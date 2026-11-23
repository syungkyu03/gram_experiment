import torch

MODEL_ID = 'facebook/dinov3-vitb16-pretrain-lvd1689m'
REVISION = '5931719e67bbdb9737e363e781fb0c67687896bc'
DINOV2_REPO = 'facebookresearch/dinov2'
DINOV2_REVISION = 'b194f00db6136677fc8a4cc2ef2168f7699dfba2'
DINOV2_VITL14 = 'dinov2_vitl14'
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DINO_BACKBONES = {
    'vit-b': (MODEL_ID, REVISION),
    'vit-l': ('facebook/dinov3-vitl16-pretrain-lvd1689m', 'ea8dc2863c51be0a264bab82070e3e8836b02d51'),
}


def load_dinov2_vitl14(device):
    return torch.hub.load(f'{DINOV2_REPO}:{DINOV2_REVISION}', DINOV2_VITL14,
                          trust_repo=True, skip_validation=True).requires_grad_(False).eval().to(device)


class Backbone:
    def __init__(self, device, *, backbone='vit-b'):
        from huggingface_hub import snapshot_download
        from transformers import AutoImageProcessor, AutoModel

        self.device = device
        self.model_id, self.revision = DINO_BACKBONES[backbone]
        snapshot = snapshot_download(self.model_id, revision=self.revision,
                                     allow_patterns=['*.json', '*.safetensors'])
        processor = AutoImageProcessor.from_pretrained(snapshot, local_files_only=True)
        mean, std = IMAGENET_MEAN, IMAGENET_STD
        if len(processor.image_mean) != 3 or len(processor.image_std) != 3 or any(abs(a-b) > 1e-7 for a,b in zip(processor.image_mean, mean)) or any(abs(a-b) > 1e-7 for a,b in zip(processor.image_std, std)):
            raise ValueError('Processor normalization differs from the experiment protocol')
        self.model = AutoModel.from_pretrained(snapshot, local_files_only=True, dtype=torch.float32, attn_implementation='eager').requires_grad_(False).eval().to(device)
        self.mean = torch.tensor(mean, device=device)[None, :, None, None]
        self.std = torch.tensor(std, device=device)[None, :, None, None]

    @torch.inference_mode()
    def __call__(self, images):
        return self.forward_normalized((images.to(self.device) - self.mean) / self.std)

    @torch.inference_mode()
    def with_attention(self, images):
        images = (images.to(self.device) - self.mean) / self.std
        output = self.model(pixel_values=images, output_attentions=True, return_dict=True)
        tokens = output.last_hidden_state
        prefix = 1 + self.model.config.num_register_tokens
        side = images.shape[-1] // self.model.config.patch_size
        if tokens.shape[1] != prefix + side * side or not output.attentions or output.attentions[-1] is None:
            raise RuntimeError('Expected patch tokens and final-layer attention')
        attention = output.attentions[-1][:, :, 0, prefix:].float().mean(1)
        return tokens[:, prefix:].float(), tokens[:, 0].float(), attention

    @torch.inference_mode()
    def forward_normalized(self, images):
        config = self.model.config
        h, w = images.shape[-2:]
        patch = config.patch_size
        if h % patch or w % patch:
            raise ValueError('Image dimensions must be divisible by patch size')
        tokens = self.model(pixel_values=images, output_hidden_states=False, output_attentions=False, return_dict=True).last_hidden_state
        prefix = 1 + config.num_register_tokens
        expected = (images.shape[0], prefix + (h // patch) * (w // patch), config.hidden_size)
        if tuple(tokens.shape) != expected:
            raise ValueError(f'Token shape {tuple(tokens.shape)} does not match {expected}')
        return tokens[:, prefix:].float(), tokens[:, 0].float()

    @torch.inference_mode()
    def forward_with_block_cls(self, normalized_images):
        # Read each block before the separate final LayerNorm. Clone the CLS slice
        # so that the hooks do not keep the whole patch sequence for each layer.
        encoder = getattr(self.model, 'model', self.model)
        blocks = encoder.layer
        cls_states = []

        def capture_cls(_module, _inputs, output):
            cls_states.append(output[:, 0, :].clone())

        handles = [block.register_forward_hook(capture_cls) for block in blocks]
        try:
            patches, final_cls = self.forward_normalized(normalized_images)
        finally:
            for handle in handles:
                handle.remove()
        if len(cls_states) != self.model.config.num_hidden_layers:
            raise RuntimeError('Expected one CLS state for each transformer block')
        return patches, final_cls, torch.stack(cls_states, dim=1).float()
