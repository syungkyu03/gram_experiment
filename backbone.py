import torch
from huggingface_hub import snapshot_download
from transformers import AutoImageProcessor, AutoModel

MODEL_ID = 'facebook/dinov3-vitb16-pretrain-lvd1689m'
REVISION = '5931719e67bbdb9737e363e781fb0c67687896bc'


class Backbone:
    def __init__(self, device):
        self.device = device
        snapshot = snapshot_download(MODEL_ID, revision=REVISION, allow_patterns=['*.json', '*.safetensors'])
        processor = AutoImageProcessor.from_pretrained(snapshot, local_files_only=True)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        if len(processor.image_mean) != 3 or len(processor.image_std) != 3 or any(abs(a-b) > 1e-7 for a,b in zip(processor.image_mean, mean)) or any(abs(a-b) > 1e-7 for a,b in zip(processor.image_std, std)):
            raise ValueError('Processor normalization differs from the experiment protocol')
        self.model = AutoModel.from_pretrained(snapshot, local_files_only=True, dtype=torch.float32, attn_implementation='eager').requires_grad_(False).eval().to(device)
        self.mean = torch.tensor(mean, device=device)[None, :, None, None]
        self.std = torch.tensor(std, device=device)[None, :, None, None]

    @torch.inference_mode()
    def __call__(self, images):
        images = (images.to(self.device) - self.mean) / self.std
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
