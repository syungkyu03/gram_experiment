import torch


class Backbone:
    def __init__(self, repo, model, weights, device):
        self.device = device
        self.model = torch.hub.load(repo, model, source='local', weights=weights).to(device).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]

    @torch.inference_mode()
    def __call__(self, images):
        images = (images.to(self.device) - self.mean) / self.std
        result = self.model.get_intermediate_layers(images, n=1, return_class_token=True, norm=True)[0]
        return result[0].float(), result[1].float()
