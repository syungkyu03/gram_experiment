import hashlib
import torch


def sample_seed(seed, path, view):
    digest = hashlib.sha256(f'{seed}:{path}:{view}'.encode()).digest()
    return int.from_bytes(digest[:8], 'little') % (2**63 - 1)


def transform(images, rows, view, seed, noise_std, local_fraction, wavelet_strength, vae=None):
    result, masks = [], []
    for image, row in zip(images, rows):
        x = image.clone()
        generator = torch.Generator(device=x.device).manual_seed(sample_seed(seed, row['path'], view))
        mask = torch.zeros(x.shape[-2:], device=x.device, dtype=torch.bool)
        if view == 'noise':
            x = x + torch.randn(x.shape, generator=generator, device=x.device) * noise_std
        elif view == 'local_noise':
            h, w = x.shape[-2:]
            side_h, side_w = max(1, round(h * local_fraction ** .5)), max(1, round(w * local_fraction ** .5))
            top = int(torch.randint(h - side_h + 1, (1,), generator=generator, device=x.device))
            left = int(torch.randint(w - side_w + 1, (1,), generator=generator, device=x.device))
            mask[top:top + side_h, left:left + side_w] = True
            x = x + torch.randn(x.shape, generator=generator, device=x.device) * noise_std * mask
        elif view == 'wavelet':
            if x.shape[-1] % 2 or x.shape[-2] % 2:
                raise ValueError('Wavelet requires even image dimensions')
            low = torch.nn.functional.avg_pool2d(x[None], 2, 2).repeat_interleave(2, -2).repeat_interleave(2, -1)[0]
            x = low + (x - low) * (1 - wavelet_strength)
        elif view == 'vae':
            with torch.inference_mode():
                latent = vae.encode(x[None] * 2 - 1).latent_dist.mode()
                x = (vae.decode(latent).sample[0] + 1) / 2
        else:
            raise ValueError('Unknown transformation: ' + view)
        result.append(x.clamp(0, 1))
        masks.append(mask)
    return torch.stack(result), torch.stack(masks)
