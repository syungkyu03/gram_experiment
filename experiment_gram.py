import argparse
import csv
import json
import math
import time
from pathlib import Path

from PIL import Image
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from torchvision.transforms.functional import gaussian_blur

from backbone import DINOV2_REVISION, DINOV2_VITL14, IMAGENET_MEAN, IMAGENET_STD, load_dinov2_vitl14
from dataset import Images, MANIFEST_FIELDS, read_image_manifest, select_images
from metrics import gram_distances, upper_gram
from transforms import haar_attenuate, rigid_transform


NOISE_VIEW = 'rigid_noise'
GRAM_VIEWS = [NOISE_VIEW, 'gaussian_blur', 'down_up', 'wavelet']
BLUR_SIGMA_AT_224 = 1.0
DOWNSAMPLE_SCALE = 0.75
WAVELET_STRENGTH = 0.5
CLS_METRIC = 'cls_cosine_change'
GRAM_METRICS = ['patch_gram_cos', 'block_cls_gram_cos', 'patch_gram_l1',
                'block_cls_gram_l1', 'block_cls_gram_norm_drop']
EVEN_GRAM_METRICS = ['block_cls_gram_even_cos', 'block_cls_gram_even_l1', 'block_cls_gram_even_norm_drop']


def block_gram_change(original, original_norm, perturbed_cls):
    perturbed = upper_gram(perturbed_cls)
    cosine, l1 = gram_distances(original, perturbed)
    norm_drop = 1 - perturbed.double().norm(dim=-1) / original_norm
    return cosine, l1, norm_drop


def gram_batch_scores(model, images, rows, seed, noise_std, blur_sigma, *, even_blocks=False):
    normalized = (images - model.mean) / model.std
    patches, final_cls, block_cls = model.forward_with_block_cls(normalized)
    patch_gram = upper_gram(patches)
    block_gram = upper_gram(block_cls)
    block_norm = block_gram.double().norm(dim=-1)
    if even_blocks:
        # Zero-based indices 1, 3, ..., 23 select blocks 2, 4, ..., 24.
        even_gram = upper_gram(block_cls[:, 1::2])
        even_norm = even_gram.double().norm(dim=-1)
    del patches

    for view in GRAM_VIEWS:
        if view == NOISE_VIEW:
            altered, masks = rigid_transform(normalized, rows, NOISE_VIEW, seed, noise_std, local_fraction=1.0)
            del masks, normalized
        else:
            if view == 'gaussian_blur':
                kernel = 2 * math.ceil(3 * blur_sigma) + 1
                transformed = gaussian_blur(images, [kernel, kernel], [blur_sigma, blur_sigma])
            elif view == 'down_up':
                size = tuple(round(side * DOWNSAMPLE_SCALE) for side in images.shape[-2:])
                downsampled = F.interpolate(images, size=size, mode='bilinear', align_corners=False, antialias=True)
                transformed = F.interpolate(downsampled, size=images.shape[-2:], mode='bilinear',
                                            align_corners=False, antialias=True)
                del downsampled
            else:
                transformed = haar_attenuate(images, WAVELET_STRENGTH).clamp(0, 1)
            altered = (transformed - model.mean) / model.std
            del transformed

        patches_t, final_cls_t, block_cls_t = model.forward_with_block_cls(altered)
        patch_cos, patch_l1 = gram_distances(patch_gram, upper_gram(patches_t))
        block_cos, block_l1, norm_drop = block_gram_change(block_gram, block_norm, block_cls_t)
        final_cos = 1 - F.cosine_similarity(final_cls, final_cls_t, dim=-1)
        direct_block_cos = 1 - F.cosine_similarity(block_cls, block_cls_t, dim=-1)
        columns = [final_cos, patch_cos, block_cos, patch_l1, block_l1, norm_drop, direct_block_cos]
        if even_blocks:
            columns.extend(block_gram_change(even_gram, even_norm, block_cls_t[:, 1::2]))
        del patches_t, altered
        yield view, torch.column_stack(columns).cpu().tolist()


def official_rigid_batch_scores(model, images, noise_std):
    mean = images.new_tensor(IMAGENET_MEAN)[None, :, None, None]
    std = images.new_tensor(IMAGENET_STD)[None, :, None, None]
    original = ((images - mean) / std).contiguous()
    cls = model(original)
    altered = original + noise_std * torch.randn_like(original)
    scores = 1 - F.cosine_similarity(cls, model(altered), dim=-1)
    yield 'official_rigid', scores[:, None].cpu().tolist()


def main():
    load_dotenv(Path(__file__).resolve().parent / '.env')
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--size', type=int, choices=(224, 512), required=True)
    parser.add_argument('--data-root', default='/data/AIGI/GenImage')
    parser.add_argument('--samples', default='50', help='Per generator and label, or all')
    parser.add_argument('--generators', default='all')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--seed', type=int, help='Defaults to 42 for Gram and 100 for official RIGID')
    parser.add_argument('--noise-std', type=float, default=0.05)
    model_mode = parser.add_mutually_exclusive_group()
    model_mode.add_argument('--backbone', choices=('vit-b', 'vit-l'), help='DINOv3 backbone (default: vit-b)')
    model_mode.add_argument('--official-rigid', action='store_true', help='Run the DINOv2 ViT-L/14 RIGID baseline')
    parser.add_argument('--manifest', help='Use an existing manifest instead of selecting images')
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in ('', '.', '..'):
        parser.error('name must be a single directory name')
    try:
        samples = None if args.samples == 'all' else int(args.samples)
    except ValueError:
        parser.error('--samples must be a positive integer or all')
    if samples is not None and samples < 1:
        parser.error('--samples must be a positive integer or all')
    if not math.isfinite(args.noise_std) or args.noise_std <= 0:
        parser.error('--noise-std must be positive and finite')
    if args.batch_size < 1 or args.threads < 1 or args.num_workers < 0:
        parser.error('invalid batch size, thread count, or worker count')
    if args.official_rigid and (args.size != 224 or args.noise_std != 0.05 or not args.manifest):
        parser.error('--official-rigid requires --size 224, --manifest, and --noise-std 0.05')
    if args.seed is None:
        args.seed = 100 if args.official_rigid else 42

    torch.set_num_threads(args.threads)
    if args.manifest:
        try:
            rows = read_image_manifest(args.manifest)
        except ValueError as error:
            parser.error(str(error))
    else:
        rows = select_images(args.data_root, samples, args.seed, args.generators)

    output = Path('results') / args.name
    if output.exists():
        parser.error(f'run already exists: {output}')
    config = dict(vars(args), score_schema=2, resize='bilinear',
                  noise_space='imagenet_normalized_unclipped', exif_transpose=not args.official_rigid)
    blur_sigma = BLUR_SIGMA_AT_224 * args.size / 224
    even_blocks = args.backbone == 'vit-l'
    if args.official_rigid:
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        model = load_dinov2_vitl14(args.device)
        views = ['official_rigid']
        metrics = [CLS_METRIC]
        config.update(transforms='official_rigid', model_id=DINOV2_VITL14,
                      model_revision=DINOV2_REVISION, noise_rng='torch_global')
    else:
        from backbone import Backbone

        backbone_name = args.backbone or 'vit-b'
        model = Backbone(args.device, backbone=backbone_name)
        num_blocks = model.model.config.num_hidden_layers
        if num_blocks < 2:
            raise ValueError('Expected at least two transformer blocks for a CLS Gram')
        if even_blocks and num_blocks != 24:
            raise ValueError('ViT-L must have 24 blocks for the 12-block Gram comparison')
        views = GRAM_VIEWS
        metrics = [CLS_METRIC, *GRAM_METRICS, *(f'block_{i:02d}_cos' for i in range(1, num_blocks + 1))]
        if even_blocks:
            metrics.extend(EVEN_GRAM_METRICS)
            config['block_cls_gram_subset_blocks'] = list(range(2, num_blocks + 1, 2))
        config.update(transforms='gram', backbone=backbone_name, model_id=model.model_id,
                      model_revision=model.revision, num_blocks=num_blocks,
                      patch_size=model.model.config.patch_size, block_cls_source='post_block_pre_final_layernorm',
                      noise_seed_view=NOISE_VIEW, blur_sigma_at_224=BLUR_SIGMA_AT_224, blur_sigma=blur_sigma,
                      blur_kernel_radius_sigmas=3, downsample_scale=DOWNSAMPLE_SCALE,
                      down_up_interpolation='bilinear_antialias', wavelet_strength=WAVELET_STRENGTH)
    config.update(views=views, metrics=metrics)

    output.mkdir(parents=True, exist_ok=False)
    (output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    with (output / 'manifest.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    loader = DataLoader(Images(rows, args.size, resample=Image.Resampling.BILINEAR,
                               exif_transpose=config['exif_transpose']),
                        batch_size=args.batch_size, num_workers=args.num_workers)
    partial = output / 'scores.partial.csv'
    start = time.monotonic()
    with partial.open('w', newline='', encoding='utf-8') as f, torch.inference_mode():
        writer = csv.DictWriter(f, fieldnames=[*MANIFEST_FIELDS, 'view', *metrics])
        writer.writeheader()
        for indices, images, _ in loader:
            positions = indices.tolist()
            batch_rows = [rows[i] for i in positions]
            images = images.to(args.device)
            if args.official_rigid:
                batch_scores = official_rigid_batch_scores(model, images, args.noise_std)
            else:
                batch_scores = gram_batch_scores(model, images, batch_rows, args.seed, args.noise_std,
                                                 blur_sigma, even_blocks=even_blocks)
            for view, values in batch_scores:
                writer.writerows({**row, 'view': view, **dict(zip(metrics, scores))}
                                 for row, scores in zip(batch_rows, values))
            f.flush()
            done = positions[-1] + 1
            elapsed = time.monotonic() - start
            eta = elapsed / done * (len(rows) - done)
            print(f'{done}/{len(rows)} images | {done * len(views)} scores | '
                  f'elapsed {elapsed/60:.1f} min | ETA {eta/60:.1f} min', flush=True)
            if images.is_cuda and (done == len(positions) or done == len(rows)):
                peak = torch.cuda.max_memory_reserved(images.device) / 1024**3
                print(f'Peak PyTorch CUDA reserved memory: {peak:.2f} GiB', flush=True)
    partial.replace(output / 'scores.csv')


if __name__ == '__main__':
    main()
