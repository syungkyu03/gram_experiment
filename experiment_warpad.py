import argparse
import csv
import json
import time
from pathlib import Path

from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from backbone import DINOV2_REPO, DINOV2_REVISION, DINOV2_VITL14, IMAGENET_MEAN, IMAGENET_STD, load_dinov2_vitl14
from dataset import Images, MANIFEST_FIELDS, read_image_manifest


PREP_SIZE = 896
TILE_SIZE = 224
TILES_PER_IMAGE = (PREP_SIZE // TILE_SIZE) ** 2
WAVELET_LEVELS = 2
ATTENUATION = 0.1
SCORE = 'cls_cosine_change'


def split_tiles(images):
    tiles = images.unfold(2, TILE_SIZE, TILE_SIZE).unfold(3, TILE_SIZE, TILE_SIZE)
    return tiles.permute(0, 2, 3, 1, 4, 5).reshape(-1, images.shape[1], TILE_SIZE, TILE_SIZE)


def wavelet_perturbation(tiles, dwt, inverse_dwt):
    low, detail = dwt(tiles)
    high = inverse_dwt((torch.zeros_like(low), detail))
    return tiles - ATTENUATION * high


def warpad_batch_scores(model, normalized_images, dwt, inverse_dwt, tile_batch_size):
    similarities = []
    for tiles in split_tiles(normalized_images).split(tile_batch_size):
        altered = wavelet_perturbation(tiles, dwt, inverse_dwt)
        original_cls = model.forward_features(tiles, None)['x_norm_clstoken']
        altered_cls = model.forward_features(altered, None)['x_norm_clstoken']
        similarities.append(F.cosine_similarity(original_cls, altered_cls, dim=-1))
    per_image = torch.cat(similarities).reshape(normalized_images.shape[0], TILES_PER_IMAGE)
    return 1 - per_image.mean(dim=1)


def main():
    parser = argparse.ArgumentParser(description='Run the official GenImage WaRPAD method on a shared manifest')
    parser.add_argument('--name', required=True)
    parser.add_argument('--manifest', required=True, help='Existing manifest.csv with original image paths')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=1, help='Full images per DataLoader batch')
    parser.add_argument('--tile-batch-size', type=int, default=TILES_PER_IMAGE, help='224px tiles per model forward')
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in ('', '.', '..'):
        parser.error('name must be a single directory name')
    if min(args.batch_size, args.tile_batch_size, args.threads) < 1 or args.num_workers < 0:
        parser.error('invalid image batch size, tile batch size, thread count, or worker count')
    try:
        rows = read_image_manifest(args.manifest)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    output = Path('results') / args.name
    if output.exists():
        parser.error(f'run already exists: {output}')
    try:
        from pytorch_wavelets import DWTForward, DWTInverse
    except ImportError as error:
        parser.error(f'WaRPAD requires pytorch-wavelets and PyWavelets: {error}')

    torch.set_num_threads(args.threads)
    torch.manual_seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = load_dinov2_vitl14(args.device)
    dwt = DWTForward(J=WAVELET_LEVELS, wave='haar', mode='zero').to(args.device)
    inverse_dwt = DWTInverse(wave='haar', mode='zero').to(args.device)
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32, device=args.device)[None, :, None, None]
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32, device=args.device)[None, :, None, None]

    config = dict(vars(args), score_schema=2, transforms='warpad', views=['warpad'], metrics=[SCORE],
                  model_repo=DINOV2_REPO, model_id=DINOV2_VITL14, model_revision=DINOV2_REVISION,
                  seed=1, size=PREP_SIZE, resize='bicubic', exif_transpose=False,
                  tile_size=TILE_SIZE, tiles_per_image=TILES_PER_IMAGE,
                  mean=IMAGENET_MEAN, std=IMAGENET_STD, wavelet='haar', wavelet_levels=WAVELET_LEVELS,
                  wavelet_padding='zero', high_frequency_attenuation=ATTENUATION,
                  score_aggregation='1_minus_mean_tile_cosine')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    with (output / 'manifest.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    loader = DataLoader(Images(rows, PREP_SIZE, resample=Image.Resampling.BICUBIC, exif_transpose=False),
                        batch_size=args.batch_size, num_workers=args.num_workers)
    partial = output / 'scores.partial.csv'
    start = time.monotonic()
    with partial.open('w', newline='', encoding='utf-8') as f, torch.inference_mode():
        writer = csv.DictWriter(f, fieldnames=[*MANIFEST_FIELDS, 'view', SCORE])
        writer.writeheader()
        for batch, (indices, images, _) in enumerate(loader, 1):
            normalized = (images.to(args.device) - mean) / std
            scores = warpad_batch_scores(model, normalized, dwt, inverse_dwt, args.tile_batch_size).cpu().tolist()
            positions = indices.tolist()
            writer.writerows({**rows[i], 'view': 'warpad', SCORE: value} for i, value in zip(positions, scores))
            f.flush()
            done = positions[-1] + 1
            if batch == 1 or batch % 100 == 0 or done == len(rows):
                elapsed = time.monotonic() - start
                eta = elapsed / done * (len(rows) - done)
                print(f'{done}/{len(rows)} images | elapsed {elapsed/60:.1f} min | ETA {eta/60:.1f} min', flush=True)
            if normalized.is_cuda and (batch == 1 or done == len(rows)):
                peak = torch.cuda.max_memory_reserved(normalized.device) / 1024**3
                print(f'Peak PyTorch CUDA reserved memory: {peak:.2f} GiB', flush=True)
    partial.replace(output / 'scores.csv')


if __name__ == '__main__':
    main()
