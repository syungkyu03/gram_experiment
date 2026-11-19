import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from dataset import Images, select_images
from metrics import lr_scores
from transforms import region_noise, sample_seed


IMAGE_SIZE = 512
PATCH_SIZE = 16
# Rectangles are (top, left, height, width) in patch coordinates.
PERTURBATIONS = {'left': (12, 4, 8, 8), 'right': (12, 20, 8, 8)}
READOUTS = {'left': (12, 0, 8, 4), 'right': (12, 28, 8, 4)}
NOISE_SEED_VIEW = 'ring_noise'
MANIFEST_FIELDS = ['path', 'generator', 'label', 'class_name']
SCORE_FIELDS = [*MANIFEST_FIELDS, 'view', 'response_ll', 'response_lr', 'response_rl',
                'response_rr', 'near_patch', 'far_patch', 'lr_cross_ratio']


def main():
    load_dotenv(Path(__file__).resolve().parent / '.env')
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--data-root', default='/data/AIGI/GenImage')
    parser.add_argument('--samples', default='50', help='Per generator and label, or all')
    parser.add_argument('--generators', default='all')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=4, help='Images per forward')
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--noise-std', type=float, default=0.05)
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

    from backbone import Backbone, MODEL_ID, REVISION

    torch.set_num_threads(args.threads)
    rows = select_images(args.data_root, samples, args.seed, args.generators)
    output = Path('results') / args.name
    output.mkdir(parents=True, exist_ok=False)
    config = dict(vars(args), model_id=MODEL_ID, model_revision=REVISION, score_schema=1,
                  transforms='lr', size=IMAGE_SIZE, patch_size=PATCH_SIZE,
                  perturbations=PERTURBATIONS, readouts=READOUTS, noise_seed_view=NOISE_SEED_VIEW)
    (output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    with (output / 'manifest.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    model = Backbone(args.device)
    if model.model.config.patch_size != PATCH_SIZE:
        raise ValueError(f'This experiment expects patch size {PATCH_SIZE}')
    grid_side = IMAGE_SIZE // PATCH_SIZE
    grid = torch.arange(grid_side ** 2, device=args.device).reshape(grid_side, grid_side)
    readout_indices = torch.stack([
        grid[top:top + height, left:left + width].flatten()
        for top, left, height, width in READOUTS.values()
    ])
    _, _, mask_height, mask_width = PERTURBATIONS['left']
    noise_shape = (3, mask_height * PATCH_SIZE, mask_width * PATCH_SIZE)
    loader = DataLoader(Images(rows, IMAGE_SIZE), batch_size=args.batch_size,
                        num_workers=args.num_workers)
    start = time.monotonic()
    with (output / 'scores.csv').open('w', newline='', encoding='utf-8') as f, torch.inference_mode():
        writer = csv.DictWriter(f, fieldnames=SCORE_FIELDS)
        writer.writeheader()
        for indices, images, _ in loader:
            batch_rows = [rows[index] for index in indices.tolist()]
            images = images.to(args.device)
            original = model(images)[0][:, readout_indices]

            noises = []
            for row in batch_rows:
                seed = sample_seed(args.seed, row['path'], NOISE_SEED_VIEW)
                generator = torch.Generator(device=images.device).manual_seed(seed)
                noises.append(torch.randn(noise_shape, generator=generator, device=images.device))
            noise = torch.stack(noises) * args.noise_std

            perturbed = []
            for top, left, _, _ in PERTURBATIONS.values():
                altered = region_noise(images, top * PATCH_SIZE, left * PATCH_SIZE, noise)
                perturbed.append(model(altered)[0][:, readout_indices])
                del altered
            values = {key: value.cpu().tolist() for key, value in lr_scores(original, *perturbed).items()}
            for i, row in enumerate(batch_rows):
                writer.writerow({**row, 'view': 'lr', **{key: value[i] for key, value in values.items()}})
            f.flush()

            done = indices[-1].item() + 1
            elapsed = time.monotonic() - start
            eta = elapsed / done * (len(rows) - done)
            print(f'{done}/{len(rows)} images | elapsed {elapsed/60:.1f} min | ETA {eta/60:.1f} min', flush=True)
            if images.is_cuda and (done == len(indices) or done == len(rows)):
                peak = torch.cuda.max_memory_reserved(images.device) / 1024**3
                print(f'Peak PyTorch CUDA reserved memory: {peak:.2f} GiB', flush=True)


if __name__ == '__main__':
    main()
