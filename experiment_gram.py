import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from dataset import Images, select_images
from metrics import gram_distances, upper_gram
from transforms import rigid_transform


PRIMARY_METRICS = ['cls_cosine_change', 'patch_gram_cos', 'block_cls_gram_cos']
MANIFEST_FIELDS = ['path', 'generator', 'label', 'class_name']
NOISE_VIEW = 'rigid_noise'


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
    model = Backbone(args.device)
    num_blocks = model.model.config.num_hidden_layers
    if num_blocks < 2:
        raise ValueError('Expected at least two transformer blocks for a CLS Gram')
    block_metrics = [f'block_{i:02d}_cos' for i in range(1, num_blocks + 1)]
    metrics = [*PRIMARY_METRICS, 'patch_gram_l1', 'block_cls_gram_l1', *block_metrics]
    config = dict(vars(args), model_id=MODEL_ID, model_revision=REVISION, score_schema=1,
                  transforms='gram', resize='bilinear', noise_seed_view=NOISE_VIEW,
                  noise_space='imagenet_normalized_unclipped', num_blocks=num_blocks,
                  patch_size=model.model.config.patch_size,
                  block_cls_source='post_block_pre_final_layernorm', primary_metrics=PRIMARY_METRICS,
                  metrics=metrics)
    (output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    with (output / 'manifest.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    loader = DataLoader(Images(rows, args.size, resample=Image.Resampling.BILINEAR),
                        batch_size=args.batch_size, num_workers=args.num_workers)
    partial = output / 'block_cls_grams.partial.npy'
    blocks_archive = np.lib.format.open_memmap(partial, mode='w+', dtype=np.float32,
                                               shape=(len(rows), 2, math.comb(num_blocks, 2)))
    start = time.monotonic()
    with (output / 'scores.csv').open('w', newline='', encoding='utf-8') as f, torch.inference_mode():
        writer = csv.DictWriter(f, fieldnames=[*MANIFEST_FIELDS, 'view', *metrics])
        writer.writeheader()
        for indices, images, _ in loader:
            positions = indices.tolist()
            batch_rows = [rows[i] for i in positions]
            normalized = (images.to(args.device) - model.mean) / model.std
            patches, final_cls, block_cls = model.forward_with_block_cls(normalized)
            altered, masks = rigid_transform(normalized, batch_rows, NOISE_VIEW,
                                             args.seed, args.noise_std, local_fraction=1.0)
            del normalized, masks
            patches_t, final_cls_t, block_cls_t = model.forward_with_block_cls(altered)
            del altered

            patch_cos, patch_l1 = gram_distances(upper_gram(patches), upper_gram(patches_t))
            del patches, patches_t
            block_gram, block_gram_t = upper_gram(block_cls), upper_gram(block_cls_t)
            block_cos, block_l1 = gram_distances(block_gram, block_gram_t)
            final_cos = 1 - F.cosine_similarity(final_cls, final_cls_t, dim=-1)
            direct_block_cos = 1 - F.cosine_similarity(block_cls, block_cls_t, dim=-1)

            blocks_archive[positions] = torch.stack((block_gram, block_gram_t), dim=1).cpu().numpy()
            batch_scores = torch.column_stack((final_cos, patch_cos, block_cos, patch_l1,
                                               block_l1, direct_block_cos)).cpu().tolist()
            writer.writerows({**row, 'view': 'gram', **dict(zip(metrics, scores))}
                             for row, scores in zip(batch_rows, batch_scores))
            f.flush()
            done = positions[-1] + 1
            elapsed = time.monotonic() - start
            eta = elapsed / done * (len(rows) - done)
            print(f'{done}/{len(rows)} images | elapsed {elapsed/60:.1f} min | ETA {eta/60:.1f} min', flush=True)
            if final_cls.is_cuda and (done == len(positions) or done == len(rows)):
                peak = torch.cuda.max_memory_reserved(final_cls.device) / 1024**3
                print(f'Peak PyTorch CUDA reserved memory: {peak:.2f} GiB', flush=True)
    blocks_archive.flush()
    del blocks_archive
    partial.replace(output / 'block_cls_grams.npy')


if __name__ == '__main__':
    main()
