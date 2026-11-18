import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from backbone import Backbone, MODEL_ID, REVISION
from dataset import Images, select_images
from metrics import ring_scores
from transforms import region_noise, sample_seed


def main():
    load_dotenv(Path(__file__).resolve().parent / '.env')
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--data-root', default='/data/AIGI/GenImage')
    parser.add_argument('--samples', default='50')
    parser.add_argument('--generators', default='all')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=1, help='Images per perturbed forward')
    parser.add_argument('--attention-batch-size', type=int, default=4, help='Maximum images per original attention forward')
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--noise-std', type=float, default=.05)
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in ('.', '..'):
        parser.error('name must be a single directory name')
    samples = None if args.samples == 'all' else int(args.samples)
    if samples is not None and samples < 1 or args.noise_std < 0:
        parser.error('invalid samples or noise standard deviation')
    if args.batch_size < 1 or args.attention_batch_size < 1:
        parser.error('batch sizes must be positive')
    torch.set_num_threads(args.threads)
    rows = select_images(args.data_root, samples, args.seed, args.generators)
    output = Path('results') / args.name
    output.mkdir(parents=True, exist_ok=False)
    config = dict(vars(args), model_id=MODEL_ID, model_revision=REVISION, size=512, patch_size=16, area=128, margin=4, ring_distance='Chebyshev from square', reference='all patches at distance > 4; self excluded')
    (output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    with (output / 'manifest.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['path', 'generator', 'label', 'class_name']); writer.writeheader(); writer.writerows(rows)
    model = Backbone(args.device)
    if model.model.config.patch_size != 16:
        raise ValueError('This experiment expects patch size 16')
    loader = DataLoader(Images(rows, 512), batch_size=args.batch_size, num_workers=args.num_workers)
    fields = ['path', 'generator', 'label', 'class_name', 'selection', 'top_patch', 'left_patch', 'attention_mean', 'cls_cosine_change', 'bb_change', 'rings']
    start = time.monotonic()
    with (output / 'scores.csv').open('w', newline='', encoding='utf-8') as f, torch.inference_mode():
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for indices, images, _ in loader:
            batch_rows = [rows[index] for index in indices.tolist()]
            images = images.to(args.device)
            # Requesting attention retains the maps for all layers during this forward.
            originals = [model.with_attention(chunk) for chunk in images.split(args.attention_batch_size)]
            z, cls, attention = (torch.cat(parts) for parts in zip(*originals))
            del originals
            if attention.shape != (len(images), 1024):
                raise ValueError(f'Unexpected attention shape {tuple(attention.shape)}')
            maps = attention.reshape(-1, 1, 32, 32)
            candidates = F.avg_pool2d(maps, 8, stride=1)[:, 0, 4:21, 4:21].flatten(1).cpu()
            best = candidates.argmax(dim=1).tolist()
            random_indices, noises = [], []
            for row in batch_rows:
                generator = torch.Generator(device=images.device).manual_seed(sample_seed(args.seed, row['path'], 'ring_position'))
                random_indices.append(torch.randint(candidates.shape[1], (1,), generator=generator, device=images.device))
                noise_gen = torch.Generator(device=images.device).manual_seed(sample_seed(args.seed, row['path'], 'ring_noise'))
                noises.append(torch.randn((3, 128, 128), generator=noise_gen, device=images.device) * args.noise_std)
            batch_results = [[] for _ in batch_rows]
            for selection, choices in [('high_attention', best), ('random', torch.cat(random_indices).tolist())]:
                positions = [(4 + choice // 17, 4 + choice % 17) for choice in choices]
                altered = torch.stack([
                    region_noise(image, top * 16, left * 16, noise)
                    for image, (top, left), noise in zip(images, positions, noises)
                ])
                zt, clst = model(altered)
                cls_changes = (1 - F.cosine_similarity(cls, clst)).tolist()
                for i, (top, left) in enumerate(positions):
                    rings, bb = ring_scores(z[i], zt[i], top, left)
                    batch_results[i].append({**batch_rows[i], 'selection': selection, 'top_patch': top, 'left_patch': left,
                                             'attention_mean': candidates[i, choices[i]].item(),
                                             'cls_cosine_change': cls_changes[i], 'bb_change': bb, 'rings': json.dumps(rings)})
                del altered, zt, clst
            for results in batch_results:
                writer.writerows(results)
            f.flush()
            done = indices[-1].item() + 1
            elapsed = time.monotonic() - start
            print(f'{done}/{len(rows)} images | elapsed {elapsed/60:.1f} min | ETA {elapsed/done*(len(rows)-done)/60:.1f} min', flush=True)
            if images.is_cuda and (done == len(indices) or done == len(rows)):
                peak = torch.cuda.max_memory_reserved(images.device) / 1024**3
                print(f'Peak PyTorch CUDA reserved memory: {peak:.2f} GiB', flush=True)


if __name__ == '__main__':
    main()
