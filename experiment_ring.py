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
    loader = DataLoader(Images(rows, 512), batch_size=1, num_workers=args.num_workers)
    fields = ['path', 'generator', 'label', 'class_name', 'selection', 'top_patch', 'left_patch', 'attention_mean', 'cls_cosine_change', 'bb_change', 'rings']
    start = time.monotonic()
    with (output / 'scores.csv').open('w', newline='', encoding='utf-8') as f, torch.inference_mode():
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for indices, images, _ in loader:
            index = indices.item(); row = rows[index]; image = images.to(args.device)
            z, cls, attention = model.with_attention(image)
            if attention.shape != (1, 1024):
                raise ValueError(f'Unexpected attention shape {tuple(attention.shape)}')
            map_ = attention.reshape(1, 1, 32, 32)
            candidates = F.avg_pool2d(map_, 8, stride=1)[0, 0, 4:21, 4:21]
            best = int(candidates.argmax())
            generator = torch.Generator(device=image.device).manual_seed(sample_seed(args.seed, row['path'], 'ring_position'))
            random_index = int(torch.randint(candidates.numel(), (1,), generator=generator, device=image.device))
            noise_gen = torch.Generator(device=image.device).manual_seed(sample_seed(args.seed, row['path'], 'ring_noise'))
            noise = torch.randn((3, 128, 128), generator=noise_gen, device=image.device) * args.noise_std
            for selection, chosen in [('high_attention', best), ('random', random_index)]:
                top, left = 4 + chosen // 17, 4 + chosen % 17
                altered = region_noise(image[0], top * 16, left * 16, noise)
                zt, clst = model(altered[None])
                rings, bb = ring_scores(z[0], zt[0], top, left)
                writer.writerow({**row, 'selection': selection, 'top_patch': top, 'left_patch': left,
                                 'attention_mean': float(candidates[top - 4, left - 4]),
                                 'cls_cosine_change': float(1 - F.cosine_similarity(cls, clst).item()),
                                 'bb_change': bb, 'rings': json.dumps(rings)})
            f.flush()
            done = index + 1; elapsed = time.monotonic() - start
            print(f'{done}/{len(rows)} images | elapsed {elapsed/60:.1f} min | ETA {elapsed/done*(len(rows)-done)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
