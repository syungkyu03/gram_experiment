import argparse
import csv
import json
from pathlib import Path

import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from backbone import Backbone, MODEL_ID, REVISION
from dataset import Images, select_images
from metrics import gram, scores, static_score
from transforms import transform


def main():
    load_dotenv(Path(__file__).resolve().parent / '.env')
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--data-root', default='/data/AIGI/GenImage')
    p.add_argument('--vae-path', help='Local diffusers VAE directory')
    p.add_argument('--samples', type=int, default=50, help='Per generator and real/fake label')
    p.add_argument('--generators', default='all')
    p.add_argument('--transforms', default='noise,local_noise,wavelet')
    p.add_argument('--size', type=int, default=512)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--noise-std', type=float, default=0.05)
    p.add_argument('--local-fraction', type=float, default=0.25)
    p.add_argument('--wavelet-strength', type=float, default=0.5)
    args = p.parse_args()
    if Path(args.name).name != args.name or args.name in ('.', '..'):
        p.error('name must be a single directory name')
    if args.samples < 1 or not 0 < args.local_fraction < 1 or not 0 <= args.wavelet_strength <= 1 or args.noise_std < 0:
        p.error('Invalid sample count or transform strength')
    views = args.transforms.split(',')
    if not set(views) <= {'noise', 'local_noise', 'wavelet', 'vae'} or len(set(views)) != len(views):
        p.error('Invalid or duplicate transformations')
    if 'vae' in views and not args.vae_path:
        p.error('--vae-path is required for VAE')
    torch.set_num_threads(args.threads)
    rows = select_images(args.data_root, args.samples, args.seed, args.generators)
    output = Path('results') / args.name
    output.mkdir(parents=True, exist_ok=False)
    (output / 'config.json').write_text(json.dumps({**vars(args), 'model_id': MODEL_ID, 'model_revision': REVISION}, indent=2), encoding='utf-8')
    with (output / 'manifest.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['path', 'generator', 'label', 'class_name'])
        writer.writeheader()
        writer.writerows(rows)
    model = Backbone(args.device)
    vae = None
    if 'vae' in views:
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(args.vae_path, local_files_only=True).to(args.device).eval()
    loader = DataLoader(Images(rows, args.size), batch_size=args.batch_size, num_workers=args.num_workers)
    columns = ['path', 'generator', 'label', 'class_name', 'view', 'static_bottom10', 'gram_l1', 'gram_signed', 'gram_near', 'gram_far', 'gram_bb', 'cka_change', 'patch_change', 'cls_change']
    with (output / 'scores.csv').open('w', newline='', encoding='utf-8') as f, torch.inference_mode():
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for indices, images in loader:
            batch_rows = [rows[i] for i in indices.tolist()]
            images = images.to(args.device)
            z, cls = model(images)
            static = static_score(gram(z)).cpu().tolist()
            for view in views:
                altered, mask = transform(images, batch_rows, view, args.seed, args.noise_std, args.local_fraction, args.wavelet_strength, vae)
                zt, clst = model(altered)
                values = scores(z, cls, zt, clst, mask if view == 'local_noise' else None)
                values = {k: v.cpu().tolist() for k, v in values.items()}
                for i, row in enumerate(batch_rows):
                    writer.writerow({**row, 'view': view, 'static_bottom10': static[i], **{k: v[i] for k, v in values.items()}})
                f.flush()
            print(f'Processed {indices[-1].item() + 1}/{len(rows)}', flush=True)


if __name__ == '__main__':
    main()
