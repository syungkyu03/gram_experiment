import argparse
import csv
import json
import time
from pathlib import Path

import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from backbone import Backbone, MODEL_ID, REVISION
from dataset import Images, select_images
from metrics import gram, scores, static_score
from transforms import transform, rigid_transform


def main():
    load_dotenv(Path(__file__).resolve().parent / '.env')
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--data-root', default='/data/AIGI/GenImage')
    p.add_argument('--vae-path', help='Local diffusers VAE directory')
    p.add_argument('--samples', default='50', help='Per generator and label, or all')
    p.add_argument('--generators', default='all')
    p.add_argument('--transforms', default='rigid_noise,rigid_local_noise,rigid_wavelet,noise,local_noise,wavelet')
    p.add_argument('--size', type=int, default=512)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--noise-std', type=float, default=0.05)
    p.add_argument('--rigid-noise-std', type=float, default=0.05, help='Noise standard deviation after normalization')
    p.add_argument('--local-fraction', type=float, default=0.25)
    p.add_argument('--wavelet-strength', type=float, default=0.5)
    p.add_argument('--resume', action='store_true', help='Continue an incomplete run with identical settings')
    p.add_argument('--exclude-manifest', help='Exclude image paths listed in a prior manifest')
    args = p.parse_args()
    if Path(args.name).name != args.name or args.name in ('.', '..'):
        p.error('name must be a single directory name')
    if args.samples != 'all':
        try:
            args.samples = int(args.samples)
        except ValueError:
            p.error('--samples must be a positive integer or all')
    if (args.samples != 'all' and args.samples < 1) or not 0 < args.local_fraction < 1 or not 0 <= args.wavelet_strength <= 1 or args.noise_std < 0 or args.rigid_noise_std < 0:
        p.error('Invalid sample count or transform strength')
    views = args.transforms.split(',')
    if not set(views) <= {'rigid_noise', 'rigid_local_noise', 'rigid_wavelet', 'noise', 'local_noise', 'wavelet', 'vae'} or len(set(views)) != len(views):
        p.error('Invalid or duplicate transformations')
    if 'vae' in views and not args.vae_path:
        p.error('--vae-path is required for VAE')
    torch.set_num_threads(args.threads)
    excluded = None
    if args.exclude_manifest:
        with open(args.exclude_manifest, newline='', encoding='utf-8') as f:
            excluded = {row['path'] for row in csv.DictReader(f)}
    rows = select_images(args.data_root, None if args.samples == 'all' else args.samples, args.seed, args.generators, excluded)
    output = Path('results') / args.name
    config = {k: v for k, v in vars(args).items() if k != 'resume'}
    config.update(model_id=MODEL_ID, model_revision=REVISION, score_schema=3, rigid_size=224, rigid_resize='bilinear', rigid_noise_space='normalized_no_clipping')
    columns = ['path', 'generator', 'label', 'class_name', 'view', 'static_bottom10', 'gram_l1', 'gram_signed', 'gram_near', 'gram_far', 'gram_aa', 'gram_ab', 'gram_bb', 'cka_change', 'patch_change', 'patch_change_a', 'patch_change_b', 'cls_change', 'cls_cosine_change']
    score_path = output / 'scores.csv'
    completed = set()
    if args.resume:
        if json.loads((output / 'config.json').read_text(encoding='utf-8')) != config:
            raise ValueError('Cannot resume: experiment settings or score schema differ')
        with (output / 'manifest.csv').open(newline='', encoding='utf-8') as f:
            if list(csv.DictReader(f)) != rows:
                raise ValueError('Cannot resume: selected manifest differs')
        if score_path.exists():
            with score_path.open(newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                if reader.fieldnames != columns:
                    raise ValueError('Cannot resume: score columns differ')
                for row in reader:
                    if None in row or any(row.get(k) is None for k in columns):
                        raise ValueError('Incomplete score row; inspect scores.csv before resuming')
                    key = (row['path'], row['generator'], row['label'], row['view'])
                    if key in completed:
                        raise ValueError(f'Duplicate score row: {key}')
                    completed.add(key)
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
        with (output / 'manifest.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['path', 'generator', 'label', 'class_name'])
            writer.writeheader()
            writer.writerows(rows)
    key_for = lambda row, view: (row['path'], row['generator'], row['label'], view)
    pending = [row for row in rows if any(key_for(row, view) not in completed for view in views)]
    if not pending:
        print('All requested scores are already complete.')
        return
    model = Backbone(args.device)
    vae = None
    if 'vae' in views:
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(args.vae_path, local_files_only=True).to(args.device).eval()
    loader = DataLoader(Images(pending, args.size, rigid=any(v.startswith('rigid_') for v in views)), batch_size=args.batch_size, num_workers=args.num_workers)
    start = time.monotonic()
    def status(stage, done):
        elapsed = time.monotonic() - start
        data = {'stage': stage, 'session_images_complete': done, 'session_images_total': len(pending),
                'elapsed_seconds': round(elapsed, 1),
                'eta_seconds': round(elapsed / done * (len(pending) - done), 1) if done else None,
                'score_rows_complete': len(completed), 'score_rows_total': len(rows) * len(views)}
        tmp = output / 'status.tmp'
        tmp.write_text(json.dumps(data, indent=2), encoding='utf-8')
        tmp.replace(output / 'status.json')
    with score_path.open('a' if score_path.exists() else 'w', newline='', encoding='utf-8') as f, torch.inference_mode():
        writer = csv.DictWriter(f, fieldnames=columns)
        if f.tell() == 0:
            writer.writeheader()
        for indices, images, rigid_images in loader:
            batch_rows = [pending[i] for i in indices.tolist()]
            images = images.to(args.device)
            standard_needed = any(key_for(row, view) not in completed for row in batch_rows for view in views if not view.startswith('rigid_'))
            if standard_needed:
                status('original', indices[0].item())
                z, cls = model(images)
                static = static_score(gram(z)).cpu().tolist()
            rigid_z = rigid_cls = rigid_base = rigid_static = None
            for view in views:
                positions = [i for i, row in enumerate(batch_rows) if key_for(row, view) not in completed]
                if not positions:
                    continue
                status(view, indices[0].item())
                view_rows = [batch_rows[i] for i in positions]
                if view.startswith('rigid_'):
                    if rigid_base is None:
                        rigid_base = (rigid_images.to(args.device) - model.mean) / model.std
                        rigid_z, rigid_cls = model.forward_normalized(rigid_base)
                        rigid_static = static_score(gram(rigid_z)).cpu().tolist()
                    if view == 'rigid_wavelet':
                        altered, mask = transform(rigid_images[positions].to(args.device), view_rows, 'wavelet', args.seed, args.noise_std, args.local_fraction, args.wavelet_strength, vae)
                        zt, clst = model(altered)
                    else:
                        altered, mask = rigid_transform(rigid_base[positions], view_rows, view, args.seed, args.rigid_noise_std, args.local_fraction)
                        zt, clst = model.forward_normalized(altered)
                    values = scores(rigid_z[positions], rigid_cls[positions], zt, clst, mask if view == 'rigid_local_noise' else None)
                    view_static = [rigid_static[i] for i in positions]
                else:
                    altered, mask = transform(images[positions], view_rows, view, args.seed, args.noise_std, args.local_fraction, args.wavelet_strength, vae)
                    zt, clst = model(altered)
                    values = scores(z[positions], cls[positions], zt, clst, mask if view == 'local_noise' else None)
                    view_static = [static[i] for i in positions]
                values = {k: v.cpu().tolist() for k, v in values.items()}
                for i, row in enumerate(view_rows):
                    writer.writerow({**row, 'view': view, 'static_bottom10': view_static[i], **{k: v[i] for k, v in values.items()}})
                    completed.add(key_for(row, view))
                f.flush()
            done = indices[-1].item() + 1
            elapsed = time.monotonic() - start
            status('complete' if done == len(pending) else 'batch_complete', done)
            print(f'Images {done}/{len(pending)} this session | elapsed {elapsed/60:.1f} min | ETA {(elapsed / done * (len(pending) - done))/60:.1f} min | score rows {len(completed)}/{len(rows)*len(views)}', flush=True)


if __name__ == '__main__':
    main()
