import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

from sklearn.metrics import roc_auc_score

from analysis_details import write_csv
from analyze import report


def plot_block_grams(archive, rows, num_blocks, run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[(row['generator'], row['label'])].append(i)
    class_means = defaultdict(list)
    for (_, label), indices in groups.items():
        original, noisy = archive[indices].transpose(1, 0, 2)
        class_means[label].append(np.stack((original.mean(axis=0, dtype=np.float64),
                                            np.abs(noisy - original).mean(axis=0, dtype=np.float64))))
    real, fake = (np.mean(class_means[label], axis=0) for label in ('real', 'fake'))
    i, j = np.triu_indices(num_blocks, k=1)
    ticks = np.linspace(0, num_blocks - 1, min(4, num_blocks), dtype=int)
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.5), layout='constrained')
    for row_idx, (real_values, fake_values, title) in enumerate(zip(real, fake, ('Original cosine', 'Absolute Gram change'))):
        diff = fake_values - real_values
        diff_max = max(float(np.abs(diff).max()), np.finfo(float).eps)
        vmin = min(real_values.min(), fake_values.min(), np.nextafter(1.0, 0.0)) if row_idx == 0 else 0.0
        vmax = 1.0 if row_idx == 0 else max(real_values.max(), fake_values.max(), np.finfo(float).eps)
        for col, (values, label) in enumerate(zip((real_values, fake_values, diff), ('Real', 'Fake', 'Fake minus Real'))):
            matrix = np.eye(num_blocks) if row_idx == 0 and col < 2 else np.zeros((num_blocks, num_blocks))
            matrix[i, j] = matrix[j, i] = values
            ax = axes[row_idx, col]
            if col < 2:
                absolute = ax.imshow(matrix, cmap='viridis', vmin=vmin, vmax=vmax)
            else:
                difference = ax.imshow(matrix, cmap='RdBu_r', vmin=-diff_max, vmax=diff_max)
            ax.set(title=f'{title} / {label}', xlabel='Block', ylabel='Block')
            ax.set_xticks(ticks, ticks + 1)
            ax.set_yticks(ticks, ticks + 1)
        fig.colorbar(absolute, ax=axes[row_idx, :2], shrink=.75)
        fig.colorbar(difference, ax=axes[row_idx, 2], shrink=.75)
    fig.savefig(run / 'figures' / 'block_cls_gram.png', dpi=160)
    plt.close(fig)


MANIFEST_FIELDS = ('path', 'generator', 'label', 'class_name')
FINAL_CLS = 'cls_cosine_change'
MACRO = 'macro'
VIEW_LABELS = {'rigid_noise': 'Gaussian noise', 'gaussian_blur': 'Gaussian blur',
               'down_up': 'Down / upsample', 'wavelet': 'Haar wavelet',
               'official_rigid': 'Official RIGID', 'warpad': 'WaRPAD'}
METHOD_LABELS = {FINAL_CLS: 'Final CLS', 'patch_gram_cos': 'Patch Gram cosine',
                 'block_cls_gram_cos': 'CLS Gram cosine', 'patch_gram_l1': 'Patch Gram L1',
                 'block_cls_gram_l1': 'CLS Gram L1', 'block_cls_gram_norm_drop': 'CLS Gram norm drop',
                 'block_cls_gram_even_norm_drop': 'CLS Gram norm drop (12 even blocks)'}
GRAM_PROTOCOL = ('resize', 'noise_space', 'noise_std', 'noise_seed_view', 'seed', 'block_cls_source',
                 'patch_size', 'views', 'exif_transpose', 'blur_sigma_at_224', 'blur_kernel_radius_sigmas',
                 'downsample_scale', 'down_up_interpolation', 'wavelet_strength')
VIT_B_ID = 'facebook/dinov3-vitb16-pretrain-lvd1689m'
VIT_L_ID = 'facebook/dinov3-vitl16-pretrain-lvd1689m'


def read_manifest(run):
    with (run / 'manifest.csv').open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def read_scores(run, manifest, views, metrics, legacy, *, save_audit=True):
    positions = {tuple(row[field] for field in MANIFEST_FIELDS): i for i, row in enumerate(manifest)}
    view_indices = {view: i for i, view in enumerate(views)}
    values = np.full((len(views), len(manifest), len(metrics)), np.nan)
    seen = np.zeros(values.shape[:2], dtype=bool)
    unexpected = duplicates = order_mismatches = row_count = 0
    path = run / 'scores.csv'
    if not path.is_file() and (run / 'scores.partial.csv').is_file():
        raise ValueError(f'{run} has only partial scores. Finish the experiment before analysis.')
    with path.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        missing_fields = set((*MANIFEST_FIELDS, 'view', *metrics)) - set(reader.fieldnames or [])
        if missing_fields:
            raise ValueError(f'{path} is missing columns: {", ".join(sorted(missing_fields))}')
        for row_count, row in enumerate(reader, 1):
            view = view_indices.get(row['view'])
            position = positions.get(tuple(row[field] for field in MANIFEST_FIELDS))
            if view is None or position is None:
                unexpected += 1
                continue
            if seen[view, position]:
                duplicates += 1
                continue
            if legacy and position != row_count - 1:
                order_mismatches += 1
            seen[view, position] = True
            for column, metric in enumerate(metrics):
                try:
                    values[view, position, column] = float(row[metric])
                except (TypeError, ValueError):
                    continue
    audit = dict(manifest_rows=len(manifest), expected_score_rows=seen.size, actual_score_rows=row_count,
                 missing_rows=int(seen.size - seen.sum()), unexpected_rows=unexpected,
                 duplicate_rows=duplicates, duplicate_manifest_rows=len(manifest) - len(positions),
                 manifest_order_mismatches=order_mismatches,
                 nonfinite_or_missing={metric: int(np.count_nonzero(seen & ~np.isfinite(values[:, :, i])))
                                       for i, metric in enumerate(metrics)})
    if save_audit:
        (run / 'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print('Audit:', json.dumps(audit), flush=True)
    if any(audit[key] for key in ('missing_rows', 'unexpected_rows', 'duplicate_rows',
                                  'duplicate_manifest_rows', 'manifest_order_mismatches')) or any(audit['nonfinite_or_missing'].values()):
        raise ValueError(f'{run} is incomplete or contains invalid scores. See the audit above.')
    return values


def plot_macro(run, size, views, metrics, aurocs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figures = run / 'figures'
    figures.mkdir(exist_ok=True)
    plotted = [metric for metric in METHOD_LABELS if metric in metrics]
    fig, ax = plt.subplots(figsize=(10, 4.6), layout='constrained')
    x = np.arange(len(plotted))
    width = .8 / len(views)
    for i, view in enumerate(views):
        dx = (i - (len(views) - 1) / 2) * width
        ax.bar(x + dx, [aurocs[view, metric, MACRO] for metric in plotted],
               width=width, label=VIEW_LABELS[view])
    ax.axhline(.5, color='0.6', linestyle='--', linewidth=.8)
    ax.set(ylim=(0, 1), ylabel='Macro AUROC', title=f'GenImage | {size} x {size}')
    ax.set_xticks(x, [METHOD_LABELS[metric] for metric in plotted], rotation=20, ha='right')
    ax.legend(ncols=min(4, len(views)), loc='lower center', bbox_to_anchor=(.5, 1.08), frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(figures / 'macro_auroc.png', dpi=180)
    plt.close(fig)


def analyze_run(run, config, manifest, *, save_reports=True):
    metrics = config['metrics']
    legacy = config['score_schema'] == 1
    views = ['gram'] if legacy else config['views']
    print(f'\n{run} ({config["size"]} px)', flush=True)
    values = read_scores(run, manifest, views, metrics, legacy, save_audit=save_reports)
    if legacy:
        num_blocks = config['num_blocks']
        archive = np.load(run / 'block_cls_grams.npy', mmap_mode='r', allow_pickle=False)
        expected = (len(manifest), 2, math.comb(num_blocks, 2))
        if archive.shape != expected or archive.dtype != np.float32 or not np.isfinite(archive).all():
            raise ValueError(f'Expected a finite float32 CLS Gram archive of shape {expected}')

    generators = np.array([row['generator'] for row in manifest])
    labels = np.array([row['label'] for row in manifest])
    if not len(labels) or not np.isin(labels, ['real', 'fake']).all():
        raise ValueError(f'{run}: manifest requires valid Real and Fake labels')

    summary = []
    aurocs = {}
    names = sorted(set(generators))
    for generator in names:
        selected = np.flatnonzero(generators == generator)
        fake = labels[selected] == 'fake'
        n_fake = int(fake.sum())
        n_real = len(fake) - n_fake
        if not (n_real and n_fake):
            raise ValueError(f'{run}: {generator} needs both Real and Fake images')
        for i, view in enumerate(views):
            for j, metric in enumerate(metrics):
                auc = roc_auc_score(fake, values[i, selected, j])
                aurocs[view, metric, generator] = auc
                summary.append(dict(generator=generator, view=view, metric=metric,
                                    n_real=n_real, n_fake=n_fake, auroc_fake_high=auc))
    for view in views:
        for metric in metrics:
            aurocs[view, metric, MACRO] = mean(aurocs[view, metric, name] for name in names)
    if not save_reports:
        return aurocs
    write_csv(run / 'summary.csv', ['generator', 'view', 'metric', 'n_real', 'n_fake', 'auroc_fake_high'], summary)
    report(summary, run, plot=legacy)
    if legacy:
        plot_block_grams(archive, manifest, num_blocks, run)
    else:
        plot_macro(run, config['size'], views, metrics, aurocs)

    if config['transforms'] == 'gram':
        comparison = [dict(view=view, metric=metric, generator=generator, auroc=auc,
                           final_cls_auroc=aurocs[view, FINAL_CLS, generator],
                           delta_vs_final_cls=auc - aurocs[view, FINAL_CLS, generator])
                      for (view, metric, generator), auc in aurocs.items()]
        write_csv(run / 'delta_vs_final_cls.csv',
                  ['view', 'metric', 'generator', 'auroc', 'final_cls_auroc', 'delta_vs_final_cls'], comparison)
        block_metrics = [f'block_{i:02d}_cos' for i in range(1, config['num_blocks'] + 1)]
        for view in views:
            block = max(block_metrics, key=lambda metric: aurocs[view, metric, MACRO])
            print(f'{view}: best individual block {block} ({aurocs[view, block, MACRO]:.4f})')
    return aurocs


def load_run(run):
    config = json.loads((run / 'config.json').read_text(encoding='utf-8'))
    kind = config.get('transforms'), config.get('score_schema')
    if kind not in (('gram', 1), ('gram', 2), ('official_rigid', 2), ('warpad', 2)):
        raise ValueError(f'{run} is not a supported Gram, official RIGID, or WaRPAD run')
    return run, config


def compare_backbones(runs, results, vit_b_runs, manifest, output):
    generators = sorted({row['generator'] for row in manifest})
    metrics = [(metric, metric) for metric in METHOD_LABELS if '_even_' not in metric]
    metrics.extend((f'block_cls_gram_{score}', f'block_cls_gram_even_{score}')
                   for score in ('cos', 'l1', 'norm_drop'))
    comparison = []
    for (_, cfg_l), aurocs_l, (run_b, cfg_b) in zip(runs, results, vit_b_runs):
        aurocs_b = analyze_run(run_b, cfg_b, manifest, save_reports=False)
        for view in cfg_l['views']:
            for metric_b, metric_l in metrics:
                values_b = {g: aurocs_b[view, metric_b, g] for g in [*generators, MACRO]}
                values_l = {g: aurocs_l[view, metric_l, g] for g in [*generators, MACRO]}
                for values in (values_b, values_l):
                    values['worst'] = min(values[g] for g in generators)
                for generator, value_b in values_b.items():
                    value_l = values_l[generator]
                    comparison.append(dict(size=cfg_l['size'], view=view, metric_vit_b=metric_b,
                                           metric_vit_l=metric_l, generator=generator, auroc_vit_b=value_b,
                                           auroc_vit_l=value_l, delta_vit_l_minus_b=value_l - value_b))
    path = output / 'backbone_comparison.csv'
    write_csv(path, ['size', 'view', 'metric_vit_b', 'metric_vit_l', 'generator',
                     'auroc_vit_b', 'auroc_vit_l', 'delta_vit_l_minus_b'], comparison)
    print(f'\nMatched ViT-B / ViT-L comparison: {path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, nargs='+', required=True, help='One run or the matched 224 and 512 Gram runs')
    parser.add_argument('--rigid-run', type=Path, help='Optional matched official RIGID run for comparison')
    parser.add_argument('--vit-b-run', type=Path, nargs='+', help='ViT-B references for the selected ViT-L sizes')
    args = parser.parse_args()
    if len(args.run) > 2:
        parser.error('--run accepts one run or a matched pair')
    runs = sorted((load_run(run) for run in args.run), key=lambda item: item[1]['size'])
    manifest = read_manifest(runs[0][0])
    if len(runs) == 2:
        (small, cfg_small), (large, cfg_large) = runs
        if (cfg_small['size'], cfg_large['size']) != (224, 512):
            raise ValueError('Resolution comparison requires one 224 run and one 512 run')
        matched = (*GRAM_PROTOCOL, 'transforms', 'score_schema', 'model_id', 'model_revision', 'num_blocks',
                   'metrics', 'primary_metrics', 'block_cls_gram_subset_blocks')
        differing = [key for key in matched if cfg_small.get(key) != cfg_large.get(key)]
        if differing:
            raise ValueError('Resolution comparison settings differ in ' + ', '.join(differing))
        if manifest != read_manifest(large):
            raise ValueError('Resolution comparison requires the same manifest at both sizes')

    if args.vit_b_run:
        vit_b_runs = sorted((load_run(run) for run in args.vit_b_run), key=lambda item: item[1]['size'])
        if [cfg['size'] for _, cfg in vit_b_runs] != [cfg['size'] for _, cfg in runs]:
            parser.error('--vit-b-run requires one ViT-B run per ViT-L resolution')
        for (_, cfg_l), (run_b, cfg_b) in zip(runs, vit_b_runs):
            if (cfg_l['model_id'], cfg_b['model_id']) != (VIT_L_ID, VIT_B_ID):
                parser.error('--vit-b-run compares ViT-L inputs to --run with ViT-B references')
            differing = [key for key in (*GRAM_PROTOCOL, 'score_schema', 'blur_sigma')
                         if cfg_l.get(key) != cfg_b.get(key)]
            if differing:
                raise ValueError('Backbone comparison settings differ in ' + ', '.join(differing))
            if manifest != read_manifest(run_b):
                raise ValueError('Backbone comparison requires the same manifest for ViT-B and ViT-L')

    if args.rigid_run:
        rigid_run, rigid_config = load_run(args.rigid_run)
        if rigid_config['transforms'] != 'official_rigid' or any(cfg['transforms'] not in ('gram', 'warpad') for _, cfg in runs):
            parser.error('--rigid-run requires an official RIGID run and Gram or WaRPAD inputs to --run')
        if manifest != read_manifest(rigid_run):
            raise ValueError('Official RIGID comparison requires the same manifest as the Gram runs')

    results = [analyze_run(run, config, manifest) for run, config in runs]
    output = runs[-1][0]
    if len(results) == 2:
        small, large = results
        comparison = [dict(view=view, metric=metric, generator=generator, auroc_224=value,
                           auroc_512=large[view, metric, generator], delta_512_minus_224=large[view, metric, generator] - value)
                      for (view, metric, generator), value in small.items()]
        write_csv(output / 'resolution_comparison.csv',
                  ['view', 'metric', 'generator', 'auroc_224', 'auroc_512', 'delta_512_minus_224'], comparison)
        print(f'\nMatched resolution comparison: {output / "resolution_comparison.csv"}')
        for row in comparison:
            if row['generator'] == MACRO:
                print(f'{row["view"]:<16} {row["metric"]:<26} {row["auroc_224"]:.4f}  {row["auroc_512"]:.4f}  '
                      f'{row["delta_512_minus_224"]:+.4f}')

    if args.rigid_run:
        rigid = analyze_run(rigid_run, rigid_config, manifest)
        comparison = []
        for (_, config), result in zip(runs, results):
            for (view, metric, generator), value in result.items():
                baseline = rigid['official_rigid', FINAL_CLS, generator]
                comparison.append(dict(size=config['size'], view=view, metric=metric, generator=generator,
                                       auroc=value, official_rigid_auroc=baseline, delta_vs_official_rigid=value - baseline))
        write_csv(output / 'official_rigid_comparison.csv',
                  ['size', 'view', 'metric', 'generator', 'auroc', 'official_rigid_auroc', 'delta_vs_official_rigid'], comparison)
        print(f'\nOfficial RIGID macro AUROC: {rigid["official_rigid", FINAL_CLS, MACRO]:.4f}')
        print(f'Matched official RIGID comparison: {output / "official_rigid_comparison.csv"}')

    if args.vit_b_run:
        compare_backbones(runs, results, vit_b_runs, manifest, output)


if __name__ == '__main__':
    main()
