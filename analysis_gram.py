import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

from analysis_details import audit_scores, write_csv
from analyze import report, summarize_aurocs


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


def analyze_run(run, config):
    with (run / 'scores.csv').open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    metrics = config['metrics']
    print(f'\n{run} ({config["size"]} px)', flush=True)
    audit = audit_scores(run, rows, metrics, ['gram'])
    mismatches = ('missing_rows', 'unexpected_rows', 'duplicate_rows', 'duplicate_manifest_rows')
    if any(audit[key] for key in mismatches) or any(audit['nonfinite_or_missing'].values()):
        raise ValueError('Gram run is incomplete or contains invalid scores. See audit.json.')

    # The Gram archive is in manifest order. Check before using labels for its rows.
    with (run / 'manifest.csv').open(newline='', encoding='utf-8') as f:
        for index, (score, sample) in enumerate(zip(rows, csv.DictReader(f)), 1):
            if any(score[key] != sample[key] for key in ('path', 'label', 'generator', 'class_name')):
                raise ValueError(f'Gram score row {index} does not match the manifest order')
    num_blocks = config['num_blocks']
    archive = np.load(run / 'block_cls_grams.npy', mmap_mode='r', allow_pickle=False)
    expected = (len(rows), 2, math.comb(num_blocks, 2))
    if archive.shape != expected or archive.dtype != np.float32 or not np.isfinite(archive).all():
        raise ValueError(f'Expected a finite float32 CLS Gram archive of shape {expected}')

    groups = defaultdict(list)
    for row in rows:
        groups[row['generator'], row['view']].append(row)
    summary = summarize_aurocs(groups, metrics)
    if not summary or len(summary) != len(groups) * len(metrics):
        raise ValueError('Each generator needs Real and Fake scores')
    write_csv(run / 'summary.csv', ['generator', 'view', 'metric', 'n_real', 'n_fake', 'auroc_fake_high'], summary)
    report(summary, run)
    plot_block_grams(archive, rows, num_blocks, run)

    aucs = defaultdict(list)
    for row in summary:
        aucs[row['metric']].append(row['auroc_fake_high'])
    macro = {metric: mean(values) for metric, values in aucs.items()}
    best_block = max((f'block_{i:02d}_cos' for i in range(1, num_blocks + 1)), key=macro.__getitem__)
    print(f'Best individual block on this run {best_block} ({macro[best_block]:.4f})')
    return macro


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, nargs='+', required=True, help='One run or the matched 224 and 512 runs')
    args = parser.parse_args()
    if len(args.run) > 2:
        parser.error('--run accepts one run or a matched pair')
    runs = [(run, json.loads((run / 'config.json').read_text(encoding='utf-8'))) for run in args.run]
    for run, config in runs:
        if config.get('transforms') != 'gram' or config.get('score_schema') != 1:
            raise ValueError(f'{run} is not a supported Gram run')
    if len(runs) == 2:
        runs.sort(key=lambda item: item[1]['size'])
        (small, cfg_small), (large, cfg_large) = runs
        if (cfg_small['size'], cfg_large['size']) != (224, 512):
            raise ValueError('Resolution comparison requires one 224 run and one 512 run')
        matched = ('model_id', 'model_revision', 'resize', 'noise_space', 'noise_std', 'noise_seed_view',
                   'seed', 'block_cls_source', 'num_blocks', 'patch_size', 'metrics', 'primary_metrics')
        differing = [key for key in matched if cfg_small[key] != cfg_large[key]]
        if differing:
            raise ValueError('Resolution comparison settings differ in ' + ', '.join(differing))
        if (small / 'manifest.csv').read_bytes() != (large / 'manifest.csv').read_bytes():
            raise ValueError('Resolution comparison requires the same manifest at both sizes')

    results = [analyze_run(run, config) for run, config in runs]
    if len(results) == 2:
        print('\nMatched manifest macro AUROC')
        print(f'{"Metric":<20} {"224":>8} {"512":>8} {"512 - 224":>10}')
        for metric in runs[0][1]['primary_metrics']:
            small, large = (result[metric] for result in results)
            print(f'{metric:<20} {small:8.4f} {large:8.4f} {large-small:+10.4f}')


if __name__ == '__main__':
    main()
