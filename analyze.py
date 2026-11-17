import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from analysis_details import detailed_analysis


def report(summary, run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if not summary:
        print('No AUROC values: each group needs both Real and Fake scores.')
        return
    generators = sorted({row['generator'] for row in summary})
    views = sorted({row['view'] for row in summary})
    figures = run / 'figures'
    figures.mkdir(exist_ok=True)
    for view in views:
        entries = [row for row in summary if row['view'] == view]
        methods = sorted({row['metric'] for row in entries})
        lookup = {(row['metric'], row['generator']): row['auroc_fake_high'] for row in entries}
        matrix = np.array([[lookup.get((method, generator), np.nan) for generator in generators] for method in methods])
        means = [float(np.mean(row)) if np.isfinite(row).all() else float('nan') for row in matrix]
        widths = [max(len(g), 6) for g in generators]
        print(f'\n{view}  (AUROC: Fake score higher; macro mean requires all generators)')
        print(f"{'Metric':<18} " + ' '.join(f'{g:>{w}}' for g, w in zip(generators, widths)) + '   Mean')
        for method, values, mean in zip(methods, matrix, means):
            cells = [f'{v:.3f}' if np.isfinite(v) else '-' for v in values]
            mean_text = f'{mean:.3f}' if np.isfinite(mean) else '-'
            print(f'{method:<18} ' + ' '.join(f'{cell:>{w}}' for cell, w in zip(cells, widths)) + f' {mean_text:>6}')
        with (run / f'auroc_{view}.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', *generators, 'macro_mean'])
            for method, values, mean in zip(methods, matrix, means):
                writer.writerow([method, *[f'{x:.6f}' if np.isfinite(x) else '' for x in values], f'{mean:.6f}' if np.isfinite(mean) else ''])
        fig, ax = plt.subplots(figsize=(max(7, len(generators) * 1.3), max(3, len(methods) * .5 + 1.5)))
        image = ax.imshow(np.ma.masked_invalid(matrix), cmap='RdBu_r', vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(range(len(generators)), generators, rotation=35, ha='right')
        ax.set_yticks(range(len(methods)), methods)
        ax.set_title(f'{view}: AUROC (Fake score higher)')
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                ax.text(j, i, f'{value:.2f}' if np.isfinite(value) else '-', ha='center', va='center', color='white' if np.isfinite(value) and (value < .25 or value > .75) else 'black')
        fig.colorbar(image, ax=ax, label='AUROC')
        fig.tight_layout()
        fig.savefig(figures / f'auroc_{view}.png', dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--bootstrap', type=int, default=200, help='Stratified bootstrap repetitions; 0 disables')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    run = Path(args.run)
    with (run / 'scores.csv').open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    metrics = ['static_bottom10', 'gram_l1', 'gram_signed', 'gram_near', 'gram_far', 'gram_aa', 'gram_ab', 'gram_bb', 'cka_change', 'patch_change', 'patch_change_a', 'patch_change_b', 'cls_change', 'cls_cosine_change']
    metrics = [metric for metric in metrics if rows and metric in rows[0]]
    groups = defaultdict(list)
    for row in rows:
        groups[(row['generator'], row['view'])].append(row)
    summary = []
    for (generator, view), items in sorted(groups.items()):
        for metric in metrics:
            pairs = [(int(r['label'] == 'fake'), float(r[metric])) for r in items if r.get(metric) and math.isfinite(float(r[metric]))]
            if len({y for y, _ in pairs}) != 2:
                continue
            summary.append({'generator': generator, 'view': view, 'metric': metric, 'n_real': sum(y == 0 for y, _ in pairs), 'n_fake': sum(y == 1 for y, _ in pairs), 'auroc_fake_high': roc_auc_score([y for y, _ in pairs], [v for _, v in pairs])})
    with (run / 'summary.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['generator', 'view', 'metric', 'n_real', 'n_fake', 'auroc_fake_high'])
        writer.writeheader()
        writer.writerows(summary)
    correlations = []
    for (generator, view), items in sorted(groups.items()):
        for metric in metrics:
            for baseline in ('patch_change', 'cls_change'):
                if metric == baseline:
                    continue
                pairs = [(float(r[metric]), float(r[baseline])) for r in items if r.get(metric) and r.get(baseline) and math.isfinite(float(r[metric])) and math.isfinite(float(r[baseline]))]
                if len(pairs) > 1:
                    x, y = np.asarray(pairs).T
                    if x.std() and y.std():
                        correlations.append({'generator': generator, 'view': view, 'metric': metric, 'baseline': baseline, 'n': len(pairs), 'pearson': np.corrcoef(x, y)[0, 1]})
    with (run / 'correlations.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['generator', 'view', 'metric', 'baseline', 'n', 'pearson'])
        writer.writeheader()
        writer.writerows(correlations)
    print(f'Wrote {len(summary)} AUROC rows and {len(correlations)} correlations')
    report(summary, run)
    detailed_analysis(run, rows, metrics, args.bootstrap, args.seed)


if __name__ == '__main__':
    main()
