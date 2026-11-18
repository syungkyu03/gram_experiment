import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    run = Path(args.run)
    with (run / 'scores.csv').open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    groups = defaultdict(list)
    for row in rows:
        ring = json.loads(row['rings'])
        for metric, column in [('patch', 2), ('gram', 3)]:
            for distance, count, patch, gram in ring:
                groups[(row['generator'], row['selection'], metric, distance)].append((row['label'], [patch, gram][column - 2]))
            values = [entry[column] for entry in ring if entry[0] <= 4]
            if len(values) == 4:
                groups[(row['generator'], row['selection'], metric + '_slope_1_4', 0)].append((row['label'], float(np.polyfit([1, 2, 3, 4], values, 1)[0])))
    output = []
    for (gen, selection, metric, distance), entries in sorted(groups.items()):
        real = [v for label, v in entries if label == 'real']; fake = [v for label, v in entries if label == 'fake']
        if real and fake:
            output.append(dict(generator=gen, selection=selection, metric=metric, ring=distance, n_real=len(real), n_fake=len(fake), real_mean=np.mean(real), fake_mean=np.mean(fake), auroc_fake_high=roc_auc_score([0]*len(real)+[1]*len(fake), real+fake)))
    with (run / 'ring_summary.csv').open('w', newline='', encoding='utf-8') as f:
        fields = ['generator','selection','metric','ring','n_real','n_fake','real_mean','fake_mean','auroc_fake_high']
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(output)
    figures = run / 'figures'; figures.mkdir(exist_ok=True)
    for gen in sorted({r['generator'] for r in output}):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, metric in zip(axes, ['patch', 'gram']):
            for selection in ('high_attention', 'random'):
                data = [r for r in output if r['generator'] == gen and r['metric'] == metric and r['selection'] == selection and r['ring'] > 0]
                for label in ('real', 'fake'):
                    ax.plot([r['ring'] for r in data], [r[label + '_mean'] for r in data], label=f'{selection} {label}')
            ax.set(title=metric, xlabel='Distance from A (patches)', ylabel='Mean change'); ax.legend(fontsize=7)
        fig.suptitle(gen + ' — available images per ring vary'); fig.tight_layout(); fig.savefig(figures / f'{gen}_rings.png', dpi=140); plt.close(fig)
    print(f'Wrote {len(output)} summary rows to {run}')


if __name__ == '__main__':
    main()
