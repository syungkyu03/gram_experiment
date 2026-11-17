import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    run = Path(args.run)
    with (run / 'scores.csv').open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    metrics = ['static_bottom10', 'gram_l1', 'gram_signed', 'gram_near', 'gram_far', 'gram_bb', 'cka_change', 'patch_change', 'cls_change']
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


if __name__ == '__main__':
    main()
