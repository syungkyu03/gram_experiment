import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def write_csv(path, fields, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def number(row, key):
    try:
        value = float(row.get(key, ''))
        return value if np.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def detailed_analysis(run, rows, metrics, repeats, seed):
    if repeats < 0:
        raise ValueError('--bootstrap must be nonnegative')
    run = Path(run)
    rng = np.random.default_rng(seed)
    config = json.loads((run / 'config.json').read_text(encoding='utf-8'))
    views = config['transforms'].split(',')
    with (run / 'manifest.csv').open(newline='', encoding='utf-8') as f:
        manifest = list(csv.DictReader(f))
    key = lambda r: (r['path'], r['generator'], r['label'])
    expected = {(*key(row), view) for row in manifest for view in views}
    actual = Counter((*key(row), row['view']) for row in rows)
    audit = {'manifest_rows': len(manifest), 'expected_score_rows': len(expected), 'actual_score_rows': len(rows),
             'missing_rows': len(expected - actual.keys()), 'unexpected_rows': len(actual.keys() - expected),
             'duplicate_rows': sum(n - 1 for n in actual.values() if n > 1),
             'duplicate_manifest_rows': len(manifest) - len({key(r) for r in manifest}),
             'nonfinite_or_missing': {metric: sum(number(r, metric) is None for r in rows if metric not in ('gram_aa', 'gram_ab', 'gram_bb') or r['view'] == 'local_noise') for metric in metrics}}
    (run / 'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print('Audit:', json.dumps(audit, ensure_ascii=False))
    groups = defaultdict(list)
    for row in rows:
        groups[(row['generator'], row['view'])].append(row)
    intervals, comparisons, correlations, distributions = [], [], [], []
    for (generator, view), items in sorted(groups.items()):
        for metric in metrics:
            usable = [r for r in items if r['label'] in ('real', 'fake') and number(r, metric) is not None]
            for label in ('real', 'fake'):
                vals = np.array([number(r, metric) for r in usable if r['label'] == label])
                if len(vals):
                    q = np.quantile(vals, [.05, .25, .5, .75, .95])
                    distributions.append(dict(generator=generator, view=view, metric=metric, label=label, n=len(vals), mean=vals.mean(), std=vals.std(), q05=q[0], q25=q[1], median=q[2], q75=q[3], q95=q[4]))
            labels = np.array([r['label'] == 'fake' for r in usable], dtype=bool)
            vals = np.array([number(r, metric) for r in usable])
            real, fake = np.flatnonzero(~labels), np.flatnonzero(labels)
            if len(real) and len(fake):
                auc = roc_auc_score(labels, vals)
                draws = []
                if len(real) >= 2 and len(fake) >= 2:
                    for _ in range(repeats):
                        ix = np.concatenate([rng.choice(real, len(real)), rng.choice(fake, len(fake))])
                        draws.append(roc_auc_score(labels[ix], vals[ix]))
                lo, hi = np.quantile(draws, [.025, .975]) if draws else ('', '')
                intervals.append(dict(generator=generator, view=view, metric=metric, n_real=len(real), n_fake=len(fake), auroc=auc, ci_low=lo, ci_high=hi, bootstrap=len(draws)))
            for baseline in ('patch_change', 'cls_change'):
                if metric == baseline:
                    continue
                paired = [r for r in usable if number(r, baseline) is not None]
                for label in ('all', 'real', 'fake'):
                    selected = [r for r in paired if label == 'all' or r['label'] == label]
                    x = np.array([number(r, metric) for r in selected]); y = np.array([number(r, baseline) for r in selected])
                    if len(x) > 1 and x.std() > 0 and y.std() > 0:
                        correlations.append(dict(generator=generator, view=view, metric=metric, baseline=baseline, subset=label, n=len(x), pearson=np.corrcoef(x, y)[0, 1]))
                if metric == 'gram_bb' and view == 'local_noise':
                    y = np.array([r['label'] == 'fake' for r in paired], dtype=bool); a = np.array([number(r, metric) for r in paired]); b = np.array([number(r, baseline) for r in paired])
                    real, fake = np.flatnonzero(~y), np.flatnonzero(y)
                    if len(real) and len(fake):
                        diff = roc_auc_score(y, a) - roc_auc_score(y, b)
                        draws = []
                        if len(real) >= 2 and len(fake) >= 2:
                            for _ in range(repeats):
                                ix = np.concatenate([rng.choice(real, len(real)), rng.choice(fake, len(fake))])
                                draws.append(roc_auc_score(y[ix], a[ix]) - roc_auc_score(y[ix], b[ix]))
                        lo, hi = np.quantile(draws, [.025, .975]) if draws else ('', '')
                        comparisons.append(dict(generator=generator, view=view, metric=metric, baseline=baseline, n_real=len(real), n_fake=len(fake), auc_difference=diff, ci_low=lo, ci_high=hi, bootstrap=len(draws)))
    write_csv(run / 'auroc_intervals.csv', ['generator','view','metric','n_real','n_fake','auroc','ci_low','ci_high','bootstrap'], intervals)
    write_csv(run / 'paired_comparisons.csv', ['generator','view','metric','baseline','n_real','n_fake','auc_difference','ci_low','ci_high','bootstrap'], comparisons)
    write_csv(run / 'correlations_by_label.csv', ['generator','view','metric','baseline','subset','n','pearson'], correlations)
    write_csv(run / 'distributions.csv', ['generator','view','metric','label','n','mean','std','q05','q25','median','q75','q95'], distributions)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures = run / 'figures'
    figures.mkdir(exist_ok=True)
    for view, metric in sorted({(r['view'], r['metric']) for r in distributions}):
        selected = [r for r in distributions if r['view'] == view and r['metric'] == metric]
        generators = sorted({r['generator'] for r in selected})
        fig, ax = plt.subplots(figsize=(max(7, len(generators) * 1.5), 4))
        for label, offset, color in [('real', -.18, 'tab:blue'), ('fake', .18, 'tab:orange')]:
            for i, generator in enumerate(generators):
                match = [r for r in selected if r['generator'] == generator and r['label'] == label]
                if match:
                    r = match[0]
                    ax.bxp([dict(med=r['median'], q1=r['q25'], q3=r['q75'], whislo=r['q05'], whishi=r['q95'], fliers=[])], positions=[i + offset], widths=.3, showfliers=False, patch_artist=True, boxprops=dict(facecolor=color, alpha=.6))
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color='tab:blue', label='Real'), Patch(color='tab:orange', label='Fake')])
        ax.set_xticks(range(len(generators)), generators, rotation=30, ha='right')
        ax.set_title(f'{view} / {metric} (whiskers: 5th–95th percentiles)')
        fig.tight_layout()
        fig.savefig(figures / f'distribution_{view}_{metric}.png', dpi=120)
        plt.close(fig)
    print('Detailed analysis: audit.json, auroc_intervals.csv, paired_comparisons.csv, correlations_by_label.csv, distributions.csv')
