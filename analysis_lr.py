import argparse
import csv
from collections import defaultdict
from pathlib import Path

from analyze import report, summarize_aurocs
from analysis_details import audit_scores, number, write_csv


METRICS = ['lr_cross_ratio', 'far_patch', 'near_patch']
RESPONSE_FIELDS = ['response_ll', 'response_lr', 'response_rl', 'response_rr']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    run = Path(args.run)
    with (run / 'scores.csv').open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    audit = audit_scores(run, rows, [*METRICS, *RESPONSE_FIELDS], ['lr'])
    mismatches = ('missing_rows', 'unexpected_rows', 'duplicate_rows', 'duplicate_manifest_rows')
    if any(audit[key] for key in mismatches):
        raise ValueError('LR score rows do not match the manifest. See audit.json.')

    # Use the same images for all metrics when a near response is zero.
    groups = defaultdict(list)
    for row in rows:
        if all(number(row, metric) is not None for metric in METRICS):
            groups[(row['generator'], row['view'])].append(row)
    kept = sum(map(len, groups.values()))
    print(f'Paired LR analysis uses {kept}/{len(rows)} images.', flush=True)
    for row in rows:
        key = (row['generator'], row['view'])
        if key not in groups:
            raise ValueError(f'No finite paired scores for {row["generator"]}. See audit.json.')
    summary = summarize_aurocs(groups, METRICS)
    if not summary or len(summary) != len(groups) * len(METRICS):
        raise ValueError('Each generator needs paired Real and Fake scores. See audit.json.')
    write_csv(run / 'summary.csv', ['generator', 'view', 'metric', 'n_real', 'n_fake', 'auroc_fake_high'], summary)
    report(summary, run)

    lookup = {(row['generator'], row['metric']): row for row in summary}
    comparisons = []
    for generator, _ in sorted(groups):
        ratio = lookup[generator, 'lr_cross_ratio']
        far = lookup[generator, 'far_patch']
        comparisons.append(dict(generator=generator, n_real=ratio['n_real'], n_fake=ratio['n_fake'],
                                cross_ratio_auroc=ratio['auroc_fake_high'], far_patch_auroc=far['auroc_fake_high'],
                                auc_difference=ratio['auroc_fake_high'] - far['auroc_fake_high']))
    write_csv(run / 'lr_comparison.csv', ['generator', 'n_real', 'n_fake', 'cross_ratio_auroc',
                                          'far_patch_auroc', 'auc_difference'], comparisons)


if __name__ == '__main__':
    main()
