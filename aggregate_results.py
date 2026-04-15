"""
Aggregate all benchmark results into a single paper-ready markdown table.

Reads every `results/<benchmark>_*_s<seed>/results.json` on disk, groups by
(benchmark, model), computes mean +/- std across seeds, and prints a
markdown table + a JSON dump suitable for dropping into EVIDENCE_PACKAGE.md
or the paper draft.

Usage:
    python aggregate_results.py                  # print markdown table
    python aggregate_results.py --json           # also dump JSON to stdout
    python aggregate_results.py --save paper_table.md  # write to file

The script discovers result files automatically. If new benchmarks get added
(e.g. overnight CLUTRR runs), they'll show up without any code changes as
long as they follow the naming convention results/<tag>_s<seed>/results.json.
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Benchmark row specifications
# ---------------------------------------------------------------------------
# Each row specifies: (display name, dir glob pattern, metric path within
# the results.json, format hint). The aggregator discovers all result files
# matching the glob, extracts the metric, and groups by architecture.

ROW_SPECS = [
    # Graph 12M
    {
        'name': 'Graph 12M V4 OOD',
        'glob': 'hwta_v4_d768*',
        'metric': lambda j: j['circuit']['ood_acc'],
        'percentage': False,
    },
    {
        'name': 'Graph 12M TF OOD',
        'glob': 'hwta_12m_tf*',
        'metric': lambda j: j.get('ood_acc') or j.get('transformer', {}).get('ood_acc'),
        'percentage': False,
    },
    # Graph 100M
    {
        'name': 'Graph 100M V4 OOD',
        'glob': 'hwta_v4_d2048*',
        'metric': lambda j: j['circuit']['ood_acc'],
        'percentage': False,
    },
    {
        'name': 'Graph 100M V4 extrap (d20+d30)/2',
        'glob': 'hwta_v4_d2048*',
        'metric': lambda j: (j['circuit']['depth_20'] + j['circuit']['depth_30']) / 2,
        'percentage': False,
    },
    {
        'name': 'Graph 100M TF OOD',
        'glob': 'hwta_v4_d2048',  # transformer baseline lives in same file
        'metric': lambda j: j.get('transformer', {}).get('ood_acc'),
        'percentage': False,
    },
    # Graph 1.5B (scaling)
    {
        'name': 'Graph 1.5B V4 OOD',
        'glob': 'hwta_v4_d8192*',
        'metric': lambda j: j['circuit']['ood_acc'],
        'percentage': False,
    },
    # SCAN V4b
    {
        'name': 'SCAN V4b jump_comp',
        'glob': 'scan_v4b_d32_s*',
        'metric': lambda j: j['final']['acc_jump_comp'],
        'percentage': True,
    },
    # SCAN V4c
    {
        'name': 'SCAN V4c jump_comp',
        'glob': 'scan_v4c_d32_h128_s*',
        'metric': lambda j: j['final']['acc_jump_comp'],
        'percentage': True,
    },
    # SCAN Tree TF (small, matched)
    {
        'name': 'SCAN TF d=128 L=4 jump_comp',
        'glob': 'scan_tree_tf_d128_l4_s*',
        'metric': lambda j: j['final']['acc_jump_comp'],
        'percentage': True,
    },
    # SCAN Tree TF (large, bigger-is-worse)
    {
        'name': 'SCAN TF d=256 L=8 jump_comp',
        'glob': 'scan_tree_tf_d256_l8_s*',
        'metric': lambda j: j['final']['acc_jump_comp'],
        'percentage': True,
    },
    # CruxMini V4
    {
        'name': 'CruxMini V4 test_mul',
        'glob': 'cruxmini_s*',
        'metric': lambda j: j['final']['acc_test_mul'],
        'percentage': True,
    },
    # CruxMini Tree TF (small, matched)
    {
        'name': 'CruxMini TF d=128 L=4 test_mul',
        'glob': 'cruxmini_tf_d128_l4_s*',
        'metric': lambda j: j['final']['acc_test_mul'],
        'percentage': True,
    },
    # CruxMini Tree TF (large)
    {
        'name': 'CruxMini TF d=256 L=8 test_mul',
        'glob': 'cruxmini_tf_d256_l8_s*',
        'metric': lambda j: j['final']['acc_test_mul'],
        'percentage': True,
    },
    # CLUTRR V4 (overnight runs)
    {
        'name': 'CLUTRR V4 overall test',
        'glob': 'clutrr_v4_d*_s*',
        'metric': lambda j: j.get('best_test_acc') or max(h['test_acc'] for h in j['history']),
        'percentage': True,
    },
    {
        'name': 'CLUTRR V4 k=10',
        'glob': 'clutrr_v4_d*_s*',
        'metric': lambda j: j.get('best_per_task_acc', {}).get('task_1.10') or
                            max(h['per_task_acc'].get('task_1.10', 0) for h in j['history']),
        'percentage': True,
    },
    # CLUTRR TF (overnight runs)
    {
        'name': 'CLUTRR TF overall test',
        'glob': 'clutrr_tf_d*_s*',
        'metric': lambda j: j.get('best_test_acc') or max(h['test_acc'] for h in j['history']),
        'percentage': True,
    },
    {
        'name': 'CLUTRR TF k=10',
        'glob': 'clutrr_tf_d*_s*',
        'metric': lambda j: j.get('best_per_task_acc', {}).get('task_1.10') or
                            max(h['per_task_acc'].get('task_1.10', 0) for h in j['history']),
        'percentage': True,
    },
    # Existing single-seed CLUTRR (for back-compat with older runs)
    {
        'name': 'CLUTRR V4 (legacy single-seed)',
        'glob': 'clutrr_v4',
        'metric': lambda j: max(h['test_acc'] for h in j['history']),
        'percentage': True,
    },
]


# Documented failed runs to exclude from aggregation.
# These are runs that are known-bad per primer / EVIDENCE_PACKAGE and should
# not be included in the published paper table, even though their result.json
# files still exist on disk.
#
# RULE: only add a run here if there is a documented reason it's a known
# failure (degenerate routing, overwritten file, config bug, etc.). Do NOT
# add runs just because the numbers are low.
FAILED_RUNS = {
    # Graph 100M V4 @ lr=3e-5 seed-2: primer says this was a DEGENERATE routing
    # run (depth-3 below random, OOD=0.6157). The reproducible recipe is lr=2e-5.
    # Reporting this seed as part of the graph 100M row is dishonest because the
    # primer documents that lr=3e-5 is unreliable across seeds. This failed run
    # is a methodological footnote, not a result.
    'hwta_v4_d2048_seed2',
}


def discover_results(glob_pattern):
    """Find every results.json under results/<glob_pattern>/.

    Skips directories listed in FAILED_RUNS (documented known failures).
    """
    base = Path('results')
    if not base.exists():
        return []
    # Handle both exact matches and wildcards
    if '*' in glob_pattern:
        dirs = sorted(base.glob(glob_pattern))
    else:
        d = base / glob_pattern
        dirs = [d] if d.is_dir() else []
    hits = []
    for d in dirs:
        if d.name in FAILED_RUNS:
            print(f"  [excluded] {d.name} (documented failed run)")
            continue
        p = d / 'results.json'
        if p.exists():
            try:
                with open(p) as f:
                    hits.append((d.name, json.load(f)))
            except Exception as e:
                print(f"  WARN: failed to read {p}: {e}")
    return hits


def compute_stats(values):
    """Return (mean, sample_std, n)."""
    n = len(values)
    if n == 0:
        return None, None, 0
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0, 1
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var), n


def aggregate():
    rows = []
    for spec in ROW_SPECS:
        hits = discover_results(spec['glob'])
        if not hits:
            rows.append({
                'name': spec['name'],
                'glob': spec['glob'],
                'status': 'NO DATA',
                'values': [],
                'mean': None,
                'std': None,
                'n': 0,
            })
            continue

        values = []
        seeds_seen = []
        for dir_name, j in hits:
            try:
                v = spec['metric'](j)
                if v is None:
                    continue
                values.append(float(v))
                seeds_seen.append(dir_name)
            except Exception as e:
                print(f"  WARN: metric extraction failed for {dir_name}: {e}")

        mean, std, n = compute_stats(values)
        rows.append({
            'name': spec['name'],
            'glob': spec['glob'],
            'status': 'OK' if n > 0 else 'NO DATA',
            'values': values,
            'seeds': seeds_seen,
            'mean': mean,
            'std': std,
            'n': n,
            'percentage': spec.get('percentage', False),
        })
    return rows


def format_table(rows):
    """Markdown table with mean +/- std and per-seed values."""
    lines = []
    lines.append('| Benchmark | Mean +/- std | n | Per-seed values |')
    lines.append('|---|---|---:|---|')
    for row in rows:
        name = row['name']
        n = row['n']
        if n == 0 or row['mean'] is None:
            lines.append(f'| {name} | **NO DATA** | 0 | ({row["glob"]}) |')
            continue
        mean = row['mean']
        std = row['std']
        if row['percentage']:
            mean_str = f'{mean*100:.1f}'
            std_str = f'{std*100:.2f}' if std is not None else '--'
            values_str = ', '.join(f'{v*100:.1f}' for v in row['values'])
        else:
            mean_str = f'{mean:.4f}'
            std_str = f'{std:.4f}' if std is not None else '--'
            values_str = ', '.join(f'{v:.4f}' for v in row['values'])
        lines.append(f'| {name} | **{mean_str} +/- {std_str}** | {n} | [{values_str}] |')
    return '\n'.join(lines)


def format_paper_table(rows):
    """The clean paper-ready table: one row per benchmark, V4 and TF side-by-side."""
    rows_by_name = {r['name']: r for r in rows}

    def get(name):
        r = rows_by_name.get(name)
        if not r or r['mean'] is None:
            return 'NO DATA'
        if r['percentage']:
            if r['std'] is None or r['n'] < 2:
                return f'{r["mean"]*100:.1f}'
            return f'{r["mean"]*100:.1f} +/- {r["std"]*100:.2f}'
        else:
            if r['std'] is None or r['n'] < 2:
                return f'{r["mean"]:.3f}'
            return f'{r["mean"]:.3f} +/- {r["std"]:.3f}'

    def get_n(name):
        r = rows_by_name.get(name)
        return r['n'] if r else 0

    lines = []
    lines.append('## Paper-ready results table')
    lines.append('')
    lines.append('| Benchmark | V4 | TF matched | seeds (V4/TF) |')
    lines.append('|---|---|---|---|')
    lines.append(
        f'| Graph 12M OOD | {get("Graph 12M V4 OOD")} | {get("Graph 12M TF OOD")} '
        f'| {get_n("Graph 12M V4 OOD")}/{get_n("Graph 12M TF OOD")} |'
    )
    lines.append(
        f'| Graph 100M OOD | {get("Graph 100M V4 OOD")} | {get("Graph 100M TF OOD")} '
        f'| {get_n("Graph 100M V4 OOD")}/{get_n("Graph 100M TF OOD")} |'
    )
    lines.append(
        f'| Graph 100M extrap (d20+d30)/2 | {get("Graph 100M V4 extrap (d20+d30)/2")} | -- '
        f'| {get_n("Graph 100M V4 extrap (d20+d30)/2")}/- |'
    )
    lines.append(
        f'| Graph 1.5B OOD (scaling) | {get("Graph 1.5B V4 OOD")} | -- '
        f'| {get_n("Graph 1.5B V4 OOD")}/- |'
    )
    lines.append(
        f'| SCAN V4b jump_comp | {get("SCAN V4b jump_comp")} | {get("SCAN TF d=128 L=4 jump_comp")} '
        f'| {get_n("SCAN V4b jump_comp")}/{get_n("SCAN TF d=128 L=4 jump_comp")} |'
    )
    lines.append(
        f'| SCAN V4c jump_comp | {get("SCAN V4c jump_comp")} | {get("SCAN TF d=128 L=4 jump_comp")} '
        f'| {get_n("SCAN V4c jump_comp")}/{get_n("SCAN TF d=128 L=4 jump_comp")} |'
    )
    lines.append(
        f'| SCAN (bigger-is-worse) | {get("SCAN V4b jump_comp")} | {get("SCAN TF d=256 L=8 jump_comp")} '
        f'| {get_n("SCAN V4b jump_comp")}/{get_n("SCAN TF d=256 L=8 jump_comp")} |'
    )
    lines.append(
        f'| CruxMini test_mul | {get("CruxMini V4 test_mul")} | {get("CruxMini TF d=128 L=4 test_mul")} '
        f'| {get_n("CruxMini V4 test_mul")}/{get_n("CruxMini TF d=128 L=4 test_mul")} |'
    )
    lines.append(
        f'| CLUTRR overall test | {get("CLUTRR V4 overall test")} | {get("CLUTRR TF overall test")} '
        f'| {get_n("CLUTRR V4 overall test")}/{get_n("CLUTRR TF overall test")} |'
    )
    lines.append(
        f'| CLUTRR k=10 | {get("CLUTRR V4 k=10")} | {get("CLUTRR TF k=10")} '
        f'| {get_n("CLUTRR V4 k=10")}/{get_n("CLUTRR TF k=10")} |'
    )
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', action='store_true',
                        help='Also print raw aggregated data as JSON')
    parser.add_argument('--save', type=str, default=None,
                        help='Write markdown output to this file')
    args = parser.parse_args()

    rows = aggregate()

    out = []
    out.append('# Aggregated Benchmark Results')
    out.append('')
    out.append('## Detailed (every discovered result file)')
    out.append('')
    out.append(format_table(rows))
    out.append('')
    out.append(format_paper_table(rows))
    out.append('')

    markdown = '\n'.join(out)
    print(markdown)

    if args.json:
        print('\n## JSON dump')
        print(json.dumps([
            {k: v for k, v in r.items() if k not in ('percentage',)}
            for r in rows
        ], indent=2, default=str))

    if args.save:
        with open(args.save, 'w') as f:
            f.write(markdown + '\n')
        print(f'\nSaved to {args.save}')


if __name__ == '__main__':
    main()
