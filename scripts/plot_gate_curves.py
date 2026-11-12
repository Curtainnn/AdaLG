#!/usr/bin/env python
import argparse
import csv
import glob
import os
import re
import statistics
from collections import defaultdict


FLOAT = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
SEED_RE = re.compile(r'(?:Running with seed=|seed)(\d+)')
EPOCH_RE = re.compile(r'Epoch:\s*(?P<epoch>\d+),\s+Steps:')
LOCAL_GATE_RE = re.compile(
    rf'\[local_gate\]\s+mean=(?P<mean>{FLOAT})\s+std=(?P<std>{FLOAT})\s+'
    rf'max=(?P<max>{FLOAT})\s+min=(?P<min>{FLOAT})\s+'
    rf'\[gate_max\]\s+(?P<gate_max>{FLOAT})\s+has_local=(?P<has_local>{FLOAT})'
)


def parse_args():
    parser = argparse.ArgumentParser(description='Plot multiple local-gate mean curves in one figure.')
    parser.add_argument(
        '--series',
        action='append',
        required=True,
        help='one curve in label=glob form, e.g. USHCN=logs/ushcn_full*.log; repeat for multiple curves',
    )
    parser.add_argument('--out', default='analysis/gate_mean_curves.png', help='output image path')
    parser.add_argument('--title', default='Local Gate Mean Curves', help='figure title')
    parser.add_argument('--csv', default='', help='optional output CSV path for plotted curve values')
    return parser.parse_args()


def expand_pattern(pattern):
    matches = glob.glob(pattern)
    if matches:
        return sorted(matches)
    if os.path.exists(pattern):
        return [pattern]
    return []


def parse_series_arg(value):
    if '=' not in value:
        raise ValueError(f'--series must use label=glob form, got: {value}')
    label, pattern = value.split('=', 1)
    label = label.strip()
    pattern = pattern.strip()
    if not label or not pattern:
        raise ValueError(f'--series must use non-empty label=glob form, got: {value}')
    return label, pattern


def parse_gate_rows(paths):
    rows = []
    for path in paths:
        seed = ''
        epoch = ''
        seed_match = SEED_RE.search(path)
        if seed_match:
            seed = seed_match.group(1)
        with open(path, 'r', encoding='utf-8', errors='ignore') as file_obj:
            for line in file_obj:
                line_seed = SEED_RE.search(line)
                if line_seed:
                    seed = line_seed.group(1)
                line_epoch = EPOCH_RE.search(line)
                if line_epoch:
                    epoch = int(line_epoch.group('epoch'))
                gate_match = LOCAL_GATE_RE.search(line)
                if gate_match and epoch != '':
                    rows.append({
                        'source': path,
                        'seed': seed,
                        'epoch': int(epoch),
                        'gate_mean': float(gate_match.group('mean')),
                        'gate_std': float(gate_match.group('std')),
                        'gate_max_observed': float(gate_match.group('max')),
                        'gate_min_observed': float(gate_match.group('min')),
                        'gate_max_bound': float(gate_match.group('gate_max')),
                        'has_local': float(gate_match.group('has_local')),
                    })
    return rows


def aggregate_rows(rows):
    by_epoch = defaultdict(list)
    gate_bounds = []
    for row in rows:
        by_epoch[int(row['epoch'])].append(float(row['gate_mean']))
        gate_bounds.append(float(row['gate_max_bound']))
    epochs = sorted(by_epoch)
    curve = []
    for epoch in epochs:
        values = by_epoch[epoch]
        mean = statistics.mean(values)
        std = statistics.pstdev(values) if len(values) > 1 else 0.0
        curve.append({
            'epoch': epoch,
            'mean': mean,
            'std': std,
            'n': len(values),
        })
    bound = statistics.mean(gate_bounds) if gate_bounds else None
    return curve, bound


def write_curve_csv(path, curves):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=['label', 'epoch', 'gate_mean', 'gate_std', 'n'])
        writer.writeheader()
        for label, curve, _ in curves:
            for row in curve:
                writer.writerow({
                    'label': label,
                    'epoch': row['epoch'],
                    'gate_mean': row['mean'],
                    'gate_std': row['std'],
                    'n': row['n'],
                })


def plot_curves(path, title, curves):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f'Could not import matplotlib: {exc}') from exc

    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    plt.figure(figsize=(7.5, 4.4))
    bounds = []
    for label, curve, bound in curves:
        if not curve:
            continue
        epochs = [row['epoch'] for row in curve]
        means = [row['mean'] for row in curve]
        stds = [row['std'] for row in curve]
        line = plt.plot(epochs, means, linewidth=2.0, label=label)[0]
        if any(std > 0.0 for std in stds):
            lower = [mean - std for mean, std in zip(means, stds)]
            upper = [mean + std for mean, std in zip(means, stds)]
            plt.fill_between(epochs, lower, upper, color=line.get_color(), alpha=0.14)
        if bound is not None:
            bounds.append(bound)

    if bounds:
        gate_bound = statistics.mean(bounds)
        plt.axhline(gate_bound, color='black', linestyle='--', linewidth=1.2, label=f'Gate max={gate_bound:.2f}')
    plt.xlabel('Epoch')
    plt.ylabel('Local gate mean')
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def main():
    args = parse_args()
    curves = []
    for series_value in args.series:
        label, pattern = parse_series_arg(series_value)
        paths = expand_pattern(pattern)
        if not paths:
            print(f'[warn] no files matched {label}: {pattern}')
            curves.append((label, [], None))
            continue
        rows = parse_gate_rows(paths)
        curve, bound = aggregate_rows(rows)
        print(f'{label}: files={len(paths)} gate_rows={len(rows)} epochs={len(curve)}')
        curves.append((label, curve, bound))

    if not any(curve for _, curve, _ in curves):
        raise SystemExit('No gate rows found. Check that logs contain [local_gate] lines.')

    plot_curves(args.out, args.title, curves)
    if args.csv:
        write_curve_csv(args.csv, curves)
    print(f'plot: {args.out}')
    if args.csv:
        print(f'csv: {args.csv}')


if __name__ == '__main__':
    raise SystemExit(main())
