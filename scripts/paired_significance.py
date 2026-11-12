#!/usr/bin/env python
import argparse
import csv
import glob
import math
import os
import re
import statistics


METRIC_RE = re.compile(r'([A-Za-z_]+):([-+0-9.eE]+)')
SEED_RE = re.compile(r'(?:seed|seed=)(\d+)')
METRICS = ['mse', 'mae', 'rmse', 'mape', 'mspe', 'smape']


def parse_args():
    parser = argparse.ArgumentParser(description='Paired significance test between two benchmark result sets.')
    parser.add_argument('--a', required=True, nargs='+', help='metrics.csv, metrics.txt, directory, or glob for method A')
    parser.add_argument('--b', required=True, nargs='+', help='metrics.csv, metrics.txt, directory, or glob for method B')
    parser.add_argument('--a-name', default='method_a')
    parser.add_argument('--b-name', default='method_b')
    parser.add_argument('--out-dir', default=os.path.join('analysis', 'significance'))
    parser.add_argument('--out-name', default='paired_significance')
    return parser.parse_args()


def expand_inputs(patterns):
    paths = []
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            paths.extend(matches)
        elif os.path.exists(pattern):
            paths.append(pattern)
    files = []
    for path in paths:
        if os.path.isdir(path):
            files.extend(glob.glob(os.path.join(path, '**', 'metrics.csv'), recursive=True))
            files.extend(glob.glob(os.path.join(path, '**', 'metrics.txt'), recursive=True))
        elif os.path.basename(path) in {'metrics.csv', 'metrics.txt'}:
            files.append(path)
    return sorted(dict.fromkeys(files))


def seed_from_text(text):
    match = SEED_RE.search(text)
    return match.group(1) if match else ''


def parse_metrics_txt(path):
    metrics = {}
    with open(path, 'r', encoding='utf-8', errors='ignore') as file_obj:
        for line in file_obj:
            for key, value in METRIC_RE.findall(line):
                metrics[key.lower()] = float(value)
    return metrics


def parse_metrics_csv(path):
    rows = {}
    with open(path, 'r', encoding='utf-8', errors='ignore') as file_obj:
        reader = csv.DictReader(file_obj)
        for idx, raw_row in enumerate(reader):
            seed = raw_row.get('seed') or seed_from_text(raw_row.get('source', '')) or str(idx)
            row = {}
            for metric in METRICS:
                value = raw_row.get(metric)
                if value not in {None, ''}:
                    row[metric] = float(value)
            if row:
                rows[str(seed)] = row
    return rows


def read_result_set(paths):
    result = {}
    for path in paths:
        if path.endswith('.csv'):
            result.update(parse_metrics_csv(path))
        else:
            seed = seed_from_text(path)
            metrics = parse_metrics_txt(path)
            if metrics:
                result[str(seed or len(result))] = metrics
    return result


def mean_std(values):
    if not values:
        return '', ''
    mean = statistics.mean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return mean, std


def paired_tests(values_a, values_b):
    if len(values_a) != len(values_b) or len(values_a) < 2:
        return '', '', '', '', '', ''
    diffs = [a_value - b_value for a_value, b_value in zip(values_a, values_b)]
    diff_mean, diff_std = mean_std(diffs)
    cohen_dz = diff_mean / diff_std if diff_std not in {'', 0.0} else ''
    t_stat = ''
    t_p = ''
    w_p = ''
    try:
        from scipy import stats

        t_result = stats.ttest_rel(values_a, values_b)
        t_stat = float(t_result.statistic)
        t_p = float(t_result.pvalue)
        if all(abs(value) <= 1e-15 for value in diffs):
            w_p = 1.0
        else:
            w_p = float(stats.wilcoxon(values_a, values_b, zero_method='wilcox').pvalue)
    except Exception:
        pass
    return diff_mean, diff_std, cohen_dz, t_stat, t_p, w_p


def holm_adjust(rows, p_key, out_key):
    valid = []
    for idx, row in enumerate(rows):
        value = row.get(p_key, '')
        if value == '' or not math.isfinite(float(value)):
            row[out_key] = ''
            continue
        valid.append((idx, float(value)))
    ordered = sorted(valid, key=lambda item: item[1])
    running_max = 0.0
    total = len(ordered)
    for rank, (idx, value) in enumerate(ordered, start=1):
        adjusted = min((total - rank + 1) * value, 1.0)
        running_max = min(max(running_max, adjusted), 1.0)
        rows[idx][out_key] = running_max


def fmt(value):
    if value == '' or value is None:
        return ''
    value = float(value)
    if value != 0.0 and abs(value) < 1e-4:
        return f'{value:.3e}'
    return f'{value:.6f}'


def write_outputs(rows, args):
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, f'{args.out_name}.csv')
    md_path = os.path.join(args.out_dir, f'{args.out_name}.md')
    fieldnames = [
        'a_name', 'b_name', 'metric', 'n', 'seeds',
        'a_mean', 'b_mean', 'diff_a_minus_b_mean', 'diff_a_minus_b_std',
        'a_improvement_mean', 'a_improvement_pct', 'a_better_seeds', 'b_better_seeds',
        'tie_seeds', 'cohen_dz', 'paired_t_stat', 'paired_t_p', 'paired_t_p_holm',
        'wilcoxon_p', 'wilcoxon_p_holm',
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in fieldnames})
    with open(md_path, 'w', encoding='utf-8') as file_obj:
        file_obj.write('| Comparison | Metric | n | A mean | B mean | A improvement | Wilcoxon p (Holm) | t-test p (Holm) | A better seeds |\n')
        file_obj.write('| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n')
        for row in rows:
            if row['metric'] not in {'mse', 'mae', 'rmse'}:
                continue
            file_obj.write(
                '| {} vs {} | {} | {} | {} | {} | {} | {} | {} | {}/{} |\n'.format(
                    row['a_name'],
                    row['b_name'],
                    row['metric'],
                    row['n'],
                    fmt(row['a_mean']),
                    fmt(row['b_mean']),
                    fmt(row['a_improvement_mean']),
                    fmt(row['wilcoxon_p_holm']),
                    fmt(row['paired_t_p_holm']),
                    row['a_better_seeds'],
                    row['n'],
                )
            )
    return csv_path, md_path


def main():
    args = parse_args()
    paths_a = expand_inputs(args.a)
    paths_b = expand_inputs(args.b)
    if not paths_a:
        raise SystemExit('No metrics files found for --a.')
    if not paths_b:
        raise SystemExit('No metrics files found for --b.')

    results_a = read_result_set(paths_a)
    results_b = read_result_set(paths_b)
    rows = []
    for metric in METRICS:
        common_seeds = [
            seed
            for seed in sorted(set(results_a) & set(results_b))
            if metric in results_a[seed] and metric in results_b[seed]
        ]
        values_a = [results_a[seed][metric] for seed in common_seeds]
        values_b = [results_b[seed][metric] for seed in common_seeds]
        if not values_a:
            continue
        a_mean, _ = mean_std(values_a)
        b_mean, _ = mean_std(values_b)
        diff_mean, diff_std, cohen_dz, t_stat, t_p, w_p = paired_tests(values_a, values_b)
        improvement = -diff_mean if diff_mean != '' else ''
        improvement_pct = improvement / b_mean * 100.0 if improvement != '' and b_mean != 0.0 else ''
        a_better = sum(1 for a_value, b_value in zip(values_a, values_b) if a_value < b_value)
        b_better = sum(1 for a_value, b_value in zip(values_a, values_b) if b_value < a_value)
        rows.append({
            'a_name': args.a_name,
            'b_name': args.b_name,
            'metric': metric,
            'n': len(common_seeds),
            'seeds': ' '.join(common_seeds),
            'a_mean': a_mean,
            'b_mean': b_mean,
            'diff_a_minus_b_mean': diff_mean,
            'diff_a_minus_b_std': diff_std,
            'a_improvement_mean': improvement,
            'a_improvement_pct': improvement_pct,
            'a_better_seeds': a_better,
            'b_better_seeds': b_better,
            'tie_seeds': len(values_a) - a_better - b_better,
            'cohen_dz': cohen_dz,
            'paired_t_stat': t_stat,
            'paired_t_p': t_p,
            'wilcoxon_p': w_p,
        })

    holm_adjust(rows, 'paired_t_p', 'paired_t_p_holm')
    holm_adjust(rows, 'wilcoxon_p', 'wilcoxon_p_holm')
    csv_path, md_path = write_outputs(rows, args)
    print(f'a_files: {len(paths_a)}')
    print(f'b_files: {len(paths_b)}')
    print(f'csv: {csv_path}')
    print(f'md: {md_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
