#!/usr/bin/env python
import argparse
import csv
import math
import os
import subprocess
import sys

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import benchmark


def parse_args():
    parser = argparse.ArgumentParser(description='Run fixed/adaptive query fusion ablations.')
    parser.add_argument('model', nargs='?', default=os.environ.get('MODEL', 'AdaLG'))
    parser.add_argument('dataset', nargs='?', default=os.environ.get('DATASET', 'human_activity'))
    parser.add_argument(
        '--strategies',
        nargs='*',
        default=None,
        help='strategy keys to run; default is global_only, fixed gates, adaptive_unbounded, feature_level, adaptive_bounded',
    )
    parser.add_argument('--fixed-gates', default=os.environ.get('FUSION_FIXED_GATES', '0.1,0.3,0.5'))
    parser.add_argument('--bounded-gate-max', default=os.environ.get('BOUNDED_GATE_MAX', '0.3'))
    parser.add_argument('--des-prefix', default=os.environ.get('FUSION_DES_PREFIX', ''))
    parser.add_argument('--seed', default=os.environ.get('SEED', ''))
    parser.add_argument('--seed-list', default=os.environ.get('SEED_LIST', ''))
    parser.add_argument(
        '--query-evidence',
        choices=['none', 'adaptive_bounded', 'all'],
        default=os.environ.get('FUSION_QUERY_EVIDENCE', 'adaptive_bounded'),
        help='which strategies should emit query gate evidence files',
    )
    parser.add_argument(
        '--local-drop-robustness',
        action='store_true',
        help='evaluate local observation degradation for every selected strategy after training',
    )
    parser.add_argument(
        '--local-drop-rates',
        default=os.environ.get('LOCAL_OBSERVATION_DROP_RATES', '0,0.25,0.5,0.75'),
        help='comma-separated local observation deletion rates used by --local-drop-robustness',
    )
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true', help='reuse completed per-seed training checkpoints and regenerate test analyses')
    parser.add_argument('--no-analyze', action='store_true')
    parser.add_argument(
        '--log-dir',
        default=os.environ.get('FUSION_LOG_DIR', os.path.join('logs', 'fusion_ablation')),
    )
    parser.add_argument(
        '--analysis-dir',
        default=os.environ.get('FUSION_ANALYSIS_DIR', os.path.join('analysis', 'fusion_ablation')),
    )
    parser.add_argument(
        '--out-dir',
        default=os.environ.get('FUSION_OUT_DIR', os.path.join('analysis', 'fusion_ablation')),
    )
    return parser.parse_args()


def split_csv_values(text):
    return [item.strip() for item in str(text).replace(';', ',').split(',') if item.strip()]


def safe_component(value):
    return ''.join(ch if ch.isalnum() else '_' for ch in str(value)).strip('_')


def build_strategy_catalog(fixed_gates, bounded_gate_max):
    catalog = {
        'global_only': {
            'label': 'Global only',
            'env': {
                'POINT_FORCE_FUSION': 'global',
                'POINT_FIXED_GATE': None,
                'POINT_GATE_MAX': '1.0',
            },
        },
        'local_only': {
            'label': 'Local only',
            'env': {
                'POINT_FORCE_FUSION': 'local',
                'POINT_FIXED_GATE': None,
                'POINT_GATE_MAX': '1.0',
            },
        },
        'adaptive_unbounded': {
            'label': 'Adaptive gate without bound',
            'env': {
                'POINT_FORCE_FUSION': 'auto',
                'POINT_FIXED_GATE': None,
                'POINT_GATE_MAX': '1.0',
            },
        },
        'feature_level': {
            'label': 'Feature-level gated fusion',
            'env': {
                'POINT_FORCE_FUSION': 'feature',
                'POINT_FIXED_GATE': None,
                'POINT_GATE_MAX': str(bounded_gate_max),
            },
        },
        'adaptive_bounded': {
            'label': 'AdaLG bounded adaptive gate',
            'env': {
                'POINT_FORCE_FUSION': 'auto',
                'POINT_FIXED_GATE': None,
                'POINT_GATE_MAX': str(bounded_gate_max),
            },
        },
    }
    # Primary pair: same backbone, gate capacity, auxiliary losses, optimizer,
    # splits, and seeds. Only the second prediction's information source differs.
    ordinary = {
        'POINT_FORCE_FUSION': 'auto', 'POINT_FIXED_GATE': None, 'POINT_GATE_MAX': '1.0',
        'POINT_GATE_MODE': 'ordinary', 'POINT_DISABLE_LOCAL_BRANCH': '0',
        'POINT_DISABLE_CAUSAL_GATE_CONTEXT': '1', 'POINT_GATE_PREDICTION_FEATURES': '1',
        'POINT_DISABLE_GATE_AVAILABILITY_FEATURES': '1',
        'POINT_SEPARATE_UTILITY_HEAD': '0', 'POINT_LOCAL_FUSION_MODE': 'dual_gate',
        'CAUSAL_LOCAL_EXPAND_POOL': '0', 'CAUSAL_LOCAL_ALPHA': '0.0',
        'UTILITY_LOSS_WEIGHT': '0.0', 'ORACLE_GATE_LOSS_WEIGHT': '0.0',
        'POINT_TRAIN_STAGE': 'full', 'POINT_GATE_INIT_CHECKPOINT': None,
    }
    for key, label, source, encoder in (
        ('shared_ordinary', 'Shared encoder + two heads + ordinary gate', 'shared_encoder', 'summary'),
        ('raw_ordinary', 'Independent raw local + ordinary gate', 'raw', 'summary'),
        ('raw_attention_ordinary', 'Independent raw attention + ordinary gate', 'raw', 'attention'),
    ):
        catalog[key] = {'label': label, 'env': dict(ordinary, POINT_LOCAL_BRANCH_SOURCE=source, POINT_LOCAL_ENCODER=encoder)}
    catalog['raw_attention_utility'] = {
        'label': 'Independent raw attention + bounded utility gate',
        'env': dict(ordinary, POINT_LOCAL_BRANCH_SOURCE='raw', POINT_LOCAL_ENCODER='attention',
                    POINT_GATE_MODE='utility', POINT_GATE_MAX=str(bounded_gate_max),
                    POINT_DISABLE_GATE_AVAILABILITY_FEATURES='0',
                    UTILITY_LOSS_WEIGHT='0.1', ORACLE_GATE_LOSS_WEIGHT='0.1'),
    }
    for gate in fixed_gates:
        key = f'fixed_{gate}'
        catalog[key] = {
            'label': f'Fixed g={gate}',
            'env': {
                'POINT_FORCE_FUSION': 'auto',
                'POINT_FIXED_GATE': str(gate),
                'POINT_GATE_MAX': '1.0',
            },
        }
    return catalog


def resolve_strategy_keys(args, fixed_gates):
    if args.strategies:
        keys = []
        for item in args.strategies:
            if item == 'fixed':
                keys.extend(f'fixed_{gate}' for gate in fixed_gates)
            else:
                keys.append(item)
        return keys
    return [
        'global_only',
        *[f'fixed_{gate}' for gate in fixed_gates],
        'adaptive_unbounded',
        'feature_level',
        'adaptive_bounded',
    ]


def apply_env(base_env, updates):
    child_env = dict(base_env)
    for key, value in updates.items():
        if value is None or value == '':
            child_env.pop(key, None)
        else:
            child_env[key] = str(value)
    return child_env


def resolve_seed_list(env):
    if env.get('SEED'):
        return [env['SEED']]
    return env.get('SEED_LIST', '2021 2022 2023 2024 2025').split()


def build_protocol_suffix(dataset, env):
    point_train_size = env.get('POINT_TRAIN_SIZE', '0.8')
    point_val_size = env.get('POINT_VAL_SIZE', '0.1')
    point_test_size = env.get('POINT_TEST_SIZE', '0.1')
    point_split_mode = dataset.get('point_split_mode', 'random')
    point_stratified = dataset.get('point_stratified_random_split', '1')
    protocol_suffix = f'_split_{point_split_mode}_{point_train_size}_{point_val_size}_{point_test_size}'
    if dataset.get('p19_split'):
        protocol_suffix = f"{protocol_suffix}_fold{dataset['p19_split']}"
    if point_split_mode in {'random', 'record'} and str(point_stratified).strip().lower() not in {'0', 'false'}:
        protocol_suffix = f'{protocol_suffix}_stratified'
    return protocol_suffix


def compute_run_des_prefix(model_key, dataset_key, env):
    dataset = benchmark.load_dataset_config(dataset_key)
    model = benchmark.load_model_config(model_key, dataset, env=env)
    protocol_suffix = build_protocol_suffix(dataset, env)
    des_prefix = f"{model['des']}{protocol_suffix}{model.get('model_des_suffix', '')}"
    return benchmark.compact_des_prefix(des_prefix)


def run_benchmark(args, env):
    command = [
        sys.executable,
        'scripts/benchmark.py',
        args.model,
        args.dataset,
        '--log-dir',
        args.log_dir,
        '--analysis-dir',
        args.analysis_dir,
    ]
    if args.dry_run:
        command.append('--dry-run')
    if args.resume:
        command.append('--resume')
    if args.no_analyze:
        command.append('--no-analyze')
    return subprocess.call(command, cwd=REPO_ROOT, env=env)


def metric_summary(collected):
    row = {'n': len(collected)}
    for metric_name in benchmark.METRIC_NAMES:
        values = [metrics[metric_name] for _, _, metrics in collected if metric_name in metrics]
        if not values:
            row[f'{metric_name}_mean'] = ''
            row[f'{metric_name}_std'] = ''
            continue
        avg, std = benchmark.mean_std(values)
        row[f'{metric_name}_mean'] = avg
        row[f'{metric_name}_std'] = std
    return row


def format_metric(row, metric_name):
    mean_value = row.get(f'{metric_name}_mean', '')
    std_value = row.get(f'{metric_name}_std', '')
    if mean_value == '':
        return ''
    if row.get('n', 0) > 1:
        return f'{mean_value:.6f} +/- {std_value:.6f}'
    return f'{mean_value:.6f}'


def write_fusion_table(rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'fusion_strategy_results.csv')
    fieldnames = [
        'fusion_strategy',
        'strategy_key',
        'run_des_prefix',
        'n',
        *[f'{metric}_mean' for metric in benchmark.METRIC_NAMES],
        *[f'{metric}_std' for metric in benchmark.METRIC_NAMES],
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in fieldnames})

    md_path = os.path.join(out_dir, 'fusion_strategy_results.md')
    with open(md_path, 'w', encoding='utf-8') as file_obj:
        file_obj.write('| Fusion Strategy | MSE | MAE | RMSE | MAPE | n |\n')
        file_obj.write('| --- | ---: | ---: | ---: | ---: | ---: |\n')
        for row in rows:
            file_obj.write(
                '| {} | {} | {} | {} | {} | {} |\n'.format(
                    row['fusion_strategy'],
                    format_metric(row, 'mse'),
                    format_metric(row, 'mae'),
                    format_metric(row, 'rmse'),
                    format_metric(row, 'mape'),
                    row.get('n', 0),
                )
            )
    return csv_path, md_path


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def holm_adjust_pvalues(rows, p_key, out_key, group_key='metric'):
    groups = {}
    for idx, row in enumerate(rows):
        p_value = to_float(row.get(p_key))
        if p_value is None or not math.isfinite(p_value):
            row[out_key] = ''
            continue
        groups.setdefault(row.get(group_key, ''), []).append((idx, p_value))

    for group_rows in groups.values():
        ordered = sorted(group_rows, key=lambda item: item[1])
        total = len(ordered)
        running_max = 0.0
        for rank, (idx, p_value) in enumerate(ordered, start=1):
            adjusted = min((total - rank + 1) * p_value, 1.0)
            running_max = min(max(running_max, adjusted), 1.0)
            rows[idx][out_key] = running_max


def paired_significance(values_a, values_b):
    if len(values_a) != len(values_b) or len(values_a) < 2:
        return '', '', '', '', '', ''
    differences = [a_value - b_value for a_value, b_value in zip(values_a, values_b)]
    diff_mean, diff_std = mean_std_values(differences)
    cohen_dz = diff_mean / diff_std if diff_std not in {'', 0.0} else ''
    t_stat = ''
    p_ttest = ''
    p_wilcoxon = ''
    try:
        from scipy import stats

        t_result = stats.ttest_rel(values_a, values_b)
        t_stat = float(t_result.statistic)
        p_ttest = float(t_result.pvalue)
        if all(abs(value) <= 1e-15 for value in differences):
            p_wilcoxon = 1.0
        else:
            p_wilcoxon = float(stats.wilcoxon(values_a, values_b, zero_method='wilcox').pvalue)
    except Exception:
        pass
    return diff_mean, diff_std, cohen_dz, t_stat, p_ttest, p_wilcoxon


def select_fusion_significance_pairs(rows):
    keys = [row['strategy_key'] for row in rows]
    key_set = set(keys)
    fixed_keys = sorted(key for key in keys if key.startswith('fixed_'))
    pairs = []
    for a, b in [('raw_ordinary', 'shared_ordinary'), ('raw_attention_ordinary', 'shared_ordinary'),
                 ('raw_attention_ordinary', 'raw_ordinary'), ('raw_attention_utility', 'raw_attention_ordinary')]:
        if a in key_set and b in key_set:
            pairs.append((a, b))
    if 'adaptive_bounded' in key_set and 'adaptive_unbounded' in key_set:
        pairs.append(('adaptive_bounded', 'adaptive_unbounded'))
    if 'adaptive_bounded' in key_set and 'feature_level' in key_set:
        pairs.append(('adaptive_bounded', 'feature_level'))
    for fixed_key in fixed_keys:
        if 'adaptive_bounded' in key_set:
            pairs.append(('adaptive_bounded', fixed_key))
        if 'adaptive_unbounded' in key_set:
            pairs.append(('adaptive_unbounded', fixed_key))
    if pairs:
        return pairs
    return [
        (keys[left], keys[right])
        for left in range(len(keys))
        for right in range(left + 1, len(keys))
    ]


def write_fusion_significance(rows, out_dir):
    if not rows:
        return None, None

    metrics_by_strategy = {}
    labels = {}
    for strategy_row in rows:
        strategy_key = strategy_row['strategy_key']
        labels[strategy_key] = strategy_row['fusion_strategy']
        seed_metrics = {}
        for seed, _, metrics in strategy_row.get('collected', []):
            seed_metrics[str(seed)] = metrics
        metrics_by_strategy[strategy_key] = seed_metrics

    output_rows = []
    for strategy_a, strategy_b in select_fusion_significance_pairs(rows):
        seeds_a = metrics_by_strategy.get(strategy_a, {})
        seeds_b = metrics_by_strategy.get(strategy_b, {})
        for metric_name in benchmark.METRIC_NAMES:
            common_seeds = [
                seed
                for seed in sorted(set(seeds_a) & set(seeds_b))
                if metric_name in seeds_a[seed] and metric_name in seeds_b[seed]
            ]
            values_a = [float(seeds_a[seed][metric_name]) for seed in common_seeds]
            values_b = [float(seeds_b[seed][metric_name]) for seed in common_seeds]
            if not common_seeds:
                continue
            diff_mean, diff_std, cohen_dz, t_stat, p_ttest, p_wilcoxon = paired_significance(values_a, values_b)
            a_mean, _ = mean_std_values(values_a)
            b_mean, _ = mean_std_values(values_b)
            improvement_mean = -diff_mean if diff_mean != '' else ''
            improvement_pct = (
                (improvement_mean / b_mean) * 100.0
                if improvement_mean != '' and b_mean not in {'', 0.0}
                else ''
            )
            a_better = sum(1 for a_value, b_value in zip(values_a, values_b) if a_value < b_value)
            b_better = sum(1 for a_value, b_value in zip(values_a, values_b) if b_value < a_value)
            ties = len(values_a) - a_better - b_better
            output_rows.append({
                'strategy_a': strategy_a,
                'strategy_b': strategy_b,
                'strategy_a_label': labels.get(strategy_a, strategy_a),
                'strategy_b_label': labels.get(strategy_b, strategy_b),
                'metric': metric_name,
                'n': len(common_seeds),
                'seeds': ' '.join(common_seeds),
                'strategy_a_mean': a_mean,
                'strategy_b_mean': b_mean,
                'diff_a_minus_b_mean': diff_mean,
                'diff_a_minus_b_std': diff_std,
                'strategy_a_improvement_mean': improvement_mean,
                'strategy_a_improvement_pct': improvement_pct,
                'strategy_a_better_seeds': a_better,
                'strategy_b_better_seeds': b_better,
                'tie_seeds': ties,
                'cohen_dz': cohen_dz,
                'paired_t_stat': t_stat,
                'paired_t_p': p_ttest,
                'wilcoxon_p': p_wilcoxon,
            })

    if not output_rows:
        return None, None

    holm_adjust_pvalues(output_rows, 'paired_t_p', 'paired_t_p_holm')
    holm_adjust_pvalues(output_rows, 'wilcoxon_p', 'wilcoxon_p_holm')
    for row in output_rows:
        p_holm = to_float(row.get('wilcoxon_p_holm'))
        row['wilcoxon_significant_0_05'] = int(p_holm is not None and p_holm < 0.05)

    csv_path = os.path.join(out_dir, 'fusion_strategy_significance.csv')
    fieldnames = [
        'strategy_a',
        'strategy_b',
        'strategy_a_label',
        'strategy_b_label',
        'metric',
        'n',
        'seeds',
        'strategy_a_mean',
        'strategy_b_mean',
        'diff_a_minus_b_mean',
        'diff_a_minus_b_std',
        'strategy_a_improvement_mean',
        'strategy_a_improvement_pct',
        'strategy_a_better_seeds',
        'strategy_b_better_seeds',
        'tie_seeds',
        'cohen_dz',
        'paired_t_stat',
        'paired_t_p',
        'paired_t_p_holm',
        'wilcoxon_p',
        'wilcoxon_p_holm',
        'wilcoxon_significant_0_05',
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in output_rows:
            writer.writerow({key: row.get(key, '') for key in fieldnames})

    md_path = os.path.join(out_dir, 'fusion_strategy_significance.md')
    with open(md_path, 'w', encoding='utf-8') as file_obj:
        file_obj.write('| Comparison | Metric | n | A mean | B mean | A improvement | Wilcoxon p (Holm) | t-test p (Holm) | A better seeds |\n')
        file_obj.write('| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n')
        for row in output_rows:
            if row['metric'] not in {'mse', 'mae', 'rmse'}:
                continue
            comparison = f"{row['strategy_a']} vs {row['strategy_b']}"
            file_obj.write(
                '| {} | {} | {} | {} | {} | {} | {} | {} | {}/{} |\n'.format(
                    comparison,
                    row['metric'],
                    row['n'],
                    _format_float(row['strategy_a_mean']),
                    _format_float(row['strategy_b_mean']),
                    _format_float(row['strategy_a_improvement_mean']),
                    _format_float(row.get('wilcoxon_p_holm')),
                    _format_float(row.get('paired_t_p_holm')),
                    row['strategy_a_better_seeds'],
                    row['n'],
                )
            )
    return csv_path, md_path


def _format_float(value):
    value = to_float(value)
    if value is None:
        return ''
    if abs(value) < 1e-4 and value != 0.0:
        return f'{value:.3e}'
    return f'{value:.6f}'


def aggregate_auxiliary_rows(strategy_rows, file_name, group_keys, value_keys, out_path):
    grouped = {}
    for strategy_row in strategy_rows:
        for _, metrics_path, _ in strategy_row.get('collected', []):
            source_path = os.path.join(os.path.dirname(metrics_path), file_name)
            if not os.path.exists(source_path):
                continue
            with open(source_path, 'r', encoding='utf-8') as file_obj:
                reader = csv.DictReader(file_obj)
                for raw_row in reader:
                    group = tuple([strategy_row['strategy_key']] + [raw_row.get(key, '') for key in group_keys])
                    grouped.setdefault(group, []).append(raw_row)

    if not grouped:
        return None

    fieldnames = ['strategy_key', *group_keys, 'n_files', 'n_queries_sum']
    for value_key in value_keys:
        fieldnames.extend([f'{value_key}_mean', f'{value_key}_std'])

    with open(out_path, 'w', newline='', encoding='utf-8') as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for group, raw_rows in sorted(grouped.items()):
            output = {'strategy_key': group[0], 'n_files': len(raw_rows), 'n_queries_sum': 0}
            for idx, group_key in enumerate(group_keys, start=1):
                output[group_key] = group[idx]
            n_queries_values = [to_float(row.get('n_queries')) for row in raw_rows]
            output['n_queries_sum'] = int(sum(value for value in n_queries_values if value is not None))
            for value_key in value_keys:
                values = [to_float(row.get(value_key)) for row in raw_rows]
                values = [value for value in values if value is not None]
                if values:
                    avg, std = benchmark.mean_std(values)
                    output[f'{value_key}_mean'] = avg
                    output[f'{value_key}_std'] = std
                else:
                    output[f'{value_key}_mean'] = ''
                    output[f'{value_key}_std'] = ''
            writer.writerow(output)
    return out_path


def write_auxiliary_tables(rows, out_dir):
    aggregate_auxiliary_rows(
        rows, 'branch_complementarity.csv', ['group'],
        ['global_mae', 'global_mse', 'local_mae_has_local', 'local_mse_has_local', 'fused_mae', 'fused_mse',
         'oracle_selection_mae', 'oracle_selection_mse', 'fusion_minus_selection_oracle_mae',
         'local_win_rate', 'fusion_rescue_rate', 'fusion_harm_rate', 'mean_local_improvement',
         'mean_fusion_improvement', 'signed_error_correlation', 'mean_gate'],
        os.path.join(out_dir, 'branch_complementarity_aggregate.csv'),
    )
    relationship_path = aggregate_auxiliary_rows(
        rows,
        'query_gate_relationships.csv',
        ['relationship', 'evidence_key', 'expected_sign'],
        ['pearson', 'spearman', 'avg_evidence', 'avg_gate'],
        os.path.join(out_dir, 'query_gate_relationships_aggregate.csv'),
    )
    recent_gap_path = aggregate_auxiliary_rows(
        rows,
        'recent_gap_gate_summary.csv',
        ['recent_gap_group', 'recent_gap_key'],
        ['avg_recent_gap', 'avg_gate', 'std_gate', 'avg_candidate_count', 'mae', 'mse'],
        os.path.join(out_dir, 'recent_gap_gate_aggregate.csv'),
    )
    local_advantage_path = aggregate_auxiliary_rows(
        rows,
        'local_advantage_gate_summary.csv',
        ['query_group'],
        [
            'query_ratio',
            'mean_advantage',
            'global_mae',
            'local_mae',
            'adalg_mae',
            'global_mse',
            'local_mse',
            'adalg_mse',
            'mean_gate',
            'std_gate',
        ],
        os.path.join(out_dir, 'local_advantage_gate_aggregate.csv'),
    )
    local_drop_path = aggregate_auxiliary_rows(
        rows,
        'local_observation_drop_robustness.csv',
        ['degradation_mode', 'drop_rate'],
        [
            'mse',
            'mae',
            'rmse',
            'mape',
            'mspe',
            'smape',
            'mean_gate',
            'std_gate',
            'mean_gate_has_local',
            'has_local_ratio',
            'mean_local_advantage',
            'local_better_ratio',
            'global_mse', 'global_mae', 'local_mae_has_local', 'oracle_regret',
            'max_global_prediction_change', 'mse_change_vs_no_drop', 'mae_change_vs_no_drop',
            'mean_candidate_count', 'mean_recent_gap_has_local',
            'mean_gate_change_vs_no_drop', 'mean_abs_gate_change_vs_no_drop',
        ],
        os.path.join(out_dir, 'local_observation_drop_robustness_aggregate.csv'),
    )
    return relationship_path, recent_gap_path, local_advantage_path, local_drop_path


def write_local_advantage_markdown(csv_path):
    if not csv_path or not os.path.exists(csv_path):
        return None
    md_path = os.path.splitext(csv_path)[0] + '.md'
    with open(csv_path, 'r', encoding='utf-8') as file_obj:
        reader = csv.DictReader(file_obj)
        rows = list(reader)
    with open(md_path, 'w', encoding='utf-8') as file_obj:
        file_obj.write('| Strategy | Query Group | Query Ratio | Global MAE | Local MAE | AdaLG MAE | Mean g | Mean A | n |\n')
        file_obj.write('| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n')
        for row in rows:
            file_obj.write(
                '| {strategy_key} | {query_group} | {query_ratio_mean} | {global_mae_mean} | '
                '{local_mae_mean} | {adalg_mae_mean} | {mean_gate_mean} | {mean_advantage_mean} | '
                '{n_queries_sum} |\n'.format(**row)
            )
    return md_path


def write_local_drop_markdown(csv_path):
    if not csv_path or not os.path.exists(csv_path):
        return None
    md_path = os.path.splitext(csv_path)[0] + '.md'
    with open(csv_path, 'r', encoding='utf-8') as file_obj:
        reader = csv.DictReader(file_obj)
        rows = list(reader)
    with open(md_path, 'w', encoding='utf-8') as file_obj:
        file_obj.write('| Strategy | Mode | Severity | MSE | MAE | RMSE | MAPE | Mean g | Has-local ratio | Local-better ratio | n |\n')
        file_obj.write('| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n')
        for row in rows:
            file_obj.write(
                '| {strategy_key} | {degradation_mode} | {drop_rate} | {mse_mean} | {mae_mean} | {rmse_mean} | {mape_mean} | '
                '{mean_gate_mean} | {has_local_ratio_mean} | {local_better_ratio_mean} | '
                '{n_queries_sum} |\n'.format(**row)
            )
    return md_path


def mean_std_values(values):
    if not values:
        return '', ''
    avg = sum(values) / len(values)
    std = math.sqrt(sum((value - avg) ** 2 for value in values) / len(values)) if len(values) > 1 else 0.0
    return avg, std


def paired_t_test(values):
    values = [value for value in values if value is not None]
    if len(values) < 2:
        return '', ''
    avg, std = mean_std_values(values)
    if std == 0:
        return '', ''
    t_stat = avg / (std / math.sqrt(len(values)))
    p_value = ''
    try:
        from scipy import stats
        p_value = float(stats.ttest_1samp(values, 0.0).pvalue)
    except Exception:
        p_value = ''
    return t_stat, p_value


def write_local_drop_significance(rows, out_dir):
    paired = {}
    metrics = ['mse', 'mae', 'rmse', 'mape', 'mean_gate', 'has_local_ratio']
    for strategy_row in rows:
        for seed, metrics_path, _ in strategy_row.get('collected', []):
            source_path = os.path.join(os.path.dirname(metrics_path), 'local_observation_drop_robustness.csv')
            if not os.path.exists(source_path):
                continue
            with open(source_path, 'r', encoding='utf-8') as file_obj:
                reader = csv.DictReader(file_obj)
                drop_rows = list(reader)
            baselines = {row.get('degradation_mode', 'drop'): row for row in drop_rows if to_float(row.get('drop_rate')) == 0.0}
            for row in drop_rows:
                mode = row.get('degradation_mode', 'drop')
                baseline = baselines.get(mode)
                if baseline is None:
                    continue
                drop_rate = to_float(row.get('drop_rate'))
                if drop_rate is None or drop_rate == 0.0:
                    continue
                key = (strategy_row['strategy_key'], mode, str(drop_rate))
                target = paired.setdefault(key, {metric: [] for metric in metrics})
                for metric in metrics:
                    value = to_float(row.get(metric))
                    base_value = to_float(baseline.get(metric))
                    if value is not None and base_value is not None:
                        target[metric].append(value - base_value)

    if not paired:
        return None

    out_path = os.path.join(out_dir, 'local_observation_drop_significance.csv')
    fieldnames = ['strategy_key', 'degradation_mode', 'drop_rate', 'n']
    for metric in metrics:
        fieldnames.extend([f'delta_{metric}_mean', f'delta_{metric}_std', f'delta_{metric}_t', f'delta_{metric}_p'])
    with open(out_path, 'w', newline='', encoding='utf-8') as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for (strategy_key, mode, drop_rate), values_by_metric in sorted(paired.items()):
            n = max(len(values) for values in values_by_metric.values())
            output = {'strategy_key': strategy_key, 'degradation_mode': mode, 'drop_rate': drop_rate, 'n': n}
            for metric, values in values_by_metric.items():
                avg, std = mean_std_values(values)
                t_stat, p_value = paired_t_test(values)
                output[f'delta_{metric}_mean'] = avg
                output[f'delta_{metric}_std'] = std
                output[f'delta_{metric}_t'] = t_stat
                output[f'delta_{metric}_p'] = p_value
            writer.writerow(output)
    return out_path


def main():
    args = parse_args()
    fixed_gates = split_csv_values(args.fixed_gates)
    catalog = build_strategy_catalog(fixed_gates, args.bounded_gate_max)
    strategy_keys = resolve_strategy_keys(args, fixed_gates)
    unknown = [key for key in strategy_keys if key not in catalog]
    if unknown:
        raise SystemExit(f'Unknown strategies: {", ".join(unknown)}')

    dataset = benchmark.load_dataset_config(args.dataset)
    base_des = args.des_prefix.strip() or os.environ.get('DES', '') or f"{dataset['slug']}_{args.model}_fusion"
    report_component = safe_component(f'{dataset["slug"]}_{args.model}_{base_des}')
    out_dir = os.path.join(args.out_dir, report_component)

    rows = []
    base_env = os.environ.copy()
    base_env.setdefault('CUDA_VISIBLE_DEVICES', os.environ.get('GPU', '0'))
    if args.seed:
        base_env['SEED'] = args.seed
        base_env.pop('SEED_LIST', None)
    elif args.seed_list:
        base_env.pop('SEED', None)
        base_env['SEED_LIST'] = args.seed_list

    for strategy_key in strategy_keys:
        strategy = catalog[strategy_key]
        env_updates = dict(strategy['env'])
        env_updates['DES'] = f'{base_des}_{safe_component(strategy_key)}'
        should_collect_evidence = (
            args.query_evidence == 'all'
            or (args.query_evidence == 'adaptive_bounded' and strategy_key == 'adaptive_bounded')
        )
        if should_collect_evidence:
            env_updates['QUERY_EVIDENCE_ANALYSIS'] = '1'
        else:
            env_updates['QUERY_EVIDENCE_ANALYSIS'] = None
        if args.local_drop_robustness:
            env_updates['LOCAL_OBSERVATION_DROP_ROBUSTNESS'] = '1'
            env_updates['LOCAL_OBSERVATION_DROP_RATES'] = args.local_drop_rates
        else:
            env_updates['LOCAL_OBSERVATION_DROP_ROBUSTNESS'] = None

        child_env = apply_env(base_env, env_updates)
        seed_list = resolve_seed_list(child_env)
        run_des_prefix = compute_run_des_prefix(args.model, args.dataset, child_env)
        print('')
        print(f'Fusion strategy: {strategy["label"]}')
        print(f'run_des_prefix: {run_des_prefix}')
        status = run_benchmark(args, child_env)
        if status != 0:
            return status
        if args.dry_run:
            continue

        collected, missing = benchmark.find_seed_metrics(run_des_prefix, seed_list)
        row = {
            'fusion_strategy': strategy['label'],
            'strategy_key': strategy_key,
            'run_des_prefix': run_des_prefix,
            'collected': collected,
        }
        row.update(metric_summary(collected))
        rows.append(row)
        if missing:
            print(f'missing metrics for {strategy_key}: {" ".join(missing)}')

    if args.dry_run:
        return 0

    csv_path, md_path = write_fusion_table(rows, out_dir)
    fusion_significance_path, fusion_significance_md_path = write_fusion_significance(rows, out_dir)
    relationship_path, recent_gap_path, local_advantage_path, local_drop_path = write_auxiliary_tables(rows, out_dir)
    local_advantage_md_path = write_local_advantage_markdown(local_advantage_path)
    local_drop_md_path = write_local_drop_markdown(local_drop_path)
    local_drop_significance_path = write_local_drop_significance(rows, out_dir)
    print('')
    print(f'fusion_table_csv: {csv_path}')
    print(f'fusion_table_md: {md_path}')
    if fusion_significance_path:
        print(f'fusion_significance_csv: {fusion_significance_path}')
    if fusion_significance_md_path:
        print(f'fusion_significance_md: {fusion_significance_md_path}')
    if relationship_path:
        print(f'query_gate_relationships_aggregate: {relationship_path}')
    if recent_gap_path:
        print(f'recent_gap_gate_aggregate: {recent_gap_path}')
    if local_advantage_path:
        print(f'local_advantage_gate_aggregate: {local_advantage_path}')
    if local_advantage_md_path:
        print(f'local_advantage_gate_aggregate_md: {local_advantage_md_path}')
    if local_drop_path:
        print(f'local_observation_drop_robustness_aggregate: {local_drop_path}')
    if local_drop_md_path:
        print(f'local_observation_drop_robustness_aggregate_md: {local_drop_md_path}')
    if local_drop_significance_path:
        print(f'local_observation_drop_significance: {local_drop_significance_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
