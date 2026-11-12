#!/usr/bin/env python
import argparse
import contextlib
import csv
import importlib
import io
import math
import os
import subprocess
import sys
from types import SimpleNamespace

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)


def reexec_with_repo_python():
    if os.environ.get('PARAM_MATCH_NO_REEXEC') == '1':
        return
    candidates = []
    if os.environ.get('PYTHON'):
        candidates.append(os.environ['PYTHON'])
    candidates.extend([
        os.path.join(REPO_ROOT, '.venv', 'Scripts', 'python.exe'),
        os.path.join(REPO_ROOT, '.venv', 'bin', 'python'),
    ])
    current = os.path.normcase(os.path.abspath(sys.executable))
    for candidate in candidates:
        if not candidate or not os.path.exists(candidate):
            continue
        candidate_abs = os.path.normcase(os.path.abspath(candidate))
        if candidate_abs == current:
            return
        child_env = dict(os.environ)
        child_env['PARAM_MATCH_NO_REEXEC'] = '1'
        raise SystemExit(subprocess.call([candidate, *sys.argv], env=child_env))


reexec_with_repo_python()

import benchmark
from exp.exp_basic import get_model_registry


MODEL_DEFAULTS = {
    'input_token_len': 576,
    'dropout': 0.1,
    'e_layers': 1,
    'd_model': 512,
    'n_heads': 8,
    'd_ff': 2048,
    'activation': 'relu',
    'covariate': False,
    'node_num': 100,
    'use_norm': False,
    'output_attention': False,
    'flash_attention': False,
    'point_hidden_dim': 128,
    'point_token_subbins': 4,
    'point_time_dim': 32,
    'point_local_fusion_mode': 'dual_gate',
    'point_force_fusion': 'auto',
    'point_disable_local_branch': False,
    'point_fixed_gate': None,
    'point_local_fusion_k': 8,
    'point_local_gate_bias': -4.0,
    'point_dual_gate_bias': -1.0,
    'point_gate_max': 1.0,
    'point_causal_context_mode': 'normal',
    'point_use_node_embedding': False,
    'causal_ode_state_dim': 32,
    'causal_ode_hidden': 128,
    'causal_ode_constraint_strength': 0.0,
    'causal_ode_h_weight': 0.001,
    'causal_ode_l1_weight': 0.0001,
    'causal_ode_diag_weight': 0.001,
    'causal_ode_laplacian_weight': 0.0001,
    'causal_ode_adjacency_scale': 1.0,
    'causal_ode_gate_init': -4.0,
    'causal_ode_steps': 8,
    'causal_prior_window': 30,
    'causal_prior_mode': 'normal',
    'causal_local_alpha': 0.0,
    'causal_local_beta': 1.0,
    'causal_local_expand_pool': False,
}


def parse_args():
    parser = argparse.ArgumentParser(description='Find parameter-matched AdaLG global-only controls.')
    parser.add_argument('dataset', nargs='?', default=os.environ.get('DATASET', 'human_activity'))
    parser.add_argument('--model', default=os.environ.get('MODEL', 'AdaLG'))
    parser.add_argument('--num-vars', type=int, default=None, help='override dataset variable count')
    parser.add_argument('--d-models', default='', help='comma/space separated candidate d_model values')
    parser.add_argument('--d-ff-ratios', default='2,4,6,8', help='candidate d_ff ratios relative to d_model')
    parser.add_argument('--e-layers', default='', help='comma/space separated candidate encoder layer counts')
    parser.add_argument('--max-d-model', type=int, default=None)
    parser.add_argument('--top-k', type=int, default=12)
    parser.add_argument('--out-dir', default=os.path.join('analysis', 'param_match'))
    return parser.parse_args()


def split_ints(text):
    values = []
    for item in str(text).replace(';', ',').replace(' ', ',').split(','):
        item = item.strip()
        if item:
            values.append(int(item))
    return values


def split_floats(text):
    values = []
    for item in str(text).replace(';', ',').replace(' ', ',').split(','):
        item = item.strip()
        if item:
            values.append(float(item))
    return values


def convert_value(raw_value, default_value):
    if raw_value is True:
        return True
    if default_value is None:
        if raw_value in {'', None}:
            return None
        try:
            return float(raw_value)
        except (TypeError, ValueError):
            return raw_value
    if isinstance(default_value, bool):
        return str(raw_value).strip().lower() in {'1', 'true', 'yes', 'on'}
    if isinstance(default_value, int) and not isinstance(default_value, bool):
        return int(raw_value)
    if isinstance(default_value, float):
        return float(raw_value)
    return raw_value


def parse_model_args(arg_list):
    values = dict(MODEL_DEFAULTS)
    idx = 0
    while idx < len(arg_list):
        token = arg_list[idx]
        if not str(token).startswith('--'):
            idx += 1
            continue
        key_value = str(token)[2:]
        if '=' in key_value:
            key, raw_value = key_value.split('=', 1)
            idx += 1
        elif idx + 1 < len(arg_list) and not str(arg_list[idx + 1]).startswith('--'):
            key = key_value
            raw_value = arg_list[idx + 1]
            idx += 2
        else:
            key = key_value
            raw_value = True
            idx += 1
        key = key.replace('-', '_')
        values[key] = convert_value(raw_value, values.get(key, ''))
    return values


def build_dataset_args(dataset):
    return SimpleNamespace(
        data=dataset['data_name'],
        root_path=dataset['root_path'],
        data_path=dataset['data_path'],
        seq_len=dataset['seq_len'],
        input_token_len=dataset['input_token_len'],
        output_token_len=dataset['output_token_len'],
        test_seq_len=dataset['test_seq_len'],
        test_pred_len=dataset['test_pred_len'],
        nonautoregressive=True,
        test_flag='T',
        subset_rand_ratio=1.0,
        point_train_size=float(os.environ.get('POINT_TRAIN_SIZE', '0.8')),
        point_val_size=float(os.environ.get('POINT_VAL_SIZE', '0.1')),
        point_test_size=float(os.environ.get('POINT_TEST_SIZE', '0.1')),
        point_split_mode=dataset.get('point_split_mode', 'random'),
        point_stratified_random_split=str(dataset.get('point_stratified_random_split', '1')) not in {'0', 'false', 'False'},
        point_stratify_bins=int(os.environ.get('POINT_STRATIFY_BINS', '10')),
        point_align_val_with_test=False,
    )


def resolve_num_vars(dataset, override):
    if override is not None:
        return int(override)
    benchmark.prepare_dataset_if_needed(dataset, benchmark.resolve_python_executable())
    try:
        from data_provider.data_factory import resolve_dataset_num_vars

        return resolve_dataset_num_vars(build_dataset_args(dataset), flag='train')
    except Exception as exc:
        raise SystemExit(
            f'Could not infer num_vars for {dataset["key"]}: {exc}. '
            'Retry with --num-vars N.'
        )


def env_with_overrides(base_env, **overrides):
    env = dict(base_env)
    for key, value in overrides.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = str(value)
    return env


def model_values(model_key, dataset, env, num_vars):
    config = benchmark.load_model_config(model_key, dataset, env=env)
    values = parse_model_args(config['args'])
    values['input_token_len'] = dataset['input_token_len']
    values['point_num_vars'] = int(num_vars)
    values['n_vars'] = int(num_vars)
    values['enc_in'] = int(num_vars)
    values['dec_in'] = int(num_vars)
    values['c_out'] = int(num_vars)
    return config, values


def count_model_params(model_name, values):
    registry, import_errors = get_model_registry()
    module = registry.get(model_name)
    if module is None:
        error_details = ''
        if import_errors:
            error_details = ' Import failures: ' + '; '.join(
                f'{name}: {exc}' for name, exc in sorted(import_errors.items())
            )
        raise ValueError(f'Unknown model {model_name!r}.{error_details}')
    args = SimpleNamespace(**values)
    with contextlib.redirect_stdout(io.StringIO()):
        model = module.Model(args)
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return int(total_params), int(trainable_params)


def round_to_multiple(value, multiple):
    multiple = max(int(multiple), 1)
    return max(multiple, int(round(float(value) / multiple) * multiple))


def candidate_d_models(base_d_model, n_heads, args):
    def is_valid_d_model(value):
        if value % n_heads != 0:
            return False
        head_dim = value // n_heads
        rotary_width = int(head_dim * 0.5)
        return rotary_width > 0 and rotary_width % 2 == 0

    if args.d_models:
        return sorted(set(value for value in split_ints(args.d_models) if is_valid_d_model(value)))
    max_d_model = args.max_d_model or max(base_d_model * 2, base_d_model + 256)
    step = max(int(n_heads), 1)
    return [value for value in range(base_d_model, max_d_model + 1, step) if is_valid_d_model(value)]


def candidate_e_layers(base_e_layers, args):
    if args.e_layers:
        return sorted(set(split_ints(args.e_layers)))
    return sorted(set([max(1, base_e_layers - 1), base_e_layers, base_e_layers + 1]))


def command_for_candidate(model_key, dataset_key, row):
    env_parts = [
        "$env:POINT_DISABLE_LOCAL_BRANCH='1'",
        "$env:POINT_FORCE_FUSION='global'",
        f"$env:D_MODEL='{row['d_model']}'",
        f"$env:D_FF='{row['d_ff']}'",
        f"$env:E_LAYERS='{row['e_layers']}'",
        "$env:DES='param_matched_global_only'",
    ]
    return '; '.join(env_parts + [f'python scripts/benchmark.py {model_key} {dataset_key}'])


def write_outputs(rows, out_dir, report_name):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f'{report_name}.csv')
    md_path = os.path.join(out_dir, f'{report_name}.md')
    fieldnames = [
        'rank',
        'target_total_params',
        'candidate_total_params',
        'candidate_trainable_params',
        'abs_diff',
        'diff_pct',
        'd_model',
        'd_ff',
        'e_layers',
        'n_heads',
        'command',
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in fieldnames})
    with open(md_path, 'w', encoding='utf-8') as file_obj:
        file_obj.write('| Rank | Params | Diff | Diff % | d_model | d_ff | layers |\n')
        file_obj.write('| ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n')
        for row in rows:
            file_obj.write(
                '| {rank} | {candidate_total_params} | {abs_diff} | {diff_pct:.3f} | '
                '{d_model} | {d_ff} | {e_layers} |\n'.format(**row)
            )
    return csv_path, md_path


def main():
    args = parse_args()
    normalized_model = str(args.model).strip().lower().replace(' ', '_').replace('-', '_')
    if normalized_model not in {'adalg', 'ada_lg'}:
        raise SystemExit('param_matched_control.py currently supports --model AdaLG only.')
    args.model = 'AdaLG'

    dataset = benchmark.load_dataset_config(args.dataset)
    num_vars = resolve_num_vars(dataset, args.num_vars)
    base_env = os.environ.copy()
    target_env = env_with_overrides(
        base_env,
        POINT_DISABLE_LOCAL_BRANCH=None,
        POINT_FORCE_FUSION='auto',
        POINT_FIXED_GATE=None,
    )
    target_config, target_values = model_values(args.model, dataset, target_env, num_vars)
    target_total, target_trainable = count_model_params(target_config['model_name'], target_values)

    base_d_model = int(target_values['d_model'])
    base_d_ff = int(target_values['d_ff'])
    base_e_layers = int(target_values['e_layers'])
    n_heads = int(target_values['n_heads'])
    d_ff_ratios = split_floats(args.d_ff_ratios)
    d_ff_values = {base_d_ff}

    candidates = []
    for d_model in candidate_d_models(base_d_model, n_heads, args):
        for ratio in d_ff_ratios:
            d_ff_values.add(round_to_multiple(d_model * ratio, n_heads))
        for d_ff in sorted(d_ff_values):
            for e_layers in candidate_e_layers(base_e_layers, args):
                candidate_env = env_with_overrides(
                    target_env,
                    POINT_DISABLE_LOCAL_BRANCH='1',
                    POINT_FORCE_FUSION='global',
                    D_MODEL=d_model,
                    D_FF=d_ff,
                    E_LAYERS=e_layers,
                )
                candidate_config, candidate_values = model_values(args.model, dataset, candidate_env, num_vars)
                total_params, trainable_params = count_model_params(candidate_config['model_name'], candidate_values)
                abs_diff = abs(total_params - target_total)
                diff_pct = abs_diff / float(target_total) * 100.0 if target_total else math.inf
                candidates.append({
                    'target_total_params': target_total,
                    'target_trainable_params': target_trainable,
                    'candidate_total_params': total_params,
                    'candidate_trainable_params': trainable_params,
                    'abs_diff': abs_diff,
                    'diff_pct': diff_pct,
                    'd_model': d_model,
                    'd_ff': d_ff,
                    'e_layers': e_layers,
                    'n_heads': n_heads,
                })

    candidates.sort(key=lambda row: (row['diff_pct'], row['candidate_total_params']))
    top_rows = []
    for rank, row in enumerate(candidates[: max(int(args.top_k), 1)], start=1):
        row = dict(row)
        row['rank'] = rank
        row['command'] = command_for_candidate(args.model, dataset['key'], row)
        top_rows.append(row)

    report_name = f'{dataset["key"]}_{args.model}_param_matched_global_only'
    csv_path, md_path = write_outputs(top_rows, args.out_dir, report_name)
    best = top_rows[0]
    print(f'target_total_params: {target_total}')
    print(
        'best_candidate: params={candidate_total_params} diff={abs_diff} '
        'diff_pct={diff_pct:.3f}% d_model={d_model} d_ff={d_ff} e_layers={e_layers}'.format(**best)
    )
    print(f'csv: {csv_path}')
    print(f'md: {md_path}')
    print(f'run_command: {best["command"]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
