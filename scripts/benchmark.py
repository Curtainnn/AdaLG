#!/usr/bin/env python
import argparse
import glob
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import shutil
import uuid
from pathlib import Path

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_CONFIG_DIR = os.path.join(REPO_ROOT, 'configs', 'models')
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


METRIC_PATTERN = re.compile(r'([A-Za-z_]+):([-+0-9.eE]+)')
METRIC_NAMES = ['mse', 'mae', 'rmse', 'mape', 'mspe', 'smape']
DATASETS = {
    'human_activity': {
        'aliases': ('human', 'astgi_human'),
        'slug': 'human_activity',
        'id': 'HumanActivity',
        'data_name': 'HumanActivityASTGIPointForecast',
        'root_path': './data_provider/Dataset/human_activity/',
        'data_path': 'human_activity_1ms.csv',
        'seq_len': 3000,
        'input_token_len': 200,
        'output_token_len': 500,
        'train_epochs': 100,
        'batch_size': 16,
    },
    'ushcn': {
        'aliases': (),
        'slug': 'ushcn',
        'id': 'USHCN',
        'data_name': 'USHCNPointForecast',
        'root_path': './data_provider/Dataset/ushcn',
        'data_path': 'small_chunked_sporadic.csv',
        'seq_len': 150,
        'input_token_len': 25,
        'output_token_len': 3,
        'train_epochs': 100,
        'batch_size': 16,
    },
    'physionet': {
        'aliases': ('p12',),
        'slug': 'p12',
        'id': 'PhysioNet',
        'data_name': 'PhysioNetPointForecast',
        'root_path': './data_provider/Dataset/physionet/processed',
        'data_path': 'all_0.0.pt',
        'seq_len': 36,
        'input_token_len': 4,
        'output_token_len': 12,
        'train_epochs': 100,
        'batch_size': 4,
    },
    'p19': {
        'aliases': ('physionet19', 'physionet_2019'),
        'slug': 'p19',
        'id': 'P19',
        'data_name': 'P19PointForecast',
        'root_path': './data_provider/Dataset/P19/P19data',
        'data_path': 'processed_data/PT_dict_list_6.npy',
        'seq_len': 36,
        'input_token_len': 4,
        'output_token_len': 12,
        'train_epochs': 100,
        'batch_size': 128,
        'point_split_mode': 'record',
        'point_stratified_random_split': '0',
    },
    'etth1': {
        'aliases': (),
        'slug': 'etth1',
        'id': 'ETTh1',
        'data_name': 'MultivariateDatasetBenchmark',
        'root_path': './data_provider/Dataset/',
        'data_path': 'ETTh1.csv',
        'seq_len': 672,
        'input_token_len': 96,
        'output_token_len': 96,
        'train_epochs': 10,
        'batch_size': 32,
    },
    'ecl': {
        'aliases': ('electricity',),
        'slug': 'ecl',
        'id': 'ECL',
        'data_name': 'MultivariateDatasetBenchmark',
        'root_path': './data_provider/Dataset/',
        'data_path': 'electricity.csv',
        'seq_len': 672,
        'input_token_len': 96,
        'output_token_len': 96,
        'train_epochs': 10,
        'batch_size': 4,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description='Run a fair forecast benchmark protocol.')
    parser.add_argument('model', nargs='?', default=os.environ.get('MODEL', 'timer_xl'))
    parser.add_argument('dataset', nargs='?', default=os.environ.get('DATASET', 'human_activity'))
    parser.add_argument('--dry-run', action='store_true', help='print resolved protocol and command without training')
    parser.add_argument('--resume', action='store_true', help='reuse completed training checkpoints; rerun test/analyses and train missing seeds')
    parser.add_argument('--diagnose-existing', action='store_true',
                        help='load existing per-seed gate checkpoints and report gate-train/validation calibration without retraining or testing')
    parser.add_argument('--existing-run-prefix', default='',
                        help='exact compact run prefix from a previous Seed Summary; only used with --diagnose-existing')
    parser.add_argument(
        '--log-dir',
        default=os.environ.get('BENCHMARK_LOG_DIR', os.path.join('logs', 'benchmark')),
        help='directory used to save per-seed benchmark logs',
    )
    parser.add_argument(
        '--analysis-dir',
        default=os.environ.get('BENCHMARK_ANALYSIS_DIR', os.path.join('analysis', 'benchmark')),
        help='directory used to save parsed benchmark reports',
    )
    parser.add_argument('--no-analyze', action='store_true', help='skip automatic log analysis after training')
    return parser.parse_args()


def env_value(name, default, cast=str):
    value = os.environ.get(name, '')
    if value == '':
        return default
    return cast(value)


def compact_path_component(value, max_len=120):
    value = str(value)
    if len(value) <= max_len:
        return value
    digest = hashlib.sha1(value.encode('utf-8')).hexdigest()[:8]
    keep_len = max(max_len - len(digest) - 2, 1)
    prefix = value[:keep_len].rstrip('_-.')
    return f'{prefix}_h{digest}'


def compact_des_prefix(des_prefix, max_len=110):
    des_prefix = str(des_prefix)
    digest = hashlib.sha1(des_prefix.encode('utf-8')).hexdigest()[:8]
    suffix = f'_h{digest}'
    if len(des_prefix) + len(suffix) <= max_len:
        return f'{des_prefix}{suffix}'
    keep_len = max(max_len - len(suffix), 1)
    prefix = des_prefix[:keep_len].rstrip('_-.')
    return f'{prefix}{suffix}'


def safe_path_component(value, max_len=140):
    value = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(value)).strip('._-') or 'run'
    return compact_path_component(value, max_len=max_len)


def canonical_dataset_key(dataset_key):
    requested = str(dataset_key).lower()
    for key, config in DATASETS.items():
        if requested == key or requested in config.get('aliases', ()):
            return key
    supported = ', '.join(sorted(DATASETS))
    raise ValueError(f'Unsupported dataset key: {dataset_key}. Supported: {supported}')


def load_dataset_config(dataset_key):
    canonical_key = canonical_dataset_key(dataset_key)
    base = dict(DATASETS[canonical_key])
    p19_split = env_value('P19_SPLIT', '1') if canonical_key == 'p19' else None
    config = {
        'key': canonical_key,
        'slug': base['slug'],
        'id': base['id'],
        'data_name': base['data_name'],
        'root_path': env_value('ROOT_PATH', base['root_path']),
        'data_path': env_value(
            'DATA_PATH',
            (
                f"processed_data/PT_dict_list_6.npy,splits/phy19_split{p19_split}_new.npy"
                if canonical_key == 'p19'
                else base['data_path']
            ),
        ),
        'seq_len': env_value('SEQ_LEN', base['seq_len'], int),
        'input_token_len': env_value('INPUT_TOKEN_LEN', base['input_token_len'], int),
        'output_token_len': env_value('OUTPUT_TOKEN_LEN', base['output_token_len'], int),
        'train_epochs': env_value('TRAIN_EPOCHS', base['train_epochs'], int),
        'batch_size': env_value('BATCH_SIZE', base['batch_size'], int),
        'features': env_value('FEATURES', 'M'),
        'point_split_mode': env_value('POINT_SPLIT_MODE', base.get('point_split_mode', 'random')),
        'point_stratified_random_split': env_value(
            'POINT_STRATIFIED_RANDOM_SPLIT',
            base.get('point_stratified_random_split', '1'),
        ),
    }
    if p19_split is not None:
        config['p19_split'] = p19_split
    config['test_seq_len'] = env_value('TEST_SEQ_LEN', config['seq_len'], int)
    config['test_pred_len'] = env_value('TEST_PRED_LEN', config['output_token_len'], int)
    return config


def prepare_dataset_if_needed(config, python_executable):
    if config['data_name'] != 'HumanActivityASTGIPointForecast':
        return

    root_path = config['root_path'].rstrip('/\\')
    processed_file = os.path.join(root_path, config['data_path'])
    raw_file = os.path.join(root_path, 'ConfLongDemo_JSI.txt')
    if os.path.exists(processed_file):
        return
    if not os.path.exists(raw_file):
        raise FileNotFoundError(
            f'Missing processed file: {processed_file}; missing raw file required for preprocessing: {raw_file}'
        )

    print(f'Processed Human Activity file not found. Generating {processed_file} from {raw_file} ...')
    subprocess.check_call(
        [
            python_executable,
            'scripts/data/prepare_human_activity_posture_dataset.py',
            '--input',
            raw_file,
            '--output-dir',
            root_path,
        ]
    )


def resolve_python_executable():
    if os.environ.get('PYTHON'):
        return os.environ['PYTHON']
    for relative_path in (os.path.join('.venv', 'Scripts', 'python.exe'), os.path.join('.venv', 'bin', 'python')):
        candidate = os.path.join(REPO_ROOT, relative_path)
        if os.path.exists(candidate):
            return candidate
    return sys.executable


def load_model_config(model_key, dataset, env=None):
    aliases = {
        'timerxl': 'timer_xl',
        'timer_xl': 'timer_xl',
        'timer-xl': 'timer_xl',
        'adalg': 'adalg',
        'ada_lg': 'adalg',
        'ada-lg': 'adalg',
        'astgi': 'astgi',
        'dlinear': 'dlinear',
        'd_linear': 'dlinear',
        'd-linear': 'dlinear',
        'grud': 'gru_d',
        'gru_d': 'gru_d',
        'gru-d': 'gru_d',
        'grafiti': 'grafiti',
        'graffiti': 'grafiti',
        'gra-fiti': 'grafiti',
        'timecheat': 'timecheat',
        'time_cheat': 'timecheat',
        'time-cheat': 'timecheat',
        'mtan': 'mtan',
        'm_tan': 'mtan',
        'm-tan': 'mtan',
    }
    requested_key = str(model_key).strip().lower().replace(' ', '_')
    model_key = aliases.get(requested_key, requested_key.replace('-', '_'))
    config_path = os.path.join(MODEL_CONFIG_DIR, f'{model_key}_config.py')
    if not os.path.exists(config_path):
        available = sorted(
            name[:-len('_config.py')]
            for name in os.listdir(MODEL_CONFIG_DIR)
            if name.endswith('_config.py')
        )
        raise ValueError(
            f"Unsupported model key: {model_key}. Available model configs: {', '.join(available)}"
        )
    spec = importlib.util.spec_from_file_location(f'benchmark_model_config_{model_key}', config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if env is not None:
        try:
            return module.load_model_config(dataset, env=env)
        except TypeError:
            pass
    return module.load_model_config(dataset)


def resolve_seed_list():
    if os.environ.get('SEED'):
        return [os.environ['SEED']]
    return os.environ.get('SEED_LIST', '2021 2022 2023 2024 2025').split()


def build_split_args(dataset):
    point_train_size = env_value('POINT_TRAIN_SIZE', '0.8')
    point_val_size = env_value('POINT_VAL_SIZE', '0.1')
    point_test_size = env_value('POINT_TEST_SIZE', '0.1')
    point_split_mode = dataset.get('point_split_mode', 'random')
    point_stratified = dataset.get('point_stratified_random_split', '1')
    point_stratify_bins = env_value('POINT_STRATIFY_BINS', '10')
    inspect_point_splits = env_value('INSPECT_POINT_SPLITS', '0')

    split_args = [
        '--point_train_size',
        point_train_size,
        '--point_val_size',
        point_val_size,
        '--point_test_size',
        point_test_size,
        '--point_split_mode',
        point_split_mode,
    ]
    protocol_suffix = f'_split_{point_split_mode}_{point_train_size}_{point_val_size}_{point_test_size}'
    if dataset.get('p19_split'):
        protocol_suffix = f"{protocol_suffix}_fold{dataset['p19_split']}"
    if point_split_mode in {'random', 'record'} and point_stratified not in {'0', 'false', 'False'}:
        split_args.extend(['--point_stratified_random_split', '--point_stratify_bins', point_stratify_bins])
        protocol_suffix = f'{protocol_suffix}_stratified'
    if inspect_point_splits not in {'0', 'false', 'False'}:
        split_args.append('--inspect_point_splits')

    split_desc = {
        'mode': point_split_mode,
        'ratios': f'{point_train_size}/{point_val_size}/{point_test_size}',
        'stratified': point_stratified,
    }
    return split_args, protocol_suffix, split_desc


def build_query_evidence_args():
    enabled = str(env_value('QUERY_EVIDENCE_ANALYSIS', '0')).strip().lower() in {'1', 'true', 'yes', 'on'}
    robustness_enabled = str(env_value('LOCAL_OBSERVATION_DROP_ROBUSTNESS', '0')).strip().lower() in {
        '1',
        'true',
        'yes',
        'on',
    }

    args = []
    if enabled:
        args.extend([
            '--query_evidence_analysis',
            '--query_evidence_density_key',
            str(env_value('QUERY_EVIDENCE_DENSITY_KEY', 'local_density')),
            '--query_evidence_bins',
            str(env_value('QUERY_EVIDENCE_BINS', 3, int)),
            '--local_advantage_comparable_quantile',
            str(env_value('LOCAL_ADVANTAGE_COMPARABLE_QUANTILE', 0.2, float)),
        ])
        out_dir = str(env_value('QUERY_EVIDENCE_OUT_DIR', '')).strip()
        if out_dir:
            args.extend(['--query_evidence_out_dir', out_dir])
        save_raw = str(env_value('QUERY_EVIDENCE_SAVE_RAW', '0')).strip().lower() in {'1', 'true', 'yes', 'on'}
        if save_raw:
            args.extend([
                '--query_evidence_save_raw',
                '--query_evidence_max_raw_rows',
                str(env_value('QUERY_EVIDENCE_MAX_RAW_ROWS', 200000, int)),
            ])
    if robustness_enabled:
        args.extend([
            '--local_observation_drop_robustness',
            '--local_observation_drop_rates',
            str(env_value('LOCAL_OBSERVATION_DROP_RATES', '0,0.25,0.5,0.75')),
            '--local_degradation_modes', str(env_value('LOCAL_DEGRADATION_MODES', 'drop')),
            '--eval_local_observation_drop_seed',
            str(env_value('EVAL_LOCAL_OBSERVATION_DROP_SEED', 0, int)),
        ])
    return args


def build_common_args(dataset, model, split_args):
    train_epochs = model['train_epochs']
    return [
        '--task_name',
        'forecast',
        '--is_training',
        '1',
        '--root_path',
        dataset['root_path'],
        '--data_path',
        dataset['data_path'],
        '--model_id',
        model['model_id'],
        '--model',
        model['model_name'],
        '--data',
        dataset['data_name'],
        '--features',
        dataset['features'],
        '--seq_len',
        str(dataset['seq_len']),
        '--input_token_len',
        str(dataset['input_token_len']),
        '--output_token_len',
        str(dataset['output_token_len']),
        '--test_seq_len',
        str(dataset['test_seq_len']),
        '--test_pred_len',
        str(dataset['test_pred_len']),
        '--batch_size',
        str(model['batch_size']),
        '--num_workers',
        str(env_value('NUM_WORKERS', 0, int)),
        '--learning_rate',
        str(model['learning_rate']),
        '--optimizer',
        model['optimizer'],
        '--loss_type',
        str(env_value('LOSS_TYPE', 'mse')).lower(),
        '--checkpoint_metric',
        str(env_value('CHECKPOINT_METRIC', 'mse')).lower(),
        '--train_epochs',
        str(train_epochs),
        '--gpu',
        '0',
        '--cosine',
        '--tmax',
        str(train_epochs),
        '--patience',
        str(env_value('PATIENCE', 5, int)),
        *split_args,
        *model['args'],
        *build_query_evidence_args(),
    ]


def print_protocol(dataset, model, split_desc, seed_list, des_prefix, run_des_prefix, python_executable):
    print('Benchmark Protocol')
    print(f"python: {python_executable}")
    print(f"model: {model['model_name']}")
    print(f"dataset: {dataset['data_name']} ({dataset['key']})")
    print(f"data: {os.path.join(dataset['root_path'], dataset['data_path'])}")
    print(f"split: mode={split_desc['mode']}, ratios={split_desc['ratios']}, stratified={split_desc['stratified']}")
    print(f"seeds: {' '.join(seed_list)}")
    print('metrics: normalized evaluation metrics (mse, mae, rmse, mape, mspe, smape)')
    print(f'des_prefix: {des_prefix}')
    print(f'run_des_prefix: {run_des_prefix}')
    print('')


def build_seed_command(python_executable, common_args, seed, des_prefix, resume=False):
    seed_des = f'{des_prefix}_seed{seed}'
    run_args = list(common_args)
    resumed = False
    if resume:
        checkpoints_root = (run_args[run_args.index('--checkpoints') + 1]
                            if '--checkpoints' in run_args else 'checkpoints')
        pattern = os.path.join(checkpoints_root, f'*{seed_des}_h*_0', 'checkpoint.pth')
        paths = sorted(glob.glob(pattern))
        if len(paths) > 1:
            raise ValueError(f'Ambiguous resume for seed={seed}: {paths}. Use distinct experiment prefixes.')
        if paths and os.path.exists(os.path.join(os.path.dirname(paths[0]), 'validation_selection.json')):
            with open(os.path.join(os.path.dirname(paths[0]), 'validation_selection.json'), encoding='utf-8') as file_obj:
                manifest = json.load(file_obj)
            if not math.isfinite(float(manifest['best_val_metric'])) or int(manifest['best_epoch']) <= 0:
                raise ValueError(f'Invalid completed-training manifest for {paths[0]}.')
            run_args[run_args.index('--is_training') + 1] = '0'
            run_args.extend(['--test_dir', os.path.basename(os.path.dirname(paths[0]))])
            resumed = True
            print(f'[resume] seed={seed} checkpoint={paths[0]}; rerunning test/analyses without training.', flush=True)
    command = [
        python_executable,
        '-u',
        'run.py',
        *run_args,
        '--seed',
        str(seed),
        '--des',
        seed_des,
    ]
    return command, resumed


def archive_incompatible_checkpoint(command, log_path, results_root='test_results',
                                    reason='checkpoint architecture incomplete or incompatible'):
    checkpoints_root = Path(command[command.index('--checkpoints') + 1] if '--checkpoints' in command else 'checkpoints').resolve()
    setting = command[command.index('--test_dir') + 1]
    source = (checkpoints_root / setting).resolve()
    source.relative_to(checkpoints_root)
    if source == checkpoints_root:
        raise ValueError('Cannot archive the checkpoint root itself.')
    archive_root = (checkpoints_root / 'incompatible_checkpoints').resolve()
    archive_root.relative_to(checkpoints_root)
    archive_root.mkdir(parents=True, exist_ok=True)
    archive = archive_root / uuid.uuid4().hex[:12]
    archive.resolve().relative_to(checkpoints_root)
    shutil.move(str(source), str(archive))
    if os.path.exists(log_path):
        shutil.copy2(log_path, archive / 'previous_run.log')
    tests_root = Path(results_root).resolve()
    previous_tests = (tests_root / setting).resolve()
    previous_tests.relative_to(tests_root)
    if previous_tests != tests_root and previous_tests.exists():
        shutil.move(str(previous_tests), str(archive / 'previous_test_results'))
    with open(archive / 'incompatible_run.json', 'w', encoding='utf-8') as file_obj:
        json.dump({'original_checkpoint_dir': str(source), 'reason': reason,
                   'evaluation_command': command}, file_obj, indent=2)
    return archive


def run_seed(python_executable, common_args, seed, des_prefix, child_env, log_dir, resume=False):
    seed_des = f'{des_prefix}_seed{seed}'
    command, resumed = build_seed_command(python_executable, common_args, seed, des_prefix, resume=resume)
    print(f'Running with seed={seed}, des={seed_des}')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'seed{seed}.log')
    print(f'log: {log_path}')
    def execute(child_command, append):
        with open(log_path, 'a' if append else 'w', encoding='utf-8', errors='replace') as log_file:
            if append:
                log_file.write('\n[resume] Reusing completed training checkpoint; rerunning test/analyses.\n')
            process = subprocess.Popen(
                child_command, env=child_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding='utf-8', errors='replace', bufsize=1,
            )
            for line in process.stdout:
                print(line, end='')
                log_file.write(line)
            return process.wait()

    status = execute(command, resumed)
    if resumed and status == 86:
        archive = archive_incompatible_checkpoint(command, log_path)
        print(f'[resume_retrain] seed={seed} incompatible checkpoint and prior results preserved at {archive}; retraining this seed.', flush=True)
        command, _ = build_seed_command(python_executable, common_args, seed, des_prefix, resume=False)
        status = execute(command, False)
    return status, log_path


def diagnose_existing_seed(python_executable, common_args, seed, des_prefix, child_env, log_dir):
    checkpoint_pattern = os.path.join(
        'checkpoints', f'*{des_prefix}_seed{seed}_h*_0', 'checkpoint.pth'
    )
    checkpoint_paths = sorted(glob.glob(checkpoint_pattern))
    if len(checkpoint_paths) != 1:
        raise ValueError(
            f'Expected exactly one saved checkpoint for seed {seed}; '
            f'found {len(checkpoint_paths)} from {checkpoint_pattern!r}: {checkpoint_paths}'
        )
    setting = os.path.basename(os.path.dirname(checkpoint_paths[0]))
    diagnostic_args = list(common_args)
    diagnostic_args[diagnostic_args.index('--is_training') + 1] = '0'
    command = [
        python_executable, '-u', 'run.py', *diagnostic_args,
        '--seed', str(seed), '--des', f'{des_prefix}_seed{seed}',
        '--test_dir', setting, '--gate_split_diagnostics_only',
    ]
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'gate_diagnostics_seed{seed}.log')
    print(f'Diagnosing seed={seed} checkpoint={checkpoint_paths[0]}')
    with open(log_path, 'w', encoding='utf-8', errors='replace') as log_file:
        process = subprocess.Popen(
            command,
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1,
        )
        for line in process.stdout:
            print(line, end='')
            log_file.write(line)
        status = process.wait()
    return status, log_path


def parse_metrics_file(path):
    metrics = {}
    with open(path, 'r', encoding='utf-8') as file_obj:
        for line in file_obj:
            for key, value in METRIC_PATTERN.findall(line):
                metrics[key] = float(value)
    return metrics


def find_seed_metrics(des_prefix, seed_list):
    collected = []
    missing = []
    for seed in seed_list:
        pattern = os.path.join('test_results', f'*{des_prefix}_seed{seed}*', 'metrics.txt')
        matches = glob.glob(pattern)
        if not matches:
            missing.append(seed)
            continue
        metrics_path = max(matches, key=os.path.getmtime)
        collected.append((seed, metrics_path, parse_metrics_file(metrics_path)))
    return collected, missing


def mean_std(values):
    avg = sum(values) / len(values)
    std = math.sqrt(sum((value - avg) ** 2 for value in values) / len(values)) if len(values) > 1 else 0.0
    return avg, std


def print_summary(des_prefix, seed_list):
    collected, missing = find_seed_metrics(des_prefix, seed_list)
    print('')
    print('Seed Summary')
    print(f'des_prefix: {des_prefix}')
    if not collected:
        print('No seed metrics found.')
        return 1

    print('completed_seeds: {}'.format(' '.join(seed for seed, _, _ in collected)))
    if missing:
        print('missing_seeds: {}'.format(' '.join(missing)))

    for metric_name in METRIC_NAMES:
        values = [metrics[metric_name] for _, _, metrics in collected if metric_name in metrics]
        if not values:
            continue
        avg, std = mean_std(values)
        print(f'{metric_name.upper()}: {avg:.6f} +/- {std:.6f} (n={len(values)})')
    return 0


def run_analysis_report(python_executable, report_name, log_paths, metric_paths, out_dir):
    command = [
        python_executable,
        'scripts/analyze_experiment_logs.py',
        '--name',
        report_name,
        '--out-dir',
        out_dir,
    ]
    if log_paths:
        command.extend(['--logs', *log_paths])
    if metric_paths:
        command.extend(['--metrics', *metric_paths])

    print('')
    print('Analysis Report')
    print(f'output: {out_dir}')
    status = subprocess.call(command)
    if status == 0:
        print(f'efficiency_report: {os.path.join(out_dir, "efficiency_report.csv")}')
    return status


def main():
    args = parse_args()
    if args.dry_run and args.diagnose_existing:
        raise ValueError('--dry-run and --diagnose-existing cannot be combined.')
    if args.existing_run_prefix and not args.diagnose_existing:
        raise ValueError('--existing-run-prefix requires --diagnose-existing.')
    os.chdir(REPO_ROOT)
    python_executable = resolve_python_executable()
    dataset = load_dataset_config(args.dataset)
    model = load_model_config(args.model, dataset)
    seed_list = resolve_seed_list()
    split_args, protocol_suffix, split_desc = build_split_args(dataset)
    des_prefix = f"{model['des']}{protocol_suffix}{model.get('model_des_suffix', '')}"
    run_des_prefix = args.existing_run_prefix or compact_des_prefix(des_prefix)
    common_args = build_common_args(dataset, model, split_args)
    report_component = safe_path_component(f"{dataset['slug']}_{model['model_name']}_{run_des_prefix}")
    log_dir = os.path.join(args.log_dir, report_component)
    analysis_out_dir = os.path.join(args.analysis_dir, report_component)
    report_name = f"{model['model_name']}_{dataset['slug']}"

    child_env = os.environ.copy()
    child_env.setdefault('CUDA_VISIBLE_DEVICES', os.environ.get('GPU', '0'))

    print_protocol(dataset, model, split_desc, seed_list, des_prefix, run_des_prefix, python_executable)
    print(f'log_dir: {log_dir}')
    print(f'analysis_out_dir: {analysis_out_dir}')
    print('')
    if args.dry_run:
        preview_command = [
            python_executable,
            '-u',
            'run.py',
            *common_args,
            '--seed',
            seed_list[0],
            '--des',
            f'{run_des_prefix}_seed{seed_list[0]}',
        ]
        print('Dry run command:')
        if args.resume:
            preview_command, _ = build_seed_command(python_executable, common_args, seed_list[0], run_des_prefix, resume=True)
        print(' '.join(preview_command))
        return 0

    if args.diagnose_existing:
        if model['model_name'] != 'AdaLG':
            raise ValueError('--diagnose-existing currently supports AdaLG only.')
        for seed in seed_list:
            status, _ = diagnose_existing_seed(
                python_executable, common_args, seed, run_des_prefix, child_env, log_dir
            )
            if status != 0:
                return status
        return 0

    prepare_dataset_if_needed(dataset, python_executable)

    log_paths = []
    for seed in seed_list:
        status, log_path = run_seed(python_executable, common_args, seed, run_des_prefix, child_env, log_dir, resume=args.resume)
        log_paths.append(log_path)
        if status != 0:
            return status

    summary_status = print_summary(run_des_prefix, seed_list)
    if args.no_analyze:
        return summary_status

    collected, _ = find_seed_metrics(run_des_prefix, seed_list)
    metric_paths = [metrics_path for _, metrics_path, _ in collected]
    analysis_status = run_analysis_report(
        python_executable,
        report_name,
        log_paths,
        metric_paths,
        analysis_out_dir,
    )
    return summary_status or analysis_status


if __name__ == '__main__':
    raise SystemExit(main())
