#!/usr/bin/env python
"""Isolate gate bound, availability inputs, and reliability supervision."""
import argparse
from contextlib import contextmanager
import csv
import hashlib
import importlib
import itertools
import json
import math
import os
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import benchmark
import fusion_ablation as fusion

FACTORS = ('availability', 'supervision', 'bounded')


class InvalidGateCheckpointError(RuntimeError):
    """A trained gate cannot be audited against its branch-only source."""


def require_training_provenance_support():
    try:
        helper = importlib.import_module('utils.gate_mechanism')
    except ModuleNotFoundError as error:
        if error.name not in {'utils', 'utils.gate_mechanism'}:
            raise
        raise RuntimeError(
            'Incomplete server update: missing utils/gate_mechanism.py. '
            'Sync the complete update package to the AdaLG project root, '
            'including utils/gate_mechanism.py, before running with --resume. '
            'No checkpoint has been archived or trained.') from error
    if not all(callable(getattr(helper, name, None))
               for name in ('reset_gate', 'unrestricted_oracle_error')):
        raise RuntimeError('Incomplete utils/gate_mechanism.py; sync it from the same update package.')
    from exp.exp_forecast import Exp_Forecast
    if getattr(Exp_Forecast, 'gate_holdout_manifest_version', 0) < 2:
        raise RuntimeError(
            'Server training code is outdated. Sync exp/exp_forecast.py, '
            'scripts/benchmark.py, scripts/gate_mechanism_ablation.py and '
            'utils/gate_mechanism.py from the '
            'same update package before running/resuming. Gate holdout provenance '
            'must be saved for both branch and gate stages.')


def read_checkpoint_record(path):
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise InvalidGateCheckpointError(f'Missing/invalid checkpoint record: {path}') from error
    if not isinstance(value, dict):
        raise InvalidGateCheckpointError(f'Checkpoint record must be an object: {path}')
    return value


def validate_frozen_gate(checkpoint, source):
    """Audit before resume; repair only from recorded training provenance."""
    protocol = completed_source(source)
    validation = read_checkpoint_record(checkpoint.parent / 'validation_selection.json')
    try:
        valid = (math.isfinite(float(validation['best_val_metric']))
                 and int(validation['best_epoch']) > 0
                 and validation['test_evaluated_during_training'] is False)
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise InvalidGateCheckpointError(f'Incomplete/invalid gate training: {checkpoint}')
    source_selection = read_checkpoint_record(source.parent / 'validation_selection.json')
    if validation.get('seed') != source_selection.get('seed'):
        raise InvalidGateCheckpointError(f'Gate/source seed mismatch: {checkpoint}')
    manifest_path = checkpoint.parent / 'gate_holdout_manifest.json'
    restore = not manifest_path.exists()
    manifest = (validation.get('gate_holdout_manifest') if restore
                else read_checkpoint_record(manifest_path))
    if (not isinstance(manifest, dict) or manifest.get('stage') != 'gate'
            or manifest.get('protocol') != protocol):
        raise InvalidGateCheckpointError(
            f'Gate holdout record missing or mismatched: {manifest_path}. '
            'Cannot infer which samples trained an old gate from its weights; '
            'this gate must be retrained on the verified holdout.')
    recorded = validation.get('gate_holdout_manifest')
    if recorded is not None and recorded != manifest:
        raise InvalidGateCheckpointError(f'Conflicting gate training records: {checkpoint}')
    source_hash, n = predictor_hash(source)
    result_hash, result_n = predictor_hash(checkpoint)
    if source_hash != result_hash or n != result_n:
        raise InvalidGateCheckpointError(f'Frozen predictor weights differ from source: {checkpoint}')
    if restore:
        temporary = manifest_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding='utf-8')
        temporary.replace(manifest_path)
        print(f'[restore gate metadata] verified completed-training record: {manifest_path}', flush=True)
    return dict(source=str(source), checkpoint=str(checkpoint), predictor_sha256=result_hash,
                frozen_elements=n, verified=True, protocol_sha256=signature(protocol))


def run_frozen_gate_seed(python, common, seed, prefix, env, log_dir, source, resume=False):
    if resume:
        paths = list((ROOT / 'checkpoints').glob(f'*{prefix}_seed{seed}_h*_0/checkpoint.pth'))
        if paths:
            checkpoint = unique_checkpoint(prefix, seed)
            try:
                validate_frozen_gate(checkpoint, source)
            except InvalidGateCheckpointError as error:
                command, _ = benchmark.build_seed_command(python, common, seed, prefix, resume=False)
                command.extend(['--test_dir', checkpoint.parent.name])
                archive = benchmark.archive_incompatible_checkpoint(command, str(Path(log_dir) / f'seed{seed}.log'),
                                                                     reason=str(error))
                print(f'[gate resume retrain] seed={seed}: {error}\n'
                      f'Previous checkpoint/results archived at {archive}; retraining only this gate.', flush=True)
                resume = False
    status, log = benchmark.run_seed(python, common, seed, prefix, env, log_dir, resume=resume)
    if status:
        return status, log, None
    checkpoint = unique_checkpoint(prefix, seed)
    try:
        audit = validate_frozen_gate(checkpoint, source)
    except InvalidGateCheckpointError as error:
        raise RuntimeError(
            f'New gate checkpoint failed its audit: {error}. '
            'Sync exp/exp_forecast.py and the experiment script from the same '
            'update package, then rerun with --resume. Existing files are preserved.') from error
    return status, log, audit


@contextmanager
def environment(env):
    previous = os.environ.copy()
    os.environ.clear()
    os.environ.update(env)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(previous)


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:12]


def build_cells(cap=0.3, initial=0.1, encoder='summary', k=12, utility=0.1, oracle=0.1):
    if (not all(math.isfinite(v) for v in (cap, initial, utility, oracle))
            or not 0 < initial < cap < 1 or min(utility, oracle) <= 0 or k <= 0):
        raise ValueError('Require 0 < initial < cap < 1, positive loss weights and k.')
    common = dict(
        POINT_LOCAL_BRANCH_SOURCE='raw', POINT_LOCAL_ENCODER=encoder,
        POINT_LOCAL_FUSION_K=str(k), POINT_LOCAL_FUSION_MODE='dual_gate',
        POINT_DISABLE_LOCAL_BRANCH='0', POINT_FORCE_FUSION='auto', POINT_FIXED_GATE=None,
        POINT_GATE_PREDICTION_FEATURES='1', POINT_SEPARATE_UTILITY_HEAD='0',
        POINT_DISABLE_CAUSAL_GATE_CONTEXT='1', POINT_CAUSAL_CONTEXT_MODE='normal',
        CAUSAL_LOCAL_ALPHA='0', CAUSAL_LOCAL_BETA='1', CAUSAL_LOCAL_EXPAND_POOL='0',
        CAUSAL_PRIOR_MODE='normal', GLOBAL_AUX_LOSS_WEIGHT='0.25', LOCAL_AUX_LOSS_WEIGHT='0.25',
        POINT_TRAIN_STAGE='full', POINT_GATE_HOLDOUT_RATIO='0', POINT_GATE_INIT_CHECKPOINT=None,
        POINT_GATE_INITIAL_WEIGHT=str(initial), INPUT_OBSERVATION_DROP_ROBUSTNESS='0',
        QUERY_EVIDENCE_ANALYSIS='1', QUERY_EVIDENCE_OUT_DIR=None,
    )
    cells = []
    for a, s, b in itertools.product((0, 1), repeat=3):
        gate_cap = cap if b else 1.0
        env = dict(common, POINT_GATE_MAX=str(gate_cap),
                   POINT_LOCAL_GATE_BIAS=str(math.log(initial / (gate_cap - initial))),
                   POINT_DISABLE_GATE_AVAILABILITY_FEATURES=str(1 - a),
                   POINT_GATE_MODE='utility' if s else 'ordinary',
                   UTILITY_LOSS_WEIGHT=str(utility if s else 0),
                   ORACLE_GATE_LOSS_WEIGHT=str(oracle if s else 0))
        cells.append(dict(key=f'a{a}_s{s}_b{b}', availability=a, supervision=s, bounded=b,
                          cap=gate_cap, initial_weight=initial, env=env))
    return cells


def conditional_pairs(cells):
    """Twelve cube edges: only one conceptual factor differs per contrast."""
    pairs = []
    for factor in FACTORS:
        others = [f for f in FACTORS if f != factor]
        for off in cells:
            if off[factor]:
                continue
            on = next(c for c in cells if c[factor] and all(c[f] == off[f] for f in others))
            pairs.append((factor, on['key'], off['key']))
    return pairs


def common_args(dataset_key, env):
    with environment(env):
        dataset = benchmark.load_dataset_config(dataset_key)
        model = benchmark.load_model_config('AdaLG', dataset, env=env)
        split, _, _ = benchmark.build_split_args(dataset)
        return benchmark.build_common_args(dataset, model, split)


def unique_checkpoint(prefix, seed):
    paths = sorted((ROOT / 'checkpoints').glob(f'*{prefix}_seed{seed}_h*_0/checkpoint.pth'))
    if len(paths) != 1:
        raise RuntimeError(f'Expected exactly one checkpoint for {prefix}, seed {seed}: {paths}')
    return paths[0]


def predictor_hash(path):
    import torch
    state = torch.load(path, map_location='cpu', weights_only=True)
    digest = hashlib.sha256()
    count = 0
    for key, tensor in sorted(state.items()):
        if key.startswith('query_head.utility_gate.'):
            continue
        digest.update(key.encode())
        digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        count += tensor.numel()
    return digest.hexdigest(), count


def completed_source(path):
    manifest = json.loads((path.parent / 'validation_selection.json').read_text(encoding='utf-8'))
    holdout = json.loads((path.parent / 'gate_holdout_manifest.json').read_text(encoding='utf-8'))
    if (not math.isfinite(float(manifest['best_val_metric'])) or int(manifest['best_epoch']) < 1
            or manifest['test_evaluated_during_training'] or holdout['stage'] != 'branch'):
        raise RuntimeError(f'Invalid branch-only training source: {path}')
    return holdout['protocol']


def train_source(dataset, env, seeds, args):
    env = fusion.apply_env(env, dict(
        POINT_TRAIN_STAGE='branch', POINT_GATE_HOLDOUT_RATIO=str(args.holdout),
        POINT_GATE_SPLIT_SEED=str(args.split_seed), POINT_GATE_INIT_CHECKPOINT=None,
        POINT_GATE_INITIAL_WEIGHT=None, POINT_FIXED_GATE='0.5', POINT_GATE_MAX='1',
        POINT_GATE_MODE='ordinary', POINT_LOCAL_GATE_BIAS='0',
        POINT_DISABLE_GATE_AVAILABILITY_FEATURES='1', UTILITY_LOSS_WEIGHT='0', ORACLE_GATE_LOSS_WEIGHT='0',
        QUERY_EVIDENCE_ANALYSIS='0', LOCAL_OBSERVATION_DROP_ROBUSTNESS='0',
    ))
    if args.branch_epochs is not None:
        env['TRAIN_EPOCHS'] = str(args.branch_epochs)
    common = common_args(dataset, env) + ['--skip_test']
    prefix = f'gm_{dataset}_branch_{signature(common)}'
    python = env.get('PYTHON') or sys.executable
    for seed in seeds:
        command, _ = benchmark.build_seed_command(python, common, seed, prefix)
        if args.dry_run:
            print('[branch/no test] ' + shlex.join(command), flush=True)
            continue
        paths = list((ROOT / 'checkpoints').glob(f'*{prefix}_seed{seed}_h*_0/checkpoint.pth'))
        if args.resume and paths:
            path = unique_checkpoint(prefix, seed)
            completed_source(path)
            print(f'[resume branch] {path}', flush=True)
        else:
            status, _ = benchmark.run_seed(python, common, seed, prefix, env,
                                          str(ROOT / 'logs' / 'gate_mechanism' / prefix), resume=False)
            if status:
                raise RuntimeError(f'Branch training failed, seed={seed}, exit={status}')
            completed_source(unique_checkpoint(prefix, seed))
    return prefix, f'checkpoints/*{prefix}_seed{{seed}}_h*_0/checkpoint.pth'


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with open(path, newline='', encoding='utf-8') as file:
        return list(csv.DictReader(file))


def paired_row(factor, on, off, endpoint, condition, values_on, values_off):
    if set(values_on) != set(values_off):
        raise RuntimeError(f'Unmatched seeds for {on} vs {off}, {endpoint}, {condition}')
    seeds = sorted(values_on)
    a, b = [values_on[s] for s in seeds], [values_off[s] for s in seeds]
    if not seeds or not all(math.isfinite(v) for v in a + b):
        raise RuntimeError(f'Empty/nonfinite paired metrics for {on} vs {off}, {endpoint}, {condition}')
    avg_a, _ = benchmark.mean_std(a)
    avg_b, _ = benchmark.mean_std(b)
    diff, std, dz, t, pt, pw = fusion.paired_significance(a, b)
    # Keep effect estimates for a one-seed smoke test, without inventing p-values.
    if len(seeds) == 1:
        diff, std = a[0] - b[0], 0.0
    elif len(seeds) > 1:
        # Paired effect sizes use the sample SD of differences (ddof=1).
        std = math.sqrt(sum((x - y - diff) ** 2 for x, y in zip(a, b)) / (len(seeds) - 1))
        dz = diff / std if std else ''
    return dict(factor=factor, on=on, off=off, endpoint=endpoint, condition=condition,
                n=len(seeds), seeds=' '.join(seeds), on_mean=avg_a, off_mean=avg_b,
                diff_on_minus_off=diff, diff_std=std, improvement=avg_b - avg_a,
                improvement_pct=100 * (avg_b - avg_a) / avg_b if avg_b and endpoint in {'mse', 'mae'} else '',
                on_lower_seeds=sum(x < y for x, y in zip(a, b)),
                cohen_dz=dz, paired_t_stat=t, paired_t_p=pt, wilcoxon_p=pw, family='all')


def correct(rows):
    # One predeclared family per report, across all factors AND endpoints.
    for p in ('paired_t_p', 'wilcoxon_p'):
        fusion.holm_adjust_pvalues(rows, p, p + '_holm', group_key='family')


def reports(cells, rows, out, modes, rates):
    clean, robust, diagnostics = {}, {}, []
    for cell, row in zip(cells, rows):
        key = cell['key']
        clean[key] = {str(seed): metrics for seed, _, metrics in row['collected']}
        robust[key] = {}
        for seed, path, _ in row['collected']:
            seed = str(seed)
            folder = Path(path).parent
            calibration = read_csv(folder / 'query_gate_calibration.csv')[0]
            diagnostics.append(dict(strategy_key=key, seed=seed, **calibration))
            degradation = read_csv(folder / 'local_observation_drop_robustness.csv')
            expected = {(m, float(r)) for m in modes for r in rates}
            observed = {(r['degradation_mode'], float(r['drop_rate'])) for r in degradation}
            if observed != expected or len(degradation) != len(expected):
                raise RuntimeError(f'Incomplete/duplicate degradation conditions in {folder}')
            for r in degradation:
                r = dict(r)
                r['mse_excess_vs_global'] = float(r['mse']) - float(r['global_mse'])
                r['mae_excess_vs_global'] = float(r['mae']) - float(r['global_mae'])
                r['strategy_key'], r['seed'] = key, seed
                condition = f"{r['degradation_mode']}:{float(r['drop_rate']):g}"
                robust[key].setdefault(condition, {})[seed] = r
    clean_pairs, robust_pairs = [], []
    for factor, on, off in conditional_pairs(cells):
        for metric in ('mse', 'mae'):
            clean_pairs.append(paired_row(factor, on, off, metric, 'clean',
                                         {s: float(v[metric]) for s, v in clean[on].items()},
                                         {s: float(v[metric]) for s, v in clean[off].items()}))
        if set(robust[on]) != set(robust[off]):
            raise RuntimeError('Unmatched robustness conditions.')
        for condition in sorted(robust[on]):
            for metric in ('mse', 'mae', 'mse_change_vs_no_drop', 'mae_change_vs_no_drop'):
                robust_pairs.append(paired_row(factor, on, off, metric, condition,
                                               {s: float(v[metric]) for s, v in robust[on][condition].items()},
                                               {s: float(v[metric]) for s, v in robust[off][condition].items()}))
    correct(clean_pairs)
    correct(robust_pairs)
    fusion.write_fusion_table(rows, str(out))
    fusion.write_auxiliary_tables(rows, str(out))
    write_csv(out / 'mechanism_clean_contrasts.csv', clean_pairs)
    write_csv(out / 'mechanism_robustness_contrasts.csv', robust_pairs)
    write_csv(out / 'gate_calibration_per_seed.csv', diagnostics)
    summaries = []
    for cell in cells:
        data = [r for r in diagnostics if r['strategy_key'] == cell['key']]
        summary = dict(strategy_key=cell['key'], n=len(data), gate_cap=cell['cap'])
        for metric in data[0]:
            if metric in {'strategy_key', 'seed'}:
                continue
            values = [float(r[metric]) for r in data if r[metric] and math.isfinite(float(r[metric]))]
            summary[metric + '_n_valid'] = len(values)
            if values:
                summary[metric + '_mean'], summary[metric + '_std'] = benchmark.mean_std(values)
        summaries.append(summary)
    write_csv(out / 'gate_calibration_summary.csv', summaries)
    write_csv(out / 'gate_robustness_per_seed.csv',
              [r for conditions in robust.values() for seeds in conditions.values() for r in seeds.values()])
    with open(out / 'mechanism_clean_contrasts.md', 'w', encoding='utf-8') as file:
        file.write('| Factor | On vs off | Metric | Improvement | n | t p (Holm) | Wilcoxon p (Holm) |\n'
                   '| --- | --- | --- | ---: | ---: | ---: | ---: |\n')
        for r in clean_pairs:
            file.write(f"| {r['factor']} | {r['on']} vs {r['off']} | {r['endpoint']} | "
                       f"{r['improvement']:.6f} | {r['n']} | {r.get('paired_t_p_holm', '')} | "
                       f"{r.get('wilcoxon_p_holm', '')} |\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', default=['ushcn', 'physionet'])
    parser.add_argument('--seeds', default='2021 2022 2023 2024 2025')
    parser.add_argument('--protocol', choices=['frozen', 'joint'], default='frozen')
    parser.add_argument('--gate-max', type=float, default=0.3)
    parser.add_argument('--initial-gate', type=float, default=0.1)
    parser.add_argument('--local-encoder', choices=['summary', 'attention'], default='summary')
    parser.add_argument('--local-k', type=int, default=12)
    parser.add_argument('--utility-weight', type=float, default=0.1)
    parser.add_argument('--oracle-weight', type=float, default=0.1)
    parser.add_argument('--holdout', type=float, default=0.2)
    parser.add_argument('--split-seed', type=int, default=1729)
    parser.add_argument('--branch-epochs', type=int)
    parser.add_argument('--gate-epochs', type=int)
    parser.add_argument('--degradation-modes', default='drop,recent_drop,noise,stale')
    parser.add_argument('--severities', default='0,0.25,0.5,0.75,1')
    parser.add_argument('--out-dir', default='analysis/gate_mechanism')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    seeds = args.seeds.split()
    modes = args.degradation_modes.split(',')
    rates = [float(v) for v in args.severities.split(',')]
    if not seeds or len(set(seeds)) != len(seeds) or any(not s.isdigit() for s in seeds):
        parser.error('Seeds must be unique nonnegative integers.')
    if (not 0 < args.holdout < 1 or len(set(modes)) != len(modes)
            or not set(modes) <= {'drop', 'recent_drop', 'noise', 'stale'}
            or not modes or len(set(rates)) != len(rates) or not {0., 1.} <= set(rates)
            or any(not 0 <= r <= 1 for r in rates)):
        parser.error('Use unique supported modes and severities in [0,1], including 0 and 1; holdout in (0,1).')
    if any(v is not None and v < 1 for v in (args.branch_epochs, args.gate_epochs)):
        parser.error('Epoch budgets must be positive.')
    os.chdir(ROOT)
    if not args.dry_run:
        require_training_provenance_support()
    cells = build_cells(args.gate_max, args.initial_gate, args.local_encoder, args.local_k,
                        args.utility_weight, args.oracle_weight)
    base = dict(os.environ)
    for key in ('SEED', 'SEED_LIST', 'MONITOR_TEST_DURING_TRAINING', 'QUERY_EVIDENCE_OUT_DIR'):
        base.pop(key, None)
    base['LOCAL_OBSERVATION_DROP_ROBUSTNESS'] = '1'
    base['LOCAL_DEGRADATION_MODES'] = ','.join(modes)
    base['LOCAL_OBSERVATION_DROP_RATES'] = ','.join(str(r) for r in rates)
    python = base.get('PYTHON') or sys.executable
    for dataset in args.datasets:
        if not args.dry_run:
            with environment(base):
                benchmark.prepare_dataset_if_needed(benchmark.load_dataset_config(dataset), python)
        source_id, source_pattern = '', ''
        if args.protocol == 'frozen':
            source_id, source_pattern = train_source(dataset, fusion.apply_env(base, cells[0]['env']), seeds, args)
        report_id = signature(dict(source=source_id, args={k: v for k, v in vars(args).items()
                                                         if k not in {'dry_run', 'resume', 'out_dir'}},
                                   common=common_args(dataset, fusion.apply_env(base, cells[0]['env']))))
        out = Path(args.out_dir) / f'{dataset}_{args.protocol}_{report_id}'
        rows, audit = [], []
        for cell in cells:
            env = fusion.apply_env(base, cell['env'])
            if args.protocol == 'frozen':
                env.update(POINT_TRAIN_STAGE='gate', POINT_GATE_HOLDOUT_RATIO=str(args.holdout),
                           POINT_GATE_SPLIT_SEED=str(args.split_seed), POINT_GATE_INIT_CHECKPOINT=source_pattern)
            if args.gate_epochs is not None:
                env['TRAIN_EPOCHS'] = str(args.gate_epochs)
            common = common_args(dataset, env)
            prefix = f"gm_{dataset}_{args.protocol}_{cell['key']}_{signature(dict(common=common, source=source_id))}"
            for seed in seeds:
                command, _ = benchmark.build_seed_command(python, common, seed, prefix)
                if args.dry_run:
                    print('[gate] ' + shlex.join(command), flush=True)
                    continue
                log_dir = str(ROOT / 'logs' / 'gate_mechanism' / prefix)
                if args.protocol == 'frozen':
                    status, _, seed_audit = run_frozen_gate_seed(
                        python, common, seed, prefix, env, log_dir,
                        unique_checkpoint(source_id, seed), resume=args.resume)
                else:
                    status, _ = benchmark.run_seed(python, common, seed, prefix, env,
                                                  log_dir, resume=args.resume)
                if status:
                    raise RuntimeError(f"Gate experiment failed: {cell['key']}, seed={seed}, exit={status}")
                checkpoint = unique_checkpoint(prefix, seed)
                validation = json.loads((checkpoint.parent / 'validation_selection.json').read_text(encoding='utf-8'))
                if validation['test_evaluated_during_training']:
                    raise RuntimeError('Test data used during training; mechanism run rejected.')
                if args.protocol == 'frozen':
                    audit.append(dict(strategy_key=cell['key'], seed=seed, **seed_audit))
            if not args.dry_run:
                collected, missing = benchmark.find_seed_metrics(prefix, seeds)
                if missing:
                    raise RuntimeError(f'Missing metrics: {missing}')
                row = dict(strategy_key=cell['key'], fusion_strategy=cell['key'], run_des_prefix=prefix,
                           collected=collected, **fusion.metric_summary(collected))
                rows.append(row)
        if args.dry_run:
            continue
        out.mkdir(parents=True, exist_ok=True)
        config = dict(protocol=args.protocol, factors=FACTORS, source=source_id, seeds=seeds,
                      arguments=vars(args), cells=[{k: v for k, v in c.items() if k != 'env'} for c in cells],
                      clean_holm_family='all 12 conditional contrasts x MSE/MAE',
                      robustness_holm_family='all conditional contrasts x conditions x four error endpoints',
                      test_used_for_selection=False)
        (out / 'experiment_protocol.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
        write_csv(out / 'frozen_predictor_audit.csv', audit)
        reports(cells, rows, out, modes, rates)
        print(f'[mechanism report] {out.resolve()}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
