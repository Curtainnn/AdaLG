#!/usr/bin/env python
"""Train two Local paths against one common frozen Global per seed."""
import argparse
import importlib
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
import gate_mechanism_ablation as mechanism


def require_support():
    mechanism.require_training_provenance_support()
    try:
        importlib.import_module('utils.frozen_global')
    except ModuleNotFoundError as error:
        if error.name != 'utils.frozen_global':
            raise
        raise RuntimeError('Missing utils/frozen_global.py. Sync the complete common-Global update package.') from error
    from exp.exp_forecast import Exp_Forecast
    if getattr(Exp_Forecast, 'frozen_global_local_version', 0) != 1:
        raise RuntimeError('Outdated exp/exp_forecast.py. Sync all files from the same common-Global update package.')
    probe = dict(os.environ, POINT_FROZEN_GLOBAL_LOCAL_TRAINING='1', POINT_TRAIN_STAGE='gate',
                 POINT_GATE_HOLDOUT_RATIO='0.2', POINT_GATE_INIT_CHECKPOINT='probe')
    with mechanism.environment(probe):
        model = benchmark.load_model_config('AdaLG', benchmark.load_dataset_config('human_activity'), env=probe)
    if '--point_frozen_global_local_training' not in model['args']:
        raise RuntimeError('Outdated configs/models/adalg_config.py. Sync the complete update package.')


def validate_source(path):
    protocol = mechanism.completed_source(path)
    selection = mechanism.read_checkpoint_record(path.parent / 'validation_selection.json')
    if selection.get('global_only_source') is not True:
        raise RuntimeError(f'Expected a Global-only pretraining checkpoint: {path}')
    return protocol


def audit_local_checkpoint(path, source):
    from utils.frozen_global import global_state_hash
    import torch
    protocol = validate_source(source)
    selection = mechanism.read_checkpoint_record(path.parent / 'validation_selection.json')
    if (selection.get('frozen_global_local_training') is not True
            or selection.get('test_evaluated_during_training') is not False
            or int(selection.get('best_epoch', 0)) < 1
            or not math.isfinite(float(selection.get('best_val_metric', float('nan'))))):
        raise mechanism.InvalidGateCheckpointError(f'Incomplete/invalid common-Global training: {path}')
    source_selection = mechanism.read_checkpoint_record(source.parent / 'validation_selection.json')
    if selection.get('seed') != source_selection.get('seed'):
        raise mechanism.InvalidGateCheckpointError('Local/source seeds differ.')
    manifest_path = path.parent / 'gate_holdout_manifest.json'
    manifest = (mechanism.read_checkpoint_record(manifest_path) if manifest_path.exists()
                else selection.get('gate_holdout_manifest'))
    if (not isinstance(manifest, dict) or manifest.get('stage') != 'gate'
            or manifest.get('protocol') != protocol or selection.get('gate_holdout_manifest') != manifest):
        raise mechanism.InvalidGateCheckpointError(f'Local/source split or scaler differs: {path}')
    reference = global_state_hash(torch.load(source, map_location='cpu', weights_only=True))
    result = global_state_hash(torch.load(path, map_location='cpu', weights_only=True))
    if reference != result:
        raise mechanism.InvalidGateCheckpointError(f'Frozen Global tensors changed: {path}')
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return dict(source=str(source), checkpoint=str(path), global_sha256=result[0], global_elements=result[1],
                split_scaler_sha256=mechanism.signature(protocol), verified=True)


def compare_global_predictions(path_a, path_b, rtol=1e-5, atol=1e-6):
    import numpy as np
    with np.load(path_a) as a, np.load(path_b) as b:
        if set(a.files) != set(b.files) or 'global_pred' not in a.files or 'true' not in a.files:
            raise RuntimeError('Missing/mismatched exported Global audit columns.')
        for key in a.files:
            if key != 'global_pred' and not np.array_equal(a[key], b[key]):
                raise RuntimeError(f'Local paths were evaluated on different queries/labels: {key}')
        if (a['global_pred'].shape != b['global_pred'].shape or not a['global_pred'].size
                or not np.isfinite(a['global_pred']).all() or not np.isfinite(b['global_pred']).all()
                or not np.allclose(a['global_pred'], b['global_pred'], rtol=rtol, atol=atol)):
            raise RuntimeError('Global predictions differ across Local paths despite a common source.')
        return dict(n_queries=int(a['global_pred'].size),
                    max_global_prediction_change=float(np.max(np.abs(a['global_pred'] - b['global_pred']))),
                    global_rtol=rtol, global_atol=atol, verified=True)


def train_source(dataset, env, args, seeds):
    updates = dict(POINT_TRAIN_STAGE='branch', POINT_GATE_HOLDOUT_RATIO=str(args.holdout),
                   POINT_GATE_SPLIT_SEED=str(args.split_seed), POINT_GATE_INIT_CHECKPOINT=None,
                   POINT_FROZEN_GLOBAL_LOCAL_TRAINING='0', POINT_GATE_INITIAL_WEIGHT=None,
                   POINT_FORCE_FUSION='global', POINT_DISABLE_LOCAL_BRANCH='1',
                   GLOBAL_AUX_LOSS_WEIGHT='0', LOCAL_AUX_LOSS_WEIGHT='0',
                   LOCAL_OBSERVATION_DROP_ROBUSTNESS='0', QUERY_EVIDENCE_ANALYSIS='0')
    env = fusion.apply_env(env, updates)
    if args.global_epochs is not None:
        env['TRAIN_EPOCHS'] = str(args.global_epochs)
    common = mechanism.common_args(dataset, env) + ['--skip_test']
    prefix = f'fg_{dataset}_global_{mechanism.signature(common)}'
    python = env.get('PYTHON') or sys.executable
    for seed in seeds:
        command, _ = benchmark.build_seed_command(python, common, seed, prefix)
        if args.dry_run:
            print('[Global-only/no test] ' + shlex.join(command), flush=True)
            continue
        paths = list((ROOT / 'checkpoints').glob(f'*{prefix}_seed{seed}_h*_0/checkpoint.pth'))
        if args.resume and paths:
            source = mechanism.unique_checkpoint(prefix, seed)
            validate_source(source)
            print(f'[resume common Global] {source}', flush=True)
        else:
            status, _ = benchmark.run_seed(python, common, seed, prefix, env,
                                          str(ROOT / 'logs/frozen_global' / prefix), resume=False)
            if status:
                raise RuntimeError(f'Global-only training failed: seed={seed}, exit={status}')
            validate_source(mechanism.unique_checkpoint(prefix, seed))
    return prefix, f'checkpoints/*{prefix}_seed{{seed}}_h*_0/checkpoint.pth'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', default=['human_activity'])
    parser.add_argument('--seeds', default='2021 2022 2023 2024 2025')
    parser.add_argument('--holdout', type=float, default=0.2)
    parser.add_argument('--split-seed', type=int, default=1729)
    parser.add_argument('--local-encoder', choices=['summary', 'attention'], default='summary')
    parser.add_argument('--local-k', type=int, default=12)
    parser.add_argument('--global-epochs', type=int)
    parser.add_argument('--local-epochs', type=int)
    parser.add_argument('--degradation-modes', default='drop,recent_drop,noise,stale')
    parser.add_argument('--severities', default='0,0.25,0.5,0.75,1')
    parser.add_argument('--out-dir', default='analysis/frozen_global')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    seeds = args.seeds.split()
    modes = args.degradation_modes.split(',')
    rates = [float(v) for v in args.severities.split(',')]
    if (not seeds or len(set(seeds)) != len(seeds) or any(not s.isdigit() for s in seeds)
            or not 0 < args.holdout < 1 or args.local_k < 1
            or any(e is not None and e < 1 for e in (args.global_epochs, args.local_epochs))
            or not modes or len(set(modes)) != len(modes)
            or not set(modes) <= {'drop', 'recent_drop', 'noise', 'stale'}
            or len(set(rates)) != len(rates) or not {0., 1.} <= set(rates)
            or any(not 0 <= r <= 1 for r in rates)):
        parser.error('Invalid seeds, holdout, epoch budgets, modes or severities (include 0 and 1).')
    os.chdir(ROOT)
    if not args.dry_run:
        require_support()
    base = dict(os.environ)
    for key in ('SEED', 'SEED_LIST', 'QUERY_EVIDENCE_OUT_DIR', 'MONITOR_TEST_DURING_TRAINING'):
        base.pop(key, None)
    # Reuse the ordinary-gate controls, fixing all three mechanism factors off.
    base = fusion.apply_env(base, mechanism.build_cells(encoder=args.local_encoder, k=args.local_k)[0]['env'])
    base.update(POINT_FROZEN_GLOBAL_LOCAL_TRAINING='0',
                GLOBAL_AUX_LOSS_WEIGHT='0', LOCAL_AUX_LOSS_WEIGHT='0.25',
                LOCAL_OBSERVATION_DROP_ROBUSTNESS='1', LOCAL_DEGRADATION_MODES=','.join(modes),
                LOCAL_OBSERVATION_DROP_RATES=','.join(str(r) for r in rates))
    python = base.get('PYTHON') or sys.executable
    for dataset in args.datasets:
        if not args.dry_run:
            with mechanism.environment(base):
                benchmark.prepare_dataset_if_needed(benchmark.load_dataset_config(dataset), python)
        source_id, pattern = train_source(dataset, base, args, seeds)
        rows, audits = [], []
        for key, source, label in (
            ('shared_ordinary', 'shared_encoder', 'Common frozen Global + shared-feature second head'),
            ('raw_ordinary', 'raw', 'Common frozen Global + independent raw Local'),
        ):
            env = dict(base, POINT_LOCAL_BRANCH_SOURCE=source,
                       POINT_LOCAL_ENCODER='summary' if source == 'shared_encoder' else args.local_encoder,
                       POINT_TRAIN_STAGE='gate', POINT_GATE_HOLDOUT_RATIO=str(args.holdout),
                       POINT_GATE_SPLIT_SEED=str(args.split_seed), POINT_GATE_INIT_CHECKPOINT=pattern,
                       POINT_FROZEN_GLOBAL_LOCAL_TRAINING='1')
            if args.local_epochs is not None:
                env['TRAIN_EPOCHS'] = str(args.local_epochs)
            common = mechanism.common_args(dataset, env)
            prefix = f'fg_{dataset}_{key}_{mechanism.signature(dict(common=common, source=source_id))}'
            log_dir = str(ROOT / 'logs/frozen_global' / prefix)
            for seed in seeds:
                command, _ = benchmark.build_seed_command(python, common, seed, prefix)
                if args.dry_run:
                    print('[Local+Gate/frozen Global] ' + shlex.join(command), flush=True)
                    continue
                source_path = mechanism.unique_checkpoint(source_id, seed)
                resume = args.resume
                existing = list((ROOT / 'checkpoints').glob(f'*{prefix}_seed{seed}_h*_0/checkpoint.pth'))
                if resume and existing:
                    checkpoint = mechanism.unique_checkpoint(prefix, seed)
                    try:
                        audit_local_checkpoint(checkpoint, source_path)
                    except mechanism.InvalidGateCheckpointError as error:
                        command.extend(['--test_dir', checkpoint.parent.name])
                        archive = benchmark.archive_incompatible_checkpoint(
                            command, str(Path(log_dir) / f'seed{seed}.log'), reason=str(error))
                        print(f'[resume retrain Local] seed={seed}; previous run preserved at {archive}', flush=True)
                        resume = False
                status, _ = benchmark.run_seed(python, common, seed, prefix, env, log_dir, resume=resume)
                if status:
                    raise RuntimeError(f'Local training/test failed: {key}, seed={seed}, exit={status}')
                audit = audit_local_checkpoint(mechanism.unique_checkpoint(prefix, seed), source_path)
                audits.append(dict(strategy_key=key, seed=seed, **audit))
            if not args.dry_run:
                collected, missing = benchmark.find_seed_metrics(prefix, seeds)
                if missing:
                    raise RuntimeError(f'Missing Local comparison metrics: {missing}')
                rows.append(dict(strategy_key=key, fusion_strategy=label, run_des_prefix=prefix,
                                 collected=collected, **fusion.metric_summary(collected)))
        if args.dry_run:
            continue
        report_id = mechanism.signature(dict(source=source_id, prefixes=[r['run_des_prefix'] for r in rows], seeds=seeds))
        out = Path(args.out_dir) / f'{dataset}_{report_id}'
        out.mkdir(parents=True, exist_ok=True)
        prediction_audits = []
        by_strategy = {r['strategy_key']: {str(s): Path(p).parent for s, p, _ in r['collected']} for r in rows}
        for seed in seeds:
            a = by_strategy['shared_ordinary'][seed] / 'frozen_global_predictions.npz'
            b = by_strategy['raw_ordinary'][seed] / 'frozen_global_predictions.npz'
            prediction_audits.append(dict(seed=seed, **compare_global_predictions(a, b)))
        fusion.write_fusion_table(rows, str(out))
        fusion.write_auxiliary_tables(rows, str(out))
        contrast = []
        metrics = {r['strategy_key']: {str(s): m for s, _, m in r['collected']} for r in rows}
        for endpoint in ('mse', 'mae'):
            contrast.append(mechanism.paired_row(
                'local_path', 'raw_ordinary', 'shared_ordinary', endpoint, 'clean',
                {s: float(m[endpoint]) for s, m in metrics['raw_ordinary'].items()},
                {s: float(m[endpoint]) for s, m in metrics['shared_ordinary'].items()}))
        mechanism.correct(contrast)
        mechanism.write_csv(out / 'common_global_contrasts.csv', contrast)
        mechanism.write_csv(out / 'frozen_global_parameter_audit.csv', audits)
        mechanism.write_csv(out / 'frozen_global_prediction_audit.csv', prediction_audits)
        (out / 'experiment_protocol.json').write_text(json.dumps(dict(
            source=source_id, arguments=vars(args), frozen='all Global tensors',
            trained='Local prediction path plus ordinary Gate', shared_head_init='pretrained Global output head',
            initial_gate=0.1, gate_cap=1, test_used_for_selection=False), indent=2), encoding='utf-8')
        print(f'[common Global report] {out.resolve()}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
