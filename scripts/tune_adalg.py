#!/usr/bin/env python
"""Select AdaLG settings using validation only, then export a reproducible config."""
import argparse
import csv
import glob
import hashlib
import itertools
import json
import math
import os
import re
import shlex
import subprocess
import sys

import benchmark


def export_environment(common_args, overrides):
    """Export only recognized experiment settings, never arbitrary environment data."""
    with open(os.path.join(benchmark.MODEL_CONFIG_DIR, 'adalg_config.py'), encoding='utf-8') as file_obj:
        model_source = file_obj.read()
    with open(benchmark.__file__, encoding='utf-8') as file_obj:
        protocol_source = file_obj.read()
    allowed = set(re.findall(r"_env_(?:value|flag)\(env,\s*'([A-Z_]+)'", model_source))
    allowed.update(re.findall(r"env_value\(\s*'([A-Z_]+)'", protocol_source))
    exported = dict(overrides)
    index = 0
    while index < len(common_args):
        argument = common_args[index]
        if argument.startswith('--'):
            name, separator, value = argument[2:].partition('=')
            key = name.upper()
            if not separator:
                if index + 1 < len(common_args) and not common_args[index + 1].startswith('--'):
                    index += 1
                    value = common_args[index]
                else:
                    value = '1'
            if key in allowed:
                exported[key] = value
        index += 1
    return exported


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', nargs='?', default='ushcn')
    parser.add_argument('--encoders', nargs='+', choices=['summary', 'attention'], default=['summary', 'attention'])
    parser.add_argument('--local-k', nargs='+', type=int, default=[4, 8, 12])
    parser.add_argument('--gates', nargs='+', default=['ordinary:1', 'utility:0.3'])
    parser.add_argument('--seeds', default='2021')
    parser.add_argument('--out-dir', default='analysis/adalg_validation_search')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    os.chdir(benchmark.REPO_ROOT)
    dataset = benchmark.load_dataset_config(args.dataset)
    split_args, _, _ = benchmark.build_split_args(dataset)
    python = benchmark.resolve_python_executable()
    out_dir = os.path.join(args.out_dir, dataset['key'])
    os.makedirs(out_dir, exist_ok=True)
    seeds = args.seeds.split()
    if not seeds or any(k <= 0 for k in args.local_k):
        parser.error('Provide at least one seed and positive local-k values.')
    rows = []
    configs = []
    if not args.dry_run:
        benchmark.prepare_dataset_if_needed(dataset, python)
    for candidate_id, (encoder, k, gate_spec) in enumerate(itertools.product(args.encoders, args.local_k, args.gates)):
        mode, maximum = gate_spec.split(':')
        if mode not in {'ordinary', 'utility'} or not 0.0 < float(maximum) <= 1.0:
            parser.error('Gate specifications must be ordinary:<max> or utility:<max>, max in (0,1].')
        overrides = {
            'POINT_LOCAL_BRANCH_SOURCE': 'raw', 'POINT_LOCAL_ENCODER': encoder, 'POINT_LOCAL_FUSION_K': str(k),
            'POINT_GATE_MODE': mode, 'POINT_GATE_MAX': maximum, 'POINT_FORCE_FUSION': 'auto',
            'POINT_LOCAL_FUSION_MODE': 'dual_gate', 'POINT_GATE_PREDICTION_FEATURES': '1',
            'POINT_SEPARATE_UTILITY_HEAD': '0', 'POINT_DISABLE_CAUSAL_GATE_CONTEXT': '1',
            'POINT_DISABLE_LOCAL_BRANCH': '0', 'CAUSAL_LOCAL_EXPAND_POOL': '0', 'CAUSAL_LOCAL_ALPHA': '0.0',
            'POINT_DISABLE_GATE_AVAILABILITY_FEATURES': '0',
            'POINT_TRAIN_STAGE': 'full', 'POINT_GATE_INIT_CHECKPOINT': '',
            'GLOBAL_AUX_LOSS_WEIGHT': '0.25', 'LOCAL_AUX_LOSS_WEIGHT': '0.25',
            'UTILITY_LOSS_WEIGHT': '0.1' if mode == 'utility' else '0',
            'ORACLE_GATE_LOSS_WEIGHT': '0.1' if mode == 'utility' else '0',
        }
        env = dict(os.environ, **overrides)
        model = benchmark.load_model_config('AdaLG', dataset, env=env)
        common = benchmark.build_common_args(dataset, model, split_args)
        scores, checkpoint_paths = [], []
        for seed in seeds:
            signature = hashlib.sha1(json.dumps([common, overrides, seed], sort_keys=True).encode()).hexdigest()[:10]
            des = f'valtune_{dataset["key"]}_c{candidate_id}_s{seed}_{signature}'
            command = [python, '-u', 'run.py', *common, '--seed', seed, '--des', des, '--skip_test']
            log_path = os.path.join(out_dir, f'{des}.log')
            manifests = glob.glob(os.path.join('checkpoints', f'*{des}_h*', 'validation_selection.json'))
            if args.dry_run:
                print(subprocess.list2cmdline(command), flush=True)
                continue
            if not (args.resume and len(manifests) == 1):
                print(f'[validation_search] candidate={candidate_id} encoder={encoder} k={k} gate={gate_spec} seed={seed}', flush=True)
                with open(log_path, 'w', encoding='utf-8') as log_file:
                    status = subprocess.call(command, env=env, stdout=log_file, stderr=subprocess.STDOUT)
                if status:
                    raise RuntimeError(f'Training failed; see {log_path}')
                manifests = glob.glob(os.path.join('checkpoints', f'*{des}_h*', 'validation_selection.json'))
            if len(manifests) != 1:
                raise RuntimeError(f'Expected one validation manifest for {des}, got {len(manifests)}.')
            with open(manifests[0], encoding='utf-8') as file_obj:
                result = json.load(file_obj)
            if result['test_evaluated_during_training']:
                raise RuntimeError('Validation search must not evaluate test during training.')
            score = float(result['best_val_metric'])
            if not math.isfinite(score):
                raise RuntimeError('Validation score is not finite.')
            scores.append(score)
            checkpoint_paths.append(result['checkpoint_path'])
            print(f'[validation_search] candidate={candidate_id} seed={seed} val={score:.8f}', flush=True)
        if args.dry_run:
            continue
        mean, std = benchmark.mean_std(scores)
        rows.append({'candidate': candidate_id, 'encoder': encoder, 'k': k, 'gate_mode': mode, 'gate_max': maximum,
                     'n_seeds': len(scores), 'validation_mean': mean, 'validation_std': std})
        configs.append({'candidate': candidate_id, 'environment': export_environment(common, overrides), 'validation_scores': scores,
                        'checkpoint_paths': checkpoint_paths, 'training_args': common})
    if args.dry_run:
        return 0
    best = min(rows, key=lambda row: row['validation_mean'])
    with open(os.path.join(out_dir, 'validation_search.csv'), 'w', newline='', encoding='utf-8') as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    selected = configs[best['candidate']]
    selected.update(dataset=dataset['key'], seeds=seeds, selection=best, protocol='validation_only')
    with open(os.path.join(out_dir, 'selected_config.json'), 'w', encoding='utf-8') as file_obj:
        json.dump(selected, file_obj, indent=2)
    with open(os.path.join(out_dir, 'selected_config.ps1'), 'w', encoding='utf-8') as file_obj:
        for key, value in selected['environment'].items():
            quoted_value = str(value).replace("'", "''")
            file_obj.write(f"$env:{key} = '{quoted_value}'\n")
    with open(os.path.join(out_dir, 'selected_config.sh'), 'w', encoding='utf-8', newline='\n') as file_obj:
        file_obj.write('# Source this file to apply the validation-selected experiment settings.\n')
        for key, value in selected['environment'].items():
            file_obj.write(f'export {key}={shlex.quote(str(value))}\n')
    print(f'Selected on validation: {best}; config: {out_dir}/selected_config.json', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
