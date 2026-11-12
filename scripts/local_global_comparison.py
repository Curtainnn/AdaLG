#!/usr/bin/env python
"""Run the shared-head rebuttal controls on two or more datasets."""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', default=['ushcn', 'physionet'])
    parser.add_argument('--strategies', nargs='+', default=['shared_ordinary', 'raw_ordinary', 'raw_attention_ordinary', 'raw_attention_utility'])
    parser.add_argument('--seeds', default='2021 2022 2023 2024 2025')
    parser.add_argument('--degradation-modes', default='drop,recent_drop,noise,stale')
    parser.add_argument('--severities', default='0,0.25,0.5,0.75,1')
    parser.add_argument('--bounded-gate-max', default='0.3')
    parser.add_argument('--des-prefix', default='local_global_comparison')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true', help='reuse trained checkpoints; finish test analyses and train missing seeds')
    args = parser.parse_args()
    env = dict(os.environ, LOCAL_DEGRADATION_MODES=args.degradation_modes)
    env.pop('SEED', None)
    env.pop('QUERY_EVIDENCE_OUT_DIR', None)  # Each run must have its own diagnostics.
    python = os.environ.get('PYTHON', sys.executable)
    for dataset in args.datasets:
        command = [python, 'scripts/fusion_ablation.py', 'AdaLG', dataset,
                   '--strategies', *args.strategies, '--seed-list', args.seeds,
                   '--query-evidence', 'all', '--local-drop-robustness', '--local-drop-rates', args.severities,
                   '--bounded-gate-max', args.bounded_gate_max, '--des-prefix', f'{args.des_prefix}_{dataset}']
        if args.dry_run:
            command.append('--dry-run')
        if args.resume:
            command.append('--resume')
        status = subprocess.call(command, cwd=ROOT, env=env)
        if status:
            return status
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
