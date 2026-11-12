#!/usr/bin/env python
"""Fresh, objective-matched AdaLG vs raw_recent_concat paired experiments."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import benchmark
import gate_mechanism_ablation as mechanism

VARIANTS = ('adalg', 'raw_recent_concat')
CODE_FILES = ('models/adalg.py', 'layers/adalg_query_point_head.py', 'layers/raw_recent_concat.py',
              'layers/recent_observations.py', 'layers/local_residual_adapter.py',
              'exp/exp_forecast.py', 'data_provider/data_loader.py', 'data_provider/data_factory.py',
              'configs/models/adalg_config.py', 'run.py', 'utils/raw_recent_analysis.py',
              'scripts/raw_recent_concat_comparison.py')
CODE_FILES = sorted(set(CODE_FILES) | {str(p.relative_to(ROOT)).replace('\\', '/')
                    for directory in ('layers', 'models', 'data_provider', 'exp', 'utils', 'configs')
                    for p in (ROOT / directory).rglob('*.py')})


def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as file:
        for block in iter(lambda: file.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def command_for(dataset, variant, settings, study, seed):
    env = dict(os.environ)
    # Do not inherit experiment overrides from an interactive shell.
    for key in list(env):
        if key.startswith(('POINT_', 'CAUSAL_', 'QUERY_EVIDENCE_', 'LOCAL_', 'INPUT_')):
            del env[key]
    for key in ('SEED', 'SEED_LIST', 'MONITOR_TEST_DURING_TRAINING', 'SUBSET_RAND_RATIO',
                'TRAIN_EPOCHS', 'LOSS_TYPE', 'CHECKPOINT_METRIC', 'BATCH_SIZE', 'PATIENCE', 'NUM_WORKERS'):
        env.pop(key, None)
    env.update(settings, POINT_PREDICTION_VARIANT=variant)
    env.setdefault('OMP_NUM_THREADS', '4')
    env.setdefault('MKL_NUM_THREADS', '4')
    env['PYTHONIOENCODING'] = 'utf-8'
    with mechanism.environment(env):
        config = benchmark.load_dataset_config(dataset)
        model = benchmark.load_model_config('AdaLG', config, env=env)
        split, _, _ = benchmark.build_split_args(config)
        common = benchmark.build_common_args(config, model, split)
    # The dedicated export records every valid query; legacy gate diagnostics do not apply to concat.
    common = [x for x in common if x != '--query_evidence_analysis']
    common += ['--checkpoints', str(study / dataset / 'checkpoints'),
               '--results_root', str(study / dataset / 'results')]
    prefix = f'rrc_{dataset}_{variant}_{study.name}'
    return env, common, prefix, config


def summarize(study):
    import numpy as np
    from scipy.stats import ttest_rel, wilcoxon
    protocol = read_json(study / 'experiment_config.json')
    rows, contrasts, groups, audits = [], [], [], []
    for dataset in protocol['datasets']:
        paired = []
        for seed in protocol['seeds']:
            exports = {}
            for variant in VARIANTS:
                status_path = study / dataset / variant / f'seed{seed}' / 'run_status.json'
                if not status_path.exists():
                    continue
                status = read_json(status_path)
                if status.get('status') != 'complete':
                    continue
                folder = Path(status['result_directory'])
                manifest = read_json(folder / 'raw_recent_export_manifest.json')
                selection = read_json(Path(status['checkpoint']).parent / 'validation_selection.json')
                if selection['test_evaluated_during_training']:
                    raise ValueError('Test-based checkpoint selection rejected.')
                with np.load(folder / 'raw_recent_queries.npz', allow_pickle=False) as data:
                    q = {key: data[key] for key in data.files}
                if not np.isfinite(q['prediction']).all() or not np.isfinite(q['true']).all():
                    raise ValueError('Nonfinite query metrics.')
                error = q['prediction'].astype(np.float64) - q['true']
                info = manifest['training']['model']
                row = dict(dataset=dataset, seed=seed, variant=variant, n_queries=len(error),
                           mse=float(np.mean(error**2)), mae=float(np.mean(np.abs(error))),
                           total_parameters=info['total_parameters'], query_head_parameters=info['query_head_parameters'],
                           best_epoch=selection['best_epoch'], best_validation_metric=selection['best_val_metric'],
                           checkpoint_metric=selection['checkpoint_metric'], checkpoint=status['checkpoint'])
                rows.append(row)
                exports[variant] = (q, manifest, row)
            if set(exports) != set(VARIANTS):
                continue
            a, am, ar = exports['adalg']
            b, bm, br = exports['raw_recent_concat']
            for key in ('sample_index', 'query_row', 'variable', 'query_time', 'true', 'true_original',
                        'recent_count', 'recent_gap', 'local_dynamics'):
                if not np.array_equal(a[key], b[key]):
                    raise ValueError(f'Paired query/data condition mismatch: {dataset}, seed={seed}, {key}')
            for split in ('train', 'val'):
                if am['training']['data'][split] != bm['training']['data'][split]:
                    raise ValueError(f'Training/validation split or normalization mismatch: {split}')
            if am['test'] != bm['test']:
                raise ValueError('Test split/scaler mismatch.')
            memberships=[set(am['training']['data']['train']['group_hashes']),
                         set(am['training']['data']['val']['group_hashes']),set(am['test']['group_hashes'])]
            if any(memberships[i] & memberships[j] for i,j in ((0,1),(0,2),(1,2))):
                raise ValueError('Record groups overlap between train, validation, or test.')
            if am['training']['model']['encoder_initial_sha256'] != bm['training']['model']['encoder_initial_sha256']:
                raise ValueError('Global Encoder initialization differs between variants.')
            # Compare resolved runtime options, excluding only model/output identity.
            ignored = {'point_prediction_variant', 'des', 'test_dir', 'is_training', 'raw_concat_hidden'}
            aa, ba = am['training']['args'], bm['training']['args']
            if {k:v for k,v in aa.items() if k not in ignored} != {k:v for k,v in ba.items() if k not in ignored}:
                raise ValueError('Resolved paired training configurations differ.')
            audits.append(dict(dataset=dataset, seed=seed, queries=len(a['true']), verified=True,
                               encoder_initial_sha256=am['training']['model']['encoder_initial_sha256']))
            joined = {key:a[key] for key in ('sample_index', 'query_row', 'variable', 'query_time', 'true',
                       'true_original', 'recent_count', 'recent_gap', 'local_dynamics')}
            joined.update(global_prediction=a['global_pred'], global_prediction_original=a['global_pred_original'],
                          raw_concat_prediction=b['prediction'], raw_concat_prediction_original=b['prediction_original'],
                          adalg_prediction=a['prediction'], adalg_prediction_original=a['prediction_original'],
                          gate_weight=a['gate'])
            # Global prediction is the AdaLG branch, not a separately trained global-only model.
            np.savez_compressed(study / dataset / f'paired_queries_seed{seed}.npz', **joined)
            paired.append((ar, br))
            count = a['recent_count']
            selectors = [('all', np.ones(len(count), dtype=bool)), ('no_local', count == 0),
                         ('count_1_3', (count >= 1) & (count <= 3)),
                         ('count_4_7', (count >= 4) & (count <= 7)), ('count_8_12', count >= 8)]
            for feature in ('recent_gap', 'local_dynamics'):
                eligible = count > 0
                if eligible.any():
                    edges = np.quantile(a[feature][eligible], [1/3, 2/3])
                    for index, name in enumerate(('low', 'middle', 'high')):
                        mask = eligible & (np.searchsorted(edges, a[feature], side='right') == index)
                        selectors.append((f'{feature}_{name}', mask))
            for group, selected in selectors:
                if not selected.any():
                    continue
                ae = a['prediction'][selected].astype(np.float64) - a['true'][selected]
                be = b['prediction'][selected].astype(np.float64) - b['true'][selected]
                groups.append(dict(dataset=dataset, seed=seed, group=group, n_queries=int(selected.sum()),
                                   adalg_mse=float(np.mean(ae**2)), raw_concat_mse=float(np.mean(be**2)),
                                   adalg_mae=float(np.mean(abs(ae))), raw_concat_mae=float(np.mean(abs(be))),
                                   mean_count=float(np.mean(count[selected])),
                                   mean_gap=float(np.mean(a['recent_gap'][selected])),
                                   mean_dynamics=float(np.mean(a['local_dynamics'][selected]))))
        for endpoint in ('mse', 'mae'):
            if not paired:
                continue
            av, bv = (np.array([pair[index][endpoint] for pair in paired]) for index in (0, 1))
            delta = av - bv
            n = len(av)
            pt = float(ttest_rel(av, bv).pvalue) if n > 1 and np.std(delta) > 0 else (1. if n > 1 and np.all(delta == 0) else '')
            pw = float(wilcoxon(av, bv, method='exact').pvalue) if n > 1 and np.any(delta) else (1. if n > 1 else '')
            contrasts.append(dict(dataset=dataset, endpoint=endpoint, n=n,
                                  adalg_mean=float(av.mean()), adalg_std=float(av.std(ddof=1)) if n > 1 else float('nan'),
                                  raw_concat_mean=float(bv.mean()), raw_concat_std=float(bv.std(ddof=1)) if n > 1 else float('nan'),
                                  diff_adalg_minus_concat=float(delta.mean()),
                                  diff_std=float(delta.std(ddof=1)) if n > 1 else float('nan'),
                                  adalg_improvement_pct=float(100*(bv.mean()-av.mean())/bv.mean()) if bv.mean() else 0.,
                                  adalg_better_seeds=int((av < bv).sum()), paired_t_p=pt, wilcoxon_p=pw,
                                  paired_t_p_holm='', wilcoxon_p_holm=''))
    # One declared family across requested datasets and both primary metrics.
    for key in ('paired_t_p', 'wilcoxon_p'):
        available = sorted([i for i,r in enumerate(contrasts) if isinstance(r[key], float) and math.isfinite(r[key])],
                           key=lambda i: contrasts[i][key])
        running = 0.
        for rank, index in enumerate(available):
            running = max(running, min(1., contrasts[index][key] * (len(available)-rank)))
            contrasts[index][key+'_holm'] = running
    write_csv(study / 'per_seed_results.csv', rows)
    write_csv(study / 'paired_statistics.csv', contrasts)
    write_csv(study / 'query_groups_per_seed.csv', groups)
    write_csv(study / 'pair_audit.csv', audits)
    summary = []
    for dataset, group in sorted({(r['dataset'],r['group']) for r in groups}):
        selected = [r for r in groups if r['dataset']==dataset and r['group']==group]
        summary.append(dict(dataset=dataset, group=group, n_seeds=len(selected),
                            n_queries_sum=sum(r['n_queries'] for r in selected),
                            **{key+'_mean':float(np.mean([r[key] for r in selected])) for key in
                               ('adalg_mse','raw_concat_mse','adalg_mae','raw_concat_mae')}))
    write_csv(study / 'query_groups_summary.csv', summary)
    expected = len(protocol['seeds'])*len(protocol['datasets'])
    complete = len(audits)==expected
    lines = ['# Raw Recent Concat vs AdaLG', '',
             f"Status: {'complete paired study' if complete else 'incomplete; only completed pairs below'}. "
             f"Verified pairs: {len(audits)}/{expected}. {'SMOKE ONLY; not paper evidence.' if protocol['smoke'] else ''}", '',
             '## Method and controlled settings', '',
             'Both models keep the same Global Encoder architecture and initial tensors. Recent-K=12 uses only '
             'same-variable observations strictly before each query. Concat encodes the flattened value/gap/mask '
             'triples by a two-layer MLP, then predicts once from [Global context, query embedding, Recent MLP, '
             'local conditions]. It contains no Local prediction head and no Adaptive Gate. No-local Recent features are zeroed.', '',
             'The primary protocol uses the same masked MSE (USHCN/Human Activity) or MAE (P12/P19) on the final prediction '
             'for BOTH models; all branch/utility/oracle auxiliary weights are zero. This tests architecture under matched '
             'objectives and is not a reproduction of the historical full auxiliary recipe. The extra causal-context '
             'gate input is disabled so AdaLG does not receive Global evidence absent from concat. Encoder weights are trained '
             'end to end; equal initialization does not mean equal final encoders.', '',
             'Settings and commands are in experiment_config.json and per-run executed_command.json. Training/validation '
             'membership and scaler hashes are in checkpoint raw_recent_training.json; test fingerprints are recorded '
             'only after training. No old results or checkpoints enter this study.', '',
             '## Parameters and per-seed runs', '',
             'All counted parameters are real modules; matching adjusts the concat prediction-head width without dummy '
             'padding. The unmodified Global frontend retains its context projection; the primary profile disables its '
             'gate input for both models, so this common unused overhead remains counted in both totals. '
             'See per_seed_results.csv and per-run model metadata.', '',
             '| Dataset | AdaLG parameters | Concat parameters | Difference (concat−AdaLG) |',
             '|---|---:|---:|---:|']
    for dataset in protocol['datasets']:
        ar=[r for r in rows if r['dataset']==dataset and r['variant']=='adalg']
        br=[r for r in rows if r['dataset']==dataset and r['variant']=='raw_recent_concat']
        if ar and br:
            ap,bp=ar[0]['total_parameters'],br[0]['total_parameters']
            lines.append(f'| {dataset} | {ap} | {bp} | {bp-ap:+d} ({100*(bp-ap)/ap:+.3f}%) |')
    lines += ['', '### Resolved training settings', '',
              '| Dataset | Loss / checkpoint | Batch | Epoch cap | Split | Gate max (AdaLG) |',
              '|---|---|---:|---:|---|---:|']
    for dataset in protocol['datasets']:
        setting=dict(protocol['profile']['common'],**protocol['profile']['datasets'][dataset])
        lines.append(f"| {dataset} | {setting['LOSS_TYPE']} / val_{setting['CHECKPOINT_METRIC']} | "
                     f"{protocol['batch_size'] or setting['BATCH_SIZE']} | "
                     f"{1 if protocol['smoke'] else protocol['epochs'] or setting['TRAIN_EPOCHS']} | record | {setting['POINT_GATE_MAX']} |")
    lines += ['', '## Overall performance and paired tests', '',
              'Mean ± sample SD across completed seeds. Positive improvement favors AdaLG. Five-seed exact two-sided '
              'Wilcoxon p-values cannot be below 0.0625 when every nonzero difference has the same sign; n=1 SD is undefined (nan). '
              'Holm correction covers requested dataset×metric contrasts; incomplete families are provisional.', '',
              '| Dataset | Metric | n | AdaLG | Raw concat | AdaLG improvement | Wins | t p (Holm) | Wilcoxon p (Holm) |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in contrasts:
        lines.append(f"| {r['dataset']} | {r['endpoint']} | {r['n']} | {r['adalg_mean']:.6f} ± {r['adalg_std']:.6f} | "
                     f"{r['raw_concat_mean']:.6f} ± {r['raw_concat_std']:.6f} | {r['adalg_improvement_pct']:+.3f}% | "
                     f"{r['adalg_better_seeds']}/{r['n']} | {r['paired_t_p_holm']} | {r['wilcoxon_p_holm']} |")
    lines += ['', '## Query-level analysis', '',
              'See query_groups_per_seed.csv and query_groups_summary.csv for count bins 0/1–3/4–7/8–12 and '
              'label-free feature-tertiles of nearest gap and local dynamics among queries with Local. Tertiles are '
              'descriptive test-covariate groups, not tuned using test errors. Do not treat repeated seeds or individual '
              'queries as independent dataset replicates. Dynamics is Recent-K population SD in the model-input scale.', '',
              'Every paired_queries_seed*.npz contains stable query IDs, original and normalized labels, the AdaLG Global '
              'branch prediction, concat and AdaLG predictions, AdaLG Gate, and common observation conditions. The Global '
              'column is a co-trained branch, not an independently supervised global-only model. No rows are truncated.', '',
              '| Dataset | Query group | Seeds | AdaLG MSE | Concat MSE | AdaLG MAE | Concat MAE |',
              '|---|---|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['dataset']} | {r['group']} | {r['n_seeds']} | {r['adalg_mse_mean']:.6f} | "
                     f"{r['raw_concat_mse_mean']:.6f} | {r['adalg_mae_mean']:.6f} | {r['raw_concat_mae_mean']:.6f} |")
    lines += ['',
              '## Interpretation and limits', '',
              'If AdaLG consistently outperforms this concat path, that supports a benefit beyond simply injecting raw '
              'recent evidence into one head under this protocol. A tie or reversal limits the necessity claim for separate '
              'Local predictions and output coordination. Parameter matching is approximate and may change effective '
              'capacity; this experiment alone cannot establish that independent branches are universally necessary.', '',
              '## Configuration reconciliation', '',
              'Historical main-text/appendix conflicts (cap 0.3 vs 0.5, all-MSE vs P12/P19 MAE, dimension128 vs64, '
              'Local auxiliary0 vs command default0.25) are recorded in docs/raw_recent_concat_experiments.md. '
              'The new manifest explicitly fixes cap0.5, d_model64, dataset-specific losses and checkpoint metrics, '
              'record splits, and zero auxiliary losses for BOTH variants. These decisions standardize this prospective '
              'study; they do not retroactively certify the old main-table runs.', '']
    for r in contrasts:
        if r['endpoint']=='mse':
            direction='lower' if r['adalg_mean'] < r['raw_concat_mean'] else 'higher or equal'
            lines += [f"Observed {r['dataset']}: AdaLG has {direction} mean MSE, with {r['adalg_better_seeds']}/{r['n']} favorable seeds. "
                      + ('This is only a smoke check.' if protocol['smoke'] else
                         'Interpret the paired effect and corrected tests above; no advantage is assumed in advance.'), '']
    (study / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return complete


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', default=['ushcn','p19'])
    parser.add_argument('--seeds', default='2021 2022 2023 2024 2025')
    parser.add_argument('--config', default='configs/experiments/raw_recent_concat.json')
    parser.add_argument('--out-dir', default='outputs/raw_recent_concat')
    parser.add_argument('--study-dir', help='existing study directory for --report-only')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--smoke', action='store_true', help='1 epoch, 2% training subset; not paper evidence')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--report-only', action='store_true')
    args=parser.parse_args()
    os.chdir(ROOT)
    if args.report_only:
        if not args.study_dir:
            parser.error('--report-only requires --study-dir')
        summarize(Path(args.study_dir).resolve())
        return 0
    datasets=[benchmark.load_dataset_config(d)['key'] for d in args.datasets]
    seeds=[int(s) for s in args.seeds.split()]
    if not seeds or len(set(seeds))!=len(seeds) or min(seeds)<0 or len(set(datasets))!=len(datasets):
        parser.error('Use unique datasets and nonnegative seeds.')
    if any(v is not None and v<1 for v in (args.epochs,args.batch_size)):
        parser.error('Budgets and batch sizes must be positive.')
    profile=read_json(args.config)
    if any(d not in profile['datasets'] for d in datasets):
        parser.error('Unsupported dataset.')
    if any(float(profile['common'][key]) != 0 for key in ('GLOBAL_AUX_LOSS_WEIGHT','LOCAL_AUX_LOSS_WEIGHT',
                                                        'UTILITY_LOSS_WEIGHT','ORACLE_GATE_LOSS_WEIGHT')):
        parser.error('This primary comparison requires zero auxiliary weights for both variants.')
    if profile['common']['POINT_LOCAL_FUSION_K'] != '12' or profile['common']['POINT_RECENT_STRICT_PAST'] != '1':
        parser.error('This study requires K=12 and strict past observations.')
    if profile['common']['POINT_DISABLE_CAUSAL_GATE_CONTEXT'] != '1':
        parser.error('The primary information-matched study must disable the extra causal gate context.')
    config=dict(profile=profile, datasets=datasets, seeds=seeds, epochs=args.epochs,
                batch_size=args.batch_size, smoke=args.smoke,
                source_sha256={p:sha(ROOT/p) for p in CODE_FILES})
    config['data_sha256']={}
    config['resolved_common']={}
    config['dataset_metadata']={}
    for dataset in datasets:
        settings=dict(profile['common'],**profile['datasets'][dataset])
        if args.epochs:
            settings['TRAIN_EPOCHS']=str(args.epochs)
        if args.batch_size:
            settings['BATCH_SIZE']=str(args.batch_size)
        if args.smoke:
            settings['TRAIN_EPOCHS']='1'
        for variant in VARIANTS:
            _, common, _, data=command_for(dataset,variant,settings,Path('__STUDY__'),seeds[0])
            config['resolved_common'][dataset+'/'+variant]=common
        config['dataset_metadata'][dataset]=data
        for name in data['data_path'].split(','):
            path=(Path(data['root_path'])/name).resolve()
            if path.is_file():
                config['data_sha256'][str(path)]=sha(path)
            elif not args.dry_run:
                raise FileNotFoundError(f'Required dataset file missing: {path}; prepare the dataset before the controlled study.')
    identifier=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:12]
    study=(Path(args.out_dir)/identifier).resolve()
    if not args.dry_run:
        study.mkdir(parents=True,exist_ok=True)
        record=study/'experiment_config.json'
        if record.exists() and read_json(record)!=config:
            raise RuntimeError('Existing experiment config differs; refusing to overwrite.')
        record.write_text(json.dumps(config,indent=2),encoding='utf-8')
        print(f'[study] {study}',flush=True)
    for dataset in datasets:
        settings=dict(profile['common'],**profile['datasets'][dataset])
        if args.epochs:
            settings['TRAIN_EPOCHS']=str(args.epochs)
        if args.batch_size:
            settings['BATCH_SIZE']=str(args.batch_size)
        if args.smoke:
            settings['TRAIN_EPOCHS']='1'
        for seed in seeds:
            for variant in VARIANTS:
                env,common,prefix,data=command_for(dataset,variant,settings,study,seed)
                if args.smoke:
                    common+=['--subset_rand_ratio','0.02']
                command,resumed=benchmark.build_seed_command(sys.executable,common,str(seed),prefix,resume=args.resume)
                if args.dry_run:
                    print(shlex.join(command))
                    continue
                run_dir=study/dataset/variant/f'seed{seed}'
                run_dir.mkdir(parents=True,exist_ok=True)
                command_path=run_dir/'executed_command.json'
                if command_path.exists() and not args.resume:
                    raise RuntimeError(f'Existing run: {run_dir}; use --resume or a different configuration.')
                record=dict(command=command,settings=settings,variant=variant,seed=seed,
                            source_sha256=config['source_sha256'])
                if not command_path.exists():
                    command_path.write_text(json.dumps(record,indent=2),encoding='utf-8')
                prior=run_dir/'run_status.json'
                if args.resume and prior.exists() and read_json(prior).get('status')=='complete':
                    old=read_json(prior)
                    if sha(old['checkpoint']) != old['checkpoint_sha256']:
                        raise ValueError('Existing checkpoint bytes changed; refusing resume.')
                    if sha(Path(old['result_directory'])/'raw_recent_queries.npz') != old['query_export_sha256']:
                        raise ValueError('Existing query export bytes changed; refusing resume.')
                    print(f'[resume verified outputs] {dataset} {variant} seed={seed}',flush=True)
                    continue
                log=run_dir/('evaluation.log' if resumed else 'training.log')
                print(f'[run] {dataset} {variant} seed={seed}; log={log}',flush=True)
                with log.open('w',encoding='utf-8') as file:
                    result=subprocess.run(command,cwd=ROOT,env=env,stdout=file,stderr=subprocess.STDOUT)
                if result.returncode:
                    prior.write_text(json.dumps(dict(status='failed',exit_code=result.returncode,log=str(log)),indent=2),encoding='utf-8')
                    summarize(study)
                    raise RuntimeError(f'Run failed, exit={result.returncode}; inspect {log}')
                paths=list((study/dataset/'checkpoints').glob(f'*{prefix}_seed{seed}_h*_0/checkpoint.pth'))
                if len(paths)!=1:
                    raise RuntimeError(f'Ambiguous checkpoint: {paths}')
                folder=study/dataset/'results'/paths[0].parent.name
                if not (folder/'raw_recent_queries.npz').exists():
                    raise RuntimeError('Missing full query export.')
                prior.write_text(json.dumps(dict(status='complete',checkpoint=str(paths[0]),
                                                checkpoint_sha256=sha(paths[0]),result_directory=str(folder),
                                                query_export_sha256=sha(folder/'raw_recent_queries.npz')),indent=2),encoding='utf-8')
            if not args.dry_run:
                summarize(study)
    if not args.dry_run:
        summarize(study)
        print(f'[report] {study / "report.md"}',flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
