# Unified Forecast Benchmark

This repository keeps the model implementations under `models/` and uses one outer
benchmark protocol for comparison.

Fairness is controlled at the protocol level:

- same dataset
- same train/val/test split arguments
- same `run.py` training and test entry point
- same metric implementation in `utils/metrics.py`
- model-specific hyperparameters isolated in `configs/models/*_config.py`

## Run

```bash
# Timer-XL on Human Activity
python scripts/benchmark.py timer_xl human_activity

# ASTGI on the same protocol
python scripts/benchmark.py astgi human_activity

# AdaLG on the same protocol
python scripts/benchmark.py AdaLG human_activity
```

Supported dataset keys:

```text
human_activity, ushcn, physionet, etth1, ecl
```

Useful overrides:

```bash
SEED_LIST="2021 2022 2023" python scripts/benchmark.py astgi ushcn
POINT_SPLIT_MODE=chronological python scripts/benchmark.py timer_xl ushcn
BATCH_SIZE=8 TRAIN_EPOCHS=50 python scripts/benchmark.py astgi human_activity
```

The benchmark writes per-run metrics to `test_results/` through `run.py`, then
prints the cross-seed summary directly in `benchmark.py`.
If `.venv` exists, `benchmark.py` uses it for the child `run.py` process unless
`PYTHON` is set.

## Script Layout

```text
scripts/benchmark.py                  # cross-platform shared outer protocol
scripts/benchmark.sh                  # optional Bash wrapper for benchmark.py
configs/models/timer_xl_config.py     # Timer-XL adapted config
configs/models/astgi_config.py        # ASTGI adapted config
configs/models/adalg_config.py        # AdaLG adapted config
scripts/data/                         # dataset preprocessing helpers
```

The old `scripts/adaptation`, `scripts/pretrain`, and `scripts/supervised`
layouts are intentionally removed from the active script surface.

## Shared-encoder controls and local evidence experiments

See [the experiment guide](docs/local_global_experiments.md) for the shared
encoder/two-head baseline, independent raw local encoders, complementarity,
local degradation, and validation-only tuning:

```powershell
.\.venv\Scripts\python.exe scripts/local_global_comparison.py --datasets ushcn physionet
.\.venv\Scripts\python.exe scripts/tune_adalg.py ushcn
```

For isolated gate-bound, availability-feature and reliability-supervision
ablations, see [the gate mechanism guide](docs/gate_mechanism_experiments.md):

```bash
python scripts/gate_mechanism_ablation.py --datasets ushcn --protocol frozen --resume
```

For the single-head raw-recent concatenation control, full keyed query exports,
and isolated paired reports, see [the raw concat study guide](docs/raw_recent_concat_experiments.md):

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/raw_recent_concat_comparison.py --datasets ushcn p19 --resume
```

This prospective study matches the final-prediction objective, disables auxiliary
branch/gate losses for both variants, and does not reuse historical manuscript runs.
