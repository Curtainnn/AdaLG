#!/usr/bin/env python
"""Generate the two AdaLG gate-analysis figures from query-level evidence."""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable


def reexec_with_repo_python() -> None:
    if os.environ.get("GATE_FIGURES_NO_REEXEC") == "1":
        return
    repo_root = Path(__file__).resolve().parents[1]
    candidates = []
    if os.environ.get("PYTHON"):
        candidates.append(Path(os.environ["PYTHON"]))
    candidates.extend(
        [
            repo_root / ".venv" / "Scripts" / "python.exe",
            repo_root / ".venv" / "bin" / "python",
        ]
    )
    current = Path(sys.executable).resolve()
    for candidate in candidates:
        if not candidate.exists():
            continue
        if candidate.resolve() == current:
            return
        child_env = dict(os.environ)
        child_env["GATE_FIGURES_NO_REEXEC"] = "1"
        raise SystemExit(subprocess.call([str(candidate), *sys.argv], env=child_env))


reexec_with_repo_python()

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.stats import pearsonr, spearmanr
except ImportError:
    pearsonr = None
    spearmanr = None


GLOBAL_COLOR = "#1F4E79"
LOCAL_COLOR = "#A6473A"
ADALG_COLOR = "#4E7F3D"
GATE_COLOR = "#1A1A1A"
SCATTER_COLOR = "#2F6F9F"
FIT_COLOR = "#111111"
ZERO_COLOR = "#8B3E2F"
GMAX_COLOR = "#666666"
GRID_COLOR = "#D0D0D0"

GROUP_ORDER = ["Local-better", "Comparable", "Global-better"]


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 11,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    lower_to_original = {column.lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    raise KeyError(
        f"None of the expected columns were found: {list(candidates)}. "
        f"Available columns: {list(df.columns)}"
    )


def expand_inputs(patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            paths = [Path(match) for match in matches]
        elif Path(pattern).exists():
            paths = [Path(pattern)]
        else:
            paths = []

        for path in paths:
            if path.is_dir():
                files.extend(path.rglob("query_evidence_raw.csv"))
                files.extend(path.rglob("gate_query_data.csv"))
            else:
                files.append(path)

    unique = {str(path): path for path in files}
    return [unique[key] for key in sorted(unique)]


def load_query_data(patterns: list[str], strict: bool = False) -> tuple[pd.DataFrame, list[str], list[str]]:
    paths = expand_inputs(patterns)
    if not paths:
        raise FileNotFoundError(f"No CSV files matched: {patterns}")

    frames: list[pd.DataFrame] = []
    loaded_paths: list[str] = []
    skipped_paths: list[str] = []
    for path in paths:
        frame = pd.read_csv(path)
        if frame.empty:
            skipped_paths.append(f"{path}: empty")
            continue
        try:
            standardized = standardize_columns(frame)
        except (KeyError, ValueError) as exc:
            if strict:
                raise
            skipped_paths.append(f"{path}: {exc}")
            continue
        standardized["source_file"] = str(path)
        frames.append(standardized)
        loaded_paths.append(str(path))

    if not frames:
        skipped = "\n".join(f"- {item}" for item in skipped_paths[:10])
        raise ValueError(f"No usable CSV files found.\n{skipped}")
    return pd.concat(frames, ignore_index=True), loaded_paths, skipped_paths


def _optional_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    lower_to_original = {column.lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    return None


def _assert_close_columns(
    actual: pd.Series,
    expected: pd.Series,
    *,
    actual_name: str,
    expected_name: str,
) -> None:
    actual_values = pd.to_numeric(actual, errors="coerce").to_numpy(dtype=np.float64)
    expected_values = pd.to_numeric(expected, errors="coerce").to_numpy(dtype=np.float64)
    finite = np.isfinite(actual_values) & np.isfinite(expected_values)
    if finite.any() and not np.allclose(
        actual_values[finite],
        expected_values[finite],
        rtol=1e-5,
        atol=1e-7,
    ):
        max_error = float(np.max(np.abs(actual_values[finite] - expected_values[finite])))
        raise ValueError(
            f"Inconsistent query evidence: {actual_name} does not match {expected_name} "
            f"(maximum absolute discrepancy {max_error:.6g}). This commonly indicates "
            "that normalized and inverse-transformed values were mixed."
        )


def validate_raw_prediction_consistency(df: pd.DataFrame) -> None:
    """Check error identities that require predictions and targets to share one scale."""
    true_col = _optional_existing_column(df, ["y_true", "true", "target", "label", "ground_truth", "truth"])
    global_col = _optional_existing_column(
        df,
        ["y_global", "global_pred", "pred_global", "global_prediction"],
    )
    local_col = _optional_existing_column(
        df,
        ["y_local", "local_pred", "pred_local", "local_prediction"],
    )

    if true_col is not None and global_col is not None:
        expected = (pd.to_numeric(df[global_col], errors="coerce") - pd.to_numeric(df[true_col], errors="coerce")).abs()
        if "global_abs_error" in df.columns:
            _assert_close_columns(
                df["global_abs_error"],
                expected,
                actual_name="global_abs_error",
                expected_name=f"|{global_col} - {true_col}|",
            )

    if true_col is not None and local_col is not None:
        expected = (pd.to_numeric(df[local_col], errors="coerce") - pd.to_numeric(df[true_col], errors="coerce")).abs()
        if "local_abs_error" in df.columns:
            _assert_close_columns(
                df["local_abs_error"],
                expected,
                actual_name="local_abs_error",
                expected_name=f"|{local_col} - {true_col}|",
            )


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    validate_raw_prediction_consistency(df)
    gate_col = first_existing_column(df, ["gate", "g_q", "gate_weight", "fusion_gate", "final_gate"])
    output = pd.DataFrame({"gate": pd.to_numeric(df[gate_col], errors="coerce")})

    if {"A_q", "g_q"}.issubset(df.columns):
        output["advantage"] = pd.to_numeric(df["A_q"], errors="coerce")
    elif {"local_advantage", "gate"}.issubset(df.columns):
        output["advantage"] = pd.to_numeric(df["local_advantage"], errors="coerce")
    elif {"global_abs_error", "local_abs_error"}.issubset(df.columns):
        output["global_abs_error"] = pd.to_numeric(df["global_abs_error"], errors="coerce")
        output["local_abs_error"] = pd.to_numeric(df["local_abs_error"], errors="coerce")
        output["advantage"] = output["global_abs_error"] - output["local_abs_error"]
    else:
        y_true_col = first_existing_column(
            df,
            ["y_true", "true", "target", "label", "ground_truth", "truth"],
        )
        y_global_col = first_existing_column(
            df,
            ["y_global", "global_pred", "pred_global", "global_prediction"],
        )
        y_local_col = first_existing_column(
            df,
            ["y_local", "local_pred", "pred_local", "local_prediction"],
        )
        output["y_true"] = pd.to_numeric(df[y_true_col], errors="coerce")
        output["y_global"] = pd.to_numeric(df[y_global_col], errors="coerce")
        output["y_local"] = pd.to_numeric(df[y_local_col], errors="coerce")
        output["global_abs_error"] = np.abs(output["y_global"] - output["y_true"])
        output["local_abs_error"] = np.abs(output["y_local"] - output["y_true"])
        output["advantage"] = output["global_abs_error"] - output["local_abs_error"]

    if "global_abs_error" not in output.columns and "global_abs_error" in df.columns:
        output["global_abs_error"] = pd.to_numeric(df["global_abs_error"], errors="coerce")
    if "local_abs_error" not in output.columns and "local_abs_error" in df.columns:
        output["local_abs_error"] = pd.to_numeric(df["local_abs_error"], errors="coerce")

    if "abs_error" in df.columns:
        output["adalg_abs_error"] = pd.to_numeric(df["abs_error"], errors="coerce")
    elif {"y_true", "y_global", "y_local"}.issubset(output.columns):
        output["y_adalg"] = (1.0 - output["gate"]) * output["y_global"] + output["gate"] * output["y_local"]
        output["adalg_abs_error"] = np.abs(output["y_adalg"] - output["y_true"])

    if "group" in df.columns:
        output["group"] = df["group"].astype(str)
    elif "query_group" in df.columns:
        output["group"] = df["query_group"].astype(str)

    if "has_local_history" in df.columns:
        output["has_local_history"] = pd.to_numeric(df["has_local_history"], errors="coerce")

    output = output.replace([np.inf, -np.inf], np.nan)
    output = output.dropna(subset=["advantage", "gate"])
    return output


def normalize_existing_groups(series: pd.Series) -> pd.Series:
    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .str.replace("_", "-", regex=False)
        .str.replace(" ", "-", regex=False)
    )
    mapping = {
        "local-better": "Local-better",
        "localbetter": "Local-better",
        "comparable": "Comparable",
        "global-better": "Global-better",
        "globalbetter": "Global-better",
    }
    mapped = normalized.map(mapping)
    if mapped.isna().any():
        invalid = sorted(series[mapped.isna()].astype(str).unique().tolist())
        raise ValueError(f"Unrecognized group labels: {invalid}")
    return mapped


def add_derived_columns(
    df: pd.DataFrame,
    epsilon: float | None,
    comparable_quantile: float,
    include_no_local: bool,
) -> tuple[pd.DataFrame, float]:
    result = df.copy()
    if not include_no_local and "has_local_history" in result.columns:
        result = result[result["has_local_history"].astype(float) > 0.5].copy()

    if "global_abs_error" not in result.columns or "local_abs_error" not in result.columns:
        raise ValueError(
            "The group bar chart needs global/local absolute errors. Use query_evidence_raw.csv "
            "generated by QUERY_EVIDENCE_SAVE_RAW=1, not a two-column A_q/g_q file."
        )
    if "adalg_abs_error" not in result.columns:
        raise ValueError(
            "The group bar chart needs AdaLG absolute error. Expected abs_error in query_evidence_raw.csv."
        )

    if "group" in result.columns:
        result["group"] = normalize_existing_groups(result["group"])
        threshold = float("nan")
    else:
        if epsilon is not None:
            if epsilon < 0:
                raise ValueError("--epsilon must be non-negative.")
            threshold = float(epsilon)
        else:
            comparable_quantile = min(max(float(comparable_quantile), 0.0), 1.0)
            threshold = (
                float(np.quantile(np.abs(result["advantage"]), comparable_quantile))
                if comparable_quantile > 0.0
                else 0.0
            )
        result["group"] = np.select(
            [result["advantage"] > threshold, result["advantage"] < -threshold],
            ["Local-better", "Global-better"],
            default="Comparable",
        )

    result = result.dropna(subset=["global_abs_error", "local_abs_error", "adalg_abs_error", "gate", "advantage"])
    if result.empty:
        raise ValueError("No finite query rows remain after filtering.")
    return result, threshold


def compute_group_stats(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby("group", observed=True)
        .agg(
            count=("group", "size"),
            global_mae=("global_abs_error", "mean"),
            local_mae=("local_abs_error", "mean"),
            adalg_mae=("adalg_abs_error", "mean"),
            gate_mean=("gate", "mean"),
            gate_std=("gate", "std"),
        )
        .reindex(GROUP_ORDER)
    )
    if grouped.isna().all(axis=1).any():
        missing_groups = grouped.index[grouped.isna().all(axis=1)].tolist()
        raise ValueError(f"Empty query groups: {missing_groups}")
    grouped["ratio"] = grouped["count"] / grouped["count"].sum()
    grouped["gate_se"] = grouped["gate_std"].fillna(0.0) / np.sqrt(grouped["count"])
    grouped["gate_ci95"] = 1.96 * grouped["gate_se"]
    return grouped


def save_figure(fig: plt.Figure, base_path: Path) -> None:
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_gate_group(stats: pd.DataFrame, out_dir: Path) -> None:
    x = np.arange(len(GROUP_ORDER))
    width = 0.23
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.bar(
        x - width,
        stats["global_mae"],
        width,
        label="Global",
        color=GLOBAL_COLOR,
        edgecolor="black",
        linewidth=0.8,
        hatch="///",
    )
    ax.bar(
        x,
        stats["local_mae"],
        width,
        label="Local",
        color=LOCAL_COLOR,
        edgecolor="black",
        linewidth=0.8,
        hatch="///",
    )
    ax.bar(
        x + width,
        stats["adalg_mae"],
        width,
        label="AdaLG",
        color=ADALG_COLOR,
        edgecolor="black",
        linewidth=0.8,
        hatch="///",
    )
    labels = [f"{group}\n({ratio * 100:.2f}%)" for group, ratio in zip(GROUP_ORDER, stats["ratio"])]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Group-wise MAE")
    ax.grid(axis="y", linewidth=0.6, color=GRID_COLOR, alpha=0.55)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax2 = ax.twinx()
    ax2.errorbar(
        x,
        stats["gate_mean"],
        yerr=stats["gate_ci95"],
        color=GATE_COLOR,
        marker="o",
        markersize=7,
        markerfacecolor="white",
        markeredgecolor=GATE_COLOR,
        markeredgewidth=1.5,
        linewidth=1.6,
        capsize=3,
        label=r"Mean $g_q$",
    )
    ax2.set_ylabel(r"Mean gate $g_q$")
    upper = max(0.32, float((stats["gate_mean"] + stats["gate_ci95"]).max()) * 1.18)
    ax2.set_ylim(0.0, upper)
    ax2.spines["top"].set_visible(False)

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(
        handles1 + handles2,
        labels1 + labels2,
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        columnspacing=1.6,
        handletextpad=0.7,
    )
    fig.tight_layout()
    save_figure(fig, out_dir / "gate_group_analysis")


def corr(x_values: np.ndarray, y_values: np.ndarray, method: str) -> float:
    if method == "pearson" and pearsonr is not None:
        return float(pearsonr(x_values, y_values).statistic)
    if method == "spearman" and spearmanr is not None:
        return float(spearmanr(x_values, y_values).statistic)
    x_series = pd.Series(x_values)
    y_series = pd.Series(y_values)
    if method == "spearman":
        x_series = x_series.rank(method="average")
        y_series = y_series.rank(method="average")
    return float(x_series.corr(y_series))


def plot_gate_scatter(df: pd.DataFrame, out_dir: Path, sample_size: int, alpha: float) -> tuple[float, float]:
    advantage = df["advantage"].to_numpy(dtype=np.float64)
    gate = df["gate"].to_numpy(dtype=np.float64)
    # Correlations are deliberately computed from every query before visual
    # sampling and before any binned summaries are constructed.
    pearson_r = (
        float(pearsonr(df["advantage"], df["gate"]).statistic)
        if pearsonr is not None
        else corr(advantage, gate, "pearson")
    )
    spearman_rho = (
        float(spearmanr(df["advantage"], df["gate"]).statistic)
        if spearmanr is not None
        else corr(advantage, gate, "spearman")
    )
    slope, intercept = np.polyfit(advantage, gate, deg=1)
    x_min = float(np.nanpercentile(advantage, 1))
    x_max = float(np.nanpercentile(advantage, 99))
    x_line = np.linspace(x_min, x_max, 300)
    plot_df = df.sample(sample_size, random_state=2026) if sample_size > 0 and len(df) > sample_size else df

    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    ax.scatter(
        plot_df["advantage"],
        plot_df["gate"],
        s=15,
        alpha=alpha,
        color=SCATTER_COLOR,
        linewidths=0,
        rasterized=True,
    )
    ax.plot(x_line, slope * x_line + intercept, color=FIT_COLOR, linewidth=1.8, label="Linear fit")
    ax.axvline(0.0, color=ZERO_COLOR, linestyle="--", linewidth=1.5)
    ax.axhline(y=0.3, color=GMAX_COLOR, linestyle=":", linewidth=1.2)
    ax.text(
        0.98,
        0.955,
        r"$g_{\max}=0.3$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color=GMAX_COLOR,
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            alpha=0.80,
            pad=1.2,
        ),
    )
    ax.set_xlabel(r"Query-level local advantage $A_q$")
    ax.set_ylabel(r"Gate weight $g_q$")
    ax.set_ylim(0.02, 0.315)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        0.98,
        0.13,
        rf"Pearson $r={pearson_r:.4f}$" + "\n" + rf"Spearman $\rho={spearman_rho:.4f}$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        bbox={
            "boxstyle": "square,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#C7C7C7",
            "linewidth": 1.0,
        },
    )
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    save_figure(fig, out_dir / "gate_advantage_scatter")
    return pearson_r, spearman_rho


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate AdaLG gate group and gate-advantage figures.")
    parser.add_argument(
        "--scatter-csv",
        nargs="+",
        required=True,
        help='CSV files, directories, or glob patterns, e.g. "test_results/*gate_scatter_5seed*/query_evidence_raw.csv"',
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="Fixed Comparable threshold. If omitted, --comparable-quantile is used.",
    )
    parser.add_argument(
        "--comparable-quantile",
        type=float,
        default=0.2,
        help="Quantile of |A_q| assigned to Comparable when no group column exists.",
    )
    parser.add_argument("--include-no-local", action="store_true", help="Keep queries with has_local_history=0.")
    parser.add_argument("--strict", action="store_true", help="Fail on the first incompatible CSV.")
    parser.add_argument("--sample-size", type=int, default=50000, help="Visual scatter sample size; <=0 plots all rows.")
    parser.add_argument("--alpha", type=float, default=0.32, help="Scatter alpha.")
    parser.add_argument(
        "--expected-query-count",
        type=int,
        default=1860,
        help="Required query count for the paper figure; use 0 to disable this check.",
    )
    parser.add_argument("--out-dir", default="figures_generated", help="Output directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_matplotlib()
    raw_df, files, skipped = load_query_data(args.scatter_csv, strict=args.strict)
    df, threshold = add_derived_columns(
        raw_df,
        epsilon=args.epsilon,
        comparable_quantile=args.comparable_quantile,
        include_no_local=args.include_no_local,
    )
    if args.expected_query_count > 0 and len(df) != args.expected_query_count:
        raise ValueError(
            f"Expected {args.expected_query_count:,} query rows for the paper figure, "
            f"but found {len(df):,} after finite-value and local-history filtering. "
            "Pearson and Spearman were not reported."
        )
    stats = compute_group_stats(df)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_gate_group(stats, out_dir)
    pearson_r, spearman_rho = plot_gate_scatter(
        df,
        out_dir,
        sample_size=args.sample_size,
        alpha=args.alpha,
    )
    stats.to_csv(out_dir / "gate_group_statistics.csv", index=True)
    df[["advantage", "gate", "global_abs_error", "local_abs_error", "adalg_abs_error", "group"]].to_csv(
        out_dir / "gate_query_data.csv",
        index=False,
    )

    print(f"files_loaded: {len(files)}")
    print(f"files_skipped: {len(skipped)}")
    if skipped:
        for item in skipped[:10]:
            print(f"skipped: {item}", file=sys.stderr)
    print(f"query_rows: {len(df)}")
    print(f"correlation_rows: {len(df)}")
    print("correlation_level: query (unbinned, unaggregated)")
    print("prediction_scale_check: passed")
    print(f"comparable_threshold: {threshold:.8f}" if np.isfinite(threshold) else "comparable_threshold: from_group_column")
    print(f"pearson_r: {pearson_r:.6f}")
    print(f"spearman_rho: {spearman_rho:.6f}")
    print(f"output_dir: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
