#!/usr/bin/env python
import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path


def reexec_with_repo_python():
    if os.environ.get("PAPER_FIGURES_NO_REEXEC") == "1":
        return
    repo_root = Path(__file__).resolve().parents[1]
    candidates = []
    if os.environ.get("PYTHON"):
        candidates.append(Path(os.environ["PYTHON"]))
    candidates.extend([
        repo_root / ".venv" / "Scripts" / "python.exe",
        repo_root / ".venv" / "bin" / "python",
    ])
    current = Path(sys.executable).resolve()
    for candidate in candidates:
        if not candidate.exists():
            continue
        if candidate.resolve() == current:
            return
        env = dict(os.environ)
        env["PAPER_FIGURES_NO_REEXEC"] = "1"
        raise SystemExit(subprocess.call([str(candidate), *sys.argv], env=env))


reexec_with_repo_python()

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageOps


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

METHODS = ["DLinear", "GRU-D", "ASTGI", "t-PatchGNN", "GraFITi", "TimeCHEAT", "AdaLG"]

TABLE2 = {
    "USHCN": {
        "MSE": [60.39, 31.75, 34.68, 34.68, 34.81, 39.45, 31.38],
        "MSE_std": [1.17, 0.64, 0.70, 0.64, 0.33, 7.85, 0.20],
        "MAE": [44.61, 29.91, 28.73, 28.04, 29.19, 31.42, 27.29],
        "MAE_std": [0.52, 0.58, 1.65, 0.93, 0.38, 2.96, 0.58],
    },
    "Human Activity": {
        "MSE": [29.56, 24.23, 23.68, 24.19, 23.89, 34.81, 23.43],
        "MSE_std": [0.12, 0.47, 0.62, 0.20, 0.21, 0.33, 0.17],
        "MAE": [36.05, 30.70, 28.74, 28.70, 28.29, 30.16, 28.19],
        "MAE_std": [0.10, 0.40, 0.91, 0.17, 0.12, 1.59, 0.31],
    },
    "P12": {
        "MSE": [82.60, 45.18, 46.04, 45.95, 45.18, 54.54, 44.59],
        "MSE_std": [0.03, 0.16, 0.26, 0.15, 0.20, 0.04, 0.07],
        "MAE": [66.61, 43.36, 41.93, 42.06, 41.38, 48.17, 41.31],
        "MAE_std": [0.02, 0.11, 0.20, 0.11, 0.09, 0.02, 0.04],
    },
    "P19": {
        "MSE": [55.60, 51.03, 50.77, 50.79, 50.41, 54.96, 48.99],
        "MSE_std": [0.01, 0.17, 0.30, 0.17, 0.26, 3.90, 0.08],
        "MAE": [52.30, 49.08, 48.73, 48.75, 48.48, 51.53, 48.44],
        "MAE_std": [0.02, 0.08, 0.29, 0.14, 0.16, 2.45, 0.02],
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate paper figures from hard-coded tables and query evidence CSV.")
    parser.add_argument("--out-dir", default="figures_generated")
    parser.add_argument(
        "--scatter-csv",
        nargs="*",
        default=["gate_query_data.csv"],
        help="CSV files, directories, or glob patterns for gate_query_data.csv/query_evidence_raw.csv.",
    )
    parser.add_argument("--scatter-prefix", default="table5_gate_scatter")
    parser.add_argument("--scatter-sample-size", type=int, default=50000)
    parser.add_argument("--scatter-alpha", type=float, default=0.08)
    parser.add_argument("--scatter-bins", type=int, default=10)
    parser.add_argument(
        "--include-no-local",
        action="store_true",
        help="Keep rows with has_local_history=0. By default these rows are filtered when the column exists.",
    )
    parser.add_argument(
        "--strict-scatter",
        action="store_true",
        help="Fail immediately when any matched scatter CSV has incompatible columns.",
    )
    return parser.parse_args()


def safe_corr(x_values, y_values):
    x_values = np.asarray(x_values, dtype=np.float64)
    y_values = np.asarray(y_values, dtype=np.float64)
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    x_values = x_values[valid]
    y_values = y_values[valid]
    if x_values.size < 2:
        return 0.0
    x_centered = x_values - np.mean(x_values)
    y_centered = y_values - np.mean(y_values)
    denom = np.sqrt(np.sum(x_centered ** 2) * np.sum(y_centered ** 2))
    if denom <= 0.0:
        return 0.0
    return float(np.sum(x_centered * y_centered) / denom)


def plot_table2_panel(dataset, output_path):
    d = TABLE2[dataset]
    x = np.arange(len(METHODS))
    width = 0.36
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.bar(x - width / 2, d["MSE"], width, yerr=d["MSE_std"], capsize=2, label="MSE")
    ax.bar(x + width / 2, d["MAE"], width, yerr=d["MAE_std"], capsize=2, label="MAE")
    ax.set_title(dataset)
    ax.set_ylabel(r"Error ($\times 10^{-2}$)")
    ax.set_xticks(x)
    ax.set_xticklabels(METHODS, rotation=28, ha="right")
    ax.legend(frameon=False, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def combine_panels(panel_paths, output_path):
    imgs = [Image.open(path).convert("RGB") for path in panel_paths]
    cell_w = max(im.width for im in imgs)
    cell_h = max(im.height for im in imgs)
    canvas = Image.new("RGB", (2 * cell_w, 2 * cell_h), "white")
    positions = [(0, 0), (cell_w, 0), (0, cell_h), (cell_w, cell_h)]
    for im, pos in zip(imgs, positions):
        contained = ImageOps.contain(im, (cell_w, cell_h))
        x_pad = pos[0] + (cell_w - contained.width) // 2
        y_pad = pos[1] + (cell_h - contained.height) // 2
        canvas.paste(contained, (x_pad, y_pad))
    canvas.save(output_path, dpi=(600, 600))


def plot_table5_groups(output_path):
    groups = ["Local-better\n(42.15%)", "Comparable\n(20.16%)", "Global-better\n(37.69%)"]
    global_mae = [0.6754, 0.2378, 0.3204]
    local_mae = [0.2785, 0.2113, 1.0747]
    adalg_mae = [0.2320, 0.1359, 0.3489]
    mean_gate = [0.23, 0.25, 0.18]
    x = np.arange(3)
    width = 0.23

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.bar(x - width, global_mae, width, label="Global")
    ax.bar(x, local_mae, width, label="Local")
    ax.bar(x + width, adalg_mae, width, label="AdaLG")
    ax.set_ylabel("Group-wise MAE")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax2 = ax.twinx()
    ax2.plot(
        x,
        mean_gate,
        color="black",
        marker="o",
        markerfacecolor="white",
        markeredgecolor="black",
        linewidth=1.5,
        label=r"Mean $g_q$",
    )
    ax2.set_ylabel(r"Mean gate $g_q$")
    ax2.set_ylim(0, 0.34)
    ax2.spines["top"].set_visible(False)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, ncol=4, loc="upper center")
    fig.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def expand_scatter_inputs(patterns):
    paths = []
    for pattern in patterns:
        matches = glob.glob(str(pattern), recursive=True)
        if matches:
            paths.extend(matches)
        elif Path(pattern).exists():
            paths.append(pattern)

    files = []
    for path in paths:
        path = Path(path)
        if path.is_dir():
            files.extend(path.rglob("gate_query_data.csv"))
            files.extend(path.rglob("query_evidence_raw.csv"))
        else:
            files.append(path)

    unique = {}
    for path in files:
        unique[str(Path(path))] = Path(path)
    return [unique[key] for key in sorted(unique)]


def read_scatter_data(csv_path, include_no_local=False):
    df = pd.read_csv(csv_path)
    if {"A_q", "g_q"}.issubset(df.columns):
        out = df[["A_q", "g_q"]].copy()
    elif {"local_advantage", "gate"}.issubset(df.columns):
        if not include_no_local and "has_local_history" in df.columns:
            df = df[df["has_local_history"].astype(float) > 0.5]
        out = pd.DataFrame({"A_q": df["local_advantage"], "g_q": df["gate"]})
    elif {"global_abs_error", "local_abs_error", "gate"}.issubset(df.columns):
        if not include_no_local and "has_local_history" in df.columns:
            df = df[df["has_local_history"].astype(float) > 0.5]
        out = pd.DataFrame({
            "A_q": df["global_abs_error"].astype(float) - df["local_abs_error"].astype(float),
            "g_q": df["gate"],
        })
    else:
        true_col = "y_true" if "y_true" in df.columns else "true"
        global_col = "y_global" if "y_global" in df.columns else "global_pred"
        local_col = "y_local" if "y_local" in df.columns else "local_pred"
        required = {true_col, global_col, local_col, "gate"}
        missing = required.difference(df.columns)
        if missing:
            available = ", ".join(str(col) for col in df.columns)
            raise ValueError(
                f"Missing columns in {csv_path}: {sorted(missing)}. "
                f"Available columns: {available}"
            )
        if not include_no_local and "has_local_history" in df.columns:
            df = df[df["has_local_history"].astype(float) > 0.5]
        out = pd.DataFrame({
            "A_q": np.abs(df[global_col].astype(float) - df[true_col].astype(float))
            - np.abs(df[local_col].astype(float) - df[true_col].astype(float)),
            "g_q": df["gate"],
        })
    out["A_q"] = pd.to_numeric(out["A_q"], errors="coerce")
    out["g_q"] = pd.to_numeric(out["g_q"], errors="coerce")
    out = out[np.isfinite(out["A_q"]) & np.isfinite(out["g_q"])].copy()
    if out.empty:
        raise ValueError(f"No finite scatter rows in {csv_path}.")
    return out


def load_scatter_data(csv_paths, include_no_local=False, strict=False):
    frames = []
    skipped = []
    for csv_path in csv_paths:
        try:
            cur = read_scatter_data(csv_path, include_no_local=include_no_local)
        except ValueError as exc:
            if strict:
                raise
            skipped.append((csv_path, str(exc)))
            continue
        cur["source_file"] = str(csv_path)
        frames.append(cur)
    if not frames:
        details = "\n".join(f"- {path}: {reason}" for path, reason in skipped[:10])
        raise ValueError(
            "No usable scatter CSV files found. Need one of these column sets: "
            "A_q/g_q, local_advantage/gate, global_abs_error/local_abs_error/gate, "
            "or true/global_pred/local_pred/gate.\n"
            f"Skipped files: {len(skipped)}\n{details}"
        )
    for path, reason in skipped[:20]:
        print(f"[scatter] skipped incompatible file: {path}", file=sys.stderr)
        print(f"[scatter] reason: {reason}", file=sys.stderr)
    if len(skipped) > 20:
        print(f"[scatter] skipped {len(skipped) - 20} more incompatible files.", file=sys.stderr)
    return pd.concat(frames, ignore_index=True), skipped


def build_binned_curve(df, bin_count):
    if bin_count <= 1 or len(df) < bin_count:
        return pd.DataFrame()
    ranked = df["A_q"].rank(method="first")
    bin_id = pd.qcut(ranked, q=bin_count, labels=False, duplicates="drop")
    rows = []
    for cur_bin, cur_df in df.groupby(bin_id, sort=True):
        g_std = float(cur_df["g_q"].std(ddof=0)) if len(cur_df) > 1 else 0.0
        rows.append({
            "bin": int(cur_bin),
            "A_q_mean": float(cur_df["A_q"].mean()),
            "g_q_mean": float(cur_df["g_q"].mean()),
            "g_q_sem": g_std / float(np.sqrt(len(cur_df))),
            "n": int(len(cur_df)),
        })
    return pd.DataFrame(rows)


def plot_gate_scatter(
    csv_paths,
    output_path,
    sample_size=50000,
    alpha=0.08,
    bins=10,
    include_no_local=False,
    strict=False,
):
    df, skipped = load_scatter_data(csv_paths, include_no_local=include_no_local, strict=strict)
    advantage = df["A_q"].to_numpy(dtype=np.float64)
    gate = df["g_q"].to_numpy(dtype=np.float64)

    pearson_r = safe_corr(advantage, gate)
    spearman_rho = safe_corr(df["A_q"].rank(method="average"), df["g_q"].rank(method="average"))
    slope, intercept = np.polyfit(advantage, gate, 1)
    xx = np.linspace(np.nanpercentile(advantage, 1), np.nanpercentile(advantage, 99), 200)
    plot_df = df.sample(sample_size, random_state=2026) if sample_size > 0 and len(df) > sample_size else df
    bin_df = build_binned_curve(df, bins)

    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.scatter(plot_df["A_q"], plot_df["g_q"], s=8, alpha=alpha, linewidths=0, rasterized=True)
    ax.plot(xx, slope * xx + intercept, linewidth=1.5, color="black", label="Linear fit")
    if not bin_df.empty:
        ax.plot(
            bin_df["A_q_mean"],
            bin_df["g_q_mean"],
            color="#d97706",
            marker="o",
            linewidth=1.6,
            label="Binned mean",
        )
        ax.fill_between(
            bin_df["A_q_mean"],
            bin_df["g_q_mean"] - bin_df["g_q_sem"],
            bin_df["g_q_mean"] + bin_df["g_q_sem"],
            color="#d97706",
            alpha=0.18,
            linewidth=0,
        )
    ax.axvline(0, linestyle="--", linewidth=1, color="#9a3412")
    ax.set_xlabel(r"Query-level local advantage $A_q$")
    ax.set_ylabel(r"Gate weight $g_q$")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        0.03,
        0.97,
        rf"Pearson $r={pearson_r:.4f}$" + "\n" + rf"Spearman $\rho={spearman_rho:.4f}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.8"),
    )
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    df[["A_q", "g_q"]].to_csv(output_path.with_name("gate_query_data.csv"), index=False)
    bin_df.to_csv(output_path.with_name(f"{output_path.stem}_binned_curve.csv"), index=False)
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return pearson_r, spearman_rho, len(df), len(skipped)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    panel_paths = []
    for dataset in TABLE2:
        panel = out_dir / f"table2_{dataset.lower().replace(' ', '_')}.png"
        plot_table2_panel(dataset, panel)
        panel_paths.append(panel)

    combine_panels(panel_paths, out_dir / "table2_main_performance.png")
    plot_table5_groups(out_dir / "table5_gate_group.png")

    scatter_csvs = expand_scatter_inputs(args.scatter_csv)
    if scatter_csvs:
        pearson_r, spearman_rho, n_rows, n_skipped = plot_gate_scatter(
            scatter_csvs,
            out_dir / f"{args.scatter_prefix}.png",
            sample_size=args.scatter_sample_size,
            alpha=args.scatter_alpha,
            bins=args.scatter_bins,
            include_no_local=args.include_no_local,
            strict=args.strict_scatter,
        )
        print(f"scatter_files: {len(scatter_csvs)}")
        print(f"scatter_files_skipped: {n_skipped}")
        print(f"scatter_rows: {n_rows}")
        print(f"scatter_pearson: {pearson_r:.6f}")
        print(f"scatter_spearman: {spearman_rho:.6f}")
    else:
        print(f"Scatter skipped: no files matched {args.scatter_csv}.")
    print(f"output_dir: {out_dir}")


if __name__ == "__main__":
    raise SystemExit(main())
