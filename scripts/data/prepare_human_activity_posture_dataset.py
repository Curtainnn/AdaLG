import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


RAW_COLUMNS = [
    "sequence",
    "tag_id",
    "timestamp_numeric",
    "date_str",
    "x",
    "y",
    "z",
    "activity",
]

DATE_FORMAT = "%d.%m.%Y %H:%M:%S:%f"

TAG_NAME_BY_ID = {
    "010-000-024-033": "ankle_left",
    "010-000-030-096": "ankle_right",
    "020-000-033-111": "chest",
    "020-000-032-221": "belt",
}

TAG_ORDER = [
    "ankle_left",
    "ankle_right",
    "chest",
    "belt",
]

AXIS_ORDER = ["x", "y", "z"]

VAR_NAMES = [f"{tag}_{axis}" for tag in TAG_ORDER for axis in AXIS_ORDER]
VAR_ID_BY_NAME = {name: idx for idx, name in enumerate(VAR_NAMES)}

EVENT_ID_COLUMNS = [
    "split",
    "subject",
    "sequence",
    "trial",
    "timestamp_numeric",
    "datetime",
    "rel_time_ms",
    "rel_time_s",
    "activity",
    "tag_id",
    "tag_name",
]

SEQUENCE_ID_COLUMNS = ["split", "subject", "sequence", "trial"]


def _parse_subject_spec(spec):
    return [item.strip() for item in str(spec).split(",") if item.strip()]


def _make_subject_to_split(train_subjects, val_subjects, test_subjects):
    subject_to_split = {}
    for subject in train_subjects:
        subject_to_split[subject] = "train"
    for subject in val_subjects:
        if subject in subject_to_split:
            raise ValueError(f"Subject '{subject}' is assigned to multiple splits.")
        subject_to_split[subject] = "val"
    for subject in test_subjects:
        if subject in subject_to_split:
            raise ValueError(f"Subject '{subject}' is assigned to multiple splits.")
        subject_to_split[subject] = "test"
    return subject_to_split


def _activity_mode(series):
    modes = series.mode(dropna=False)
    if modes.empty:
        return None
    return modes.iloc[0]


def load_raw_events(input_path, subject_to_split):
    df = pd.read_csv(input_path, header=None, names=RAW_COLUMNS)
    df["datetime"] = pd.to_datetime(df["date_str"], format=DATE_FORMAT)
    df["tag_name"] = df["tag_id"].map(TAG_NAME_BY_ID)
    if df["tag_name"].isna().any():
        missing_tags = sorted(df.loc[df["tag_name"].isna(), "tag_id"].unique().tolist())
        raise ValueError(f"Unknown tag ids found: {missing_tags}")

    df["subject"] = df["sequence"].str[0]
    df["trial"] = df["sequence"].str[1:].astype(int)
    df["split"] = df["subject"].map(subject_to_split)
    if df["split"].isna().any():
        missing_subjects = sorted(df.loc[df["split"].isna(), "subject"].unique().tolist())
        raise ValueError(f"Subjects missing split assignment: {missing_subjects}")

    df = df.sort_values(["sequence", "datetime", "tag_id"]).reset_index(drop=True)
    seq_start = df.groupby("sequence")["datetime"].transform("min")
    df["rel_time_ms"] = (df["datetime"] - seq_start).dt.total_seconds() * 1000.0
    df["rel_time_s"] = df["rel_time_ms"] / 1000.0
    return df


def build_point_observations(raw_df):
    point_df = raw_df.melt(
        id_vars=EVENT_ID_COLUMNS,
        value_vars=AXIS_ORDER,
        var_name="axis",
        value_name="value",
    )
    point_df["var_name"] = point_df["tag_name"] + "_" + point_df["axis"]
    point_df["var_id"] = point_df["var_name"].map(VAR_ID_BY_NAME).astype(int)
    point_df = point_df.sort_values(
        ["sequence", "rel_time_ms", "var_id", "timestamp_numeric"]
    ).reset_index(drop=True)
    return point_df


def build_dense_grid(point_df, raw_df, grid_ms):
    dense_long = point_df.copy()
    dense_long["bin_idx"] = np.floor(dense_long["rel_time_ms"] / float(grid_ms)).astype(int)

    value_df = (
        dense_long.groupby(SEQUENCE_ID_COLUMNS + ["bin_idx", "var_name", "var_id"], as_index=False)["value"]
        .mean()
    )
    dense_df = value_df.pivot_table(
        index=SEQUENCE_ID_COLUMNS + ["bin_idx"],
        columns="var_name",
        values="value",
        aggfunc="mean",
    )
    dense_df = dense_df.reindex(columns=VAR_NAMES)
    dense_df = dense_df.reset_index()
    dense_df.columns.name = None

    activity_df = raw_df.copy()
    activity_df["bin_idx"] = np.floor(activity_df["rel_time_ms"] / float(grid_ms)).astype(int)
    activity_df = (
        activity_df.groupby(SEQUENCE_ID_COLUMNS + ["bin_idx"], as_index=False)["activity"]
        .agg(_activity_mode)
    )

    dense_df = dense_df.merge(activity_df, on=SEQUENCE_ID_COLUMNS + ["bin_idx"], how="left")
    dense_df["rel_time_ms"] = dense_df["bin_idx"] * float(grid_ms)
    dense_df["rel_time_s"] = dense_df["rel_time_ms"] / 1000.0
    dense_df["observed_var_count"] = dense_df[VAR_NAMES].notna().sum(axis=1)
    dense_df = dense_df[
        SEQUENCE_ID_COLUMNS
        + ["bin_idx", "rel_time_ms", "rel_time_s", "activity", "observed_var_count"]
        + VAR_NAMES
    ].sort_values(["sequence", "bin_idx"]).reset_index(drop=True)
    return dense_df


def build_ushcn_style_sporadic(dense_df):
    sporadic_df = dense_df[SEQUENCE_ID_COLUMNS + ["bin_idx", "activity"] + VAR_NAMES].copy()
    sporadic_df = sporadic_df.rename(
        columns={
            "split": "Split",
            "subject": "Subject",
            "sequence": "ID",
            "trial": "Trial",
            "bin_idx": "Time",
            "activity": "Activity",
        }
    )

    ordered_cols = ["Split", "Subject", "ID", "Trial", "Time", "Activity"]
    for var_idx, var_name in enumerate(VAR_NAMES):
        value_col = f"Value_{var_idx}"
        mask_col = f"Mask_{var_idx}"
        sporadic_df[value_col] = sporadic_df[var_name].astype(np.float32)
        sporadic_df[mask_col] = sporadic_df[var_name].notna().astype(np.float32)
        ordered_cols.extend([value_col, mask_col])

    return sporadic_df[ordered_cols].sort_values(["ID", "Time"]).reset_index(drop=True)


def build_human_activity_1ms(raw_df):
    activity_df = raw_df.copy()
    first_tp = activity_df.groupby("sequence")["timestamp_numeric"].transform("min").astype(float)
    activity_df["time_ms"] = np.round((activity_df["timestamp_numeric"].astype(float) - first_tp) / 1e4).astype(int)

    point_df = activity_df.melt(
        id_vars=SEQUENCE_ID_COLUMNS + ["time_ms", "activity", "tag_name"],
        value_vars=AXIS_ORDER,
        var_name="axis",
        value_name="value",
    )
    point_df["var_name"] = point_df["tag_name"] + "_" + point_df["axis"]
    point_df["var_id"] = point_df["var_name"].map(VAR_ID_BY_NAME).astype(int)

    value_df = (
        point_df.groupby(SEQUENCE_ID_COLUMNS + ["time_ms", "var_name", "var_id"], as_index=False)["value"]
        .mean()
    )
    value_wide = value_df.pivot_table(
        index=SEQUENCE_ID_COLUMNS + ["time_ms"],
        columns="var_name",
        values="value",
        aggfunc="mean",
    )
    value_wide = value_wide.reindex(columns=VAR_NAMES).reset_index()
    value_wide.columns.name = None

    activity_mode_df = (
        activity_df.groupby(SEQUENCE_ID_COLUMNS + ["time_ms"], as_index=False)["activity"]
        .agg(_activity_mode)
    )
    human_df = value_wide.merge(
        activity_mode_df,
        on=SEQUENCE_ID_COLUMNS + ["time_ms"],
        how="left",
    )
    human_df["observed_var_count"] = human_df[VAR_NAMES].notna().sum(axis=1)

    ordered_cols = SEQUENCE_ID_COLUMNS + ["time_ms", "activity", "observed_var_count"]
    for var_idx, var_name in enumerate(VAR_NAMES):
        human_df[f"Value_{var_idx}"] = human_df[var_name].astype(np.float32)
        human_df[f"Mask_{var_idx}"] = human_df[var_name].notna().astype(np.float32)
        ordered_cols.extend([f"Value_{var_idx}", f"Mask_{var_idx}"])

    return human_df[ordered_cols].sort_values(["sequence", "time_ms"]).reset_index(drop=True)


def build_sequence_metadata(raw_df, grid_ms):
    seq_meta = (
        raw_df.groupby(SEQUENCE_ID_COLUMNS, as_index=False)
        .agg(
            start_datetime=("datetime", "min"),
            end_datetime=("datetime", "max"),
            num_event_rows=("sequence", "size"),
            num_tags=("tag_id", "nunique"),
            num_activities=("activity", "nunique"),
            duration_ms=("rel_time_ms", "max"),
        )
    )
    seq_meta["duration_s"] = seq_meta["duration_ms"] / 1000.0
    seq_meta["grid_ms"] = int(grid_ms)
    seq_meta["num_dense_bins_covered"] = np.floor(seq_meta["duration_ms"] / float(grid_ms)).astype(int) + 1
    return seq_meta.sort_values(["split", "sequence"]).reset_index(drop=True)


def build_metadata(raw_df, point_df, dense_df, sporadic_df, human_activity_df, sequence_df, input_path, output_dir, grid_ms, train_subjects, val_subjects, test_subjects):
    split_sequence_counts = sequence_df.groupby("split")["sequence"].nunique().to_dict()
    split_event_counts = raw_df.groupby("split").size().to_dict()
    split_point_counts = point_df.groupby("split").size().to_dict()
    split_dense_counts = dense_df.groupby("split").size().to_dict()

    metadata = {
        "dataset_name": "Localization Data for Posture Reconstruction",
        "input_file": str(input_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "grid_ms": int(grid_ms),
        "split_subjects": {
            "train": train_subjects,
            "val": val_subjects,
            "test": test_subjects,
        },
        "tag_name_by_id": TAG_NAME_BY_ID,
        "var_names": VAR_NAMES,
        "var_id_by_name": VAR_ID_BY_NAME,
        "num_sequences": int(sequence_df["sequence"].nunique()),
        "num_event_rows": int(len(raw_df)),
        "num_point_rows": int(len(point_df)),
        "num_dense_rows": int(len(dense_df)),
        "num_sporadic_rows": int(len(sporadic_df)),
        "num_human_activity_rows": int(len(human_activity_df)),
        "sequence_counts_by_split": {k: int(v) for k, v in split_sequence_counts.items()},
        "event_counts_by_split": {k: int(v) for k, v in split_event_counts.items()},
        "point_counts_by_split": {k: int(v) for k, v in split_point_counts.items()},
        "dense_counts_by_split": {k: int(v) for k, v in split_dense_counts.items()},
        "activity_counts": {k: int(v) for k, v in raw_df["activity"].value_counts().sort_index().to_dict().items()},
        "subject_event_counts": {k: int(v) for k, v in raw_df["subject"].value_counts().sort_index().to_dict().items()},
        "dense_missing_ratio": float(dense_df[VAR_NAMES].isna().mean().mean()),
    }
    return metadata


def write_split_tables(df, output_dir, stem):
    for split_name in ["train", "val", "test"]:
        split_df = df[df["split"] == split_name]
        split_df.to_csv(output_dir / f"{stem}_{split_name}.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Prepare the human activity posture dataset for irregular and dense experiments.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("./Dataset/Human Activity/ConfLongDemo_JSI.txt"),
        help="Path to ConfLongDemo_JSI.txt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./data_provider/Dataset/human_activity"),
        help="Directory where processed files will be written",
    )
    parser.add_argument(
        "--grid-ms",
        type=int,
        default=100,
        help="Dense baseline resampling grid in milliseconds",
    )
    parser.add_argument("--train-subjects", type=str, default="A,B,C", help="Comma-separated training subject ids")
    parser.add_argument("--val-subjects", type=str, default="D", help="Comma-separated validation subject ids")
    parser.add_argument("--test-subjects", type=str, default="E", help="Comma-separated test subject ids")
    args = parser.parse_args()

    train_subjects = _parse_subject_spec(args.train_subjects)
    val_subjects = _parse_subject_spec(args.val_subjects)
    test_subjects = _parse_subject_spec(args.test_subjects)
    subject_to_split = _make_subject_to_split(train_subjects, val_subjects, test_subjects)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_raw_events(args.input, subject_to_split)
    point_df = build_point_observations(raw_df)
    dense_df = build_dense_grid(point_df, raw_df, grid_ms=args.grid_ms)
    sporadic_df = build_ushcn_style_sporadic(dense_df)
    human_activity_df = build_human_activity_1ms(raw_df)
    sequence_df = build_sequence_metadata(raw_df, grid_ms=args.grid_ms)
    metadata = build_metadata(
        raw_df=raw_df,
        point_df=point_df,
        dense_df=dense_df,
        sporadic_df=sporadic_df,
        human_activity_df=human_activity_df,
        sequence_df=sequence_df,
        input_path=args.input,
        output_dir=output_dir,
        grid_ms=args.grid_ms,
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        test_subjects=test_subjects,
    )

    raw_out = output_dir / "raw_events.csv"
    point_out = output_dir / "point_observations.csv"
    dense_out = output_dir / f"dense_grid_{int(args.grid_ms)}ms.csv"
    sporadic_out = output_dir / "ushcn_style_sporadic.csv"
    human_activity_out = output_dir / "human_activity_1ms.csv"
    seq_out = output_dir / "sequence_metadata.csv"
    metadata_out = output_dir / "metadata.json"

    raw_df.to_csv(raw_out, index=False)
    point_df.to_csv(point_out, index=False)
    dense_df.to_csv(dense_out, index=False)
    sporadic_df.to_csv(sporadic_out, index=False)
    human_activity_df.to_csv(human_activity_out, index=False)
    sequence_df.to_csv(seq_out, index=False)
    with open(metadata_out, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    write_split_tables(raw_df, output_dir, "raw_events")
    write_split_tables(point_df, output_dir, "point_observations")
    write_split_tables(dense_df, output_dir, f"dense_grid_{int(args.grid_ms)}ms")
    write_split_tables(
        sporadic_df.rename(
            columns={
                "Split": "split",
                "Subject": "subject",
                "ID": "sequence",
                "Trial": "trial",
                "Time": "bin_idx",
                "Activity": "activity",
            }
        ),
        output_dir,
        "ushcn_style_sporadic",
    )
    write_split_tables(human_activity_df, output_dir, "human_activity_1ms")

    print(f"input_file={args.input.resolve()}")
    print(f"output_dir={output_dir.resolve()}")
    print(f"num_event_rows={len(raw_df)}")
    print(f"num_point_rows={len(point_df)}")
    print(f"num_dense_rows={len(dense_df)}")
    print(f"num_sporadic_rows={len(sporadic_df)}")
    print(f"num_human_activity_rows={len(human_activity_df)}")
    print(f"dense_missing_ratio={metadata['dense_missing_ratio']:.6f}")
    print(f"splits={metadata['sequence_counts_by_split']}")


if __name__ == "__main__":
    main()
