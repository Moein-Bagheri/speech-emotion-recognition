"""Create metadata, EDA figures and both requested dataset partitions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ser.config import (
    DOC_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    SEED,
)
from ser.data import (
    collect_metadata,
    download_iemocap,
    find_parquet_files,
    random_stratified_split,
    speaker_disjoint_split,
    summarize_split,
)
from ser.plotting import (
    plot_duration_boxplot,
    plot_duration_distribution,
    plot_label_distribution,
)
from ser.utils import ensure_directory, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROCESSED_DATA_DIR,
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=DOC_DIR / "figures" / "eda",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=DOC_DIR / "tables",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--download-if-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directory(args.processed_dir)
    ensure_directory(args.figures_dir)
    ensure_directory(args.tables_dir)

    try:
        parquet_files = find_parquet_files(args.raw_dir)
    except FileNotFoundError:
        if not args.download_if_missing:
            raise
        parquet_files = download_iemocap(args.raw_dir)

    metadata, dataset_manifest = collect_metadata(parquet_files)
    random_split = random_stratified_split(metadata, seed=args.seed)
    speaker_split = speaker_disjoint_split(metadata)

    metadata.to_csv(
        args.processed_dir / "metadata.csv",
        index=False,
        encoding="utf-8",
    )
    random_split.to_csv(
        args.processed_dir / "split_random.csv",
        index=False,
        encoding="utf-8",
    )
    speaker_split.to_csv(
        args.processed_dir / "split_speaker.csv",
        index=False,
        encoding="utf-8",
    )

    plot_label_distribution(
        metadata,
        args.figures_dir / "label_distribution.png",
    )
    plot_duration_distribution(
        metadata,
        args.figures_dir / "duration_distribution.png",
    )
    plot_duration_boxplot(
        metadata,
        args.figures_dir / "duration_by_label.png",
    )

    label_counts = (
        metadata.groupby(["label_id", "label_name"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    label_counts["percentage"] = 100.0 * label_counts["count"] / len(metadata)
    label_counts.to_csv(
        args.tables_dir / "label_counts.csv",
        index=False,
        encoding="utf-8",
    )

    duration_statistics = pd.DataFrame(
        [
            {
                "mean_seconds": metadata["duration_sec"].mean(),
                "std_seconds": metadata["duration_sec"].std(ddof=1),
                "min_seconds": metadata["duration_sec"].min(),
                "max_seconds": metadata["duration_sec"].max(),
                "total_hours": metadata["duration_sec"].sum() / 3600.0,
            }
        ]
    )
    duration_statistics.to_csv(
        args.tables_dir / "duration_statistics.csv",
        index=False,
        encoding="utf-8",
    )

    split_report = {
        "dataset": dataset_manifest,
        "random": summarize_split(random_split),
        "speaker_disjoint": summarize_split(speaker_split),
        "random_speaker_overlap": {
            split_name: sorted(
                random_split.loc[
                    random_split["split"] == split_name,
                    "speaker",
                ]
                .unique()
                .tolist()
            )
            for split_name in ("train", "val", "test")
        },
    }
    write_json(
        split_report,
        args.processed_dir / "dataset_summary.json",
    )
    write_json(
        split_report,
        args.tables_dir / "dataset_summary.json",
    )

    print(f"Selected utterances: {len(metadata):,}")
    print(f"Speakers: {metadata['speaker'].nunique()}")
    print("Random split:", summarize_split(random_split))
    print("Speaker-disjoint split:", summarize_split(speaker_split))


if __name__ == "__main__":
    main()
