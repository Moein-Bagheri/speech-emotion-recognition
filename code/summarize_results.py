"""Aggregate experiment JSON files into CSV and Markdown tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ser.config import DOC_DIR, OUTPUT_DIR
from ser.utils import ensure_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=DOC_DIR / "tables",
    )
    return parser.parse_args()


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows]) + "\n"


def main() -> None:
    args = parse_args()
    ensure_directory(args.tables_dir)
    records = []
    for metrics_path in sorted(args.output_dir.glob("*/metrics.json")):
        with metrics_path.open("r", encoding="utf-8") as stream:
            metrics = json.load(stream)
        config = metrics["config"]
        split_name = Path(config["split_csv"]).stem.removeprefix("split_")
        truncation = Path(config["feature_path"]).stem.removeprefix("features_")
        records.append(
            {
                "experiment": metrics["experiment_name"],
                "split": split_name,
                "model": config["architecture"],
                "features": config["feature_mode"],
                "truncation": truncation,
                "masked_GAP": config["masked_pooling"],
                "accuracy": round(metrics["test"]["accuracy"], 4),
                "weighted_F1": round(metrics["test"]["weighted_f1"], 4),
                "macro_F1": round(metrics["test"]["macro_f1"], 4),
                "best_epoch": metrics["best_epoch"],
                "parameters": metrics["num_trainable_parameters"],
            }
        )
    if not records:
        raise FileNotFoundError(
            f"No metrics.json files were found under {args.output_dir}"
        )

    results = pd.DataFrame.from_records(records)
    results.to_csv(
        args.tables_dir / "experiment_results.csv",
        index=False,
        encoding="utf-8",
    )
    (args.tables_dir / "experiment_results.md").write_text(
        markdown_table(results),
        encoding="utf-8",
    )
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
