"""Extract and cache MFCC, delta and delta-delta matrices."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from ser.config import MAX_LEN, PROCESSED_DATA_DIR, RAW_DATA_DIR
from ser.data import find_parquet_files
from ser.features import extract_feature_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROCESSED_DATA_DIR,
    )
    parser.add_argument("--max-len", type=int, default=MAX_LEN)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata_path = args.processed_dir / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"{metadata_path} is missing. Run code/prepare_data.py first."
        )
    metadata = pd.read_csv(metadata_path)
    parquet_files = find_parquet_files(args.raw_dir)
    result = extract_feature_cache(
        parquet_files=parquet_files,
        metadata=metadata,
        output_dir=args.processed_dir,
        max_len=args.max_len,
        workers=args.workers,
        force=args.force,
    )
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
