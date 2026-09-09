"""Download the IEMOCAP Parquet shards referenced in the assignment."""

from __future__ import annotations

import argparse
from pathlib import Path

from ser.config import RAW_DATA_DIR
from ser.data import download_iemocap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DATA_DIR,
        help="Directory that receives the Hugging Face snapshot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = download_iemocap(args.raw_dir)
    total_bytes = sum(path.stat().st_size for path in files)
    print(f"Downloaded/verified {len(files)} Parquet shards.")
    print(f"Total size: {total_bytes / 1024**3:.2f} GiB")
    for path in files:
        print(path)


if __name__ == "__main__":
    main()
