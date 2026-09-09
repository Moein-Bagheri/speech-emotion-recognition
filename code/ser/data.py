"""Download, inspect and split the IEMOCAP Parquet dataset."""

from __future__ import annotations

import io
import itertools
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import snapshot_download
from sklearn.model_selection import train_test_split

from .config import (
    HF_REPO_ID,
    HF_REPO_TYPE,
    ID_TO_LABEL,
    LABEL_TO_ID,
    RAW_DATA_DIR,
    SEED,
)
from .utils import ensure_directory

SPEAKER_PATTERN = re.compile(r"^(Ses\d{2}[FM])_")
SPLIT_ORDER = ("train", "val", "test")


def download_iemocap(raw_dir: Path = RAW_DATA_DIR) -> list[Path]:
    """Download only the three Parquet shards from Hugging Face.

    ``snapshot_download`` resumes interrupted downloads and validates the
    resulting files against the repository metadata.
    """

    ensure_directory(raw_dir)
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        allow_patterns=["data/*.parquet"],
        local_dir=raw_dir,
    )
    return find_parquet_files(raw_dir)


def find_parquet_files(raw_dir: Path = RAW_DATA_DIR) -> list[Path]:
    """Return all downloaded dataset shards in deterministic order."""

    files = sorted(raw_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No Parquet files were found under {raw_dir}. "
            "Run code/download_data.py first."
        )
    return files


def audio_bytes_from_value(audio_value: Any) -> bytes:
    """Extract encoded audio bytes from an Arrow/Hugging Face audio value."""

    if isinstance(audio_value, (bytes, bytearray, memoryview)):
        return bytes(audio_value)
    if isinstance(audio_value, dict):
        raw = audio_value.get("bytes")
        if raw is not None:
            return bytes(raw)
        path = audio_value.get("path")
        if path and Path(path).is_file():
            return Path(path).read_bytes()
    raise ValueError("The audio field contains neither bytes nor a readable path.")


def speaker_from_filename(filename: str) -> str:
    """Parse one of the ten IEMOCAP speaker IDs (for example ``Ses01F``)."""

    match = SPEAKER_PATTERN.match(filename)
    if not match:
        raise ValueError(f"Cannot parse speaker ID from filename: {filename}")
    return match.group(1)


def iter_dataset_rows(
    parquet_files: Iterable[Path],
    columns: tuple[str, ...] = ("file", "audio", "major_emotion"),
    batch_size: int = 64,
) -> Iterator[tuple[Path, int, dict[str, Any]]]:
    """Yield ``(shard_path, row_number, row)`` without loading all audio."""

    for parquet_path in parquet_files:
        parquet_file = pq.ParquetFile(parquet_path)
        row_offset = 0
        for batch in parquet_file.iter_batches(
            batch_size=batch_size,
            columns=list(columns),
        ):
            rows = batch.to_pylist()
            for local_index, row in enumerate(rows):
                yield parquet_path, row_offset + local_index, row
            row_offset += len(rows)


def collect_metadata(
    parquet_files: Iterable[Path],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Collect metadata and duration statistics for the four requested labels."""

    records: list[dict[str, Any]] = []
    all_label_counts: Counter[str] = Counter()
    total_rows = 0

    for parquet_path, row_index, row in iter_dataset_rows(parquet_files):
        total_rows += 1
        raw_label = str(row["major_emotion"]).strip().lower()
        all_label_counts[raw_label] += 1
        if raw_label not in LABEL_TO_ID:
            continue

        filename = str(row["file"])
        audio_bytes = audio_bytes_from_value(row["audio"])
        audio_info = sf.info(io.BytesIO(audio_bytes))
        duration = float(audio_info.frames / audio_info.samplerate)
        records.append(
            {
                "file": filename,
                "label_name": ID_TO_LABEL[LABEL_TO_ID[raw_label]],
                "label_id": LABEL_TO_ID[raw_label],
                "speaker": speaker_from_filename(filename),
                "duration_sec": duration,
                "sample_rate": int(audio_info.samplerate),
                "num_samples": int(audio_info.frames),
                "source_shard": parquet_path.name,
                "source_row": int(row_index),
            }
        )

    metadata = pd.DataFrame.from_records(records)
    if metadata.empty:
        raise RuntimeError("No samples for the requested four emotions were found.")
    metadata = metadata.sort_values("file", kind="stable").reset_index(drop=True)

    duplicates = metadata.loc[metadata["file"].duplicated(), "file"].tolist()
    if duplicates:
        raise ValueError(f"Duplicate filenames were found: {duplicates[:5]}")

    manifest = {
        "total_rows": total_rows,
        "selected_rows": int(len(metadata)),
        "all_label_counts": dict(sorted(all_label_counts.items())),
        "selected_label_counts": {
            ID_TO_LABEL[label_id]: int((metadata["label_id"] == label_id).sum())
            for label_id in sorted(ID_TO_LABEL)
        },
        "speakers": sorted(metadata["speaker"].unique().tolist()),
        "num_speakers": int(metadata["speaker"].nunique()),
        "duration_seconds": {
            "mean": float(metadata["duration_sec"].mean()),
            "std": float(metadata["duration_sec"].std(ddof=1)),
            "min": float(metadata["duration_sec"].min()),
            "max": float(metadata["duration_sec"].max()),
            "total": float(metadata["duration_sec"].sum()),
        },
    }
    return metadata, manifest


def random_stratified_split(
    metadata: pd.DataFrame,
    seed: int = SEED,
) -> pd.DataFrame:
    """Create the requested 70/10/20 utterance-level random split."""

    all_indices = np.arange(len(metadata))
    train_val_indices, test_indices = train_test_split(
        all_indices,
        test_size=0.20,
        random_state=seed,
        stratify=metadata["label_id"].to_numpy(),
    )
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=0.125,
        random_state=seed,
        stratify=metadata.iloc[train_val_indices]["label_id"].to_numpy(),
    )

    assignments = np.empty(len(metadata), dtype=object)
    assignments[train_indices] = "train"
    assignments[val_indices] = "val"
    assignments[test_indices] = "test"
    result = metadata.copy()
    result["split"] = assignments
    validate_partition(result)
    return result


def _distribution_distance(
    subset: pd.DataFrame,
    global_distribution: np.ndarray,
) -> tuple[int, float]:
    counts = (
        subset["label_id"]
        .value_counts()
        .reindex(sorted(ID_TO_LABEL), fill_value=0)
        .to_numpy(dtype=float)
    )
    missing = int((counts == 0).sum())
    distribution = counts / max(counts.sum(), 1.0)
    distance = float(np.abs(distribution - global_distribution).sum())
    return missing, distance


def speaker_disjoint_split(metadata: pd.DataFrame) -> pd.DataFrame:
    """Find a deterministic 7/1/2-speaker train/val/test allocation.

    IEMOCAP has ten speakers.  Every possible choice of two test speakers and
    one validation speaker is scored by sample-ratio error and label-
    distribution error.  Missing labels receive a large penalty.  This keeps
    the requested approximate 70/10/20 ratios while guaranteeing that a
    speaker appears in exactly one partition.
    """

    speakers = sorted(metadata["speaker"].unique().tolist())
    if len(speakers) < 4:
        raise ValueError("At least four speakers are required for a group split.")

    num_test_speakers = max(1, round(len(speakers) * 0.20))
    num_val_speakers = max(1, round(len(speakers) * 0.10))
    if num_test_speakers + num_val_speakers >= len(speakers):
        raise ValueError("Not enough speakers remain for the training split.")

    global_counts = (
        metadata["label_id"]
        .value_counts()
        .reindex(sorted(ID_TO_LABEL), fill_value=0)
        .to_numpy(dtype=float)
    )
    global_distribution = global_counts / global_counts.sum()
    target_ratios = {"train": 0.70, "val": 0.10, "test": 0.20}

    best_key: tuple[float, ...] | None = None
    best_groups: dict[str, tuple[str, ...]] | None = None

    for test_speakers in itertools.combinations(speakers, num_test_speakers):
        remaining = [s for s in speakers if s not in test_speakers]
        for val_speakers in itertools.combinations(
            remaining,
            num_val_speakers,
        ):
            train_speakers = tuple(s for s in remaining if s not in val_speakers)
            groups = {
                "train": train_speakers,
                "val": tuple(val_speakers),
                "test": tuple(test_speakers),
            }

            missing_labels = 0
            distribution_error = 0.0
            ratio_error = 0.0
            for split_name, split_speakers in groups.items():
                subset = metadata[metadata["speaker"].isin(split_speakers)]
                missing, distance = _distribution_distance(
                    subset,
                    global_distribution,
                )
                missing_labels += missing
                distribution_error += distance
                actual_ratio = len(subset) / len(metadata)
                ratio_error += abs(actual_ratio - target_ratios[split_name])

            # The string signature makes ties reproducible.
            signature = "|".join(",".join(groups[name]) for name in SPLIT_ORDER)
            key = (
                float(missing_labels),
                ratio_error * 3.0 + distribution_error,
                ratio_error,
                distribution_error,
                float(sum(ord(char) for char in signature)),
            )
            if best_key is None or key < best_key:
                best_key = key
                best_groups = groups

    if best_groups is None:
        raise RuntimeError("No valid speaker-disjoint allocation was found.")

    speaker_to_split = {
        speaker: split_name
        for split_name, group in best_groups.items()
        for speaker in group
    }
    result = metadata.copy()
    result["split"] = result["speaker"].map(speaker_to_split)
    validate_partition(result)
    validate_speaker_disjoint(result)
    return result


def validate_partition(split_frame: pd.DataFrame) -> None:
    """Ensure every row is assigned exactly once to train, val or test."""

    if split_frame["split"].isna().any():
        raise ValueError("At least one sample has no split assignment.")
    observed = set(split_frame["split"].unique())
    if observed != set(SPLIT_ORDER):
        raise ValueError(f"Unexpected split names: {sorted(observed)}")
    if split_frame["file"].duplicated().any():
        raise ValueError("A filename occurs more than once in the split table.")


def validate_speaker_disjoint(split_frame: pd.DataFrame) -> None:
    """Raise if any speaker occurs in more than one partition."""

    speaker_split_counts = split_frame.groupby("speaker")["split"].nunique()
    leaking = speaker_split_counts[speaker_split_counts > 1]
    if not leaking.empty:
        raise ValueError(
            "Speaker leakage detected for: " + ", ".join(leaking.index.astype(str))
        )


def summarize_split(split_frame: pd.DataFrame) -> dict[str, Any]:
    """Return counts, ratios, speakers and per-label counts for a split."""

    validate_partition(split_frame)
    summary: dict[str, Any] = {}
    for split_name in SPLIT_ORDER:
        subset = split_frame[split_frame["split"] == split_name]
        summary[split_name] = {
            "num_samples": int(len(subset)),
            "ratio": float(len(subset) / len(split_frame)),
            "speakers": sorted(subset["speaker"].unique().tolist()),
            "label_counts": {
                ID_TO_LABEL[label_id]: int((subset["label_id"] == label_id).sum())
                for label_id in sorted(ID_TO_LABEL)
            },
        }
    return summary
