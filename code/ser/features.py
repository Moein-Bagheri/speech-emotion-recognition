"""MFCC extraction, fixed-length handling and normalized PyTorch datasets."""

from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .config import HOP_LENGTH, MAX_LEN, N_MFCC, SAMPLE_RATE
from .data import audio_bytes_from_value, iter_dataset_rows
from .utils import ensure_directory, write_json


def decode_audio(audio_value: Any, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Decode an Arrow audio field, convert to mono and resample to 16 kHz."""

    audio_bytes = audio_bytes_from_value(audio_value)
    waveform, original_sr = sf.read(
        io.BytesIO(audio_bytes),
        dtype="float32",
        always_2d=False,
    )
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1, dtype=np.float32)
    if original_sr != target_sr:
        waveform = librosa.resample(
            waveform,
            orig_sr=int(original_sr),
            target_sr=target_sr,
        ).astype(np.float32, copy=False)
    if waveform.size == 0:
        raise ValueError("Decoded waveform is empty.")
    return waveform


def _safe_deltas(mfcc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute first and second deltas, including for unusually short files."""

    num_frames = mfcc.shape[1]
    width = min(9, num_frames if num_frames % 2 else num_frames - 1)
    if width < 3:
        zeros = np.zeros_like(mfcc, dtype=np.float32)
        return zeros, zeros.copy()
    delta = librosa.feature.delta(
        mfcc,
        order=1,
        width=width,
        mode="interp",
    )
    delta_delta = librosa.feature.delta(
        mfcc,
        order=2,
        width=width,
        mode="interp",
    )
    return (
        delta.astype(np.float32, copy=False),
        delta_delta.astype(np.float32, copy=False),
    )


def pad_or_truncate(
    features: np.ndarray,
    max_len: int = MAX_LEN,
    keep: str = "head",
) -> tuple[np.ndarray, int]:
    """Make a ``(channels, max_len)`` matrix using zero padding.

    ``keep='head'`` implements the assignment's baseline (retain the start);
    ``keep='tail'`` implements the requested comparison (retain the end).
    """

    if keep not in {"head", "tail"}:
        raise ValueError("keep must be either 'head' or 'tail'.")
    if features.ndim != 2:
        raise ValueError("features must have shape (channels, time).")

    num_frames = features.shape[1]
    valid_length = min(num_frames, max_len)
    output = np.zeros((features.shape[0], max_len), dtype=np.float32)
    if num_frames >= max_len:
        selected = features[:, :max_len] if keep == "head" else features[:, -max_len:]
        output[...] = selected
    else:
        output[:, :num_frames] = features
    return output, valid_length


def extract_utterance_features(
    audio_value: Any,
    max_len: int = MAX_LEN,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Extract MFCC, delta and delta-delta and return head/tail variants."""

    waveform = decode_audio(audio_value)
    mfcc = librosa.feature.mfcc(
        y=waveform,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        hop_length=HOP_LENGTH,
    ).astype(np.float32, copy=False)
    delta, delta_delta = _safe_deltas(mfcc)
    features = np.concatenate((mfcc, delta, delta_delta), axis=0)
    if not np.isfinite(features).all():
        raise FloatingPointError("Feature extraction produced NaN or infinity.")

    head, valid_length = pad_or_truncate(
        features,
        max_len=max_len,
        keep="head",
    )
    tail, _ = pad_or_truncate(
        features,
        max_len=max_len,
        keep="tail",
    )
    return head, tail, valid_length, int(features.shape[1])


def extract_feature_cache(
    parquet_files: Iterable[Path],
    metadata: pd.DataFrame,
    output_dir: Path,
    max_len: int = MAX_LEN,
    workers: int = 1,
    force: bool = False,
) -> dict[str, Any]:
    """Extract all selected utterances into memory-mappable NumPy arrays."""

    ensure_directory(output_dir)
    head_path = output_dir / "features_head.npy"
    tail_path = output_dir / "features_tail.npy"
    lengths_path = output_dir / "valid_lengths.npy"
    original_lengths_path = output_dir / "original_lengths.npy"
    manifest_path = output_dir / "feature_manifest.json"

    expected_paths = (
        head_path,
        tail_path,
        lengths_path,
        original_lengths_path,
        manifest_path,
    )
    if all(path.exists() for path in expected_paths) and not force:
        return {
            "head_path": str(head_path),
            "tail_path": str(tail_path),
            "lengths_path": str(lengths_path),
            "original_lengths_path": str(original_lengths_path),
            "manifest_path": str(manifest_path),
            "reused": True,
        }

    if metadata["file"].duplicated().any():
        raise ValueError("Metadata filenames must be unique.")
    file_to_index = {
        filename: int(index)
        for index, filename in enumerate(metadata["file"].astype(str))
    }
    selected_files = set(file_to_index)
    num_samples = len(metadata)

    head_cache = np.lib.format.open_memmap(
        head_path,
        mode="w+",
        dtype=np.float32,
        shape=(num_samples, N_MFCC * 3, max_len),
    )
    tail_cache = np.lib.format.open_memmap(
        tail_path,
        mode="w+",
        dtype=np.float32,
        shape=(num_samples, N_MFCC * 3, max_len),
    )
    valid_lengths = np.zeros(num_samples, dtype=np.int32)
    original_lengths = np.zeros(num_samples, dtype=np.int32)
    found = np.zeros(num_samples, dtype=bool)

    def compute_one(
        item: tuple[str, Any],
    ) -> tuple[str, np.ndarray, np.ndarray, int, int]:
        filename, audio_value = item
        try:
            head, tail, valid, original = extract_utterance_features(
                audio_value,
                max_len=max_len,
            )
        except Exception as error:
            raise RuntimeError(f"Feature extraction failed for {filename}") from error
        return filename, head, tail, valid, original

    def store_results(
        items: list[tuple[str, Any]],
        executor: ThreadPoolExecutor | None,
        progress: tqdm,
    ) -> None:
        iterator = (
            executor.map(compute_one, items)
            if executor is not None
            else map(compute_one, items)
        )
        for filename, head, tail, valid, original in iterator:
            index = file_to_index[filename]
            if found[index]:
                raise ValueError(f"Duplicate audio row for {filename}")
            head_cache[index] = head
            tail_cache[index] = tail
            valid_lengths[index] = valid
            original_lengths[index] = original
            found[index] = True
            progress.update(1)

    buffer: list[tuple[str, Any]] = []
    buffer_size = max(1, workers) * 4
    executor = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        with tqdm(
            total=num_samples,
            desc="Extracting MFCC features",
            unit="utterance",
        ) as progress:
            for _, _, row in iter_dataset_rows(parquet_files):
                filename = str(row["file"])
                if filename not in selected_files:
                    continue
                buffer.append((filename, row["audio"]))
                if len(buffer) >= buffer_size:
                    store_results(buffer, executor, progress)
                    buffer = []
            if buffer:
                store_results(buffer, executor, progress)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        head_cache.flush()
        tail_cache.flush()

    missing_files = metadata.loc[~found, "file"].astype(str).tolist()
    if missing_files:
        raise RuntimeError(
            f"{len(missing_files)} selected audio files were not found; "
            f"examples: {missing_files[:5]}"
        )

    np.save(lengths_path, valid_lengths)
    np.save(original_lengths_path, original_lengths)
    feature_index = metadata[["file", "label_id", "label_name", "speaker"]].copy()
    feature_index.insert(0, "feature_index", np.arange(num_samples))
    feature_index.to_csv(
        output_dir / "feature_index.csv",
        index=False,
        encoding="utf-8",
    )

    manifest = {
        "num_samples": num_samples,
        "shape": [num_samples, N_MFCC * 3, max_len],
        "dtype": "float32",
        "sample_rate": SAMPLE_RATE,
        "n_mfcc": N_MFCC,
        "hop_length": HOP_LENGTH,
        "max_len": max_len,
        "channel_layout": {
            "0:40": "MFCC",
            "40:80": "delta",
            "80:120": "delta-delta",
        },
        "truncated_utterances": int((original_lengths > max_len).sum()),
        "padded_utterances": int((original_lengths < max_len).sum()),
        "exact_length_utterances": int((original_lengths == max_len).sum()),
    }
    write_json(manifest, manifest_path)
    return {
        "head_path": str(head_path),
        "tail_path": str(tail_path),
        "lengths_path": str(lengths_path),
        "original_lengths_path": str(original_lengths_path),
        "manifest_path": str(manifest_path),
        "reused": False,
    }


def compute_normalization_statistics(
    feature_path: Path,
    train_indices: np.ndarray,
    num_channels: int,
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-channel mean/std on training samples only.

    Time positions, including baseline zero-padding, are included exactly as in
    the formula supplied in the assignment.
    """

    features = np.load(feature_path, mmap_mode="r")
    if features.ndim != 3 or num_channels > features.shape[1]:
        raise ValueError("Feature cache has an unexpected shape.")

    channel_sum = np.zeros(num_channels, dtype=np.float64)
    channel_squared_sum = np.zeros(num_channels, dtype=np.float64)
    count = 0
    for start in range(0, len(train_indices), batch_size):
        indices = train_indices[start : start + batch_size]
        batch = np.asarray(
            features[indices, :num_channels, :],
            dtype=np.float64,
        )
        channel_sum += batch.sum(axis=(0, 2))
        channel_squared_sum += np.square(batch).sum(axis=(0, 2))
        count += batch.shape[0] * batch.shape[2]

    mean = channel_sum / count
    variance = channel_squared_sum / count - np.square(mean)
    variance = np.maximum(variance, 0.0)
    std = np.sqrt(variance)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


class IemocapFeatureDataset(Dataset):
    """A memory-efficient normalized view over a feature cache."""

    def __init__(
        self,
        feature_path: Path,
        indices: np.ndarray,
        labels: np.ndarray,
        valid_lengths: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        num_channels: int,
    ) -> None:
        self.features = np.load(feature_path, mmap_mode="r")
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.valid_lengths = np.asarray(valid_lengths, dtype=np.int64)
        self.mean = mean[:num_channels, np.newaxis]
        self.std = std[:num_channels, np.newaxis]
        self.num_channels = num_channels

        if not (len(self.indices) == len(self.labels) == len(self.valid_lengths)):
            raise ValueError("Indices, labels and lengths must align.")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        global_index = int(self.indices[item])
        matrix = np.array(
            self.features[global_index, : self.num_channels, :],
            dtype=np.float32,
            copy=True,
        )
        matrix = (matrix - self.mean) / (self.std + 1e-8)
        return {
            "features": torch.from_numpy(matrix),
            "label": torch.tensor(self.labels[item], dtype=torch.long),
            "length": torch.tensor(
                self.valid_lengths[item],
                dtype=torch.long,
            ),
            "index": torch.tensor(global_index, dtype=torch.long),
        }
