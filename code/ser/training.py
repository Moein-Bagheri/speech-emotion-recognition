"""Training and evaluation loop for the two SER architectures."""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch import nn
from torch.utils.data import DataLoader
from tqdm import trange

from .config import (
    CHANNELS_BY_FEATURE_MODE,
    ID_TO_LABEL,
    SEED,
)
from .data import SPLIT_ORDER, summarize_split
from .features import (
    IemocapFeatureDataset,
    compute_normalization_statistics,
)
from .models import build_model, count_trainable_parameters
from .plotting import (
    plot_confusion_matrix,
    plot_loss_curves,
    plot_metric_curves,
    plot_per_class_f1,
)
from .utils import (
    ensure_directory,
    read_json,
    resolve_device,
    set_global_seed,
    write_json,
)


@dataclass(frozen=True)
class TrainingConfig:
    """All values needed to reproduce one experiment."""

    experiment_name: str
    metadata_csv: Path
    split_csv: Path
    feature_path: Path
    lengths_path: Path
    output_dir: Path
    architecture: str = "plain"
    feature_mode: str = "mfcc"
    masked_pooling: bool = False
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 30
    seed: int = SEED
    device: str = "auto"
    num_workers: int = 0
    use_amp: bool = True
    class_weighting: str = "none"

    def validate(self) -> None:
        if Path(self.experiment_name).name != self.experiment_name:
            raise ValueError("experiment_name must not contain path separators.")
        if self.architecture not in {"plain", "residual"}:
            raise ValueError("Unknown architecture.")
        if self.feature_mode not in CHANNELS_BY_FEATURE_MODE:
            raise ValueError("Unknown feature mode.")
        if self.class_weighting not in {"none", "balanced"}:
            raise ValueError("class_weighting must be 'none' or 'balanced'.")
        if self.batch_size <= 0 or self.epochs <= 0:
            raise ValueError("batch_size and epochs must be positive.")

    def serializable(self) -> dict[str, Any]:
        data = asdict(self)
        for key, value in list(data.items()):
            if isinstance(value, Path):
                data[key] = str(value)
        return data


def _split_arrays(
    metadata: pd.DataFrame,
    split_frame: pd.DataFrame,
    all_lengths: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    """Map split CSV rows onto the sorted feature-cache indices."""

    if metadata["file"].duplicated().any():
        raise ValueError("Metadata filenames are not unique.")
    file_to_index = {
        str(filename): int(index) for index, filename in enumerate(metadata["file"])
    }
    unknown = sorted(set(split_frame["file"].astype(str)) - set(file_to_index))
    if unknown:
        raise ValueError(f"Split table contains unknown files: {unknown[:5]}")

    arrays: dict[str, dict[str, np.ndarray]] = {}
    for split_name in SPLIT_ORDER:
        subset = split_frame[split_frame["split"] == split_name].copy()
        indices = np.array(
            [file_to_index[str(name)] for name in subset["file"]],
            dtype=np.int64,
        )
        labels = subset["label_id"].to_numpy(dtype=np.int64)
        arrays[split_name] = {
            "indices": indices,
            "labels": labels,
            "lengths": all_lengths[indices].astype(np.int64, copy=False),
            "files": subset["file"].astype(str).to_numpy(),
            "speakers": subset["speaker"].astype(str).to_numpy(),
        }
    return arrays


def _make_loader(
    feature_path: Path,
    arrays: dict[str, np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
    num_channels: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    seed: int,
) -> DataLoader:
    dataset = IemocapFeatureDataset(
        feature_path=feature_path,
        indices=arrays["indices"],
        labels=arrays["labels"],
        valid_lengths=arrays["lengths"],
        mean=mean,
        std=std,
        num_channels=num_channels,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def _classification_metrics(
    true_labels: list[int],
    predictions: list[int],
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(true_labels, predictions)),
        "weighted_f1": float(
            f1_score(
                true_labels,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                true_labels,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
    }


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    amp_enabled: bool,
) -> tuple[dict[str, float], list[int], list[int], list[int]]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_samples = 0
    true_labels: list[int] = []
    predictions: list[int] = []
    global_indices: list[int] = []

    context = torch.enable_grad if training else torch.inference_mode
    with context():
        for batch in loader:
            features = batch["features"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            lengths = batch["length"].to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = model(features, lengths)
                loss = criterion(logits, labels)

            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            batch_size = labels.shape[0]
            total_loss += float(loss.detach().cpu()) * batch_size
            total_samples += batch_size
            true_labels.extend(labels.detach().cpu().tolist())
            predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
            global_indices.extend(batch["index"].tolist())

    metrics = _classification_metrics(true_labels, predictions)
    metrics["loss"] = total_loss / max(total_samples, 1)
    return metrics, true_labels, predictions, global_indices


def _balanced_class_weights(train_labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(train_labels, minlength=len(ID_TO_LABEL))
    if (counts == 0).any():
        raise ValueError("Balanced weights require every class in training.")
    weights = len(train_labels) / (len(ID_TO_LABEL) * counts)
    return weights.astype(np.float32)


def _per_class_report(
    true_labels: list[int],
    predictions: list[int],
) -> dict[str, dict[str, float | int]]:
    label_ids = sorted(ID_TO_LABEL)
    target_names = [ID_TO_LABEL[index] for index in label_ids]
    report = classification_report(
        true_labels,
        predictions,
        labels=label_ids,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    result: dict[str, dict[str, float | int]] = {}
    for name in target_names:
        values = report[name]
        result[name] = {
            "precision": float(values["precision"]),
            "recall": float(values["recall"]),
            "f1": float(values["f1-score"]),
            "support": int(values["support"]),
        }
    return result


def run_training(
    config: TrainingConfig,
    force: bool = False,
) -> dict[str, Any]:
    """Train one configuration, save checkpoints and evaluate the best epoch."""

    config.validate()
    experiment_dir = ensure_directory(config.output_dir / config.experiment_name)
    metrics_path = experiment_dir / "metrics.json"
    if metrics_path.exists() and not force:
        return read_json(metrics_path)

    set_global_seed(config.seed)
    device = resolve_device(config.device)
    amp_enabled = config.use_amp and device.type == "cuda"
    num_channels = CHANNELS_BY_FEATURE_MODE[config.feature_mode]

    metadata = pd.read_csv(config.metadata_csv)
    split_frame = pd.read_csv(config.split_csv)
    all_lengths = np.load(config.lengths_path)
    if len(metadata) != len(all_lengths):
        raise ValueError("Metadata and valid-length arrays do not align.")
    split_arrays = _split_arrays(metadata, split_frame, all_lengths)

    mean, std = compute_normalization_statistics(
        config.feature_path,
        split_arrays["train"]["indices"],
        num_channels=num_channels,
    )
    np.savez(
        experiment_dir / "normalization_stats.npz",
        mean=mean,
        std=std,
    )

    pin_memory = device.type == "cuda"
    loaders = {
        split_name: _make_loader(
            feature_path=config.feature_path,
            arrays=split_arrays[split_name],
            mean=mean,
            std=std,
            num_channels=num_channels,
            batch_size=config.batch_size,
            shuffle=split_name == "train",
            num_workers=config.num_workers,
            pin_memory=pin_memory,
            seed=config.seed,
        )
        for split_name in SPLIT_ORDER
    }

    model = build_model(
        architecture=config.architecture,
        input_channels=num_channels,
        masked_pooling=config.masked_pooling,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
    )
    class_weights: np.ndarray | None = None
    if config.class_weighting == "balanced":
        class_weights = _balanced_class_weights(split_arrays["train"]["labels"])
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, device=device)
        )
    else:
        criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    history_rows: list[dict[str, float | int]] = []
    best_epoch = 0
    best_val_f1 = -1.0
    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    started = time.perf_counter()

    epoch_iterator = trange(
        1,
        config.epochs + 1,
        desc=config.experiment_name,
        unit="epoch",
    )
    for epoch in epoch_iterator:
        train_metrics, _, _, _ = _run_epoch(
            model=model,
            loader=loaders["train"],
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            amp_enabled=amp_enabled,
        )
        val_metrics, _, _, _ = _run_epoch(
            model=model,
            loader=loaders["val"],
            criterion=criterion,
            device=device,
            optimizer=None,
            scaler=scaler,
            amp_enabled=amp_enabled,
        )
        history_rows.append(
            {
                "epoch": epoch,
                **{f"train_{name}": value for name, value in train_metrics.items()},
                **{f"val_{name}": value for name, value in val_metrics.items()},
            }
        )
        epoch_iterator.set_postfix(
            train_loss=f"{train_metrics['loss']:.3f}",
            val_loss=f"{val_metrics['loss']:.3f}",
            val_f1=f"{val_metrics['weighted_f1']:.3f}",
        )

        is_better = val_metrics["weighted_f1"] > best_val_f1 or (
            np.isclose(val_metrics["weighted_f1"], best_val_f1)
            and val_metrics["loss"] < best_val_loss
        )
        if is_better:
            best_epoch = epoch
            best_val_f1 = val_metrics["weighted_f1"]
            best_val_loss = val_metrics["loss"]
            best_state = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": best_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_metrics": val_metrics,
                    "normalization_mean": mean,
                    "normalization_std": std,
                    "config": config.serializable(),
                },
                experiment_dir / "best_checkpoint.pt",
            )

    elapsed_seconds = time.perf_counter() - started
    if best_state is None:
        raise RuntimeError("Training did not produce a best checkpoint.")

    torch.save(
        {
            "epoch": config.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config.serializable(),
        },
        experiment_dir / "last_checkpoint.pt",
    )
    model.load_state_dict(best_state)

    test_metrics, true_labels, predictions, prediction_indices = _run_epoch(
        model=model,
        loader=loaders["test"],
        criterion=criterion,
        device=device,
        optimizer=None,
        scaler=scaler,
        amp_enabled=amp_enabled,
    )
    matrix = confusion_matrix(
        true_labels,
        predictions,
        labels=sorted(ID_TO_LABEL),
    )
    per_class = _per_class_report(true_labels, predictions)

    history = pd.DataFrame(history_rows)
    history.to_csv(
        experiment_dir / "history.csv",
        index=False,
        encoding="utf-8",
    )
    plot_loss_curves(history, experiment_dir / "loss_curve.png")
    plot_metric_curves(history, experiment_dir / "validation_metrics.png")
    plot_confusion_matrix(matrix, experiment_dir / "confusion_matrix.png")
    plot_per_class_f1(per_class, experiment_dir / "per_class_f1.png")

    index_to_file = metadata["file"].astype(str).to_dict()
    index_to_speaker = metadata["speaker"].astype(str).to_dict()
    prediction_frame = pd.DataFrame(
        {
            "feature_index": prediction_indices,
            "file": [index_to_file[index] for index in prediction_indices],
            "speaker": [index_to_speaker[index] for index in prediction_indices],
            "true_id": true_labels,
            "true_label": [ID_TO_LABEL[index] for index in true_labels],
            "predicted_id": predictions,
            "predicted_label": [ID_TO_LABEL[index] for index in predictions],
        }
    )
    prediction_frame["correct"] = (
        prediction_frame["true_id"] == prediction_frame["predicted_id"]
    )
    prediction_frame.to_csv(
        experiment_dir / "test_predictions.csv",
        index=False,
        encoding="utf-8",
    )

    train_counts = np.bincount(
        split_arrays["train"]["labels"],
        minlength=len(ID_TO_LABEL),
    )
    results: dict[str, Any] = {
        "experiment_name": config.experiment_name,
        "config": config.serializable(),
        "device": {
            "type": device.type,
            "name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
            ),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "amp_enabled": amp_enabled,
        },
        "num_trainable_parameters": count_trainable_parameters(model),
        "best_epoch": best_epoch,
        "best_validation_weighted_f1": best_val_f1,
        "best_validation_loss": best_val_loss,
        "training_seconds": elapsed_seconds,
        "test": test_metrics,
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "split_summary": summarize_split(split_frame),
        "training_label_counts": {
            ID_TO_LABEL[index]: int(train_counts[index])
            for index in sorted(ID_TO_LABEL)
        },
        "class_weights": (
            None if class_weights is None else class_weights.astype(float).tolist()
        ),
        "normalization": {
            "computed_from": "training split only",
            "num_channels": num_channels,
        },
    }
    write_json(results, metrics_path)
    return results
