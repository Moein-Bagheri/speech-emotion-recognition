"""Plotting helpers used by data preparation and model evaluation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import ID_TO_LABEL
from .utils import ensure_directory

plt.switch_backend("Agg")

COLORS = ["#c84b31", "#f2b134", "#4a90a4", "#596275"]


def _finish_figure(path: Path) -> None:
    ensure_directory(path.parent)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_label_distribution(metadata: pd.DataFrame, path: Path) -> None:
    """Draw the requested bar chart for the four selected labels."""

    counts = (
        metadata["label_name"]
        .value_counts()
        .reindex([ID_TO_LABEL[index] for index in sorted(ID_TO_LABEL)])
    )
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(7.2, 4.6))
    axis = sns.barplot(
        x=counts.index,
        y=counts.values,
        hue=counts.index,
        palette=COLORS,
        legend=False,
    )
    for patch, count in zip(axis.patches, counts.values, strict=True):
        axis.annotate(
            f"{int(count):,}",
            (patch.get_x() + patch.get_width() / 2, patch.get_height()),
            ha="center",
            va="bottom",
            xytext=(0, 5),
            textcoords="offset points",
        )
    axis.set_title("IEMOCAP label distribution (selected emotions)")
    axis.set_xlabel("Emotion")
    axis.set_ylabel("Number of utterances")
    _finish_figure(path)


def plot_duration_distribution(metadata: pd.DataFrame, path: Path) -> None:
    """Draw the requested utterance-duration frequency distribution."""

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(7.2, 4.6))
    axis = sns.histplot(
        metadata,
        x="duration_sec",
        hue="label_name",
        hue_order=[ID_TO_LABEL[index] for index in sorted(ID_TO_LABEL)],
        bins=45,
        element="step",
        stat="count",
        common_norm=False,
        palette=COLORS,
        alpha=0.30,
    )
    axis.axvline(
        metadata["duration_sec"].mean(),
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Overall mean",
    )
    axis.set_title("Distribution of utterance duration")
    axis.set_xlabel("Duration (seconds)")
    axis.set_ylabel("Frequency")
    _finish_figure(path)


def plot_duration_boxplot(metadata: pd.DataFrame, path: Path) -> None:
    """Draw a supplementary per-label duration comparison."""

    order = [ID_TO_LABEL[index] for index in sorted(ID_TO_LABEL)]
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(7.2, 4.6))
    axis = sns.boxplot(
        data=metadata,
        x="label_name",
        y="duration_sec",
        order=order,
        hue="label_name",
        palette=COLORS,
        legend=False,
        showfliers=False,
    )
    axis.set_title("Utterance duration by emotion")
    axis.set_xlabel("Emotion")
    axis.set_ylabel("Duration (seconds)")
    _finish_figure(path)


def plot_loss_curves(history: pd.DataFrame, path: Path) -> None:
    """Plot train and validation loss over all epochs."""

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(7.2, 4.6))
    plt.plot(
        history["epoch"],
        history["train_loss"],
        label="Train loss",
        linewidth=2,
    )
    plt.plot(
        history["epoch"],
        history["val_loss"],
        label="Validation loss",
        linewidth=2,
    )
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Training and validation loss")
    plt.legend()
    _finish_figure(path)


def plot_metric_curves(history: pd.DataFrame, path: Path) -> None:
    """Plot validation accuracy and both F1 variants."""

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(7.2, 4.6))
    plt.plot(
        history["epoch"],
        history["val_accuracy"],
        label="Validation accuracy",
        linewidth=2,
    )
    plt.plot(
        history["epoch"],
        history["val_weighted_f1"],
        label="Validation weighted F1",
        linewidth=2,
    )
    plt.plot(
        history["epoch"],
        history["val_macro_f1"],
        label="Validation macro F1",
        linewidth=2,
        linestyle="--",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.0)
    plt.title("Validation metrics")
    plt.legend()
    _finish_figure(path)


def plot_confusion_matrix(matrix: np.ndarray, path: Path) -> None:
    """Plot a four-class test confusion matrix."""

    labels = [ID_TO_LABEL[index] for index in sorted(ID_TO_LABEL)]
    sns.set_theme(style="white")
    plt.figure(figsize=(6.2, 5.4))
    axis = sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
    )
    axis.set_title("Test confusion matrix")
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    _finish_figure(path)


def plot_per_class_f1(per_class: dict[str, dict[str, float]], path: Path) -> None:
    """Draw per-class F1 scores to make minority-class failures visible."""

    labels = [ID_TO_LABEL[index] for index in sorted(ID_TO_LABEL)]
    values = [float(per_class[label]["f1"]) for label in labels]
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(7.2, 4.6))
    axis = sns.barplot(
        x=labels,
        y=values,
        hue=labels,
        palette=COLORS,
        legend=False,
    )
    for patch, score in zip(axis.patches, values, strict=True):
        axis.annotate(
            f"{score:.3f}",
            (patch.get_x() + patch.get_width() / 2, patch.get_height()),
            ha="center",
            va="bottom",
            xytext=(0, 4),
            textcoords="offset points",
        )
    axis.set_ylim(0.0, 1.0)
    axis.set_title("Test F1 by emotion")
    axis.set_xlabel("Emotion")
    axis.set_ylabel("F1 score")
    _finish_figure(path)
