"""Run the required comparisons and optional ablations sequentially."""

from __future__ import annotations

import argparse
from pathlib import Path

from ser.config import OUTPUT_DIR, PROCESSED_DATA_DIR
from ser.training import TrainingConfig, run_training

CORE_EXPERIMENTS = (
    {
        "experiment_name": "random_plain_mfcc_head",
        "split": "random",
        "architecture": "plain",
        "feature_mode": "mfcc",
        "truncation": "head",
        "masked_pooling": False,
    },
    {
        "experiment_name": "speaker_plain_mfcc_head",
        "split": "speaker",
        "architecture": "plain",
        "feature_mode": "mfcc",
        "truncation": "head",
        "masked_pooling": False,
    },
    {
        "experiment_name": "speaker_residual_mfcc_head",
        "split": "speaker",
        "architecture": "residual",
        "feature_mode": "mfcc",
        "truncation": "head",
        "masked_pooling": False,
    },
    {
        "experiment_name": "speaker_plain_mfcc_tail",
        "split": "speaker",
        "architecture": "plain",
        "feature_mode": "mfcc",
        "truncation": "tail",
        "masked_pooling": False,
    },
)

OPTIONAL_EXPERIMENTS = (
    {
        "experiment_name": "speaker_plain_mfcc_delta2_head",
        "split": "speaker",
        "architecture": "plain",
        "feature_mode": "mfcc_delta2",
        "truncation": "head",
        "masked_pooling": False,
    },
    {
        "experiment_name": "speaker_plain_mfcc_head_masked",
        "split": "speaker",
        "architecture": "plain",
        "feature_mode": "mfcc",
        "truncation": "head",
        "masked_pooling": True,
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-optional", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiments = list(CORE_EXPERIMENTS)
    if args.include_optional:
        experiments.extend(OPTIONAL_EXPERIMENTS)

    for specification in experiments:
        config = TrainingConfig(
            experiment_name=specification["experiment_name"],
            metadata_csv=args.processed_dir / "metadata.csv",
            split_csv=(args.processed_dir / f"split_{specification['split']}.csv"),
            feature_path=(
                args.processed_dir / f"features_{specification['truncation']}.npy"
            ),
            lengths_path=args.processed_dir / "valid_lengths.npy",
            output_dir=args.output_dir,
            architecture=specification["architecture"],
            feature_mode=specification["feature_mode"],
            masked_pooling=specification["masked_pooling"],
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            seed=args.seed,
            device=args.device,
        )
        result = run_training(config, force=args.force)
        print(
            f"{config.experiment_name}: "
            f"Acc={result['test']['accuracy']:.4f}, "
            f"weighted F1={result['test']['weighted_f1']:.4f}"
        )


if __name__ == "__main__":
    main()
