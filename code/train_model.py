"""Train one configurable 1D-CNN experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from ser.config import OUTPUT_DIR, PROCESSED_DATA_DIR
from ser.training import TrainingConfig, run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--architecture",
        choices=("plain", "residual"),
        default="plain",
    )
    parser.add_argument(
        "--feature-mode",
        choices=("mfcc", "mfcc_delta", "mfcc_delta2"),
        default="mfcc",
    )
    parser.add_argument(
        "--split-strategy",
        choices=("random", "speaker"),
        default="speaker",
    )
    parser.add_argument(
        "--truncation",
        choices=("head", "tail"),
        default="head",
    )
    parser.add_argument("--masked-pooling", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--class-weighting",
        choices=("none", "balanced"),
        default="none",
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--experiment-name", type=str)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_name = args.experiment_name
    if not experiment_name:
        parts = [
            args.split_strategy,
            args.architecture,
            args.feature_mode,
            args.truncation,
        ]
        if args.masked_pooling:
            parts.append("masked")
        experiment_name = "_".join(parts)

    config = TrainingConfig(
        experiment_name=experiment_name,
        metadata_csv=args.processed_dir / "metadata.csv",
        split_csv=args.processed_dir / f"split_{args.split_strategy}.csv",
        feature_path=args.processed_dir / f"features_{args.truncation}.npy",
        lengths_path=args.processed_dir / "valid_lengths.npy",
        output_dir=args.output_dir,
        architecture=args.architecture,
        feature_mode=args.feature_mode,
        masked_pooling=args.masked_pooling,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        use_amp=not args.no_amp,
        class_weighting=args.class_weighting,
    )
    result = run_training(config, force=args.force)
    print(
        f"{experiment_name}: "
        f"accuracy={result['test']['accuracy']:.4f}, "
        f"weighted_f1={result['test']['weighted_f1']:.4f}, "
        f"macro_f1={result['test']['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
