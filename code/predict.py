"""Run a saved best checkpoint on one WAV file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ser.config import CHANNELS_BY_FEATURE_MODE, ID_TO_LABEL
from ser.features import extract_utterance_features
from ser.models import build_model
from ser.utils import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("audio_file", type=Path)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )
    if "normalization_mean" not in checkpoint:
        raise ValueError(
            "Use best_checkpoint.pt; the last checkpoint does not include "
            "normalization statistics."
        )

    config = checkpoint["config"]
    feature_mode = config["feature_mode"]
    num_channels = CHANNELS_BY_FEATURE_MODE[feature_mode]
    architecture = config["architecture"]
    masked_pooling = bool(config["masked_pooling"])
    truncation = "tail" if "features_tail" in str(config["feature_path"]) else "head"

    audio_value = {
        "bytes": args.audio_file.read_bytes(),
        "path": str(args.audio_file),
    }
    head, tail, valid_length, _ = extract_utterance_features(audio_value)
    features = head if truncation == "head" else tail
    features = features[:num_channels]

    mean = np.asarray(
        checkpoint["normalization_mean"],
        dtype=np.float32,
    )[:num_channels, np.newaxis]
    std = np.asarray(
        checkpoint["normalization_std"],
        dtype=np.float32,
    )[:num_channels, np.newaxis]
    features = (features - mean) / (std + 1e-8)

    model = build_model(
        architecture=architecture,
        input_channels=num_channels,
        masked_pooling=masked_pooling,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    inputs = torch.from_numpy(features).unsqueeze(0).to(device)
    lengths = torch.tensor([valid_length], device=device)
    with torch.inference_mode():
        probabilities = model(inputs, lengths).softmax(dim=1)[0].cpu()

    predicted_id = int(probabilities.argmax())
    result = {
        "predicted_id": predicted_id,
        "predicted_label": ID_TO_LABEL[predicted_id],
        "probabilities": {
            ID_TO_LABEL[index]: float(probabilities[index])
            for index in sorted(ID_TO_LABEL)
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
