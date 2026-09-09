from __future__ import annotations

import numpy as np
import pandas as pd

from ser.training import TrainingConfig, run_training


def test_one_epoch_training_writes_required_artifacts(tmp_path) -> None:
    rng = np.random.default_rng(7)
    num_samples = 40
    features = rng.normal(size=(num_samples, 120, 32)).astype(np.float32)
    labels = np.tile(np.arange(4), num_samples // 4)
    speakers = [f"Ses{index % 5 + 1:02d}F" for index in range(num_samples)]
    files = [
        f"{speakers[index]}_sample_{index:03d}.wav" for index in range(num_samples)
    ]

    metadata = pd.DataFrame(
        {
            "file": files,
            "label_id": labels,
            "label_name": ["Angry", "Happy", "Neutral", "Sad"] * 10,
            "speaker": speakers,
        }
    )
    split_names = np.array(["train"] * 24 + ["val"] * 8 + ["test"] * 8)
    split_frame = metadata.copy()
    split_frame["split"] = split_names

    metadata_path = tmp_path / "metadata.csv"
    split_path = tmp_path / "split.csv"
    feature_path = tmp_path / "features.npy"
    lengths_path = tmp_path / "lengths.npy"
    metadata.to_csv(metadata_path, index=False)
    split_frame.to_csv(split_path, index=False)
    np.save(feature_path, features)
    np.save(lengths_path, np.full(num_samples, 32, dtype=np.int32))

    config = TrainingConfig(
        experiment_name="smoke_test",
        metadata_csv=metadata_path,
        split_csv=split_path,
        feature_path=feature_path,
        lengths_path=lengths_path,
        output_dir=tmp_path / "outputs",
        epochs=1,
        batch_size=8,
        device="cpu",
        use_amp=False,
    )
    result = run_training(config)
    experiment_dir = tmp_path / "outputs" / "smoke_test"

    assert 0.0 <= result["test"]["accuracy"] <= 1.0
    assert (experiment_dir / "best_checkpoint.pt").is_file()
    assert (experiment_dir / "last_checkpoint.pt").is_file()
    assert (experiment_dir / "loss_curve.png").is_file()
    assert (experiment_dir / "confusion_matrix.png").is_file()
    assert (experiment_dir / "test_predictions.csv").is_file()
