from __future__ import annotations

import numpy as np

from ser.features import (
    compute_normalization_statistics,
    pad_or_truncate,
)


def test_head_and_tail_truncation_are_distinct() -> None:
    features = np.arange(12, dtype=np.float32).reshape(2, 6)

    head, head_length = pad_or_truncate(features, max_len=4, keep="head")
    tail, tail_length = pad_or_truncate(features, max_len=4, keep="tail")

    np.testing.assert_array_equal(head, features[:, :4])
    np.testing.assert_array_equal(tail, features[:, -4:])
    assert head_length == tail_length == 4


def test_short_sequence_is_zero_padded_at_end() -> None:
    features = np.ones((3, 2), dtype=np.float32)
    padded, valid_length = pad_or_truncate(
        features,
        max_len=5,
        keep="head",
    )

    np.testing.assert_array_equal(padded[:, :2], 1.0)
    np.testing.assert_array_equal(padded[:, 2:], 0.0)
    assert valid_length == 2


def test_statistics_use_training_indices_only(tmp_path) -> None:
    feature_path = tmp_path / "features.npy"
    features = np.zeros((3, 2, 4), dtype=np.float32)
    features[0] = 1.0
    features[1] = 3.0
    features[2] = 1000.0
    np.save(feature_path, features)

    mean, std = compute_normalization_statistics(
        feature_path,
        train_indices=np.array([0, 1]),
        num_channels=2,
    )

    np.testing.assert_allclose(mean, [2.0, 2.0])
    np.testing.assert_allclose(std, [1.0, 1.0])
