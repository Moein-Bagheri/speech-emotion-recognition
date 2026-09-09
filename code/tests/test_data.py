from __future__ import annotations

import pandas as pd

from ser.data import (
    speaker_disjoint_split,
    validate_speaker_disjoint,
)


def test_speaker_disjoint_split_has_7_1_2_speakers() -> None:
    rows = []
    for speaker_index in range(10):
        speaker = f"Ses{speaker_index // 2 + 1:02d}"
        speaker += "F" if speaker_index % 2 == 0 else "M"
        for label_id, label_name in enumerate(("Angry", "Happy", "Neutral", "Sad")):
            for repetition in range(label_id + 1):
                rows.append(
                    {
                        "file": (
                            f"{speaker}_synthetic_" f"{label_id}_{repetition}.wav"
                        ),
                        "label_id": label_id,
                        "label_name": label_name,
                        "speaker": speaker,
                    }
                )
    metadata = pd.DataFrame(rows)

    result = speaker_disjoint_split(metadata)
    validate_speaker_disjoint(result)

    speaker_counts = result.groupby("split")["speaker"].nunique().to_dict()
    assert speaker_counts == {"train": 7, "val": 1, "test": 2}
    assert len(result) == len(metadata)
