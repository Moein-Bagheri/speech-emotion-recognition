from __future__ import annotations

import pandas as pd

import pytest

from ser.data import (
    speaker_disjoint_split,
    speaker_from_filename,
    validate_speaker_disjoint,
)

LABELS = ("Angry", "Happy", "Neutral", "Sad")


@pytest.mark.parametrize(
    ("filename", "speaker"),
    [
        ("Ses04M_impro01_F000.wav", "Ses04F"),
        ("Ses04M_impro01_M000.wav", "Ses04M"),
        ("Ses01F_script02_2_M013.wav", "Ses01M"),
        ("Ses03M_impro05b_F021.wav", "Ses03F"),
        ("Ses05F_script01_1b_F003.wav", "Ses05F"),
    ],
)
def test_speaker_comes_from_the_utterance_not_the_dialog(
    filename: str, speaker: str
) -> None:
    assert speaker_from_filename(filename) == speaker


def test_speaker_from_filename_rejects_other_names() -> None:
    with pytest.raises(ValueError):
        speaker_from_filename("Ses04M_impro01.wav")


def test_split_from_real_names_never_shares_a_voice() -> None:
    # Real IEMOCAP naming: both actors of a session speak in every dialog of that session.
    rows = []
    for session in range(1, 6):
        for dialog in "FM":
            for voice in "FM":
                for label_id, label_name in enumerate(LABELS):
                    for repetition in range(label_id + 2):
                        filename = (
                            f"Ses{session:02d}{dialog}_impro0{label_id + 1}_"
                            f"{voice}{repetition:03d}.wav"
                        )
                        rows.append(
                            {
                                "file": filename,
                                "label_id": label_id,
                                "label_name": label_name,
                                "speaker": speaker_from_filename(filename),
                            }
                        )
    metadata = pd.DataFrame(rows)

    result = speaker_disjoint_split(metadata)
    validate_speaker_disjoint(result)

    # the voice, read independently of the parser: session plus the letter of the last part
    voice = result["file"].map(lambda name: name[:5] + name.rsplit("_", 1)[1][0])
    assert result.assign(voice=voice).groupby("voice")["split"].nunique().max() == 1
    assert result.groupby("split")["speaker"].nunique().to_dict() == {
        "train": 7,
        "val": 1,
        "test": 2,
    }


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
