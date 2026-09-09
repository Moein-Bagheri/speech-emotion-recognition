"""Shared constants and paths for the speech-emotion project."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_ROOT / "raw"
PROCESSED_DATA_DIR = DATA_ROOT / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DOC_DIR = PROJECT_ROOT / "doc"

HF_REPO_ID = "AbstractTTS/IEMOCAP"
HF_REPO_TYPE = "dataset"

# The numeric order is fixed by the assignment.
LABEL_TO_ID = {
    "angry": 0,
    "happy": 1,
    "neutral": 2,
    "sad": 3,
}
ID_TO_LABEL = {
    0: "Angry",
    1: "Happy",
    2: "Neutral",
    3: "Sad",
}

SAMPLE_RATE = 16_000
N_MFCC = 40
HOP_LENGTH = 512
MAX_LEN = 300
SEED = 42

CHANNELS_BY_FEATURE_MODE = {
    "mfcc": 40,
    "mfcc_delta": 80,
    "mfcc_delta2": 120,
}
