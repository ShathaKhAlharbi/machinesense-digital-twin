import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

with open(MODELS_DIR / "preprocessing_stats.json", "r", encoding="utf-8") as fh:
    PREPROCESSING_STATS = json.load(fh)

FEATURE_ORDER = PREPROCESSING_STATS["feature_order"]
NUMERIC_FEATURES = list(PREPROCESSING_STATS["numeric_median"].keys())
CATEGORICAL_FEATURES = list(PREPROCESSING_STATS["categorical_mode"].keys())


def prepare_input(raw_record: dict) -> pd.DataFrame:
    payload = {}

    for column in NUMERIC_FEATURES:
        if column not in raw_record:
            raise ValueError(f"Missing required numeric feature: {column}")
        payload[column] = raw_record[column]

    for column in CATEGORICAL_FEATURES:
        if column not in raw_record:
            raise ValueError(f"Missing required categorical feature: {column}")
        payload[column] = str(raw_record[column]).strip()

    frame = pd.DataFrame([payload])

    for column in NUMERIC_FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame[column] = frame[column].fillna(PREPROCESSING_STATS["numeric_median"][column])

    for column in CATEGORICAL_FEATURES:
        frame[column] = frame[column].fillna(PREPROCESSING_STATS["categorical_mode"][column])
        frame[column] = frame[column].astype(str)

    frame = pd.get_dummies(frame, columns=CATEGORICAL_FEATURES, dtype=int)

    for feature in FEATURE_ORDER:
        if feature not in frame.columns:
            frame[feature] = 0

    frame = frame.reindex(columns=FEATURE_ORDER, fill_value=0)

    for column in NUMERIC_FEATURES:
        train_min = PREPROCESSING_STATS["train_min"][column]
        train_max = PREPROCESSING_STATS["train_max"][column]
        train_range = PREPROCESSING_STATS["train_range"][column]
        safe_range = train_range if train_range != 0 else 1.0
        frame[column] = (frame[column] - train_min) / safe_range

    return frame
