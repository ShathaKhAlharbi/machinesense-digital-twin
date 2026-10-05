from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"


@lru_cache(maxsize=1)
def load_model_bundle():
    model = joblib.load(MODELS_DIR / "FINAL_random_forest_model.joblib")
    feature_names = pd.read_csv(MODELS_DIR / "feature_names.csv").iloc[:, 0].tolist()
    risk_thresholds = pd.read_csv(MODELS_DIR / "risk_state_thresholds.csv")
    return model, feature_names, risk_thresholds


def classify_probability(probability: float, thresholds: pd.DataFrame) -> str:
    normal_upper = float(thresholds.loc[thresholds["Risk State"] == "Normal", "Maximum Probability"].iloc[0])
    warning_upper = float(thresholds.loc[thresholds["Risk State"] == "Warning", "Maximum Probability"].iloc[0])

    if probability < normal_upper:
        return "Normal"
    if probability < warning_upper:
        return "Warning"
    return "High Risk"
