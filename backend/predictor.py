from backend.model import classify_probability, load_model_bundle
from backend.preprocessing import prepare_input


def predict_record(raw_record: dict) -> dict:
    model, _, thresholds = load_model_bundle()
    feature_frame = prepare_input(raw_record)
    probability = float(model.predict_proba(feature_frame)[0, 1])
    risk_level = classify_probability(probability, thresholds)
    return {
        "failure_probability": round(probability, 6),
        "risk_level": risk_level,
    }
