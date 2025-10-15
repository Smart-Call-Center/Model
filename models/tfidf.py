# models/tfidf.py
import joblib
from pathlib import Path

MODEL_PATH = Path("artifacts/tfidf/model.joblib")

def load_model():
    return joblib.load(MODEL_PATH)

def predict(texts):
    m = load_model()
    return m.predict(texts).tolist()

def predict_proba(texts):
    m = load_model()
    if hasattr(m, "predict_proba"):
        return m.predict_proba(texts).tolist()
    return None
