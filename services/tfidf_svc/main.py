import os, json, joblib, numpy as np, time
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

MODEL_PATH = os.getenv("MODEL_PATH", "/app/model/model.joblib")
LABELS_PATH = os.path.join(os.path.dirname(MODEL_PATH), "labels.json")

app = FastAPI()
clf = None
labels = None

# Prometheus (optional but nice)
PREDICTIONS_TOTAL = Counter("tfidf_predictions_total", "Total predictions (tfidf)")
PREDICTION_LATENCY = Histogram("tfidf_prediction_latency_seconds", "Latency (tfidf)")
UP = Gauge("tfidf_up", "Service up (tfidf)")

class Input(BaseModel):
    text: str

def _softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x -= np.max(x)
    e = np.exp(x)
    return e / e.sum()

@app.on_event("startup")
def load_model():
    global clf, labels
    # load trained artifact
    obj = joblib.load(MODEL_PATH)

    # many trainers save the Pipeline directly -> use it
    clf = obj

    # optional labels
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            labels_json = json.load(f)
            if isinstance(labels_json, list):
                labels = labels_json

    UP.set(1)

@app.get("/health")
def health():
    return {"model_loaded": clf is not None}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict")
def predict(inp: Input):
    start = time.perf_counter()
    text = inp.text

    # prefer calibrated probabilities if available
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba([text])[0]
    else:
        # fall back to decision_function + softmax
        if hasattr(clf, "decision_function"):
            scores = clf.decision_function([text])[0]
            # binary margins may be scalar; normalize to vector
            scores = np.atleast_1d(scores)
            if scores.ndim == 0:
                scores = np.array([1 - scores, scores])
            probs = _softmax(scores)
        else:
            # final fallback: hard prediction -> fake one-hot
            pred = clf.predict([text])[0]
            n = len(labels) if labels else 1
            probs = np.zeros(n, dtype=np.float64)
            idx = int(pred) if isinstance(pred, (int, np.integer)) else 0
            idx = max(0, min(idx, n - 1))
            probs[idx] = 1.0

    # build response
    if labels and len(labels) == len(probs):
        preds = [{"label": labels[i], "score": float(probs[i])} for i in range(len(probs))]
    else:
        preds = [{"label": str(i), "score": float(probs[i])} for i in range(len(probs))]

    preds.sort(key=lambda x: x["score"], reverse=True)

    PREDICTIONS_TOTAL.inc()
    PREDICTION_LATENCY.observe(time.perf_counter() - start)
    return {"predictions": preds}

