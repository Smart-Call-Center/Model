import os
import json
import time

import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

MODEL_PATH = os.getenv("MODEL_PATH", "/app/model/model.joblib")
LABELS_PATH = os.path.join(os.path.dirname(MODEL_PATH), "labels.json")

app = FastAPI(title="tfidf_svc")
clf = None
labels = None

# Prometheus
PREDICTIONS_TOTAL = Counter("tfidf_predictions_total", "Total predictions (tfidf)")
PREDICTION_LATENCY = Histogram("tfidf_prediction_latency_seconds", "Latency (tfidf)")
UP = Gauge("tfidf_up", "Service up (tfidf)")


class PredictIn(BaseModel):
    text: str


class PredictOut(BaseModel):
    label: str
    confidences: dict[str, float]


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x -= np.max(x)
    e = np.exp(x)
    return e / e.sum()


@app.on_event("startup")
def load_model():
    """
    Charge le modèle TF-IDF + SVM et, si disponible, la liste des labels.
    """
    global clf, labels

    obj = joblib.load(MODEL_PATH)
    clf = obj

    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            labels_json = json.load(f)
            # On accepte soit une simple liste, soit un dict selon le training
            if isinstance(labels_json, list):
                # ex: ["Email Issue", "Network Issue", ...]
                labels = labels_json
            elif isinstance(labels_json, dict) and "id2label" in labels_json:
                # ex: {"0": "Email Issue", "1": "Network Issue", ...}
                # on ordonne par clé numérique
                id2label = labels_json["id2label"]
                labels = [id2label[str(i)] for i in range(len(id2label))]

    UP.set(1.0)


@app.get("/health")
def health():
    return {"model_loaded": clf is not None}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictOut)
def predict(inp: PredictIn) -> PredictOut:
    """
    Prend un texte et renvoie:
      - label: label avec la plus forte probabilité
      - confidences: dict {label: proba}
    """
    start = time.perf_counter()

    text = inp.text

    # 1) Probabilités selon les capacités du modèle
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba([text])[0]
    else:
        if hasattr(clf, "decision_function"):
            scores = clf.decision_function([text])[0]
            scores = np.atleast_1d(scores)
            if scores.ndim == 0:
                scores = np.array([1 - scores, scores])
            probs = _softmax(scores)
        else:
            # cas très dégradé: on ne dispose que de predict()
            pred = clf.predict([text])[0]
            n = len(labels) if labels else 1
            probs = np.zeros(n, dtype=np.float64)
            idx = int(pred) if isinstance(pred, (int, np.integer)) else 0
            idx = max(0, min(idx, n - 1))
            probs[idx] = 1.0

    # 2) Noms de classes
    if labels and len(labels) == len(probs):
        label_names = list(labels)
    else:
        label_names = [str(i) for i in range(len(probs))]

    # 3) Dict de confidences
    confidences = {
        label_names[i]: float(probs[i]) for i in range(len(probs))
    }

    # 4) Label avec la proba max
    best_label = max(confidences.items(), key=lambda x: x[1])[0]

    # 5) métriques
    PREDICTIONS_TOTAL.inc()
    PREDICTION_LATENCY.observe(time.perf_counter() - start)

    return PredictOut(label=best_label, confidences=confidences)
