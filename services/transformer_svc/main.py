# services/transformer_svc/main.py
import os
import time
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/app/model"))

PRED_COUNT = Counter("transformer_predictions_total", "Number of predictions")
PRED_LAT = Histogram("transformer_prediction_latency_seconds", "Prediction latency")
MODEL_UP = Gauge("transformer_model_loaded", "1 if model loaded, else 0")

app = FastAPI(title="transformer_svc")


class PredictIn(BaseModel):
    text: str


class PredictOut(BaseModel):
    label: str
    confidences: Dict[str, float]
    model: str = "transformer_svc"


clf = None  # HF pipeline


@app.on_event("startup")
def load_model():
    """
    Charge le modèle HF fine-tuné (AutoModelForSequenceClassification) + tokenizer.
    Les labels métiers viennent de model.config.id2label qui a été mis à jour
    pendant le training via save_hf_artifacts.
    """
    global clf

    if (MODEL_DIR / "config.json").exists():
        tok = AutoTokenizer.from_pretrained(MODEL_DIR)
        mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

        clf_local = pipeline(
            "text-classification",
            model=mdl,
            tokenizer=tok,
            return_all_scores=True,
        )
        clf = clf_local
        MODEL_UP.set(1.0)
    else:
        MODEL_UP.set(0.0)


@app.get("/health")
def health():
    return {"model_loaded": clf is not None}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictOut)
def predict(inp: PredictIn) -> PredictOut:
    """
    Renvoie:
      - label: label avec score max (nom métier si training bien fait)
      - confidences: dict {label: score}
    """
    if clf is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t0 = time.time()

    outputs = clf(inp.text)[0]  # liste de dicts {label, score}
    conf = {o["label"]: float(o["score"]) for o in outputs}
    pred_label = max(conf.items(), key=lambda x: x[1])[0]

    PRED_COUNT.inc()
    PRED_LAT.observe(time.time() - t0)

    return PredictOut(label=pred_label, confidences=conf)
