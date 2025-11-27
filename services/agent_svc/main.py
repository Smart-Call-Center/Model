import os
import time
from typing import Dict, Any

import requests
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

from router import route_ticket
from pii import scrub_pii


TFIDF_URL = os.getenv("TFIDF_URL", "http://tfidf_svc:8001/predict")
TRANSFORMER_URL = os.getenv("TRANSFORMER_URL", "http://transformer_svc:8002/predict")
REQUEST_TIMEOUT = float(os.getenv("AGENT_DOWNSTREAM_TIMEOUT", "5.0"))



AGENT_REQUESTS = Counter(
    "agent_requests_total", "Number of prediction requests received by the agent"
)
AGENT_LATENCY = Histogram(
    "agent_prediction_latency_seconds", "End-to-end prediction latency of the agent"
)
AGENT_ROUTE_COUNT = Counter(
    "agent_route_total", "Routing decisions made by the agent", ["model"]
)
AGENT_UP = Gauge("agent_up", "Agent service health flag")

AGENT_UP.set(1.0)


app = FastAPI(title="CallCenterAI Agent Service")


class AgentPredictIn(BaseModel):
    text: str


class AgentPredictOut(BaseModel):
    label: str
    model_used: str
    confidences: Dict[str, float]
    router_decision: Dict[str, Any]


@app.get("/health")
def health() -> Dict[str, Any]:
    """
    Healthcheck simple.
    """
    return {
        "status": "ok",
        "tfidf_url": TFIDF_URL,
        "transformer_url": TRANSFORMER_URL,
    }


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=AgentPredictOut)
def predict(inp: AgentPredictIn) -> AgentPredictOut:
    """
    1. Scrub PII
    2. Route (tfidf / transformer)
    3. Appel HTTP au service choisi
    4. Retourne la prédiction + info routage
    """

    AGENT_REQUESTS.inc()
    t0 = time.time()

    clean_text = scrub_pii(inp.text or "")

    # Décision de routage
    model_choice, debug = route_ticket(clean_text)

    if model_choice == "tfidf":
        url = TFIDF_URL
    else:
        url = TRANSFORMER_URL

    # Appel HTTP downstream
    try:
        resp = requests.post(
            url,
            json={"text": clean_text},
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as exc:
        AGENT_LATENCY.observe(time.time() - t0)
        raise HTTPException(
            status_code=502,
            detail=f"Error calling downstream {model_choice} service: {exc}",
        )

    if resp.status_code != 200:
        AGENT_LATENCY.observe(time.time() - t0)
        raise HTTPException(
            status_code=502,
            detail=f"Downstream {model_choice} service returned {resp.status_code}: {resp.text}",
        )

    data = resp.json()

    label = data.get("label")
    confidences = data.get("confidences") or {}

    if label is None:
        AGENT_LATENCY.observe(time.time() - t0)
        raise HTTPException(
            status_code=500,
            detail=f"Downstream {model_choice} service did not return a 'label' field",
        )

    AGENT_ROUTE_COUNT.labels(model=model_choice).inc()
    AGENT_LATENCY.observe(time.time() - t0)

    return AgentPredictOut(
        label=label,
        model_used=model_choice,
        confidences={k: float(v) for k, v in confidences.items()},
        router_decision=debug,
    )
