from __future__ import annotations

import os
from typing import Literal, Tuple, Dict, Any

import langdetect
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MAX_WORDS_SIMPLE: int = int(os.getenv("ROUTER_MAX_WORDS_SIMPLE", "40"))

COMPLEXITY_THRESHOLD: float = float(os.getenv("ROUTER_COMPLEXITY_THRESHOLD", "0.5"))

_SUPPORTED_LANGS_TFIDF = os.getenv("ROUTER_TFIDF_LANGS", "en,fr")
SUPPORTED_LANGS_TFIDF = {
    lang.strip().lower()
    for lang in _SUPPORTED_LANGS_TFIDF.split(",")
    if lang.strip()
}

ROUTER_MODEL_NAME: str = os.getenv("ROUTER_MODEL_NAME", "google/flan-t5-small")

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


_router_tokenizer = AutoTokenizer.from_pretrained(ROUTER_MODEL_NAME)
_router_model = AutoModelForSeq2SeqLM.from_pretrained(ROUTER_MODEL_NAME).to(DEVICE)
_router_model.eval()




def detect_language(text: str) -> str:
    """
    Détecte la langue principale du texte.
    Retourne un code ISO (ex: 'en', 'fr', 'ar') ou 'unknown' en cas d'erreur.
    """
    try:
        return langdetect.detect(text).lower()
    except Exception:
        return "unknown"


def estimate_complexity(text: str) -> float:

    prompt = (
        "You are evaluating the complexity of an IT support ticket.\n"
        "Consider aspects like the number of concepts, technical difficulty, ambiguity, "
        "and the mental effort needed to diagnose and resolve the issue.\n"
        "Return a single number between 0.0 and 1.0 where:\n"
        " - 0.0 means very simple and easy to understand.\n"
        " - 1.0 means highly complex and difficult.\n"
        "Respond with ONLY the numeric value.\n\n"
        f"Ticket: {text}\n"
    )

    inputs = _router_tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = _router_model.generate(**inputs, max_new_tokens=8)

    result = _router_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    try:
        score = float(result)
    except ValueError:
        score = 0.5  

    if score < 0.0:
        score = 0.0
    elif score > 1.0:
        score = 1.0

    return score



def route_ticket(text: str) -> Tuple[Literal["tfidf", "transformer"], Dict[str, Any]]:
   
    text = (text or "").strip()

    if not text:
        # Cas extrême : texte vide → on renvoie TF-IDF par défaut
        choice: Literal["tfidf", "transformer"] = "tfidf"
        debug = {
            "language": "unknown",
            "word_count": 0,
            "complexity": None,
            "reason": "empty_text_default_tfidf",
            "max_words_simple": MAX_WORDS_SIMPLE,
            "complexity_threshold": COMPLEXITY_THRESHOLD,
        }
        return choice, debug

    lang = detect_language(text)
    word_count = len(text.split())

    if lang not in SUPPORTED_LANGS_TFIDF:
        choice = "transformer"
        debug = {
            "language": lang,
            "word_count": word_count,
            "complexity": None,
            "reason": "lang_not_supported_for_tfidf",
            "max_words_simple": MAX_WORDS_SIMPLE,
            "complexity_threshold": COMPLEXITY_THRESHOLD,
        }
        return choice, debug

    if word_count > MAX_WORDS_SIMPLE:
        choice = "transformer"
        debug = {
            "language": lang,
            "word_count": word_count,
            "complexity": None,
            "reason": "too_long_for_tfidf",
            "max_words_simple": MAX_WORDS_SIMPLE,
            "complexity_threshold": COMPLEXITY_THRESHOLD,
        }
        return choice, debug
    complexity = estimate_complexity(text)

    if complexity >= COMPLEXITY_THRESHOLD:
        choice = "transformer"
        reason = "complex_ticket"
    else:
        choice = "tfidf"
        reason = "simple_ticket"

    debug = {
        "language": lang,
        "word_count": word_count,
        "complexity": complexity,
        "reason": reason,
        "max_words_simple": MAX_WORDS_SIMPLE,
        "complexity_threshold": COMPLEXITY_THRESHOLD,
    }
    return choice, debug


if __name__ == "__main__":
    simple_ticket = "Je n'arrive pas à me connecter à mon compte après avoir réinitialisé mon mot de passe."
    complex_ticket = (
        "Notre cluster de microservices distribués tombe en panne de manière intermittente sous forte charge, "
        "provoquant des timeouts en cascade dans les services dépendants. Nous observons des fuites mémoire "
        "dans la couche de cache et des conditions de course dans la réplication de la base de données."
    )

    for sample in (simple_ticket, complex_ticket):
        c, dbg = route_ticket(sample)
        print("TEXT:", sample)
        print("CHOICE:", c)
        print("DEBUG:", dbg)
        print("-" * 40)
