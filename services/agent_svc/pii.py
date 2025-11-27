import re
from typing import Pattern

# Patterns basiques de PII (tu peux les améliorer si tu veux)
EMAIL_RE: Pattern[str] = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_RE: Pattern[str] = re.compile(r"\+?\d[\d\s().-]{7,}\d")
IP_RE: Pattern[str] = re.compile(
    r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
)

ID_RE: Pattern[str] = re.compile(r"\b[A-Z]{2,5}-\d{3,10}\b")


def scrub_pii(text: str) -> str:


    if not text:
        return text

    text = EMAIL_RE.sub("<EMAIL>", text)
    text = PHONE_RE.sub("<PHONE>", text)
    text = IP_RE.sub("<IP>", text)
    text = ID_RE.sub("<TICKET_ID>", text)

    return text
