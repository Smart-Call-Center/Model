# models/transformers.py
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import json
import torch
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    TextClassificationPipeline,
)


@dataclass
class TransformerConfig:
    model_name: str = "distilbert-base-multilingual-cased"
    max_length: int = 256
    device: Optional[int] = 0 if torch.cuda.is_available() else -1  # -1 = CPU
    fp16: bool = True


class TransformerClassifier:
    def __init__(self, model_dir: str | Path, max_length: int = 256, device: int = -1):
        self.model_dir = Path(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.config = AutoConfig.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_dir,
            config=self.config,
        )
        self.pipeline = TextClassificationPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            top_k=None,
            truncation=True,
            padding=True,
            max_length=max_length,
            device=device,
            return_all_scores=True,
        )

    def predict(self, texts: List[str]) -> List[Dict[str, float]]:
        if isinstance(texts, str):
            texts = [texts]
        outputs = self.pipeline(texts)
        results: List[Dict[str, float]] = []
        for out in outputs:
            conf = {item["label"]: float(item["score"]) for item in out}
            results.append(conf)
        return results


def save_hf_artifacts(
    model,
    tokenizer,
    out_dir: str | Path,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
) -> None:
    """
    Sauvegarde propre d'un modèle HF pour l'inférence :
      - met à jour config.label2id / config.id2label
      - sauvegarde model + tokenizer
      - écrit labels.json (utile côté services)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model.config.label2id = label2id
    model.config.id2label = id2label

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    labels_payload = {
        "label2id": label2id,
        "id2label": {str(k): v for k, v in id2label.items()},
    }
    (out_dir / "labels.json").write_text(
        json.dumps(labels_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_hf_for_inference(
    model_dir: str | Path,
    cfg: Optional[TransformerConfig] = None,
) -> TransformerClassifier:
    cfg = cfg or TransformerConfig()
    device = cfg.device if cfg.device is not None else -1
    return TransformerClassifier(
        model_dir,
        max_length=cfg.max_length,
        device=device,
    )
