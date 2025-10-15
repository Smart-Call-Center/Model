from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import (AutoTokenizer, AutoConfig,
                          AutoModelForSequenceClassification, TextClassificationPipeline)

@dataclass
class TransformerConfig:
    model_name: str = "distilbert-base-multilingual-cased"  # multi-lang
    max_length: int = 256
    device: Optional[int] = 0 if torch.cuda.is_available() else -1  # -1 = CPU
    fp16: bool = True

class TransformerClassifier:
    """
    Wrapper d'inférence: charge un dossier HF (config, tokenizer, weights)
    et expose .predict(texts) -> [(label, score), ...]
    """
    def __init__(self, model_dir: str | Path, max_length: int = 256, device: int = -1):
        model_dir = str(model_dir)
        tok = AutoTokenizer.from_pretrained(model_dir)
        mdl = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.pipe = TextClassificationPipeline(
            model=mdl, tokenizer=tok, device=device, top_k=None, truncation=True
        )
        self.max_length = max_length

    def predict(self, texts: List[str]) -> List[Dict]:
        return self.pipe(texts)

def save_hf_artifacts(model, tokenizer, out_dir: str | Path, label2id: Dict[str, int], id2label: Dict[int, str]) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.config.label2id = label2id
    model.config.id2label = id2label
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    # dump mapping à part aussi (utile)
    (out_dir / "labels.json").write_text(
        __import__("json").dumps({"label2id": label2id, "id2label": {str(k): v for k, v in id2label.items()}}, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

def load_hf_for_inference(model_dir: str | Path, cfg: Optional[TransformerConfig] = None) -> TransformerClassifier:
    cfg = cfg or TransformerConfig()
    device = cfg.device if cfg.device is not None else (-1)
    return TransformerClassifier(model_dir, max_length=cfg.max_length, device=device)
