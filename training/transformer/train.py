# training/transformer/train.py
from pathlib import Path
import os
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# NEW
import mlflow

ARTIFACT_DIR = Path("artifacts/transformer"); ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_NAME = os.getenv("MODEL_NAME", "distilbert-base-multilingual-cased")  # FR/EN/AR friendly
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "CallCenterAI")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:mlruns")   # local folder tracking (no server needed)

def main():
    # ---------- MLflow setup ----------
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # ---------- Load splits ----------
    train = pd.read_pickle("data/processed/train.pkl")
    val   = pd.read_pickle("data/processed/val.pkl")
    test  = pd.read_pickle("data/processed/test.pkl")

    # ---------- Encode labels ----------
    le = LabelEncoder()
    y_all = pd.concat([train["label"], val["label"], test["label"]], ignore_index=True)
    le.fit(y_all)

    def encode(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"text": df["text"], "labels": le.transform(df["label"])})

    dtr = Dataset.from_pandas(encode(train), preserve_index=False)
    dvl = Dataset.from_pandas(encode(val),   preserve_index=False)
    dte = Dataset.from_pandas(encode(test),  preserve_index=False)

    # ---------- Tokenizer & padding ----------
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    collator = DataCollatorWithPadding(tokenizer=tok, pad_to_multiple_of=8)

    def tok_fn(batch):
        return tok(batch["text"], truncation=True)

    dtr = dtr.map(tok_fn, batched=True).remove_columns(["text"])
    dvl = dvl.map(tok_fn, batched=True).remove_columns(["text"])
    dte = dte.map(tok_fn, batched=True).remove_columns(["text"])

    # ---------- GPU settings ----------
    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    use_fp16 = use_cuda and not use_bf16
    torch_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)

    # ---------- Model ----------
    num_labels = len(le.classes_)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        torch_dtype=torch_dtype,
    )

    # ---------- Training args ----------
    args = TrainingArguments(
        output_dir=str(ARTIFACT_DIR / "runs"),
        num_train_epochs=int(os.getenv("EPOCHS", 2)),
        learning_rate=float(os.getenv("LR", 5e-5)),
        per_device_train_batch_size=int(os.getenv("TRAIN_BS", 16)),
        per_device_eval_batch_size=int(os.getenv("EVAL_BS", 16)),
        gradient_accumulation_steps=int(os.getenv("GRAD_ACC", 1)),
        warmup_ratio=float(os.getenv("WARMUP_RATIO", 0.1)),
        weight_decay=float(os.getenv("WEIGHT_DECAY", 0.01)),
        lr_scheduler_type=os.getenv("SCHEDULER", "linear"),

        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=int(os.getenv("LOG_STEPS", 50)),

        # >>> THIS is the key line to make Trainer talk to MLflow
        report_to="mlflow",
        run_name=os.getenv("RUN_NAME", "transformer-distilmultilingual"),

        # Mixed precision / perf
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=True,
        optim=os.getenv("OPTIM", "adamw_torch"),
        dataloader_num_workers=int(os.getenv("NUM_WORKERS", 2)),
        dataloader_pin_memory=True,
        group_by_length=True,
        torch_compile=bool(int(os.getenv("TORCH_COMPILE", "0"))),
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = (preds == labels).mean().item()
        return {"accuracy": float(acc)}

    # ---------- Training with MLflow run ----------
    with mlflow.start_run(run_name=args.run_name):
        # log a few params up-front
        mlflow.log_params({
            "model_name": MODEL_NAME,
            "num_labels": num_labels,
            "epochs": args.num_train_epochs,
            "lr": args.learning_rate,
            "train_bs": args.per_device_train_batch_size,
            "eval_bs": args.per_device_eval_batch_size,
            "fp16": args.fp16,
            "bf16": args.bf16,
        })

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=dtr,
            eval_dataset=dvl,
            tokenizer=tok,
            data_collator=collator,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        eval_val = trainer.evaluate(dvl)
        mlflow.log_metrics({f"val_{k}": float(v) for k, v in eval_val.items() if isinstance(v, (int, float))})

        # Optional: evaluate on test and log
        eval_test = trainer.evaluate(dte)
        mlflow.log_metrics({f"test_{k}": float(v) for k, v in eval_test.items() if isinstance(v, (int, float))})

        # ---------- Save small, reusable export ----------
        # (Consider saving under artifacts/transformer/final and track only that in DVC)
        model.save_pretrained(ARTIFACT_DIR)
        tok.save_pretrained(ARTIFACT_DIR)
        (ARTIFACT_DIR / "labels.txt").write_text("\n".join(le.classes_), encoding="utf-8")

        # log the folder to MLflow so it appears under "Artifacts"
        mlflow.log_artifacts(str(ARTIFACT_DIR), artifact_path="transformer")

        print(f"GPU used: {use_cuda} | bf16={use_bf16} fp16={use_fp16}")
        print(f"Saved HF model → {ARTIFACT_DIR}")

if __name__ == "__main__":
    main()
