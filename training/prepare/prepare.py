# training/prepare/prepare.py
from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Config (override via env if needed)
RAW_CSV      = Path(os.getenv("RAW_CSV", "data/raw/tickets.csv"))
OUT_DIR      = Path(os.getenv("OUT_DIR", "data/processed"))
TEXT_COL     = os.getenv("TEXT_COL", "Document")
LABEL_COL    = os.getenv("LABEL_COL", "Topic_group")
TEST_SIZE    = float(os.getenv("TEST_SIZE", "0.2"))
VAL_SIZE     = float(os.getenv("VAL_SIZE", "0.1"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"Missing RAW_CSV at {RAW_CSV}")

    df = pd.read_csv(RAW_CSV)
    if TEXT_COL not in df.columns or LABEL_COL not in df.columns:
        raise ValueError(f"Expected columns [{TEXT_COL}, {LABEL_COL}] in {RAW_CSV}. Found {list(df.columns)}")

    df = df[[TEXT_COL, LABEL_COL]].dropna().rename(columns={TEXT_COL: "text", LABEL_COL: "label"})
    # train/test split
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["label"]
    )
    # train/val split
    train_df, val_df = train_test_split(
        train_df, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=train_df["label"]
    )

    train_df.to_pickle(OUT_DIR / "train.pkl", protocol=4)
    val_df.to_pickle(OUT_DIR / "val.pkl", protocol=4)
    test_df.to_pickle(OUT_DIR / "test.pkl", protocol=4)

    print(f"[OK] Saved: {OUT_DIR/'train.pkl'}, {OUT_DIR/'val.pkl'}, {OUT_DIR/'test.pkl'}")

if __name__ == "__main__":
    main()
