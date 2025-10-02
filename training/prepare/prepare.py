# training/prepare/prepare.py
import pandas as pd
from pathlib import Path
import re

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1) Charger
df = pd.read_csv(RAW_DIR / "tickets.csv")  
df = df.rename(columns={"Document": "text", "Topic_group": "label"})

# 2) Nettoyage minimal
def clean_text(t: str) -> str:
    t = str(t)
    t = t.strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t

df["text"] = df["text"].apply(clean_text)

# 3) Filtrer lignes vides
df = df[df["text"].str.len() > 0]

# 4) Split train/val/test (80/10/10)
from sklearn.model_selection import train_test_split
train_df, temp = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
val_df, test_df = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp["label"])

# 5) Sauvegarder
train_df.to_csv(OUT_DIR / "train.csv", index=False)
val_df.to_csv(OUT_DIR / "val.csv", index=False)
test_df.to_csv(OUT_DIR / "test.csv", index=False)

print("Prepared:", { "train": len(train_df), "val": len(val_df), "test": len(test_df) })
