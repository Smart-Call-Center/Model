# training/tfidf/train.py
from pathlib import Path
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib
import json

DATA_DIR = Path("data/processed")
ART_DIR = Path("artifacts/tfidf")
ART_DIR.mkdir(parents=True, exist_ok=True)

def load_split():
    train = pd.read_csv(DATA_DIR / "train.csv")
    val   = pd.read_csv(DATA_DIR / "val.csv")
    for dfname, df in [("train", train), ("val", val)]:
        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError(f"{dfname}.csv doit contenir les colonnes 'text' et 'label'")
    # Assurer le bon type
    train["text"] = train["text"].astype(str)
    val["text"]   = val["text"].astype(str)
    train["label"] = train["label"].astype(str)
    val["label"]   = val["label"].astype(str)
    return train, val

def build_model(max_features=50000, C=1.0, min_df=2, ngram_max=2):
    vec = TfidfVectorizer(ngram_range=(1, ngram_max), max_features=max_features, min_df=min_df)
    base = LinearSVC(C=C)
    clf  = CalibratedClassifierCV(base, cv=3)  
    pipe = Pipeline([("tfidf", vec), ("clf", clf)])
    return pipe

def main():
    # 1) Données
    train, val = load_split()
    X_train, y_train = train["text"], train["label"]
    X_val,   y_val   = val["text"],   val["label"]

    # 2) MLflow local (dossier ./mlruns)
    tracking_dir = Path("mlruns").resolve()
    mlflow.set_tracking_uri(tracking_dir.as_uri())
    mlflow.set_experiment("tfidf_baseline")

    params = {"max_features": 50000, "C": 1.0, "min_df": 2, "ngram_max": 2}

    with mlflow.start_run(run_name="tfidf_linearSVC_calibrated"):
        mlflow.log_params(params)

        model = build_model(**params)
        model.fit(X_train, y_train)

        # 3) Eval
        pred = model.predict(X_val)
        acc = accuracy_score(y_val, pred)
        f1m = f1_score(y_val, pred, average="macro")
        mlflow.log_metric("val_accuracy", acc)
        mlflow.log_metric("val_f1_macro", f1m)

        # Rapport détaillé (utile pour debug)
        report = classification_report(y_val, pred, output_dict=True)
        report_path = ART_DIR / "classification_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        mlflow.log_artifact(str(report_path))

        # 4) Sauvegarde du modèle (artefact principal)
        model_path = ART_DIR / "model.joblib"
        joblib.dump(model, model_path)
        mlflow.log_artifact(str(model_path))

        print({
            "val_accuracy": round(acc, 4),
            "val_f1_macro": round(f1m, 4),
            "model_path": str(model_path)
        })

if __name__ == "__main__":
    main()