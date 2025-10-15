# training/tfidf/train.py
import json, joblib, pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report

# (optionnel) MLflow
import mlflow

DATA_DIR = Path("data/processed")
ARTIFACT_DIR = Path("artifacts/tfidf"); ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    train = pd.read_pickle(DATA_DIR/"train.pkl")
    val   = pd.read_pickle(DATA_DIR/"val.pkl")
    test  = pd.read_pickle(DATA_DIR/"test.pkl")

    X_train = pd.concat([train["text"], val["text"]], ignore_index=True)
    y_train = pd.concat([train["label"], val["label"]], ignore_index=True)
    X_test, y_test = test["text"], test["label"]

    # Vectorizer + SVM linéaire
    vec  = TfidfVectorizer(max_features=50000, ngram_range=(1,2))
    base = LinearSVC()  # pas de proba native
    clf  = CalibratedClassifierCV(base, cv=3)  # calibration -> predict_proba disponible

    pipe = make_pipeline(vec, clf)

    # MLflow (facultatif mais recommandé)
    mlflow.set_experiment("CallCenterAI")
    with mlflow.start_run(run_name="tfidf_svm_calibrated"):
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1w = f1_score(y_test, y_pred, average="weighted")

        mlflow.log_param("max_features", 50000)
        mlflow.log_param("ngram_range", "1,2")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_weighted", f1w)

        # rapport détaillé
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        (ARTIFACT_DIR/"classification_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        joblib.dump(pipe, ARTIFACT_DIR/"model.joblib")
        mlflow.log_artifact(ARTIFACT_DIR/"classification_report.json")
        mlflow.log_artifact(ARTIFACT_DIR/"model.joblib")

    print(f"OK: acc={acc:.4f} f1_w={f1w:.4f} | saved → {ARTIFACT_DIR/'model.joblib'}")

if __name__ == "__main__":
    main()
