# Smart Call Center – Model

Projet MLOps pour la classification automatique de tickets de support dans un centre d’appel.

Ce dépôt contient :

- Deux modèles de classification :
  - **TF-IDF + SVM** (modèle rapide / baseline)
  - **Transformer multilingue** (modèle Hugging Face finetuné)
- Un **service agent** qui :
  - nettoie le texte (suppression de PII),
  - décide quel modèle utiliser (TF-IDF vs Transformer),
  - appelle le bon service via HTTP.
- Une stack MLOps autour :
  - **MLflow** pour le suivi des expériences,
  - **DVC** pour la pipeline de training,
  - **Prometheus + Grafana** pour le monitoring,
  - **Docker Compose** pour lancer tous les services.

---

## 1. Architecture globale

### 1.1 Composants principaux

- **`tfidf_svc`**  
  Service FastAPI exposant le modèle TF-IDF + SVM :
  - `GET /health`
  - `POST /predict`
  - `GET /metrics` (métriques Prometheus)

- **`transformer_svc`**  
  Service FastAPI exposant le modèle Transformer (HuggingFace) :
  - `GET /health`
  - `POST /predict`
  - `GET /metrics`

- **`agent_svc`**  
  Service FastAPI qui :
  - nettoie le texte via `scrub_pii`,
  - utilise `route_ticket` pour décider **tfidf** ou **transformer** en fonction :
    - de la langue (FR/EN/AR…),
    - de la longueur du ticket,
    - de la complexité (modèle HF de routage),
  - appelle le service choisi via HTTP (`TFIDF_URL` / `TRANSFORMER_URL`),
  - expose des métriques Prometheus :
    - `agent_requests_total`
    - `agent_prediction_latency_seconds`
    - `agent_route_total{model="tfidf"|"transformer"}`
    - `agent_up`

- **MLflow**  
  Serveur MLflow utilisé pour :
  - **enregistrer les runs** de training (métriques, hyperparamètres, artifacts),
  - préparer la **Model Registry** (stages `Production` / `Staging`).

- **Prometheus**  
  Scrape les métriques de :
  - `agent_svc`, `tfidf_svc`, `transformer_svc`,
  - lui-même.

- **Grafana**  
  Dashboards pour :
  - état des services (up/down),
  - nombre de requêtes,
  - latence des prédictions,
  - répartition TF-IDF / Transformer.

### 1.2 Ports utilisés

Par défaut :

- Agent : `http://localhost:8080`
- TF-IDF : `http://localhost:8081`
- Transformer : `http://localhost:8082`
- Prometheus : `http://localhost:9090`
- Grafana : `http://localhost:3000`
- MLflow : `http://localhost:5000`

(Ports définis dans `docker-compose.yml`.)

---

## 2. Structure du projet

Aperçu simplifié (les chemins exacts peuvent légèrement varier) :

```text
.
├── services/
│   ├── tfidf_svc/
│   │   ├── main.py          # API TF-IDF
│   │   └── Dockerfile
│   ├── transformer_svc/
│   │   ├── main.py          # API Transformer
│   │   ├── train.py         # training Transformer
│   │   └── Dockerfile
│   └── agent_svc/
│       ├── main.py          # API Agent
│       ├── router.py        # logique de routage TF-IDF / Transformer
│       ├── pii.py           # nettoyage PII
│       └── Dockerfile
│
├── models/
│   ├── tfidf.py             # pipeline TF-IDF + SVM
│   └── transformers.py      # wrapper pour le modèle HF
│
├── training/
│   ├── prepare.py           # préparation des données
│   ├── tfidf/
│   │   └── train.py         # training TF-IDF
│   └── transformer/
│       └── train.py         # training Transformer
│
├── artifacts/
│   ├── tfidf/               # modèle TF-IDF entraîné (model.joblib, labels, etc.)
│   └── transformer/         # modèle Transformer (répertoires HF)
│
├── monitoring/
│   ├── prometheus.yml       # config Prometheus (scrape /metrics)
│   └── grafana/
│       ├── provisioning/
│       │   ├── datasources/ # datasource Prometheus
│       │   └── dashboards/  # dashboards pré-configurés (JSON)
│       └── ...              # éventuelles données/volumes Grafana
│
├── dvc.yaml                 # pipeline DVC (prepare, train_tfidf, train_transformer)
├── docker-compose.yml       # lance tous les services (modèles + monitoring + mlflow)
├── requirements.txt
└── README.md
