# Malaria Cell Classifier — End-to-End MLOps Pipeline

Image classification (**Parasitized** vs **Uninfected** blood-smear cells, NIH malaria dataset) with a full ML cycle:
data acquisition → preprocessing → model creation → testing → retraining → API → UI → cloud deployment → load testing.

## Links
- 🎥 **Video demo (YouTube):** _TODO: paste link_
- 🌐 **Live app (Render):** _TODO: paste URL_
- 💻 **GitHub repo:** _TODO: paste URL_

## Project description
| Piece | What it does |
|---|---|
| `src/preprocessing.py` | Download/split the dataset, image loading, feature extraction for insights |
| `src/model.py` | MobileNetV2 transfer-learning model (or custom CNN), training with regularisation + early stopping, evaluation (accuracy, precision, recall, F1, ROC-AUC, loss, confusion matrix) |
| `src/prediction.py` | Single-image prediction (auto-reloads the model after retraining) |
| `src/retrain.py` | Ingest uploaded data, fine-tune, evaluate, hot-swap model. Trigger: UI button / `POST /retrain`, or automatically when pending uploads ≥ `RETRAIN_THRESHOLD` (default 50) |
| `src/database.py` | SQLite log of every uploaded image (`data/app.db`) |
| `src/api.py` | FastAPI: `/predict`, `/upload`, `/retrain`, `/retrain/status`, `/health`, `/metrics`, `/insights/*` (Swagger at `/docs`) |
| `ui/streamlit_app.py` | Dashboard (up-time, metrics), Predict, Data insights, Upload & Retrain |
| `locust/locustfile.py` | Flood-request simulation |
| `notebook/malaria_mlops.ipynb` | Preprocessing, training, evaluation, prediction functions |

Labels: `0 = Uninfected`, `1 = Parasitized` (positive class = disease present).

## Setup (local)
```bash
git clone <repo-url> && cd malaria_mlops
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

# 1. Get data (Kaggle credentials required) — or use --raw-dir with a local copy
python -m src.preprocessing --per-class 2000

# 2. Train + evaluate offline (saves models/malaria_model.h5 and models/metrics.json)
python -m src.model --epochs 8 --fine-tune-epochs 4
#    (add --model-type custom_cnn to train from scratch)

# 3. Run API and UI
uvicorn src.api:app --port 8000                          # http://localhost:8000/docs
API_URL=http://localhost:8000 streamlit run ui/streamlit_app.py
```

## Docker
```bash
docker build -t malaria-classifier .
docker run -p 8000:8000 -p 8501:8501 malaria-classifier   # UI :8501, API :8000
```

## Deploy on Render
1. Push the repo (including `models/malaria_model.h5` and the small `data/` sample) to GitHub.
2. Render → **New → Blueprint** (uses `render.yaml`) or **New → Web Service → Docker**.
3. The public URL serves the Streamlit UI; the API runs inside the same container.
> Free tier has 512 MB RAM: prediction works; retraining may need a larger plan.

## Flood-request simulation (Locust)
```bash
docker compose up --build --scale app=1 -d     # then 2, then 4 containers (nginx load-balances on :8080)
pip install locust
locust -f locust/locustfile.py --host http://localhost:8080 --headless -u 50 -r 5 -t 60s --csv locust/results/containers_1
```
Repeat with `--scale app=2` and `--scale app=4` (change the `--csv` prefix) and fill in the table:

| Containers | Users | Requests | RPS | Median (ms) | 95th pct (ms) | Failures |
|---|---|---|---|---|---|---|
| 1 | 50 | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ |
| 2 | 50 | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ |
| 4 | 50 | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ |

## Results (test set)
_TODO: paste accuracy / precision / recall / F1 / ROC-AUC from `models/metrics.json`._

## Directory structure
```
malaria_mlops/
├── README.md
├── notebook/malaria_mlops.ipynb
├── src/ (preprocessing.py, model.py, prediction.py, retrain.py, database.py, api.py, config.py)
├── ui/streamlit_app.py
├── data/ (train/, test/, uploads/)
├── models/malaria_model.h5
├── locust/locustfile.py
├── deploy/ (start.sh, nginx.conf)
├── Dockerfile  docker-compose.yml  render.yaml
└── requirements.txt
```
