# Malaria Cell Classifier: End-to-End MLOps Pipeline

An image-classification system that decides whether a microscope image of a single blood cell is **Parasitized** (malaria) or **Uninfected**. It covers the full ML cycle: data acquisition, preprocessing, model creation, testing, prediction, retraining, API, UI, Docker packaging and cloud deployment.

## Links

-  **Video demo (YouTube):** https://youtu.be/6Xqf0ALhmqk
-  **GitHub repo:** https://github.com/davidngarambe/malaria-mlops-summative

## Project description

The system takes a cell image, resizes it to 128x128 and classifies it with a MobileNetV2 transfer-learning model. Users can:

1. **Predict** one image and see the label, confidence and response time.
2. **Explore the data** through charts of six image features and a written interpretation.
3. **Upload new labelled images in bulk** (saved to a SQLite database and to the training data).
4. **Retrain the model** with a button, or automatically when 50 or more uploaded images are waiting. The new model is evaluated and swapped in without restarting the service.
5. **Monitor the service**: status, up-time, predictions served, pending uploads and test-set metrics.

**Dataset:** NIH Malaria Cell Images (Kaggle: `iarunava/cell-images-for-detecting-malaria`), two classes. The repository contains only a small sample (`data/train`, `data/test`) to keep it light. The full dataset is downloaded by `src/preprocessing.py`.

**Model:** MobileNetV2 (ImageNet weights) with a regularised classifier head. Optimisation techniques: L2 regularisation, dropout, Adam, learning-rate reduction on plateau, early stopping with best-weights restore, data augmentation inside the model, and a second fine-tuning phase on the last 30 backbone layers. Labels: `0 = Uninfected`, `1 = Parasitized`.

## Results

Test set of 600 images (300 per class), saved in `models/metrics.json`:

| Metric | Value |
|---|---|
| Accuracy | 0.925 |
| Precision | 0.941 |
| Recall | 0.907 |
| F1 | 0.924 |
| ROC-AUC | 0.976 |

Confusion matrix (rows = actual, columns = predicted): Uninfected 283 / 17, Parasitized 28 / 272.

Retraining in the demo uses only the small sample in this repo, so its metrics are noisier than the numbers above. In production the model would be retrained on the full dataset.

## Repository structure

```
malaria-mlops-summative/
├── README.md
├── notebook/malaria_mlops.ipynb      preprocessing, training, evaluation, prediction
├── src/
│   ├── config.py                     paths and settings
│   ├── preprocessing.py              data acquisition, loading, image features
│   ├── model.py                      model creation, training, evaluation
│   ├── prediction.py                 single-image prediction
│   ├── retrain.py                    ingest uploads, retrain, hot-swap the model
│   ├── database.py                   SQLite log of uploaded images
│   └── api.py                        FastAPI service
├── ui/streamlit_app.py               Streamlit user interface
├── data/ (train/, test/)             small sample of the dataset
├── models/                           malaria_model.h5 and metrics.json
├── locust/locustfile.py              flood-request simulation script
├── deploy/ (start.sh, nginx.conf)
├── Dockerfile  docker-compose.yml  render.yaml
└── requirements.txt  requirements-dev.txt
```

## How to run it

### Option 1: Docker (recommended)

```bash
git clone https://github.com/davidngarambe/malaria-mlops-summative.git
cd malaria-mlops-summative
docker build -t malaria-classifier .
docker run -p 8000:8000 -p 8501:8501 malaria-classifier
```

- UI: http://localhost:8501
- API and Swagger docs: http://localhost:8000/docs

### Option 2: Google Colab (no Docker needed)

```python
!git clone -q https://github.com/davidngarambe/malaria-mlops-summative.git
%cd malaria-mlops-summative
!pip install -q fastapi "uvicorn[standard]" python-multipart streamlit seaborn
```

Then start `uvicorn src.api:app --port 8000` and `streamlit run ui/streamlit_app.py --server.port 8501` as background processes and open port 8501 through a tunnel.

### Option 3: Local Python

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn src.api:app --port 8000
API_URL=http://localhost:8000 streamlit run ui/streamlit_app.py
```

### Re-training the model from scratch (offline)

```bash
pip install -r requirements-dev.txt
python -m src.preprocessing --per-class 2000     # needs Kaggle credentials
python -m src.model --epochs 8 --fine-tune-epochs 4
```

## API endpoints

| Endpoint | Purpose |
|---|---|
| `POST /predict` | Predict one image (`file`) |
| `POST /upload` | Bulk upload labelled images (`label`, `files`) |
| `POST /retrain` | Start retraining |
| `GET /retrain/status` | Retraining progress and upload counts |
| `GET /health` | Status, up-time, predictions served, pending uploads |
| `GET /metrics` | Latest test-set metrics |
| `GET /insights/*` | Class counts, image features, sample cells |

## Retraining process

1. Images are uploaded through the UI (`POST /upload`) and recorded in SQLite.
2. Retraining starts from the button (`POST /retrain`) or automatically once the pending count reaches `RETRAIN_THRESHOLD` (default 50).
3. Pending images are moved into `data/train/<class>/`, the current model is fine-tuned at a low learning rate, evaluated on `data/test`, saved, and loaded by the API without a restart. A backup of the previous model is kept.

## Flood-request simulation (Locust)

The Locust script is `locust/locustfile.py`, and `docker-compose.yml` with `deploy/nginx.conf` runs several containers behind a load balancer (`docker compose up --scale app=N`). The flood test with 1, 2 and 4 containers was not completed before the submission deadline, so no results table is included.

Run it with:

```bash
docker compose up --build --scale app=2 -d
pip install locust
locust -f locust/locustfile.py --host http://localhost:8080 --headless -u 50 -r 5 -t 60s
```

## Known limitations

- The repository holds only a small sample of the dataset, so retraining inside the demo is for showing the pipeline, not for improving accuracy.
- The flood-test results with different container counts are not included.
