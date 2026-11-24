"""FastAPI service: prediction, bulk upload, retraining trigger, monitoring, insights.

Run:  uvicorn src.api:app --host 0.0.0.0 --port 8000   (docs at /docs)
"""
import base64
import io
import threading
import time
import uuid
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from . import database, prediction, retrain
from .config import (CLASS_NAMES, IMAGE_EXTENSIONS, MODEL_PATH, RETRAIN_THRESHOLD,
                     TRAIN_DIR, UPLOAD_DIR)
from .model import load_metrics
from .preprocessing import FEATURE_NAMES, count_images, dataset_features, sample_images

app = FastAPI(title="Malaria Cell Classifier API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

START_TIME = time.time()


@app.on_event("startup")
def warm_up_model():
    """Load the model in the background at startup so the first prediction is fast."""
    def _load():
        try:
            prediction.get_model()
        except Exception as e:
            print("Model warm-up failed:", e)
    threading.Thread(target=_load, daemon=True).start()

_stats = {"predictions": 0, "errors": 0}
_features_cache = {"df": None}


# ------------------------------------------------------------------ monitoring
@app.get("/health")
def health():
    """Model up-time and basic service status."""
    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "model_file_present": MODEL_PATH.exists(),
        "model_loaded": prediction.model_loaded(),
        "predictions_served": _stats["predictions"],
        "errors": _stats["errors"],
        "pending_uploads": retrain.pending_count(),
        "retrain_threshold": RETRAIN_THRESHOLD,
    }


@app.get("/metrics")
def metrics():
    m = load_metrics()
    if m is None:
        raise HTTPException(404, "No evaluation metrics yet. Train the model first.")
    return m


# ------------------------------------------------------------------ prediction
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    try:
        result = prediction.predict_image(data)
    except FileNotFoundError as e:
        _stats["errors"] += 1
        raise HTTPException(503, str(e))
    except Exception:
        _stats["errors"] += 1
        raise HTTPException(400, "Could not read the file as an image")
    _stats["predictions"] += 1
    return result


# ------------------------------------------------------------ upload + retrain
@app.post("/upload")
async def upload(label: str = Form(...), files: List[UploadFile] = File(...)):
    """Bulk-upload labelled images (all files in one call share the given label)."""
    if label not in CLASS_NAMES:
        raise HTTPException(400, f"label must be one of {CLASS_NAMES}")
    dest = UPLOAD_DIR / label
    dest.mkdir(parents=True, exist_ok=True)
    saved, rejected = 0, []
    for f in files:
        data = await f.read()
        try:
            Image.open(io.BytesIO(data)).verify()
        except Exception:
            rejected.append(f.filename)
            continue
        name = f"{uuid.uuid4().hex[:8]}_{(f.filename or 'img.png').replace('/', '_')}"
        if not name.lower().endswith(IMAGE_EXTENSIONS):
            name += ".png"
        path = dest / name
        path.write_bytes(data)
        database.add_upload(f.filename, label, path)       # saved to the database
        saved += 1

    auto = False
    if saved and retrain.should_retrain():                  # automatic retraining trigger
        auto = retrain.start_retrain()
    return {"saved": saved, "rejected": rejected,
            "pending_total": retrain.pending_count(),
            "auto_retrain_started": auto}


@app.post("/retrain")
def trigger_retrain(epochs: int = 3):
    """Manual retraining trigger (the UI button calls this)."""
    started = retrain.start_retrain(epochs=epochs)
    if not started:
        raise HTTPException(409, "A retraining job is already running")
    return {"message": "Retraining started", "pending_images": retrain.pending_count()}


@app.get("/retrain/status")
def retrain_status():
    return {**retrain.get_status(), "pending_images": retrain.pending_count(),
            "uploads": database.upload_summary()}


# -------------------------------------------------------------------- insights
@app.get("/insights/counts")
def insight_counts():
    return {"train": count_images(TRAIN_DIR)}


@app.get("/insights/features")
def insight_features(refresh: bool = False, per_class: int = 200):
    if _features_cache["df"] is None or refresh:
        _features_cache["df"] = dataset_features(per_class=per_class)
    df = _features_cache["df"]
    return {"features": FEATURE_NAMES, "rows": df.to_dict(orient="records")}


@app.get("/insights/samples")
def insight_samples(n: int = 4):
    out = {}
    for cls, paths in sample_images(n=n).items():
        thumbs = []
        for p in paths:
            img = Image.open(p).convert("RGB")
            img.thumbnail((128, 128))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            thumbs.append(base64.b64encode(buf.getvalue()).decode())
        out[cls] = thumbs
    return out
