"""Model retraining: ingest uploaded data, fine-tune, evaluate, hot-swap the model."""
import json
import shutil
import threading
import time
import uuid
from pathlib import Path

from . import database
from .config import (CLASS_NAMES, MODEL_PATH, MODELS_DIR, PREV_MODEL_PATH,
                     RETRAIN_STATUS_PATH, RETRAIN_THRESHOLD, TRAIN_DIR)

_run_lock = threading.Lock()


def _write_status(**kw):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RETRAIN_STATUS_PATH.write_text(json.dumps(kw, indent=2))


def get_status():
    if RETRAIN_STATUS_PATH.exists():
        return json.loads(RETRAIN_STATUS_PATH.read_text())
    return {"state": "idle"}


def pending_count():
    return len(database.pending_uploads())


def should_retrain(threshold=RETRAIN_THRESHOLD):
    """Trigger rule: enough new labelled images are waiting."""
    return pending_count() >= threshold


def ingest_uploads():
    """Move pending uploads into data/train/<class>/ and mark them in the DB."""
    rows = database.pending_uploads()
    moved, ids = 0, []
    for r in rows:
        src = Path(r["path"])
        if not src.exists() or r["label"] not in CLASS_NAMES:
            continue
        dest_dir = TRAIN_DIR / r["label"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), dest_dir / f"upload_{uuid.uuid4().hex[:8]}_{src.name}")
        ids.append(r["id"])
        moved += 1
    database.mark_ingested(ids)
    return moved


def retrain(epochs=3):
    """Full retraining run. Blocking; use start_retrain() to run in the background."""
    from .model import (compile_model, evaluate_model, load_trained_model, build_model,
                        save_model, unfreeze_backbone, _callbacks)
    from .preprocessing import make_datasets
    from . import prediction

    if not _run_lock.acquire(blocking=False):
        return {"state": "running", "message": "A retraining job is already running"}
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        _write_status(state="running", step="ingesting uploaded data", started_at=started)
        n_new = ingest_uploads()

        _write_status(state="running", step="preparing datasets", started_at=started,
                      new_images=n_new)
        train_ds, val_ds, test_ds = make_datasets()

        _write_status(state="running", step="training", started_at=started, new_images=n_new)
        if MODEL_PATH.exists():
            model = load_trained_model(MODEL_PATH)          # continue from current model
            shutil.copy2(MODEL_PATH, PREV_MODEL_PATH)       # keep a rollback copy
        else:
            model = build_model()
        unfreeze_backbone(model)
        compile_model(model, lr=1e-5 if MODEL_PATH.exists() else 1e-3)
        hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                         callbacks=_callbacks(patience=2))
        history = {k: list(map(float, v)) for k, v in hist.history.items()}

        _write_status(state="running", step="evaluating", started_at=started, new_images=n_new)
        save_model(model)
        metrics = evaluate_model(model, test_ds, history=history,
                                 extra={"retrained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                                        "new_images_used": n_new})
        prediction.reload_model()
        status = {"state": "done", "started_at": started,
                  "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                  "new_images": n_new,
                  "accuracy": metrics["accuracy"], "f1": metrics["f1"]}
        _write_status(**status)
        return status
    except Exception as e:  # surface the error to the UI
        status = {"state": "failed", "started_at": started, "error": str(e)}
        _write_status(**status)
        return status
    finally:
        _run_lock.release()


def start_retrain(epochs=3):
    """Kick off retraining in a background thread. Returns False if already running."""
    if _run_lock.locked():
        return False
    threading.Thread(target=retrain, kwargs={"epochs": epochs}, daemon=True).start()
    return True
