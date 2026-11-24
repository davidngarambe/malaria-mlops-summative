"""Single-image prediction (used by the API, the UI and the notebook)."""
import threading

from .config import CLASS_NAMES, MODEL_PATH
from .preprocessing import load_image_array

_model = None
_model_mtime = None
_lock = threading.Lock()


def get_model():
    """Lazy-load the saved model; reloads automatically if the file changed (retraining)."""
    global _model, _model_mtime
    with _lock:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"No trained model at {MODEL_PATH}. Train one first.")
        mtime = MODEL_PATH.stat().st_mtime
        if _model is None or mtime != _model_mtime:
            from .model import load_trained_model
            _model = load_trained_model(MODEL_PATH)
            _model_mtime = mtime
        return _model


def model_loaded():
    return _model is not None


def reload_model():
    global _model
    with _lock:
        _model = None
    return get_model()


def predict_image(source, threshold=0.5):
    """Predict one cell image (path, bytes or file-like).

    Returns {"label", "confidence", "probabilities"}; label 1 = Parasitized.
    """
    import numpy as np

    arr = load_image_array(source)
    prob_parasitized = float(get_model().predict(np.expand_dims(arr, 0), verbose=0)[0][0])
    idx = int(prob_parasitized >= threshold)
    return {
        "label": CLASS_NAMES[idx],
        "confidence": prob_parasitized if idx == 1 else 1 - prob_parasitized,
        "probabilities": {"Uninfected": 1 - prob_parasitized, "Parasitized": prob_parasitized},
    }
