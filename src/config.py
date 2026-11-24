"""Central configuration for the malaria MLOps pipeline."""
import os
from pathlib import Path

ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parent.parent))

DATA_DIR = ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "app.db"

MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "malaria_model.h5"
PREV_MODEL_PATH = MODELS_DIR / "malaria_model_previous.h5"
METRICS_PATH = MODELS_DIR / "metrics.json"
RETRAIN_STATUS_PATH = MODELS_DIR / "retrain_status.json"

# Label 0 = Uninfected, label 1 = Parasitized (positive class = disease present).
# image_dataset_from_directory assigns labels in this order.
CLASS_NAMES = ["Uninfected", "Parasitized"]
IMG_SIZE = (128, 128)
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
SEED = 42

# "mobilenet" (transfer learning) or "custom_cnn" (trained from scratch)
MODEL_TYPE = os.getenv("MODEL_TYPE", "mobilenet")

# Automatic retraining trigger: when this many new labelled images are waiting.
RETRAIN_THRESHOLD = int(os.getenv("RETRAIN_THRESHOLD", "50"))

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")
