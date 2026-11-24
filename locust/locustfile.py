"""Flood-request simulation against the prediction API.

Headless example (60 s, 50 users, 5 new users/s), CSV results saved in locust/results/:
    locust -f locust/locustfile.py --host http://localhost:8080 \
           --headless -u 50 -r 5 -t 60s --csv locust/results/containers_1

Web UI: locust -f locust/locustfile.py --host http://localhost:8080  -> http://localhost:8089
"""
import io
import os
import random
from pathlib import Path

from locust import HttpUser, between, task
from PIL import Image

SAMPLE_DIR = Path(os.getenv("SAMPLE_DIR", Path(__file__).resolve().parent.parent / "data" / "test"))


def _load_samples(limit=20):
    files = [p for p in SAMPLE_DIR.rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
    random.shuffle(files)
    return [p.read_bytes() for p in files[:limit]]


def _fake_cell():
    img = Image.effect_noise((128, 128), 60).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


SAMPLES = _load_samples() or [_fake_cell()]


class PredictionUser(HttpUser):
    wait_time = between(0.1, 1.0)

    @task(8)
    def predict(self):
        self.client.post("/predict", files={"file": ("cell.png", random.choice(SAMPLES), "image/png")},
                         name="/predict")

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")
