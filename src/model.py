"""Model creation, training and testing (evaluation)."""
import json
import time

import numpy as np

from .config import (CLASS_NAMES, IMG_SIZE, METRICS_PATH, MODEL_PATH, MODEL_TYPE,
                     MODELS_DIR)


# --------------------------------------------------------------------------
# Model creation
# --------------------------------------------------------------------------
def _augmentation(x):
    from tensorflow.keras import layers
    x = layers.RandomFlip("horizontal_and_vertical")(x)
    x = layers.RandomRotation(0.1)(x)
    return layers.RandomZoom(0.1)(x)


def build_mobilenet(dropout=0.3, l2=1e-4):
    """Transfer learning: frozen ImageNet MobileNetV2 + regularised classifier head."""
    import tensorflow as tf
    from tensorflow.keras import layers, regularizers

    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    x = _augmentation(inputs)
    x = layers.Rescaling(1 / 127.5, offset=-1)(x)          # MobileNetV2 expects [-1, 1]
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet")
    base.trainable = False
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs, outputs, name="malaria_mobilenetv2")


def build_custom_cnn(dropout=0.4, l2=1e-4):
    """A CNN trained from scratch (no pretrained weights)."""
    import tensorflow as tf
    from tensorflow.keras import layers, regularizers

    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    x = _augmentation(inputs)
    x = layers.Rescaling(1 / 255.0)(x)
    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu",
                          kernel_regularizer=regularizers.l2(l2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(l2))(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs, outputs, name="malaria_custom_cnn")


def build_model(model_type=MODEL_TYPE, **kwargs):
    return build_custom_cnn(**kwargs) if model_type == "custom_cnn" else build_mobilenet(**kwargs)


def compile_model(model, lr=1e-3):
    import tensorflow as tf
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall"),
                 tf.keras.metrics.AUC(name="auc")])
    return model


def _backbone(model):
    import tensorflow as tf
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            return layer
    return None


def unfreeze_backbone(model, n_layers=30):
    """Fine-tuning: unfreeze the last n layers of the pretrained backbone
    (BatchNorm layers stay frozen)."""
    import tensorflow as tf
    base = _backbone(model)
    if base is None:
        return model
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    return model


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def _callbacks(patience=3):
    import tensorflow as tf
    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                         restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ]


def train_model(train_ds, val_ds, epochs=8, fine_tune_epochs=4, model_type=MODEL_TYPE):
    """Train from scratch / from ImageNet. Returns (model, history dict)."""
    model = compile_model(build_model(model_type), lr=1e-3)
    hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=_callbacks())
    history = {k: list(map(float, v)) for k, v in hist.history.items()}

    if model_type == "mobilenet" and fine_tune_epochs:
        unfreeze_backbone(model)
        compile_model(model, lr=1e-5)
        hist2 = model.fit(train_ds, validation_data=val_ds, epochs=fine_tune_epochs,
                          callbacks=_callbacks(patience=2))
        for k, v in hist2.history.items():
            history.setdefault(k, []).extend(map(float, v))
    return model, history


def save_model(model, path=MODEL_PATH):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(path)
    return path


def load_trained_model(path=MODEL_PATH):
    import tensorflow as tf
    return tf.keras.models.load_model(path, compile=False)


# --------------------------------------------------------------------------
# Testing / evaluation
# --------------------------------------------------------------------------
def predict_dataset(model, ds):
    """Return (y_true, y_prob) for a tf.data dataset of (image, label)."""
    y_true, y_prob = [], []
    for images, labels in ds:
        y_prob.append(model.predict(images, verbose=0).ravel())
        y_true.append(labels.numpy().ravel())
    return np.concatenate(y_true).astype(int), np.concatenate(y_prob)


def evaluate_model(model, test_ds, threshold=0.5, save=True, history=None, extra=None):
    """Compute accuracy, precision, recall, F1, ROC-AUC, loss, confusion matrix."""
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix, f1_score, log_loss,
                                 precision_score, recall_score, roc_auc_score)

    y_true, y_prob = predict_dataset(model, test_ds)
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else None,
        "loss": float(log_loss(y_true, np.clip(y_prob, 1e-7, 1 - 1e-7), labels=[0, 1])),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "classes": CLASS_NAMES,
        "n_test": int(len(y_true)),
        "report": classification_report(y_true, y_pred, target_names=CLASS_NAMES,
                                        output_dict=True, zero_division=0),
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if history:
        metrics["history"] = history
    if extra:
        metrics.update(extra)
    if save:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    return metrics


def load_metrics():
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text())
    return None


if __name__ == "__main__":
    import argparse

    from .preprocessing import make_datasets

    ap = argparse.ArgumentParser(description="Train the malaria model offline")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--fine-tune-epochs", type=int, default=4)
    ap.add_argument("--model-type", default=MODEL_TYPE, choices=["mobilenet", "custom_cnn"])
    args = ap.parse_args()

    train_ds, val_ds, test_ds = make_datasets()
    model, history = train_model(train_ds, val_ds, args.epochs, args.fine_tune_epochs,
                                 args.model_type)
    save_model(model)
    m = evaluate_model(model, test_ds, history=history, extra={"model_type": args.model_type})
    print({k: m[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "loss")})
