"""Streamlit UI: up-time, prediction, data insights, bulk upload and retraining."""
import base64
import io
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="Malaria Cell Classifier", page_icon="🦟", layout="wide")

# Hide Streamlit's top-right "three dots" menu and the "Deploy" button
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    [data-testid="stMainMenu"] {display: none;}
    [data-testid="stToolbar"] {display: none;}
    [data-testid="stAppDeployButton"] {display: none;}
    </style>
    """,
    unsafe_allow_html=True,
)


def api_get(path, **kw):
    r = requests.get(f"{API_URL}{path}", timeout=kw.pop("timeout", 30), **kw)
    r.raise_for_status()
    return r.json()


def fmt_uptime(sec):
    sec = int(sec)
    return f"{sec // 3600}h {sec % 3600 // 60}m {sec % 60}s"


st.sidebar.title("🦟 Malaria Cell Classifier")
page = st.sidebar.radio("Navigate", ["Dashboard", "Predict", "Data insights",
                                      "Upload & Retrain"])
st.sidebar.caption(f"API: {API_URL}")

# =============================================================== Dashboard
if page == "Dashboard":
    st.title("Model dashboard")
    try:
        h = api_get("/health")
    except Exception as e:
        st.error(f"API unreachable: {e}")
        st.stop()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status", h["status"].upper())
    c2.metric("Up-time", fmt_uptime(h["uptime_seconds"]))
    c3.metric("Predictions served", h["predictions_served"])
    c4.metric("Pending upload images", f"{h['pending_uploads']} / {h['retrain_threshold']}")
    st.caption("Retraining starts automatically when pending images reach the threshold.")

    st.subheader("Model performance on the held-out test set")
    try:
        m = api_get("/metrics")
    except Exception:
        st.warning("No metrics yet — train the model first.")
        st.stop()
    cols = st.columns(5)
    for col, key in zip(cols, ["accuracy", "precision", "recall", "f1", "roc_auc"]):
        col.metric(key.upper().replace("_", "-"), f"{m[key]:.3f}" if m.get(key) is not None else "n/a")
    left, right = st.columns(2)
    with left:
        cm = np.array(m["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(4, 3.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=m["classes"], yticklabels=m["classes"], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion matrix")
        st.pyplot(fig)
    with right:
        hist = m.get("history")
        if hist and "loss" in hist:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.plot(hist["loss"], label="train loss")
            if "val_loss" in hist:
                ax.plot(hist["val_loss"], label="val loss")
            ax.set_xlabel("epoch")
            ax.legend()
            ax.set_title("Training curves")
            st.pyplot(fig)
    st.caption(f"Last evaluated: {m.get('evaluated_at', 'n/a')} · "
               f"test images: {m.get('n_test', 'n/a')}")

# ================================================================ Predict
elif page == "Predict":
    st.title("Predict a single cell image")
    f = st.file_uploader("Upload a blood-smear cell image", type=["png", "jpg", "jpeg"])
    if f is not None:
        left, right = st.columns([1, 2])
        left.image(f, caption=f.name, width=220)
        if right.button("Predict", type="primary"):
            t0 = time.time()
            r = requests.post(f"{API_URL}/predict",
                              files={"file": (f.name, f.getvalue(), f.type)}, timeout=60)
            latency = (time.time() - t0) * 1000
            if r.ok:
                res = r.json()
                color = "red" if res["label"] == "Parasitized" else "green"
                right.markdown(f"### Prediction: :{color}[{res['label']}]")
                right.progress(min(max(res["confidence"], 0.0), 1.0),
                               text=f"Confidence {res['confidence']:.1%}")
                right.json(res["probabilities"])
                right.caption(f"Round-trip latency: {latency:.0f} ms")
            else:
                right.error(f"Error {r.status_code}: {r.text}")

# =========================================================== Data insights
elif page == "Data insights":
    st.title("What does the data tell us?")
    try:
        counts = api_get("/insights/counts")["train"]
        feats = api_get("/insights/features", timeout=300)
        samples = api_get("/insights/samples")
    except Exception as e:
        st.error(f"Could not load insights: {e}")
        st.stop()
    df = pd.DataFrame(feats["rows"])
    if df.empty:
        st.warning("No training images found.")
        st.stop()

    st.subheader("1. Class balance")
    st.bar_chart(pd.Series(counts, name="images"))

    st.subheader("2. Example cells")
    cols = st.columns(8)
    i = 0
    for cls, thumbs in samples.items():
        for t in thumbs:
            cols[i % 8].image(base64.b64decode(t), caption=cls, width=100)
            i += 1

    st.subheader("3. Image features by class")
    names = feats["features"]
    pick = st.multiselect("Features", names, default=names[:3])
    for name in pick:
        c1, c2 = st.columns(2)
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.kdeplot(data=df, x=name, hue="label", fill=True, common_norm=False, ax=ax)
        ax.set_title(f"{name} — distribution")
        c1.pyplot(fig)
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.boxplot(data=df, x="label", y=name, ax=ax)
        ax.set_title(f"{name} — by class")
        c2.pyplot(fig)

    st.subheader("4. Interpretation (computed from the data)")
    summary = df.groupby("label")[names].mean().T
    summary["difference %"] = ((summary.get("Parasitized", 0) - summary.get("Uninfected", 0))
                               / summary.get("Uninfected", 1).replace(0, np.nan) * 100)
    st.dataframe(summary.style.format("{:.3f}"))
    top = summary["difference %"].abs().sort_values(ascending=False).head(3).index
    for feat in top:
        d = summary.loc[feat, "difference %"]
        direction = "higher" if d > 0 else "lower"
        st.markdown(f"- **{feat}**: Parasitized cells have a mean {abs(d):.1f}% {direction} "
                    f"than uninfected cells.")
    st.caption("The three features with the largest class gap are listed above. "
               "These simple statistics hint at what the CNN can pick up on "
               "(e.g. stained parasite regions raise the share of dark pixels).")
    corr = df[names].corr()
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Feature correlation")
    st.pyplot(fig)

# ========================================================= Upload & Retrain
else:
    st.title("Upload new data & retrain")
    st.subheader("1. Upload labelled images (bulk)")
    label = st.selectbox("Class of ALL images in this batch", ["Parasitized", "Uninfected"])
    files = st.file_uploader("Select multiple images", type=["png", "jpg", "jpeg"],
                             accept_multiple_files=True)
    if files and st.button(f"Upload {len(files)} images as {label}"):
        payload = [("files", (f.name, f.getvalue(), f.type)) for f in files]
        r = requests.post(f"{API_URL}/upload", data={"label": label}, files=payload, timeout=300)
        if r.ok:
            res = r.json()
            st.success(f"Saved {res['saved']} images · pending total: {res['pending_total']}")
            if res["rejected"]:
                st.warning(f"Rejected (not valid images): {res['rejected']}")
            if res["auto_retrain_started"]:
                st.info("Pending images reached the threshold — automatic retraining started.")
        else:
            st.error(r.text)

    st.subheader("2. Trigger retraining")
    epochs = st.slider("Epochs", 1, 10, 3)
    if st.button("🔁 Retrain model now", type="primary"):
        r = requests.post(f"{API_URL}/retrain", params={"epochs": epochs}, timeout=30)
        st.success("Retraining started") if r.ok else st.error(r.text)

    st.subheader("3. Retraining status")
    try:
        s = api_get("/retrain/status")
        st.json(s)
        if s.get("state") == "running":
            time.sleep(5)
            st.rerun()
        elif s.get("state") == "done":
            st.success("Retraining finished — the new model is live. "
                       "See the Dashboard for updated metrics.")
    except Exception as e:
        st.error(e)
