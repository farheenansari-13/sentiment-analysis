# app.py
"""
Streamlit Sentiment Analysis App
Place `vectorizer.joblib` and `sentiment_model.joblib` in the same folder as this file.
Requirements (put in requirements.txt):
streamlit
scikit-learn
joblib
pandas
"""

import streamlit as st
from joblib import load
import os
import pandas as pd
import numpy as np

st.set_page_config(page_title="Sentiment Classifier", page_icon="📝", layout="centered")

@st.cache_resource
def load_artifacts(vectorizer_path="vectorizer.joblib", model_path="sentiment_model.joblib"):
    """Load vectorizer and model from disk. Returns tuple (vectorizer, model)."""
    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(f"Vectorizer not found at {vectorizer_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    vectorizer = load(vectorizer_path)
    model = load(model_path)
    return vectorizer, model

def predict_text(text, vectorizer, model):
    """Return prediction, probabilities (if available), and class names."""
    X = vectorizer.transform([text])
    # Determine class labels if model exposes them
    classes = None
    if hasattr(model, "classes_"):
        classes = list(model.classes_)
    # Try predict_proba
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]  # shape (n_classes,)
        # create dataframe for display
        if classes is None:
            classes = [str(i) for i in range(len(probs))]
        prob_df = pd.DataFrame({"label": classes, "probability": probs})
        # sort descending probability for nicer display
        prob_df = prob_df.sort_values("probability", ascending=False).reset_index(drop=True)
        predicted_label = prob_df.loc[0, "label"]
        predicted_prob = prob_df.loc[0, "probability"]
        return predicted_label, float(predicted_prob), prob_df
    else:
        # fallback to predict()
        pred = model.predict(X)[0]
        return pred, None, None

# ---------- App UI ----------
st.title("📝 Sentiment Analysis — Streamlit")
st.write(
    "Enter a sentence or paragraph and click **Predict**. "
    "The app loads `vectorizer.joblib` and `sentiment_model.joblib` from the app folder."
)

with st.sidebar:
    st.header("Model files")
    st.markdown(
        "If you want to use custom files, upload them here. "
        "Uploaded files will override files in the repo for this session."
    )
    vec_uploader = st.file_uploader("Upload vectorizer.joblib", type=["joblib"])
    model_uploader = st.file_uploader("Upload sentiment_model.joblib", type=["joblib"])
    st.markdown("---")
    st.markdown("Deployment tips:")
    st.markdown("- Put `app.py`, `vectorizer.joblib`, and `sentiment_model.joblib` in your repo root.")
    st.markdown("- Add `requirements.txt` with packages: streamlit, scikit-learn, joblib, pandas.")
    st.markdown("- Deploy on Streamlit Cloud or any server that can run Streamlit.")
    st.caption("Your timezone: Asia/Kolkata")

# If user uploads files in the sidebar, save them to temporary filenames and load from them.
VECTOR_FILE = "vectorizer.joblib"
MODEL_FILE = "sentiment_model.joblib"

if vec_uploader is not None:
    with open(VECTOR_FILE, "wb") as f:
        f.write(vec_uploader.getbuffer())
    st.sidebar.success("Uploaded vectorizer.joblib")

if model_uploader is not None:
    with open(MODEL_FILE, "wb") as f:
        f.write(model_uploader.getbuffer())
    st.sidebar.success("Uploaded sentiment_model.joblib")

# Load artifacts (with error handling)
try:
    vectorizer, model = load_artifacts(VECTOR_FILE, MODEL_FILE)
except Exception as e:
    st.error(f"Error loading artifacts: {e}")
    st.stop()

# Example inputs
examples = [
    "I love this product! It works perfectly.",
    "This is the worst experience I've had with a service.",
    "The movie was okay, not great but not terrible either."
]

st.subheader("Input")
text = st.text_area("Type text to analyze", value=examples[0], height=150)

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("Predict"):
        if not text.strip():
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("Predicting..."):
                try:
                    label, prob, prob_df = predict_text(text, vectorizer, model)
                except Exception as e:
                    st.error(f"Prediction error: {e}")
                else:
                    st.markdown("### Result")
                    st.write(f"**Predicted label:** `{label}`")
                    if prob is not None:
                        st.write(f"**Confidence:** {prob:.3f}")
                    # show probability table if available
                    if prob_df is not None:
                        st.markdown("**Probabilities:**")
                        # display nicely
                        prob_df["probability"] = prob_df["probability"].apply(lambda x: float(x))
                        st.dataframe(prob_df)
with col2:
    st.subheader("Quick examples")
    for ex in examples:
        if st.button(ex, key=ex):
            text = ex
            # rerun with the example inserted - set in session_state so the textarea updates
            st.experimental_rerun()

st.markdown("---")
st.markdown("If your model uses numeric labels (e.g., 0 / 1), the app will display whatever labels `model.classes_` contains.")
st.markdown("Need any extra features (batch CSV prediction, explanation, custom label mapping)? Tell me and I’ll add them.")

# Optional: let user download result as CSV after prediction
if "prob_df" in locals() and prob_df is not None:
    csv = prob_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download probabilities CSV", data=csv, file_name="prediction_probs.csv", mime="text/csv")

