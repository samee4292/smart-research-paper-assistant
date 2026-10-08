"""Reusable local inference; no Gemini, retrieval, or classifier API calls."""

from pathlib import Path

import streamlit as st
import torch

from classifier.model import INTENTS, MODEL_CONFIG, IntentClassifier
from rag.embeddings import MODEL_NAME, embed_texts

MODEL_PATH = Path(__file__).resolve().parent / "intent_classifier.pt"


@st.cache_resource(show_spinner=False, max_entries=1)
def _load_classifier(path: str, file_version: tuple[int, int]):
    # File modification time/size invalidate the cache when retraining finishes.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    if (config["labels"] != INTENTS or config["embedding_model"] != MODEL_NAME
            or config["architecture"] != MODEL_CONFIG or not config["normalize_embeddings"]):
        raise ValueError("Classifier configuration does not match the current feature extractor/architecture.")
    model = IntentClassifier(**config["architecture"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()  # Dropout must be disabled for repeatable predictions.
    return model


def predict_intent(question: str) -> dict:
    """Return the predicted class and its uncalibrated softmax confidence."""
    if not question.strip():
        raise ValueError("Enter a question to classify.")
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Train the classifier first: python -m classifier.train")
    info = MODEL_PATH.stat()
    model = _load_classifier(str(MODEL_PATH), (info.st_mtime_ns, info.st_size))
    vector = torch.from_numpy(embed_texts([question.strip()]))
    with torch.inference_mode():
        probabilities = torch.softmax(model(vector), dim=1)[0]
    index = int(probabilities.argmax().item())
    return {"intent": INTENTS[index], "confidence": float(probabilities[index].item())}
