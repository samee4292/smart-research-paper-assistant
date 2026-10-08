"""Independent classifier development panel; never changes the RAG pipeline."""

import streamlit as st
from contextlib import nullcontext

from classifier.predict import MODEL_PATH, predict_intent


def render_intent_classifier(embedded: bool = False):
    with nullcontext() if embedded else st.expander("Test Intent Classifier"):
        if embedded:
            st.subheader("Test Intent Classifier")
        st.caption("Custom trained PyTorch FFNN · Testing only; retrieval and Gemini are unaffected.")
        if not MODEL_PATH.exists():
            st.info("Model not trained yet. Run: python -m classifier.train")
            return
        question = st.text_input("Question to classify", key="intent_question")
        if st.button("Classify", key="classify_intent"):
            if not question.strip():
                st.warning("Enter a question to classify.")
                return
            try:
                with st.spinner("Classifying question locally…"):
                    prediction = predict_intent(question)
            except Exception:
                st.error("Could not load or run the classifier. Check the model files and try retraining.")
            else:
                st.text(f"Predicted Intent: {prediction['intent']}")
                st.text(f"Confidence: {prediction['confidence']:.1%}")
                st.caption("Confidence is an uncalibrated model score, not a guarantee of correctness.")
