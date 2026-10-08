"""Question and answer presentation using the unchanged session RAG pipeline."""

import re

import streamlit as st

from llm.generator import INSUFFICIENT_CONTEXT, generate_answer, get_api_key, unique_sources
from classifier.predict import predict_intent
from rag.retriever import basic_retrieve, retrieve_with_intent

QUESTION_LIMIT = 10
EXAMPLE_QUESTIONS = [
    "Summarize the main findings.",
    "What methodology did the researchers use?",
    "What were the main results?",
    "What limitations did the authors identify?",
    "Compare the approaches discussed in these papers.",
]


def emphasize_citations(answer: str, sources: list[dict]) -> str:
    """Bold only known source labels; stored/generated answer text stays unchanged."""
    labels = {f"[{s['filename']}, Page {s['page']}]" for s in sources}
    if not labels:
        return answer
    pattern = "|".join(re.escape(label) for label in sorted(labels, key=len, reverse=True))
    return re.sub(pattern, lambda match: f"**{match.group(0)}**", answer)


def render_answer_section(has_papers: bool) -> None:
    st.session_state.setdefault("generated_answers", 0)
    signature = st.session_state.get("index_signature")
    if signature != st.session_state.get("answer_signature"):
        st.session_state.update(answer_signature=signature, rag_answer=None,
                                rag_sources=[], answered_question=None,
                                detected_intent=None, intent_confidence=None, retrieval_debug=[])

    remaining = max(0, QUESTION_LIMIT - st.session_state["generated_answers"])
    remaining_label = st.sidebar.empty()
    remaining_label.caption(f"Demo questions remaining: {remaining} / {QUESTION_LIMIT}")
    api_key = get_api_key()
    if not api_key:
        st.sidebar.info("Add a Gemini API key to enable answers. Setup instructions are in the README.")

    st.subheader("Ask your papers")
    store = st.session_state.get("vector_store")
    if not has_papers:
        st.info("Upload at least one research paper to begin.")
    elif store is None:
        if st.session_state.get("chunks"):
            st.info("Your documents aren't ready yet. Retry preparation in the sidebar.")
        else:
            st.info("Your papers need readable text before you can ask a question.")
    if not remaining:
        st.info("You've used the 10 questions available in this demo session.")

    with st.form("ask_papers", border=False):
        question = st.text_input("Question about your papers", key="rag_question",
                                 placeholder="What would you like to understand?")
        asked = st.form_submit_button("Ask", key="rag_ask", type="primary",
                                      disabled=not remaining or not api_key or not has_papers or store is None)

    if asked:
        st.session_state.update(rag_answer=None, rag_sources=[], answered_question=None,
                                detected_intent=None, intent_confidence=None, retrieval_debug=[])
        # Preserve server-side validation as well as the normal disabled states.
        if not remaining:
            st.info("You've used the 10 questions available in this demo session.")
        elif not api_key:
            st.info("Configure a Gemini API key before asking a question.")
        elif not question.strip():
            st.warning("Enter a question before asking.")
        elif not has_papers:
            st.info("Upload at least one research paper to begin.")
        elif store is None:
            st.info("Your papers are not ready yet. Try preparing them again in the sidebar.")
        else:
            st.session_state["answered_question"] = question.strip()
            try:
                with st.spinner("Understanding your question…"):
                    prediction = predict_intent(question.strip())
                st.session_state.update(detected_intent=prediction["intent"],
                                        intent_confidence=prediction["confidence"])
            except Exception:
                prediction = None
            try:
                with st.spinner("Finding supporting passages…"):
                    context = (retrieve_with_intent(question.strip(), prediction["intent"],
                        prediction["confidence"], store, top_k=5) if prediction else
                        basic_retrieve(question.strip(), store, top_k=5))
                st.session_state["retrieval_debug"] = context
            except Exception:
                st.error("We couldn't search your papers right now. Please retry preparing them in the sidebar.")
            else:
                if not context:
                    st.info(INSUFFICIENT_CONTEXT)
                else:
                    try:
                        with st.spinner("Preparing your answer…"):
                            answer = generate_answer(question.strip(), context, api_key)
                    except Exception:
                        # Never display exception text, including safe SDK diagnostics.
                        st.error("Unable to generate an answer right now. Please try again.")
                    else:
                        st.session_state.update(rag_answer=answer, rag_sources=unique_sources(context),
                                                answered_question=question.strip())
                        st.session_state["generated_answers"] += 1
                        remaining_label.caption(f"Demo questions remaining: {QUESTION_LIMIT - st.session_state['generated_answers']} / {QUESTION_LIMIT}")
                        if st.session_state["generated_answers"] >= QUESTION_LIMIT:
                            st.rerun()

    if st.session_state.get("rag_answer"):
        st.divider()
        st.caption(f"Question: {st.session_state['answered_question']}")
        if st.session_state.get("detected_intent"):
            st.caption(f"Detected Intent: {st.session_state['detected_intent']} · Confidence: {st.session_state['intent_confidence']:.0%}")
        st.subheader("Answer")
        st.markdown(emphasize_citations(st.session_state["rag_answer"], st.session_state["rag_sources"]))
        with st.expander("Sources used"):
            st.caption("Unique paper pages supplied as context for this answer.")
            for source in st.session_state["rag_sources"]:
                st.text(f"{source['filename']} — Page {source['page']}")
    elif has_papers and store is not None:
        st.caption("Example questions")
        st.markdown("\n".join(f"- {question}" for question in EXAMPLE_QUESTIONS))

