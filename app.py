"""Research-paper workspace: documents in the sidebar, answers in the main view."""

import streamlit as st

from rag.chunker import chunk_pages
from research_assistant.pdf_extraction import extract_pdf
from research_assistant.search_ui import ensure_session_index
from research_assistant.answer_ui import render_answer_section
from research_assistant.technical_ui import render_technical_details


def main() -> None:
    st.set_page_config(page_title="Smart Research Paper Assistant",
                       page_icon=":material/menu_book:", layout="centered")
    st.title("Smart Research Paper Assistant")
    st.write("Ask questions across research papers using intent-aware retrieval and grounded AI responses.")
    st.caption("Upload research papers, ask questions, and receive answers grounded in the uploaded documents with page-level citations.")

    with st.sidebar:
        st.header("Research Papers")
        uploads = st.file_uploader(
            "Upload PDFs", type=["pdf"], accept_multiple_files=True,
            help="Choose one or more text-based PDFs. Scanned pages need OCR, which is not included.",
        )

        # Preserve the existing session-only extraction cache and upload identity.
        previous_results = st.session_state.get("extractions", {})
        current_results, results = {}, []
        for upload in uploads or []:
            content = upload.getvalue()
            key = (upload.name, content)
            if key in previous_results:
                result = previous_results[key]
            else:
                with st.spinner(f"Reading {upload.name}…"):
                    result = extract_pdf(upload.name, content)
            current_results[key] = result
            results.append(result)
        st.session_state["extractions"] = current_results

        chunks = chunk_pages(page for result in results if not result.error for page in result.pages)
        st.session_state["chunks"] = chunks
        ensure_session_index(chunks)
        store = st.session_state.get("vector_store")
        papers, pages, indexed = st.columns(3)
        papers.metric("Papers", len(results))
        pages.metric("Pages", sum(len(r.pages) for r in results if not r.error))
        indexed.metric("Chunks", store.index.ntotal if store is not None else 0,
                       help="Chunks indexed from the current documents.")

        for result in results:
            st.text(result.filename)
            if result.error:
                st.error(result.error)
            else:
                page_label = "page" if result.page_count == 1 else "pages"
                st.caption(f"{result.page_count} {page_label} · {result.status}")
                if result.warnings:
                    st.caption("Some pages contain no readable text. Check document inspection in Technical details.")

        if store is not None:
            st.success("Documents indexed and ready", icon=":material/check_circle:")
        elif chunks:
            st.error("We couldn't prepare your documents. Please retry.")
            if st.button("Retry indexing", key="retry_indexing"):
                st.session_state.pop("index_signature", None)
                st.rerun()
        elif results:
            st.info("Choose a PDF with selectable text to begin.")

    render_answer_section(has_papers=bool(results))
    st.sidebar.caption("Documents stay in this session. Selected passages are sent to Gemini when you ask a question.")

    with st.expander("How does intent detection help?"):
        st.write("The application uses a custom feed-forward neural network to classify research questions as Summary, Comparison, Methodology, Results, Limitations, or Definition. High-confidence predictions are used to guide retrieval toward more relevant evidence. Low-confidence predictions fall back to standard semantic retrieval.")

    render_technical_details(results, chunks)


if __name__ == "__main__":
    main()

