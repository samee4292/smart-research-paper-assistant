# Smart Research Paper Assistant

A research-paper assistant that retrieves evidence from uploaded PDFs and generates grounded Gemini answers with page-level citations. A custom trained neural network identifies the question's intent and guides retrieval when its confidence is high enough.

## Demo

**Deployed Streamlit URL:** [Click Here](https://smart-research-paper-assistant.streamlit.app)

Run the app locally today. Documents are managed in the sidebar; the main view focuses on questions, answers, and sources.

## What It Does

- Upload multiple research papers in PDF format.
- Extract text page by page and preserve filename/page metadata.
- Create overlapping, page-aware chunks and retrieve them by semantic similarity.
- Classify questions as Summary, Comparison, Methodology, Results, Limitations, or Definition.
- Enhance the original FAISS search with intent-guided queries or comparison source diversity.
- Generate Gemini answers from the selected evidence, with `[paper.pdf, Page 4]` citations.
- Inspect unique context sources and optionally expand technical details.

The normal interface hides development tools. Open **Technical details → Show development tools** to inspect extracted text/chunks, test basic semantic search, or test the classifier independently.

## Architecture

```text
PDFs
  ↓
PyMuPDF text extraction — filename + one-based page number
  ↓
Page-aware chunking — up to 250 words, 40-word overlap
  ↓
MiniLM embeddings — normalized 384-dimensional vectors
  ↓
FAISS in-memory vector index + chunk metadata

User question
  ↓
MiniLM question embedding
  ↓
Custom feed-forward neural network
  ↓
Intent + confidence
  ↓
Existing FAISS search of the ORIGINAL question (always)
  ├─ Confidence < 0.70: standard semantic retrieval
  └─ Confidence ≥ 0.70:
       ├─ Intent query expansion + merge/deduplicate
       └─ Comparison: encourage relevant source diversity
  ↓
Up to 5 selected evidence chunks
  ↓
Original question + evidence + source metadata → Gemini
  ↓
Grounded answer + validated filename/page citations
```

The original question's strongest result is retained during intent expansion. Ranked candidate lists are combined using reciprocal-rank fusion; no reranking model is used. Comparisons can use one paper or multiple papers. A second paper is included only when its similarity passes the existing relevance heuristic.

Gemini receives the original question and selected chunk text/source metadata. It does **not** receive PDF files, full papers, unretrieved chunks, embeddings, the index, or previous answers. When the context is insufficient, the app asks it to respond:

> I couldn't find enough information in the uploaded papers to answer this question.

Citation source IDs are validated and citation labels are built from retrieved metadata. This prevents invented source identities; it does not guarantee that every statement is supported, so users should inspect the passages.

## Custom Neural Network

MiniLM is a pretrained, frozen text encoder. The **feed-forward intent classifier was trained specifically for this project** using PyTorch; Gemini does not classify questions.

- **Input:** a normalized, 384-dimensional MiniLM question embedding.
- **Architecture:** `384 → 128 → 64 → 6`, with ReLU activations and dropout during training.
- **Outputs:** Summary, Comparison, Methodology, Results, Limitations, Definition.
- **Dataset:** 390 authored examples, balanced at 65 per intent.
- **Training:** Adam, cross-entropy loss, and checkpoint selection by validation loss.

The saved classifier is included under `classifier/`. You do not need to retrain it to run the app. See [classifier/README.md](classifier/README.md) for training details.

## Evaluation

Phase 7 used a fixed **60-question classifier evaluation** and **12 retrieval questions** with manually assigned relevant pages from the existing synthetic paper fixture.

| Measurement | Before / Basic | After / Intent-aware |
|---|---:|---:|
| Classifier accuracy | 86.67% (52/60) | 93.33% (56/60) |
| Retrieval Top-3 hit rate | 83.33% (10/12) | 91.67% (11/12) |
| Retrieval Top-5 hit rate | 100% (12/12) | 100% (12/12) |

There was **no Top-5 retrieval improvement**. One limitations question gained a Top-3 hit; no measured hit worsened on this fixture. These results come from a small project-specific development evaluation and **must not be interpreted as broad benchmark performance**. Baseline errors informed the training additions, so the evaluation is not an untouched final test set.

RESULTS/METHODOLOGY confusion remains. The known question, “What performance did the adaptive planner achieve?”, still predicts METHODOLOGY after retraining, but confidence fell from 81.1% to 50.9%. This activates the existing **0.70 confidence fallback** to standard semantic retrieval. Softmax confidence is uncalibrated and is not a guarantee of correctness.

See [the measured evaluation report](evaluation/results.md) for per-intent accuracies, confusion matrices, errors, and question-level retrieval results. [results.json](evaluation/results.json) also contains confidences, retrieved text/scores, and model/data hashes. Original artifacts are preserved in `evaluation/baseline/`.

## Tech Stack

Python · Streamlit · PyMuPDF · Sentence Transformers · all-MiniLM-L6-v2 · FAISS · PyTorch · Google Gemini API

NumPy supports vector operations; scikit-learn supports training splits and evaluation metrics.

## Running Locally

Use **Python 3.12** and Streamlit **1.65 or newer**, then run these commands
from the project folder. The dependency file specifies the tested UI API floor.

### Windows / PowerShell

```powershell
py -3.12 -m venv .venv-search
.\.venv-search\Scripts\python.exe -m pip install -r requirements.txt
```

If `.venv-search` already exists, skip creating it and install/update dependencies with the second command.

Configure `GEMINI_API_KEY` using either the environment or Streamlit secrets. The environment takes precedence.

```powershell
$env:GEMINI_API_KEY = "YOUR_KEY_HERE"
.\.venv-search\Scripts\python.exe -m streamlit run app.py
```

Or copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill in:

```toml
GEMINI_API_KEY = "YOUR_KEY_HERE"
```

Then run:

```powershell
.\.venv-search\Scripts\python.exe -m streamlit run app.py
```

Get a key from [Google AI Studio](https://aistudio.google.com/apikey). Restart Streamlit after changing its secrets or theme configuration. Open the local URL printed by Streamlit, usually `http://localhost:8501`.

### macOS / Linux

```bash
python3.12 -m venv .venv-search
source .venv-search/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY="YOUR_KEY_HERE"
streamlit run app.py
```

The first embedding-model load needs internet access to download MiniLM. Subsequent inference runs locally from `.model-cache/`. Gemini answers require network access and available API quota. Model selection remains configurable through `GEMINI_MODEL` in `llm/generator.py`.

No key is needed for local PDF extraction, embeddings, or development search/classification. A `.env` file is not automatically loaded.

### Try the app

1. Upload one or more text-based PDFs in **Research Papers**.
2. Wait for **Documents indexed and ready**.
3. Enter a question in **Ask your papers** and click **Ask**.
4. Read the answer, detected intent/confidence, and emphasized page citations.
5. Expand **Sources used** to see unique paper/page pairs supplied as context.
6. Expand **Technical details** to inspect retrieval mode, scores, provenance, and passage text.

Example questions are suggestions only; they never submit automatically. The demo allows **10 successful generated answers per Streamlit session**. Failed requests do not consume a question. Changing uploads clears the displayed answer but preserves the session count.

## Project Structure

```text
app.py                           Streamlit workspace and document sidebar
research_assistant/
  pdf_extraction.py              Page-by-page extraction and readable errors
  models.py                      Paper/page metadata structures
  answer_ui.py                   Questions, answers, citations, session limit
  technical_ui.py                Optional evidence and development tools
  search_ui.py                    Session index lifecycle and search tester
  classifier_ui.py               Independent classifier tester
rag/
  chunker.py                     Page-aware 250-word / 40-word-overlap chunks
  embeddings.py                  Cached MiniLM and normalized vectors
  vector_store.py                 Existing FAISS index and metadata mapping
  retriever.py                    Basic and intent-aware retrieval
classifier/
  dataset.csv                    Balanced intent training data
  model.py / train.py / predict.py
  intent_classifier.pt            Trained model required by the app
  intent_config.json / training_report.json
evaluation/
  eval_questions.csv / evaluate_classifier.py
  retrieval_questions.json / evaluate_retrieval.py
  compare_retrieval.py / build_report.py
  results.md / results.json       Recorded measurements
  baseline/                      Original model and dataset
tests/                           Unit, interface, and optional live checks
.streamlit/config.toml            Native Streamlit theme
.streamlit/secrets.toml.example   Credential template with placeholder only
requirements.txt
.gitignore
```

## Testing and Reproducing Results

Run regression tests; Gemini calls are mocked:

```powershell
.\.venv-search\Scripts\python.exe -m unittest discover -s tests -v
```

Reproduce the current classifier/retrieval evaluation without Gemini:

```powershell
.\.venv-search\Scripts\python.exe -m evaluation.evaluate_classifier
.\.venv-search\Scripts\python.exe -m evaluation.evaluate_retrieval
.\.venv-search\Scripts\python.exe -m evaluation.build_report
```

See [DEVELOPMENT.md](DEVELOPMENT.md) for the preserved baseline command, development tools, and optional real-model/live-Gemini checks. Re-running evaluation overwrites its output reports; it does not retrain the classifier.

## Credentials and Repository Hygiene

The key is loaded only from `GEMINI_API_KEY` or Streamlit secrets. It is never placed in prompts or displayed in the interface. Gemini exceptions are replaced with fixed public messages without logging their raw contents.

Local secrets, environment files, virtual environments, model-download caches, Python/test caches, logs, and temporary upload folders are ignored by Git. Keep `.streamlit/secrets.toml` local; commit only its placeholder template. Uploads, extracted text, and FAISS indexes are held in session memory, not saved as a database.

## Limitations

- The custom intent dataset and evaluation benchmark are small.
- Ambiguous questions can be misclassified, including with high confidence. Confidence fallback reduces the influence of uncertain predictions but does not eliminate mistakes.
- Retrieval depends on PDF extraction quality. Columns, equations, and tables can lose reading order; scanned PDFs need OCR, which is not implemented.
- Chunking uses words rather than sentences or embedding tokens. MiniLM truncates beyond its 256-word-piece limit, so longer chunks may not be represented completely.
- Similarity and comparison cutoffs are heuristics, not guarantees of relevance.
- Grounding instructions and citation checks do not establish factual correctness.
- Context sources include every selected page, even when an answer does not explicitly cite all of them.
- The session quota is a demo safeguard, not authentication or persistent rate limiting.

## Future Improvements

- A larger, manually reviewed intent dataset.
- More comprehensive retrieval evaluation on real papers.
- Improved awareness of PDF sections and structure.
- Optional reranking, evaluated against the current baseline.

These are future possibilities, not implemented features. Deployment is handled separately.

## Phase 9: Manual Deployment to Streamlit Community Cloud

The entry point is **app.py** in the repository root. Select **Python 3.12**, matching the tested local runtime. Deployment preparation does not publish the app.

### Dependencies and model artifacts

`requirements.txt` pins the tested Streamlit, PyMuPDF, sentence-transformers, FAISS CPU, NumPy, PyTorch, scikit-learn, official `google-genai` SDK, and Transformers encoder dependency. The official PyTorch CPU index supplies the CPU-only wheel on Linux/Windows. No CUDA, GPU FAISS, torchvision, torchaudio, Docker, or system-package file is needed by this implementation.

Include `classifier/intent_classifier.pt` (weights plus embedded labels/configuration), `classifier/intent_config.json` (readable configuration), and all app packages. Keep training/evaluation code, datasets, reports, and `evaluation/baseline/`. No retraining is required. The loader resolves the checkpoint relative to its module and loads onto CPU.

MiniLM downloads from the public `sentence-transformers/all-MiniLM-L6-v2` Hugging Face model on first indexing. A fresh host needs outbound access and enough disk/memory. Do not set `HF_HUB_OFFLINE=1` or `TRANSFORMERS_OFFLINE=1` on a fresh deployment. `.model-cache/` contains pretrained model files only and must not be committed. The embedding model and classifier are cached shared inference resources; uploaded PDF bytes, pages, chunks, answers, counters, and FAISS indexes remain in user session state. Indexing is skipped on reruns when the chunks have not changed.

### 1. Check locally

From the project root in PowerShell:

```powershell
.\.venv-search\Scripts\python.exe -m pip install -r requirements.txt
.\.venv-search\Scripts\python.exe -m pip check
.\.venv-search\Scripts\python.exe -m unittest discover -s tests -v
.\.venv-search\Scripts\python.exe -m streamlit run app.py
```

For a new computer, first run `py -3.12 -m venv .venv-search`. Launch with this environment's Python to avoid the missing-PyTorch error caused by using another interpreter.

### 2. Push to GitHub when ready

Create an empty GitHub repository named `smart-research-paper-assistant`, without an initial README. The local repository currently has no committed app files. Review and stage only the project files:

```powershell
git status --short
git check-ignore .streamlit/secrets.toml .env .model-cache/example .venv-search/pyvenv.cfg tmp/example
git add .gitignore requirements.txt README.md DEVELOPMENT.md DEPLOYMENT_CHECKS.md app.py .streamlit/config.toml .streamlit/secrets.toml.example classifier evaluation llm rag research_assistant tests
git diff --cached --stat
git diff --cached --name-only
```

Confirm no real secrets, environment files, uploads, caches, virtual environments, or temporary files are staged. Confirm the classifier `.pt` and `.json` artifacts ARE included. Do not force-add ignored files. Then publish manually:

```powershell
git commit -m "Prepare research paper assistant for Streamlit deployment"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/smart-research-paper-assistant.git
git push -u origin main
```

Replace `YOUR_USERNAME`. If `origin` already exists, inspect `git remote -v` and use the intended existing remote instead of adding another. Do not publish uploaded papers. Ignore rules do not remove old credentials from Git history; rotate any key that was previously committed.

### 3. Connect Streamlit and configure secrets

1. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) and connect GitHub.
2. Choose **Create app**, select the repository and branch **main**, and set **Main file path** to **app.py**.
3. In **Advanced settings**, choose **Python 3.12**. Select the Python version here; do not add `runtime.txt`.
4. Paste this into the **Secrets** field, replacing the placeholder with your actual key:

```toml
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"
```

5. Click **Deploy** when ready and wait for installation/startup. The first PDF can take longer while MiniLM downloads.
6. Later, update credentials through the app's **Settings → Secrets**.

The existing loader reads the environment first, then Streamlit secrets. `.env` is not automatically loaded. Keep actual `.streamlit/secrets.toml` local; commit only its placeholder example. Never put the key into source, screenshots, build commands, or repository issues. The unchanged Gemini model is configured in `llm/generator.py`; confirm its availability and quota for your project before publishing.

If dashboard labels change, follow the [official deployment instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).

### 4. Test the public URL yourself

- Confirm the clean upload prompt before uploading anything.
- Upload a small text-based PDF; check paper/page/chunk counts and the ready status.
- Ask a supported question and verify filename/page citations against the PDF. Inspect Sources used and Technical details when needed.
- Ask an unsupported question and check the insufficient-information behavior. Confirm invalid PDFs display readable errors.
- Open a second independent browser/private window with a different paper. Check that documents, retrieved sources, answers, and counters are independent.
- Generate ten successful answers in one session and confirm further generation is disabled. This consumes Gemini quota.
- Review owner-only cloud logs for dependency, download, or resource failures; review logs for private information before sharing them.

### Hosting, privacy, and quota limits

Community Cloud has limited CPU, memory, and storage, and idle apps may sleep. Large PDFs or many simultaneous sessions can exceed resources. Existing handlers provide safe messages for caught indexing/search/API failures; an operating-system out-of-memory kill cannot be caught by Python and may restart the app. Unexpected exceptions are configured to show generic browser messages rather than stack traces; owner logs can still contain diagnostic details. See [current hosting limitations](https://docs.streamlit.io/deploy/streamlit-community-cloud/status).

The app does not write uploaded PDFs, extracted text, or FAISS indexes to persistent storage. They are session-local and are not shared in model caches. Selected passages and source metadata are sent to Gemini when a question is asked. Session data/counts are lost when the session ends or the server restarts; no persistent user history is provided.

The **10 successful answers per session** limit is a demo safeguard, not strong protection against deliberate abuse. New sessions can reset it; failed API requests may still consume provider quota. All users share the configured Gemini project's quota. Restrict the key to its intended API where supported, review active limits in Google AI Studio, and set conservative provider-side project quotas or spend caps where available. Billing alerts alone do not impose a hard spending limit. Free-tier availability and request/token/daily limits vary by model, project, region, and account. Check [Gemini rate limits](https://ai.google.dev/gemini-api/docs/rate-limits) and [pricing](https://ai.google.dev/gemini-api/docs/pricing) before publishing.

Local verification does not establish Linux hosting capacity, a fresh cloud model download, or public URL behavior. These must be tested after manual publishing. No GitHub push or deployment was performed during preparation.

See [DEPLOYMENT_CHECKS.md](DEPLOYMENT_CHECKS.md) for the local verification results and remaining checks.

