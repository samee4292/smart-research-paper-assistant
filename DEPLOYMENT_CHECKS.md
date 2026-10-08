# Phase 9 local verification — 8 October 2026

Deployment has not been performed. No commit, GitHub push, or cloud app was created.

## Verified locally

- Entry point: repository-root `app.py`; Python 3.12 in `.venv-search`.
- All required runtime packages import; `pip check` reports no broken requirements.
- Runtime versions are pinned in `requirements.txt`; Linux/Windows select the official CPU-only PyTorch build. The official wheel index lists a Python 3.12 Linux x86-64 wheel for this PyTorch version.
- Existing classifier weights load with `map_location="cpu"`; checkpoint/readable label configuration matches the six existing intents. Real classifier inference works without retraining.
- Cached MiniLM loads on CPU, produces real embeddings, and reuses the same model object. The semantic integration test retrieves the expected PDF page.
- Synthetic local PDFs pass through real extraction, chunking, MiniLM, and FAISS in Streamlit AppTest. Two independent sessions hold different documents/indexes. An unchanged-upload rerun reuses the same index. Missing-key Ask controls remain disabled.
- Regression suite: **48 tests discovered, 47 passed, 1 skipped** with `RUN_MODEL_TESTS=1`. The skipped opt-in live-Gemini test was not run; no live Gemini requests were made. Mocked API success/failure, citation checks, invalid-PDF handling, and question-limit tests passed.
- Three stale test assertions from the reverted UI polish were aligned with the current working UI. Application UI files were not edited.
- Real secrets/environment files, virtual environments, pretrained caches, and temporary files are Git-ignored. Backup secrets files are ignored too; the placeholder example and essential trained model/evaluation artifacts remain eligible for commit.
- A pattern scan of Git-visible text files found no Gemini key patterns. This is a limited check, not a guarantee that every possible credential format has been detected. The actual secret file was not read or displayed. The repository currently has no tracked app files or committed history to audit.
- `client.showErrorDetails = "none"` is recognized by Streamlit. Unexpected browser errors are generic; owner-side logs may still contain diagnostics.
- SHA-256 comparison confirmed `app.py`, all application/AI package files, trained artifacts, and evaluation files stayed unchanged during this phase.

## Not verified here

- A fresh Linux installation of the pinned dependencies or actual Community Cloud resource capacity.
- A cold download of MiniLM on a fresh cloud machine. Local tests used the existing model cache; the unchanged loader supports first-use download from the public Hugging Face repository.
- Actual Gemini credentials, current project quota/model availability, or paid/free-tier access. The app's existing model selection and prompts were preserved.
- Public deployment, public URL, cloud session behavior, or owner dashboard access.

These are manual post-deployment checks, not claims of a tested public service. Hosting resource limits, model-download connectivity, and Gemini project quotas can still prevent successful operation. Follow the Phase 9 section of `README.md` before publishing and test a small PDF first.
