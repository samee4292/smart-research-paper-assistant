# Development and local verification

The AI architecture is frozen after Phase 7. Phase 8 changes presentation only.
The Streamlit file watcher is disabled to avoid importing optional Transformers
image modules while scanning loaded packages. Restart the app after source edits.

## Optional interface tools

Expand **Technical details** and enable **Show development tools** to inspect
extracted pages/chunks, run basic **Test Semantic Search**, or run **Test Intent
Classifier**. Neither tester calls Gemini. These functions remain reusable in
their original modules; the normal question interface uses the same index.

The evidence table shows ORIGINAL/INTENT/BOTH provenance, individual query
cosines, and the ranking metric. RRF scores are rank-fusion values, not cosine
similarities or probabilities. Standard retrieval is used below 70% confidence,
when the classifier cannot run, or when query expansion fails.

## Reproduce the original classifier baseline

From the project root in PowerShell:

```powershell
.\.venv-search\Scripts\python.exe -m evaluation.evaluate_classifier --model evaluation\baseline\intent_classifier.pt --training-data evaluation\baseline\dataset.csv --output evaluation\classifier_before.json
.\.venv-search\Scripts\python.exe -m evaluation.evaluate_classifier
.\.venv-search\Scripts\python.exe -m evaluation.evaluate_retrieval
.\.venv-search\Scripts\python.exe -m evaluation.build_report
```

These commands use saved weights, preserve the app's current checkpoint, and
do not call Gemini. The comparison report is a development measurement: errors
in the 60-question evaluation informed the training-data additions. The larger
training dataset changed its internal split, so the old/new internal test scores
are not a paired comparison. The original Phase 6 report remains historical.

For basic/intent-aware chunk comparisons on your own PDFs:

```powershell
.\.venv-search\Scripts\python.exe -m evaluation.compare_retrieval --pdfs "C:\papers\adaptive.pdf" "C:\papers\fixed.pdf" --output evaluation\my_papers.json
```

## Automated checks

```powershell
.\.venv-search\Scripts\python.exe -m unittest discover -s tests -v
```

The regular suite mocks MiniLM/Gemini where appropriate. A real-model extraction
and FAISS check can be enabled separately:

```powershell
$env:RUN_MODEL_TESTS = "1"
.\.venv-search\Scripts\python.exe -m unittest discover -s tests -p test_semantic_integration.py -v
Remove-Item Env:RUN_MODEL_TESTS
```

The optional live portfolio journey uses the real PDF pipeline, saved FFNN,
FAISS, Gemini, and Streamlit interface against artificial in-memory PDFs.
It makes two API requests, needs a configured key, and may use API quota:

```powershell
$env:RUN_LIVE_GEMINI_TESTS = "1"
.\.venv-search\Scripts\python.exe -m unittest discover -s tests -p test_portfolio_journey.py -v
Remove-Item Env:RUN_LIVE_GEMINI_TESTS
```

It checks high-confidence intent retrieval and the known low-confidence fallback,
validated page citations, unique context sources, hidden development tools,
and the session question counter. Synthetic findings are not actual research.

## Manual edge cases

- Start with no uploads: Ask is disabled and upload guidance appears.
- Upload multiple PDFs: confirm names, page counts, indexed chunks, and readiness.
- Use a short paper: fewer than five chunks should work without errors.
- Use a blank/scanned, empty, damaged, or locked PDF alongside a valid PDF:
  the valid paper remains usable and the invalid paper has a readable status.
- Remove/change files: the index and displayed answer must reflect current
  uploads; the session quota must not reset.
- Ask with whitespace only: no retrieval/API call should occur.
- Ask an unsupported question: verify the insufficient-information response
  and inspect context if the model instead produces an apparently cited claim.
- Remove or invalidate the key: no credential value or raw SDK error should
  appear. API failures must not consume a question.
- Reach ten successful answers: Ask becomes disabled.
- Test desktop and a narrow viewport: sidebar controls remain reachable,
  answer text wraps, and optional tables scroll within their container.

No uploaded paper is saved to disk. Keep credentials and model-download caches
local and ignored; preserve the trained classifier and evaluation artifacts.
