# Custom research-question intent classifier

Phase 5 trains our own 57,926-parameter PyTorch feed-forward network. MiniLM
is a pretrained, fixed feature extractor, not the network trained in this phase.
Classification never calls Gemini or an external classification service.
The independent development tester does not change retrieval. Phase 6's main
Ask interface uses this same trained classifier to guide existing FAISS search;
Gemini still only receives the original question and selected source context.

## Dataset

`dataset.csv` now contains 390 authored research-style questions: 65 each
for SUMMARY, COMPARISON, METHODOLOGY, RESULTS, LIMITATIONS, and DEFINITION.
These are curated educational examples, not questions collected from real users.
Wording includes direct requests, paraphrases, and questions about different
research domains. Intent words are not required: for example, "What remains
unresolved after this study?" is LIMITATIONS. These examples were written during
development; no classifier API or runtime LLM labeling is used.
Phase 7 added 90 varied examples (15 per intent), after measuring the baseline
errors. The original 300-row dataset and checkpoint remain in `evaluation/baseline/`.

Conventions: SUMMARY requests an overview of the whole paper; RESULTS requests
specific observations, scores, or experimental conclusions. COMPARISON takes
precedence when explicitly relating two studies/methods. METHODOLOGY concerns
how work was done; LIMITATIONS asks about shortcomings or boundaries;
DEFINITION asks what a term means. Some real questions combine intents; this
single-label starter dataset does not capture all that ambiguity.

The loader rejects unknown labels, empty text, and normalized exact duplicates.
When adding a question containing a comma, quote its text using standard CSV syntax.
Row splits are stratified and disjoint: 273 training, 58 validation, 59 test.
A fixed seed (42) determines the split, initialization, dropout,
and batch order. No learned preprocessing or embedding fine-tuning uses the
validation/test data. Shared vocabulary/paraphrase styles in this small curated
dataset can still make held-out accuracy optimistic about real-world queries.

## Features and network

The existing `rag.embeddings.embed_texts` encodes each question as a normalized
384-number MiniLM vector. The same CPU MiniLM model/cache used elsewhere in the
project is reused. Encoding runs without gradients. Only `IntentClassifier`
parameters are passed to Adam; MiniLM weights are not updated.

```text
MiniLM vector (384)
  -> Linear(384, 128) -> ReLU -> Dropout(0.2)
  -> Linear(128, 64)  -> ReLU
  -> Linear(64, 6)   -> raw class logits
```

- **ReLU** keeps positive values and turns negative values into zero. It lets
  the network learn nonlinear patterns instead of only a linear rule.
- **Dropout** randomly hides 20% of the first hidden layer's activations during
  training, discouraging reliance on just a few features. It is disabled for
  validation, test evaluation, and inference through `model.eval()`.
- **CrossEntropyLoss** compares the six raw scores with the correct class and
  penalizes giving that class low probability. It handles normalization itself;
  training does not put softmax before the loss.
- **Adam** uses gradients and running statistics to adjust each trainable weight.
  The default learning rate is 0.001 and batch size is 32.

## Train or retrain

From the project root in PowerShell:

```powershell
.\.venv-search\Scripts\python.exe -m pip install -r requirements.txt
.\.venv-search\Scripts\python.exe -m classifier.train
```

No Gemini key is needed. MiniLM needs internet only if its model cache is missing.
The default training run has at most 150 epochs and stops after 20 epochs without
validation-loss improvement. The best validation-loss checkpoint is restored
before evaluating the test set. Do not repeatedly tune against test accuracy;
use validation data and reserve new independent questions for later evaluation.

Options, if needed:

```powershell
.\.venv-search\Scripts\python.exe -m classifier.train --seed 42 --epochs 150 --patience 20 --learning-rate 0.001
```

Use `--dataset PATH` for an expanded dataset and `--output-dir PATH` for an
experiment that should not replace the app's default model. The default output
directory is `classifier/`. Retraining there replaces the saved model/reports;
the inference cache notices the file timestamp/size and reloads it on the next
classification. Exact numerical reproducibility is intended on the same CPU
environment; library/hardware changes can affect it.

## Saved artifacts

- `intent_classifier.pt`: tensor weights plus model configuration, ordered
  labels, MiniLM model name, normalization flag, and training metadata.
- `intent_config.json`: human-readable copy of configuration/label mapping.
- `training_report.json`: epoch history, final loss/accuracy on all splits,
  held-out classification report, confusion matrix, test predictions, and
  split row indices. Matrix rows are actual classes; columns are predictions.

Final training accuracy is measured with the selected checkpoint and dropout
disabled. Per-epoch training accuracy in the history is measured during batch
updates with dropout active, so those numbers need not be identical.

## Historical Phase 5 first-run evaluation (300 rows)

The original saved checkpoint is preserved in `evaluation/baseline/`.
Seed 42 selected epoch 24 by validation loss; early stopping ended training
at epoch 44. The test set was not used for checkpoint selection or tuning.

| Split | Questions | Final accuracy |
|---|---:|---:|
| Training | 210 | 100.00% |
| Validation | 45 | 91.11% |
| Test | 45 | 84.44% (38/45) |

Held-out classification report:

| Intent | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| SUMMARY | 1.000 | 0.857 | 0.923 | 7 |
| COMPARISON | 0.857 | 0.857 | 0.857 | 7 |
| METHODOLOGY | 0.857 | 0.750 | 0.800 | 8 |
| RESULTS | 0.778 | 0.875 | 0.824 | 8 |
| LIMITATIONS | 0.778 | 0.875 | 0.824 | 8 |
| DEFINITION | 0.857 | 0.857 | 0.857 | 7 |
| Macro average | 0.854 | 0.845 | 0.847 | 45 |
| Weighted average | 0.851 | 0.844 | 0.845 | 45 |

Confusion matrix: rows are actual labels; columns are predicted labels.

| Actual / Predicted | SUMMARY | COMPARISON | METHODOLOGY | RESULTS | LIMITATIONS | DEFINITION |
|---|---:|---:|---:|---:|---:|---:|
| SUMMARY | 6 | 0 | 0 | 0 | 0 | 1 |
| COMPARISON | 0 | 6 | 0 | 0 | 1 | 0 |
| METHODOLOGY | 0 | 1 | 6 | 1 | 0 | 0 |
| RESULTS | 0 | 0 | 0 | 7 | 1 | 0 |
| LIMITATIONS | 0 | 0 | 0 | 1 | 7 | 0 |
| DEFINITION | 0 | 0 | 1 | 0 | 0 | 6 |

Historical predictions from the original model (not an additional test set):

| Question | Prediction | Confidence |
|---|---|---:|
| Can you give me the big-picture takeaway from this article? | SUMMARY | 97.9% |
| How do these two studies differ in their evaluation design? | COMPARISON | 99.9% |
| Which steps did the authors take to build the training dataset? | METHODOLOGY | 99.8% |
| What F1 score did their system obtain on held-out examples? | RESULTS | 93.2% |
| What weaknesses did the authors identify? | LIMITATIONS | 87.9% |
| In this paper what is meant by federated learning? | DEFINITION | 95.9% |

The gap between training and test accuracy is a reason to keep evaluating
independently authored questions. In particular, methodology/results and
results/limitations can overlap in natural language. A high softmax score
does not eliminate those mistakes.

## Predict and manually test

```python
from classifier.predict import predict_intent
print(predict_intent("What weaknesses did the authors identify?"))
```

Returns `{"intent": "LIMITATIONS", "confidence": ...}`. Confidence is the
largest softmax output; it is not calibrated and is not a guarantee of correctness.
The classifier always selects one of six classes, even for unrelated questions.
Missing model files and empty questions produce readable errors.

Run Streamlit as before and expand **Test Intent Classifier**. Enter unseen
paraphrases of each intent and click **Classify**. PDFs and a Gemini key are
not needed for this panel. Confirm semantic search and RAG still work separately.

The starter dataset is small and synthetic in the sense of being authored for
the demo. Held-out metrics are evidence about this dataset, not proof of broad
generalization. Additional independently written questions are the next step
for evaluating robustness. See the root README for Phase 6 intent-aware retrieval.

## Phase 7 retraining and fixed evaluation

The architecture, frozen MiniLM features, seed, learning rate, and early-stopping
settings were retained. One retraining run on 390 examples selected epoch 16
and stopped at epoch 36. Internal train/validation/test accuracies were 98.90%,
87.93%, and 91.53% (54/59). The internal split changed with the dataset, so these
test scores should not be interpreted as a paired comparison with Phase 5.

The paired comparison uses the same separate 60-question evaluation CSV before
and after retraining. Full metrics, incorrect predictions, and the known planner
failure check are in [evaluation/results.md](../evaluation/results.md).
Since evaluation errors informed development, this is a development benchmark,
not an untouched final test set. No evaluation question or the exact known
failure was added to the training dataset.
