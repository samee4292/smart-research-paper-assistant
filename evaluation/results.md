# Phase 7 measured evaluation

## Classifier: fixed 60-question comparison

Before: **86.67%** (52/60).

After: **93.33%** (56/60).

| Intent | Before | After |
|---|---:|---:|
| SUMMARY | 90% | 100% |
| COMPARISON | 100% | 100% |
| METHODOLOGY | 70% | 80% |
| RESULTS | 80% | 80% |
| LIMITATIONS | 90% | 100% |
| DEFINITION | 90% | 100% |

## Known failure (separate from the 60 questions)

What performance did the adaptive planner achieve?

Before: expected RESULTS, predicted **METHODOLOGY**, confidence **81.1%**.
After: expected RESULTS, predicted **METHODOLOGY**, confidence **50.9%**.

## Before confusion matrix

Rows are expected; columns are predicted.

| Expected | SUMMARY | COMPARISON | METHODOLOGY | RESULTS | LIMITATIONS | DEFINITION |
|---|---:|---:|---:|---:|---:|---:|
| SUMMARY | 9 | 0 | 1 | 0 | 0 | 0 |
| COMPARISON | 0 | 10 | 0 | 0 | 0 | 0 |
| METHODOLOGY | 0 | 0 | 7 | 3 | 0 | 0 |
| RESULTS | 0 | 0 | 2 | 8 | 0 | 0 |
| LIMITATIONS | 1 | 0 | 0 | 0 | 9 | 0 |
| DEFINITION | 0 | 1 | 0 | 0 | 0 | 9 |

### Before incorrect predictions

- Can you give me the broad picture without dwelling on individual experiments? Expected **SUMMARY**, predicted **METHODOLOGY**, confidence 39.6%.
- What happened between recruiting volunteers and computing the final estimates? Expected **METHODOLOGY**, predicted **RESULTS**, confidence 65.9%.
- What was done to measure the system's performance? Expected **METHODOLOGY**, predicted **RESULTS**, confidence 83.8%.
- What sequence of actions would reproduce their evaluation? Expected **METHODOLOGY**, predicted **RESULTS**, confidence 68.7%.
- What did they actually obtain at the end of the trial? Expected **RESULTS**, predicted **METHODOLOGY**, confidence 73.9%.
- What did the adaptive system achieve on the final assessment? Expected **RESULTS**, predicted **METHODOLOGY**, confidence 71.5%.
- What keeps this system from being ready for everyday use? Expected **LIMITATIONS**, predicted **SUMMARY**, confidence 64.9%.
- What idea is captured by the label sample efficiency? Expected **DEFINITION**, predicted **COMPARISON**, confidence 45.0%.

## After confusion matrix

Rows are expected; columns are predicted.

| Expected | SUMMARY | COMPARISON | METHODOLOGY | RESULTS | LIMITATIONS | DEFINITION |
|---|---:|---:|---:|---:|---:|---:|
| SUMMARY | 10 | 0 | 0 | 0 | 0 | 0 |
| COMPARISON | 0 | 10 | 0 | 0 | 0 | 0 |
| METHODOLOGY | 0 | 0 | 8 | 2 | 0 | 0 |
| RESULTS | 0 | 0 | 2 | 8 | 0 | 0 |
| LIMITATIONS | 0 | 0 | 0 | 0 | 10 | 0 |
| DEFINITION | 0 | 0 | 0 | 0 | 0 | 10 |

### After incorrect predictions

- What was done to measure the system's performance? Expected **METHODOLOGY**, predicted **RESULTS**, confidence 65.6%.
- What sequence of actions would reproduce their evaluation? Expected **METHODOLOGY**, predicted **RESULTS**, confidence 41.7%.
- What did they actually obtain at the end of the trial? Expected **RESULTS**, predicted **METHODOLOGY**, confidence 81.1%.
- What did the adaptive system achieve on the final assessment? Expected **RESULTS**, predicted **METHODOLOGY**, confidence 77.7%.

## Retrieval: fixed 12-question fixture

At least one manually labeled (filename, page) occurs within the first k of the same five selected chunks.

| Metric | Basic | Intent-aware |
|---|---:|---:|
| Top-1 hit rate | 75.00% | 75.00% |
| Top-3 hit rate | 83.33% | 91.67% |
| Top-5 hit rate | 100.00% | 100.00% |
| All labeled pages in Top 5 | 100.00% | 100.00% |

### Top-1 changes

Improved: none.

Worsened: none.

### Top-3 changes

Improved: What prevents the adaptive planning trial from establishing long-term benefits?.

Worsened: none.

### Top-5 changes

Improved: none.

Worsened: none.

## Question-level retrieval

### What is the broad purpose of the adaptive planning research?

Expected: adaptive_planning.pdf page 1. Classifier: METHODOLOGY (35.1%).

Basic: adaptive_planning.pdf p5, fixed_planning.pdf p5, adaptive_planning.pdf p3, adaptive_planning.pdf p1, adaptive_planning.pdf p2. Top-3 hit: False; Top-5 hit: True.

Intent-aware: adaptive_planning.pdf p5, fixed_planning.pdf p5, adaptive_planning.pdf p3, adaptive_planning.pdf p1, adaptive_planning.pdf p2. Top-3 hit: False; Top-5 hit: True.

### Give an overall account of why the fixed timetable study was undertaken.

Expected: fixed_planning.pdf page 1. Classifier: METHODOLOGY (46.6%).

Basic: fixed_planning.pdf p2, fixed_planning.pdf p1, fixed_planning.pdf p5, fixed_planning.pdf p3, adaptive_planning.pdf p3. Top-3 hit: True; Top-5 hit: True.

Intent-aware: fixed_planning.pdf p2, fixed_planning.pdf p1, fixed_planning.pdf p5, fixed_planning.pdf p3, adaptive_planning.pdf p3. Top-3 hit: True; Top-5 hit: True.

### How were students assigned and followed in the adaptive planning experiment?

Expected: adaptive_planning.pdf page 2. Classifier: METHODOLOGY (97.5%).

Basic: adaptive_planning.pdf p2, adaptive_planning.pdf p5, fixed_planning.pdf p2, fixed_planning.pdf p5, adaptive_planning.pdf p3. Top-3 hit: True; Top-5 hit: True.

Intent-aware: adaptive_planning.pdf p2, adaptive_planning.pdf p5, fixed_planning.pdf p2, fixed_planning.pdf p5, adaptive_planning.pdf p1. Top-3 hit: True; Top-5 hit: True.

### How did researchers conduct the fixed timetable experiment?

Expected: fixed_planning.pdf page 2. Classifier: METHODOLOGY (90.5%).

Basic: fixed_planning.pdf p2, adaptive_planning.pdf p2, adaptive_planning.pdf p3, fixed_planning.pdf p4, fixed_planning.pdf p1. Top-3 hit: True; Top-5 hit: True.

Intent-aware: fixed_planning.pdf p2, adaptive_planning.pdf p2, adaptive_planning.pdf p3, fixed_planning.pdf p3, fixed_planning.pdf p4. Top-3 hit: True; Top-5 hit: True.

### What retention accuracy did the adaptive planning system achieve?

Expected: adaptive_planning.pdf page 3. Classifier: METHODOLOGY (53.4%).

Basic: adaptive_planning.pdf p3, fixed_planning.pdf p3, adaptive_planning.pdf p5, fixed_planning.pdf p5, adaptive_planning.pdf p2. Top-3 hit: True; Top-5 hit: True.

Intent-aware: adaptive_planning.pdf p3, fixed_planning.pdf p3, adaptive_planning.pdf p5, fixed_planning.pdf p5, adaptive_planning.pdf p2. Top-3 hit: True; Top-5 hit: True.

### What measured scores did the fixed planning group obtain?

Expected: fixed_planning.pdf page 3. Classifier: RESULTS (62.3%).

Basic: fixed_planning.pdf p3, fixed_planning.pdf p2, adaptive_planning.pdf p3, fixed_planning.pdf p5, adaptive_planning.pdf p2. Top-3 hit: True; Top-5 hit: True.

Intent-aware: fixed_planning.pdf p3, fixed_planning.pdf p2, adaptive_planning.pdf p3, fixed_planning.pdf p5, adaptive_planning.pdf p2. Top-3 hit: True; Top-5 hit: True.

### What prevents the adaptive planning trial from establishing long-term benefits?

Expected: adaptive_planning.pdf page 4. Classifier: LIMITATIONS (85.3%).

Basic: adaptive_planning.pdf p5, fixed_planning.pdf p5, adaptive_planning.pdf p3, adaptive_planning.pdf p4, adaptive_planning.pdf p1. Top-3 hit: False; Top-5 hit: True.

Intent-aware: adaptive_planning.pdf p5, adaptive_planning.pdf p4, fixed_planning.pdf p5, adaptive_planning.pdf p3, adaptive_planning.pdf p1. Top-3 hit: True; Top-5 hit: True.

### Why can a fixed schedule neglect a learner's weak topics?

Expected: fixed_planning.pdf page 4. Classifier: LIMITATIONS (80.9%).

Basic: fixed_planning.pdf p4, fixed_planning.pdf p1, adaptive_planning.pdf p2, adaptive_planning.pdf p1, adaptive_planning.pdf p5. Top-3 hit: True; Top-5 hit: True.

Intent-aware: fixed_planning.pdf p4, fixed_planning.pdf p1, adaptive_planning.pdf p2, adaptive_planning.pdf p1, adaptive_planning.pdf p4. Top-3 hit: True; Top-5 hit: True.

### How do fixed and adaptive planning differ in their response to quiz scores?

Expected: fixed_planning.pdf page 5. Classifier: COMPARISON (79.5%).

Basic: fixed_planning.pdf p5, adaptive_planning.pdf p3, adaptive_planning.pdf p5, adaptive_planning.pdf p2, adaptive_planning.pdf p1. Top-3 hit: True; Top-5 hit: True.

Intent-aware: fixed_planning.pdf p5, adaptive_planning.pdf p3, adaptive_planning.pdf p5, adaptive_planning.pdf p2, adaptive_planning.pdf p1. Top-3 hit: True; Top-5 hit: True.

### What delayed retention scores distinguish the adaptive and fixed approaches?

Expected: adaptive_planning.pdf page 3, fixed_planning.pdf page 3. Classifier: COMPARISON (67.9%).

Basic: fixed_planning.pdf p3, adaptive_planning.pdf p3, adaptive_planning.pdf p2, adaptive_planning.pdf p4, adaptive_planning.pdf p1. Top-3 hit: True; Top-5 hit: True.

Intent-aware: fixed_planning.pdf p3, adaptive_planning.pdf p3, adaptive_planning.pdf p2, adaptive_planning.pdf p4, adaptive_planning.pdf p1. Top-3 hit: True; Top-5 hit: True.

### What does adaptive study planning mean in the authors' terminology?

Expected: adaptive_planning.pdf page 5. Classifier: DEFINITION (82.3%).

Basic: adaptive_planning.pdf p5, fixed_planning.pdf p5, adaptive_planning.pdf p1, adaptive_planning.pdf p3, adaptive_planning.pdf p2. Top-3 hit: True; Top-5 hit: True.

Intent-aware: adaptive_planning.pdf p5, fixed_planning.pdf p5, adaptive_planning.pdf p1, adaptive_planning.pdf p2, adaptive_planning.pdf p3. Top-3 hit: True; Top-5 hit: True.

### What is meant by fixed study planning?

Expected: fixed_planning.pdf page 5. Classifier: DEFINITION (78.6%).

Basic: fixed_planning.pdf p5, fixed_planning.pdf p1, fixed_planning.pdf p4, adaptive_planning.pdf p1, fixed_planning.pdf p2. Top-3 hit: True; Top-5 hit: True.

Intent-aware: fixed_planning.pdf p5, fixed_planning.pdf p1, fixed_planning.pdf p4, adaptive_planning.pdf p1, adaptive_planning.pdf p5. Top-3 hit: True; Top-5 hit: True.

## Scope and limitations

- The same 60 evaluation questions were frozen before training-data changes; no exact normalized evaluation question or known failure was added to training.
- Baseline errors informed the added examples. Therefore the before/after evaluation is a development comparison, not an untouched final test set or proof of broad generalization.
- Training stayed balanced: 300 to 390 authored examples (50 to 65 per intent). One retraining run used unchanged architecture and default hyperparameters, selecting a checkpoint by training validation loss, not by this evaluation accuracy.
- The larger training dataset changes its internal 70/15/15 split. Its internal test accuracy is not a directly paired before/after measurement; the fixed 60-question CSV is the paired comparison.
- Retrieval annotations were authored before executing the benchmark and the Phase 6 fixture, retrieval algorithm, threshold, and embedding model were not tuned to its outcomes.
- A page hit is weak evidence of retrieval success: it does not measure precision, completeness, valid cross-paper comparisons, or answer quality. The fixture has only 11 chunks and 12 questions.
- No Gemini request or Gemini-based scoring was used.

Full classification reports, all prediction confidences, retrieved chunk text/scores, and dataset/model hashes are in `results.json`.
