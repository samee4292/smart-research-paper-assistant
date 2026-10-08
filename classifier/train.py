"""Train our FFNN on frozen MiniLM features: python -m classifier.train."""

import argparse
from collections import Counter
import copy
import csv
from hashlib import sha256
import json
from pathlib import Path
import random

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from classifier.model import INTENTS, MODEL_CONFIG, IntentClassifier
from rag.embeddings import MODEL_NAME, embed_texts

DIRECTORY = Path(__file__).resolve().parent


def load_dataset(path: Path):
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != ["text", "intent"]:
            raise ValueError("Dataset columns must be text,intent.")
        rows = list(reader)
    seen = set()
    texts, labels = [], []
    for row in rows:
        text, label = row["text"].strip(), row["intent"].strip()
        normalized = " ".join(text.casefold().split()).rstrip(".?!")
        if not text or label not in INTENTS or normalized in seen:
            raise ValueError("Dataset contains empty questions, unknown intents, or duplicate questions.")
        seen.add(normalized)
        texts.append(text)
        labels.append(INTENTS.index(label))
    if any(count < 10 for count in Counter(labels).values()) or len(set(labels)) != 6:
        raise ValueError("Include at least 10 examples of each intent for stratified splitting.")
    return texts, np.asarray(labels, dtype=np.int64)


def split_indices(labels, seed=42):
    """Disjoint 70/15/15 split, preserving class proportions at both steps."""
    train, rest = train_test_split(np.arange(len(labels)), test_size=0.30,
                                  random_state=seed, stratify=labels)
    validation, test = train_test_split(rest, test_size=0.50,
                                      random_state=seed, stratify=labels[rest])
    return {"train": train, "validation": validation, "test": test}


def evaluate(model, features, labels):
    model.eval()  # Disable dropout for measured validation/test accuracy.
    with torch.inference_mode():
        logits = model(features)
        loss = nn.functional.cross_entropy(logits, labels).item()
        probabilities = torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)
    return {"loss": loss, "accuracy": (predictions == labels).float().mean().item()}, predictions.tolist(), probabilities.tolist()


def train(dataset_path=DIRECTORY / "dataset.csv", output_dir=DIRECTORY,
          seed=42, epochs=150, patience=20, learning_rate=0.001):
    if epochs <= 0 or patience <= 0 or learning_rate <= 0:
        raise ValueError("Epochs, patience, and learning rate must be positive.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)

    texts, labels = load_dataset(Path(dataset_path))
    splits = split_indices(labels, seed)
    print("Dataset counts:", dict(Counter(INTENTS[i] for i in labels)), flush=True)
    print("Split sizes:", {name: len(ids) for name, ids in splits.items()}, flush=True)
    print("Encoding questions with frozen MiniLM (no MiniLM training)...", flush=True)
    # Encode once without gradients. The model and optimizer below contain ONLY
    # our small FFNN; no vocabulary/scaler is fitted on held-out text.
    with torch.inference_mode():
        vectors = embed_texts(texts)
    if vectors.shape != (len(texts), 384) or not np.isfinite(vectors).all():
        raise ValueError("Expected one finite 384-dimensional MiniLM vector per question.")
    features = torch.from_numpy(vectors.copy())
    targets = torch.from_numpy(labels)
    model = IntentClassifier(**MODEL_CONFIG)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    train_ids = splits["train"]
    loader = DataLoader(TensorDataset(features[train_ids], targets[train_ids]), batch_size=32,
                        shuffle=True, generator=torch.Generator().manual_seed(seed))
    history, best_loss, best_state, best_epoch, stale = [], float("inf"), None, 0, 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, count = 0.0, 0, 0
        for batch_features, batch_labels in loader:
            optimizer.zero_grad()
            logits = model(batch_features)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_labels)
            correct += (logits.argmax(dim=1) == batch_labels).sum().item()
            count += len(batch_labels)
        validation, _, _ = evaluate(model, features[splits["validation"]], targets[splits["validation"]])
        history.append({"epoch": epoch, "train_loss": total_loss / count,
                        "train_accuracy": correct / count, "validation_loss": validation["loss"],
                        "validation_accuracy": validation["accuracy"]})
        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:3}: train loss={total_loss/count:.4f} accuracy={correct/count:.3f}; "
                  f"validation loss={validation['loss']:.4f} accuracy={validation['accuracy']:.3f}", flush=True)
        if validation["loss"] < best_loss - 0.0001:
            best_loss, best_state, best_epoch, stale = validation["loss"], copy.deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
        if stale >= patience:
            print(f"Early stopping at epoch {epoch}; selected epoch {best_epoch} by validation loss.", flush=True)
            break

    model.load_state_dict(best_state)
    final_metrics = {}
    for name in ("train", "validation"):
        final_metrics[name], _, _ = evaluate(model, features[splits[name]], targets[splits[name]])
    # The held-out test set is evaluated only AFTER selecting the checkpoint.
    test_metrics, test_predictions, test_probabilities = evaluate(model, features[splits["test"]], targets[splits["test"]])
    final_metrics["test"] = test_metrics
    test_labels = labels[splits["test"]]
    report_text = classification_report(test_labels, test_predictions, labels=list(range(6)),
                                        target_names=INTENTS, digits=3, zero_division=0)
    matrix = confusion_matrix(test_labels, test_predictions, labels=list(range(6))).tolist()
    config = {"architecture": MODEL_CONFIG, "labels": INTENTS,
              "label_to_index": {label: i for i, label in enumerate(INTENTS)},
              "embedding_model": MODEL_NAME, "normalize_embeddings": True,
              "seed": seed, "learning_rate": learning_rate, "batch_size": 32,
              "max_epochs": epochs, "patience": patience, "best_epoch": best_epoch,
              "split_ratios": [0.70, 0.15, 0.15],
              "dataset_sha256": sha256(Path(dataset_path).read_bytes()).hexdigest(),
              "torch_version": str(torch.__version__)}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": config}, output_dir / "intent_classifier.pt")
    (output_dir / "intent_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    report = {"config": config, "metrics": final_metrics, "history": history,
              "split_indices": {name: ids.tolist() for name, ids in splits.items()},
              "classification_report": classification_report(test_labels, test_predictions,
                  labels=list(range(6)), target_names=INTENTS, output_dict=True, zero_division=0),
              "confusion_matrix": matrix,
              "confusion_matrix_order": INTENTS,
              "test_predictions": [dict(text=texts[int(row)], actual=INTENTS[int(actual)],
                   predicted=INTENTS[predicted], confidence=max(probabilities))
                   for row, actual, predicted, probabilities in
                   zip(splits["test"], test_labels, test_predictions, test_probabilities)]}
    (output_dir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, values in final_metrics.items():
        print(f"Final {name} accuracy: {values['accuracy']:.2%}")
    print("\nHeld-out test classification report:\n" + report_text)
    print("Confusion matrix (rows=actual, columns=predicted):", INTENTS)
    for row in matrix:
        print(row)
    print("Saved trained FFNN and reports to", output_dir)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the custom research query classifier on CPU.")
    parser.add_argument("--dataset", type=Path, default=DIRECTORY / "dataset.csv")
    parser.add_argument("--output-dir", type=Path, default=DIRECTORY)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    args = parser.parse_args()
    train(args.dataset, args.output_dir, args.seed, args.epochs, args.patience, args.learning_rate)
