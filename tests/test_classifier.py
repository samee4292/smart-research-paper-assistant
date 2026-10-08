from collections import Counter
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
from streamlit.testing.v1 import AppTest
import torch

from classifier.model import INTENTS, IntentClassifier
from classifier.predict import predict_intent
from classifier.train import DIRECTORY, load_dataset, split_indices, train


class ClassifierTests(unittest.TestCase):
    def test_balanced_dataset_and_disjoint_reproducible_splits(self):
        texts, labels = load_dataset(DIRECTORY / "dataset.csv")
        self.assertEqual(len(texts), 390)
        self.assertEqual(Counter(labels), {i: 65 for i in range(6)})
        split = split_indices(labels)
        self.assertEqual([len(split[name]) for name in ["train", "validation", "test"]], [273, 58, 59])
        sets = [set(ids) for ids in split.values()]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        self.assertEqual(set.union(*sets), set(range(len(texts))))
        for name in split:
            np.testing.assert_array_equal(split[name], split_indices(labels)[name])
            self.assertEqual(set(labels[split[name]]), set(range(6)))

    def test_network_shape_gradients_and_eval_mode(self):
        model = IntentClassifier()
        inputs = torch.randn(4, 384)
        logits = model(inputs)
        self.assertEqual(tuple(logits.shape), (4, 6))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 57926)
        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2, 3])).backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))
        model.eval()
        torch.testing.assert_close(model(inputs), model(inputs))

    def test_training_saves_reloadable_bundle_and_reports(self):
        _, labels = load_dataset(DIRECTORY / "dataset.csv")
        # Controlled features exercise training without downloading MiniLM.
        vectors = np.zeros((len(labels), 384), dtype=np.float32)
        vectors[np.arange(len(labels)), labels] = 1
        with tempfile.TemporaryDirectory() as folder, patch("classifier.train.embed_texts", return_value=vectors), redirect_stdout(io.StringIO()):
            report = train(output_dir=Path(folder), epochs=2)
            checkpoint = torch.load(Path(folder) / "intent_classifier.pt", weights_only=True)
            self.assertEqual(checkpoint["config"]["labels"], INTENTS)
            test_count = len(split_indices(labels)["test"])
            self.assertEqual(len(report["test_predictions"]), test_count)
            self.assertEqual(np.asarray(report["confusion_matrix"]).sum(), test_count)
            self.assertEqual(len(report["history"]), 2)
            self.assertEqual(json.loads((Path(folder) / "intent_config.json").read_text())["seed"], 42)
            with patch("classifier.predict.MODEL_PATH", Path(folder) / "intent_classifier.pt"), patch("classifier.predict.embed_texts", return_value=vectors[:1]):
                first = predict_intent("What is the overall contribution?")
                self.assertEqual(first, predict_intent("What is the overall contribution?"))
                self.assertIn(first["intent"], INTENTS)
                self.assertTrue(0 <= first["confidence"] <= 1)
        with self.assertRaises(ValueError):
            predict_intent("  ")
        with patch("classifier.predict.MODEL_PATH", Path("missing_model.pt")), self.assertRaises(FileNotFoundError):
            predict_intent("A question")

    def test_ui_prediction_is_independent_of_rag(self):
        app = AppTest.from_file(str(DIRECTORY.parent / "app.py"))
        with patch("streamlit.file_uploader", return_value=[SimpleNamespace(name="invalid.pdf", getvalue=lambda: b"invalid")]), patch("research_assistant.answer_ui.get_api_key", return_value=None), patch("research_assistant.classifier_ui.MODEL_PATH") as path, patch("research_assistant.classifier_ui.predict_intent", return_value={"intent": "LIMITATIONS", "confidence": 0.91}) as predict:
            path.exists.return_value = True
            app.run(timeout=20)
            app.checkbox(key="show_development_tools").check().run(timeout=20)
            app.button(key="classify_intent").click().run(timeout=20)
            predict.assert_not_called()
            app.text_input(key="intent_question").set_value("Where could the approach fail?")
            app.button(key="classify_intent").click().run(timeout=20)
            self.assertFalse(app.exception)
            self.assertTrue(any(t.value == "Predicted Intent: LIMITATIONS" for t in app.text))
            self.assertTrue(any(t.value == "Confidence: 91.0%" for t in app.text))
            self.assertEqual(app.session_state["generated_answers"], 0)
            self.assertIsNone(app.session_state["vector_store"])


if __name__ == "__main__":
    unittest.main()
