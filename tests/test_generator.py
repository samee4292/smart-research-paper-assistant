import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from llm.generator import (GEMINI_MODEL, GenerationError, INSUFFICIENT_CONTEXT,
                           build_prompt, generate_answer, get_api_key, unique_sources, validate_answer,
                           render_answer_response)


class GeneratorTests(unittest.TestCase):
    def setUp(self):
        self.context = [dict(chunk_id="p1", filename="paper.pdf", page=6,
                             text="The method improved accuracy.", similarity=0.8)]

    def test_prompt_and_unique_context_sources(self):
        prompt = json.loads(build_prompt(" What improved? ", self.context))
        self.assertEqual(prompt["question"], "What improved?")
        self.assertEqual(set(prompt["research_paper_context"][0]), {"source_id", "filename", "page", "text"})
        self.assertEqual(prompt["research_paper_context"][0]["source_id"], 1)
        self.assertEqual(unique_sources(self.context * 2), [dict(filename="paper.pdf", page=6)])

    def test_credentials_precedence_and_missing_secrets(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "env-test"}), patch("llm.generator.st.secrets", {"GEMINI_API_KEY": "secret-test"}):
            self.assertEqual(get_api_key(), "env-test")
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}), patch("llm.generator.st.secrets", {"GEMINI_API_KEY": "secret-test"}):
            self.assertEqual(get_api_key(), "secret-test")
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}), patch("llm.generator.st.secrets", {}):
            self.assertIsNone(get_api_key())

    def test_citation_allowlist_and_fallback(self):
        answer = "Accuracy improved [paper.pdf, Page 6]."
        self.assertEqual(validate_answer(answer, self.context), answer)
        self.assertEqual(validate_answer(INSUFFICIENT_CONTEXT, self.context), INSUFFICIENT_CONTEXT)
        bracket_source = [dict(filename="paper [draft].pdf", page=1)]
        bracket_answer = "A supported claim [paper [draft].pdf, Page 1]."
        self.assertEqual(validate_answer(bracket_answer, bracket_source), bracket_answer)
        for invalid in ["", "No citation", "Result [paper.pdf, Page 99]", "Result [fake.pdf, Page 6]",
                        "Result [paper.pdf, Page 6] [invented]", "Result [paper.pdf, Page 6] [broken"]:
            with self.subTest(invalid=invalid), self.assertRaises(GenerationError):
                validate_answer(invalid, self.context)

    def test_official_sdk_call_and_safe_api_failure(self):
        client = Mock()
        client.models.generate_content.return_value = SimpleNamespace(text=json.dumps({
            "insufficient_context": False,
            "statements": [{"text": "Accuracy improved.", "source_ids": [1]}],
        }))
        with patch("google.genai.Client") as create:
            create.return_value.__enter__.return_value = client
            answer = generate_answer("What improved?", self.context, "fake-test-credential")
            self.assertIn("[paper.pdf, Page 6]", answer)
            request = client.models.generate_content.call_args.kwargs
            self.assertEqual(request["model"], GEMINI_MODEL)
            self.assertNotIn("fake-test-credential", request["contents"])
            self.assertIn("ONLY", request["config"].system_instruction)
            self.assertEqual(request["config"].response_mime_type, "application/json")
            client.models.generate_content.side_effect = RuntimeError("fake-test-credential sensitive request")
            with self.assertRaises(GenerationError) as error:
                generate_answer("What improved?", self.context, "fake-test-credential")
            self.assertNotIn("fake-test-credential", str(error.exception))

    def test_no_question_or_context(self):
        with self.assertRaises(GenerationError):
            generate_answer("  ", self.context, "fake")
        self.assertEqual(generate_answer("question", [], ""), INSUFFICIENT_CONTEXT)

    def test_structured_citations_and_invalid_source_ids(self):
        response = {"insufficient_context": False,
                    "statements": [{"text": "Accuracy improved.", "source_ids": [1, 1]}]}
        self.assertEqual(render_answer_response(json.dumps(response), self.context),
                         "Accuracy improved. [paper.pdf, Page 6]")
        for ids in [[0], [2], [-1], [True], ["1"], []]:
            response["statements"][0]["source_ids"] = ids
            with self.subTest(ids=ids), self.assertRaises(GenerationError):
                render_answer_response(json.dumps(response), self.context)
        self.assertEqual(render_answer_response(json.dumps({"insufficient_context": True,
                                                            "statements": []}), self.context),
                         INSUFFICIENT_CONTEXT)
        with self.assertRaises(GenerationError):
            render_answer_response("not JSON", self.context)


if __name__ == "__main__":
    unittest.main()
