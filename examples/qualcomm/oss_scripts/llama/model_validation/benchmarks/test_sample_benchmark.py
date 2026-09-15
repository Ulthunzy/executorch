# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Test source isolation, few-shot rendering, conservative scoring, and reporting."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from benchmark import run_session
from report import digest, read_json, write_json
from sample_benchmark import prepare_cases, render_messages, summarize_samples
from sample_scoring import boxed_answers, extract_choice, numeric_value, score_answer


class CharacterTokenizer:
    """Use character counts to make context-boundary tests deterministic."""

    def encode(self, text, *, bos, eos):
        """Return distinct positions while checking that no extra BOS/EOS is added."""
        assert bos == eos == 0
        return list(range(len(text)))


class SampleBenchmarkTest(unittest.TestCase):
    """Cover the sample bundle without requiring a model, GPU, or device."""

    def test_preserve_fewshot_and_system(self):
        """Keep source roles and demonstration answers, opening only the final assistant."""
        prompt, question = render_messages(
            [
                {"role": "SYSTEM", "prompt": "Think normally."},
                {"role": "HUMAN", "prompt": "Example?"},
                {"role": "BOT", "prompt": "A"},
                {"role": "HUMAN", "prompt": "Actual question?"},
            ]
        )
        self.assertEqual(question, "Actual question?")
        self.assertEqual(
            prompt,
            (
                "<|im_start|>system\nThink normally.<|im_end|>\n"
                "<|im_start|>user\nExample?<|im_end|>\n"
                "<|im_start|>assistant\nA<|im_end|>\n"
                "<|im_start|>user\nActual question?<|im_end|>\n"
                "<|im_start|>assistant\n"
            ),
        )
        self.assertNotIn("/no_think", prompt)
        with self.assertRaises(ValueError):
            render_messages([{"role": "tool", "content": "not supported"}])
        with self.assertRaises(ValueError):
            render_messages([{"role": "BOT", "prompt": "already answered"}])

    def test_answer_extraction(self):
        """Accept explicit answers without picking letters from ordinary prose."""
        for answer in (
            "ANSWER: B",
            r"\boxed{B}",
            "The answer is B.",
            "答案：B。",
            "(B)",
        ):
            self.assertEqual(extract_choice(answer), "B")
        for answer in (
            "I discuss A and B.",
            "B could be right but I am not sure",
            "Answer: A or B",
        ):
            self.assertIsNone(extract_choice(answer))
        self.assertIsNone(score_answer("MMLU", "A", "I cannot decide.")["correct"])
        self.assertFalse(score_answer("MMLU", "A", "ANSWER: B")["correct"])

    def test_math_scoring(self):
        """Handle nested boxes and numeric equivalence, never eval generated text."""
        self.assertEqual(boxed_answers(r"x=\boxed{\frac{6}{3}}"), [r"\frac{6}{3}"])
        self.assertTrue(
            score_answer("GSM8K", "solution\n#### 2", r"\boxed{\frac{6}{3}}")["correct"]
        )
        self.assertTrue(
            score_answer("AIME25", "336^\\circ", r"\boxed{0336}")["correct"]
        )
        self.assertTrue(
            score_answer("MATH-500", r"result \boxed{15x-80}", r"\boxed{15x - 80}")[
                "correct"
            ]
        )
        self.assertIsNone(
            score_answer("MATH-500", r"\boxed{\cot x}", r"\boxed{1/\tan(x)}")["correct"]
        )
        self.assertIsNone(numeric_value("1,2"))
        self.assertIsNone(numeric_value("1e999999999"))
        self.assertEqual(numeric_value("1,200"), 1200)
        self.assertIsNone(numeric_value("__import__('os').system('false')"))
        self.assertIsNone(
            score_answer("GSM8K", "#### 2", "My intermediate result was 2.")["correct"]
        )

    def test_missing_ifeval_dependencies_are_not_incorrect_answers(self):
        """Mark unavailable graders as unscored instead of silently failing an answer."""
        with patch("sample_scoring.ifeval_tools", side_effect=ImportError("missing")):
            result = score_answer("IFEval", {}, "{}")
        self.assertIsNone(result["correct"])
        self.assertEqual(result["score_method"], "grader_unavailable")

    def test_prepare_score_and_keep_skips_in_denominator(self):
        """Never inject old model responses or references, and never truncate long inputs."""
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            source = bundle / "all_benchmarks_140.jsonl"
            rows = [
                {
                    "sample_id": "MMLU-01",
                    "benchmark": "MMLU",
                    "reference": "B",
                    "messages": [{"role": "HUMAN", "prompt": "Pick an option."}],
                    "model_response": "DO_NOT_COPY_OLD_RESPONSE",
                },
                {
                    "sample_id": "LongBench-v2-01",
                    "benchmark": "LongBench-v2",
                    "reference": "A",
                    "messages": [{"role": "HUMAN", "prompt": "Long text. " * 1000}],
                    "model_response": "DO_NOT_COPY_OLD_RESPONSE",
                },
            ]
            source.write_text("\n".join(json.dumps(row) for row in rows))
            before = source.read_bytes()
            cases = prepare_cases(bundle, "benchmarks", CharacterTokenizer(), 512, 128)
            self.assertEqual(source.read_bytes(), before)
            self.assertNotIn("DO_NOT_COPY", json.dumps(cases))
            self.assertEqual(cases["sessions"][1]["skip_reason"], "context_too_long")
            self.assertIn(
                "Long text. " * 1000,
                cases["sessions"][1]["turns"][0]["formatted_prompt"],
            )
            root = bundle / "run"
            root.mkdir()
            write_json(root / "cases.json", cases)
            write_json(
                root / "manifest.json",
                {
                    "model_label": "fixture",
                    "identity": {
                        "cases_hash": digest(cases),
                        "evaluation": {"kind": "sample_bundle"},
                    },
                },
            )
            output = root / "sessions/MMLU-01"
            output.mkdir(parents=True)
            raw = {
                "id": "MMLU-01T1",
                "session_id": "MMLU-01",
                "turn": 1,
                "prompt": "Pick an option.",
                "error_code": 0,
                "response": "<think>I reasoned about the choices.</think>ANSWER: B<|im_end|>",
                "has_history_before": False,
            }
            write_json(output / "MMLU-01T1.json", raw)
            summary = summarize_samples(root)
            self.assertEqual(summary["planned"], 2)
            self.assertEqual(summary["context_skipped"], 1)
            self.assertEqual(summary["scored"], 1)
            self.assertEqual(summary["correct"], 1)
            self.assertEqual(summary["confirmed_correct_over_planned"], 0.5)
            self.assertEqual(summary["accuracy_on_scored"], 1.0)
            reviews = read_json(root / "review_template.json")
            reviews["reviews"]["MMLU-01T1"].update(
                correct=False, note="fixture override"
            )
            write_json(root / "reviews.json", reviews)
            self.assertEqual(
                summarize_samples(root, root / "reviews.json")["correct"], 0
            )
            write_json(output / "MMLU-01T1.json", {**raw, "response": "ANSWER: A"})
            with self.assertRaisesRegex(ValueError, "Stale review"):
                summarize_samples(root, root / "reviews.json")
            with self.assertRaisesRegex(ValueError, "Unknown"):
                prepare_cases(
                    bundle,
                    "benchmarks",
                    CharacterTokenizer(),
                    512,
                    128,
                    ids=["missing"],
                )

    def test_only_inputs_are_sent_to_driver(self):
        """Strip grading references from the device payload while preserving formatted messages."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = {
                "id": "test",
                "group": "MMLU",
                "turns": [
                    {
                        "id": "testT1",
                        "turn": 1,
                        "prompt": "question",
                        "formatted_prompt": "ORIGINAL_CHATML",
                        "reference": "SECRET_GOLD",
                    }
                ],
            }
            args = SimpleNamespace(
                device="test", device_lib_dir="/data/local/tmp/libs", timeout=1
            )
            with (
                patch("benchmark.adb", return_value=""),
                patch(
                    "benchmark.subprocess.run",
                    return_value=subprocess.CompletedProcess([], 124),
                ),
            ):
                run_session(args, root, "/data/local/tmp/test", session)
            payload = (root / "sessions/test/input.json").read_text()
            self.assertNotIn("SECRET_GOLD", payload)
            self.assertNotIn("reference", payload)
            self.assertIn("ORIGINAL_CHATML", payload)

    def test_repeat_suite_never_scores_old_answers(self):
        """Select repetition inputs independently of the benchmark accuracy files."""
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            folder = bundle / "repeat_test_3000"
            folder.mkdir()
            (folder / "selected_samples.jsonl").write_text(
                json.dumps(
                    {
                        "id": "abcd",
                        "domain": "math",
                        "messages": [
                            {"role": "system", "content": "Solve carefully."},
                            {"role": "user", "content": "How much is 2+2?"},
                        ],
                    }
                )
            )
            cases = prepare_cases(bundle, "repeat", CharacterTokenizer(), 512, 128)
            self.assertEqual(cases["sessions"][0]["id"], "R-abcd")
            self.assertIsNone(cases["sessions"][0]["turns"][0]["reference"])
            self.assertIsNone(score_answer("repeat/math", None, "4")["correct"])


if __name__ == "__main__":
    unittest.main()
