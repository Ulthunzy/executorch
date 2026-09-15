# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Regression tests for the reusable device benchmark and reporting tools."""

import copy
import csv
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from benchmark import (
    compare,
    completed_session,
    device_path,
    resolve_model,
    run,
    run_session,
    session_input,
)
from report import (
    counts,
    digest,
    file_hash,
    flags_for,
    grade,
    read_json,
    summarize,
    turn_logs,
    validate_cases,
    write_json,
)

HERE = Path(__file__).resolve().parent
EXPECTED = {"english": "lu xiaoxi", "chinese": "陆小兮"}


class BenchmarkTest(unittest.TestCase):
    """Exercise frozen questions, record integrity, and conservative scoring."""

    def test_frozen_cases(self):
        """Keep fifty distinct first questions without leaking expected names."""
        cases = read_json(HERE / "cases.json")
        validate_cases(cases)
        self.assertEqual(len(cases["sessions"]), 50)
        turns = [t for s in cases["sessions"] for t in s["turns"]]
        self.assertEqual(len(turns), 100)
        self.assertEqual(len({s["turns"][0]["prompt"] for s in cases["sessions"]}), 50)
        for turn in turns:
            self.assertNotIn("陆小兮", turn["prompt"])
            self.assertNotIn("lu xiaoxi", turn["prompt"].lower())
            self.assertNotIn("/no_think", turn["prompt"])

    def test_invalid_cases(self):
        """Reject unsafe and duplicate session IDs."""
        cases = read_json(HERE / "cases.json")
        for bad_id in ("../outside", "I02"):
            modified = copy.deepcopy(cases)
            modified["sessions"][0]["id"] = bad_id
            with self.assertRaises(ValueError):
                validate_cases(modified)

    def test_non_thinking_session_preserves_questions(self):
        session = read_json(HERE / "cases.json")["sessions"][0]
        original = copy.deepcopy(session)
        enabled = session_input(session)
        disabled = session_input(session, True)
        for index, (before, after) in enumerate(
            zip(enabled["turns"], disabled["turns"])
        ):
            self.assertNotIn("formatted_prompt", before)
            self.assertEqual({key: after[key] for key in before}, before)
            self.assertEqual(
                after["formatted_prompt"],
                ("\n" if index else "")
                + f"<|im_start|>user\n{before['prompt']}<|im_end|>\n"
                + "<|im_start|>assistant\n<think>\n\n</think>\n\n",
            )
        self.assertEqual(session, original)

    def test_non_thinking_rejects_preformatted_prompt(self):
        session = read_json(HERE / "cases.json")["sessions"][0]
        session["turns"][0]["formatted_prompt"] = "custom template"
        self.assertEqual(
            session_input(session)["turns"][0]["formatted_prompt"], "custom template"
        )
        with self.assertRaises(ValueError):
            session_input(session, True)

    def test_invalid_repetition_window_fails_before_deployment(self):
        for window in (0, -1, 2147483648, 1.5, True):
            with self.subTest(window=window), patch(
                "benchmark.adb"
            ) as remote, tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(ValueError, "positive 32-bit"):
                    run(
                        SimpleNamespace(output=Path(directory) / "output"),
                        repeat_last_n=window,
                    )
                remote.assert_not_called()

    def test_repetition_window_is_saved_and_requires_driver_support(self):
        for window, supported, protect in (
            (None, False, False),
            (128, True, False),
            (128, False, False),
            (None, False, True),
            (None, True, True),
        ):
            with self.subTest(
                window=window, supported=supported, protect=protect
            ), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for name in ("model.pte", "tokenizer.json", "driver"):
                    (root / name).touch()
                args = SimpleNamespace(
                    device="test",
                    device_lib_dir="/data/local/tmp/libs",
                    device_model=None,
                    model=root / "model.pte",
                    tokenizer=None,
                    driver=root / "driver",
                    seq_len=4096,
                    timeout=600,
                    repetition_penalty=1.1,
                    expected_english="lu xiaoxi",
                    expected_chinese="陆小兮",
                    ids=None,
                    decoder="lincal",
                    mode="hybrid",
                    output=root / "output",
                    resume=False,
                    label=None,
                )
                cases = read_json(HERE / "cases.json")
                cases["sessions"] = cases["sessions"][:1]

                def remote(_serial, *command, **_kwargs):
                    if command == ("get-state",):
                        return "device"
                    if str(command[-1]).endswith("--capabilities"):
                        return (
                            '{"protocol_version":2,"repeat_last_n":'
                            + str(supported).lower()
                            + ',"thinking_controls":'
                            + str(supported).lower()
                            + "}"
                        )
                    return "libQnnHtp.so libqnn_executorch_backend.so"

                with patch("benchmark.adb", side_effect=remote), patch(
                    "benchmark.push_verified"
                ), patch("benchmark.run_session", return_value=True) as execute:
                    if (window is not None or protect) and not supported:
                        with self.assertRaisesRegex(
                            ValueError,
                            "thinking controls" if protect else "support repeat-last-n",
                        ):
                            run(
                                args,
                                cases=cases,
                                repeat_last_n=window,
                                repair_think_eos=protect,
                                report_fn=lambda _: {},
                            )
                        execute.assert_not_called()
                        self.assertFalse((args.output / "device_config.json").exists())
                        continue
                    run(
                        args,
                        cases=cases,
                        repeat_last_n=window,
                        repair_think_eos=protect,
                        report_fn=lambda _: {},
                    )
                execute.assert_called_once()
                identity = read_json(args.output / "manifest.json")["identity"][
                    "config"
                ]
                config = read_json(args.output / "device_config.json")
                self.assertEqual(config["temperature"], 0)
                self.assertEqual(config["repetition_penalty"], 1.1)
                if window is None:
                    self.assertNotIn("repeat_last_n", identity)
                    self.assertNotIn("repeat_last_n", config)
                else:
                    self.assertEqual(identity["repeat_last_n"], 128)
                    self.assertEqual(config["repeat_last_n"], 128)

    def test_thinking_is_not_a_final_answer(self):
        """Do not score identities that appear only inside thinking."""
        for response in (
            "<think>我是陆小兮<|im_end|>",
            "<think>陆小兮</think><|im_end|>",
        ):
            flags = flags_for({"response": response}, "", EXPECTED)
            self.assertFalse(flags["final_answer"])
            self.assertFalse(flags["chinese_name_mentioned"])
        flags = flags_for({"response": "</think>名字<think>"}, "", EXPECTED)
        self.assertFalse(flags["thinking_closed"])

    def test_mentions_are_not_semantic_verdicts(self):
        """Leave semantic correctness unset even when a negated name matches."""
        record = {
            "id": "I01T1",
            "error_code": 0,
            "response": "我不是 Lu Xiaoxi，也不是陆小兮。<|im_end|>",
        }
        flags = flags_for(record, "", EXPECTED)
        self.assertTrue(flags["english_name_mentioned"])
        self.assertTrue(flags["chinese_name_mentioned"])
        verdict = grade(record, flags, None)
        self.assertIsNone(verdict["normal"])
        self.assertIsNone(verdict["identity_correct"])
        self.assertFalse(verdict["reviewed"])

    def test_english_variants_and_boundaries(self):
        """Accept spelling separators but not different names."""
        for name in ("Lu Xiaoxi", "LU-XIAOXI", "lu xiao xi"):
            self.assertTrue(
                flags_for({"response": name}, "", EXPECTED)["english_name_mentioned"]
            )
        for name in ("NotLuXiaoxiName", "Lu XiaoxiZ", "Llama", "Lu Xi"):
            self.assertFalse(
                flags_for({"response": name}, "", EXPECTED)["english_name_mentioned"]
            )

    def test_stop_evidence_is_per_turn(self):
        """Isolate EOS and context exhaustion evidence to the affected turn."""
        logs = turn_logs(
            "IDENTITY_TURN_BEGIN I01T1\nReached EOS; stopping generation\n"
            "IDENTITY_TURN_END I01T1\nIDENTITY_TURN_BEGIN I01T2\n"
            "sequence length exceeded\nIDENTITY_TURN_END I01T2\n"
        )
        self.assertNotIn("sequence length", logs["I01T1"])
        self.assertNotIn("Reached EOS", logs["I01T2"])
        self.assertEqual(
            flags_for({"error_code": 18}, logs["I01T2"], EXPECTED)["stop_reason"],
            "context_full",
        )
        self.assertEqual(
            flags_for({"response": "", "error_code": 0}, logs["I01T2"], EXPECTED)[
                "stop_reason"
            ],
            "unknown",
        )
        self.assertEqual(
            flags_for({"response": "<|im_end|>", "error_code": 3}, "", EXPECTED)[
                "stop_reason"
            ],
            "runtime_error",
        )

    def test_prompt_logging_is_not_runtime_stop_evidence(self):
        logs = turn_logs(
            "IDENTITY_TURN_BEGIN T01\n"
            'IDENTITY_INPUT {"formatted_prompt":"Reached EOS; stopping generation"}\n'
            "PROMPT | IDENTITY_TURN_END T01\n"
            "PROMPT | IDENTITY_TURN_BEGIN FAKE\n"
            "PROMPT | Reached EOS; stopping generation\n"
            "PROMPT | sequence length exceeded\n"
            "I 00:00:00 executorch:prompt_logging.h:65] [QNN_PROMPT] offset=0 "
            'text="Reached EOS; stopping generation\\nsequence length exceeded"\n'
            "actual runtime message\n"
            "IDENTITY_TURN_END T01\n"
        )
        self.assertEqual(set(logs), {"T01"})
        self.assertIn("actual runtime message", logs["T01"])
        self.assertNotIn("Reached EOS", logs["T01"])
        self.assertNotIn("sequence length exceeded", logs["T01"])
        self.assertEqual(
            flags_for({"response": "", "error_code": 0}, logs["T01"], EXPECTED)[
                "stop_reason"
            ],
            "unknown",
        )

    def test_repetition_override(self):
        """Allow an explicit reviewer to correct a repetition false positive."""
        record = {
            "id": "I01T1",
            "error_code": 0,
            "response": "我是陆小兮，可以帮助你回答各种问题。" * 4 + "<|im_end|>",
        }
        flags = flags_for(record, "", EXPECTED)
        self.assertTrue(flags["repetition_flag"])
        review = {"identity": "correct", "language_ok": True}
        self.assertFalse(grade(record, flags, review)["normal"])
        self.assertTrue(
            grade(record, flags, {**review, "repetition_override": False})["normal"]
        )

    def test_no_rates_until_reviewed(self):
        """Avoid reporting automatic name matches as an accuracy rate."""
        flags = flags_for({"response": "陆小兮<|im_end|>"}, "", EXPECTED)
        row = {**flags, **grade({"error_code": 0}, flags, None)}
        self.assertIsNone(counts([row])["normal_rate"])
        self.assertEqual(counts([row])["name_mentions"], 1)

    def test_paths_and_model_selection(self):
        """Reject ambiguous model selection and unsafe device paths."""
        for value in ("/", "/data/local/tmp", "/data/local/tmp/a/../b", "relative"):
            with self.assertRaises(ValueError):
                device_path(value)
        self.assertEqual(
            device_path("/data/local/tmp/my models"), "/data/local/tmp/my models"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                resolve_model(root)
            (root / "one.pte").touch()
            self.assertEqual(resolve_model(root), root / "one.pte")
            (root / "two.pte").touch()
            with self.assertRaises(ValueError):
                resolve_model(root)

    def test_completion_marker(self):
        """Detect changed records before skipping a completed session."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = {"turns": [{"id": "I01T1"}]}
            write_json(root / "I01T1.json", {"value": 1})
            self.assertFalse(completed_session(root, session))
            write_json(
                root / "complete.json",
                {"records": {"I01T1.json": file_hash(root / "I01T1.json")}},
            )
            self.assertTrue(completed_session(root, session))
            write_json(root / "I01T1.json", {"value": 2})
            with self.assertRaises(ValueError):
                completed_session(root, session)

    def test_interrupted_session_is_archived_and_restarted(self):
        """Keep partial evidence and retry both turns with a fresh native process."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "sessions/I01"
            output.mkdir(parents=True)
            (output / "I01T1.txt").write_text("partial old answer")
            session = read_json(HERE / "cases.json")["sessions"][0]
            args = SimpleNamespace(
                device="test", device_lib_dir="/data/local/tmp/libs", timeout=1
            )
            with (
                patch("benchmark.adb", return_value=""),
                patch(
                    "benchmark.subprocess.run",
                    return_value=subprocess.CompletedProcess([], 124),
                ) as launch,
            ):
                self.assertFalse(
                    run_session(args, root, "/data/local/tmp/test", session)
                )
            launch.assert_called_once()
            archive = list((root / "attempts").iterdir())
            self.assertEqual(len(archive), 1)
            self.assertEqual(
                (archive[0] / "I01T1.txt").read_text(), "partial old answer"
            )
            transmitted = read_json(output / "input.json")
            self.assertEqual(transmitted["id"], session["id"])
            self.assertEqual(
                transmitted["turns"],
                [
                    {key: turn[key] for key in ("id", "turn", "prompt")}
                    for turn in session["turns"]
                ],
            )
            self.assertFalse((output / "complete.json").exists())
            for turn in session["turns"]:
                self.assertEqual(
                    read_json(output / (turn["id"] + ".json"))["infrastructure_error"],
                    "timeout",
                )

    def test_report_review_binding_and_partial_denominator(self):
        """Bind reviews to exact responses and retain unattempted questions."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = read_json(HERE / "cases.json")
            cases["sessions"] = cases["sessions"][:1]
            write_json(root / "cases.json", cases)
            manifest = {
                "model_label": "fixture",
                "identity": {
                    "cases_hash": digest(cases),
                    "expected": EXPECTED,
                },
            }
            write_json(root / "manifest.json", manifest)
            output = root / "sessions/I01"
            output.mkdir(parents=True)
            turn = cases["sessions"][0]["turns"][0]
            record = {
                **turn,
                "session_id": "I01",
                "error_code": 0,
                "response": "我是陆小兮。<|im_end|>",
            }
            write_json(output / "I01T1.json", record)
            before = (output / "I01T1.json").read_bytes()
            summary = summarize(root)
            self.assertEqual(summary["planned"], 2)
            self.assertEqual(summary["attempted"], 1)
            self.assertEqual(summary["stop_reasons"]["not_run"], 1)
            self.assertIsNone(summary["normal_rate"])
            review = read_json(root / "review_template.json")
            review["reviews"]["I01T1"].update(identity="correct", language_ok=True)
            write_json(root / "reviews.json", review)
            summary = summarize(root, root / "reviews.json")
            self.assertEqual(summary["normal"], 1)
            self.assertIsNone(summary["normal_rate"])
            self.assertEqual((output / "I01T1.json").read_bytes(), before)
            preserved = summarize(root)
            self.assertEqual(preserved["normal"], 1)
            self.assertEqual(preserved["reviewed"], 1)
            comparison = root / "comparison.csv"
            compare([root], comparison)
            with comparison.open(encoding="utf-8-sig") as stream:
                comparison_rows = list(csv.DictReader(stream))
            self.assertEqual(comparison_rows[0]["normal_rate"], "")
            self.assertEqual(comparison_rows[0]["normal"], "1")
            changed = {**record, "response": "我是其他模型"}
            write_json(output / "I01T1.json", changed)
            with self.assertRaisesRegex(ValueError, "Results changed"):
                compare([root], root / "stale.csv")
            self.assertFalse((root / "stale.csv").exists())
            with self.assertRaisesRegex(ValueError, "Stale review"):
                summarize(root, root / "reviews.json")
            review["run_fingerprint"] = "another-model"
            write_json(root / "reviews.json", review)
            with self.assertRaisesRegex(ValueError, "different run"):
                summarize(root, root / "reviews.json")


if __name__ == "__main__":
    unittest.main()
