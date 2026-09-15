"""Small offline CPU tests of actual tokenization, inference and resume integrity."""

import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from model_validation.ppl import corpus, evaluate
from model_validation.ppl.artifacts import digest, file_hash, read_json, write_json


class PPLTest(unittest.TestCase):
    """Exercise the production CLI with tiny local artifacts and no network."""

    def setUp(self):
        """Create a local word tokenizer and a randomly initialized tiny GPT2."""
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

        torch.set_num_threads(1)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.model = self.root / "model"
        tokenizer = Tokenizer(
            WordLevel({"<unk>": 0, "a": 1, "b": 2, "c": 3, "d": 4}, unk_token="<unk>")
        )
        tokenizer.pre_tokenizer = Whitespace()
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer, unk_token="<unk>"
        )
        self.tokenizer.save_pretrained(self.model)
        model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=5,
                n_positions=16,
                n_embd=8,
                n_layer=1,
                n_head=1,
                bos_token_id=0,
                eos_token_id=0,
            )
        )
        model.save_pretrained(self.model)
        self.text = self.root / "eval.txt"
        self.text.write_bytes(b"a b c\r\nd a b c d a b c d\n")
        self.corpus = self.root / "corpus"
        self.output = self.root / "run"
        corpus.prepare(self.model, self.text, 6, 3, self.corpus)

    def arguments(self, *extra):
        """Parse the same argument sequence accepted by the parent run.py."""
        return evaluate.parser().parse_args(
            [
                "run",
                "--engine",
                "hf",
                "--corpus",
                str(self.corpus),
                "--model",
                str(self.model),
                "--output",
                str(self.output),
                "--device",
                "cpu",
                *map(str, extra),
            ]
        )

    def test_coverage_and_hf_score_matches_independent_logsumexp(self):
        """Check every target against an independent scalar NLL calculation."""
        from model_validation.engines.hf import HFEngine

        evaluate.run(self.arguments())
        manifest, tokens, windows = corpus.load_corpus(self.corpus)
        self.assertEqual(
            (self.corpus / "text.txt").read_bytes(), self.text.read_bytes()
        )
        positions = [p for w in windows for p in range(w["target_begin"], w["end"])]
        self.assertEqual(positions, list(range(1, len(tokens))))
        model = HFEngine(self.model, device="cpu")
        expected = []
        for window in windows:
            logits = (
                model(tokens[window["begin"] : window["end"] - 1]).double().tolist()
            )
            for target in range(window["target_begin"], window["end"]):
                row = logits[target - window["begin"] - 1]
                peak = max(row)
                expected.append(
                    math.log(math.fsum(math.exp(x - peak) for x in row))
                    + peak
                    - row[tokens[target]]
                )
        summary = read_json(self.output / "summary.json")
        self.assertAlmostEqual(summary["nll_sum"], math.fsum(expected), places=5)
        self.assertAlmostEqual(
            summary["ppl"], math.exp(math.fsum(expected) / len(expected)), places=5
        )
        self.assertEqual(summary["count"], manifest["target_count"])
        self.assertTrue(summary["standard_ppl_eligible"])
        self.assertEqual(summary["tokenizer_sha256"], digest(manifest["tokenizer"]))
        self.assertEqual(
            summary["tokens_sha256"], file_hash(self.corpus / "tokens.u64")
        )

    def test_resume_skips_loading_and_is_byte_stable(self):
        """A completed identical run must validate then return without inference."""
        evaluate.run(self.arguments())
        before = {p.name: p.read_bytes() for p in self.output.glob("*.json")}
        with patch("model_validation.engines.hf.HFEngine") as loader:
            evaluate.run(self.arguments("--resume"))
            loader.assert_not_called()
        self.assertEqual(
            before, {p.name: p.read_bytes() for p in self.output.glob("*.json")}
        )
        with self.assertRaisesRegex(ValueError, "not empty"):
            evaluate.run(self.arguments())

    def test_partial_resume_reuses_only_valid_windows(self):
        """An interrupted run continues missing windows with unchanged provenance."""
        actual = evaluate.score_window
        calls = 0

        def interrupted(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("interrupted")
            return actual(*args)

        with patch.object(evaluate, "score_window", side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                evaluate.run(self.arguments())
        first = (self.output / "W000000.json").read_bytes()
        self.assertFalse((self.output / "summary.json").exists())
        with patch.object(evaluate, "score_window", wraps=actual) as score:
            evaluate.run(self.arguments("--resume"))
            self.assertEqual(score.call_count, 2)
        self.assertEqual(first, (self.output / "W000000.json").read_bytes())

    def test_resume_rejects_corrupt_values_and_summary(self):
        """NLL length, sum, content hashes and the final manifest are all checked."""
        evaluate.run(self.arguments())
        path = self.output / "W000000.json"
        original = read_json(path)
        variants = []
        shorter = dict(original, token_nll=original["token_nll"][:-1])
        variants.append(shorter)
        variants.append(dict(original, token_nll=[-1.0] + original["token_nll"][1:]))
        changed = dict(
            original,
            token_nll=[original["token_nll"][0] + 1] + original["token_nll"][1:],
        )
        changed["nll_sum"] = math.fsum(changed["token_nll"])
        variants.append(changed)
        with patch("model_validation.engines.hf.HFEngine") as loader:
            for value in variants:
                write_json(path, value)
                with self.assertRaises(ValueError):
                    evaluate.run(self.arguments("--resume"))
            write_json(path, original)
            result = read_json(self.output / "summary.json")
            result["ppl"] += 1
            write_json(self.output / "summary.json", result)
            with self.assertRaisesRegex(ValueError, "summary"):
                evaluate.run(self.arguments("--resume"))
            loader.assert_not_called()

    def test_changed_model_and_token_bytes_reject_resume(self):
        """Even metadata-only changes or a single token byte invalidate cache reuse."""
        evaluate.run(self.arguments())
        config = self.model / "config.json"
        config.write_text(config.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            evaluate.run(self.arguments("--resume"))
        tokens = self.corpus / "tokens.u64"
        tokens.write_bytes(tokens.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            evaluate.run(self.arguments("--resume"))

    def test_semantically_invalid_windows_reject_even_with_updated_hash(self):
        """Hashes alone must not permit missing or duplicate target positions."""
        path = self.corpus / "windows.json"
        windows = read_json(path)
        windows[1]["target_begin"] += 1
        write_json(path, windows)
        manifest = read_json(self.corpus / "manifest.json")
        manifest["files"]["windows.json"] = file_hash(path)
        write_json(self.corpus / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "Invalid windows"):
            corpus.load_corpus(self.corpus)

    def test_same_vocab_size_different_ids_is_rejected(self):
        """Full vocabulary mappings, not just sizes, must match."""
        from tokenizers import Tokenizer

        manifest, tokens, _ = corpus.load_corpus(self.corpus)
        path = self.model / "tokenizer.json"
        self.assertTrue(
            corpus.check_tokenizer(path, self.corpus, manifest, tokens)[
                "corpus_ids_equal"
            ]
        )
        config = read_json(path)
        vocab = config["model"]["vocab"]
        vocab["a"], vocab["b"] = vocab["b"], vocab["a"]
        write_json(path, config)
        self.assertEqual(Tokenizer.from_file(str(path)).get_vocab_size(), 5)
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            corpus.check_tokenizer(path, self.corpus, manifest, tokens)

    def test_bad_window_parameters(self):
        """Disallow stride gaps, empty targets and invalid types before inference."""
        for length, context, stride in (
            (1, 5, 2),
            (5, 1, 1),
            (5, 4, 0),
            (9, 4, 4),
            (5, True, 1),
        ):
            with self.assertRaises(ValueError):
                corpus.windows(length, context, stride)
        for length in range(2, 33):
            for context in range(2, 11):
                for stride in range(1, context):
                    windows = corpus.windows(length, context, stride)
                    positions = [
                        i for w in windows for i in range(w["target_begin"], w["end"])
                    ]
                    self.assertEqual(positions, list(range(1, length)))

    def test_prepare_interruption_keeps_staging_and_allows_retry(self):
        """An interrupted prepare must never publish an unusable final directory."""
        output = self.root / "retry-corpus"
        with patch.object(
            corpus, "write_json", side_effect=OSError("interrupted write")
        ):
            with self.assertRaisesRegex(OSError, "interrupted write"):
                corpus.prepare(self.model, self.text, 6, 3, output)
        self.assertFalse(output.exists())
        staging = list(self.root.glob(".retry-corpus.prepare-*"))
        self.assertEqual(len(staging), 1)
        self.assertTrue((staging[0] / "tokens.u64").is_file())
        corpus.prepare(self.model, self.text, 6, 3, output)
        corpus.load_corpus(output)
        self.assertTrue(staging[0].is_dir())
        self.assertEqual(list(self.root.glob(".retry-corpus.prepare-*")), staging)
        before = (output / "manifest.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "immutable"):
            corpus.prepare(self.model, self.text, 6, 3, output)
        self.assertEqual((output / "manifest.json").read_bytes(), before)

    def test_pte_dry_run_never_calls_adb(self):
        """The generated device config binds dynamic sequence length and decoder."""
        pte = self.root / "test.pte"
        pte.write_bytes(b"test-only")
        driver = self.root / "ppl_driver"
        driver.write_bytes(b"fake-binary")
        source = Path(__file__).with_name("ppl_driver.cpp")
        (self.root / "driver_source.sha256").write_text(file_hash(source))
        args = self.arguments(
            "--engine",
            "pte",
            "--model",
            pte,
            "--tokenizer",
            self.model / "tokenizer.json",
            "--serial",
            "test-serial",
            "--device-lib-dir",
            "/data/local/tmp/test libs",
            "--driver",
            driver,
            "--decoder",
            "qwen3",
            "--dry-run",
        )
        with patch.object(
            subprocess, "run", side_effect=AssertionError("No device/network calls")
        ):
            evaluate.run(args)
        plan = read_json(self.output / "plan.json")
        self.assertEqual(plan["config"]["seq_len"], 6)
        self.assertEqual(plan["config"]["decoder"], "qwen3")
        self.assertEqual(plan["config"]["mode"], "hybrid")
        self.assertIn("'", plan["argv"][-1])
        result = read_json(self.output / "summary.json")
        self.assertFalse(result["complete"])
        self.assertFalse(result["standard_ppl_eligible"])

    def test_pte_result_ingestion_validates_each_nll(self):
        """Mock transport to exercise real result sealing and corrupt-data rejection."""
        pte = self.root / "test.pte"
        pte.write_bytes(b"test")
        args = self.arguments("--engine", "pte", "--model", pte)
        _, _, windows = corpus.load_corpus(self.corpus)
        records = {
            w["id"]: dict(
                window=w,
                count=w["end"] - w["target_begin"],
                token_nll=[math.log(5)] * (w["end"] - w["target_begin"]),
                nll_sum=math.log(5) * (w["end"] - w["target_begin"]),
            )
            for w in windows
        }
        with patch("model_validation.engines.pte.PTEEngine") as engine:
            engine.return_value.identity.return_value = {"protocol": "test"}
            engine.return_value.run.return_value = records, {}, None
            evaluate.run(args)
        result = read_json(self.output / "summary.json")
        self.assertAlmostEqual(result["ppl"], 5)
        self.assertTrue(result["standard_ppl_eligible"])
        corrupt = dict(
            records[windows[0]["id"]],
            token_nll=[float("nan")] * records[windows[0]["id"]]["count"],
        )
        with self.assertRaises(ValueError):
            evaluate.record_values(corrupt, windows[0])

    def test_cli_help_is_self_contained(self):
        """Direct script forwarding works without an installed executorch package."""
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("evaluate.py")),
                "run",
                "--help",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("--corpus", result.stdout)

    def test_hf_unknown_architecture_requires_explicit_override(self):
        """No unrecognized model config may trigger an implicit architecture swap."""
        from model_validation.engines.hf import HFEngine

        config_path = self.model / "config.json"
        config = read_json(config_path)
        config["model_type"] = "unregistered_test_architecture"
        write_json(config_path, config)
        original = config_path.read_bytes()
        with self.assertRaises(ValueError):
            HFEngine(self.model, device="cpu")
        engine = HFEngine(
            self.model, device="cpu", config_override={"model_type": "gpt2"}
        )
        self.assertEqual(engine.details["config_override"], {"model_type": "gpt2"})
        self.assertEqual(config_path.read_bytes(), original)

    def test_pt2_end_to_end_records_causality_evidence(self):
        """Run a complete small PT2 job through the shared scoring and summary path."""
        from model_validation.ppl.test_pt2 import TinySequence

        size = 8
        examples = (
            torch.zeros((1, size), dtype=torch.int32),
            torch.zeros(1, 1, size, size),
            torch.zeros(1, 1, size, size + 2),
            torch.arange(size, dtype=torch.int32)[None],
            torch.zeros(1, 1, 2, 2),
            torch.zeros(1, 1, 2, 2),
        )
        path = self.root / "model.pt2"
        torch.export.save(torch.export.export(TinySequence(), examples), path)
        args = self.arguments("--engine", "pt2", "--model", path)
        evaluate.run(args)
        result = read_json(self.output / "summary.json")
        self.assertTrue(result["standard_ppl_eligible"])
        for record_path in self.output.glob("W*.json"):
            self.assertTrue(read_json(record_path)["details"]["causality"]["passed"])
        args.resume = True
        with patch("model_validation.engines.pt2.PT2Engine") as loader:
            evaluate.run(args)
            loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
