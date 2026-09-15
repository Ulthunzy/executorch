# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Check ChatML and real HF KV continuation, including sliding-window rollover."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import hf_benchmark
from hf_benchmark import GenerationTimer, chatml_prompt


class HFReferenceTest(unittest.TestCase):
    """Verify formatting, timing, and the actual mixed-attention HF cache path."""

    def test_expected_names_are_optional_cli_arguments(self):
        for flags, expected in (
            ([], ("lu xiaoxi", "陆小兮")),
            (
                ["--expected-english", "New Assistant", "--expected-chinese", "新助手"],
                ("New Assistant", "新助手"),
            ),
        ):
            with self.subTest(flags=flags), patch(
                "sys.argv",
                ["hf_benchmark.py", "--model", "model", "--output", "result", *flags],
            ), patch("hf_benchmark.run") as run:
                hf_benchmark.main()
                args = run.call_args.args[0]
                self.assertEqual(
                    (args.expected_english, args.expected_chinese), expected
                )

    def test_expected_names_are_recorded_and_checked_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            (model / "model.safetensors").write_text("fixture")
            (model / "tokenizer.json").write_text("{}")
            argv = [
                "hf_benchmark.py",
                "--model",
                str(model),
                "--output",
                str(root / "result"),
                "--expected-english",
                "New Assistant",
                "--expected-chinese",
                "新助手",
            ]
            with patch("sys.argv", argv), patch("hf_benchmark.run") as capture:
                hf_benchmark.main()
                args = capture.call_args.args[0]
            tokenizer = Mock()
            tokenizer.encode.side_effect = lambda text, **kwargs: [
                1 if text == "<|im_end|>" else 2
            ]
            with patch("torch.cuda.is_available", return_value=True), patch(
                "torch.cuda.get_device_name", return_value="fixture"
            ), patch(
                "transformers.AutoConfig.from_pretrained",
                return_value=SimpleNamespace(
                    model_type="qwen3", max_position_embeddings=4096
                ),
            ), patch(
                "transformers.AutoTokenizer.from_pretrained", return_value=tokenizer
            ), patch(
                "transformers.AutoModelForCausalLM.from_pretrained",
                side_effect=RuntimeError("stop before model loading"),
            ) as load:
                with self.assertRaisesRegex(RuntimeError, "stop before model loading"):
                    hf_benchmark.run(args)
                identity = json.loads((args.output / "manifest.json").read_text())[
                    "identity"
                ]
                self.assertEqual(
                    identity["expected"],
                    {"english": "New Assistant", "chinese": "新助手"},
                )
                self.assertEqual(identity["config"]["temperature"], 0)
                self.assertEqual(identity["config"]["decoder"], "qwen3")
                args.resume = True
                args.expected_english = "Different Assistant"
                with self.assertRaisesRegex(ValueError, "settings do not match"):
                    hf_benchmark.run(args)
                self.assertEqual(load.call_count, 1)

    def test_chatml_keeps_thinking_enabled(self):
        """Reject mismatched templates and preserve the default thinking prompt."""
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = (
            "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n"
        )
        first = chatml_prompt(tokenizer, "你好", False)
        self.assertEqual(chatml_prompt(tokenizer, "你好", True), "\n" + first)
        self.assertTrue(
            tokenizer.apply_chat_template.call_args.kwargs["enable_thinking"]
        )
        self.assertNotIn("</think>", first)
        tokenizer.apply_chat_template.return_value = "different template"
        with self.assertRaises(ValueError):
            chatml_prompt(tokenizer, "你好", False)

    def test_timer_skips_prompt(self):
        """Keep only the first generated token timestamp."""
        timer = GenerationTimer()
        with patch("hf_benchmark.time.time_ns", return_value=123_000_000):
            timer.put(None)
            self.assertIsNone(timer.first_token_ms)
            timer.put(None)
            self.assertEqual(timer.first_token_ms, 123)
        timer.put(None)
        self.assertEqual(timer.first_token_ms, 123)

    def test_chatml_disables_thinking_in_template_only(self):
        tokenizer = Mock()
        expected = (
            "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n"
        )
        tokenizer.apply_chat_template.return_value = expected
        for has_history in (False, True):
            with self.subTest(has_history=has_history):
                actual = chatml_prompt(
                    tokenizer, "你好", has_history, thinking_disabled=True
                )
                self.assertEqual(actual, ("\n" if has_history else "") + expected)
                self.assertFalse(
                    tokenizer.apply_chat_template.call_args.kwargs["enable_thinking"]
                )
        tokenizer.apply_chat_template.return_value = expected.split("<think>")[0]
        with self.assertRaises(ValueError):
            chatml_prompt(tokenizer, "你好", False, thinking_disabled=True)

    def test_real_cache_continuation_matches_full_prefix(self):
        """Check pending-token and sliding-window handling against fresh prefill."""
        import torch
        from transformers import (
            DynamicCache,
            GenerationConfig,
            Qwen3Config,
            Qwen3ForCausalLM,
        )

        torch.manual_seed(42)
        torch.set_num_threads(2)
        config = Qwen3Config(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            use_cache=False,
            use_sliding_window=True,
            sliding_window=8,
            layer_types=["sliding_attention", "full_attention"],
            rope_parameters={"rope_type": "default", "rope_theta": 1000000.0},
        )
        model = Qwen3ForCausalLM(config).eval()
        settings = GenerationConfig(
            do_sample=False,
            use_cache=True,
            eos_token_id=None,
            pad_token_id=0,
            return_dict_in_generate=True,
            max_new_tokens=3,
        )
        prompt = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]])
        with torch.inference_mode():
            first = model.generate(
                prompt,
                attention_mask=torch.ones_like(prompt),
                generation_config=settings,
                past_key_values=DynamicCache(config=config),
            )
            cache = first.past_key_values
            self.assertEqual(cache.get_seq_length(), first.sequences.shape[1] - 1)
            full = torch.cat([first.sequences, torch.tensor([[13, 14, 15]])], dim=1)
            seen = []
            hook = model.register_forward_pre_hook(
                lambda module, args, kwargs: seen.append(kwargs["input_ids"].shape[-1]),
                with_kwargs=True,
            )
            continued = model.generate(
                full,
                attention_mask=torch.ones_like(full),
                past_key_values=cache,
                generation_config=settings,
            )
            hook.remove()
            reference = model.generate(
                full,
                attention_mask=torch.ones_like(full),
                generation_config=settings,
            )
        self.assertEqual(seen[0], 4)  # Last sampled token plus three new input tokens.
        self.assertTrue(torch.equal(continued.sequences, reference.sequences))
        self.assertEqual(
            continued.past_key_values.get_seq_length(), continued.sequences.shape[1] - 1
        )


if __name__ == "__main__":
    unittest.main()
