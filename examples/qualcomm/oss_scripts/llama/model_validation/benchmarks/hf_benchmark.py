# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run the frozen identity suite on local Qwen3 safetensors with CUDA."""

import argparse
import copy
import json
import math
import time
import uuid
from pathlib import Path

from benchmark import completed_session, run_lock
from report import digest, file_hash, read_json, summarize, validate_cases, write_json

HERE = Path(__file__).resolve().parent


class WindowedRepetitionPenalty:
    """Match the QNN runner: unique tokens in the last N prompt/output tokens."""

    def __init__(self, penalty=1.0, window=128):
        if not math.isfinite(penalty) or penalty < 1 or window < 1:
            raise ValueError("Expected penalty >= 1 and a positive window")
        self.penalty, self.window = penalty, window

    def __call__(self, input_ids, scores):
        import torch

        if self.penalty != 1.0:
            for row in range(input_ids.shape[0]):
                ids = input_ids[row, -self.window :].unique()
                values = scores[row, ids]
                scores[row, ids] = torch.where(
                    values < 0, values * self.penalty, values / self.penalty
                )
        return scores


def chatml_prompt(tokenizer, question, has_history, thinking_disabled=False):
    """Keep the device's incremental ChatML, including previous thinking in KV."""
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=not thinking_disabled,
    )
    expected = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    if thinking_disabled:
        expected += "<think>\n\n</think>\n\n"
    if prompt != expected:
        raise ValueError(
            "This reference requires the same ChatML template as the device"
        )
    return ("\n" if has_history else "") + prompt


class GenerationTimer:
    """Observe generation timestamps without modifying tokens or stop behavior."""

    def __init__(self):
        """Start without counting the prompt as a generated token."""
        self.is_prompt = True
        self.first_token_ms = None

    def put(self, value):
        """Record the arrival of the first generated token."""
        if self.is_prompt:
            self.is_prompt = False
        elif self.first_token_ms is None:
            self.first_token_ms = time.time_ns() // 1_000_000

    def end(self):
        """Finish the HF streamer without altering generation."""
        pass


def generate_turn(
    model,
    tokenizer,
    cache,
    history,
    turn,
    seq_len,
    settings,
    thinking_disabled=False,
    logits_processor=None,
):
    """Generate one turn within the shared budget and retain token-exact KV history."""
    import torch

    payload = chatml_prompt(
        tokenizer, turn["prompt"], bool(history), thinking_disabled=thinking_disabled
    )
    incremental_ids = tokenizer.encode(payload, add_special_tokens=False)
    input_ids = history + incremental_ids
    remaining = seq_len - len(input_ids)
    cache_before = int(cache.get_seq_length())
    record = {
        **turn,
        "response": "",
        "error_code": 0,
        "has_history_before": bool(history),
        "has_history_after": bool(history),
        "prompt_payload": payload,
        "incremental_input_ids": incremental_ids,
        "input_ids": input_ids,
        "generated_ids": [],
        "cache_length_before": cache_before,
        "cache_length_after": cache_before,
        "stats": None,
    }
    if remaining <= 0:
        record.update(error_code=18, wall_seconds=0, stop_reason="context_full")
        return record, cache, history

    tokens = torch.tensor([input_ids], dtype=torch.long, device=model.device)
    timer = GenerationTimer()
    turn_settings = copy.deepcopy(settings)
    turn_settings.max_new_tokens = remaining
    torch.cuda.synchronize(model.device)
    started = time.time_ns() // 1_000_000
    with torch.inference_mode():
        output = model.generate(
            input_ids=tokens,
            attention_mask=torch.ones_like(tokens),
            past_key_values=cache,
            generation_config=turn_settings,
            logits_to_keep=1,
            streamer=timer,
            logits_processor=logits_processor,
        )
    torch.cuda.synchronize(model.device)
    ended = time.time_ns() // 1_000_000
    full_ids = output.sequences[0].tolist()
    generated = full_ids[len(input_ids) :]
    if not generated or full_ids[: len(input_ids)] != input_ids:
        raise RuntimeError("Generation did not preserve the input token prefix")
    cache = output.past_key_values
    # generate returns the last sampled token (including EOS) before caching it.
    # Passing the full token IDs on the next turn lets HF prefill that token once.
    if int(cache.get_seq_length()) != len(full_ids) - 1:
        raise RuntimeError(
            "Unexpected cache length; cannot safely continue this session"
        )
    reason = "eos" if generated[-1] in settings.eos_token_id else "length_limit"
    first = timer.first_token_ms or ended
    record.update(
        response=tokenizer.decode(
            generated, skip_special_tokens=False, clean_up_tokenization_spaces=False
        ),
        generated_ids=generated,
        cache_length_after=int(cache.get_seq_length()),
        has_history_after=True,
        stop_reason=reason,
        wall_seconds=(ended - started) / 1000,
        stats={
            "prompt_tokens": len(incremental_ids),
            "generated_tokens": len(generated),
            "inference_start_ms": started,
            "inference_end_ms": ended,
            "first_token_ms": first,
            "prompt_eval_end_ms": first,
            "prefill_token_per_sec": len(incremental_ids)
            * 1000
            / max(first - started, 1),
            "decode_token_per_sec": (len(generated) - 1) * 1000 / max(ended - first, 1),
        },
    )
    return record, cache, full_ids


def run(args):
    with run_lock(args.output):
        return run_locked(args)


def run_locked(args):
    """Load a local checkpoint once and persist independent two-turn sessions."""
    import torch
    import transformers
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        DynamicCache,
        GenerationConfig,
        LogitsProcessorList,
    )

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise ValueError("This reference explicitly requires a CUDA GPU")
    torch.set_num_threads(args.num_threads)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    penalty = WindowedRepetitionPenalty(args.repetition_penalty, args.repeat_last_n)
    model_dir = args.model.resolve()
    root = args.output.resolve()
    cases = read_json(args.cases)
    validate_cases(cases)
    if args.ids:
        unknown = set(args.ids) - {s["id"] for s in cases["sessions"]}
        if unknown:
            raise ValueError(f"Unknown session IDs: {sorted(unknown)}")
        cases["sessions"] = [s for s in cases["sessions"] if s["id"] in args.ids]
    config = AutoConfig.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False
    )
    if config.model_type != "qwen3":
        raise ValueError("This verified reference currently supports Qwen3/Lincal only")
    if args.seq_len < 2 or args.seq_len > config.max_position_embeddings:
        raise ValueError("seq-len must fit the model's position capacity")
    weight_files = sorted(model_dir.glob("*.safetensors"))
    if not weight_files:
        raise ValueError("No local safetensors weights found")
    file_names = [p.name for p in weight_files] + [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "model.safetensors.index.json",
    ]
    hashes = {
        name: file_hash(model_dir / name)
        for name in file_names
        if (model_dir / name).is_file()
    }
    tokenizer_dir = (args.tokenizer or model_dir).resolve()
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir, local_files_only=True, trust_remote_code=False
    )
    eos_ids = [
        tokenizer.encode(t, add_special_tokens=False)
        for t in ("<|im_end|>", "<|endoftext|>")
    ]
    if any(len(ids) != 1 for ids in eos_ids):
        raise ValueError("Expected single-token ChatML EOS markers")
    eos_ids = [ids[0] for ids in eos_ids]
    identity = {
        "engine": "transformers-cuda",
        "cases_hash": digest(cases),
        "model_hash": digest(hashes),
        "model_files": hashes,
        "tokenizer_hash": file_hash(tokenizer_dir / "tokenizer.json"),
        "tokenizer_files": {
            p.name: file_hash(p)
            for p in tokenizer_dir.iterdir()
            if p.name
            in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
        },
        "driver_hash": file_hash(Path(__file__)),
        "device": args.device,
        "device_build": torch.cuda.get_device_name(args.device),
        "runtime_libraries": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "expected": {
            "english": args.expected_english,
            "chinese": args.expected_chinese,
        },
        "config": {
            "decoder": "qwen3",
            "mode": "hf_kv",
            "seq_len": args.seq_len,
            "temperature": 0,
            "repetition_penalty": args.repetition_penalty,
            "repeat_last_n": args.repeat_last_n,
            "ignore_eos": False,
            "max_new_tokens": -1,
            "thinking_disabled": args.disable_thinking,
            "prompt_format": "chatml-v1",
            "history": (
                "incremental_kv"
                if args.disable_thinking
                else "incremental_kv_with_thinking"
            ),
            "dtype": args.dtype,
            "tf32": False,
            "attention": args.attention,
            "eos_ids": eos_ids,
            "use_cache": True,
            "seed": 42,
            "num_threads": args.num_threads,
        },
    }
    if root.exists():
        if not args.resume:
            raise ValueError(
                "Output already exists; choose a new directory or use --resume"
            )
        if read_json(root / "manifest.json")["identity"] != identity:
            raise ValueError("Resume model, script, questions or settings do not match")
        if digest(read_json(root / "cases.json")) != identity["cases_hash"]:
            raise ValueError("Saved cases were modified")
        if (root / "summary.json").is_file() and all(
            completed_session(root / "sessions" / s["id"], s) for s in cases["sessions"]
        ):
            from common.audit import audit_run

            audit_run(root)
            print("SKIP completed run", root)
            return
    else:
        root.mkdir(parents=True)
        write_json(
            root / "manifest.json",
            {
                "schema_version": 1,
                "identity": identity,
                "model_label": model_dir.name + "-transformers-" + args.dtype,
                "model_path": str(model_dir),
                "tokenizer_path": str(tokenizer_dir / "tokenizer.json"),
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "note": "Full input IDs accompany generate for indexing; existing KV is reused, not re-prefilled. No history thinking is stripped. GPU throughput is not a controlled comparison to device throughput.",
            },
        )
        write_json(root / "cases.json", cases)

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    started = time.perf_counter()
    model, loading = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        dtype=dtype,
        attn_implementation=args.attention,
        output_loading_info=True,
    )
    if any(
        loading.get(k)
        for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ):
        raise ValueError(f"Incomplete checkpoint loading: {loading}")
    model = model.to(args.device).eval()
    torch.cuda.synchronize(args.device)
    write_json(
        root / "model_loading.json",
        {
            "seconds": time.perf_counter() - started,
            "loading_info": {
                key: list(loading.get(key, []))
                for key in (
                    "missing_keys",
                    "unexpected_keys",
                    "mismatched_keys",
                    "error_msgs",
                )
            },
            "dtype": str(model.dtype),
            "device": str(model.device),
            "layer_types": model.config.layer_types,
            "sliding_window": model.config.sliding_window,
            "rope_parameters": model.config.rope_parameters,
            "checkpoint_use_cache": model.config.use_cache,
            "generation_use_cache": True,
        },
    )
    settings = GenerationConfig(
        do_sample=False,
        num_beams=1,
        temperature=1.0,
        top_p=1.0,
        top_k=50,
        repetition_penalty=1.0,
        use_cache=True,
        return_dict_in_generate=True,
        eos_token_id=eos_ids,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
    )
    print(f"Loaded {model_dir.name}: {model.dtype}, {model.device}", flush=True)
    print(
        f"Generation: temperature=0, repetition_penalty={args.repetition_penalty}, "
        f"repeat_last_n={args.repeat_last_n}, "
        f"thinking_disabled={args.disable_thinking}",
        flush=True,
    )
    for session in cases["sessions"]:
        directory = root / "sessions" / session["id"]
        if completed_session(directory, session):
            print("SKIP", session["id"], flush=True)
            continue
        if directory.exists():
            archive = root / "attempts" / (session["id"] + "-" + uuid.uuid4().hex[:12])
            archive.parent.mkdir(exist_ok=True)
            directory.rename(archive)
        directory.mkdir(parents=True)
        write_json(directory / "input.json", session)
        cache, history = DynamicCache(config=model.config), []
        with (directory / "session.log").open("w", encoding="utf-8") as log:
            for turn in session["turns"]:
                log.write(f"IDENTITY_TURN_BEGIN {turn['id']}\n")
                log.flush()
                record, cache, history = generate_turn(
                    model,
                    tokenizer,
                    cache,
                    history,
                    turn,
                    args.seq_len,
                    settings,
                    thinking_disabled=args.disable_thinking,
                    logits_processor=LogitsProcessorList([penalty]),
                )
                record["session_id"] = session["id"]
                log.write(
                    {
                        "eos": "Reached EOS; stopping generation\n",
                        "length_limit": "Generation stopped at seq_len limit\n",
                        "context_full": "sequence length exceeded\n",
                    }[record["stop_reason"]]
                )
                log.write(f"IDENTITY_TURN_END {turn['id']}\n")
                log.flush()
                write_json(directory / (turn["id"] + ".json"), record)
                (directory / (turn["id"] + ".txt")).write_text(
                    record["response"], encoding="utf-8"
                )
                print(
                    time.strftime("%H:%M:%S"),
                    turn["id"],
                    record["stop_reason"],
                    len(record["generated_ids"]),
                    "tokens",
                    flush=True,
                )
        write_json(
            directory / "complete.json",
            {
                "records": {
                    t["id"] + ".json": file_hash(directory / (t["id"] + ".json"))
                    for t in session["turns"]
                }
            },
        )
        del cache
    summary = summarize(root)
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "planned",
                    "attempted",
                    "eos",
                    "stop_reasons",
                    "name_mentions",
                )
            },
            ensure_ascii=False,
        )
    )
    print("Report:", root / "REPORT.md")


def main():
    """Parse reference-run options without enabling model remote code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument(
        "--tokenizer", type=Path, help="Optional shared tokenizer directory"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cases", type=Path, default=HERE / "cases.json")
    parser.add_argument("--expected-english", default="lu xiaoxi")
    parser.add_argument("--expected-chinese", default="陆小兮")
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--attention", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--repeat-last-n", type=int, default=128)
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Use enable_thinking=False in the model chat template; keep raw outputs and KV history unchanged.",
    )
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
