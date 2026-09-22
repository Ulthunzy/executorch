"""Conservative CUDA generation from an unmodified full-sequence QDQ export."""

import argparse
import importlib.metadata
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark import completed_session  # noqa: E402
from common.output import atomic_directory, run_lock  # noqa: E402
from hf_benchmark import chatml_prompt, WindowedRepetitionPenalty  # noqa: E402
from report import (
    digest,
    file_hash,
    read_json,
    summarize,
    validate_cases,
    write_json,
)  # noqa: E402


def generate_turn(engine, tokenizer, history, turn, args, eos):
    import torch

    payload = chatml_prompt(
        tokenizer, turn["prompt"], bool(history), args.disable_thinking
    )
    incremental = tokenizer.encode(payload, add_special_tokens=False)
    inputs = history + incremental
    remaining = args.seq_len - len(inputs)
    max_new_tokens = getattr(args, "max_new_tokens", -1)
    record = {
        **turn,
        "response": "",
        "prompt_payload": payload,
        "incremental_input_ids": incremental,
        "input_ids": inputs,
        "generated_ids": [],
        "error_code": 0,
        "stats": None,
        "has_history_before": bool(history),
        "has_history_after": bool(history),
    }
    if remaining <= 0:
        record.update(error_code=18, stop_reason="context_full", wall_seconds=0)
        return record, history
    remaining = remaining if max_new_tokens == -1 else min(remaining, max_new_tokens)
    generated = []
    started = time.perf_counter()
    first_token_seconds = None
    penalty = WindowedRepetitionPenalty(args.repetition_penalty, args.repeat_last_n)
    while len(generated) < remaining:
        prefix = inputs + generated
        scores = engine(prefix)[-1:].clone()
        prefix_tensor = torch.tensor([prefix], device=scores.device)
        token = int(penalty(prefix_tensor, scores).argmax(-1).item())
        if first_token_seconds is None:
            first_token_seconds = time.perf_counter() - started
        generated.append(token)
        if token in eos:
            break
    elapsed = time.perf_counter() - started
    record.update(
        response=tokenizer.decode(
            generated, skip_special_tokens=False, clean_up_tokenization_spaces=False
        ),
        generated_ids=generated,
        has_history_after=True,
        stop_reason="eos" if generated[-1] in eos else "length_limit",
        wall_seconds=elapsed,
        first_token_seconds=first_token_seconds,
        causality_check=engine.last_causality_check,
        execution="full_sequence_replay_on_cuda_not_incremental_kv",
    )
    return record, inputs + generated


def run(args, *, report_fn=summarize, evaluation=None):
    with run_lock(args.output):
        return run_locked(args, report_fn=report_fn, evaluation=evaluation)


def run_locked(args, *, report_fn=summarize, evaluation=None):
    import torch

    from engines.pt2 import PT2Engine
    from transformers import AutoTokenizer

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise ValueError("PT2 generation requires CUDA")
    if args.seq_len < 2:
        raise ValueError("--seq-len must be >= 2")
    max_new_tokens = getattr(args, "max_new_tokens", -1)
    if max_new_tokens == 0 or max_new_tokens < -1:
        raise ValueError("max-new-tokens must be -1 or positive")
    WindowedRepetitionPenalty(args.repetition_penalty, args.repeat_last_n)
    torch.manual_seed(42)
    torch.set_num_threads(args.num_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cases = read_json(args.cases)
    validate_cases(cases)
    if args.ids:
        if set(args.ids) - {s["id"] for s in cases["sessions"]}:
            raise ValueError("Unknown session IDs")
        cases["sessions"] = [s for s in cases["sessions"] if s["id"] in args.ids]
    if any(
        "formatted_prompt" in turn for s in cases["sessions"] for turn in s["turns"]
    ):
        raise ValueError("PT2 ChatML adapter does not accept preformatted prompts")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True, trust_remote_code=False
    )
    encoded_eos = [
        tokenizer.encode(marker, add_special_tokens=False)
        for marker in ("<|im_end|>", "<|endoftext|>")
    ]
    if any(len(ids) != 1 for ids in encoded_eos):
        raise ValueError("Expected single-token ChatML EOS markers")
    eos = {ids[0] for ids in encoded_eos}
    identity = {
        "engine": "pt2-full-sequence-cuda",
        "model_hash": file_hash(args.model),
        "tokenizer_hash": file_hash(args.tokenizer / "tokenizer.json"),
        "tokenizer_files": {
            p.name: file_hash(p)
            for p in args.tokenizer.iterdir()
            if p.suffix in {".json", ".jinja"}
        },
        "cases_hash": digest(cases),
        "driver_hash": file_hash(Path(__file__)),
        "adapter_hash": file_hash(ROOT / "engines/pt2.py"),
        "helper_hashes": {
            name: file_hash(ROOT / "benchmarks" / name)
            for name in ("hf_benchmark.py", "benchmark.py", "report.py")
        },
        "device": args.device,
        "device_name": torch.cuda.get_device_name(args.device),
        "runtime_libraries": {
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "tokenizers": importlib.metadata.version("tokenizers"),
        },
        "expected": (
            {}
            if evaluation is not None
            else {
                "english": args.expected_english,
                "chinese": args.expected_chinese,
            }
        ),
        "config": {
            "temperature": 0,
            "seq_len": args.seq_len,
            "repetition_penalty": args.repetition_penalty,
            "repeat_last_n": args.repeat_last_n,
            "ignore_eos": False,
            "max_new_tokens": max_new_tokens,
            "thinking_disabled": args.disable_thinking,
            "repair_think_eos": False,
            "enable_think_budget": False,
            "dtype": "float32",
            "tf32": False,
            "layout": args.pt2_layout,
            "graph_modified": False,
            "num_threads": args.num_threads,
        },
    }
    if evaluation is not None:
        identity["evaluation"] = evaluation
    root = args.output.resolve()
    if root.exists():
        if not args.resume or read_json(root / "manifest.json")["identity"] != identity:
            raise ValueError(
                "Existing output requires --resume with identical model, inputs and settings"
            )
        if digest(read_json(root / "cases.json")) != identity["cases_hash"]:
            raise ValueError("Saved cases changed")
    else:
        with atomic_directory(root) as working:
            write_json(working / "cases.json", cases)
            write_json(
                working / "manifest.json",
                {
                    "schema_version": 1,
                    "identity": identity,
                    "model_label": args.model.parent.name + "-pt2",
                    "model_path": str(args.model.resolve()),
                    "note": "Unmodified QDQ graph; full token-history replay, not production KV speed.",
                },
            )
    if (
        all(
            completed_session(root / "sessions" / s["id"], s) for s in cases["sessions"]
        )
        and (root / "summary.json").exists()
    ):
        from common.audit import audit_run

        audit_run(root)
        print("SKIP completed run", root)
        return
    engine = PT2Engine(
        args.model,
        device=args.device,
        layout=args.pt2_layout,
        pad_token_id=encoded_eos[1][0],
    )
    if args.seq_len > engine.max_input_length:
        raise ValueError("seq-len exceeds the exported full-sequence input shape")
    write_json(root / "input_specs.json", engine.input_specs)
    for session in cases["sessions"]:
        target = root / "sessions" / session["id"]
        if completed_session(target, session):
            print("SKIP", session["id"], flush=True)
            continue
        if target.exists():
            archive = root / "attempts" / (session["id"] + "-" + uuid.uuid4().hex)
            archive.parent.mkdir(exist_ok=True)
            target.rename(archive)
        target.mkdir(parents=True)
        write_json(target / "input.json", session)
        history = []
        with (target / "session.log").open("w", encoding="utf-8") as log:
            for turn in session["turns"]:
                log.write(f"IDENTITY_TURN_BEGIN {turn['id']}\n")
                record, history = generate_turn(
                    engine, tokenizer, history, turn, args, eos
                )
                record["session_id"] = session["id"]
                message = {
                    "eos": "Reached EOS; stopping generation",
                    "length_limit": "Generation stopped at seq_len limit",
                    "context_full": "sequence length exceeded",
                }[record["stop_reason"]]
                log.write(message + f"\nIDENTITY_TURN_END {turn['id']}\n")
                log.flush()
                write_json(target / (turn["id"] + ".json"), record)
                (target / (turn["id"] + ".txt")).write_text(
                    record["response"], encoding="utf-8"
                )
                print(
                    turn["id"],
                    record["stop_reason"],
                    len(record["generated_ids"]),
                    "tokens",
                    flush=True,
                )
        write_json(
            target / "complete.json",
            {
                "records": {
                    turn["id"] + ".json": file_hash(target / (turn["id"] + ".json"))
                    for turn in session["turns"]
                }
            },
        )
    report_fn(root)
    print(root / "REPORT.md")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument(
        "--pt2-layout", choices=("lincal-full-sequence",), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=ROOT / "benchmarks/cases.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=-1)
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--repeat-last-n", type=int, default=128)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--expected-english", default="Lu Xiaoxi")
    parser.add_argument("--expected-chinese", default="陆小兮")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
