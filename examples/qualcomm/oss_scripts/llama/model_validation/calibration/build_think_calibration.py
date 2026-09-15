# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Curate bounded, complete thinking conversations without training a model."""

import argparse
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
PUBLIC_REPO = "Congliu/Chinese-DeepSeek-R1-Distill-data-110k"
PUBLIC_REVISION = "8520b649430617c2be4490f424d251d09d835ed3"
BUCKETS = ((32, 128), (128, 512), (512, 1536), (1536, 2501))
SPECIAL = re.compile(r"<\|[^>]+\|>|</?think>")
FOREIGN_IDENTITY = re.compile(
    r"ChatGPT|DeepSeek|Qwen|通义千问|深度求索|小度|豆包|腾讯混元|文心一言", re.I
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def thinking_parts(content):
    match = re.fullmatch(r"<think>\s*(.*?)\s*</think>\s*(.+)", content, re.S)
    if not match or not match[1].strip() or not match[2].strip():
        raise ValueError("missing_thinking_or_final")
    if SPECIAL.search(match[1]) or SPECIAL.search(match[2]):
        raise ValueError("nested_or_foreign_special_tokens")
    return match[1], match[2]


def has_repetition(text):
    sentences = [
        s.strip() for s in re.split(r"[。！？!?\n]", text) if len(s.strip()) >= 12
    ]
    return max(Counter(sentences).values(), default=0) >= 4 or bool(
        re.search(r"(.{12,160}?)\1{3,}", text, re.S)
    )


def format_chatml(messages):
    if not messages or len(messages) % 2:
        raise ValueError("expected_complete_user_assistant_pairs")
    chunks = []
    for index, message in enumerate(messages):
        role = "user" if index % 2 == 0 else "assistant"
        content = message["content"]
        if (
            message["role"] != role
            or not isinstance(content, str)
            or not content.strip()
        ):
            raise ValueError("invalid_message")
        if role == "assistant":
            thinking_parts(content)
        elif SPECIAL.search(content):
            raise ValueError("special_tokens_in_question")
        if "\ufffd" in content or has_repetition(content):
            raise ValueError("garbled_or_repetitive")
        chunks.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    return "\n".join(chunks)


def measure(messages, tokenizer):
    text = format_chatml(messages)
    return {
        "total_tokens": len(tokenizer.encode(text, add_special_tokens=False)),
        "think_tokens": [
            len(
                tokenizer.encode(
                    thinking_parts(m["content"])[0], add_special_tokens=False
                )
            )
            for m in messages
            if m["role"] == "assistant"
        ],
        "final_tokens": [
            len(
                tokenizer.encode(
                    thinking_parts(m["content"])[1], add_special_tokens=False
                )
            )
            for m in messages
            if m["role"] == "assistant"
        ],
    }


def question_key(text):
    return re.sub(r"\W", "", text).casefold()


def excluded_questions(paths):
    return {
        question_key(turn["prompt"])
        for path in paths
        for session in read_json(path)["sessions"]
        for turn in session["turns"]
    }


def public_candidate(row, index, tokenizer, max_context_len, min_score):
    if (
        not isinstance(row.get("score"), (int, float))
        or not math.isfinite(row["score"])
        or row["score"] < min_score
    ):
        raise ValueError("low_score")
    fields = [row.get(k) for k in ("input", "reasoning_content", "content")]
    if any(not isinstance(s, str) or not s.strip() for s in fields):
        raise ValueError("missing_field")
    if any(SPECIAL.search(s) for s in fields):
        raise ValueError("source_special_tokens")
    question, reasoning, answer = fields
    if not re.search(r"[\u4e00-\u9fff]", question) or not re.search(
        r"[\u4e00-\u9fff]", reasoning
    ):
        raise ValueError("not_chinese")
    if FOREIGN_IDENTITY.search("\n".join(fields)):
        raise ValueError("foreign_model_reference")
    if len(tokenizer.encode(question, add_special_tokens=False)) > 256:
        raise ValueError("long_question")
    messages = [
        {"role": "user", "content": question.strip()},
        {
            "role": "assistant",
            "content": f"<think>\n{reasoning.strip()}\n</think>\n\n{answer.strip()}",
        },
    ]
    sizes = measure(messages, tokenizer)
    if sizes["total_tokens"] > max_context_len or sizes["final_tokens"][0] > 768:
        raise ValueError("long_conversation_or_final")
    bucket = next(
        (
            i
            for i, (low, high) in enumerate(BUCKETS)
            if low <= sizes["think_tokens"][0] < high
        ),
        None,
    )
    if bucket is None:
        raise ValueError("think_outside_buckets")
    source = row.get("repo_name", "unknown")
    general = source.startswith(("coig/", "zhihu/", "COIG", "m-a-p/"))
    return {
        "id": f"R1-{index:06d}",
        "messages": messages,
        **sizes,
        "bucket": bucket,
        "general": general,
        "provenance": {
            "kind": "public_r1",
            "repo": PUBLIC_REPO,
            "revision": PUBLIC_REVISION,
            "row_index": index,
            "row_hash": digest(row),
            "upstream_source": source,
            "score": row["score"],
            "dataset_card_license": "apache-2.0",
        },
    }


def public(args, tokenizer):
    import requests

    if args.output.exists():
        raise ValueError(
            "Choose a new output directory; source snapshots are immutable"
        )
    args.output.mkdir(parents=True)
    url = f"https://huggingface.co/datasets/{PUBLIC_REPO}/resolve/{PUBLIC_REVISION}/distill_r1_110k.jsonl"
    excluded = excluded_questions(args.exclude_cases)
    seen = set(excluded)
    pools = [[] for _ in BUCKETS]
    rejected = Counter()
    scanned = 0
    # The source is JSONL: stream only the bounded prefix, not the whole repository.
    with (
        args.source_jsonl.open("rb")
        if args.source_jsonl
        else requests.get(url, stream=True, timeout=(20, 45))
    ) as response:
        if not args.source_jsonl:
            response.raise_for_status()
        lines = response if args.source_jsonl else response.iter_lines(chunk_size=65536)
        with (args.output / "source-prefix.jsonl").open("wb") as cache:
            for line in lines:
                line = line.rstrip(b"\r\n")
                if not line:
                    continue
                cache.write(line + b"\n")
                row = json.loads(line)
                index = scanned
                scanned += 1
                try:
                    candidate = public_candidate(
                        row, index, tokenizer, args.max_context_len, args.min_score
                    )
                    key = question_key(candidate["messages"][0]["content"])
                    if key in seen:
                        raise ValueError("duplicate_or_heldout_question")
                    seen.add(key)
                    pools[candidate["bucket"]].append(candidate)
                except ValueError as error:
                    rejected[str(error)] += 1
                if scanned % 500 == 0:
                    print(
                        "Scanned",
                        scanned,
                        "eligible buckets",
                        list(map(len, pools)),
                        flush=True,
                    )
                if scanned >= args.max_scanned:
                    break
    available = list(map(len, pools))
    write_json(
        args.output / "scan.json",
        {"scanned": scanned, "available": available, "rejected": dict(rejected)},
    )
    if any(len(pool) < count for pool, count in zip(pools, args.bucket_counts)):
        raise ValueError(
            f"Insufficient eligible samples: {available}, requested {args.bucket_counts}; no duplication or truncation used"
        )
    rng = random.Random(args.seed)
    selected = []
    for pool, count in zip(pools, args.bucket_counts):
        rng.shuffle(pool)
        general = [r for r in pool if r["general"]]
        others = [r for r in pool if not r["general"]]
        chosen = general[: round(count * 0.7)]
        chosen += others[: count - len(chosen)]
        chosen_ids = {r["id"] for r in chosen}
        chosen += [r for r in pool if r["id"] not in chosen_ids][: count - len(chosen)]
        selected.extend(chosen)
    if sum(r["general"] for r in selected) < len(selected) * 0.6:
        raise ValueError(
            "Fewer than 60% general-domain samples; inspect source coverage"
        )
    rng.shuffle(selected)
    write_json(args.output / "selected.json", selected)
    manifest = {
        "source_url": url,
        "source_revision": PUBLIC_REVISION,
        "source_prefix_sha256": file_hash(args.output / "source-prefix.jsonl"),
        "replayed_source_jsonl": str(args.source_jsonl) if args.source_jsonl else None,
        "selected_sha256": file_hash(args.output / "selected.json"),
        "tokenizer_sha256": file_hash(args.tokenizer / "tokenizer.json"),
        "script_sha256": file_hash(Path(__file__)),
        "seed": args.seed,
        "max_context_len": args.max_context_len,
        "min_score": args.min_score,
        "bucket_counts": args.bucket_counts,
        "scanned": scanned,
        "excluded_cases": {str(p): file_hash(p) for p in args.exclude_cases},
        "review": "Automatic structural/length/repetition screening plus upstream model scores, not an independent correctness guarantee.",
    }
    write_json(args.output / "manifest.json", manifest)
    print("Selected", len(selected), "public conversations", flush=True)


def gpu_candidates(root, tokenizer, max_context_len):
    manifest = read_json(root / "manifest.json")
    cases = read_json(root / "cases.json")
    if digest(cases) != manifest["identity"]["cases_hash"]:
        raise ValueError("GPU question set changed")
    config = manifest["identity"]["config"]
    if (
        manifest["identity"]["engine"] != "transformers-cuda"
        or config["thinking_disabled"]
        or config["ignore_eos"]
    ):
        raise ValueError("Expected CUDA reference with thinking and EOS enabled")
    rows = []
    for session in cases["sessions"]:
        messages, hashes, finals, errors = [], {}, [], []
        directory = root / "sessions" / session["id"]
        completion = read_json(directory / "complete.json")
        for number, turn in enumerate(session["turns"]):
            path = directory / (turn["id"] + ".json")
            hashes[turn["id"]] = file_hash(path)
            if completion["records"][path.name] != hashes[turn["id"]]:
                raise ValueError(f"GPU record changed: {path}")
            record = read_json(path)
            try:
                if (
                    record["prompt"] != turn["prompt"]
                    or record["stop_reason"] != "eos"
                    or record["error_code"]
                ):
                    raise ValueError("failed_generation")
                if record["has_history_before"] != bool(number):
                    raise ValueError("incorrect_history_state")
                raw = record["response"]
                if not raw.endswith("<|im_end|>"):
                    raise ValueError("unexpected_eos")
                content = raw[: -len("<|im_end|>")]
                _, final = thinking_parts(content)
                finals.append(final)
                if FOREIGN_IDENTITY.search(content):
                    raise ValueError("foreign_model_reference")
                if not re.search(turn["expected_answer_regex"], final, re.S):
                    raise ValueError("expected_answer_not_found")
                messages.extend(
                    [
                        {"role": "user", "content": turn["prompt"]},
                        {"role": "assistant", "content": content},
                    ]
                )
            except ValueError as error:
                errors.append(f"{turn['id']}: {error}")
        sizes = {}
        if not errors:
            try:
                sizes = measure(messages, tokenizer)
                if sizes["total_tokens"] > max_context_len:
                    raise ValueError("long_conversation")
            except ValueError as error:
                errors.append(str(error))
        rows.append(
            {
                "id": session["id"],
                "group": session["group"],
                "messages": messages,
                "final_answers": finals,
                "errors": errors,
                "record_hashes": hashes,
                **sizes,
                "provenance": {
                    "kind": "luxi_gpu",
                    "run_fingerprint": digest(manifest["identity"]),
                    "session_id": session["id"],
                },
            }
        )
    return rows


def inspect_gpu(args, tokenizer):
    rows = [
        row
        for root in args.gpu_run
        for row in gpu_candidates(root, tokenizer, args.max_context_len)
    ]
    if args.output.exists():
        raise ValueError("Choose a new review snapshot filename")
    write_json(args.output, rows)
    for row in rows:
        print(
            row["id"],
            row["errors"] or "structural checks passed",
            json.dumps(row["final_answers"], ensure_ascii=False),
        )


def merge(args, tokenizer):
    if args.output.exists():
        raise ValueError("Choose a new output directory")
    public_manifest = read_json(args.public / "manifest.json")
    if public_manifest["selected_sha256"] != file_hash(args.public / "selected.json"):
        raise ValueError("Public selection changed")
    tokenizer_hash = file_hash(args.tokenizer / "tokenizer.json")
    if public_manifest["tokenizer_sha256"] != tokenizer_hash:
        raise ValueError("Public selection used a different tokenizer")
    gpu_manifests = [read_json(root / "manifest.json") for root in args.gpu_run]
    if any(
        manifest["identity"]["tokenizer_hash"] != tokenizer_hash
        for manifest in gpu_manifests
    ):
        raise ValueError("GPU generation used a different tokenizer")
    reviews = read_json(args.gpu_reviews)
    if set(reviews["run_fingerprints"]) != {
        digest(manifest["identity"]) for manifest in gpu_manifests
    }:
        raise ValueError("Reviews belong to a different GPU run")
    public_rows = read_json(args.public / "selected.json")
    gpu_rows = [
        row
        for root in args.gpu_run
        for row in gpu_candidates(root, tokenizer, args.max_context_len)
    ]
    if len({row["id"] for row in gpu_rows}) != len(gpu_rows):
        raise ValueError("Duplicate GPU session IDs across runs")
    if reviews["reviews"].keys() - {row["id"] for row in gpu_rows}:
        raise ValueError("Review contains unknown GPU session IDs")
    accepted = []
    for row in gpu_rows:
        review = reviews["reviews"].get(row["id"])
        if not review or review.get("accept") is not True:
            continue
        if review["record_hashes"] != row["record_hashes"]:
            raise ValueError(f"Stale review: {row['id']}")
        if row["errors"]:
            raise ValueError(
                f"Review cannot override structural failure: {row['id']}: {row['errors']}"
            )
        accepted.append(row)
    if len(accepted) != args.gpu_count:
        raise ValueError(
            f"Expected {args.gpu_count} explicitly reviewed GPU sessions, got {len(accepted)}"
        )
    rows = public_rows + accepted
    excluded = excluded_questions(args.exclude_cases)
    seen = set()
    texts = []
    for row in rows:
        sizes = measure(row["messages"], tokenizer)
        if sizes["total_tokens"] > args.max_context_len:
            raise ValueError(f"Conversation too long: {row['id']}")
        for message in row["messages"]:
            if (
                message["role"] == "user"
                and question_key(message["content"]) in excluded
            ):
                raise ValueError(f"Held-out question included: {row['id']}")
        key = question_key(row["messages"][0]["content"])
        if key in seen:
            raise ValueError(f"Duplicate first question: {row['id']}")
        seen.add(key)
        row.update(sizes)
        texts.append({"id": row["id"], "text": format_chatml(row["messages"])})
    args.output.mkdir(parents=True)
    write_json(
        args.output / "messages.json", [{"messages": row["messages"]} for row in rows]
    )
    with (args.output / "calibration.chatml.jsonl").open(
        "w", encoding="utf-8"
    ) as stream:
        for row in texts:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(
        args.output / "provenance.json",
        [
            {k: v for k, v in row.items() if k not in ("messages", "final_answers")}
            for row in rows
        ],
    )
    write_json(args.output / "gpu_reviews.json", reviews)
    write_json(args.output / "public_manifest.json", public_manifest)
    write_json(args.output / "gpu_manifests.json", gpu_manifests)
    thoughts = [n for row in rows for n in row["think_tokens"]]
    summary = {
        "conversations": len(rows),
        "assistant_turns": len(thoughts),
        "public_conversations": len(public_rows),
        "gpu_multiturn_conversations": len(accepted),
        "total_tokens": sum(row["total_tokens"] for row in rows),
        "max_conversation_tokens": max(row["total_tokens"] for row in rows),
        "think_tokens": {
            "min": min(thoughts),
            "median": statistics.median(thoughts),
            "mean": statistics.mean(thoughts),
            "max": max(thoughts),
        },
        "think_length_buckets": {
            f"{low}-{high - 1}": sum(low <= n < high for n in thoughts)
            for low, high in ((0, 32), *BUCKETS)
        },
        "upstream_sources": dict(
            Counter(
                row["provenance"].get("upstream_source", "luxi_gpu") for row in rows
            )
        ),
    }
    write_json(args.output / "summary.json", summary)
    write_json(
        args.output / "manifest.json",
        {
            "schema_version": 1,
            "tokenizer_sha256": tokenizer_hash,
            "script_sha256": file_hash(Path(__file__)),
            "max_context_len": args.max_context_len,
            "source_files": {
                p.name: file_hash(p)
                for p in sorted(args.output.iterdir())
                if p.is_file()
            },
            "excluded_cases": {str(p): file_hash(p) for p in args.exclude_cases},
            "limitations": "PTQ input, not SFT. Public rows have model-based scores plus automatic checks; GPU rows additionally have explicit semantic review. Exact question exclusion does not prove semantic non-overlap. No PTE exported or accuracy improvement established.",
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("public", "inspect-gpu", "merge"))
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-context-len", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-score", type=int, default=9)
    parser.add_argument("--max-scanned", type=int, default=8000)
    parser.add_argument("--bucket-counts", nargs=4, type=int, default=[0, 128, 112, 16])
    parser.add_argument(
        "--source-jsonl",
        type=Path,
        help="Replay a saved source-prefix.jsonl without downloading again",
    )
    parser.add_argument("--gpu-run", nargs="+", type=Path)
    parser.add_argument("--gpu-reviews", type=Path)
    parser.add_argument("--gpu-count", type=int, default=32)
    parser.add_argument("--public", type=Path)
    parser.add_argument(
        "--exclude-cases",
        nargs="*",
        type=Path,
        default=[HERE.parent / "benchmarks/cases.json"],
    )
    args = parser.parse_args()
    if (
        args.max_context_len < 2
        or args.max_scanned < 1
        or min(args.bucket_counts) < 0
        or not sum(args.bucket_counts)
    ):
        parser.error("Invalid context length, scan limit or bucket counts")
    if args.command in ("inspect-gpu", "merge") and not args.gpu_run:
        parser.error("--gpu-run is required")
    if args.command == "merge" and (not args.public or not args.gpu_reviews):
        parser.error("merge requires --public and --gpu-reviews")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True, trust_remote_code=False
    )
    {"public": public, "inspect-gpu": inspect_gpu, "merge": merge}[args.command](
        args, tokenizer
    )


if __name__ == "__main__":
    main()
