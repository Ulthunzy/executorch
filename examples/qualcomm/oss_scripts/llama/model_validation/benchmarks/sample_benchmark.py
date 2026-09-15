# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run a local benchmark sample bundle on a ChatML QNN model."""

import argparse
import csv
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

from benchmark import add_device_arguments, resolve_model, run
from report import (
    digest,
    file_hash,
    flags_for,
    read_json,
    turn_logs,
    validate_cases,
    write_json,
)
from sample_scoring import score_answer

HERE = Path(__file__).resolve().parent
ROLES = {
    "HUMAN": "user",
    "BOT": "assistant",
    "SYSTEM": "system",
    "user": "user",
    "assistant": "assistant",
    "system": "system",
}


def render_messages(messages):
    """Preserve source system instructions and few-shot answers in ChatML."""
    if not isinstance(messages, list) or not messages:
        raise ValueError("Expected a nonempty messages list")
    rendered, roles, last_prompt = [], [], None
    for message in messages:
        role = ROLES.get(message.get("role"))
        content = message.get("prompt", message.get("content"))
        if role is None or not isinstance(content, str):
            raise ValueError("Unsupported message role or non-text content")
        roles.append(role)
        last_prompt = content
        rendered.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    if roles[-1] != "user" or not last_prompt.strip():
        raise ValueError("The final message must be a nonempty user question")
    return "".join(rendered) + "<|im_start|>assistant\n", last_prompt


def prepare_cases(
    bundle,
    suite,
    tokenizer,
    seq_len,
    reserve_tokens,
    benchmarks=None,
    ids=None,
    limit=None,
):
    """Read only sample inputs and references, screening length without truncation."""
    source = bundle / (
        "all_benchmarks_140.jsonl"
        if suite == "benchmarks"
        else "repeat_test_3000/selected_samples.jsonl"
    )
    if limit is not None and limit <= 0:
        raise ValueError("--limit must be positive")
    if reserve_tokens < 1 or reserve_tokens >= seq_len:
        raise ValueError("--reserve-tokens must be >=1 and smaller than --seq-len")
    selected, all_ids, all_benchmarks = [], set(), set()
    with source.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            sid = row["sample_id"] if suite == "benchmarks" else "R-" + str(row["id"])
            group = (
                row["benchmark"] if suite == "benchmarks" else "repeat/" + row["domain"]
            )
            if sid in all_ids or not re.fullmatch(r"[A-Za-z0-9_-]+", sid):
                raise ValueError(
                    f"Invalid/duplicate sample ID at line {line_number}: {sid}"
                )
            all_ids.add(sid)
            all_benchmarks.add(group)
            if (benchmarks and group not in benchmarks) or (ids and sid not in ids):
                continue
            if limit is not None and len(selected) >= limit:
                continue
            prompt, question = render_messages(row["messages"])
            selected.append(
                {
                    "id": sid,
                    "group": group,
                    "source_line": line_number,
                    "subset": row.get("subset", row.get("domain")),
                    "source_record_id": row.get("source_record_id", row.get("id")),
                    "turns": [
                        {
                            "id": sid + "T1",
                            "turn": 1,
                            "prompt": question,
                            "formatted_prompt": prompt,
                            "reference": row["reference"]
                            if suite == "benchmarks"
                            else None,
                        }
                    ],
                }
            )
    if benchmarks and set(benchmarks) - all_benchmarks:
        raise ValueError(f"Unknown benchmarks: {set(benchmarks) - all_benchmarks}")
    if ids and set(ids) - all_ids:
        raise ValueError(f"Unknown sample IDs: {set(ids) - all_ids}")
    if not selected:
        raise ValueError("No samples selected")
    for index, session in enumerate(selected, 1):
        prompt = session["turns"][0]["formatted_prompt"]
        count = len(tokenizer.encode(prompt, bos=0, eos=0))
        session["prompt_tokens_preflight"] = count
        if count + reserve_tokens > seq_len:
            session["skip_reason"] = "context_too_long"
        print(
            f"PREFLIGHT {index}/{len(selected)} {session['id']} "
            f"prompt_tokens={count} {session.get('skip_reason', 'ready')}",
            flush=True,
        )
    cases = {
        "schema_version": 1,
        "name": "sample-bundle-" + suite,
        "source_path": str(source.resolve()),
        "source_sha256": file_hash(source),
        "reserve_tokens": reserve_tokens,
        "sessions": selected,
    }
    validate_cases(cases)
    return cases


def sample_counts(rows):
    """Expose score coverage alongside correctness and generation failures."""
    scored = [row for row in rows if row["correct"] is not None]
    ready = [
        row for row in rows if row["stop_reason"] not in ("not_run", "context_too_long")
    ]
    reference_rows = [row for row in rows if not row["benchmark"].startswith("repeat/")]
    correct = sum(row["correct"] is True for row in rows)
    finals = [
        re.sub(r"\s+", " ", row["final_answer"]).strip()
        for row in ready
        if row["final_answer"]
    ]
    return {
        "planned": len(rows),
        "attempted": len(ready),
        "context_skipped": sum(
            row["stop_reason"] == "context_too_long" for row in rows
        ),
        "not_run": sum(row["stop_reason"] == "not_run" for row in rows),
        "scored": len(scored),
        "correct": correct,
        "pending_review": sum(
            row["correct"] is None
            and row["score_method"]
            in (
                "needs_review",
                "symbolic_review_required",
                "grader_unavailable",
                "unsupported_benchmark",
            )
            for row in reference_rows
        ),
        "accuracy_on_scored": correct / len(scored) if scored else None,
        "confirmed_correct_over_planned": correct / len(reference_rows)
        if reference_rows
        else None,
        "eos": sum(row["stop_reason"] == "eos" for row in rows),
        "repetition_flags": sum(row["repetition_flag"] for row in rows),
        "normal_generation": sum(row["normal_generation"] for row in rows),
        "duplicate_final_excess": len(finals) - len(set(finals)),
        "stop_reasons": dict(Counter(row["stop_reason"] for row in rows)),
        "score_methods": dict(Counter(row["score_method"] for row in rows)),
    }


def summarize_samples(root, reviews_path=None):
    """Write per-benchmark reports with reference-safe scoring and complete denominators."""
    manifest = read_json(root / "manifest.json")
    cases = read_json(root / "cases.json")
    if digest(cases) != manifest["identity"]["cases_hash"]:
        raise ValueError("Question snapshot changed; refusing to score")
    if manifest["identity"].get("evaluation", {}).get("kind") != "sample_bundle":
        raise ValueError("Not a sample-bundle run")
    fingerprint = digest(manifest["identity"])
    reviews = read_json(reviews_path) if reviews_path else {}
    if reviews and reviews.get("run_fingerprint") != fingerprint:
        raise ValueError("Reviews belong to another model/run")
    entries = reviews.get("reviews", {})
    expected_ids = {session["turns"][0]["id"] for session in cases["sessions"]}
    if entries.keys() - expected_ids:
        raise ValueError("Unknown review IDs")
    rows, template, source_hashes = [], {}, {}
    answers = [
        "# Benchmark 样本回答\n\n保留输入中的示例对话；只评分模型新生成的最终答复。\n"
    ]
    for session in cases["sessions"]:
        turn = session["turns"][0]
        folder = root / "sessions" / session["id"]
        path = folder / (turn["id"] + ".json")
        log_path = folder / "session.log"
        for source in (path, log_path):
            source_hashes[str(source.relative_to(root))] = (
                file_hash(source) if source.exists() else None
            )
        raw = (
            read_json(path)
            if path.exists()
            else {
                "id": turn["id"],
                "session_id": session["id"],
                "turn": 1,
                "prompt": turn["prompt"],
                "response": "",
                "error_code": None,
                "infrastructure_error": session.get("skip_reason", "not_run"),
            }
        )
        if (
            raw["id"] != turn["id"]
            or raw["prompt"] != turn["prompt"]
            or raw["session_id"] != session["id"]
            or raw["turn"] != 1
        ):
            raise ValueError(f"Mismatched result: {path}")
        if raw.get("has_history_before"):
            raise ValueError(f"Cross-sample KV contamination: {path}")
        log = log_path.read_text(errors="replace") if log_path.exists() else ""
        flags = flags_for(raw, turn_logs(log).get(turn["id"], ""), {})
        verdict = {
            "correct": None,
            "prediction": None,
            "target": None,
            "score_method": flags["stop_reason"],
        }
        if raw.get("error_code") == 0:
            verdict = score_answer(
                session["group"], turn["reference"], flags["final_answer"]
            )
        record_hash = digest(raw)
        review = entries.get(turn["id"])
        if review:
            if review.get("record_hash") != record_hash:
                raise ValueError(f"Stale review: {turn['id']}")
            if review.get("correct") is not None:
                if type(review["correct"]) is not bool or not path.exists():
                    raise ValueError(f"Invalid review: {turn['id']}")
                verdict.update(correct=review["correct"], score_method="manual_review")
        if path.exists() and turn["reference"] is not None:
            template[turn["id"]] = review or {
                "record_hash": record_hash,
                "correct": None,
                "note": "",
            }
        normal = (
            raw.get("error_code") == 0
            and flags["stop_reason"] == "eos"
            and flags["thinking_closed"]
            and bool(flags["final_answer"])
            and not flags["repetition_flag"]
            and not flags["garbled_flag"]
        )
        row = {
            "id": turn["id"],
            "sample_id": session["id"],
            "benchmark": session["group"],
            "subset": session["subset"],
            "source_line": session["source_line"],
            "prompt": turn["prompt"],
            "prompt_tokens_preflight": session["prompt_tokens_preflight"],
            "generated_tokens": (raw.get("stats") or {}).get("generated_tokens"),
            **flags,
            **verdict,
            "normal_generation": normal,
            "note": (review or {}).get("note", ""),
        }
        rows.append(row)
        target = verdict.get("target")
        status = (
            "正确"
            if row["correct"] is True
            else ("错误" if row["correct"] is False else "未评分")
        )
        answers.append(
            f"\n## {session['id']} — {status}\n\n"
            f"停止：{row['stop_reason']}；评分方式：{row['score_method']}。\n\n"
            f"问题（末条 user 消息）：\n\n"
            + "".join("> " + line + "\n" for line in turn["prompt"].splitlines())
            + f"\n参考答案：{target if target is not None else '见 cases.json 中 reference'}\n\n"
            + "".join(
                "> " + line + "\n"
                for line in (row["final_answer"] or "（没有完整最终答复）").splitlines()
            )
            + f"\n[原始输出（含 thinking）](sessions/{session['id']}/{turn['id']}.txt)"
            + f" · [完整输入](sessions/{session['id']}/input.json)\n"
        )
    summary = {
        "schema_version": 1,
        "model": manifest["model_label"],
        "run_fingerprint": fingerprint,
        "source_hashes": source_hashes,
        "review_hash": digest(reviews),
        "scoring_protocol": "conservative-local-v1",
        **sample_counts(rows),
        "benchmarks": {
            name: sample_counts([row for row in rows if row["benchmark"] == name])
            for name in dict.fromkeys(row["benchmark"] for row in rows)
        },
    }
    write_json(root / "summary.json", summary)
    write_json(root / "scored_results.json", rows)
    write_json(
        root / "review_template.json",
        {
            "run_fingerprint": fingerprint,
            "reviews": template,
        },
    )
    fields = [
        "id",
        "benchmark",
        "subset",
        "source_line",
        "prompt_tokens_preflight",
        "generated_tokens",
        "stop_reason",
        "correct",
        "prediction",
        "target",
        "score_method",
        "score_error",
        "normal_generation",
        "repetition_flag",
        "thinking_closed",
        "garbled_flag",
        "final_answer",
        "note",
    ]
    with (root / "results.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (root / "ANSWERS.md").write_text("".join(answers), encoding="utf-8")
    lines = [
        f"# {manifest['model_label']} 样本测试\n\n",
        f"计划 {summary['planned']}，已尝试 {summary['attempted']}，"
        f"上下文超长跳过 {summary['context_skipped']}，未运行 {summary['not_run']}。\n\n",
        f"已评分 {summary['scored']}，确认正确 {summary['correct']}，"
        f"待复核 {summary['pending_review']}，重复警报 {summary['repetition_flags']}。\n\n",
        "| 数据集 | 计划 | 已尝试 | 超长跳过 | 已评分 | 正确 | 待复核 |\n",
        "|---|---:|---:|---:|---:|---:|---:|\n",
    ]
    for name, count in summary["benchmarks"].items():
        lines.append(
            f"| {name} | {count['planned']} | {count['attempted']} | "
            f"{count['context_skipped']} | {count['scored']} | {count['correct']} | "
            f"{count['pending_review']} |\n"
        )
    lines.append(
        "\n采用本地保守评分，不是完整 OpenCompass 或官方榜单复现。"
        "选择题只取明确最终选项；数学数值精确比较，未证实的符号等价留待复核；"
        "IFEval 使用已安装 lm_eval 的 strict/loose 检查，缺少依赖单独标记。\n\n"
        "超长题没有被截断，也没有从计划分母删除。accuracy_on_scored 只针对已经评分的子集，"
        "需结合 scored/planned；confirmed_correct_over_planned 是已确认正确数/计划数的下界。\n\n"
        "默认保留 thinking、temperature=0，不能直接对比包内 BF16 随机采样成绩，"
        "或关闭 thinking、512-token 上限的旧 Q4_K_M 重复测试。重复警报使用当前工具的"
        "字符/句段启发式，不是旧脚本的 token-loop 指标。相同短选项不代表回答塌缩。\n\n"
        "[逐题回答](ANSWERS.md) · [CSV](results.csv) · [详细评分](scored_results.json)\n"
    )
    (root / "REPORT.md").write_text("".join(lines), encoding="utf-8")
    return summary


def ensure_driver(args):
    """Build only the dedicated bundle driver, leaving the identity runner intact."""
    if args.driver is not None:
        if not args.driver.is_file():
            raise ValueError(f"Driver not found: {args.driver}")
        return
    build_output = args.build_folder.resolve() / "benchmark_bundle"
    args.driver = build_output / "identity_eval_driver"
    marker = build_output / "driver_source.sha256"
    if (
        not args.driver.is_file()
        or not marker.exists()
        or marker.read_text().strip() != file_hash(HERE / "driver.cpp")
    ):
        subprocess.run(
            [
                sys.executable,
                str(HERE / "build_driver.py"),
                "--build-folder",
                str(args.build_folder.resolve()),
                "--output",
                str(build_output),
                "--jobs",
                "2",
            ],
            check=True,
        )


def main():
    """Run or score the 140 benchmark samples or 3000 repetition inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    add_device_arguments(run_parser)
    run_parser.set_defaults(driver=None)
    run_parser.add_argument("--bundle", type=Path, required=True)
    run_parser.add_argument(
        "--suite", choices=("benchmarks", "repeat"), default="benchmarks"
    )
    run_parser.add_argument("--benchmarks", nargs="+")
    run_parser.add_argument("--limit", type=int)
    run_parser.add_argument("--max-new-tokens", type=int, default=-1)
    run_parser.add_argument("--reserve-tokens", type=int, default=128)
    run_parser.add_argument("--build-folder", type=Path, default=Path("build-android"))
    run_parser.add_argument("--dry-run", action="store_true")
    report_parser = commands.add_parser("summarize")
    report_parser.add_argument("output", type=Path)
    report_parser.add_argument("--reviews", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "summarize":
            print(
                json.dumps(
                    summarize_samples(args.output, args.reviews),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return
        if args.max_new_tokens == 0 or args.max_new_tokens < -1:
            raise ValueError("--max-new-tokens must be -1 or positive")
        if args.output.exists() and not args.resume and not args.dry_run:
            raise ValueError("Output exists; choose a new directory or use --resume")
        from pytorch_tokenizers import CppHFTokenizer

        model = resolve_model(args.model)
        tokenizer_path = (args.tokenizer or model.parent / "tokenizer.json").resolve()
        tokenizer = CppHFTokenizer()
        tokenizer.load(str(tokenizer_path))
        cases = prepare_cases(
            args.bundle,
            args.suite,
            tokenizer,
            args.seq_len,
            args.reserve_tokens,
            args.benchmarks,
            args.ids,
            args.limit,
        )
        print(
            json.dumps(
                {
                    "planned": len(cases["sessions"]),
                    "context_skipped": sum(
                        bool(s.get("skip_reason")) for s in cases["sessions"]
                    ),
                    "benchmarks": dict(Counter(s["group"] for s in cases["sessions"])),
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        if args.dry_run:
            return
        ensure_driver(args)
        run(
            args,
            cases=cases,
            report_fn=summarize_samples,
            evaluation={
                "kind": "sample_bundle",
                "suite": args.suite,
                "source_sha256": cases["source_sha256"],
            },
            max_new_tokens=args.max_new_tokens,
        )
    except (
        ValueError,
        OSError,
        KeyError,
        ImportError,
        subprocess.SubprocessError,
    ) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
