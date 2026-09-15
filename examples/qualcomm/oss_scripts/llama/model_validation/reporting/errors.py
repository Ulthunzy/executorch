# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Write a session error table from existing records and explicitly bound reviews."""

import argparse
import csv
import html
import os
import re
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))
from common.output import atomic_directory  # noqa: E402
from report import (  # noqa: E402
    digest,
    file_hash,
    flags_for,
    grade,
    read_json,
    turn_logs,
    validate_cases,
    write_json,
)

STATUSES = {
    "reviewed_failed": "已复核未通过",
    "pending_review": "待复核",
    "reviewed_passed": "已复核通过",
}


def markdown_cell(value):
    text = html.escape(str(value), quote=False)
    text = re.sub(r"([\\`*_\[\]|])", r"\\\1", text)
    return text.replace("\r", "").replace("\n", "<br>")


def build_report(run_directory, output, reviews_path=None, references_path=None):
    """Validate source bindings and publish the completed report directory atomically."""
    root, output = Path(run_directory).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError(
            "Choose a new output directory; existing files are not overwritten"
        )
    sources = {}

    def source_hash(path):
        sources[str(path)] = file_hash(path) if path.is_file() else None

    source_hash(root / "manifest.json")
    source_hash(root / "cases.json")
    manifest = read_json(root / "manifest.json")
    cases = read_json(root / "cases.json")
    validate_cases(cases)
    identity = manifest["identity"]
    fingerprint = digest(identity)
    if digest(cases) != identity["cases_hash"]:
        raise ValueError("Saved question set no longer matches this run")
    ids = {turn["id"] for session in cases["sessions"] for turn in session["turns"]}
    reviews, references = {}, {}
    if reviews_path is not None:
        reviews_path = Path(reviews_path).resolve()
        source_hash(reviews_path)
        reviews = read_json(reviews_path)
        if (
            not isinstance(reviews, dict)
            or reviews.get("run_fingerprint") != fingerprint
        ):
            raise ValueError("Reviews belong to a different run/model/config")
    entries = reviews.get("reviews", {})
    if not isinstance(entries, dict) or entries.keys() - ids:
        raise ValueError("Review contains invalid or unknown turn IDs")
    if references_path is not None:
        references_path = Path(references_path).resolve()
        source_hash(references_path)
        references = read_json(references_path)
        if (
            not isinstance(references, dict)
            or references.keys() - ids
            or any(not isinstance(value, str) for value in references.values())
        ):
            raise ValueError("References must map known turn IDs to reference text")

    rows, turns = [], []
    max_turns = max(len(session["turns"]) for session in cases["sessions"])
    fields = [
        "session_id",
        "group",
        "status",
        "failed_turns",
        "pending_turns",
        "unattempted_turns",
    ]
    turn_fields = [
        "id",
        "input",
        "final_answer_summary",
        "reference",
        "review_status",
        "identity",
        "language_ok",
        "stop_reason",
        "auto_generation_ok",
        "review_note",
        "record_path",
        "text_path",
        "log_path",
    ]
    fields += [
        f"turn_{index}_{field}"
        for index in range(1, max_turns + 1)
        for field in turn_fields
    ]
    for session in cases["sessions"]:
        directory = root / "sessions" / session["id"]
        log_path = directory / "session.log"
        source_hash(log_path)
        logs = (
            turn_logs(log_path.read_text(encoding="utf-8", errors="replace"))
            if log_path.is_file()
            else {}
        )
        row = {"session_id": session["id"], "group": session.get("group", "")}
        outcomes = []
        attempted_count = 0
        for turn in session["turns"]:
            path = directory / (turn["id"] + ".json")
            text_path = path.with_suffix(".txt")
            source_hash(path)
            source_hash(text_path)
            attempted = path.is_file()
            attempted_count += attempted
            record = (
                read_json(path)
                if attempted
                else {
                    **turn,
                    "session_id": session["id"],
                    "response": "",
                    "error_code": None,
                    "infrastructure_error": "not_run",
                }
            )
            for key, value in (
                ("id", turn["id"]),
                ("prompt", turn["prompt"]),
                ("turn", turn["turn"]),
                ("session_id", session["id"]),
            ):
                if record.get(key) != value:
                    raise ValueError(f"Mismatched saved record: {turn['id']} / {key}")
            review = entries.get(turn["id"])
            if turn["id"] in entries:
                if not attempted:
                    raise ValueError(f"Cannot review an unattempted turn: {turn['id']}")
                if not isinstance(review, dict) or review.get("record_hash") != digest(
                    record
                ):
                    raise ValueError(f"Stale review for {turn['id']}; record changed")
                if not isinstance(review.get("note", ""), str):
                    raise ValueError(f"Review note must be text for {turn['id']}")
            flags = flags_for(
                record, logs.get(turn["id"], ""), identity.get("expected", {})
            )
            result = grade(record, flags, review)
            status = (
                "pending_review"
                if not result["reviewed"]
                else ("reviewed_passed" if result["normal"] else "reviewed_failed")
            )
            outcomes.append(status)
            turns.append({"status": status, "attempted": attempted})
            answer = flags["final_answer"]
            if len(answer) > 400:
                answer = answer[:400] + "…（摘要截断；见原始记录）"
            reference = references.get(turn["id"], "")
            values = {
                "id": turn["id"],
                "input": turn["prompt"],
                "final_answer_summary": answer or "（没有完整正式输出）",
                "reference": reference if reference.strip() else "未提供参考答案",
                "review_status": status,
                "identity": result["identity"],
                "language_ok": (review or {}).get("language_ok"),
                "stop_reason": flags["stop_reason"],
                "auto_generation_ok": result["auto_generation_ok"],
                "review_note": (review or {}).get("note", ""),
                "record_path": os.path.relpath(path, output) if attempted else "",
                "text_path": (
                    os.path.relpath(text_path, output) if text_path.is_file() else ""
                ),
                "log_path": (
                    os.path.relpath(log_path, output) if log_path.is_file() else ""
                ),
            }
            row.update(
                {f"turn_{turn['turn']}_{key}": value for key, value in values.items()}
            )
        row.update(
            status=next(status for status in STATUSES if status in outcomes),
            failed_turns=outcomes.count("reviewed_failed"),
            pending_turns=outcomes.count("pending_review"),
            unattempted_turns=len(outcomes) - attempted_count,
        )
        rows.append(row)
    rows.sort(key=lambda row: list(STATUSES).index(row["status"]))

    for filename, expected_hash in sources.items():
        path = Path(filename)
        if (file_hash(path) if path.is_file() else None) != expected_hash:
            raise ValueError(f"Source changed while preparing report: {path}")
    with atomic_directory(output) as working:
        with (working / "errors.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

        lines = [
            "# 按会话错误与待复核表",
            "",
            f"运行：{markdown_cell(manifest.get('model_label', root.name))}",
            "",
            "沿用原有 identity/language_ok 复核与生成结构检查；参考文本只展示，不用于自动判分。",
            "待复核（包括未运行轮次）不计为未通过；同一会话可同时有未通过和待复核轮次。",
            "正式输出只作至多 400 字符的摘录；完整原文与日志通过链接查看。",
            "",
            "[会话 CSV](errors.csv) · [来源与产物哈希](manifest.json)",
            "",
        ]
        headings = ["会话", "未通过轮数", "待复核轮数", "未运行轮数"]
        for index in range(1, max_turns + 1):
            headings += [
                f"第{index}轮输入",
                f"第{index}轮正式输出摘要",
                f"第{index}轮参考",
                f"第{index}轮复核",
            ]
        headings += ["原始记录"]
        for status, label in STATUSES.items():
            subset = [row for row in rows if row["status"] == status]
            lines += [f"## {label}会话（{len(subset)}）", ""]
            if not subset:
                lines += ["无。", ""]
                continue
            lines += ["| " + " | ".join(headings) + " |", "|" + "---|" * len(headings)]
            for row in subset:
                cells = [
                    row["session_id"],
                    row["failed_turns"],
                    row["pending_turns"],
                    row["unattempted_turns"],
                ]
                links = []
                for index in range(1, max_turns + 1):
                    prefix = f"turn_{index}_"
                    cells += [
                        row.get(prefix + field, "—")
                        for field in ("input", "final_answer_summary", "reference")
                    ]
                    review_status = row.get(prefix + "review_status")
                    cells += [
                        (
                            STATUSES[review_status]
                            + (
                                "；" + row[prefix + "review_note"]
                                if row[prefix + "review_note"]
                                else ""
                            )
                            if review_status
                            else "—"
                        )
                    ]
                    for field, title in (
                        ("record_path", "JSON"),
                        ("text_path", "原文"),
                    ):
                        if row.get(prefix + field):
                            url = quote(
                                row[prefix + field].replace(os.sep, "/"), safe="/"
                            )
                            links.append(f"[{row[prefix + 'id']} {title}]({url})")
                if row.get("turn_1_log_path"):
                    url = quote(row["turn_1_log_path"].replace(os.sep, "/"), safe="/")
                    links.append(f"[会话日志]({url})")
                lines.append(
                    "| "
                    + " | ".join(map(markdown_cell, cells))
                    + " | "
                    + (" · ".join(links) or "未运行")
                    + " |"
                )
            lines.append("")
        (working / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        report_manifest = {
            "schema_version": 1,
            "run_directory": str(root),
            "run_fingerprint": fingerprint,
            "source_hashes": sources,
            "reviews_hash": digest(reviews),
            "references_hash": digest(references),
            "script_hashes": {
                str(Path(__file__).resolve()): file_hash(Path(__file__)),
                str(ROOT / "common/output.py"): file_hash(ROOT / "common/output.py"),
                str(ROOT / "benchmarks/report.py"): file_hash(
                    ROOT / "benchmarks/report.py"
                ),
            },
            "session_counts": dict(Counter(row["status"] for row in rows)),
            "turn_counts": dict(Counter(turn["status"] for turn in turns)),
            "unattempted_turns": sum(not turn["attempted"] for turn in turns),
            "table_hash": digest(rows),
            "artifact_hashes": {
                name: file_hash(working / name) for name in ("REPORT.md", "errors.csv")
            },
        }
        write_json(working / "manifest.json", report_manifest)
    return report_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reviews", type=Path)
    parser.add_argument(
        "--references", type=Path, help="JSON mapping turn IDs to reference text"
    )
    args = parser.parse_args()
    try:
        build_report(args.run_directory, args.output, args.reviews, args.references)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(args.output.resolve() / "REPORT.md")


if __name__ == "__main__":
    main()
