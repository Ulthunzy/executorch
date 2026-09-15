# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Report generation without importing PyTorch or requiring a device."""

import csv
import hashlib
import json
import re
import statistics
import unicodedata
from collections import Counter


def read_json(path):
    """Read UTF-8 benchmark artifacts."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    """Replace a generated JSON artifact atomically."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def digest(value):
    """Fingerprint structured values independently of dictionary key order."""
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def file_hash(path):
    """Hash large model files without loading them into memory."""
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_cases(cases):
    """Check question IDs, turn ordering, and supported name requirements."""
    if cases.get("schema_version") != 1 or not cases.get("sessions"):
        raise ValueError("Expected schema_version=1 and nonempty sessions")
    sessions, turns = set(), set()
    for session in cases["sessions"]:
        sid = session["id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", sid) or sid in sessions:
            raise ValueError(f"Invalid/duplicate session ID: {sid}")
        sessions.add(sid)
        if not session["turns"]:
            raise ValueError(f"Empty session: {sid}")
        for index, turn in enumerate(session["turns"], 1):
            tid = turn["id"]
            if not re.fullmatch(r"[A-Za-z0-9_-]+", tid) or tid in turns:
                raise ValueError(f"Invalid/duplicate turn ID: {tid}")
            turns.add(tid)
            if turn["turn"] != index or not turn["prompt"].strip():
                raise ValueError(f"Invalid turn: {tid}")
            if turn.get("required_name_language", "either") not in (
                "either",
                "chinese",
                "english",
            ):
                raise ValueError(f"Invalid name language: {tid}")
            if "formatted_prompt" in turn and not isinstance(
                turn["formatted_prompt"], str
            ):
                raise ValueError(f"Invalid formatted prompt: {tid}")


def flags_for(record, log, expected):
    """Extract final answers and screening flags, not semantic judgments."""
    text = record.get("response", "")
    opens, closes = text.count("<think>"), text.count("</think>")
    final = text.rsplit("</think>", 1)[-1] if closes else ("" if opens else text)
    final = final.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
    tokens = re.findall(r"</?think>", text)
    balanced = True
    depth = 0
    for token in tokens:
        depth += 1 if token == "<think>" else -1
        balanced = balanced and depth in (0, 1)
    balanced = balanced and depth == 0
    english = unicodedata.normalize("NFKC", final).casefold()
    letters = re.sub(r"[\s-]+", "", expected.get("english", "").casefold())
    pattern = r"(?<![a-z])" + r"[\s-]*".join(map(re.escape, letters)) + r"(?![a-z])"
    sentences = [
        part.strip()
        for part in re.split(r"[。！？!?\n]", text)
        if len(part.strip()) >= 12
    ]
    repeated = max(Counter(sentences).values(), default=0)
    tandem = re.search(r"(.{12,160}?)\1{3,}", text, flags=re.DOTALL)
    reason = "unknown"
    if record.get("infrastructure_error"):
        reason = record["infrastructure_error"]
    elif record.get("error_code", 0):
        reason = (
            "context_full"
            if "sequence length exceeded" in log
            or "Prompt exceeds the compiled prefill capacity" in log
            else "runtime_error"
        )
    elif text.rstrip().endswith(("<|im_end|>", "<|endoftext|>")) or re.search(
        r"Reached EOS(?: after prefill)?; stopping generation", log
    ):
        reason = "eos"
    elif "Generation stopped at seq_len limit" in log:
        reason = "length_limit"
    return {
        "stop_reason": reason,
        "final_answer": final,
        "thinking_closed": balanced,
        "think_open": opens,
        "think_close": closes,
        "english_name_mentioned": bool(letters and re.search(pattern, english)),
        "chinese_name_mentioned": bool(
            expected.get("chinese") and expected["chinese"] in final
        ),
        "repetition_flag": repeated >= 4 or tandem is not None,
        "garbled_flag": "\ufffd" in text,
    }


def turn_logs(log):
    """Separate native turn logs to avoid attributing another turn's EOS."""
    result, active = {}, None
    for line in log.splitlines(keepends=True):
        if line.startswith(("IDENTITY_INPUT ", "PROMPT | ")) or "[QNN_PROMPT] " in line:
            continue
        if line.startswith(("IDENTITY_TURN_BEGIN ", "BATCH_TURN_BEGIN ")):
            active = line.split()[1]
            result[active] = ""
        if active:
            result[active] += line
        if line.startswith("IDENTITY_TURN_END "):
            active = None
    return result


def grade(record, flags, review):
    """Combine explicit semantic review with structural generation checks."""
    reviewed = review is not None and review.get("identity") is not None
    if reviewed:
        if (
            review["identity"]
            not in ("correct", "missing", "wrong", "partial", "variant", "not_answered")
            or type(review.get("language_ok")) is not bool
        ):
            raise ValueError(f"Invalid review for {record['id']}")
        for field in ("other_issue", "repetition_override", "garbled_override"):
            if field in review and type(review[field]) is not bool:
                raise ValueError(f"{field} must be boolean for {record['id']}")
    review = review or {}
    identity = review.get("identity") if reviewed else None
    generation_ok = (
        record.get("error_code") == 0
        and flags["stop_reason"] == "eos"
        and flags["thinking_closed"]
        and bool(flags["final_answer"])
        and not review.get("repetition_override", flags["repetition_flag"])
        and not review.get("garbled_override", flags["garbled_flag"])
        and not review.get("other_issue", False)
    )
    return {
        "reviewed": reviewed,
        "identity": identity or "pending",
        "identity_correct": identity == "correct" if reviewed else None,
        "normal": (
            generation_ok and identity == "correct" and review["language_ok"]
            if reviewed
            else None
        ),
        "auto_generation_ok": generation_ok,
    }


def counts(rows):
    """Count every planned answer, withholding rates until fully reviewed."""
    reviewed = sum(row["reviewed"] for row in rows)
    normal = sum(row["normal"] is True for row in rows)
    identity = sum(row["identity_correct"] is True for row in rows)
    complete = bool(rows) and reviewed == len(rows)
    return {
        "planned": len(rows),
        "attempted": sum(row["stop_reason"] != "not_run" for row in rows),
        "reviewed": reviewed,
        "pending_review": len(rows) - reviewed,
        "normal": normal,
        "identity_correct": identity,
        "normal_rate": normal / len(rows) if complete else None,
        "identity_correct_rate": identity / len(rows) if complete else None,
        "eos": sum(row["stop_reason"] == "eos" for row in rows),
        "repetition_flags": sum(row["repetition_flag"] for row in rows),
        "name_mentions": sum(
            row["english_name_mentioned"] or row["chinese_name_mentioned"]
            for row in rows
        ),
        "stop_reasons": dict(Counter(row["stop_reason"] for row in rows)),
    }


def summarize(root, reviews_path=None):
    """Write reviewable reports without modifying raw inference records."""
    manifest = read_json(root / "manifest.json")
    fingerprint = digest(manifest["identity"])
    cases = read_json(root / "cases.json")
    validate_cases(cases)
    if digest(cases) != manifest["identity"]["cases_hash"]:
        raise ValueError("Saved question set no longer matches this run")
    reviews = read_json(reviews_path) if reviews_path else {}
    if reviews_path is None and (root / "review_template.json").is_file():
        reviews = read_json(root / "review_template.json")
        reviews["reviews"] = {
            key: entry
            for key, entry in reviews["reviews"].items()
            if entry.get("identity")
        }
    if reviews and reviews.get("run_fingerprint") != fingerprint:
        raise ValueError("Reviews belong to a different run/model/config")
    entries = reviews.get("reviews", {})
    expected_ids = {
        turn["id"] for session in cases["sessions"] for turn in session["turns"]
    }
    if entries.keys() - expected_ids:
        raise ValueError("Review contains unknown turn IDs")
    rows, template, source_hashes = [], {}, {}
    for session in cases["sessions"]:
        log_path = root / "sessions" / session["id"] / "session.log"
        source_hashes[str(log_path.relative_to(root))] = (
            file_hash(log_path) if log_path.exists() else None
        )
        logs = (
            turn_logs(log_path.read_text(errors="replace")) if log_path.exists() else {}
        )
        for turn in session["turns"]:
            path = root / "sessions" / session["id"] / (turn["id"] + ".json")
            source_hashes[str(path.relative_to(root))] = (
                file_hash(path) if path.exists() else None
            )
            record = (
                read_json(path)
                if path.exists()
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
            record_hash = digest(record)
            review = entries.get(turn["id"])
            if review and review.get("record_hash") != record_hash:
                raise ValueError(f"Stale review for {turn['id']}; response changed")
            if review and review.get("identity") and not path.exists():
                raise ValueError(f"Cannot review an unattempted turn: {turn['id']}")
            flags = flags_for(
                record, logs.get(turn["id"], ""), manifest["identity"]["expected"]
            )
            row = {
                **record,
                **turn,
                **flags,
                **grade(record, flags, review),
                "group": session["group"],
                "review": review or {},
            }
            rows.append(row)
            if path.exists():
                template[turn["id"]] = review or {
                    "record_hash": record_hash,
                    "identity": None,
                    "language_ok": None,
                    "note": "",
                }
    summary = {
        "schema_version": 1,
        "model": manifest["model_label"],
        "run_fingerprint": fingerprint,
        "source_hashes": source_hashes,
        "review_hash": digest(reviews),
        **counts(rows),
        "identity_outcomes": dict(Counter(row["identity"] for row in rows)),
        "first_turn": counts([row for row in rows if row["turn"] == 1]),
        "followup_turns": counts([row for row in rows if row["turn"] > 1]),
        "groups": {
            group: counts([row for row in rows if row["group"] == group])
            for group in dict.fromkeys(
                session["group"] for session in cases["sessions"]
            )
        },
        "both_turns_normal": sum(
            all(
                row["normal"] is True
                for row in rows
                if row["session_id"] == session["id"]
            )
            for session in cases["sessions"]
        ),
    }
    speeds = [
        row["stats"]["decode_token_per_sec"]
        for row in rows
        if (row.get("stats") or {}).get("decode_token_per_sec", 0) > 0
    ]
    summary["decode_token_per_sec_median"] = (
        statistics.median(speeds) if speeds else None
    )
    write_json(root / "summary.json", summary)
    write_json(
        root / "review_template.json",
        {"run_fingerprint": fingerprint, "reviews": template},
    )
    fields = [
        "id",
        "session_id",
        "group",
        "turn",
        "prompt",
        "stop_reason",
        "reviewed",
        "identity",
        "identity_correct",
        "normal",
        "auto_generation_ok",
        "english_name_mentioned",
        "chinese_name_mentioned",
        "repetition_flag",
        "thinking_closed",
        "generated_tokens",
        "final_answer",
        "note",
    ]
    with (root / "results.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "generated_tokens": (row.get("stats") or {}).get(
                        "generated_tokens"
                    ),
                    "note": row["review"].get("note", ""),
                }
            )
    answers = ["# 自我认知逐题回答\n\n仅展示最终答复；原始文件保留 thinking。\n"]
    for row in rows:
        verdict = (
            "待复核" if not row["reviewed"] else ("通过" if row["normal"] else "未通过")
        )
        answers.append(
            f"\n## {row['id']} — {verdict}\n\n问题：{row['prompt']}\n\n"
            f"停止：{row['stop_reason']}\n\n"
            + "".join(
                "> " + line + "\n"
                for line in (row["final_answer"] or "（没有完整最终答复）").splitlines()
            )
            + f"\n[完整输出](sessions/{row['session_id']}/{row['id']}.txt)"
            + f" · [会话日志](sessions/{row['session_id']}/session.log)\n"
        )
    (root / "ANSWERS.md").write_text("".join(answers), encoding="utf-8")
    status = (
        f"严格通过 {summary['normal']}/{summary['planned']}；"
        f"身份正确 {summary['identity_correct']}/{summary['planned']}。"
        if summary["normal_rate"] is not None
        else f"语义复核尚未完成：{summary['reviewed']}/{summary['planned']}，不发布最终正确率。"
    )
    (root / "REPORT.md").write_text(
        f"# {manifest['model_label']} 自我认知测试\n\n{status}\n\n"
        f"已尝试 {summary['attempted']}；EOS {summary['eos']}；"
        f"重复警报 {summary['repetition_flags']}；名字出现 {summary['name_mentions']}。\n\n"
        "名字出现仅是自动筛查，否认、引用和模板也可能命中，不能代表身份正确。\n\n"
        "严格通过要求：执行成功、EOS、完整最终答复、think 标签闭合、无明显重复/乱码、"
        "自我名称正确且符合题目的中英文要求。开发者、公司、知识截止日期等不在本次判定范围。\n\n"
        "这是固定专项题集，不代表通用准确率。上下文不足的后续轮次也保留在计划分母中。\n\n"
        "[逐题回答](ANSWERS.md) · [明细 CSV](results.csv) · [分组统计](summary.json)\n",
        encoding="utf-8",
    )
    return summary
