"""Shared, conservative scoring for HF/PT2/PTE business validation cases."""

import csv
import re
import unicodedata
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    from report import (
        digest,
        file_hash,
        flags_for,
        read_json,
        turn_logs,
        validate_cases,
        write_json,
    )
    from sample_scoring import extract_choice, extract_number, numeric_value
else:
    from .report import (
        digest,
        file_hash,
        flags_for,
        read_json,
        turn_logs,
        validate_cases,
        write_json,
    )
    from .sample_scoring import extract_choice, extract_number, numeric_value


SCORERS = {"manual", "exact_match", "contains", "choice", "numeric"}


def _normalize(value, case_sensitive=False):
    value = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value))).strip()
    return value if case_sensitive else value.casefold()


def _validate_scoring(turn):
    scoring = turn.get("scoring", {"type": "manual"})
    if not isinstance(scoring, dict) or scoring.get("type") not in SCORERS:
        raise ValueError(f"Unsupported scorer for {turn['id']}")
    if set(scoring) - {"type", "mode", "case_sensitive"}:
        raise ValueError(f"Unknown scorer option for {turn['id']}")
    if "case_sensitive" in scoring and type(scoring["case_sensitive"]) is not bool:
        raise ValueError(f"case_sensitive must be boolean: {turn['id']}")
    kind = scoring["type"]
    if kind != "manual" and "reference" not in turn:
        raise ValueError(f"{kind} requires reference: {turn['id']}")
    if kind == "contains" and scoring.get("mode", "all") not in ("all", "any"):
        raise ValueError(f"contains mode must be all or any: {turn['id']}")
    if kind in ("exact_match", "contains"):
        references = turn["reference"]
        references = references if isinstance(references, list) else [references]
        if not references or any(
            not isinstance(value, str) or not _normalize(value, True)
            for value in references
        ):
            raise ValueError(
                f"{kind} references must be nonempty strings: {turn['id']}"
            )
    elif kind == "choice" and not re.fullmatch(
        r"[A-P]", str(turn["reference"]).strip().upper()
    ):
        raise ValueError(f"Invalid choice reference: {turn['id']}")
    elif kind == "numeric" and numeric_value(str(turn["reference"])) is None:
        raise ValueError(f"Invalid numeric reference: {turn['id']}")


def validate_business_cases(cases):
    validate_cases(cases)
    for session in cases["sessions"]:
        if "critical" in session and type(session["critical"]) is not bool:
            raise ValueError(f"critical must be boolean: {session['id']}")
        for turn in session["turns"]:
            if "formatted_prompt" in turn:
                raise ValueError(
                    "Shared business cases must use plain prompts, not formatted_prompt"
                )
            if "critical" in turn and type(turn["critical"]) is not bool:
                raise ValueError(f"critical must be boolean: {turn['id']}")
            _validate_scoring(turn)


def score_turn(turn, answer):
    scoring = turn.get("scoring", {"type": "manual"})
    kind = scoring["type"]
    result = {
        "semantic_passed": None,
        "score_method": kind,
        "prediction": None,
        "target": turn.get("reference"),
    }
    if kind == "manual":
        return result
    if not answer:
        return {**result, "semantic_passed": False, "score_method": "no_answer"}
    references = turn["reference"]
    references = references if isinstance(references, list) else [references]
    case_sensitive = scoring.get("case_sensitive", False)
    normalized_answer = _normalize(answer, case_sensitive)
    if kind == "exact_match":
        normalized = [_normalize(value, case_sensitive) for value in references]
        return {
            **result,
            "semantic_passed": normalized_answer in normalized,
            "prediction": normalized_answer,
            "target": normalized,
        }
    if kind == "contains":
        normalized = [_normalize(value, case_sensitive) for value in references]
        matches = [value in normalized_answer for value in normalized]
        passed = all(matches) if scoring.get("mode", "all") == "all" else any(matches)
        return {
            **result,
            "semantic_passed": passed,
            "prediction": matches,
            "target": normalized,
        }
    if len(references) != 1:
        raise ValueError(f"{kind} accepts exactly one reference: {turn['id']}")
    if kind == "choice":
        target = str(references[0]).strip().upper()
        if not re.fullmatch(r"[A-P]", target):
            raise ValueError(f"Invalid choice reference: {turn['id']}")
        prediction = extract_choice(answer)
        return {
            **result,
            "semantic_passed": prediction == target if prediction else None,
            "prediction": prediction,
            "target": target,
            "score_method": "explicit_choice" if prediction else "manual",
        }
    target = numeric_value(str(references[0]))
    prediction_text = extract_number(answer)
    prediction = numeric_value(prediction_text) if prediction_text else None
    if target is None:
        raise ValueError(f"Invalid numeric reference: {turn['id']}")
    return {
        **result,
        "semantic_passed": prediction == target if prediction is not None else None,
        "prediction": str(prediction) if prediction is not None else None,
        "target": str(target),
        "score_method": "exact_number" if prediction is not None else "manual",
    }


def collect_business_rows(root, reviews=None, *, require_complete=False):  # noqa: C901
    root = Path(root)
    manifest = read_json(root / "manifest.json")
    identity = manifest["identity"]
    if identity.get("evaluation", {}).get("kind") != "business":
        raise ValueError("Run is not marked as a business evaluation")
    cases = read_json(root / "cases.json")
    validate_business_cases(cases)
    if digest(cases) != identity["cases_hash"]:
        raise ValueError("Business cases changed after inference")
    fingerprint = digest(identity)
    reviews = reviews or {}
    if reviews and reviews.get("run_fingerprint") != fingerprint:
        raise ValueError("Reviews belong to another run")
    entries = reviews.get("reviews", {})
    expected_ids = {
        turn["id"] for session in cases["sessions"] for turn in session["turns"]
    }
    if entries.keys() - expected_ids:
        raise ValueError("Unknown business review IDs")
    rows, template, source_hashes = [], {}, {}
    for session in cases["sessions"]:
        directory = root / "sessions" / session["id"]
        complete_path = directory / "complete.json"
        source_hashes[complete_path.relative_to(root).as_posix()] = (
            file_hash(complete_path) if complete_path.is_file() else None
        )
        complete = read_json(complete_path) if complete_path.is_file() else None
        expected_names = {turn["id"] + ".json" for turn in session["turns"]}
        if require_complete and (
            not isinstance(complete, dict)
            or set(complete.get("records", {})) != expected_names
        ):
            raise ValueError(f"Incomplete business session: {session['id']}")
        log_path = directory / "session.log"
        if require_complete and not log_path.is_file():
            raise ValueError(f"Missing business session log: {session['id']}")
        source_hashes[log_path.relative_to(root).as_posix()] = (
            file_hash(log_path) if log_path.is_file() else None
        )
        logs = (
            turn_logs(log_path.read_text(encoding="utf-8", errors="replace"))
            if log_path.is_file()
            else {}
        )
        for turn in session["turns"]:
            path = directory / (turn["id"] + ".json")
            source_hashes[path.relative_to(root).as_posix()] = (
                file_hash(path) if path.is_file() else None
            )
            record = (
                read_json(path)
                if path.is_file()
                else {
                    **turn,
                    "session_id": session["id"],
                    "response": "",
                    "error_code": None,
                    "infrastructure_error": session.get("skip_reason", "not_run"),
                }
            )
            if require_complete and (
                not path.is_file() or complete["records"][path.name] != file_hash(path)
            ):
                raise ValueError(f"Invalid business completion marker: {turn['id']}")
            for key, value in {
                "id": turn["id"],
                "turn": turn["turn"],
                "prompt": turn["prompt"],
                "session_id": session["id"],
            }.items():
                if record.get(key) != value:
                    raise ValueError(f"Mismatched business result {turn['id']}: {key}")
            incremental_input_ids = record.get("incremental_input_ids")
            if incremental_input_ids is not None and (
                not isinstance(incremental_input_ids, list)
                or not incremental_input_ids
                or any(
                    type(value) is not int or value < 0
                    for value in incremental_input_ids
                )
            ):
                raise ValueError(f"Invalid incremental input IDs: {turn['id']}")
            record_hash = digest(record)
            review = entries.get(turn["id"])
            if review:
                if review.get("record_hash") != record_hash:
                    raise ValueError(f"Stale business review: {turn['id']}")
                if (
                    review.get("passed") is not None
                    and type(review["passed"]) is not bool
                ):
                    raise ValueError(f"Invalid business review: {turn['id']}")
            flags = flags_for(record, logs.get(turn["id"], ""), {})
            score = score_turn(turn, flags["final_answer"])
            if review and review.get("passed") is not None:
                score.update(
                    semantic_passed=review["passed"], score_method="manual_review"
                )
            generation_ok = (
                record.get("error_code") == 0
                and flags["stop_reason"] == "eos"
                and flags["thinking_closed"]
                and bool(flags["final_answer"])
                and not flags["repetition_flag"]
                and not flags["garbled_flag"]
            )
            passed = (
                generation_ok and score["semantic_passed"]
                if score["semantic_passed"] is not None
                else None
            )
            critical = bool(turn.get("critical", session.get("critical", False)))
            rows.append(
                {
                    "id": turn["id"],
                    "session_id": session["id"],
                    "group": session["group"],
                    "turn": turn["turn"],
                    "prompt": turn["prompt"],
                    "critical": critical,
                    "record_hash": record_hash,
                    "incremental_input_ids": incremental_input_ids,
                    "native_input_ids": record.get("native_input_ids"),
                    **flags,
                    **score,
                    "generation_ok": generation_ok,
                    "passed": passed,
                    "note": (review or {}).get("note", ""),
                }
            )
            if path.is_file():
                template[turn["id"]] = review or {
                    "record_hash": record_hash,
                    "passed": None,
                    "note": "",
                }
    return manifest, cases, rows, template, source_hashes


def business_counts(rows):
    scored = sum(row["passed"] is not None for row in rows)
    passed = sum(row["passed"] is True for row in rows)
    return {
        "planned": len(rows),
        "attempted": sum(row["stop_reason"] != "not_run" for row in rows),
        "reviewed": scored,
        "pending_review": len(rows) - scored,
        "passed": passed,
        "pass_rate": passed / len(rows) if rows and scored == len(rows) else None,
        "eos": sum(row["stop_reason"] == "eos" for row in rows),
        "repetition_flags": sum(row["repetition_flag"] for row in rows),
        "garbled_flags": sum(row["garbled_flag"] for row in rows),
        "stop_reasons": dict(Counter(row["stop_reason"] for row in rows)),
    }


def summarize_business(root, reviews_path=None):
    root = Path(root)
    reviews = read_json(reviews_path) if reviews_path else {}
    manifest, _, rows, template, source_hashes = collect_business_rows(root, reviews)
    fingerprint = digest(manifest["identity"])
    published_reviews = {"run_fingerprint": fingerprint, "reviews": template}
    summary = {
        "schema": "executorch.business.summary.v1",
        "schema_version": 1,
        "model": manifest["model_label"],
        "run_fingerprint": fingerprint,
        "source_hashes": source_hashes,
        "review_hash": digest(published_reviews),
        **business_counts(rows),
        "groups": {
            group: business_counts([row for row in rows if row["group"] == group])
            for group in dict.fromkeys(row["group"] for row in rows)
        },
        "critical_ids": [row["id"] for row in rows if row["critical"]],
    }
    write_json(root / "summary.json", summary)
    write_json(root / "scored_results.json", rows)
    write_json(root / "review_template.json", published_reviews)
    with (root / "results.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = [
            "id",
            "session_id",
            "group",
            "critical",
            "passed",
            "semantic_passed",
            "score_method",
            "stop_reason",
            "generation_ok",
            "repetition_flag",
            "garbled_flag",
            "thinking_closed",
            "prediction",
            "target",
            "final_answer",
            "note",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# 业务模型验证\n\n",
        f"计划 {summary['planned']}，已评分 {summary['reviewed']}，通过 {summary['passed']}。\n\n",
        f"通过率：{summary['pass_rate'] if summary['pass_rate'] is not None else '待完整复核'}；",
        f"EOS {summary['eos']}；重复 {summary['repetition_flags']}；乱码 {summary['garbled_flags']}。\n\n",
        "| ID | 分组 | 关键 | 通过 | 评分 | 停止原因 |\n",
        "|---|---|---|---|---|---|\n",
    ]
    for row in rows:
        lines.append(
            f"| {row['id']} | {row['group']} | {row['critical']} | "
            f"{row['passed']} | {row['score_method']} | {row['stop_reason']} |\n"
        )
    (root / "REPORT.md").write_text("".join(lines), encoding="utf-8")
    return summary
