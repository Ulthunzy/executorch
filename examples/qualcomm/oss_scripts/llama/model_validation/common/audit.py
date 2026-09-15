"""Read-only checks of saved generation and PPL evidence using the standard library."""

import math
import statistics
import sys
from collections import Counter
from pathlib import Path

if __package__ == "common":
    from benchmarks import report
    from ppl.artifacts import digest as ppl_digest
else:
    from ..benchmarks import report
    from ..ppl.artifacts import digest as ppl_digest


def _match(saved, expected, label):
    for key, value in expected.items():
        if (
            key not in saved
            or saved[key] != value
            or type(saved[key]) is not type(value)
        ):
            raise ValueError(f"{label}: mismatched {key}")
        if isinstance(value, dict):
            _match(saved[key], value, f"{label}.{key}")


def _dialogue(root, summary):
    manifest = report.read_json(root / "manifest.json")
    identity = manifest["identity"]
    # Dialogue hashes use report.digest's default JSON separators (with spaces).
    fingerprint = report.digest(identity)
    _match(
        summary,
        {"run_fingerprint": fingerprint, "model": manifest["model_label"]},
        "summary",
    )
    cases = report.read_json(root / "cases.json")
    report.validate_cases(cases)
    _match(identity, {"cases_hash": report.digest(cases)}, "manifest identity")
    reviews = report.read_json(root / "review_template.json")
    _match(reviews, {"run_fingerprint": fingerprint}, "review template")
    entries = reviews["reviews"]
    expected_ids = {
        turn["id"] for session in cases["sessions"] for turn in session["turns"]
    }
    if not isinstance(entries, dict) or set(entries) != expected_ids:
        raise ValueError("Review template must contain every expected turn ID")
    rows, source_hashes = [], {}
    for session in cases["sessions"]:
        directory = root / "sessions" / session["id"]
        complete = report.read_json(directory / "complete.json")
        expected_names = {turn["id"] + ".json" for turn in session["turns"]}
        if set(complete["records"]) != expected_names:
            raise ValueError(f"Invalid completion marker: {session['id']}")
        log_path = directory / "session.log"
        source_hashes[log_path.relative_to(root).as_posix()] = report.file_hash(
            log_path
        )
        logs = report.turn_logs(log_path.read_text(encoding="utf-8", errors="replace"))
        for turn in session["turns"]:
            path = directory / (turn["id"] + ".json")
            checksum = report.file_hash(path)
            _match(
                complete["records"], {path.name: checksum}, "completed record SHA256"
            )
            source_hashes[path.relative_to(root).as_posix()] = checksum
            record = report.read_json(path)
            _match(
                record,
                {
                    **{key: turn[key] for key in ("id", "prompt", "turn")},
                    "session_id": session["id"],
                },
                "saved record",
            )
            review = entries[turn["id"]]
            _match(review, {"record_hash": report.digest(record)}, "review record_hash")
            flags = report.flags_for(
                record, logs.get(turn["id"], ""), identity["expected"]
            )
            rows.append(
                {
                    **record,
                    **turn,
                    **flags,
                    **report.grade(record, flags, review),
                    "group": session["group"],
                }
            )
    _match(summary, {"source_hashes": source_hashes}, "summary")
    # review_hash can include external review metadata absent from the template;
    # recompute every grade from the retained reviews and hash-bound records.
    expected = {
        **report.counts(rows),
        "identity_outcomes": dict(Counter(row["identity"] for row in rows)),
        "first_turn": report.counts([row for row in rows if row["turn"] == 1]),
        "followup_turns": report.counts([row for row in rows if row["turn"] > 1]),
        "groups": {
            group: report.counts([row for row in rows if row["group"] == group])
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
    expected["decode_token_per_sec_median"] = (
        statistics.median(speeds) if speeds else None
    )
    _match(summary, expected, "summary counts/grade")


def _ppl(root, summary):
    identity = report.read_json(root / "metadata.json")
    # PPL hashes use compact JSON and reject NaN/Infinity (ppl.artifacts.digest).
    fingerprint = ppl_digest(identity)
    _match(identity, {"schema": "executorch.ppl.run.v1"}, "metadata")
    _match(
        summary,
        {
            "identity_sha256": fingerprint,
            "complete": True,
            "standard_ppl_eligible": True,
        },
        "PPL summary",
    )
    length, context, stride = (
        identity["target_count"] + 1,
        identity["context_len"],
        identity["stride"],
    )
    if (
        any(
            type(value) is not int
            for value in (identity["target_count"], context, stride)
        )
        or length < 2
        or context < 2
        or not 1 <= stride < context
    ):
        raise ValueError("Invalid PPL target_count/context_len/stride")
    windows, previous = [], 1
    # Match ppl.corpus.windows without importing its NumPy dependency.
    for begin in range(0, length - 1, stride):
        end = min(begin + context, length)
        windows.append(
            dict(id=f"W{len(windows):06d}", begin=begin, end=end, target_begin=previous)
        )
        previous = end
        if end == length:
            break
    _match(identity, {"targets_sha256": ppl_digest(windows)}, "window coverage")
    if {path.name for path in root.glob("W*.json")} != {
        window["id"] + ".json" for window in windows
    }:
        raise ValueError("Missing or unexpected PPL window records")
    records, record_hashes = [], {}
    for window in windows:
        record = report.read_json(root / (window["id"] + ".json"))
        body = {key: value for key, value in record.items() if key != "record_sha256"}
        _match(
            record,
            {"identity_sha256": fingerprint, "record_sha256": ppl_digest(body)},
            window["id"],
        )
        _match(
            record,
            {"window": window, "count": window["end"] - window["target_begin"]},
            "window coverage/count",
        )
        values, total = record["token_nll"], record["nll_sum"]
        if (
            not isinstance(values, list)
            or len(values) != record["count"]
            or any(
                type(value) not in (int, float) or not math.isfinite(value) or value < 0
                for value in values
            )
        ):
            raise ValueError(f"Invalid token NLL values: {window['id']}")
        if (
            type(total) not in (int, float)
            or not math.isfinite(total)
            or total < 0
            or not math.isclose(math.fsum(values), total, rel_tol=1e-10, abs_tol=1e-10)
        ):
            raise ValueError(f"Invalid NLL sum: {window['id']}")
        if identity["engine"] == "pt2":
            check = record.get("details", {}).get("causality", {})
            if (
                check.get("passed") is not True
                or check.get("policy") != "bit_exact_prefix_logits"
                or check.get("prefix_length") != window["end"] - window["begin"] - 1
                or not check.get("checks")
                or not all(item.get("bit_exact") is True for item in check["checks"])
            ):
                raise ValueError("PT2 result lacks passing causality evidence")
        records.append(record)
        record_hashes[window["id"]] = record["record_sha256"]
    count = sum(record["count"] for record in records)
    total = math.fsum(value for record in records for value in record["token_nll"])
    mean = total / count
    ppl = math.exp(mean) if mean <= math.log(sys.float_info.max) else None
    _match(
        summary,
        {
            "count": count,
            "target_count": count,
            "nll_sum": total,
            "mean_nll": mean,
            "ppl": ppl,
            "ppl_overflow": ppl is None,
            "model_sha256": identity["model"]["sha256"],
            "record_hashes": record_hashes,
            **{
                key: identity[key]
                for key in (
                    "engine",
                    "metric",
                    "corpus_sha256",
                    "tokens_sha256",
                    "targets_sha256",
                    "tokenizer_sha256",
                    "context_len",
                    "stride",
                )
            },
        },
        "PPL summary aggregation",
    )


def audit_run(directory) -> None:
    """Validate a supported completed run, raising ValueError for invalid evidence."""
    try:
        root = Path(directory)
        summary = report.read_json(root / "summary.json")
        if summary.get("schema") == "executorch.ppl.summary.v1":
            _ppl(root, summary)
        elif (
            "planned" in summary
            and summary.get("schema") is None
            and summary.get("schema_version", 1) == 1
        ):
            _dialogue(root, summary)
        else:
            raise ValueError("Unsupported summary schema; evidence is unverified")
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        AttributeError,
        OverflowError,
    ) as error:
        raise ValueError(f"Invalid evidence in {directory}: {error}") from error
