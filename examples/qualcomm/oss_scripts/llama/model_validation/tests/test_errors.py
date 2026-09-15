# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tiny JSON fixtures for bound reviews and arbitrary-length session tables."""

import csv
import importlib.util
import json
import re
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import unquote

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "reporting/errors.py"
SPEC = importlib.util.spec_from_file_location("session_errors", SCRIPT)
errors = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(errors)


@pytest.fixture
def run(tmp_path):
    root = tmp_path / "run with spaces"
    root.mkdir()
    sessions, records, reviews = [], {}, {}
    for sid, length in (("passed", 1), ("pending", 2), ("failed", 3)):
        directory = root / "sessions" / sid
        directory.mkdir(parents=True)
        turns = []
        for index in range(1, length + 1):
            tid = f"{sid}T{index}"
            turn = {
                "id": tid,
                "turn": index,
                "prompt": f"问题 {sid} {index} | <tag>\n下一行",
            }
            turns.append(turn)
            if sid == "failed" and index == 3:
                continue
            response = "<think>不应复制的思考</think>实际回答 " + tid + "<|im_end|>"
            record = {**turn, "session_id": sid, "response": response, "error_code": 0}
            errors.write_json(directory / f"{tid}.json", record)
            (directory / f"{tid}.txt").write_text(response, encoding="utf-8")
            records[tid] = record
            if sid != "pending":
                reviews[tid] = {
                    "record_hash": errors.digest(record),
                    "identity": (
                        "wrong" if sid == "failed" and index == 1 else "correct"
                    ),
                    "language_ok": True,
                    "note": "与已绑定的复核一致",
                }
        sessions.append({"id": sid, "group": "tiny", "turns": turns})
    cases = {"schema_version": 1, "sessions": sessions}
    manifest = {
        "model_label": "any-model",
        "identity": {"model_hash": "model-a", "cases_hash": errors.digest(cases)},
    }
    errors.write_json(root / "cases.json", cases)
    errors.write_json(root / "manifest.json", manifest)
    reviews_path = tmp_path / "reviews.json"
    errors.write_json(
        reviews_path,
        {"run_fingerprint": errors.digest(manifest["identity"]), "reviews": reviews},
    )
    return root, reviews_path, records


def test_multiturn_table_prioritizes_failures_and_preserves_sources(run, tmp_path):
    root, reviews, records = run
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    references = tmp_path / "references.json"
    errors.write_json(references, {"failedT1": "指定参考文本"})
    output = tmp_path / "report"
    manifest = errors.build_report(root, output, reviews, references)
    with (output / "errors.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["session_id"] for row in rows] == ["failed", "pending", "passed"]
    assert rows[0]["failed_turns"] == "1"
    assert rows[0]["pending_turns"] == rows[0]["unattempted_turns"] == "1"
    assert rows[0]["turn_3_review_status"] == "pending_review"
    assert rows[0]["turn_1_reference"] == "指定参考文本"
    assert rows[1]["turn_1_reference"] == "未提供参考答案"
    assert rows[1]["failed_turns"] == "0"
    assert rows[1]["turn_1_identity"] == "pending"
    assert manifest["turn_counts"] == {
        "reviewed_failed": 1,
        "reviewed_passed": 2,
        "pending_review": 3,
    }
    markdown = (output / "REPORT.md").read_text()
    assert "第3轮输入" in markdown
    assert "不应复制的思考" not in markdown
    assert "陆小兮" not in markdown and "lu xiaoxi" not in markdown
    assert "\\| &lt;tag&gt;<br>下一行" in markdown
    assert set(path.name for path in output.iterdir()) == {
        "errors.csv",
        "REPORT.md",
        "manifest.json",
    }
    assert {
        path: path.read_bytes() for path in root.rglob("*") if path.is_file()
    } == before
    for row in rows:
        for key, value in row.items():
            if key.endswith("_record_path") and value:
                assert (output / value).resolve() in before
    for name, expected in manifest["artifact_hashes"].items():
        assert errors.file_hash(output / name) == expected
    for filename, expected in manifest["source_hashes"].items():
        assert (
            errors.file_hash(Path(filename)) if Path(filename).is_file() else None
        ) == expected
    for link in re.findall(r"\]\(([^)]+)\)", markdown):
        assert (output / unquote(link)).is_file()


def test_without_reviews_every_turn_remains_pending(run, tmp_path):
    root, _, _ = run
    manifest = errors.build_report(root, tmp_path / "report")
    assert manifest["turn_counts"] == {"pending_review": 6}
    assert manifest["session_counts"] == {"pending_review": 3}
    assert "未提供参考答案" in (tmp_path / "report/REPORT.md").read_text()


@pytest.mark.parametrize(
    "change,message",
    [
        ("model", "different run"),
        ("record", "Stale review"),
        ("cases", "question set"),
        ("wrong_record_id", "Mismatched saved record"),
        ("unknown_review", "unknown turn IDs"),
        ("missing_record", "unattempted turn"),
        ("invalid_decision", "Invalid review"),
        ("invalid_boolean", "boolean"),
        ("invalid_note", "note must be text"),
    ],
)
def test_rejects_invalid_bindings_before_output(run, tmp_path, change, message):
    root, reviews_path, records = run
    reviews = errors.read_json(reviews_path)
    if change == "model":
        manifest = errors.read_json(root / "manifest.json")
        manifest["identity"]["model_hash"] = "another-model"
        errors.write_json(root / "manifest.json", manifest)
    elif change in {"record", "wrong_record_id"}:
        record = dict(records["failedT1"])
        record["response" if change == "record" else "id"] = "changed"
        errors.write_json(root / "sessions/failed/failedT1.json", record)
    elif change == "cases":
        cases = errors.read_json(root / "cases.json")
        cases["sessions"][0]["turns"][0]["prompt"] = "changed"
        errors.write_json(root / "cases.json", cases)
    elif change == "unknown_review":
        reviews["reviews"]["unknown"] = reviews["reviews"]["failedT1"]
    elif change == "missing_record":
        reviews["reviews"]["failedT3"] = reviews["reviews"]["failedT1"]
    elif change == "invalid_decision":
        reviews["reviews"]["failedT1"]["identity"] = "made-up-score"
    elif change == "invalid_note":
        reviews["reviews"]["failedT1"]["note"] = {"not": "text"}
    else:
        reviews["reviews"]["failedT1"]["repetition_override"] = "false"
    errors.write_json(reviews_path, reviews)
    with pytest.raises(ValueError, match=message):
        errors.build_report(root, tmp_path / "report", reviews_path)
    assert not (tmp_path / "report").exists()


def test_turn_logs_do_not_leak_eos_between_turns(run, tmp_path):
    root, reviews, records = run
    for tid in ("failedT1", "failedT2"):
        record = {**records[tid], "response": "正式输出"}
        errors.write_json(root / f"sessions/failed/{tid}.json", record)
    (root / "sessions/failed/session.log").write_text(
        "IDENTITY_TURN_BEGIN failedT1\nReached EOS; stopping generation\nIDENTITY_TURN_END failedT1\n"
        "IDENTITY_TURN_BEGIN failedT2\nGeneration stopped at seq_len limit\nIDENTITY_TURN_END failedT2\n"
    )
    output = tmp_path / "report"
    errors.build_report(root, output)
    with (output / "errors.csv").open(encoding="utf-8-sig", newline="") as stream:
        row = next(
            row for row in csv.DictReader(stream) if row["session_id"] == "failed"
        )
    assert row["turn_1_stop_reason"] == "eos"
    assert row["turn_2_stop_reason"] == "length_limit"
    assert row["failed_turns"] == "0"


def test_cli_and_new_output_requirement(run, tmp_path):
    root, reviews, _ = run
    output = tmp_path / "cli-report"
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            str(SCRIPT),
            str(root),
            "--output",
            str(output),
            "--reviews",
            str(reviews),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (output / "REPORT.md").is_file()
    before = (output / "manifest.json").read_bytes()
    with pytest.raises(ValueError, match="new output directory"):
        errors.build_report(root, output)
    assert (output / "manifest.json").read_bytes() == before


@pytest.mark.parametrize("references", [{"unknown": "answer"}, {"passedT1": 42}])
def test_invalid_reference_mapping_creates_no_report(run, tmp_path, references):
    root, _, _ = run
    path = tmp_path / "references.json"
    errors.write_json(path, references)
    with pytest.raises(ValueError, match="References must map"):
        errors.build_report(root, tmp_path / "report", references_path=path)
    assert not (tmp_path / "report").exists()


def test_long_answer_is_an_explicit_excerpt_with_a_raw_link(run, tmp_path):
    root, _, records = run
    record = {
        **records["passedT1"],
        "response": "<think>thought</think>" + "正文" * 250 + "<|im_end|>",
    }
    source = root / "sessions/passed/passedT1.json"
    errors.write_json(source, record)
    before = source.read_bytes()
    output = tmp_path / "report"
    errors.build_report(root, output)
    with (output / "errors.csv").open(encoding="utf-8-sig", newline="") as stream:
        row = next(
            row for row in csv.DictReader(stream) if row["session_id"] == "passed"
        )
    assert (
        row["turn_1_final_answer_summary"] == "正文" * 200 + "…（摘要截断；见原始记录）"
    )
    assert (output / row["turn_1_record_path"]).resolve() == source
    assert source.read_bytes() == before


def test_failed_write_preserves_pending_and_can_retry(run, tmp_path, monkeypatch):
    root, reviews, _ = run
    output = tmp_path / "atomic-report"
    write_json = errors.write_json

    def interrupted(path, value):
        if path.name == "manifest.json":
            assert not output.exists()
            assert (path.parent / "errors.csv").is_file()
            assert (path.parent / "REPORT.md").is_file()
            raise OSError("simulated disk failure")
        write_json(path, value)

    with monkeypatch.context() as patch:
        patch.setattr(errors, "write_json", interrupted)
        with pytest.raises(OSError, match="simulated disk failure"):
            errors.build_report(root, output, reviews)
    assert not output.exists()
    pending = list(tmp_path.glob(".atomic-report.pending-*"))
    assert len(pending) == 1
    retained = {path.name: path.read_bytes() for path in pending[0].iterdir()}
    errors.build_report(root, output, reviews)
    assert set(path.name for path in output.iterdir()) == {
        "errors.csv",
        "REPORT.md",
        "manifest.json",
    }
    assert list(tmp_path.glob(".atomic-report.pending-*")) == pending
    assert {path.name: path.read_bytes() for path in pending[0].iterdir()} == retained


def test_links_are_computed_from_destination_before_atomic_publish(
    run, tmp_path, monkeypatch
):
    root, reviews, _ = run
    output = tmp_path / "published/report"
    working = tmp_path / "separate/staging/pending/report"

    @contextmanager
    def staged(destination):
        assert destination == output
        working.mkdir(parents=True)
        yield working
        assert not output.exists()
        assert set(path.name for path in working.iterdir()) == {
            "errors.csv",
            "REPORT.md",
            "manifest.json",
        }
        output.parent.mkdir(parents=True)
        working.rename(output)

    monkeypatch.setattr(errors, "atomic_directory", staged)
    manifest = errors.build_report(root, output, reviews)
    assert not working.exists()
    with (output / "errors.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            for field, value in row.items():
                if field.endswith(("_record_path", "_text_path")) and value:
                    assert (output / value).resolve().is_file()
    markdown = (output / "REPORT.md").read_text()
    for link in re.findall(r"\]\(([^)]+)\)", markdown):
        assert (output / unquote(link)).is_file()
    for name, expected in manifest["artifact_hashes"].items():
        assert errors.file_hash(output / name) == expected
