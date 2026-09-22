"""Offline tests for shared HF/PT2/PTE business scoring evidence."""

from pathlib import Path

import pytest

from business_scoring import score_turn, summarize_business, validate_business_cases
from report import digest, file_hash, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]


def _run(root, review=True):
    cases = {
        "schema_version": 1,
        "sessions": [
            {
                "id": "B1",
                "group": "fixture",
                "critical": True,
                "turns": [
                    {
                        "id": "B1T1",
                        "turn": 1,
                        "prompt": "只回答数字：2+2？",
                        "reference": "4",
                        "scoring": {"type": "numeric"},
                    },
                    {
                        "id": "B1T2",
                        "turn": 2,
                        "prompt": "说明原因。",
                        "scoring": {"type": "manual"},
                    },
                ],
            }
        ],
    }
    identity = {
        "cases_hash": digest(cases),
        "evaluation": {
            "kind": "business",
            "schema_version": 1,
            "scorer_hashes": {
                name: file_hash(ROOT / "benchmarks" / name)
                for name in (
                    "business_benchmark.py",
                    "business_scoring.py",
                    "report.py",
                    "sample_scoring.py",
                )
            },
        },
        "config": {"temperature": 0},
    }
    root.mkdir()
    write_json(root / "cases.json", cases)
    write_json(
        root / "manifest.json",
        {"schema_version": 1, "identity": identity, "model_label": "fixture"},
    )
    session = root / "sessions/B1"
    session.mkdir(parents=True)
    responses = {"B1T1": "4<|im_end|>", "B1T2": "因为加法定义。<|im_end|>"}
    for index, turn in enumerate(cases["sessions"][0]["turns"]):
        write_json(
            session / f"{turn['id']}.json",
            {
                **turn,
                "session_id": "B1",
                "response": responses[turn["id"]],
                "error_code": 0,
                "incremental_input_ids": [100, index],
            },
        )
    (session / "session.log").write_text(
        "".join(
            f"IDENTITY_TURN_BEGIN {turn['id']}\n"
            "Reached EOS; stopping generation\n"
            f"IDENTITY_TURN_END {turn['id']}\n"
            for turn in cases["sessions"][0]["turns"]
        ),
        encoding="utf-8",
    )
    write_json(
        session / "complete.json",
        {
            "records": {
                f"{turn['id']}.json": file_hash(session / f"{turn['id']}.json")
                for turn in cases["sessions"][0]["turns"]
            }
        },
    )
    summary = summarize_business(root)
    reviews = read_json(root / "review_template.json")
    assert summary["review_hash"] == digest(reviews)
    if not review:
        return root
    reviews["reviews"]["B1T2"]["passed"] = True
    write_json(root / "reviews.json", reviews)
    summarize_business(root, root / "reviews.json")
    return root


@pytest.mark.parametrize(
    "turn,answer,expected",
    [
        (
            {
                "id": "E",
                "reference": ["OK", "好的"],
                "scoring": {"type": "exact_match"},
            },
            "好的",
            True,
        ),
        (
            {
                "id": "C",
                "reference": ["alpha", "beta"],
                "scoring": {"type": "contains"},
            },
            "alpha and beta",
            True,
        ),
        (
            {"id": "Q", "reference": "B", "scoring": {"type": "choice"}},
            "答案：B",
            True,
        ),
        (
            {"id": "Q", "reference": "B", "scoring": {"type": "choice"}},
            "A. 红色<|im_end|>",
            False,
        ),
        (
            {"id": "N", "reference": "1/2", "scoring": {"type": "numeric"}},
            "答案是 0.5",
            True,
        ),
    ],
)
def test_shared_scorers(turn, answer, expected):
    assert score_turn(turn, answer)["semantic_passed"] is expected


def test_shipped_business_cases_validate():
    validate_business_cases(read_json(ROOT / "configs/business_cases.example.json"))


def test_business_run_is_hash_bound_and_auditable(tmp_path):
    from common.audit import audit_run, generation_evidence
    from reporting.compare import collect

    root = _run(tmp_path / "run")
    audit_run(root)
    evidence = generation_evidence(root)
    assert [row["passed"] for row in evidence["rows"]] == [True, True]
    assert all(row["critical"] for row in evidence["rows"])
    assert evidence["summary"]["pass_rate"] == 1.0
    assert collect(root)["normal_rate"] == 1.0

    (root / "sessions/B1/complete.json").unlink()
    with pytest.raises(ValueError, match="Incomplete business session"):
        audit_run(root)


def test_pending_business_review_is_valid_auditable_evidence(tmp_path):
    from common.audit import audit_run, generation_evidence

    root = _run(tmp_path / "pending", review=False)
    audit_run(root)
    evidence = generation_evidence(root)
    assert [row["reviewed"] for row in evidence["rows"]] == [True, False]
    assert evidence["summary"]["pass_rate"] is None


def test_business_audit_requires_raw_session_log(tmp_path):
    from common.audit import audit_run

    root = _run(tmp_path / "missing-log")
    (root / "sessions/B1/session.log").unlink()
    summarize_business(root, root / "reviews.json")
    with pytest.raises(ValueError, match="Missing business session log"):
        audit_run(root)
