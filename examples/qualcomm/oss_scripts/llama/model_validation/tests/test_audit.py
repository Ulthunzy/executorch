"""Offline audit fixtures contain only JSON/logs, without models or devices."""

import json
import math
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from model_validation.benchmarks import report
from model_validation.common.audit import audit_run
from model_validation.ppl.artifacts import digest as ppl_digest
from model_validation.reporting import compare

PPL_FIXTURE_LOSS = math.log(2)


def dialogue(root, reviewed=True):
    root.mkdir(parents=True, exist_ok=True)
    turns = [dict(id=f"T{i}", turn=i, prompt="你是谁？") for i in (1, 2)]
    cases = dict(
        schema_version=1, sessions=[dict(id="S1", group="identity", turns=turns)]
    )
    identity = dict(
        cases_hash=report.digest(cases), expected=dict(chinese="陆小兮", english="Luxi")
    )
    report.write_json(root / "cases.json", cases)
    report.write_json(
        root / "manifest.json",
        dict(schema_version=1, identity=identity, model_label="fixture"),
    )
    session = root / "sessions/S1"
    session.mkdir(parents=True)
    for turn in turns:
        report.write_json(
            session / (turn["id"] + ".json"),
            dict(
                **turn, session_id="S1", response="我是陆小兮。<|im_end|>", error_code=0
            ),
        )
    (session / "session.log").write_text(
        "".join(
            f"IDENTITY_TURN_BEGIN {turn['id']}\nReached EOS; stopping generation\nIDENTITY_TURN_END {turn['id']}\n"
            for turn in turns
        ),
        encoding="utf-8",
    )
    report.write_json(
        session / "complete.json",
        dict(
            records={
                turn["id"] + ".json": report.file_hash(session / (turn["id"] + ".json"))
                for turn in turns
            }
        ),
    )
    report.summarize(root)
    if reviewed:
        reviews = report.read_json(root / "review_template.json")
        reviews["reviewer_metadata"] = {"source": "external review", "version": 3}
        for review in reviews["reviews"].values():
            review.update(identity="correct", language_ok=True)
        report.write_json(root / "external_reviews.json", reviews)
        report.summarize(root, root / "external_reviews.json")
    return root


def ppl_run(root, engine="hf", loss=PPL_FIXTURE_LOSS, top1=True):
    root.mkdir(parents=True, exist_ok=True)
    windows = [
        dict(id="W000000", begin=0, end=4, target_begin=1),
        dict(id="W000001", begin=2, end=6, target_begin=4),
    ]
    identity = dict(
        schema="executorch.ppl.run.v1",
        engine=engine,
        model={"sha256": "model"},
        corpus_sha256="corpus",
        tokens_sha256="tokens",
        targets_sha256=ppl_digest(windows),
        tokenizer_sha256="tokenizer",
        target_count=5,
        context_len=4,
        stride=2,
        metric={"name": "standard_teacher_forced_perplexity"},
    )
    report.write_json(root / "metadata.json", identity)
    records = {}
    for window in windows:
        values = [loss] * (window["end"] - window["target_begin"])
        details = (
            {
                "causality": dict(
                    passed=True,
                    policy="bit_exact_prefix_logits",
                    prefix_length=window["end"] - window["begin"] - 1,
                    checks=[{"bit_exact": True}],
                )
            }
            if engine == "pt2"
            else {}
        )
        record = dict(
            window=window,
            count=len(values),
            token_nll=values,
            nll_sum=math.fsum(values),
            identity_sha256=ppl_digest(identity),
            details=details,
        )
        if top1:
            record["top1_token_ids"] = [0] * len(values)
        record["record_sha256"] = ppl_digest(record)
        records[window["id"]] = record
        report.write_json(root / (window["id"] + ".json"), record)
    total = math.fsum(
        value for record in records.values() for value in record["token_nll"]
    )
    mean = total / 5
    ppl = math.exp(mean) if mean <= math.log(sys.float_info.max) else None
    saved_summary = dict(
        schema="executorch.ppl.summary.v1",
        complete=True,
        standard_ppl_eligible=True,
        count=5,
        target_count=5,
        nll_sum=total,
        mean_nll=mean,
        ppl=ppl,
        ppl_overflow=ppl is None,
        identity_sha256=ppl_digest(identity),
        model_sha256=identity["model"]["sha256"],
        record_hashes={key: value["record_sha256"] for key, value in records.items()},
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
    )
    if top1:
        saved_summary["top1_available"] = True
    report.write_json(root / "summary.json", saved_summary)
    return root


def change(path, mutate):
    value = report.read_json(path)
    mutate(value)
    report.write_json(path, value)


def assert_invalid(root):
    with pytest.raises(ValueError):
        audit_run(root)
    row = compare.collect(root)
    assert row["status"] == "invalid_evidence"
    assert row["error"]
    assert all(
        value is None
        for key, value in row.items()
        if key not in ("run", "status", "error")
    )


def test_dialogue_recomputes_full_reviews_with_external_metadata(tmp_path):
    root = dialogue(tmp_path)
    assert audit_run(root) is None
    row = compare.collect(root)
    assert row["status"] == "reviewed"
    assert row["normal_rate"] == 1.0
    assert row["reviewed"] == 2


def test_pending_review_is_valid_evidence_without_a_score(tmp_path):
    root = dialogue(tmp_path, reviewed=False)
    assert audit_run(root) is None
    row = compare.collect(root)
    assert row["status"] == "pending_review"
    assert row["normal_rate"] is None


@pytest.mark.parametrize(
    "name",
    [
        "manifest.json",
        "cases.json",
        "review_template.json",
        "sessions/S1/complete.json",
        "sessions/S1/session.log",
        "sessions/S1/T1.json",
    ],
)
def test_dialogue_requires_all_evidence(tmp_path, name):
    root = dialogue(tmp_path)
    (root / name).unlink()
    assert_invalid(root)


@pytest.mark.parametrize(
    "name,mutate",
    [
        (
            "manifest.json",
            lambda value: value["identity"].update(expected={"english": "other"}),
        ),
        (
            "cases.json",
            lambda value: value["sessions"][0]["turns"][0].update(prompt="changed"),
        ),
        ("sessions/S1/complete.json", lambda value: value["records"].pop("T2.json")),
        (
            "sessions/S1/complete.json",
            lambda value: value["records"].update({"T1.json": "fake"}),
        ),
        ("sessions/S1/T1.json", lambda value: value.update(response="changed")),
        ("review_template.json", lambda value: value["reviews"].pop("T2")),
        ("review_template.json", lambda value: value["reviews"].update(unknown={})),
        ("review_template.json", lambda value: value.update(run_fingerprint="other")),
        (
            "review_template.json",
            lambda value: value["reviews"]["T1"].update(record_hash="fake"),
        ),
        (
            "review_template.json",
            lambda value: value["reviews"]["T1"].update(identity="wrong"),
        ),
        (
            "review_template.json",
            lambda value: value["reviews"]["T1"].update(language_ok=False),
        ),
        (
            "review_template.json",
            lambda value: value["reviews"]["T1"].update(other_issue=True),
        ),
        (
            "summary.json",
            lambda value: value.update(planned=100, attempted=100, reviewed=100),
        ),
        ("summary.json", lambda value: value.update(normal_rate=0.25)),
        ("summary.json", lambda value: value["groups"]["identity"].update(normal=0)),
        (
            "summary.json",
            lambda value: value["source_hashes"].pop("sessions/S1/T2.json"),
        ),
        (
            "summary.json",
            lambda value: value["source_hashes"].update(
                {"sessions/S1/session.log": None}
            ),
        ),
    ],
)
def test_dialogue_rejects_forged_evidence(tmp_path, name, mutate):
    root = dialogue(tmp_path)
    change(root / name, mutate)
    assert_invalid(root)


def test_dialogue_log_hash_is_verified(tmp_path):
    root = dialogue(tmp_path)
    (root / "sessions/S1/session.log").write_text("changed", encoding="utf-8")
    assert_invalid(root)


@pytest.mark.parametrize("engine", ["hf", "pte", "pt2"])
def test_valid_ppl(tmp_path, engine):
    root = ppl_run(tmp_path, engine)
    assert audit_run(root) is None
    row = compare.collect(root)
    assert row["status"] == "complete"
    assert row["targets"] == 5
    assert row["ppl"] == pytest.approx(2)


def test_overflow_ppl_is_audited_but_not_published_as_a_score(tmp_path):
    root = ppl_run(tmp_path, loss=1000.0)
    assert audit_run(root) is None
    assert compare.collect(root)["ppl"] is None


def test_legacy_ppl_without_top1_remains_auditable(tmp_path):
    assert audit_run(ppl_run(tmp_path, top1=False)) is None


@pytest.mark.parametrize("name", ["metadata.json", "W000000.json", "W000001.json"])
def test_ppl_requires_raw_evidence(tmp_path, name):
    root = ppl_run(tmp_path)
    (root / name).unlink()
    assert_invalid(root)


@pytest.mark.parametrize(
    "field,value",
    [
        ("ppl", 1.0),
        ("count", 4),
        ("count", True),
        ("nll_sum", 0.0),
        ("mean_nll", 0.0),
        ("record_hashes", {}),
        ("identity_sha256", "fake"),
        ("targets_sha256", "fake"),
        ("standard_ppl_eligible", False),
    ],
)
def test_ppl_summary_is_recomputed(tmp_path, field, value):
    root = ppl_run(tmp_path)
    change(root / "summary.json", lambda summary: summary.update({field: value}))
    assert_invalid(root)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda record: record.update(count=2),
        lambda record: record.update(count=True),
        lambda record: record.update(token_nll=[0.0]),
        lambda record: record.update(token_nll=[-1.0, 0.0, 0.0], nll_sum=-1.0),
        lambda record: record.update(token_nll=[True, 0.0, 0.0], nll_sum=1.0),
        lambda record: record.update(top1_token_ids=[0]),
        lambda record: record.update(nll_sum=100.0),
        lambda record: record.update(identity_sha256="fake"),
        lambda record: record["window"].update(target_begin=2),
        lambda record: record["window"].update(begin=1),
        lambda record: record["window"].update(begin=False),
        lambda record: record["window"].update(end=5),
    ],
)
def test_resealed_ppl_records_still_require_valid_nll_and_coverage(tmp_path, mutate):
    root = ppl_run(tmp_path)
    record = report.read_json(root / "W000000.json")
    mutate(record)
    record.pop("record_sha256")
    record["record_sha256"] = ppl_digest(record)
    report.write_json(root / "W000000.json", record)
    change(
        root / "summary.json",
        lambda summary: summary["record_hashes"].update(
            W000000=record["record_sha256"]
        ),
    )
    assert_invalid(root)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_nll_is_rejected(tmp_path, value):
    root = ppl_run(tmp_path)
    change(
        root / "W000000.json", lambda record: record.update(token_nll=[value, 0.0, 0.0])
    )
    assert_invalid(root)


def test_ppl_rejects_extra_windows_and_metadata_changes(tmp_path):
    root = ppl_run(tmp_path)
    (root / "W999999.json").write_bytes((root / "W000000.json").read_bytes())
    assert_invalid(root)
    (root / "W999999.json").unlink()
    change(root / "metadata.json", lambda identity: identity.update(stride=1))
    assert_invalid(root)


@pytest.mark.parametrize("wrong_digest", [True, False])
def test_ppl_record_digest_uses_its_own_canonical_json(tmp_path, wrong_digest):
    root = ppl_run(tmp_path)
    record = report.read_json(root / "W000000.json")
    if wrong_digest:
        record.pop("record_sha256")
        record["record_sha256"] = report.digest(record)
    else:
        record["token_nll"][0] += 1
        record["nll_sum"] = math.fsum(record["token_nll"])
    report.write_json(root / "W000000.json", record)
    assert_invalid(root)


def test_malformed_session_log_raises_value_error(tmp_path):
    root = dialogue(tmp_path)
    (root / "sessions/S1/session.log").write_text(
        "IDENTITY_TURN_BEGIN \n", encoding="utf-8"
    )
    assert_invalid(root)


def test_pt2_requires_causality_even_when_hashes_are_updated(tmp_path):
    root = ppl_run(tmp_path, "pt2")
    record = report.read_json(root / "W000000.json")
    record["details"] = {}
    record.pop("record_sha256")
    record["record_sha256"] = ppl_digest(record)
    report.write_json(root / "W000000.json", record)
    change(
        root / "summary.json",
        lambda summary: summary["record_hashes"].update(
            W000000=record["record_sha256"]
        ),
    )
    assert_invalid(root)


@pytest.mark.parametrize(
    "summary",
    [
        dict(planned=100, reviewed=100, normal_rate=1.0),
        dict(
            schema="executorch.ppl.summary.v1",
            complete=True,
            standard_ppl_eligible=True,
            ppl=2.0,
            count=5,
        ),
    ],
)
def test_summary_alone_is_not_valid_evidence(tmp_path, summary):
    report.write_json(tmp_path / "summary.json", summary)
    assert_invalid(tmp_path)


@pytest.mark.parametrize(
    "summary",
    [
        dict(ppl=2.0, count=5, complete=True, standard_ppl_eligible=True),
        dict(schema="unknown.v2", planned=100, reviewed=100, normal_rate=1.0),
    ],
)
def test_unknown_schema_never_becomes_verified(tmp_path, summary):
    report.write_json(tmp_path / "summary.json", summary)
    row = compare.collect(tmp_path)
    assert row["status"] == "unverified_or_incomplete"
    assert row["ppl"] is row["normal_rate"] is None


@pytest.mark.parametrize("contents", ["{", "[]", "null"])
def test_malformed_summary_is_invalid(tmp_path, contents):
    (tmp_path / "summary.json").write_text(contents, encoding="utf-8")
    assert_invalid(tmp_path)


def test_audit_is_read_only_and_imports_no_compute_libraries(tmp_path):
    roots = [dialogue(tmp_path / "dialogue"), ppl_run(tmp_path / "ppl")]
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    script = """
import sys
sys.path.insert(0, sys.argv[1])
from model_validation.common.audit import audit_run
def guard(event, args):
    if event == 'open' and (isinstance(args[1], str) and any(c in args[1] for c in 'wax+') or args[2] & 3):
        raise AssertionError('write attempted')
    if event.startswith(('subprocess.', 'socket.')) or event in ('os.mkdir', 'os.rename', 'os.remove'):
        raise AssertionError(event)
sys.addaudithook(guard)
for root in sys.argv[2:]:
    audit_run(root)
assert not any(name.split('.')[0] in ('torch', 'numpy', 'transformers') or name.endswith('.evaluate') for name in sys.modules)
"""
    subprocess.run(
        [sys.executable, "-B", "-S", "-c", script, str(ROOT.parent), *map(str, roots)],
        check=True,
    )
    after = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert before == after


def test_comparison_publication_can_retry_after_interruption(tmp_path, monkeypatch):
    root = dialogue(tmp_path / "run")
    output = tmp_path / "comparison"
    monkeypatch.setattr(sys, "argv", ["compare", str(root), "--output", str(output)])
    original = Path.write_text

    def interrupted(path, *args, **kwargs):
        if path.name == "REPORT.md":
            raise OSError("interrupted publication")
        return original(path, *args, **kwargs)

    with patch.object(Path, "write_text", interrupted), pytest.raises(OSError):
        compare.main()
    assert not output.exists()
    assert list(tmp_path.glob(".comparison.pending-*"))
    compare.main()
    assert {path.name for path in output.iterdir()} == {
        "comparison.csv",
        "comparison.json",
        "REPORT.md",
    }
    assert (
        json.loads((output / "comparison.json").read_text())[0]["status"] == "reviewed"
    )
