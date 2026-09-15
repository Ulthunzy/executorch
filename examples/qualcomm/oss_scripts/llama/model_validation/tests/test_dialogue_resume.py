"""Completed dialogue resumes are read-only and direct writers are exclusive."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
import benchmark  # noqa: E402
import hf_benchmark  # noqa: E402
from report import file_hash, read_json, summarize, write_json  # noqa: E402


@pytest.mark.parametrize("module", [benchmark, hf_benchmark])
def test_direct_runs_cannot_share_output(module, tmp_path, monkeypatch):
    args = SimpleNamespace(output=tmp_path / "run")
    monkeypatch.setattr(
        module, "run_locked", lambda *a, **k: pytest.fail("lock bypassed")
    )
    with benchmark.run_lock(args.output):
        with pytest.raises(ValueError, match="Another process owns"):
            module.run(args)
    assert not args.output.exists()


def test_device_completed_resume_keeps_reviews_without_redeploy(tmp_path, monkeypatch):
    for name in ("model.pte", "tokenizer.json", "driver"):
        (tmp_path / name).write_bytes(b"fixture")
    args = SimpleNamespace(
        model=tmp_path / "model.pte",
        tokenizer=None,
        driver=tmp_path / "driver",
        device="fixture",
        device_lib_dir="/data/local/tmp/libs",
        device_model=None,
        seq_len=96,
        timeout=60,
        repetition_penalty=1.0,
        expected_english="Lu Xiaoxi",
        expected_chinese="陆小兮",
        ids=None,
        decoder="lincal",
        mode="hybrid",
        output=tmp_path / "run",
        resume=False,
        label=None,
    )
    cases = {
        "schema_version": 1,
        "sessions": [
            {
                "id": "S",
                "group": "fixture",
                "turns": [
                    {"id": "T1", "turn": 1, "prompt": "你好"},
                    {"id": "T2", "turn": 2, "prompt": "你是谁"},
                ],
            }
        ],
    }

    def adb(_serial, *command, **kwargs):
        return (
            "device"
            if command == ("get-state",)
            else "libQnnHtp.so libqnn_executorch_backend.so"
        )

    def session(_args, root, remote, case, **kwargs):
        directory = root / "sessions/S"
        directory.mkdir(parents=True)
        logs = []
        for turn in case["turns"]:
            write_json(
                directory / (turn["id"] + ".json"),
                {
                    **turn,
                    "session_id": "S",
                    "error_code": 0,
                    "response": "我是陆小兮。<|im_end|>",
                },
            )
            logs.extend(
                [
                    f"IDENTITY_TURN_BEGIN {turn['id']}",
                    "Reached EOS; stopping generation",
                    f"IDENTITY_TURN_END {turn['id']}",
                ]
            )
        (directory / "session.log").write_text("\n".join(logs))
        write_json(
            directory / "complete.json",
            {
                "records": {
                    t["id"] + ".json": file_hash(directory / (t["id"] + ".json"))
                    for t in case["turns"]
                }
            },
        )
        return True

    monkeypatch.setattr(benchmark, "adb", adb)
    monkeypatch.setattr(benchmark, "push_verified", lambda *a: None)
    monkeypatch.setattr(benchmark, "run_session", session)
    benchmark.run(args, cases=cases)
    review = read_json(args.output / "review_template.json")
    for entry in review["reviews"].values():
        entry.update(identity="correct", language_ok=True)
    write_json(tmp_path / "reviews.json", review)
    summarize(args.output, tmp_path / "reviews.json")
    before = {
        str(p.relative_to(args.output)): p.read_bytes()
        for p in args.output.rglob("*")
        if p.is_file()
    }
    monkeypatch.setattr(
        benchmark, "push_verified", lambda *a: pytest.fail("unexpected redeploy")
    )
    monkeypatch.setattr(
        benchmark, "run_session", lambda *a, **k: pytest.fail("unexpected inference")
    )
    args.resume = True
    benchmark.run(args, cases=cases)
    assert before == {
        str(p.relative_to(args.output)): p.read_bytes()
        for p in args.output.rglob("*")
        if p.is_file()
    }
    assert read_json(args.output / "summary.json")["normal"] == 2
    (args.output / "sessions/S/T2.json").unlink()
    with pytest.raises(ValueError, match="Completed records changed"):
        benchmark.run(args, cases=cases)
