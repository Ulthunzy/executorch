import importlib.util
import json
from pathlib import Path

import pytest


source = Path(__file__).resolve().parents[1] / "tools/moe_benchmark.py"
spec = importlib.util.spec_from_file_location("moe_benchmark", source)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


@pytest.mark.parametrize("modality,prefill", [("text", 100), ("image", 80)])
def test_prefill_matches_reference_timing_scope_and_rejects_truncation(
    tmp_path, modality, prefill
):
    stats = dict(
        prompt_tokens=256,
        generated_tokens=256,
        context_capacity=2048,
        stop_reason="max_new_tokens",
        prefill_token_per_sec=80,
        decode_token_per_sec=12,
    )
    log = tmp_path / "console.log"
    prefix = "PREFILL_TOTAL tokens=256 chunks=2 tokens_per_second=100.0\n"
    log.write_text(prefix + "CHAT_STATS turn=1 " + json.dumps(stats))
    assert benchmark.read_measurement(log, 256, modality) == dict(
        prefill=prefill, decode=12
    )
    stats["generated_tokens"] = 255
    log.write_text(prefix + "CHAT_STATS turn=1 " + json.dumps(stats))
    with pytest.raises(ValueError, match="Truncated"):
        benchmark.read_measurement(log, 256, modality)


def test_incomplete_matrix_does_not_pass():
    row = dict(
        modality="text",
        tokens=256,
        slots=32,
        rates=dict(prefill=200, decode=15),
    )
    summary = benchmark.summarize([row], repeats=2)
    assert len(summary) == 12
    assert not any(case["exceeds_sm8975"] for case in summary)
    summary = benchmark.summarize([row, row], repeats=2)
    assert summary[0]["exceeds_sm8975"]
    assert not all(case["exceeds_sm8975"] for case in summary)


def test_rates_must_strictly_exceed_reference():
    row = dict(
        modality="image",
        tokens=1024,
        slots=64,
        rates=dict(prefill=251.90, decode=12.19),
    )
    case = benchmark.summarize([row], repeats=1)[-1]
    assert case["complete"]
    assert not case["exceeds_sm8975"]


def test_plan_preserves_io_and_build_selection(tmp_path):
    image = tmp_path / "image.jpg"
    image.write_bytes(b"benchmark input")
    output = tmp_path / "plan"
    benchmark.main(
        [
            "--weights-path",
            str(tmp_path / "weights"),
            "--image",
            str(image),
            "--output",
            str(output),
            "--repeats",
            "1",
            "--build-dir",
            str(tmp_path / "build"),
            "--io-workers",
            "16",
            "--prefill-io-workers",
            "8",
            "--read-chunk-kib",
            "512",
        ]
    )
    report = json.loads((output / "RESULTS.json").read_text())
    assert len(report["commands"]) == 12
    assert not report["all_throughput_targets_passed"]
    for row in report["commands"]:
        command = row["command"]
        for flag, value in (
            ("--build-dir", str(tmp_path / "build")),
            ("--io-workers", "16"),
            ("--prefill-io-workers", "8"),
            ("--read-chunk-kib", "512"),
            ("--temperature", "0"),
        ):
            assert command[command.index(flag) + 1] == value
