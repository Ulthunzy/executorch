import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from moe_profile import compare, summarize  # noqa: E402


def test_profile_keeps_overlapped_times_separate_and_reports_hmx_duty():
    data = summarize(
        """
MOE_PHASE_PROFILE phase=prefill tokens=256 gdn_attention_ms=20 native_ms=60 io_wall_ms=40 io_overlap_ms=30 io_wait_ms=10
CHAT_DECODE_STEP position=256 total_ms=100 gdn_graph_ms=20 read_ms=10 native_ms=60 direct_bytes=1024 cache_hits=3 cache_requests=8
CHAT_DECODE_STEP position=257 total_ms=100 gdn_graph_ms=20 read_ms=10 native_ms=60 direct_bytes=1024 cache_hits=5 cache_requests=8
MOE_NATIVE_PROFILE phase=decode enabled=1 hmx_busy_ms=40 hmx_loop_ms=100 weight_dma_worker_sum_ms=200
"""
    )
    assert data["decode_steps"] == 2
    assert data["host"]["decode"]["total_ms"] == 200
    assert data["host"]["decode"]["cache_hit_fraction"] == 0.5
    assert data["host"]["decode"]["gdn_graph_wall_fraction"] == 0.2
    assert data["host"]["prefill"]["io_overlap_fraction"] == 0.75
    assert data["dsp"]["decode"]["hmx_service_duty_fraction"] == 0.4
    assert data["dsp"]["decode"]["weight_dma_worker_sum_ms"] == 200


def test_profile_comparison_checks_workload_and_does_not_report_zero_division():
    before = summarize("CHAT_DECODE_STEP total_ms=100 gdn_graph_ms=20")
    after = summarize("CHAT_DECODE_STEP total_ms=80 gdn_graph_ms=10")
    changes = compare(before, after)
    assert changes["host.decode.total_ms"]["speedup"] == 1.25
    assert changes["host.decode.gdn_graph_ms"]["speedup"] == 2
    with pytest.raises(ValueError, match="decode steps"):
        compare(before, summarize("MOE_PHASE_PROFILE phase=prefill tokens=128"))
    with pytest.raises(ValueError, match="prefill token"):
        compare(
            summarize("MOE_PHASE_PROFILE phase=prefill tokens=128"),
            summarize("MOE_PHASE_PROFILE phase=prefill tokens=256"),
        )


@pytest.mark.parametrize("text", ["", "MOE_NATIVE_PROFILE phase=decode enabled=0"])
def test_profile_rejects_missing_or_disabled_measurements(text):
    with pytest.raises(ValueError):
        summarize(text)


@pytest.mark.parametrize("value", ["nan", "inf", "-1", "invalid"])
def test_profile_rejects_invalid_numeric_measurements(value):
    with pytest.raises(ValueError, match="invalid numeric"):
        summarize(f"CHAT_DECODE_STEP position=256 total_ms={value}")


def test_profile_rejects_multiple_cumulative_sessions():
    with pytest.raises(ValueError, match="multiple cumulative"):
        summarize(
            "MOE_NATIVE_PROFILE phase=decode enabled=1 hmx_busy_ms=40 hmx_loop_ms=100\n"
            "MOE_NATIVE_PROFILE phase=decode enabled=1 hmx_busy_ms=40 hmx_loop_ms=90"
        )


def test_profile_comparison_requires_matching_context_positions():
    with pytest.raises(ValueError, match="decode context positions"):
        compare(
            summarize("CHAT_DECODE_STEP position=256 total_ms=100"),
            summarize("CHAT_DECODE_STEP position=1024 total_ms=80"),
        )


def test_profile_rejects_partial_hmx_measurements():
    with pytest.raises(ValueError, match="missing HMX busy"):
        summarize("MOE_NATIVE_PROFILE phase=decode enabled=1 hmx_loop_ms=100")
