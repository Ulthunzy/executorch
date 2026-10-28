import importlib.util
import re
from pathlib import Path

import pytest

source = Path(__file__).resolve().parents[1] / "tools/gdn_phase_report.py"
spec = importlib.util.spec_from_file_location("gdn_phase_report", source)
phase_report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(phase_report)


def test_parallel_worker_intervals_are_not_added():
    def record(slice_id, reduce):
        return (
            f"GDN_PHASE slice={slice_id} slices=2 tokens=128 heads=32 k=128 v=128 "
            f"vtcm=0 tasks=16 qtimer_hz=19200000 setup_ticks=0 reduce_ticks={reduce} "
            "delta_ticks=0 update_ticks=19200 copy_ticks=0 "
            "vector_reductions=2048 integer_updates=2048"
        )

    first = record(0, 19200)
    second = record(1, 38400)
    case = phase_report.analyze(first + "\n" + second)["cases"][0]
    assert case["critical_slice"] == 1
    assert case["critical_worker_ms"] == 3
    assert case["phase_ms"]["reduce"] == 2
    assert case["vector_reductions"] == 4096
    with pytest.raises(ValueError, match="Incomplete"):
        phase_report.analyze(first)
    with pytest.raises(ValueError, match="Inconsistent"):
        phase_report.analyze(first + "\n" + first)


@pytest.mark.parametrize(
    "replacement",
    ["qtimer_hz=0", "qtimer_hz=-1", "qtimer_hz=nan", "slice=8", "vtcm=2"],
)
def test_phase_report_rejects_invalid_worker_counters(replacement):
    record = (
        "GDN_PHASE slice=0 slices=1 tokens=1 heads=1 k=128 v=128 "
        "vtcm=0 tasks=1 qtimer_hz=19200000 setup_ticks=0 reduce_ticks=19200 "
        "delta_ticks=0 update_ticks=19200 copy_ticks=0 "
        "vector_reductions=1 integer_updates=1"
    )
    key = replacement.split("=", 1)[0]
    record = re.sub(rf"\b{key}=\S+", replacement, record)
    with pytest.raises(ValueError, match="Invalid"):
        phase_report.analyze(record)


def test_phase_report_rejects_truncated_and_duplicate_fields():
    with pytest.raises(ValueError, match="Incomplete GDN_PHASE"):
        phase_report.analyze("GDN_PHASE slice=0 slices=1 tokens=128")
    with pytest.raises(ValueError, match="Duplicate"):
        phase_report.analyze("GDN_PHASE slice=0 slice=1")
