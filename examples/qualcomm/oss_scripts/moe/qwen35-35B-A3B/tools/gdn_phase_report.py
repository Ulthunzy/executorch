"""Summarize DSP qtimer phases from a DELTANET_PROFILE_PHASES=1 logcat capture."""

import argparse
import json
import re
from pathlib import Path

PHASES = ("copy", "setup", "reduce", "delta", "update")
REQUIRED_FIELDS = {
    "slice",
    "slices",
    "tokens",
    "heads",
    "k",
    "v",
    "vtcm",
    "tasks",
    "qtimer_hz",
    "vector_reductions",
    "integer_updates",
    *(name + "_ticks" for name in PHASES),
}


def analyze(text):
    groups = {}
    for line in text.splitlines():
        if "GDN_PHASE " not in line:
            continue
        fields = re.findall(r"(\w+)=([^\s]+)", line.split("GDN_PHASE ", 1)[1])
        if len(dict(fields)) != len(fields):
            raise ValueError("Duplicate GDN_PHASE fields")
        missing = REQUIRED_FIELDS - dict(fields).keys()
        if missing:
            raise ValueError(f"Incomplete GDN_PHASE record: missing {sorted(missing)}")
        try:
            record = {key: int(value) for key, value in fields}
        except ValueError as error:
            raise ValueError("Invalid integer in GDN_PHASE record") from error
        if (
            any(value < 0 for value in record.values())
            or any(
                record[name] <= 0
                for name in ("slices", "tokens", "heads", "k", "v", "qtimer_hz")
            )
            or record["slice"] >= record["slices"]
            or record["vtcm"] not in (0, 1)
            or any(
                record[name] > record["tasks"] * record["tokens"]
                for name in ("vector_reductions", "integer_updates")
            )
        ):
            raise ValueError("Invalid GDN_PHASE worker counters")
        shape = tuple(record[name] for name in ("tokens", "heads", "k", "v", "vtcm"))
        groups.setdefault(shape, []).append(record)
    if not groups:
        raise ValueError("No GDN_PHASE records; enable the DSP profiling build")
    cases = []
    for shape, records in groups.items():
        slices = records[-1]["slices"]
        if slices <= 0 or len(records) % slices:
            raise ValueError(
                f"Incomplete invocation for {shape}: {len(records)} records"
            )
        for offset in range(0, len(records), slices):
            invocation = records[offset : offset + slices]
            if (
                {record["slice"] for record in invocation} != set(range(slices))
                or any(record["slices"] != slices for record in invocation)
                or any(
                    record["qtimer_hz"] != invocation[0]["qtimer_hz"]
                    for record in invocation
                )
            ):
                raise ValueError(f"Inconsistent worker records for {shape}")
        invocation = records[-slices:]
        critical = max(
            invocation,
            key=lambda record: sum(record[name + "_ticks"] for name in PHASES),
        )
        total = sum(critical[name + "_ticks"] for name in PHASES)
        if total <= 0:
            raise ValueError("Empty worker timing interval")
        cases.append(
            dict(
                shape=dict(zip(("tokens", "heads", "k", "v", "vtcm"), shape)),
                captured_invocations=len(records) // slices,
                scope="Last complete invocation, slowest worker's accumulated intervals; excludes graph scheduling, boundary copies and FARF logging",
                critical_slice=critical["slice"],
                critical_worker_ms=1000 * total / critical["qtimer_hz"],
                phase_ms={
                    name: 1000 * critical[name + "_ticks"] / critical["qtimer_hz"]
                    for name in PHASES
                },
                phase_fraction={
                    name: critical[name + "_ticks"] / total for name in PHASES
                },
                vector_reductions=sum(
                    record["vector_reductions"] for record in invocation
                ),
                integer_updates=sum(record["integer_updates"] for record in invocation),
                records=invocation,
            )
        )
    return dict(
        caveat="Instrumented diagnostics; compare production latency with profiling disabled. Worker intervals are parallel and must not be summed as wall time. These are elapsed intervals, not DDR or instruction issue utilization counters.",
        cases=cases,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logcat", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.logcat.read_text(errors="replace"))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for case in report["cases"]:
        print(case["shape"], case["critical_worker_ms"], case["phase_ms"])


if __name__ == "__main__":
    main()
