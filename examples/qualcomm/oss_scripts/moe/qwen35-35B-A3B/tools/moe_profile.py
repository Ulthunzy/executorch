"""Summarize host and DSP timings without adding overlapping worker times."""

import argparse
import json
import math
import re
from pathlib import Path


def summarize(text):
    records = {
        name: []
        for name in (
            "MOE_PHASE_PROFILE",
            "MOE_NATIVE_PROFILE",
            "CHAT_DECODE_STEP",
        )
    }
    for line in text.splitlines():
        for name in records:
            if name + " " not in line:
                continue
            fields = dict(re.findall(r"(\w+)=([^\s]+)", line.split(name + " ", 1)[1]))
            for key, value in fields.items():
                if key == "phase":
                    continue
                try:
                    number = float(value)
                except ValueError as error:
                    raise ValueError(
                        f"{name}: invalid numeric field {key}={value}"
                    ) from error
                if not math.isfinite(number) or number < 0:
                    raise ValueError(f"{name}: invalid numeric field {key}={value}")
                fields[key] = number
            records[name].append(fields)
    output = dict(
        decode_steps=len(records["CHAT_DECODE_STEP"]),
        decode_positions=[row.get("position") for row in records["CHAT_DECODE_STEP"]],
        host={},
        dsp={},
        notes=[
            "Decode GDN graphs include fused Router/shared computation.",
            "Native execution overlaps expert IO; worker sums and host wall times are not additive.",
            "HMX duty is busy time / HMX service-loop time, not percent of peak arithmetic throughput.",
        ],
    )
    for phase in ("prefill", "prefill_append", "decode"):
        rows = (
            records["CHAT_DECODE_STEP"]
            if phase == "decode"
            else [
                row for row in records["MOE_PHASE_PROFILE"] if row.get("phase") == phase
            ]
        )
        if not rows:
            continue
        keys = set().union(*(row.keys() for row in rows)) - {
            "phase",
            "chunk",
            "position",
            "input_id",
        }
        totals = {
            key: sum(row.get(key, 0) for row in rows)
            for key in sorted(keys)
            if all(isinstance(row.get(key, 0), (int, float)) for row in rows)
        }
        if phase == "decode":
            if totals.get("total_ms", 0):
                totals["gdn_graph_wall_fraction"] = (
                    totals.get("gdn_graph_ms", 0) / totals["total_ms"]
                )
                totals["expert_bytes_per_decode_second"] = (
                    totals.get("direct_bytes", 0) * 1000 / totals["total_ms"]
                )
            if totals.get("cache_requests", 0):
                totals["cache_hit_fraction"] = (
                    totals.get("cache_hits", 0) / totals["cache_requests"]
                )
        else:
            if totals.get("io_wall_ms", 0):
                totals["io_overlap_fraction"] = (
                    totals.get("io_overlap_ms", 0) / totals["io_wall_ms"]
                )
            if totals.get("total_ms", 0):
                totals["gdn_attention_wall_fraction"] = (
                    totals.get("gdn_attention_ms", 0) / totals["total_ms"]
                )
        output["host"][phase] = totals
    for row in records["MOE_NATIVE_PROFILE"]:
        phase = row.get("phase")
        if phase not in ("prefill", "decode"):
            continue
        if phase in output["dsp"]:
            raise ValueError(
                f"{phase}: multiple cumulative DSP totals; analyze one session at a time"
            )
        summary = {key: value for key, value in row.items() if key != "phase"}
        if summary.get("enabled") != 1:
            raise ValueError(
                f"{phase}: DSP profiling was not enabled in the native build"
            )
        if summary.get("hmx_loop_ms", 0):
            if "hmx_busy_ms" not in summary:
                raise ValueError(f"{phase}: missing HMX busy time")
            summary["hmx_service_duty_fraction"] = (
                summary["hmx_busy_ms"] / summary["hmx_loop_ms"]
            )
        output["dsp"][phase] = summary
    if not output["host"] and not output["dsp"]:
        raise ValueError("No MoE profiling records found")
    return output


def compare(baseline, candidate):
    if baseline["decode_steps"] != candidate["decode_steps"]:
        raise ValueError("Compare logs with the same number of decode steps")
    if baseline["decode_positions"] != candidate["decode_positions"]:
        raise ValueError("Compare logs with the same decode context positions")
    changes = {}
    for source in ("host", "dsp"):
        for phase in candidate[source].keys() & baseline[source].keys():
            before, after = baseline[source][phase], candidate[source][phase]
            if before.get("tokens") != after.get("tokens"):
                raise ValueError("Compare logs with the same prefill token count")
            for key in before.keys() & after.keys():
                if key.endswith("_ms") and before[key] > 0:
                    changes[f"{source}.{phase}.{key}"] = dict(
                        baseline=before[key],
                        candidate=after[key],
                        speedup=before[key] / after[key] if after[key] > 0 else None,
                    )
    return changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = summarize(args.log.read_text())
        if args.baseline:
            report["comparison"] = compare(summarize(args.baseline.read_text()), report)
        data = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(data)
        else:
            print(data, end="")
    except (OSError, ValueError) as error:
        parser.exit(1, f"Profile analysis failed: {error}\n")


if __name__ == "__main__":
    main()
