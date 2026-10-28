#!/usr/bin/env python3
# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Measure the same isolated GDN graph with two interchangeable DSP libraries.

Rates use a recurrence operation count and logical traffic estimates, not PMU
counters. In particular, logical state accesses may hit cache or VTCM and must
not be interpreted as measured DDR bandwidth or hardware utilization.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import time


MODEL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MODEL_ROOT.parents[4]
DEFAULT_SDK = Path("/hunter/qnn_sdk/src/backends/qnn/2.51.200.261021")
DEFAULT_PACKAGE = (
    REPO_ROOT / "examples/qualcomm/custom_op/deltanet_op_package_htp/DeltaNetOpPackage"
)
LIBRARY = "libQnnDeltaNetOpPackage_HTP.so"


def traffic_model(batch, tokens, heads, key_dim, value_dim, element_bytes=2):
    if min(batch, tokens, heads, key_dim, value_dim, element_bytes) <= 0:
        raise ValueError("Shape dimensions and element size must be positive")
    state_elements = batch * heads * key_dim * value_dim
    vector_elements = batch * tokens * heads * (2 * key_dim + 2 * value_dim + 2)
    external_bytes = element_bytes * (2 * state_elements + vector_elements)
    loop_bytes = element_bytes * (3 * tokens * state_elements + vector_elements)
    # Decay S, k^T S, delta, rank-one update, and q^T S; FMA counts as two.
    operations = batch * tokens * heads * (7 * key_dim * value_dim + 2 * value_dim)
    return {
        "state_bytes": state_elements * element_bytes,
        "minimum_external_bytes": external_bytes,
        "estimated_loop_bytes": loop_bytes,
        "mathematical_operations": operations,
        "loop_arithmetic_intensity_ops_per_byte": operations / loop_bytes,
    }


def rates(model, latency_ms, peak_gbps=None, peak_gops=None):
    if not math.isfinite(latency_ms) or latency_ms <= 0:
        raise ValueError("Latency must be finite and positive")
    result = {
        "estimated_loop_GB_per_s": model["estimated_loop_bytes"] / latency_ms / 1e6,
        "minimum_external_GB_per_s": model["minimum_external_bytes"] / latency_ms / 1e6,
        "mathematical_Gops_per_s": model["mathematical_operations"] / latency_ms / 1e6,
    }
    if peak_gbps is not None:
        if not math.isfinite(peak_gbps) or peak_gbps <= 0:
            raise ValueError("Calibrated bandwidth must be finite and positive")
        result["loop_rate_over_calibrated_bandwidth"] = (
            result["estimated_loop_GB_per_s"] / peak_gbps
        )
    if peak_gops is not None:
        if not math.isfinite(peak_gops) or peak_gops <= 0:
            raise ValueError(
                "Calibrated compute throughput must be finite and positive"
            )
        result["math_rate_over_calibrated_compute"] = (
            result["mathematical_Gops_per_s"] / peak_gops
        )
    if peak_gbps is not None and peak_gops is not None:
        result["estimated_loop_roofline_lower_bound_ms"] = max(
            model["estimated_loop_bytes"] / peak_gbps / 1e6,
            model["mathematical_operations"] / peak_gops / 1e6,
        )
    return result


def parse_latency(output):
    matches = re.findall(
        r"(?m)^.*?(?<!Total\s)\b\d+\s+inferences?\s+took\s+[0-9.]+\s+ms,"
        r"\s+avg\s+([0-9.]+)\s+ms",
        output,
    )
    if not matches:
        raise RuntimeError("Runner did not report an inference latency")
    latency = float(matches[-1])
    if not math.isfinite(latency) or latency <= 0:
        raise RuntimeError(f"Invalid runner latency: {latency}")
    return latency


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def output_hashes(directory):
    paths = sorted(directory.glob("*.raw"))
    if len(paths) != 2:
        raise RuntimeError(f"Expected attention and state outputs in {directory}")
    return {
        path.name: {"bytes": path.stat().st_size, "sha256": file_hash(path)}
        for path in paths
    }


def compare_outputs(
    directory, baseline_directory, reference_directory, output_specs, atol, rtol
):
    import numpy as np

    metrics = []
    compatible = True
    for index, spec in enumerate(output_specs):
        filename = Path(spec["file"]).name
        dtype = np.dtype(spec["dtype"])
        actual = np.fromfile(directory / filename, dtype=dtype).astype(np.float64)
        baseline = np.fromfile(baseline_directory / filename, dtype=dtype).astype(
            np.float64
        )
        reference = np.fromfile(
            reference_directory / f"output_{index}.raw", dtype=dtype
        ).astype(np.float64)
        if actual.size != reference.size or baseline.size != reference.size:
            raise RuntimeError(f"Output size mismatch: {filename}")
        finite = bool(np.isfinite(actual).all())
        difference = actual - baseline
        error = actual - reference
        noise = float(np.sum(error * error))
        signal = float(np.sum(reference * reference))
        compatible &= finite and bool(
            np.allclose(actual, baseline, atol=atol, rtol=rtol)
        )
        metrics.append(
            {
                "output": filename,
                "finite": finite,
                "max_abs_diff_from_baseline": (
                    float(np.max(np.abs(difference))) if finite else None
                ),
                "rmse_from_baseline": (
                    float(np.sqrt(np.mean(difference * difference))) if finite else None
                ),
                "reference_rmse": (
                    float(np.sqrt(np.mean(error * error))) if finite else None
                ),
                "reference_sqnr_db": (
                    10 * math.log10(signal / noise)
                    if signal > 0 and noise > 0
                    else None
                ),
            }
        )
    return compatible, metrics


def run_command(command, log_path=None, env=None):
    result = subprocess.run(
        command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env
    )
    if log_path is not None:
        log_path.write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {shlex.join(command)}\n{result.stdout[-6000:]}"
        )
    return result.stdout


def device_snapshot(adb):
    return {
        "unix_time": time.time(),
        "soc": run_command([*adb, "shell", "getprop", "ro.soc.model"]).strip(),
        "thermalservice": run_command([*adb, "shell", "dumpsys", "thermalservice"]),
        "battery": run_command([*adb, "shell", "dumpsys", "battery"]),
    }


def analyze(report, peak_gbps=None, peak_gops=None):
    for case in report["cases"]:
        model = traffic_model(**case["shape"])
        case["traffic_model"] = model
        for variant in case["variants"].values():
            samples = variant["round_avg_ms"]
            variant["median_ms"] = statistics.median(samples)
            variant["min_ms"] = min(samples)
            variant["max_ms"] = max(samples)
            variant["stdev_ms"] = statistics.stdev(samples) if len(samples) > 1 else 0
            variant["rates"] = rates(model, variant["median_ms"], peak_gbps, peak_gops)
        case["speedup"] = (
            case["variants"]["baseline"]["median_ms"]
            / case["variants"]["optimized"]["median_ms"]
        )
    return report


def run_benchmark(args):
    if min(args.iterations, args.rounds) <= 0 or args.warm_up < 0:
        raise ValueError(
            "Iterations/rounds must be positive; warm-up must be nonnegative"
        )
    if any(not math.isfinite(value) or value < 0 for value in (args.atol, args.rtol)):
        raise ValueError("Output tolerances must be finite and nonnegative")
    if args.calibration_samples <= 0:
        raise ValueError("Calibration samples must be positive")
    if args.state_vtcm and args.quant_dtype == "fp16":
        raise ValueError("VTCM-state variant requires QUInt16 activations")
    for tokens in args.ar_lens:
        traffic_model(1, tokens, args.heads, args.key_dim, args.value_dim)
    args.artifact.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["QNN_SDK_ROOT"] = str(args.sdk.resolve(strict=True))
    env["LD_LIBRARY_PATH"] = (
        str(args.sdk / "lib/x86_64-linux-clang") + ":" + env.get("LD_LIBRARY_PATH", "")
    )
    env["PYTHONPATH"] = str(REPO_ROOT.parent) + ":" + env.get("PYTHONPATH", "")
    adb = ["adb", "-s", args.device]
    libraries = {}
    for name, source in (
        ("baseline", args.baseline_library),
        ("optimized", args.optimized_library),
    ):
        source = source.resolve(strict=True)
        destination = args.artifact / "libraries" / name / LIBRARY
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        libraries[name] = {
            "path": str(destination),
            "sha256": file_hash(destination),
            "source": str(source),
        }
    report = {
        "soc": args.soc,
        "device": args.device,
        "device_environment": device_snapshot(adb),
        "sdk": str(args.sdk.resolve()),
        "quant_dtype": args.quant_dtype,
        "share_state_qparams": not args.independent_state_qparams,
        "state_vtcm": args.state_vtcm,
        "scope": "isolated GDN core graph; host-precomputed Q/K normalization and exp",
        "timing_scope": "runner method average including graph dispatch and boundary copies; not internal worker timing",
        "rate_caveat": "Logical traffic and mathematical ops are estimates, not DDR/PMU counters. Calibrated rates must match dtype, clocks and memory placement; ratios do not prove saturation.",
        "warm_up": args.warm_up,
        "iterations_per_round": args.iterations,
        "rounds": args.rounds,
        "seed": args.seed,
        "calibration_samples": args.calibration_samples,
        "exporter_sha256": file_hash(
            Path(__file__).with_name("gdn_benchmark_export.py")
        ),
        "workspace": args.workspace,
        "libraries": libraries,
        "cases": [],
    }
    driver = Path(__file__).with_name("gdn_benchmark_export.py")
    for tokens in args.ar_lens:
        case_dir = args.artifact / f"ar{tokens}"
        case_dir.mkdir(exist_ok=True)
        workspace = f"{args.workspace}/ar{tokens}"
        command = [
            sys.executable,
            str(driver),
            "--build_folder",
            str(args.build_folder),
            "--device",
            args.device,
            "--soc_model",
            args.soc,
            "--backend",
            "htp",
            "--ar_len",
            str(tokens),
            "--heads",
            str(args.heads),
            "--head_k_dim",
            str(args.key_dim),
            "--head_v_dim",
            str(args.value_dim),
            "--variants",
            "core_only",
            "--quant_dtypes",
            args.quant_dtype,
            "--input_dtype",
            "fp16" if args.quant_dtype == "fp16" else "fp32",
            "--iteration",
            "1",
            "--warm_up",
            str(args.warm_up),
            "--profile_level",
            "0",
            "--calibration_samples",
            str(args.calibration_samples),
            "--seed",
            str(args.seed),
            "--op_package_dir",
            str(args.op_package_dir),
            "--workspace",
            workspace,
            "-a",
            str(case_dir / "export"),
        ]
        if args.qnn_binding:
            command += ["--qnn-binding", str(args.qnn_binding.resolve(strict=True))]
        if args.independent_state_qparams:
            command += ["--independent-state-qparams"]
        if args.state_vtcm:
            command += ["--state-vtcm"]
        if args.reuse_prepared_artifact:
            source_report = json.loads(
                (args.reuse_prepared_artifact / "comparison.json").read_text()
            )
            if (
                source_report["quant_dtype"] != args.quant_dtype
                or source_report["share_state_qparams"]
                != (not args.independent_state_qparams)
                or source_report.get("state_vtcm", False) != args.state_vtcm
                or source_report["soc"] != args.soc
                or source_report["sdk"] != str(args.sdk.resolve())
                or source_report.get("seed", 666) != args.seed
                or source_report.get("calibration_samples", 2)
                != args.calibration_samples
            ):
                raise ValueError(
                    "Prepared artifact must use the same dtype and state encoding"
                )
            source_case = next(
                case
                for case in source_report["cases"]
                if case["shape"]["tokens"] == tokens
            )
            if source_case["shape"] != {
                "batch": 1,
                "tokens": tokens,
                "heads": args.heads,
                "key_dim": args.key_dim,
                "value_dim": args.value_dim,
            }:
                raise ValueError("Prepared artifact shape does not match")
            shutil.copytree(
                args.reuse_prepared_artifact / f"ar{tokens}/export",
                case_dir / "export",
                dirs_exist_ok=True,
            )
            prepared_pte = next((case_dir / "export").glob("*/*.pte"))
            device_hash = run_command(
                [*adb, "shell", "sha256sum", f"{workspace}/{prepared_pte.name}"]
            ).split()[0]
            if device_hash != file_hash(prepared_pte):
                raise ValueError(
                    "Device PTE differs from the prepared artifact; prepare again"
                )
            print(
                f"Reusing AR{tokens} prepared graph and existing device inputs",
                flush=True,
            )
        else:
            print(f"Preparing AR{tokens} isolated GDN graph", flush=True)
            run_command(command, case_dir / "export.log", env=env)
        exported = json.loads((case_dir / "export/summary.json").read_text())[0]
        if len(exported.get("outputs", [])) != 2 or any(
            "error" in out for out in exported["outputs"]
        ):
            raise RuntimeError(f"Missing reference output checks for AR{tokens}")
        pte = next((case_dir / "export").glob("*/*.pte"))
        case = {
            "shape": {
                "batch": 1,
                "tokens": tokens,
                "heads": args.heads,
                "key_dim": args.key_dim,
                "value_dim": args.value_dim,
            },
            "pte_sha256": file_hash(pte),
            "reference_checks": exported["outputs"],
            "variants": {
                name: {"round_avg_ms": [], "environment": []} for name in libraries
            },
        }
        baseline_hashes = None
        exact_match = True
        numerically_compatible = True
        baseline_directory = case_dir / "outputs-baseline-0"
        for round_index in range(args.rounds):
            order = (
                ("baseline", "optimized")
                if round_index % 2 == 0
                else ("optimized", "baseline")
            )
            for name in order:
                environment = {"before": device_snapshot(adb)}
                run_command(
                    [*adb, "push", libraries[name]["path"], f"{workspace}/{LIBRARY}"]
                )
                remote_output = f"{workspace}/measure-{name}-{round_index}"
                runner_args = [
                    "./qnn_executor_runner",
                    "--model_path",
                    pte.name,
                    "--input_list_path",
                    "input_list.txt",
                    "--output_folder_path",
                    remote_output,
                    "--warm_up",
                    str(args.warm_up),
                    "--iteration",
                    str(args.iterations),
                ]
                shell = f"cd {shlex.quote(workspace)} && export LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. && mkdir -p {shlex.quote(remote_output)} && {shlex.join(runner_args)}"
                output = run_command(
                    [*adb, "shell", shell], case_dir / f"{name}-{round_index}.log"
                )
                latency = parse_latency(output)
                environment["after"] = device_snapshot(adb)
                case["variants"][name]["round_avg_ms"].append(latency)
                case["variants"][name]["environment"].append(environment)
                local_output = case_dir / f"outputs-{name}-{round_index}"
                run_command([*adb, "pull", remote_output, str(local_output)])
                hashes = output_hashes(local_output)
                if baseline_hashes is None:
                    baseline_hashes = hashes
                exact_match &= hashes == baseline_hashes
                compatible, metrics = compare_outputs(
                    local_output,
                    baseline_directory,
                    case_dir / "export/reference",
                    exported["outputs"],
                    args.atol,
                    args.rtol,
                )
                numerically_compatible &= compatible
                case["variants"][name]["output_hashes"] = hashes
                case["variants"][name]["numerical_checks"] = metrics
                print(
                    f"AR{tokens} round {round_index + 1} {name}: {latency:.5f} ms",
                    flush=True,
                )
        case["bit_exact_across_variants_and_rounds"] = exact_match
        case["within_tolerance_across_variants_and_rounds"] = numerically_compatible
        case["comparison_tolerance"] = {"atol": args.atol, "rtol": args.rtol}
        if args.capture_profile:
            profile_package = case_dir / "profile-package"
            arch = {"SM8750": 79, "SM8850": 81, "SM8975": 85}[args.soc]
            for platform in ("x86_64-linux-clang", "aarch64-android"):
                destination = profile_package / "build" / platform
                destination.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(
                    args.op_package_dir
                    / "build"
                    / platform
                    / LIBRARY.replace("_HTP", ""),
                    destination / LIBRARY.replace("_HTP", ""),
                )
            dsp_dir = profile_package / "build" / f"hexagon-v{arch}"
            dsp_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(libraries["optimized"]["path"], dsp_dir / LIBRARY)
            profile_command = command.copy()
            profile_workspace = workspace + "-profile"
            for flag, value in (
                ("--profile_level", "2"),
                ("--op_package_dir", str(profile_package)),
                ("--workspace", profile_workspace),
                ("-a", str(case_dir / "profile-export")),
            ):
                profile_command[profile_command.index(flag) + 1] = value
            run_command(profile_command, case_dir / "profile-export.log", env=env)
            etdump_path = case_dir / "profile.etdp"
            run_command(
                [*adb, "pull", profile_workspace + "/etdump.etdp", str(etdump_path)]
            )
            profile_code = (
                "import sys; from executorch.devtools import Inspector; "
                "from executorch.devtools.inspector._inspector_utils import TimeScale; "
                "Inspector(etdump_path=sys.argv[1], source_time_scale=TimeScale.CYCLES, "
                "target_time_scale=TimeScale.CYCLES).save_data_to_tsv(sys.argv[2])"
            )
            run_command(
                [
                    sys.executable,
                    "-c",
                    profile_code,
                    str(etdump_path),
                    str(case_dir / "profile-cycles.tsv"),
                ],
                case_dir / "profile-analysis.log",
                env=env,
            )
            case["separate_profile"] = {
                "etdump": str(etdump_path),
                "table": str(case_dir / "profile-cycles.tsv"),
                "scope": "Detailed profiling is a separate run, excluded from latency medians. Values retain the units in event names (cycles/us/count); the generic table header is not a unit conversion.",
            }
        report["cases"].append(case)
        analyze(report, args.peak_gbps, args.peak_gops)
        (args.artifact / "comparison.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        if not numerically_compatible or (
            args.atol == args.rtol == 0 and not exact_match
        ):
            raise RuntimeError(f"AR{tokens} outputs differ; see comparison.json")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser(
        "run", help="Export, run paired library swaps, and check exact outputs"
    )
    run.add_argument("--device", required=True)
    run.add_argument("--soc", default="SM8850", choices=("SM8750", "SM8850", "SM8975"))
    run.add_argument("--sdk", type=Path, default=DEFAULT_SDK)
    run.add_argument("--build-folder", type=Path, default=REPO_ROOT / "build-android")
    run.add_argument("--qnn-binding", type=Path)
    run.add_argument("--op-package-dir", type=Path, default=DEFAULT_PACKAGE)
    run.add_argument("--baseline-library", type=Path, required=True)
    run.add_argument("--optimized-library", type=Path, required=True)
    run.add_argument("--artifact", type=Path, required=True)
    run.add_argument(
        "--reuse-prepared-artifact",
        type=Path,
        help="Reuse a saved comparison's exported graphs and existing device workspace; checks graph SHA256 and shape",
    )
    run.add_argument(
        "--workspace", default="/data/local/tmp/executorch/gdn-paired-benchmark"
    )
    run.add_argument("--ar-lens", nargs="+", type=int, default=[1, 128])
    run.add_argument("--heads", type=int, default=32)
    run.add_argument("--key-dim", type=int, default=128)
    run.add_argument("--value-dim", type=int, default=128)
    run.add_argument(
        "--quant-dtype", default="use_16a4w", choices=("fp16", "use_16a8w", "use_16a4w")
    )
    run.add_argument("--warm-up", type=int, default=5)
    run.add_argument("--iterations", type=int, default=100)
    run.add_argument("--rounds", type=int, default=3)
    run.add_argument("--seed", type=int, default=666)
    run.add_argument("--calibration-samples", type=int, default=2)
    run.add_argument(
        "--state-vtcm",
        action="store_true",
        help="Export the VTCM-state op; this changes the graph, so compare layouts as separate exports",
    )
    run.add_argument(
        "--independent-state-qparams",
        action="store_true",
        help="Reproduce the generic benchmark's separate initial/final state encodings; default shares them like the MoE core",
    )
    run.add_argument(
        "--atol",
        type=float,
        default=0.0,
        help="Absolute tolerance vs baseline outputs; default requires exact values",
    )
    run.add_argument(
        "--rtol",
        type=float,
        default=0.0,
        help="Relative tolerance vs baseline outputs; inspect reference error for non-exact kernels",
    )
    run.add_argument(
        "--capture-profile",
        action="store_true",
        help="Capture a separate detailed QNN ETDump and per-op cycle table after timing",
    )
    analysis = subparsers.add_parser(
        "analyze",
        help="Recompute traffic/rates from saved measurements without QNN or a device",
    )
    analysis.add_argument("report", type=Path)
    analysis.add_argument("--output", type=Path)
    for command in (run, analysis):
        command.add_argument(
            "--peak-gbps",
            type=float,
            help="Calibrated bandwidth for the same memory placement and clocks",
        )
        command.add_argument(
            "--peak-gops",
            type=float,
            help="Calibrated compute throughput for the same dtype and clocks",
        )
    args = parser.parse_args()
    if args.command == "run":
        report = run_benchmark(args)
    else:
        report = analyze(
            json.loads(args.report.read_text()), args.peak_gbps, args.peak_gops
        )
    output = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.command == "analyze" and args.output:
        args.output.write_text(output)
    else:
        print(output)


if __name__ == "__main__":
    main()
