# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export and run a fully delegated FP16 HTP Add on an Android device."""

import argparse
import json
import os
import re
import shlex
import subprocess
import uuid
from pathlib import Path

import numpy as np


def export_model(artifact, report):
    import torch
    from executorch.backends.qualcomm.utils.utils import (
        generate_htp_compiler_spec,
        generate_qnn_executorch_compiler_spec,
        get_qnn_context_binary_alignment,
        get_soc_to_chipset_map,
        get_soc_to_htp_arch_map,
        to_edge_transform_and_lower_to_qnn,
    )
    from executorch.exir import ExecutorchBackendConfig
    from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass

    class Add(torch.nn.Module):
        def forward(self, x, y):
            return x + y

    base = np.arange(256, dtype=np.float32).reshape(1, 4, 8, 8)
    x = (base - 128) / 16
    y = ((base * 7) % 64 - 32) / 8
    cases = [(x, y), (-x, -y), (np.flip(x, -1), np.flip(y, -2))]
    input_lines = []
    for index, inputs in enumerate(cases):
        names = []
        for input_index, tensor in enumerate(inputs):
            assert np.array_equal(tensor, tensor.astype(np.float16).astype(np.float32))
            name = f"input_{index}_{input_index}.raw"
            tensor.tofile(artifact / name)
            names.append(name)
        expected = inputs[0] + inputs[1]
        assert np.array_equal(expected, expected.astype(np.float16).astype(np.float32))
        expected.tofile(artifact / f"expected_{index}.raw")
        input_lines.append(" ".join(names))
    (artifact / "input_list.txt").write_text("\n".join(input_lines) + "\n")

    specs = generate_qnn_executorch_compiler_spec(
        soc_model=get_soc_to_chipset_map()[report["soc"]],
        backend_options=generate_htp_compiler_spec(use_fp16=True),
        debug=True,
    )
    lowered = to_edge_transform_and_lower_to_qnn(
        Add().eval(), tuple(torch.from_numpy(t) for t in cases[0]), specs
    )
    (artifact / "delegate_graph.txt").write_text(
        str(lowered.exported_program().graph_module.graph) + "\n"
    )
    program = lowered.to_executorch(
        ExecutorchBackendConfig(
            memory_planning_pass=MemoryPlanningPass(
                alloc_graph_input=True, alloc_graph_output=True
            ),
            segment_alignment=get_qnn_context_binary_alignment(),
        )
    )
    plan = program.executorch_program.execution_plan[0]
    delegate_ids = [delegate.id for delegate in plan.delegates]
    assert delegate_ids == ["QnnBackend"], delegate_ids
    assert not plan.operators, f"Unexpected host kernels: {plan.operators}"
    with (artifact / "add_fp16.pte").open("wb") as output:
        program.write_to_file(output)
    report.update(
        status="compiled",
        shape=list(x.shape),
        io_dtype="float32",
        htp_precision="fp16",
        delegate_ids=delegate_ids,
        host_kernel_count=len(plan.operators),
        case_count=len(cases),
        htp_arch=f"v{get_soc_to_htp_arch_map()[report['soc']].value}",
    )


def run_device(args, artifact, report):
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", artifact.name)
    workspace = f"/data/local/tmp/executorch-sm8975-smoke-{name}-{uuid.uuid4().hex[:8]}"
    report["device_workspace"] = workspace
    report.pop("cases", None)

    def adb(command):
        cmd = ["adb", "-s", args.device, *command]
        result = subprocess.run(cmd, capture_output=True, text=True)
        with (artifact / "adb.log").open("a") as log:
            log.write(shlex.join(cmd) + "\n" + result.stdout + result.stderr)
        if command[0] == "shell" and "./qnn_executor_runner" in command[-1]:
            (artifact / "runner.log").write_text(result.stdout + result.stderr)
        result.check_returncode()
        return result.stdout + result.stderr

    sdk = args.sdk_root
    arch = report["htp_arch"].removeprefix("v")
    files = [
        args.build_folder / "examples/qualcomm/executor_runner/qnn_executor_runner",
        args.build_folder / "backends/qualcomm/libqnn_executorch_backend.so",
        sdk / "lib/aarch64-android/libQnnHtp.so",
        sdk / "lib/aarch64-android/libQnnSystem.so",
        sdk / "lib/aarch64-android/libQnnHtpPrepare.so",
        sdk / f"lib/aarch64-android/libQnnHtpV{arch}Stub.so",
        sdk / f"lib/hexagon-v{arch}/unsigned/libQnnHtpV{arch}Skel.so",
        artifact / "add_fp16.pte",
        artifact / "input_list.txt",
        *sorted(artifact.glob("input_*.raw")),
    ]
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
    adb(["shell", "mkdir", workspace])
    adb(["push", *map(str, files), workspace + "/"])
    adb(["shell", "mkdir", workspace + "/outputs"])
    command = (
        f"cd {shlex.quote(workspace)} && chmod +x qnn_executor_runner && "
        'export LD_LIBRARY_PATH="$PWD" && export ADSP_LIBRARY_PATH="$PWD" && '
        "./qnn_executor_runner "
        "--model_path add_fp16.pte --input_list_path input_list.txt "
        "--output_folder_path outputs --iteration 10 --warm_up 2 --log_level 5"
    )
    adb(["shell", command])
    for index in range(report["case_count"]):
        (artifact / "outputs" / f"output_{index}_0.raw").unlink(missing_ok=True)
    adb(["pull", workspace + "/outputs", str(artifact)])
    report["device"] = args.device


def verify_outputs(artifact, report):
    results = []
    for index in range(report["case_count"]):
        expected = np.fromfile(artifact / f"expected_{index}.raw", dtype=np.float32)
        output = np.fromfile(
            artifact / "outputs" / f"output_{index}_0.raw", dtype=np.float32
        )
        assert output.shape == expected.shape, (output.shape, expected.shape)
        exact = np.array_equal(output, expected)
        results.append(
            {
                "case": index,
                "elements": int(output.size),
                "max_abs_error": float(np.max(np.abs(output - expected))),
                "exact_match": exact,
            }
        )
    report["cases"] = results
    assert all(result["exact_match"] for result in results), results
    report["status"] = "passed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--device", default="A7SR026915000282")
    parser.add_argument("--build-folder", type=Path, default=Path("build-android"))
    parser.add_argument("--soc-model", default="SM8975", help="Offline compile target.")
    parser.add_argument("--host-build-folder", type=Path)
    parser.add_argument(
        "--sdk-root",
        type=Path,
        default=Path("/hunter/qnn_sdk/src/backends/qnn/2.51.40.260930"),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--compile-only", action="store_true")
    mode.add_argument("--run-only", action="store_true")
    mode.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    artifact.mkdir(parents=True, exist_ok=True)
    args.sdk_root = args.sdk_root.resolve()
    args.build_folder = args.build_folder.resolve()
    os.environ["QNN_SDK_ROOT"] = str(args.sdk_root)
    result_path = artifact / "result.json"
    if args.run_only or args.verify_only:
        report = json.loads(result_path.read_text())
    else:
        report = {
            "soc": args.soc_model,
            "sdk_root": str(args.sdk_root),
            "device": args.device,
        }
    try:
        if not (args.run_only or args.verify_only):
            if args.host_build_folder:
                import executorch.backends.qualcomm.python as qnn_python

                adaptor_dir = args.host_build_folder.resolve() / "backends/qualcomm"
                if not adaptor_dir.is_dir():
                    raise FileNotFoundError(adaptor_dir)
                qnn_python.__path__ = [str(adaptor_dir), *qnn_python.__path__]
            export_model(artifact, report)
        if not (args.compile_only or args.verify_only):
            run_device(args, artifact, report)
        if not args.compile_only:
            verify_outputs(artifact, report)
        report.pop("error", None)
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        result_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
