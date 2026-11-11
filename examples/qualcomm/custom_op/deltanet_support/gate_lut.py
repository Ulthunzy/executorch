# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Generate calibrated AR1 decay tables with the target SDK and cache them."""

import hashlib
import json
import logging
import operator
import os
import shlex
import subprocess
import tempfile
import uuid
from contextlib import ExitStack
from pathlib import Path

import torch

from .decode_gate import gate_activation, gate_signature


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _export_gate(module, activation, output, folder, soc_model):
    from executorch.backends.qualcomm._passes import TagQuantIO
    from executorch.backends.qualcomm._passes.build_quant_io import BuildQuantIo
    from executorch.backends.qualcomm._passes.qnn_pass_manager import (
        get_qnn_pass_manager_cls,
    )
    from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset
    from executorch.backends.qualcomm.utils.constants import (
        QCOM_PASS_ACTIVATE_KEY,
        QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY,
    )
    from executorch.backends.qualcomm.utils.utils import (
        dump_context_from_pte,
        generate_htp_compiler_spec,
        generate_qnn_executorch_compiler_spec,
        to_edge_transform_and_lower_to_qnn,
    )
    from executorch.exir import ExecutorchBackendConfig
    from executorch.exir.delegate import executorch_call_delegate

    graph = torch.fx.Graph()
    values = {activation: graph.placeholder("activation")}

    def clone(node):
        if node not in values:
            values[node] = graph.node_copy(node, clone)
        return values[node]

    graph.output(clone(output))
    gate = torch.fx.GraphModule(module, graph)
    grid = activation.args[1:]

    class Gate(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = gate

        def forward(self, value):
            q = torch.ops.quantized_decomposed.quantize_per_tensor.default
            dq = torch.ops.quantized_decomposed.dequantize_per_tensor.default
            return self.gate(dq(q(value, *grid), *grid))

    spec = generate_qnn_executorch_compiler_spec(
        QcomChipset[soc_model],
        generate_htp_compiler_spec(use_fp16=False),
        shared_buffer=True,
    )
    manager = get_qnn_pass_manager_cls()
    passes = manager.get_capture_program_passes()
    passes[TagQuantIO][QCOM_PASS_ACTIVATE_KEY] = True
    passes[TagQuantIO][QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY]["get_quant_io_dtype_fn"] = (
        lambda node: torch.uint16
    )
    sample = torch.zeros(tuple(activation.meta["val"].shape))
    edge = to_edge_transform_and_lower_to_qnn(
        Gate(),
        (sample,),
        spec,
        passes_job=passes,
        dep_table=manager.get_passes_dependency_for_capture_program(),
    )
    calls = [
        n.target for n in edge.exported_program().graph.nodes if n.op == "call_function"
    ]
    if calls.count(executorch_call_delegate) != 1 or any(
        target not in (executorch_call_delegate, operator.getitem) for target in calls
    ):
        raise RuntimeError("Gate table generation requires one complete QNN graph")
    program = edge.to_executorch(ExecutorchBackendConfig(passes=[BuildQuantIo()]))
    path = folder / "gate.pte"
    with path.open("wb") as stream:
        program.write_to_file(stream)
    dump_context_from_pte(str(path))
    return folder / "forward_0.bin"


class _DeviceGenerator:
    def __init__(self, directory, sdk, arch, device, host):
        self.adb = ["adb"]
        if host:
            self.adb += ["-H", host]
        if device:
            self.adb += ["-s", device]
        self.remote = "/data/local/tmp/executorch-gate-lut-" + uuid.uuid4().hex
        self.directory, self.sdk, self.arch = directory, sdk, arch

    def call(self, *args, log):
        subprocess.run(
            [*self.adb, *args], check=True, stdout=log, stderr=subprocess.STDOUT
        )

    def __enter__(self):
        ndk = Path(os.environ["ANDROID_NDK_ROOT"])
        compiler = (
            ndk
            / "toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android28-clang++"
        )
        runner = self.directory / "gate_lut"
        with (self.directory / "generator.log").open("w") as log:
            self.call("get-state", log=log)
            subprocess.run(
                [
                    str(compiler),
                    "-std=c++17",
                    "-O2",
                    "-static-libstdc++",
                    "-I" + str(self.sdk / "include/QNN"),
                    str(Path(__file__).with_suffix(".cpp")),
                    "-L" + str(self.sdk / "lib/aarch64-android"),
                    "-lQnnHtp",
                    "-lQnnSystem",
                    "-Wl,--allow-shlib-undefined",
                    "-o",
                    str(runner),
                ],
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            self.call("shell", "mkdir -p " + self.remote, log=log)
            try:
                libraries = [
                    self.sdk / "lib/aarch64-android" / name
                    for name in (
                        "libQnnHtp.so",
                        "libQnnSystem.so",
                        "libQnnHtpPrepare.so",
                        f"libQnnHtpV{self.arch}Stub.so",
                    )
                ]
                libraries.append(
                    self.sdk
                    / f"lib/hexagon-v{self.arch}/unsigned/libQnnHtpV{self.arch}Skel.so"
                )
                self.call(
                    "push",
                    str(runner),
                    *map(str, libraries),
                    self.remote + "/",
                    log=log,
                )
                self.call("shell", "chmod 755 " + self.remote + "/gate_lut", log=log)
            except BaseException:
                self.cleanup(log)
                raise
        return self

    def cleanup(self, log):
        result = subprocess.run(
            [*self.adb, "shell", "rm -rf " + self.remote],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        if result.returncode:
            logging.warning("Could not remove temporary gate workspace %s", self.remote)

    def __exit__(self, *exc):
        with (self.directory / "generator.log").open("a") as log:
            self.cleanup(log)

    def generate(self, context, output, log):
        self.call("push", str(context), self.remote + "/gate.bin", log=log)
        environment = (
            f"export LD_LIBRARY_PATH={shlex.quote(self.remote)}\n"
            "export ADSP_LIBRARY_PATH="
            + shlex.quote(
                self.remote + ";/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/vendor/dsp;/dsp"
            )
            + "\n"
        )
        command = [
            self.remote + "/gate_lut",
            self.remote + "/gate.bin",
            self.remote + "/table.bin",
        ]
        self.call("shell", environment + shlex.join(command), log=log)
        self.call("pull", self.remote + "/table.bin", str(output), log=log)


def prepare_gate_tables(module, directory, soc_model, device=None, host=None):
    """Reuse matching tables or generate exact SDK AR1 tables on the target."""
    from executorch.backends.qualcomm.serialization.qc_schema import (
        _soc_info_table,
        QcomChipset,
    )

    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    sdk = Path(os.environ["QNN_SDK_ROOT"])
    arch = _soc_info_table[QcomChipset[soc_model]].htp_info.htp_arch
    identity = dict(
        soc=soc_model,
        htp=_digest(sdk / "lib/aarch64-android/libQnnHtp.so"),
        skel=_digest(sdk / f"lib/hexagon-v{arch}/unsigned/libQnnHtpV{arch}Skel.so"),
        compiler=_digest(sdk / "lib/x86_64-linux-clang/libQnnHtp.so"),
        prepare=_digest(sdk / "lib/aarch64-android/libQnnHtpPrepare.so"),
        generator=_digest(Path(__file__)),
        runner=_digest(Path(__file__).with_suffix(".cpp")),
    )
    cores = [
        n
        for n in module.graph.nodes
        if str(n.target) == "qnn_custom_ops.deltanet_core.default"
    ]
    if not cores:
        raise ValueError("Expected GDN decode cores before gate fusion")
    entries = []
    with ExitStack() as stack:
        generator = None
        for ordinal, core in enumerate(cores):
            output = core.args[3]
            activation = gate_activation(output)
            shape = tuple(activation.meta["val"].shape)
            if len(shape) != 3 or shape[:2] != (1, 1) or not 0 < shape[2] <= 256:
                raise ValueError("Decode gate tables require the calibrated AR1 graph")
            for node in (activation, output):
                if str(
                    node.target
                ) != "quantized_decomposed.dequantize_per_tensor.default" or node.args[
                    3:5
                ] != (0, 65535):
                    raise ValueError("Decode gate tables require calibrated U16 IO")
            entry = dict(
                heads=shape[2],
                input_grid=list(activation.args[1:-1]),
                output_grid=list(output.args[1:-1]),
                signature=gate_signature(module, activation, output),
            )
            key = hashlib.sha256(
                json.dumps([identity, entry], sort_keys=True).encode()
            ).hexdigest()
            cache = directory / "cache" / key
            table, metadata = cache / "table.bin", cache / "entry.json"
            expected_bytes = entry["heads"] * 65536 * 2
            try:
                cached = json.loads(metadata.read_text())
                valid = (
                    isinstance(cached, dict)
                    and table.stat().st_size == expected_bytes
                    and cached.get("sha256") == _digest(table)
                )
            except (OSError, ValueError):
                valid = False
            if not valid:
                if generator is None:
                    logging.info(
                        "Generating GDN decode tables on the target; log: %s",
                        directory / "generator.log",
                    )
                    generator = stack.enter_context(
                        _DeviceGenerator(directory, sdk, arch, device, host)
                    )
                cache.mkdir(parents=True, exist_ok=True)
                logging.info(
                    "Generating SDK AR1 decode gate table %d/%d",
                    ordinal + 1,
                    len(cores),
                )
                with tempfile.TemporaryDirectory(dir=cache) as temporary:
                    temporary = Path(temporary)
                    context = _export_gate(
                        module, activation, output, temporary, soc_model
                    )
                    with (cache / "generate.log").open("w") as log:
                        generator.generate(context, temporary / "table.bin", log)
                    if (temporary / "table.bin").stat().st_size != expected_bytes:
                        raise RuntimeError("SDK gate table has an unexpected size")
                    (temporary / "table.bin").replace(table)
                    cached = dict(sha256=_digest(table), identity=identity)
                    temp_meta = temporary / "entry.json"
                    temp_meta.write_text(json.dumps(cached, indent=2) + "\n")
                    temp_meta.replace(metadata)
            entry.update(
                file=str(table.relative_to(directory)), sha256=cached["sha256"]
            )
            entries.append(entry)
    index = dict(version=1, ar_len=1, packing="u16le", sdk=identity, layers=entries)
    with tempfile.NamedTemporaryFile(mode="w", dir=directory, delete=False) as stream:
        json.dump(index, stream, indent=2)
        stream.write("\n")
    Path(stream.name).replace(directory / "index.json")
    return directory
