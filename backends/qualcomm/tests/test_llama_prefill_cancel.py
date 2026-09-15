# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import torch
from executorch.exir import ExecutorchBackendConfig, to_edge
from executorch.exir.passes import MemoryPlanningPass


def test_native_prefill_cancel(tmp_path):
    root = Path(__file__).resolve().parents[3]
    build = Path(os.environ.get("EXECUTORCH_HOST_BUILD", root / "build-x86"))
    compiler = shutil.which("c++")
    libraries = [
        build / "extension/module/libextension_module.a",
        build / "extension/tensor/libextension_tensor.a",
        build / "extension/data_loader/libextension_data_loader.a",
        build / "extension/flat_tensor/libextension_flat_tensor.a",
        build / "extension/named_data_map/libextension_named_data_map.a",
        build / "extension/evalue_util/libextension_evalue_util.a",
        build / "extension/llm/sampler/libextension_llm_sampler.a",
        build / "kernels/portable/libportable_kernels.a",
        build / "libexecutorch.a",
        build / "libexecutorch_core.a",
    ]
    ops = build / "kernels/portable/libportable_ops_lib.a"
    if compiler is None or not all(p.exists() for p in [*libraries, ops]):
        pytest.skip("Build host module, sampler and portable kernels first")

    class Fixture(torch.nn.Module):
        def forward(self, tokens, mask, window_mask, pos, k, v):
            values = tokens.to(torch.float32).unsqueeze(1).unsqueeze(-1)
            values = values.expand(1, 1, tokens.shape[1], 2).clone()
            return (
                tokens.to(torch.float32).unsqueeze(-1).expand(-1, -1, 16).clone(),
                values.permute(0, 1, 3, 2).contiguous(),
                values,
            )

    methods = {}
    for ar in (4, 128):
        inputs = (
            torch.zeros(1, ar, dtype=torch.int64),
            torch.zeros(1, 1, ar, 2048),
            torch.zeros(1, 1, ar, 511 + ar),
            torch.zeros(1, ar, dtype=torch.int32),
            torch.zeros(1, 1, 2, 511),
            torch.zeros(1, 1, 511, 2),
        )
        methods[f"ar{ar}"] = torch.export.export(Fixture(), inputs)
    pte = to_edge(methods).to_executorch(
        config=ExecutorchBackendConfig(
            memory_planning_pass=MemoryPlanningPass(
                alloc_graph_input=False, alloc_graph_output=False
            )
        )
    )
    fixture = tmp_path / "prefill_cancel.pte"
    fixture.write_bytes(pte.buffer)
    binary = tmp_path / "prefill_cancel_test"
    runner = root / "examples/qualcomm/oss_scripts/llama/runner"
    sources = [
        runner / "test/prefill_cancel_test.cpp",
        runner / "prompt_processor.cpp",
        runner / "kv_manager.cpp",
        runner / "decoder_runner.cpp",
        runner / "attention_sink_rope_runner.cpp",
    ]
    compiled = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O1",
            "-g",
            "-fsanitize=address,undefined",
            "-fno-omit-frame-pointer",
            "-DET_EVENT_TRACER_ENABLED",
            "-DC10_USING_CUSTOM_GENERATED_MACROS",
            f"-I{root.parent}",
            f"-I{build / 'include'}",
            f"-I{build / 'include/executorch/runtime/core/portable_type/c10'}",
            *map(str, sources),
            "-Wl,--whole-archive",
            str(ops),
            "-Wl,--no-whole-archive",
            "-Wl,--start-group",
            *map(str, libraries),
            "-Wl,--end-group",
            "-lpthread",
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    result = subprocess.run([str(binary), str(fixture)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SWA continuation passed" in result.stdout
