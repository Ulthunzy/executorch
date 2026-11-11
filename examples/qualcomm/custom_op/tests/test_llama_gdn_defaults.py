# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from executorch.backends.qualcomm.quantizer.custom_annotation import annotate_kv_8bit
from executorch.backends.qualcomm.serialization.qc_schema import (
    QnnExecuTorchOpPackageOptions,
)
from executorch.examples.qualcomm.oss_scripts.llama import gdn
from executorch.examples.qualcomm.oss_scripts.llama.llama import _build_parser
from executorch.examples.qualcomm.oss_scripts.llama.static_llm_quant_recipe import (
    Qwen3_5_0_8BQuantRecipe,
    Qwen3_5_2BQuantRecipe,
)
from executorch.examples.qualcomm.oss_scripts.llama.wrappers import llm_wrappers


def test_removed_implementation_switches():
    parser = _build_parser()
    for option in (
        "--attention_core",
        "--attention_kv_bits",
        "--gdn_state_update",
        "--gdn_decode_lut_dir",
    ):
        assert option not in parser._option_string_actions


def test_default_package_registers_gated_decode(tmp_path, monkeypatch):
    host = tmp_path / "build/x86_64-linux-clang/libQnnDeltaNetOpPackage.so"
    host.parent.mkdir(parents=True)
    host.touch()
    monkeypatch.setattr(gdn.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        gdn,
        "setup_op_package",
        lambda *args, **kwargs: (QnnExecuTorchOpPackageOptions(), []),
    )
    options, _ = gdn.setup_gdn_op_package(
        SimpleNamespace(soc_model="SM8850", gdn_op_package_dir=str(tmp_path))
    )
    gated = [
        info
        for info in options.op_package_infos
        if info.qnn_op_type_name == "DeltaNetCoreGatedVtcm"
    ]
    assert len(gated) == 2
    assert all(
        info.custom_op_name == "qnn_custom_ops.deltanet_core_gated.default"
        for info in gated
    )


@pytest.mark.parametrize(
    "recipe_type", [Qwen3_5_0_8BQuantRecipe, Qwen3_5_2BQuantRecipe]
)
def test_recipe_selects_kv_precision(recipe_type):
    recipe = recipe_type()
    assert recipe.get_kv_io_bit_width() == 8
    recipe.recipe.custom_quant_annotations.remove(annotate_kv_8bit)
    assert recipe.get_kv_io_bit_width() == 16


@pytest.mark.parametrize(
    "soc,types,sink,online,fused",
    [
        ("SM8850", ["linear_attention", "full_attention"], None, True, True),
        ("SM8850", ["linear_attention", "sliding_attention"], None, False, True),
        ("SM8850", ["full_attention", "sliding_attention"], None, False, False),
        ("SM8850", ["linear_attention"], None, False, True),
        ("SM8850", ["full_attention"], "4,32", False, False),
        ("SM8850", ["full_attention"], None, True, False),
        ("SM8750", ["linear_attention", "full_attention"], None, False, False),
    ],
)
def test_hardware_and_attention_defaults(soc, types, sink, online, fused):
    args = SimpleNamespace(soc_model=soc, use_attention_sink=sink)
    base = SimpleNamespace(model_args=SimpleNamespace(layer_types=types))
    with patch.object(llm_wrappers, "TextDecoder", return_value=base):
        decoder = llm_wrappers.Qwen35TextDecoder(args, None)
    assert decoder.online_attention is online
    assert decoder.fused_gdn is fused
