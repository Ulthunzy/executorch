# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from executorch.backends.qualcomm.quantizer.quantizer import QnnQuantizer, QuantDtype
from executorch.examples.models.llama.model_args import ModelArgs
from executorch.examples.qualcomm.oss_scripts.llama.model.static_llama import (
    CausalMaskedSoftmax,
    MultiScopeAwareLlamaModel_swa,
)
from executorch.examples.qualcomm.oss_scripts.llama.static_llm_quant_recipe import (
    Lincal_1_7BQuantRecipe,
)
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e


PREFILL_AR = int(os.environ.get("ET_MASK_QNN_PREFILL_AR", "128"))


@pytest.fixture(scope="module", autouse=True)
def deterministic_math():
    threads = torch.get_num_threads()
    tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    yield
    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    return request.param


def softmax_inputs(width=4096):
    torch.manual_seed(42)
    scores = torch.randn(1, 2, 4, width) * 8
    scores[:, :, 1].abs_()
    scores[:, :, 2] = -scores[:, :, 2].abs()
    mask = torch.full((1, 1, 4, width), -255.0)
    mask[:, :, 0, :3] = 0
    mask[:, :, 1, :17] = 0
    mask[:, :, 2, : width // 2] = 0
    return scores, mask


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_masked_softmax_excludes_scores_and_values(device, dtype):
    scores, mask = [t.to(device=device, dtype=dtype) for t in softmax_inputs()]
    allowed = mask == 0
    module = CausalMaskedSoftmax().to(device)
    expected = module(scores, mask)
    assert torch.isfinite(expected).all()
    assert torch.count_nonzero(expected.masked_select(~allowed)) == 0
    for fill in (-10000.0, 10000.0, float("nan")):
        changed = torch.where(allowed, scores, fill)
        assert torch.equal(module(changed, mask), expected)
    reference = torch.softmax(scores.float().masked_fill(~allowed, -torch.inf), -1)
    reference = reference.nan_to_num().to(dtype)
    torch.testing.assert_close(expected, reference, atol=1e-6, rtol=0.01)
    values = torch.randn_like(scores)
    changed_values = torch.where(allowed, values, 1e4)
    assert torch.equal((expected * values).sum(-1), (expected * changed_values).sum(-1))


def test_qdq_softmax_roundtrip_is_causal(device, tmp_path):
    inputs = softmax_inputs()
    quantizer = QnnQuantizer()
    quantizer.set_default_quant_config(QuantDtype.use_16a8w)
    prepared = prepare_pt2e(
        torch.export.export(CausalMaskedSoftmax(), inputs, strict=True).module(),
        quantizer,
    )
    with torch.no_grad():
        prepared(*inputs)
    converted = convert_pt2e(prepared)
    path = tmp_path / "masked-softmax.pt2"
    torch.export.save(torch.export.export(converted, inputs, strict=True), path)
    restored = torch.export.load(path).module().to(device)
    scores, mask = [t.to(device) for t in inputs]
    with torch.no_grad():
        expected = restored(scores, mask)
        assert torch.isfinite(expected).all()
        assert torch.count_nonzero(expected.masked_select(mask != 0)) == 0
        for fill in (-10000.0, 10000.0):
            assert torch.equal(
                restored(torch.where(mask == 0, scores, fill), mask), expected
            )


def make_model(
    ar, conv=True, context=256, window=128, head_dim=32, n_heads=2, n_kv_heads=2
):
    torch.manual_seed(17)
    args = ModelArgs(
        dim=n_heads * head_dim,
        n_layers=2,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        hidden_dim=4 * head_dim,
        vocab_size=128,
        max_seq_len=context,
        max_context_len=context,
        use_kv_cache=True,
        use_hf_rope=True,
        use_qk_norm=True,
        qk_norm_before_rope=True,
        local_rope_theta=10000.0,
        layer_types=["full_attention", "sliding_attention"],
        sliding_window=window,
    )
    args.enable_r3 = True
    args.enable_masked_softmax = True
    model = MultiScopeAwareLlamaModel_swa(args, ar_len=ar).eval()
    if conv:
        for layer in model.layers:
            layer.attention.prepare_attention_conv()
            layer.feed_forward.prepare_feedforward_conv()
        model.prepare_output_conv()
    tokens, masks, pos, key, value = model.get_example_inputs()
    pos.copy_(torch.arange(ar, dtype=pos.dtype))
    return model, (tokens, *masks, pos, *key, *value)


def quantize(model, inputs):
    quantizer = QnnQuantizer()
    quantizer.set_recipe(Lincal_1_7BQuantRecipe().recipe)
    prepared = prepare_pt2e(
        torch.export.export(model, inputs, strict=True).module(), quantizer
    )
    with torch.no_grad():
        prepared(*inputs)
    return convert_pt2e(prepared)


def changed_inputs(inputs, cut, fill):
    changed = [t.clone() for t in inputs]
    changed[0][:, cut:] = fill
    # The initial-prefill mask excludes every pre-existing cache slot.
    for tensor in changed[4:]:
        tensor.fill_(fill - 64.0)
    return tuple(changed)


@pytest.mark.parametrize("ar", [1, 128, 256])
@pytest.mark.parametrize("conv", [False, True])
def test_float_attention_paths_are_causal(ar, conv, device):
    model, inputs = make_model(ar, conv)
    model = model.to(device)
    inputs = tuple(t.to(device) for t in inputs)
    cut = max(1, ar // 2)
    with torch.no_grad():
        expected = model(*inputs)[0][:, :cut]
        for fill in (1, 127):
            actual = model(*changed_inputs(inputs, cut, fill))[0][:, :cut]
            assert torch.equal(actual, expected)


@pytest.fixture(scope="module")
def quantized_models():
    from executorch.examples.qualcomm.oss_scripts.llama.wrappers.llm_wrappers import (
        HybridTextDecoder,
    )

    assert (
        1 < PREFILL_AR < 256
    ), "Prefill AR must be between decode AR1 and calibration AR256"
    result = {}
    for ar in (256, 1, PREFILL_AR):
        model, inputs = make_model(ar, head_dim=128, n_heads=4)
        result[ar] = (quantize(model, inputs), inputs)
    hybrid = object.__new__(HybridTextDecoder)
    hybrid.control_args = SimpleNamespace(decoder_model="lincal-1_7b")
    hybrid.decode = SimpleNamespace(meta={"get_n_layers": 2})
    for source, target in ((256, 1), (1, PREFILL_AR)):
        hybrid._encoding_override(
            result[source][0], result[target][0], override_kv_cache=True
        )
    return result


@pytest.mark.parametrize("ar", [1, PREFILL_AR, 256])
def test_exported_qdq_after_encoding_override(ar, device, quantized_models, tmp_path):
    model, inputs = quantized_models[ar]
    path = tmp_path / f"ar{ar}.pt2"
    torch.export.save(torch.export.export(model, inputs, strict=True), path)
    restored = torch.export.load(path).module().to(device)
    inputs = tuple(t.to(device) for t in inputs)
    with torch.no_grad():
        expected = restored(*inputs)[0]
        assert torch.isfinite(expected).all()
        for cut in sorted({1, max(1, ar // 2), ar}):
            for fill in (1, 127):
                actual = restored(*changed_inputs(inputs, cut, fill))[0][:, :cut]
                assert torch.equal(actual, expected[:, :cut])


@pytest.mark.parametrize("ar", [1, PREFILL_AR])
def test_qnn_masked_softmax_deployment(ar, quantized_models, tmp_path, monkeypatch):
    serial = os.environ.get("ET_MASK_QNN_DEVICE")
    if not serial:
        pytest.skip("Set ET_MASK_QNN_DEVICE to test QNN on a connected device")
    from executorch.backends.qualcomm.debugger.utils import generate_optrace
    from executorch.backends.qualcomm.export_utils import QnnConfig, SimpleADB
    from executorch.backends.qualcomm.utils.utils import (
        dump_context_from_pte,
        generate_htp_compiler_spec,
        generate_qnn_executorch_compiler_spec,
        get_soc_to_chipset_map,
        to_edge_transform_and_lower_to_qnn,
    )
    from executorch.examples.qualcomm.oss_scripts.llama.llama import (
        _build_qnn_profile_inputs,
    )

    root = Path(__file__).resolve().parents[3]
    config = QnnConfig(
        backend="htp",
        soc_model=os.environ.get("ET_MASK_QNN_SOC", "SM8750"),
        device=serial,
        build_folder=os.environ.get(
            "EXECUTORCH_ANDROID_BUILD", str(root / "build-android")
        ),
    )
    monkeypatch.chdir(tmp_path)
    model, inputs = quantized_models[ar]
    mha2sha = os.environ.get("ET_MASK_QNN_MHA2SHA", "1")
    assert mha2sha in {"0", "1"}
    compiler_spec = generate_qnn_executorch_compiler_spec(
        soc_model=get_soc_to_chipset_map()[config.soc_model],
        backend_options=generate_htp_compiler_spec(use_fp16=False),
        use_mha2sha=mha2sha == "1",
        profile_level=3,
    )
    program = to_edge_transform_and_lower_to_qnn(
        copy.deepcopy(model), inputs, compiler_spec
    ).to_executorch()
    delegates = program.executorch_program.execution_plan[0].delegates
    assert len(delegates) == 1 and delegates[0].id == "QnnBackend"
    path = tmp_path / f"masked-ar{ar}.pte"
    path.write_bytes(program.buffer)
    cut = max(1, ar // 2)
    variants = [
        inputs,
        changed_inputs(inputs, cut, 1),
        changed_inputs(inputs, cut, 127),
    ]
    workspace = subprocess.check_output(
        [
            "adb",
            "-s",
            serial,
            "shell",
            "mktemp -d /data/local/tmp/et-mask-regression.XXXXXX",
        ],
        text=True,
        timeout=30,
    ).strip()
    assert workspace.startswith("/data/local/tmp/et-mask-regression.")
    adb = SimpleADB(config, str(path), workspace=workspace)
    adb.push(
        inputs=variants,
        files=adb.backend_library_paths[config.backend],
        init_env=False,
    )
    adb.execute()
    adb.pull(str(tmp_path))
    outputs = [
        np.fromfile(
            tmp_path / "outputs" / f"output_{i}_0.raw", dtype=np.float32
        ).reshape(1, ar, 128)
        for i in range(len(variants))
    ]
    assert all(np.isfinite(output).all() for output in outputs)
    for output in outputs[1:]:
        np.testing.assert_array_equal(output[:, :cut], outputs[0][:, :cut])
    with torch.no_grad():
        reference = model(*inputs)[0].numpy()
    np.testing.assert_allclose(outputs[0], reference, atol=0.1, rtol=0.05)

    trace_adb = SimpleADB(config, str(path), workspace=f"{workspace}/optrace")
    binaries = dump_context_from_pte(str(path))
    assert len(binaries) == 1
    # Profiling consumes the delegate's IO order/dtypes, not the outer PTE inputs.
    profile_inputs = _build_qnn_profile_inputs(
        SimpleNamespace(model=config.soc_model), binaries[0], [inputs]
    )
    traces = generate_optrace(
        str(tmp_path),
        get_soc_to_chipset_map()[config.soc_model],
        trace_adb,
        str(path),
        profile_inputs,
    )
    operators = []
    qnn_operators = []
    for _, qhas_path in traces.values():
        qhas = json.loads(Path(qhas_path).read_text())
        operators.extend(row["op"] for row in qhas["data"]["htp_op_types"]["data"])
        qnn_operators.extend(row["op"] for row in qhas["data"]["qnn_op_types"]["data"])
    assert any("MaskedSoftmax" in operator for operator in operators)
    assert any("Hadamard" in operator for operator in qnn_operators)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_full_context_qdq_gpu_causality(tmp_path):
    from executorch.examples.qualcomm.oss_scripts.llama.model_validation.engines.pt2 import (
        PT2Engine,
    )

    model, inputs = make_model(4096, context=4096, window=512)
    converted = quantize(model, inputs)
    path = tmp_path / "full-context.pt2"
    torch.export.save(torch.export.export(converted, inputs, strict=True), path)
    engine = PT2Engine(path, device="cuda", pad_token_id=0)
    assert engine.sliding_window == 512
    ids = inputs[0][0].tolist()
    for prefix in (ids[:147], ids):
        logits = engine(prefix)
        assert logits.shape == (len(prefix), 128)
        assert engine.last_causality_check["passed"]
