# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock, patch
import weakref

import pytest
import torch
from executorch.examples.models.llama.model_args import ModelArgs
from executorch.examples.qualcomm.oss_scripts.llama.dataset.collators import (
    LLMCalibCollator,
)
from executorch.examples.qualcomm.oss_scripts.llama.inference.decoder import (
    DecoderInference,
)
from executorch.examples.qualcomm.oss_scripts.llama.masking_utils import (
    AttentionMask,
    CausalAttentionMask,
    PADDING_MASK_VALUE,
    SlidingWindowAttentionMask,
)
from executorch.examples.qualcomm.oss_scripts.llama.model.static_llama import (
    LlamaModel,
    Qwen35ModelWithoutEmbedding_swa,
)
from executorch.exir import to_edge


@pytest.fixture(autouse=True)
def limit_threads():
    original = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(original)


def model_args(context_len, sliding_window):
    return ModelArgs(
        dim=8,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        vocab_size=16,
        hidden_dim=16,
        head_dim=4,
        max_seq_len=context_len,
        max_context_len=context_len,
        use_kv_cache=True,
        layer_types=["full_attention", "sliding_attention"],
        sliding_window=sliding_window,
        use_hf_rope=True,
        kv_io_bit_width=8,
        mrope_section=[1, 1, 0],
    )


@pytest.mark.parametrize("window", [4, 8, 16])
def test_true_swa_calibration_accepts_collated_masks(window):
    model = Qwen35ModelWithoutEmbedding_swa(model_args(8, window), ar_len=8).eval()
    hidden, template, pos, k, v = model.get_example_inputs()
    batch = LLMCalibCollator(template, 8)([([1, 2, 3], None)])
    inputs = (hidden, *template, pos, *k, *v)
    exported = torch.export.export(model, inputs, strict=True).module()
    with torch.no_grad():
        expected = model(*inputs)[0][:, :3]
        actual = exported(hidden, *batch["attention_mask"], pos, *k, *v)[0]
    torch.testing.assert_close(actual[:, :3], expected)
    assert torch.isfinite(actual).all()
    assert batch["attention_mask"][1].shape == (1, 1, 8, min(8, window) + 7)


def test_calibration_preserves_template_and_batch_padding():
    template = AttentionMask(
        [CausalAttentionMask(2, 8, 8), SlidingWindowAttentionMask(2, 8, 15, 8)]
    )
    original = [m.clone() for m in template]
    batch = LLMCalibCollator(template, 8)([([1, 2], None), ([3, 4, 5, 6], None)])
    for mask, base in zip(batch["attention_mask"], original):
        assert torch.all(mask[0, :, 2:] == PADDING_MASK_VALUE)
        assert torch.all(mask[1, :, 4:] == PADDING_MASK_VALUE)
        torch.testing.assert_close(mask[0, :, :2], base[0, :, :2])
        torch.testing.assert_close(mask[1, :, :4], base[1, :, :4])
    for mask, base in zip(template, original):
        torch.testing.assert_close(mask, base)
    assert torch.all(batch["attention_mask"][1][..., :7] == PADDING_MASK_VALUE)


@pytest.mark.parametrize("ar_len", [1, 2, 8, 16])
def test_true_swa_edge_masks_have_matching_quantized_io(ar_len):
    from executorch.examples.qualcomm.oss_scripts.llama.wrappers.llm_wrappers import (
        TextDecoder,
    )

    args = model_args(16, 8)
    args.layer_types = ["sliding_attention", "sliding_attention"]
    model = Qwen35ModelWithoutEmbedding_swa(args, ar_len=ar_len).eval()
    hidden, masks, pos, k, v = model.get_example_inputs()
    ep = torch.export.export(model, (hidden, *masks, pos, *k, *v), strict=True)
    graph = to_edge(ep).exported_program().graph
    decoder = object.__new__(TextDecoder)
    decoder.control_args = SimpleNamespace(
        decoder_model="lincal3_5-0_8b", enable_true_swa=True
    )
    decoder.meta = model.get_metadata()
    decoder.cache_io_shape = {tuple(cache.shape) for cache in (*k, *v)}
    decoder.kv_cache_shape = set()
    decoder.io_shape = set()
    fixed_point_type = {"io_type": torch.uint16, "kv_type": torch.uint8}
    mask_nodes = [
        n for n in graph.nodes if n.op == "placeholder" and "atten_mask" in n.name
    ]
    assert len(mask_nodes) == 2
    for node in mask_nodes:
        assert TextDecoder._tag_ios(decoder, node, fixed_point_type) == torch.uint16


@pytest.mark.parametrize("skip_qdq_save", [False, True])
def test_saved_sqnr_graph_accepts_full_sequence(tmp_path, skip_qdq_save):
    from executorch.backends.qualcomm.quantizer.quantizer import (
        QnnQuantizer,
        QuantDtype,
    )
    from executorch.examples.qualcomm.oss_scripts.llama.decoder_constants import (
        DECODE_QDQ_FILENAME,
        TEXT_DECODER,
    )
    from executorch.examples.qualcomm.oss_scripts.llama.wrappers import llm_wrappers
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    args = model_args(8, 8)
    calib_model = LlamaModel(args, ar_len=8).eval()
    decode_model = LlamaModel(args, ar_len=1).eval()
    wrappers = []
    for model in (decode_model, calib_model):
        tokens, masks, pos, k, v = model.get_example_inputs()
        inputs = (tokens, *masks, pos, *k, *v)
        quantizer = QnnQuantizer()
        quantizer.set_default_quant_config(QuantDtype.use_16a8w)
        prepared = prepare_pt2e(
            torch.export.export(model, inputs, strict=True).module(), quantizer
        )
        with torch.no_grad():
            prepared(*inputs)
        wrappers.append(
            SimpleNamespace(
                decoder=convert_pt2e(prepared),
                export_input=inputs,
                meta=model.get_metadata(),
                control_args=SimpleNamespace(artifact=str(tmp_path)),
                dep_table={},
                passes_job={},
                _save_logits_quant_attrs=Mock(),
                _save_output_kv_cache_quant_attrs=Mock(),
            )
        )
    hybrid = object.__new__(llm_wrappers.HybridTextDecoder)
    hybrid.decode, hybrid.calibration_prefill = wrappers
    hybrid.prefill = SimpleNamespace(decoder=None)
    hybrid.apply_embedding = False
    hybrid.control_args = SimpleNamespace(
        decoder_model="stories260k", skip_qdq_save=skip_qdq_save
    )
    hybrid._encoding_override = Mock()
    request = SimpleNamespace(
        method_data={
            TEXT_DECODER: SimpleNamespace(skip_quantize=False, compile_spec=[[]])
        }
    )
    saved_graph_refs = []
    save_exported_program = torch.export.save

    def save_and_track(program, path):
        saved_graph_refs.append(weakref.ref(program))
        save_exported_program(program, path)

    def stop_before_lowering(**kwargs):
        assert len(saved_graph_refs) == (0 if skip_qdq_save else 1)
        assert all(ref() is None for ref in saved_graph_refs)
        raise RuntimeError("stop before QNN compilation")

    with (
        patch.object(torch.export, "save", new=save_and_track),
        patch.object(
            llm_wrappers,
            "to_edge_transform_and_lower_to_qnn",
            side_effect=stop_before_lowering,
        ) as lower,
    ):
        with pytest.raises(RuntimeError, match="stop before QNN compilation"):
            hybrid.compile(request)
    assert lower.call_args.kwargs["inputs"]["kv_forward"][0].shape == (1, 1)
    if skip_qdq_save:
        assert not (tmp_path / DECODE_QDQ_FILENAME).exists()
        hybrid._encoding_override.assert_called_once()
        hybrid.decode._save_logits_quant_attrs.assert_called_once()
        hybrid.decode._save_output_kv_cache_quant_attrs.assert_called_once()
        return
    # The serialized evaluation graph and deployed graph deliberately have different ARs.
    saved = torch.export.load(tmp_path / DECODE_QDQ_FILENAME).module()
    batch = LLMCalibCollator(calib_model.get_example_inputs()[1], 8)(
        [([1, 2, 3], None)]
    )
    inference = DecoderInference(calib_model.get_example_inputs, max_context_len=8)
    logits = inference.predict_step(saved, batch["input_ids"], batch["attention_mask"])
    assert logits.shape == (1, 8, 16)


def test_native_kv_manager(tmp_path):
    root = Path(__file__).resolve().parents[3]
    build = Path(os.environ.get("EXECUTORCH_HOST_BUILD", root / "build-x86"))
    core = build / "libexecutorch_core.a"
    loader = build / "extension/data_loader/libextension_data_loader.a"
    compiler = shutil.which("c++")
    if compiler is None or not core.exists() or not loader.exists():
        pytest.skip("Build host executorch_core and extension_data_loader first")

    class CacheFixture(torch.nn.Module):
        def __init__(self, ar, cache_len, recurrent=False, mask_dtype=torch.float32):
            super().__init__()
            self.ar = ar
            self.recurrent = recurrent
            k_shape = (1, 12, 4) if recurrent else (1, 1, 4, cache_len)
            v_shape = (1, 1, 4, 4) if recurrent else (1, 1, cache_len, 4)
            self.inputs = (
                torch.zeros(1, ar, 8),
                torch.zeros(1, 1, ar, 256, dtype=mask_dtype),
                torch.zeros(1, 1, ar, cache_len + ar, dtype=mask_dtype),
                torch.zeros(1, ar, dtype=torch.int32),
                torch.zeros(k_shape),
                torch.zeros(v_shape),
            )

        def forward(self, hidden, mask, window_mask, pos, k, v):
            if self.recurrent:
                return hidden, torch.zeros_like(k), torch.zeros_like(v)
            return hidden, torch.zeros(1, 1, 4, self.ar), torch.zeros(1, 1, self.ar, 4)

    fixtures = {
        "prefill": CacheFixture(128, 128),
        "decode": CacheFixture(1, 255),
        "recurrent": CacheFixture(128, 128, recurrent=True),
        "circular_float": CacheFixture(16, 7),
        "circular_uint16": CacheFixture(16, 7, mask_dtype=torch.uint16),
    }
    pte = to_edge(
        {
            name: torch.export.export(model, model.inputs)
            for name, model in fixtures.items()
        }
    ).to_executorch()
    fixture_path = tmp_path / "cache_fixtures.pte"
    fixture_path.write_bytes(pte.buffer)
    binary = tmp_path / "kv_manager_test"
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
            str(
                root
                / "examples/qualcomm/oss_scripts/llama/runner/test/kv_manager_test.cpp"
            ),
            str(root / "examples/qualcomm/oss_scripts/llama/runner/kv_manager.cpp"),
            str(loader),
            str(core),
            "-lpthread",
            "-o",
            str(binary),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert compiled.returncode == 0, compiled.stderr
    result = subprocess.run(
        [str(binary), str(fixture_path)], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "SWA masks passed" in result.stdout
