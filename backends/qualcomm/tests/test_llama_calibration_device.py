# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from executorch.backends.qualcomm._passes.seq_mse import SeqMseModule
from executorch.backends.qualcomm.quantizer.observers.per_block_param_observer import (
    PerBlockParamObserver,
)
from executorch.backends.qualcomm.quantizer.quantizer import QnnQuantizer
from executorch.examples.models.llama.model_args import ModelArgs
from executorch.examples.qualcomm.oss_scripts.llama.dataset.collators import (
    LLMCalibCollator,
)
from executorch.examples.qualcomm.oss_scripts.llama.decoder_constants import (
    TEXT_DECODER,
    VISION_ENCODER,
)
from executorch.examples.qualcomm.oss_scripts.llama.inference import DecoderInference
from executorch.examples.qualcomm.oss_scripts.llama.llama import (
    _build_parser,
    export_llama,
)
from executorch.examples.qualcomm.oss_scripts.llama.masking_utils import (
    AttentionMask,
    CausalAttentionMask,
    SlidingWindowAttentionMask,
)
from executorch.examples.qualcomm.oss_scripts.llama.model.static_llama import (
    MultiScopeAwareLlamaModel_swa,
)
from executorch.examples.qualcomm.oss_scripts.llama.quantize.ptq import PTQStrategy
from executorch.examples.qualcomm.oss_scripts.llama.static_llm_quant_recipe import (
    Lincal_1_7BQuantRecipe,
)
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e


@pytest.fixture(autouse=True)
def limit_threads():
    original = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(original)


def make_calibration_model():
    torch.manual_seed(17)
    args = ModelArgs(
        dim=64,
        n_layers=2,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        hidden_dim=128,
        vocab_size=128,
        max_seq_len=8,
        max_context_len=8,
        use_kv_cache=True,
        use_hf_rope=True,
        local_rope_theta=10000.0,
        layer_types=["full_attention", "sliding_attention"],
        sliding_window=8,
    )
    args.enable_r3 = True
    args.enable_masked_softmax = True
    model = MultiScopeAwareLlamaModel_swa(args, ar_len=8).eval()
    for layer in model.layers:
        layer.attention.prepare_attention_conv()
        layer.feed_forward.prepare_feedforward_conv()
    model.prepare_output_conv()
    tokens, masks, pos, k, v = model.get_example_inputs()
    quantizer = QnnQuantizer()
    quantizer.set_recipe(Lincal_1_7BQuantRecipe().recipe)
    prepared = prepare_pt2e(
        torch.export.export(model, (tokens, *masks, pos, *k, *v), strict=True).module(),
        quantizer,
    )
    batch = LLMCalibCollator(masks, 8)([([1, 2, 3, 4, 5], None)])
    return model, prepared, batch


def test_r3_cache_annotation_preserves_rotation_input_precision(capsys):
    _, prepared, _ = make_calibration_model()
    rotations = [
        node
        for node in prepared.graph.nodes
        if node.target == torch.ops.qnn_custom.hadamard_transform.default
    ]
    if not rotations:
        pytest.skip("QNN SDK does not recompose Hadamard transforms")
    assert len(rotations) == 4
    for rotation in rotations:
        observer = prepared.get_submodule(rotation.args[0].target)
        assert (observer.quant_min, observer.quant_max) == (0, 65535)
        # Check the upstream RoPE concat itself, not only a re-quantization edge.
        concat = rotation.args[0].args[0]
        assert concat.target == torch.ops.aten.cat.default
        for user in concat.users:
            assert user.op == "call_module"
            observer = prepared.get_submodule(user.target)
            assert (observer.quant_min, observer.quant_max) == (0, 65535)
    # Inputs are tokens, two attention masks, positions, then four KV tensors.
    cache_inputs = [node for node in prepared.graph.nodes if node.op == "placeholder"][
        4:
    ]
    assert len(cache_inputs) == 4
    for cache in cache_inputs:
        for user in cache.users:
            observer = prepared.get_submodule(user.target)
            assert observer.quant_max - observer.quant_min == 255
    assert "not expected in the input1" not in capsys.readouterr().out


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("seq_mse", [0, 2])
def test_calibration_roundtrip(device, seq_mse, tmp_path):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    model, prepared, batch = make_calibration_model()
    inference = DecoderInference(model.get_example_inputs, max_context_len=8)
    calls = []

    def check_device(module, inputs):
        assert all(tensor.device.type == device for tensor in inputs)
        assert all(p.device.type == device for p in module.parameters())
        if device == "cuda":
            assert not torch.backends.cuda.matmul.allow_tf32
            assert not torch.backends.cudnn.allow_tf32
        calls.append(True)

    handle = prepared.register_forward_pre_hook(check_device)
    tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    result = PTQStrategy(
        inference, prepared, seq_mse_candidates=seq_mse, device=device
    ).quantize({TEXT_DECODER: [batch, batch]})
    handle.remove()
    assert len(calls) >= 2
    assert result is prepared
    assert all(p.device.type == "cpu" for p in prepared.parameters())
    assert all(b.device.type == "cpu" for b in prepared.buffers())
    assert all(t.device.type == "cpu" for t in inference.get_inputs(batch["input_ids"]))
    assert (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    ) == tf32
    assert batch["input_ids"].device.type == "cpu"
    assert all(mask.device.type == "cpu" for mask in batch["attention_mask"])
    observers = [m for m in prepared.modules() if isinstance(m, PerBlockParamObserver)]
    assert observers
    assert all(
        m.min_val is not None and m.min_val.device.type == "cpu" for m in observers
    )
    if seq_mse:
        assert all(m.calculate_qparams()[0].device.type == "cpu" for m in observers)
        assert not any(isinstance(m, SeqMseModule) for m in prepared.modules())

    qdq = convert_pt2e(prepared)
    with torch.no_grad():
        logits = inference.predict_step(
            qdq, batch["input_ids"], batch["attention_mask"]
        )
    assert logits.shape == (1, 8, 128)
    assert torch.isfinite(logits).all()
    inputs = tuple(inference.get_inputs(batch["input_ids"], batch["attention_mask"]))
    path = tmp_path / "calibrated.pt2"
    torch.export.save(torch.export.export(qdq, inputs, strict=True), path)
    restored = torch.export.load(path).module()
    torch.testing.assert_close(restored(*inputs)[0], logits)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_calibration_matches_cpu():
    model, prepared, batch = make_calibration_model()
    _, gpu_prepared, _ = make_calibration_model()
    for device, module in [("cpu", prepared), ("cuda", gpu_prepared)]:
        PTQStrategy(
            DecoderInference(model.get_example_inputs, max_context_len=8),
            module,
            device=device,
        ).quantize({TEXT_DECODER: [batch]})
    cpu_stats = dict(prepared.named_buffers(remove_duplicate=False))
    gpu_stats = dict(gpu_prepared.named_buffers(remove_duplicate=False))
    assert cpu_stats.keys() == gpu_stats.keys(), cpu_stats.keys() ^ gpu_stats.keys()
    for name in cpu_stats:
        torch.testing.assert_close(
            cpu_stats[name], gpu_stats[name], atol=2e-5, rtol=1e-4
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_failure_restores_cpu():
    model, prepared, batch = make_calibration_model()
    inference = DecoderInference(model.get_example_inputs, max_context_len=8)
    tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)

    def fail(module, inputs, outputs):
        raise RuntimeError("calibration failed deliberately")

    handle = prepared.register_forward_hook(fail)
    with pytest.raises(RuntimeError, match="failed deliberately"):
        PTQStrategy(inference, prepared, device="cuda").quantize(
            {TEXT_DECODER: [batch]}
        )
    handle.remove()
    assert all(p.device.type == "cpu" for p in prepared.parameters())
    assert all(b.device.type == "cpu" for b in prepared.buffers())
    assert all(t.device.type == "cpu" for t in inference.get_inputs(batch["input_ids"]))
    assert (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    ) == tf32


def test_calibration_cli_validation(monkeypatch):
    parser = _build_parser()
    argv = ["--decoder_model", "lincal-1_7b", "--prompt", "hello", "--compile_only"]
    assert parser.parse_args(argv).calibration_device == "cpu"
    args = parser.parse_args([*argv, "--calibration_device", "cuda"])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA_VISIBLE_DEVICES"):
        export_llama(args)
    args.qat = True
    with pytest.raises(ValueError, match="not QAT"):
        export_llama(args)


def test_skip_qdq_save_cli_validation():
    parser = _build_parser()
    argv = ["--decoder_model", "lincal-1_7b", "--prompt", "hello", "--compile_only"]
    assert not parser.parse_args(argv).skip_qdq_save
    args = parser.parse_args(
        [*argv, "--skip_qdq_save", "--eval_methods", "sqnr_eval"]
    )
    with pytest.raises(ValueError, match="skip_qdq_save"):
        export_llama(args)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_multimodal_calibration_inputs(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    template = AttentionMask(
        [
            CausalAttentionMask(1, 8, 8),
            SlidingWindowAttentionMask(1, 8, 15, 8),
        ]
    )
    example = (
        torch.zeros(1, 8, 4),
        template,
        torch.zeros(3, 1, 8, dtype=torch.int32),
        [torch.zeros(1, 1, 4, 7)],
        [torch.zeros(1, 1, 7, 4)],
    )
    inference = DecoderInference(lambda: example, image_token_id=7, max_context_len=8)
    hidden = torch.randn(1, 4, 4)
    calls = []

    class Decoder(torch.nn.Module):
        def forward(self, embeddings, mask, window_mask, positions, k, v):
            for tensor in (embeddings, mask, window_mask, positions, k, v):
                assert tensor.device.type == device
            assert positions.shape == (3, 1, 8)
            torch.testing.assert_close(embeddings[:, 1:5].cpu(), hidden)
            calls.append(True)
            return embeddings, k, v

    embedding = torch.nn.Embedding(16, 4)
    batch = LLMCalibCollator(template, 8)([([1, 7, 7, 7, 7, 2], None)])
    PTQStrategy(inference, Decoder(), tok_embedding=embedding, device=device).quantize(
        {
            TEXT_DECODER: [batch],
            VISION_ENCODER: [(hidden,)],
        }
    )
    assert calls == [True]
    assert embedding.weight.device.type == "cpu"
    assert hidden.device.type == "cpu"
    assert inference.pos_ids.device.type == "cpu"
