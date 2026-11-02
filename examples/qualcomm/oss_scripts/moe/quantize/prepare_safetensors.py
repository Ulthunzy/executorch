"""Build new calibration/QDQ artifacts without importing historical model graphs."""

import argparse
import gc
import hashlib
import json
from pathlib import Path

from executorch.examples.qualcomm.oss_scripts.moe.model.qwen35 import (
    add_model_config_argument,
    load_model_config,
)


def prepare_data(args, settings):
    import torch
    from transformers import AutoTokenizer
    from executorch.examples.qualcomm.oss_scripts.moe.quantize.checkpoint import (
        Checkpoint,
        rope_tables,
    )

    checkpoint = Checkpoint(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    with args.corpus.open() as stream:
        lines = [json.loads(line)["text"] for line in stream]
    sequences, sources = [], []
    offsets = [
        0,
        500,
        1000,
        1500,
        2000,
        2500,
        3000,
        3500,
        len(lines) - 1000,
        len(lines) - 500,
    ]
    for offset in offsets:
        ids = tokenizer.encode(
            "".join(lines[offset : offset + 400]), add_special_tokens=False
        )[:1024]
        if len(ids) != 1024:
            raise ValueError(f"Corpus segment {offset} is too short")
        sequences.append(ids)
        sources.append(dict(first_line=offset, last_line=offset + 399))
    ids = torch.tensor(sequences)
    table = checkpoint.tensor("model.language_model.embed_tokens.weight")
    embedded = table[ids].float()
    with (args.output / "embedding.bf16").open("wb") as handle:
        for start in range(0, len(table), 4096):
            handle.write(
                table[start : start + 4096]
                .bfloat16()
                .view(torch.uint16)
                .numpy()
                .tobytes()
            )
    torch.save(
        dict(
            calibration=embedded[:8],
            heldout=embedded[8:],
            tokens=ids[:8],
            heldout_ids=ids[8:],
            reference_calibration=embedded[:8],
            reference_heldout=embedded[8:],
        ),
        args.output / "INPUTS-fp32.pt",
    )
    cos, sin = rope_tables(checkpoint.config, max(2048, settings.context.max_seq_len))
    (args.output / "assets").mkdir(exist_ok=True)
    cos.numpy().tofile(args.output / "assets/cos.f32")
    sin.numpy().tofile(args.output / "assets/sin.f32")
    shards = {
        name: dict(
            bytes=(args.model / name).stat().st_size,
            mtime_ns=(args.model / name).stat().st_mtime_ns,
        )
        for name in sorted(set(checkpoint.mapping.values()))
    }
    report = dict(
        model=str(args.model),
        index_sha256=hashlib.sha256(
            (args.model / "model.safetensors.index.json").read_bytes()
        ).hexdigest(),
        corpus=str(args.corpus),
        corpus_sha256=hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
        token_sha256=hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
        sources=sources,
        fit_sequences=[0, 1, 2, 4, 5, 6, 7],
        selection_sequence=3,
        heldout_sequences=[8, 9],
        heldout_used_for_fitting=False,
        safetensors=shards,
        old_qdq_used=False,
        old_quantized_weights_used=False,
    )
    (args.output / "DATA.json").write_text(json.dumps(report, indent=2) + "\n")
    print("RAW_DATA_READY", json.dumps(report), flush=True)


def quantized_embedding_inputs(args):
    import numpy as np
    import torch

    data = torch.load(args.output / "INPUTS-fp32.pt", weights_only=True, mmap=True)
    ids = torch.cat((data["tokens"], data["heldout_ids"])).numpy()
    table = np.memmap(
        args.output / "embedding.int4",
        mode="r",
        dtype=np.uint8,
        offset=64,
        shape=(248320, 64, 20),
    )
    blocks = np.asarray(table[ids])
    code = np.empty((*blocks.shape[:-1], 32), dtype=np.float32)
    code[..., ::2] = blocks[..., :16] & 15
    code[..., 1::2] = blocks[..., :16] >> 4
    scale = blocks[..., 16:18].copy().view("<f2").astype(np.float32)
    bias = blocks[..., 18:20].copy().view("<f2").astype(np.float32)
    hidden = torch.from_numpy((code * scale + bias).reshape(10, 1024, 2048))
    # Preserve separate FP32 chains; only the quantized chain consumes INT4 lookup values.
    values = dict(data)
    values["calibration"] = hidden[:8]
    values["heldout"] = hidden[8:]
    temporary = args.output / "INPUTS-updated.pt"
    torch.save(values, temporary)
    temporary.replace(args.output / "INPUTS.pt")
    (args.output / "EMBEDDING_INPUTS.json").write_text(
        json.dumps(
            dict(
                embedding="int4_g32",
                reference_chain="original BF16 lookup",
                quantized_chain="new INT4 lookup",
            )
        )
        + "\n"
    )
    print("INT4_CALIBRATION_INPUTS_READY", flush=True)


def prepare_attention(args, settings):
    import torch
    from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
    from torchao.quantization.pt2e.observer import MinMaxObserver
    from executorch.backends.qualcomm.export_utils import make_quantizer
    from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
    from executorch.backends.qualcomm.serialization.qc_schema import (
        QnnExecuTorchBackendType,
    )
    from executorch.backends.qualcomm.utils.utils import convert_linear_to_conv2d
    from executorch.examples.qualcomm.custom_op.deltanet import (
        deltanet_quant_annotations,
    )
    from executorch.examples.qualcomm.oss_scripts.llama.static_llm_quant_recipe import (
        Qwen3_5QuantRecipe,
    )
    from executorch.examples.qualcomm.oss_scripts.llama.quantize.generation import (
        remove_export_checks,
    )
    from executorch.examples.qualcomm.oss_scripts.llama.quantize.qwen35 import (
        move_graph,
        stabilize_gdn_weight_scales,
    )
    from executorch.examples.qualcomm.oss_scripts.moe.quantize.checkpoint import (
        Checkpoint,
        AttentionFront,
        rope_tables,
        run_attention,
        attention_inputs,
    )
    from executorch.examples.qualcomm.oss_scripts.moe.quantize.matrix import (
        HessianCapture,
        original_weights,
        apply_gptq,
    )
    from executorch.examples.qualcomm.oss_scripts.moe.wrappers.graph_utils import (
        specialize_front_ar1,
    )

    layer = args.layer
    directory = args.output / f"layers/layer{layer}"
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = Checkpoint(args.model)
    linear = layer % 4 != 3
    checkpoint.constants(layer, directory / "moe-constants.npz")
    from executorch.examples.qualcomm.oss_scripts.moe.quantize.reference import (
        reference_inputs,
    )

    hidden = reference_inputs(
        args.output, layer, settings.quantization.calibration_dtype
    )
    cos, sin = rope_tables(checkpoint.config)
    model = AttentionFront(checkpoint, layer).eval().cuda()
    inputs = attention_inputs(hidden[:1, :128].contiguous(), linear, cos, sin)
    model = convert_linear_to_conv2d(model.cpu())
    graph = torch.export.export(model, inputs, strict=True).module()
    weights = original_weights(graph)
    recipe = Qwen3_5QuantRecipe().recipe
    if linear:
        recipe.custom_quant_annotations.extend(
            deltanet_quant_annotations(
                QuantDtype.use_16a4w,
                act_observer=MinMaxObserver,
                share_state_qparams=True,
            )
        )
    quantizer = make_quantizer(
        backend=QnnExecuTorchBackendType.kHtpBackend,
        soc_model=settings.target.soc_model,
    )
    quantizer.set_recipe(recipe)
    prepared = prepare_pt2e(graph, quantizer)
    move_graph(prepared, "cuda")
    hidden = hidden.cuda()
    fit = [0, 1, 2, 4, 5, 6, 7]
    for sequence, start, _, _ in run_attention(prepared, hidden, linear, cos, sin, fit):
        if start == 896:
            print("ACTIVATIONS_CALIBRATED", layer, sequence, flush=True)
    quantized = convert_pt2e(prepared)
    quantized = torch.export.export(
        quantized, tuple(v.cuda() for v in inputs), strict=False
    ).module()
    remove_export_checks(quantized)
    stabilized = stabilize_gdn_weight_scales(quantized)
    capture = HessianCapture(quantized)
    for _ in run_attention(capture.run, hidden, linear, cos, sin, fit):
        pass
    details = apply_gptq(
        quantized, weights, capture, settings.quantization.matrix_gptq_damping
    )
    norm, residual = torch.empty_like(hidden), torch.empty_like(hidden)
    for sequence, start, _, out in run_attention(quantized, hidden, linear, cos, sin):
        norm[sequence, start : start + 128] = out[0][0]
        residual[sequence, start : start + 128] = out[1][0]
    if not torch.isfinite(norm).all() or not torch.isfinite(residual).all():
        raise ValueError("Nonfinite GPTQ front output")
    torch.save(
        dict(norm=norm.cpu(), residual=residual.cpu()), directory / "front-output.pt"
    )
    move_graph(quantized, "cpu")
    exported = torch.export.export(quantized, inputs, strict=False)
    torch.export.save(exported, directory / "front-qdq.pt2")
    decoded, decode_inputs = specialize_front_ar1(exported.module(), inputs)
    torch.export.save(
        torch.export.export(decoded, decode_inputs, strict=False),
        directory / "decode-qdq.pt2",
    )
    report = dict(
        layer=layer,
        source="safetensors",
        method="all_projection_gptq",
        projections=details,
        stabilized_weight_scales=stabilized,
        fit_sequences=fit,
        heldout_used_for_fitting=False,
        excluded_parameters="Norm/bias/decay vectors retain the existing floating-point or derived bias encoding; GPTQ applies to matrix/filter weights",
    )
    (directory / "GPTQ.json").write_text(json.dumps(report, indent=2) + "\n")
    print("RAW_ATTENTION_READY", layer, len(details), flush=True)


def prepare_head(args, settings):
    import torch
    from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
    from executorch.backends.qualcomm.export_utils import make_quantizer
    from executorch.backends.qualcomm.serialization.qc_schema import (
        QnnExecuTorchBackendType,
    )
    from executorch.backends.qualcomm.utils.utils import convert_linear_to_conv2d
    from executorch.examples.qualcomm.oss_scripts.llama.static_llm_quant_recipe import (
        Qwen3_5QuantRecipe,
    )
    from executorch.examples.qualcomm.oss_scripts.llama.quantize.qwen35 import (
        move_graph,
    )
    from executorch.examples.qualcomm.oss_scripts.llama.quantize.generation import (
        remove_export_checks,
    )
    from executorch.examples.qualcomm.oss_scripts.moe.quantize.checkpoint import (
        Checkpoint,
        Head,
    )
    from executorch.examples.qualcomm.oss_scripts.moe.quantize.matrix import (
        HessianCapture,
        original_weights,
        apply_gptq,
    )
    from executorch.examples.qualcomm.oss_scripts.moe.wrappers.graph_utils import (
        specialize_front_ar1,
    )

    directory = args.output / "head"
    directory.mkdir(exist_ok=True)
    from executorch.examples.qualcomm.oss_scripts.moe.quantize.reference import (
        reference_inputs,
    )

    hidden = reference_inputs(args.output, 40, settings.quantization.calibration_dtype)
    model = convert_linear_to_conv2d(Head(Checkpoint(args.model)).eval())
    sample = hidden[:1, :128].contiguous()
    graph = torch.export.export(model, (sample,), strict=True).module()
    del model
    gc.collect()
    weights = original_weights(graph)
    quantizer = make_quantizer(
        backend=QnnExecuTorchBackendType.kHtpBackend,
        soc_model=settings.target.soc_model,
    )
    quantizer.set_recipe(Qwen3_5QuantRecipe().recipe)
    prepared = prepare_pt2e(graph, quantizer)
    move_graph(prepared, "cuda")
    for sequence in [0, 1, 2, 4, 5, 6, 7]:
        for start in range(0, 1024, 128):
            prepared(
                hidden[sequence : sequence + 1, start : start + 128].contiguous().cuda()
            )
    converted = convert_pt2e(prepared)
    converted = torch.export.export(converted, (sample.cuda(),), strict=False).module()
    remove_export_checks(converted)
    capture = HessianCapture(converted)
    for sequence in [0, 1, 2, 4, 5, 6, 7]:
        for start in range(0, 1024, 128):
            capture.run(
                hidden[sequence : sequence + 1, start : start + 128].contiguous().cuda()
            )
    report = apply_gptq(
        converted, weights, capture, settings.quantization.matrix_gptq_damping
    )
    move_graph(converted, "cpu")
    exported = torch.export.export(converted, (sample,), strict=False)
    graph, values = specialize_front_ar1(exported.module(), (sample,))
    torch.export.save(
        torch.export.export(graph, values, strict=False), directory / "head-qdq.pt2"
    )
    (directory / "GPTQ.json").write_text(
        json.dumps(dict(method="all_projection_gptq", projections=report), indent=2)
        + "\n"
    )
    print("RAW_HEAD_READY", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_config_argument(parser)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=["data", "embedding-inputs", "reference", "attention", "head"],
        required=True,
    )
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--layer", type=int)
    parser.add_argument("--qnn-binding", type=Path)
    parser.add_argument("--first-layer", type=int, default=0)
    parser.add_argument("--last-layer", type=int, default=3)
    args = parser.parse_args()
    settings = load_model_config(args.model_config)
    if args.stage == "attention" and settings.quantization.attention != "gptq":
        parser.error("Use attention=gptq for raw attention calibration")
    if args.stage == "head" and settings.quantization.head != "w4a16_gptq":
        parser.error("Use head=w4a16_gptq for raw head calibration")
    if args.stage == "attention" and args.layer not in range(40):
        parser.error("attention needs --layer 0..39")
    if args.stage == "reference" and not 0 <= args.first_layer <= args.last_layer < 40:
        parser.error("Reference layer range must be within 0..39")
    if args.stage == "data" and args.corpus is None:
        parser.error("data needs --corpus")
    args.output.mkdir(parents=True, exist_ok=True)
    from executorch.examples.qualcomm.oss_scripts.moe.wrappers.binding import (
        load_binding,
    )

    load_binding(args.qnn_binding)
    import torch

    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.stage == "data":
        prepare_data(args, settings)
    elif args.stage == "embedding-inputs":
        quantized_embedding_inputs(args)
    elif args.stage == "reference":
        from executorch.examples.qualcomm.oss_scripts.moe.quantize.reference import (
            prepare_reference,
        )

        prepare_reference(args, settings)
    elif args.stage == "attention":
        prepare_attention(args, settings)
    else:
        prepare_head(args, settings)


if __name__ == "__main__":
    main()
