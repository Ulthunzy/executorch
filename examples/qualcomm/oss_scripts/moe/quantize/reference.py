"""Original-model activation windows for independent layer quantization."""

import gc
import json
from pathlib import Path
import time

import torch
from safetensors import safe_open

from executorch.backends.qualcomm.quantizer.metrics import log_quantization


def reference_output(root, layer, dtype):
    record = torch.load(
        Path(root) / "reference" / f"layer{layer}.pt", weights_only=True, mmap=True
    )
    if (
        record.get("layer") != layer
        or record.get("calibration_dtype") != dtype
        or record.get("calibration_source") != "original_model"
        or record["all_output"].dtype
        != {"bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
    ):
        raise ValueError(f"Layer {layer}: incompatible original-model activation cache")
    return record


def reference_inputs(root, layer, dtype):
    if layer:
        return reference_output(root, layer - 1, dtype)["all_output"].float()
    data = torch.load(Path(root) / "INPUTS-fp32.pt", weights_only=True, mmap=True)
    return torch.cat((data["reference_calibration"], data["reference_heldout"])).float()


@torch.no_grad()
def prepare_reference(args, settings):
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
        Qwen3_5MoeTextConfig,
    )
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeDecoderLayer,
        Qwen3_5MoeTextRotaryEmbedding,
    )

    dtype_name = settings.quantization.calibration_dtype
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[dtype_name]
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU requires calibration_dtype=float32")
    config = Qwen3_5MoeTextConfig(
        **json.loads((args.model / "config.json").read_text())["text_config"]
    )
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    mapping = json.loads((args.model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    hidden = reference_inputs(args.output, args.first_layer, dtype_name).to(
        device="cuda", dtype=dtype
    )
    positions = torch.arange(hidden.shape[1], device="cuda")[None]
    rotary = Qwen3_5MoeTextRotaryEmbedding(config, device="cuda")
    position_embeddings = rotary(hidden[:1], positions)
    mask = torch.triu(
        torch.full(
            (hidden.shape[1], hidden.shape[1]),
            torch.finfo(dtype).min,
            device="cuda",
            dtype=dtype,
        ),
        diagonal=1,
    )[None, None]
    directory = args.output / "reference"
    directory.mkdir(exist_ok=True)
    for layer in range(args.first_layer, args.last_layer + 1):
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        with torch.device("meta"):
            model = Qwen3_5MoeDecoderLayer(config, layer)
        state = {}
        for name in model.state_dict():
            key = f"model.language_model.layers.{layer}.{name}"
            with safe_open(str(args.model / mapping[key]), framework="pt") as shard:
                state[name] = shard.get_tensor(key).to(dtype=dtype)
        model.load_state_dict(state, assign=True)
        del state
        model = model.eval().to("cuda")
        norms, outputs = [], []

        def capture_norm(module, inputs):
            norms.append(inputs[0].detach().cpu())

        handle = model.mlp.register_forward_pre_hook(capture_norm)
        try:
            for sequence in range(len(hidden)):
                result = model(
                    hidden[sequence : sequence + 1],
                    position_embeddings=position_embeddings,
                    attention_mask=(
                        mask if config.layer_types[layer] == "full_attention" else None
                    ),
                    position_ids=positions,
                )
                outputs.append(result)
        finally:
            handle.remove()
        hidden = torch.cat(outputs).to(dtype)
        del result
        if not torch.isfinite(hidden).all():
            raise FloatingPointError(
                f"Layer {layer}: nonfinite original-model activations"
            )
        output = hidden.cpu()
        path = directory / f"layer{layer}.pt"
        temporary = path.with_suffix(".pt.tmp")
        torch.save(
            dict(
                layer=layer,
                calibration_source="original_model",
                calibration_dtype=dtype_name,
                all_output=output,
                norm=torch.cat(norms),
                test=dict(output=output[8:9]),
            ),
            temporary,
        )
        temporary.replace(path)
        log_quantization(
            f"reference/layer{layer}",
            event="calibration_activations_ready",
            calibration_dtype=dtype_name,
            samples=len(hidden),
            tokens=hidden.shape[1],
            seconds=time.monotonic() - started,
            cuda_peak_mib=torch.cuda.max_memory_allocated() / 2**20,
        )
        norms.clear()
        del model, outputs, output
        gc.collect()
        torch.cuda.empty_cache()
