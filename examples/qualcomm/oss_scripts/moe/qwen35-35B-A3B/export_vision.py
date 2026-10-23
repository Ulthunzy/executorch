import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Export the checkpoint's W8A16 vision encoder for SM8975"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calib-images", type=Path, nargs="+", required=True)
    parser.add_argument("--image-height", type=int, default=448)
    parser.add_argument("--image-width", type=int, default=448)
    parser.add_argument("--calibration-device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--qnn-binding", type=Path)
    args = parser.parse_args()
    if (
        min(args.image_height, args.image_width) < 32
        or args.image_height % 32
        or args.image_width % 32
    ):
        parser.error("Image dimensions must be positive multiples of 32")

    from export.binding import load_binding

    qnn = load_binding(args.qnn_binding)
    import numpy as np
    import torch
    from PIL import Image
    from safetensors import safe_open
    from transformers import AutoConfig
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
    from executorch.backends.qualcomm.quantizer.quantizer import QnnQuantizer
    from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset
    from executorch.backends.qualcomm.utils.utils import (
        generate_htp_compiler_spec,
        generate_qnn_executorch_compiler_spec,
        to_edge_transform_and_lower_to_qnn,
    )
    from executorch.examples.qualcomm.oss_scripts.llama.dataset.preprocessors import (
        _qwen_vl_image_to_patches,
    )
    from executorch.examples.qualcomm.oss_scripts.llama.encoder.encoder_quant_recipe import (
        Qwen35VLEncoderQuantRecipe,
    )
    from executorch.examples.qualcomm.oss_scripts.llama.model.vision_encoder import (
        Qwen35VLVisionEncoder,
    )
    from executorch.examples.qualcomm.oss_scripts.llama.quantize.qwen35 import (
        move_graph,
        protect_linear_bias,
    )
    from executorch.exir._serialize._program import deserialize_pte_binary

    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    config.dtype = config.vision_config.dtype = torch.float32
    model = Qwen35VLVisionEncoder(config, args.image_height, args.image_width).eval()
    index = json.loads((args.model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    keys = {
        key: filename
        for key, filename in index.items()
        if key.startswith("model.visual.")
    }
    if not keys:
        raise ValueError("Checkpoint has no model.visual weights")
    state = {}
    for filename in sorted(set(keys.values())):
        with safe_open(args.model / filename, framework="pt", device="cpu") as handle:
            for key, shard in keys.items():
                if shard == filename:
                    state[key.removeprefix("model.")] = handle.get_tensor(key).float()
    incompatible = model.load_state_dict(state, strict=False)
    if (
        set(incompatible.missing_keys)
        != {"visual.patch_embed_weight_flat", "visual.patch_embed_bias"}
        or incompatible.unexpected_keys
    ):
        raise ValueError(f"Incomplete vision checkpoint: {incompatible}")
    del state
    print(
        f"VISION_LOADED layers={config.vision_config.depth} tensors={len(keys)}",
        flush=True,
    )
    examples = []
    for path in args.calib_images:
        with Image.open(path) as image:
            examples.append(
                _qwen_vl_image_to_patches(
                    image, args.image_height, args.image_width
                ).unsqueeze(0)
            )
    inputs = (examples[0],)
    model.to(args.calibration_device)
    reference = model(inputs[0].to(args.calibration_device)).cpu()
    np.asarray(reference, dtype="<f4").tofile(args.output / "reference.f32")
    np.asarray(inputs[0], dtype="<f4").tofile(args.output / "reference-pixels.f32")
    model.cpu()
    model = torch.export.export(model, inputs, strict=True).module()
    quantizer = QnnQuantizer(soc_model=QcomChipset.SM8975)
    quantizer.set_recipe(Qwen35VLEncoderQuantRecipe().recipe)
    model = prepare_pt2e(model, quantizer)
    move_graph(model, args.calibration_device)
    for number, pixels in enumerate(examples, 1):
        model(pixels.to(args.calibration_device))
        print(
            f"VISION_CALIBRATED {number}/{len(examples)} on {args.calibration_device}",
            flush=True,
        )
    protect_linear_bias(model)
    model = convert_pt2e(model)
    outputs = torch.utils._pytree.tree_leaves(
        model(inputs[0].to(args.calibration_device))
    )
    if len(outputs) != 1:
        raise ValueError("Expected one vision embedding output")
    qdq_output = outputs[0].cpu()
    signal = reference.double().square().sum()
    error = (reference.double() - qdq_output.double()).square().sum()
    sqnr = float(10 * torch.log10(signal / error))
    if not torch.isfinite(qdq_output).all():
        raise ValueError("Nonfinite quantized vision output")
    np.asarray(qdq_output, dtype="<f4").tofile(args.output / "qdq-reference.f32")
    move_graph(model, "cpu")
    if args.calibration_device == "cuda":
        torch.cuda.empty_cache()
    torch.export.save(
        torch.export.export(model, inputs, strict=False), args.output / "vision-qdq.pt2"
    )
    print(f"VISION_QDQ sqnr_db={sqnr:.4f}", flush=True)
    specs = generate_qnn_executorch_compiler_spec(
        QcomChipset.SM8975,
        generate_htp_compiler_spec(use_fp16=False),
        shared_buffer=False,
    )
    manager = to_edge_transform_and_lower_to_qnn(model, inputs, specs)
    with (args.output / "vision.pte").open("wb") as handle:
        manager.to_executorch().write_to_file(handle)
    program = deserialize_pte_binary((args.output / "vision.pte").read_bytes()).program
    plan = program.execution_plan[0]
    instructions = [
        instruction.instr_args
        for chain in plan.chains
        for instruction in chain.instructions
    ]
    if (
        len(instructions) != 1
        or type(instructions[0]).__name__ != "DelegateCall"
        or len(plan.inputs) != 1
        or len(plan.outputs) != 1
    ):
        raise RuntimeError(
            "Vision must have one QNN delegate, one input and one output"
        )
    blob = program.backend_delegate_data[plan.delegates[0].processed.index].data
    (args.output / "vision.bin").write_bytes(qnn.StripProtocol(blob))
    metadata = dict(
        version=1,
        model_config_sha256=hashlib.sha256(
            (args.model / "config.json").read_bytes()
        ).hexdigest(),
        precision="W8A16",
        image_height=args.image_height,
        image_width=args.image_width,
        image_token_id=config.image_token_id,
        input_shape=list(inputs[0].shape),
        output_shape=list(reference.shape),
        qdq_sqnr_db=sqnr,
        calibration_images=[str(p.resolve()) for p in args.calib_images],
    )
    (args.output / "VISION.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("EXPORTED", args.output, flush=True)


if __name__ == "__main__":
    main()
