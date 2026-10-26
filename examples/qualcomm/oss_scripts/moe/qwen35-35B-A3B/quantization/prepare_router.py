import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from workflow.model_config import add_model_config_argument, load_model_config


def attention_inputs(args, layer):
    import numpy as np
    import torch
    from executorch.examples.qualcomm.oss_scripts.llama.quantize.qwen35 import (
        move_graph,
    )
    import executorch.examples.qualcomm.custom_op.deltanet
    from executorch.examples.qualcomm.oss_scripts.llama.model import chunk_gdn

    if args.normalized_inputs:
        return torch.load(args.normalized_inputs, weights_only=True, mmap=True)[
            "norm"
        ].to(args.device)

    if layer:
        hidden = torch.load(
            args.sequential / f"layer{layer - 1}.pt", weights_only=True, mmap=True
        )["hidden"]
    else:
        data = torch.load(args.inputs, weights_only=True, mmap=True)
        hidden = torch.cat((data["calibration"], data["heldout"]))
    hidden = hidden.to(args.device)
    model = move_graph(
        torch.export.load(args.source_dir / f"layer{layer}/front-qdq.pt2").module(),
        args.device,
    )
    cos = torch.from_numpy(
        np.fromfile(args.source_dir / "cos.f32", "<f4").reshape(-1, 32)
    ).to(args.device)
    sin = torch.from_numpy(
        np.fromfile(args.source_dir / "sin.f32", "<f4").reshape(-1, 32)
    ).to(args.device)
    norm = torch.empty_like(hidden)
    for sequence in range(len(hidden)):
        if layer % 4 != 3:
            first = torch.zeros(1, 8192, 4, device=args.device)
            second = torch.zeros(1, 32, 128, 128, device=args.device)
        else:
            first = torch.zeros(1, 2, 256, 896, device=args.device)
            second = torch.zeros(1, 2, 896, 256, device=args.device)
        for start in range(0, 1024, 128):
            x = hidden[sequence : sequence + 1, start : start + 128].contiguous()
            if layer % 4 != 3:
                out = model(x, first, second)
                first, second = out[2:]
            else:
                mask = torch.full((1, 1, 128, 1024), -255.0, device=args.device)
                if start:
                    mask[..., 896 - start : 896] = 0
                mask[..., 896:] = torch.triu(
                    torch.full((128, 128), -255.0, device=args.device), diagonal=1
                )
                out = model(
                    x,
                    cos[start : start + 128].contiguous(),
                    sin[start : start + 128].contiguous(),
                    mask,
                    first,
                    second,
                )
                first = torch.cat((first[..., 128:], out[2]), -1)
                second = torch.cat((second[:, :, 128:], out[3]), 2)
            norm[sequence, start : start + 128] = out[0][0]
    return norm


def prepare_layer(args, layer):
    settings = load_model_config(args.model_config)
    from export.binding import load_binding

    load_binding(args.qnn_binding)
    import hashlib
    import torch
    import torch.nn.functional as F
    from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
    from torchao.quantization.pt2e.observer import MinMaxObserver
    from executorch.backends.qualcomm.quantizer.quantizer import (
        QnnQuantizer,
        QuantDtype,
    )
    from executorch.backends.qualcomm.quantizer.quant_recipe import (
        QuantRecipe,
        QuantGranularity,
    )
    from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset
    from executorch.examples.qualcomm.oss_scripts.llama.quantize.gptq import (
        gptq_fixed_grid,
    )
    from export.model import MoeFront

    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    directory = args.output / f"layer{layer}"
    directory.mkdir(parents=True, exist_ok=True)
    norm = attention_inputs(args, layer)
    fit = norm[[0, 1, 2, 4, 5, 6, 7]].reshape(1, -1, 2048)
    heldout = norm[8:]
    model = (
        MoeFront(args.source_dir / f"layer{layer}/moe-constants.npz")
        .eval()
        .to(args.device)
    )
    reference = model(heldout)
    names = [
        "gate_weight",
        "shared_expert_gate_proj_weight",
        "shared_expert_up_proj_weight",
        "shared_expert_down_proj_weight",
        "shared_expert_gate_weight",
    ]
    weights = [getattr(model, name).clone() for name in names]
    recipe = QuantRecipe(
        QuantDtype.use_16a8w,
        False,
        act_observer=MinMaxObserver,
        granularity=QuantGranularity.PER_TENSOR,
    )
    recipe.add_node_target(
        {torch.ops.aten.linear.default, torch.ops.aten.conv2d.default},
        QuantDtype.use_16a8w,
        False,
        act_observer=MinMaxObserver,
        granularity=QuantGranularity.PER_CHANNEL,
    )
    quantizer = QnnQuantizer(soc_model=QcomChipset[settings.target.soc_model])
    quantizer.set_recipe(recipe)
    prepared = prepare_pt2e(
        torch.export.export(model, (fit[:, :128].contiguous(),)).module(), quantizer
    )
    for start in range(0, fit.shape[1], 128):
        prepared(fit[:, start : start + 128].contiguous())
    converted = convert_pt2e(prepared)
    converted._forward_pre_hooks.clear()

    class Capture(torch.fx.Interpreter):
        def __init__(self, module):
            super().__init__(module)
            self.inputs = []

        def call_function(self, target, inputs, kwargs):
            if target == torch.ops.aten.linear.default:
                self.inputs.append(inputs[0].reshape(-1, inputs[0].shape[-1]))
            return super().call_function(target, inputs, kwargs)

    linear = [
        n for n in converted.graph.nodes if n.target == torch.ops.aten.linear.default
    ]
    if len(linear) != 5:
        raise ValueError("Expected five Router/shared projections")
    capture = Capture(converted)
    capture.run(fit)
    common = capture.inputs[0]
    reports = []
    for index, (node, weight, name) in enumerate(zip(linear, weights, names)):
        dq = node.args[1]
        if (
            dq.target != torch.ops.quantized_decomposed.dequantize_per_channel.default
            or dq.args[3:7] != (0, -127, 127, torch.int8)
        ):
            raise ValueError("Expected symmetric per-channel INT8 weight grid")
        codes = getattr(converted, dq.args[0].target)
        scale = getattr(converted, dq.args[1].target)
        if index == 3:
            capture = Capture(converted)
            capture.run(fit)
            inputs = capture.inputs[index]
        else:
            inputs = common
        original = codes.clone()
        gptq = gptq_fixed_grid(
            weight,
            inputs,
            scale[:, None],
            group_size=weight.shape[1],
            quant_min=-127,
            quant_max=127,
            damping=args.damping,
        )
        codes.copy_(gptq)
        reconstructed = gptq.float() * scale[:, None]
        target = F.linear(inputs, weight)
        rtn_error = (
            (F.linear(inputs, original.float() * scale[:, None]) - target)
            .double()
            .square()
            .sum()
        )
        gptq_error = (F.linear(inputs, reconstructed) - target).double().square().sum()
        reports.append(
            dict(
                name=name,
                weight_bits=8,
                quant_min=-127,
                quant_max=127,
                group="per_output_channel",
                calibration_input_tokens=len(inputs),
                changed_codes=int((original != gptq).sum()),
                rtn_fit_mse=float(rtn_error / target.numel()),
                gptq_fit_mse=float(gptq_error / target.numel()),
                code_sha256=hashlib.sha256(gptq.cpu().numpy().tobytes()).hexdigest(),
            )
        )
    actual = converted(heldout)
    if not all(torch.isfinite(v).all() for v in actual):
        raise ValueError("Nonfinite heldout output")

    def sqnr(ref, value):
        return float(
            10
            * torch.log10(
                ref.double().square().sum()
                / (ref.double() - value.double()).square().sum()
            )
        )

    stats = dict(
        layer=layer,
        method="fixed_grid_gptq",
        damping=args.damping,
        weight_bits=8,
        activation_bits=16,
        calibration_sequences=[0, 1, 2, 4, 5, 6, 7],
        calibration_tokens=7168,
        activation_grid_tokens=7168,
        validation_used_for_fitting=False,
        heldout_used_for_fitting=False,
        calibration_state=(
            str(args.normalized_inputs)
            if args.normalized_inputs
            else "existing sequential W2 baseline; unchanged attention"
        ),
        shared_output_sqnr_db=sqnr(reference[2], actual[2]),
        router_score_sqnr_db=sqnr(reference[1], actual[1]),
        router_top8_overlap=float(
            (reference[0].unsqueeze(-1) == actual[0].unsqueeze(-2))
            .any(-1)
            .float()
            .mean()
        ),
        projections=reports,
    )
    converted = converted.cpu()
    for ar_len in (128, 1):
        example = norm[:1, :ar_len].contiguous().cpu()
        program = torch.export.export(converted, (example,))
        torch.export.save(program, directory / f"router-ar{ar_len}-qdq.pt2")
    (directory / "REPORT.json").write_text(json.dumps(stats, indent=2) + "\n")
    print("GPTQ_ROUTER_READY", layer, json.dumps(stats), flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate W8A16 Router/shared using GPTQ on frozen sequential W2 inputs"
    )
    add_model_config_argument(parser)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--sequential", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--normalized-inputs", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qnn-binding", type=Path)
    parser.add_argument("--layers", type=int, nargs="+", default=list(range(40)))
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--damping", type=float)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    settings = load_model_config(args.model_config)
    if settings.quantization.router_shared != "w8a16_gptq":
        parser.error(
            "FP16 Router/shared uses checkpoint constants; skip GPTQ preparation"
        )
    args.damping = (
        settings.quantization.router_gptq_damping
        if args.damping is None
        else args.damping
    )
    if (
        not args.layers
        or any(layer not in range(40) for layer in args.layers)
        or not math.isfinite(args.damping)
        or args.damping <= 0
    ):
        parser.error("Expected layer IDs 0..39 and positive damping")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.normalized_inputs and len(args.layers) != 1:
        parser.error("--normalized-inputs requires one layer")
    if args.worker:
        if len(args.layers) != 1:
            parser.error("Worker requires exactly one layer")
        prepare_layer(args, args.layers[0])
        return
    for layer in args.layers:
        command = [
            sys.executable,
            "-u",
            __file__,
            "--model-config",
            str(args.model_config),
            "--worker",
            "--layers",
            str(layer),
            "--source-dir",
            str(args.source_dir),
            "--sequential",
            str(args.sequential),
            "--inputs",
            str(args.inputs),
            "--output",
            str(args.output),
            "--device",
            args.device,
            "--damping",
            str(args.damping),
        ]
        if args.qnn_binding:
            command += ["--qnn-binding", str(args.qnn_binding)]
        if args.normalized_inputs:
            command += ["--normalized-inputs", str(args.normalized_inputs)]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
