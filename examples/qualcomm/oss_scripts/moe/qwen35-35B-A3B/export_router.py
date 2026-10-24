import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from model_config import add_model_config_argument, load_model_config


def export_layer(args, layer):
    from export.binding import load_binding
    from export.quantization import validate_router_graph, validate_router_source

    calibration = validate_router_source(
        args.source_dir / f"layer{layer}",
        load_model_config(args.model_config).quantization.router_gptq_damping,
    )

    qnn = load_binding(args.qnn_binding)
    import torch
    from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset
    from executorch.backends.qualcomm.utils.utils import (
        generate_htp_compiler_spec,
        generate_qnn_executorch_compiler_spec,
        to_edge_transform_and_lower_to_qnn,
    )
    from executorch.exir._serialize._program import deserialize_pte_binary

    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    for phase, ar_len in (("prefill", 128), ("decode", 1)):
        source = args.source_dir / f"layer{layer}/router-ar{ar_len}-qdq.pt2"
        model = torch.export.load(source).module()
        validate_router_graph(model)
        example = (torch.zeros(1, ar_len, 2048),)
        specs = generate_qnn_executorch_compiler_spec(
            QcomChipset.SM8975,
            generate_htp_compiler_spec(use_fp16=False),
            shared_buffer=False,
        )
        manager = to_edge_transform_and_lower_to_qnn(model, example, specs)
        directory = args.output / phase / f"layer{layer}"
        directory.mkdir(parents=True, exist_ok=True)
        pte = directory / "moe-front.pte"
        with pte.open("wb") as file:
            manager.to_executorch().write_to_file(file)
        program = deserialize_pte_binary(pte.read_bytes()).program
        plan = program.execution_plan[0]
        instructions = [
            i.instr_args for chain in plan.chains for i in chain.instructions
        ]
        if len(instructions) != 1 or type(instructions[0]).__name__ != "DelegateCall":
            raise ValueError("Router/shared requires one QNN delegate")
        blob = program.backend_delegate_data[plan.delegates[0].processed.index].data
        binary = bytes(qnn.StripProtocol(blob))
        (directory / "moe-front.bin").write_bytes(binary)
        arguments = instructions[0].args
        count = len(plan.inputs)
        report = dict(
            layer=layer,
            ar_len=ar_len,
            weight_bits=8,
            activation_bits=16,
            method="gptq",
            calibration=calibration,
            context_bytes=len(binary),
            source_sha256=hashlib.file_digest(source.open("rb"), "sha256").hexdigest(),
            input_order=[arguments[:count].index(i) for i in plan.inputs],
            output_order=[arguments[count:].index(i) for i in plan.outputs],
        )
        (directory / "MOE_FRONT.json").write_text(json.dumps(report, indent=2) + "\n")
        print("W8_ROUTER_EXPORTED", phase, layer, len(binary), flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Export independent W8A16 GPTQ Router/shared contexts"
    )
    add_model_config_argument(parser)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qnn-binding", type=Path)
    parser.add_argument("--layers", type=int, nargs="+", default=list(range(40)))
    args = parser.parse_args()
    if load_model_config(args.model_config).quantization.router_shared != "w8a16_gptq":
        parser.error(
            "export_router.py compiles the w8a16_gptq recipe; fp16 is handled by the fused export"
        )
    if not args.layers or any(layer not in range(40) for layer in args.layers):
        parser.error("Layer IDs must be in 0..39")
    if len(args.layers) == 1:
        export_layer(args, args.layers[0])
        return
    for layer in args.layers:
        command = [
            sys.executable,
            "-u",
            __file__,
            "--model-config",
            str(args.model_config),
            "--source-dir",
            str(args.source_dir),
            "--output",
            str(args.output),
            "--layers",
            str(layer),
        ]
        if args.qnn_binding:
            command += ["--qnn-binding", str(args.qnn_binding)]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
