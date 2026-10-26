import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
from dataclasses import replace

from model_config import add_model_config_argument, load_model_config


def export_layer(args, layer):
    settings = load_model_config(args.model_config)
    if settings.quantization.attention == "gptq":
        provenance = json.loads(
            (args.source_dir / f"layer{layer}/GPTQ.json").read_text()
        )
        if (
            provenance.get("source") != "safetensors"
            or provenance.get("method") != "all_projection_gptq"
        ):
            raise ValueError(f"Layer {layer} is missing all-projection GPTQ provenance")
    from export.binding import load_binding

    qnn = load_binding(args.qnn_binding)
    import torch
    from executorch.backends.qualcomm._passes.qnn_pass_manager import (
        get_qnn_pass_manager_cls,
    )
    from executorch.backends.qualcomm._passes.tag_quant_io import TagQuantIO
    from executorch.backends.qualcomm.serialization.qc_schema import (
        QcomChipset,
        QnnExecuTorchBackendType,
        QnnExecuTorchOpPackageInfo,
        QnnExecuTorchOpPackageOptions,
        QnnExecuTorchOpPackagePlatform,
        QnnExecuTorchOpPackageTarget,
    )
    from executorch.backends.qualcomm.utils.constants import (
        QCOM_PASS_ACTIVATE_KEY,
        QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY,
    )
    from executorch.backends.qualcomm.utils.utils import (
        generate_htp_compiler_spec,
        generate_qnn_executorch_compiler_spec,
        to_edge_transform_and_lower_to_qnn,
    )
    from executorch.exir._serialize._program import deserialize_pte_binary
    from export.fp16_attention_core import fp16_attention_core
    from export.graph_utils import resize_attention
    from export.model import FusedFront, QuantizedFusedFront
    from export.quantization import validate_router_graph, validate_router_source
    import executorch.examples.qualcomm.custom_op.deltanet  # Registers the frozen graph's operator.

    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    linear = layer % 4 != 3
    source = args.source_dir / f"layer{layer}/decode-qdq.pt2"
    constants = args.source_dir / f"layer{layer}/moe-constants.npz"
    if linear and args.op_package_host is None:
        raise ValueError("GDN export requires --op-package-host")
    router_calibration = (
        validate_router_source(
            args.router_qdq / f"layer{layer}",
            settings.quantization.router_gptq_damping,
        )
        if args.router_qdq
        else None
    )
    for capacity in [args.capacities[0]] if linear else args.capacities:
        graph = torch.export.load(source).module()
        inputs = resize_attention(graph, capacity)
        if not linear:
            fp16_attention_core(graph, inputs)
        if not all(torch.isfinite(value).all() for value in graph(*inputs)):
            raise ValueError(f"Layer {layer}: nonfinite reference output")
        if args.router_qdq:
            moe = torch.export.load(
                args.router_qdq / f"layer{layer}/router-ar1-qdq.pt2"
            ).module()
            validate_router_graph(moe)
            model = QuantizedFusedFront(graph, moe)
        else:
            model = FusedFront(graph, constants)
        package = None
        if linear:
            fields = dict(
                op_package_name="DeltaNetOpPackage",
                interface_provider="DeltaNetOpPackageInterfaceProvider",
                custom_op_name=str(torch.ops.qnn_custom_ops.deltanet_core.default),
                qnn_op_type_name="DeltaNetCore",
            )
            package = QnnExecuTorchOpPackageOptions(
                op_package_infos=[
                    QnnExecuTorchOpPackageInfo(
                        **fields,
                        op_package_path=args.op_package_device,
                        target=QnnExecuTorchOpPackageTarget.HTP,
                        platform=QnnExecuTorchOpPackagePlatform.AARCH64_ANDROID,
                    ),
                    QnnExecuTorchOpPackageInfo(
                        **fields,
                        op_package_path=str(args.op_package_host.resolve()),
                        target=QnnExecuTorchOpPackageTarget.CPU,
                        platform=QnnExecuTorchOpPackagePlatform.X86_64,
                    ),
                ]
            )
        specs = generate_qnn_executorch_compiler_spec(
            QcomChipset.SM8975,
            generate_htp_compiler_spec(use_fp16=False),
            shared_buffer=False,
            op_package_options=package,
        )
        jobs = get_qnn_pass_manager_cls(
            QnnExecuTorchBackendType.kHtpBackend
        ).get_capture_program_passes()
        shapes = (
            {(1, 8192, 4), (1, 32, 128, 128)}
            if linear
            else {
                (1, 2, 256, capacity - 1),
                (1, 2, capacity - 1, 256),
                (1, 2, 256, 1),
                (1, 2, 1, 256),
            }
        )

        def tag(node):
            value = node.meta.get("val")
            if (
                isinstance(value, torch.Tensor)
                and tuple(value.shape) in shapes
                and (
                    node.op == "placeholder"
                    or any(user.op == "output" for user in node.users)
                )
            ):
                return {"uint16": torch.uint16}[settings.cache.kv_dtype]

        jobs[TagQuantIO][QCOM_PASS_ACTIVATE_KEY] = True
        jobs[TagQuantIO][QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY][
            "get_quant_io_dtype_fn"
        ] = tag
        manager = to_edge_transform_and_lower_to_qnn(
            model, inputs, specs, passes_job=jobs
        )
        directory = (
            args.output
            / ("gdn-fused" if linear else f"context{capacity}-fused")
            / f"layer{layer}"
        )
        directory.mkdir(parents=True, exist_ok=True)
        pte = directory / "attention.pte"
        with pte.open("wb") as file:
            manager.to_executorch().write_to_file(file)
        program = deserialize_pte_binary(pte.read_bytes()).program
        plan = program.execution_plan[0]
        instructions = [
            instruction.instr_args
            for chain in plan.chains
            for instruction in chain.instructions
        ]
        if len(instructions) != 1 or type(instructions[0]).__name__ != "DelegateCall":
            raise RuntimeError(
                f"Layer {layer}: expected a single QNN delegate without CPU fallback"
            )
        blob = program.backend_delegate_data[plan.delegates[0].processed.index].data
        (directory / "attention.bin").write_bytes(qnn.StripProtocol(blob))
        arguments = instructions[0].args
        count = len(plan.inputs)
        info = dict(
            input_order=[arguments[:count].index(index) for index in plan.inputs],
            output_order=[arguments[count:].index(index) for index in plan.outputs],
        )
        (directory / "io.bin").write_bytes(
            struct.pack(
                "<" + "I" * (2 + len(info["input_order"]) + len(info["output_order"])),
                len(info["input_order"]),
                len(info["output_order"]),
                *info["input_order"],
                *info["output_order"],
            )
        )
        info.update(
            compile_settings=replace(
                settings, context=replace(settings.context, max_seq_len=capacity)
            ).compile_settings(),
            layer=layer,
            ar_len=1,
            context=None if linear else capacity,
            frozen_qparams=True,
            router_quantization="w8a16_gptq" if args.router_qdq else "fp16",
            router_calibration=router_calibration,
            source=str(source),
            source_sha256=hashlib.file_digest(source.open("rb"), "sha256").hexdigest(),
        )
        (directory / "IO.json").write_text(json.dumps(info, indent=2) + "\n")
        print("EXPORTED", directory, flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Export optimized AR1 attention/router/shared graphs from calibrated per-layer QDQ programs"
    )
    add_model_config_argument(parser)
    parser.add_argument("--max-seq-len", type=int)
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Contains layerN/decode-qdq.pt2 and moe-constants.npz",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output assets directory"
    )
    parser.add_argument("--layers", type=int, nargs="+", default=list(range(40)))
    parser.add_argument(
        "--router-qdq", type=Path, help="Calibrated W8A16 GPTQ Router/shared graphs"
    )
    parser.add_argument(
        "--capacities",
        type=int,
        nargs="+",
    )
    parser.add_argument(
        "--qnn-binding",
        type=Path,
        help="Optional PyQnnManagerAdaptor shared library built with the selected SDK",
    )
    parser.add_argument(
        "--op-package-host", type=Path, help="Host libQnnDeltaNetOpPackage.so"
    )
    parser.add_argument(
        "--op-package-device",
        default="/data/local/tmp/qwen35-35B-A3B/libQnnDeltaNetOpPackage_HTP.so",
    )
    args = parser.parse_args()
    settings = load_model_config(args.model_config)
    if args.capacities and args.max_seq_len:
        parser.error("Choose --max-seq-len or --capacities")
    args.capacities = args.capacities or [
        args.max_seq_len or settings.context.max_seq_len
    ]
    try:
        for capacity in args.capacities:
            replace(
                settings, context=replace(settings.context, max_seq_len=capacity)
            ).validate()
    except ValueError as error:
        parser.error(str(error))
    if bool(args.router_qdq) != (settings.quantization.router_shared == "w8a16_gptq"):
        parser.error(
            "w8a16_gptq needs --router-qdq; fp16 uses checkpoint constants without --router-qdq"
        )
    if not args.layers or any(layer not in range(40) for layer in args.layers):
        parser.error("--layers must contain layer IDs from 0 through 39")
    if len(args.layers) == 1:
        export_layer(args, args.layers[0])
        return
    # Each SDK process owns one layer to bound host memory and package state.
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
            "--capacities",
            *map(str, args.capacities),
            "--op-package-device",
            args.op_package_device,
        ]
        if args.qnn_binding:
            command.extend(["--qnn-binding", str(args.qnn_binding)])
        if args.op_package_host:
            command.extend(["--op-package-host", str(args.op_package_host)])
        if args.router_qdq:
            command.extend(["--router-qdq", str(args.router_qdq)])
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
