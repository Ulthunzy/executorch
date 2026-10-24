import argparse
import hashlib
import json
from pathlib import Path
import struct


def main():
    parser = argparse.ArgumentParser(
        description="Compile a small group of AR128/AR1 layers with shared constants"
    )
    parser.add_argument("--prefill-source", type=Path, required=True)
    parser.add_argument("--decode-source", type=Path, required=True)
    parser.add_argument("--router-qdq", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--capacity", type=int, choices=[1152, 2048], default=1152)
    parser.add_argument("--qnn-binding", type=Path)
    parser.add_argument("--op-package-host", type=Path, required=True)
    parser.add_argument("--op-package-device", required=True)
    parser.add_argument("--independent", action="store_true")
    args = parser.parse_args()
    if len(set(args.layers)) != len(args.layers) or not 1 <= len(args.layers) <= 4:
        parser.error("Select one to four distinct layers")
    if any(layer not in range(40) for layer in args.layers):
        parser.error("Layer IDs must be 0..39")

    from export.binding import load_binding

    qnn = load_binding(args.qnn_binding)
    import torch
    import executorch.examples.qualcomm.custom_op.deltanet
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
    from executorch.examples.qualcomm.oss_scripts.llama.quantize.generation import (
        remove_export_checks,
        replace_prefill_cores,
    )
    from export.fp16_attention_core import fp16_attention_core
    from export.graph_utils import resize_attention
    from export.model import FusedFront, MoeFront, QuantizedFusedFront

    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    models, examples = {}, {}
    for layer in args.layers:
        prefill = torch.export.load(
            args.prefill_source / f"layer{layer}/front-qdq.pt2"
        ).module()
        prefill_inputs = tuple(
            torch.zeros(node.meta["val"].shape)
            for node in prefill.graph.nodes
            if node.op == "placeholder"
        )
        if layer % 4 != 3:
            replace_prefill_cores(prefill)
            prefill = torch.export.export(
                prefill, prefill_inputs, strict=False
            ).module()
            remove_export_checks(prefill)
        decode = torch.export.load(
            args.decode_source / f"layer{layer}/decode-qdq.pt2"
        ).module()
        decode_inputs = resize_attention(decode, args.capacity)
        if layer % 4 == 3:
            fp16_attention_core(decode, decode_inputs)
        constants = args.prefill_source / f"layer{layer}/moe-constants.npz"
        if args.router_qdq:
            router = torch.export.load(
                args.router_qdq / f"layer{layer}/router-ar128-qdq.pt2"
            ).module()
            decoder = QuantizedFusedFront(
                decode,
                torch.export.load(
                    args.router_qdq / f"layer{layer}/router-ar1-qdq.pt2"
                ).module(),
            )
        else:
            # Match FusedFront's FP16 projection boundary in both phases.
            class HalfRouter(torch.nn.Module):
                def __init__(self, path):
                    super().__init__()
                    self.moe = MoeFront(path).half()

                def forward(self, hidden):
                    ids, scores, shared = self.moe(hidden.half())
                    return ids, scores.float(), shared.float()

            router = HalfRouter(constants)
            decoder = FusedFront(decode, constants)
        for suffix, model, inputs in (
            ("prefill_attention", prefill, prefill_inputs),
            ("prefill_router", router, (torch.zeros(1, 128, 2048),)),
            ("decode", decoder, decode_inputs),
        ):
            name = f"layer{layer}_{suffix}"
            models[name] = model
            examples[name] = inputs

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
                op_package_path=str(args.op_package_host.resolve()),
                target=QnnExecuTorchOpPackageTarget.CPU,
                platform=QnnExecuTorchOpPackagePlatform.X86_64,
            ),
            QnnExecuTorchOpPackageInfo(
                **fields,
                op_package_path=args.op_package_device,
                target=QnnExecuTorchOpPackageTarget.HTP,
                platform=QnnExecuTorchOpPackagePlatform.AARCH64_ANDROID,
            ),
        ]
    )
    specs = generate_qnn_executorch_compiler_spec(
        QcomChipset.SM8975,
        generate_htp_compiler_spec(
            use_fp16=False, use_weight_sharing=not args.independent
        ),
        shared_buffer=False,
        op_package_options=package,
    )
    jobs = get_qnn_pass_manager_cls(
        QnnExecuTorchBackendType.kHtpBackend
    ).get_capture_program_passes()
    state_shapes = {(1, 8192, 4), (1, 32, 128, 128)}
    for length in (1, 128, 896, args.capacity - 1):
        state_shapes.update({(1, 2, 256, length), (1, 2, length, 256)})

    def tag(node):
        value = node.meta.get("val")
        if (
            isinstance(value, torch.Tensor)
            and tuple(value.shape) in state_shapes
            and (
                node.op == "placeholder"
                or any(user.op == "output" for user in node.users)
            )
        ):
            return torch.uint16

    jobs[TagQuantIO][QCOM_PASS_ACTIVATE_KEY] = True
    jobs[TagQuantIO][QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY]["get_quant_io_dtype_fn"] = tag
    manager = to_edge_transform_and_lower_to_qnn(
        models, examples, specs, passes_job=jobs
    )
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "methods.pte"
    with path.open("wb") as handle:
        manager.to_executorch().write_to_file(handle)
    program = deserialize_pte_binary(path.read_bytes()).program
    contexts, graphs = {}, {}
    for plan in program.execution_plan:
        instructions = [
            instruction.instr_args
            for chain in plan.chains
            for instruction in chain.instructions
        ]
        if len(instructions) != 1 or type(instructions[0]).__name__ != "DelegateCall":
            raise ValueError(f"{plan.name}: expected one QNN delegate")
        binary = bytes(
            qnn.StripProtocol(
                program.backend_delegate_data[plan.delegates[0].processed.index].data
            )
        )
        digest = hashlib.sha256(binary).hexdigest()
        filename = contexts.setdefault(digest, f"context{len(contexts)}.bin")
        (args.output / filename).write_bytes(binary)
        arguments = instructions[0].args
        count = len(plan.inputs)
        graphs[plan.name] = dict(
            context=filename,
            input_order=[arguments[:count].index(i) for i in plan.inputs],
            output_order=[arguments[count:].index(i) for i in plan.outputs],
        )
        order = graphs[plan.name]
        values = [
            len(order["input_order"]),
            len(order["output_order"]),
            *order["input_order"],
            *order["output_order"],
        ]
        (args.output / f"{plan.name}.io").write_bytes(
            struct.pack("<" + "I" * len(values), *values)
        )
    if not args.independent and len(contexts) != 1:
        raise ValueError("Shared methods must refer to one context")
    report = dict(
        layers=args.layers,
        capacity=args.capacity,
        router="w8a16_gptq" if args.router_qdq else "fp16",
        weight_sharing=not args.independent,
        prefill_core="chunked_mixed_fp16",
        io_order="QNN input_N_/output_N_ name IDs; resolve against runtime tensor metadata",
        graphs=graphs,
        contexts={v: k for k, v in contexts.items()},
    )
    (args.output / "GRAPHS.json").write_text(json.dumps(report, indent=2) + "\n")
    print("SHARED_CONTEXT_READY", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
