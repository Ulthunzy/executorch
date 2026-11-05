import argparse
import hashlib
import json
from pathlib import Path
import struct
from dataclasses import replace


# Internal stage scripts also support direct execution for diagnostics.

from executorch.examples.qualcomm.oss_scripts.moe.model.qwen35 import (
    add_model_config_argument,
    load_model_config,
)


def main():
    parser = argparse.ArgumentParser(
        description="Compile a small group of AR128/AR1 layers with shared constants"
    )
    add_model_config_argument(parser)
    parser.add_argument("--prefill-source", type=Path, required=True)
    parser.add_argument("--decode-source", type=Path, required=True)
    parser.add_argument("--router-qdq", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+")
    parser.add_argument("--max-seq-len", "--capacity", dest="capacity", type=int)
    parser.add_argument("--qnn-binding", type=Path)
    parser.add_argument("--op-package-host", type=Path, required=True)
    parser.add_argument("--op-package-device", required=True)
    parser.add_argument(
        "--prefill-gdn-core",
        choices=("chunked_mixed_fp16", "custom_fp16", "custom_fp32", "custom_u16"),
        default="chunked_mixed_fp16",
    )
    parser.add_argument("--gdn-state-vtcm", action="store_true")
    args = parser.parse_args()
    if args.gdn_state_vtcm and args.prefill_gdn_core in ("custom_fp16", "custom_fp32"):
        parser.error("GDN VTCM layout requires a UINT16 core")
    args.layers = args.layers or [0, 1, 2, 3]
    settings = load_model_config(args.model_config)
    fp16_state = settings.cache.gdn_state_dtype == "float16"
    if fp16_state and (
        args.prefill_gdn_core != "chunked_mixed_fp16" or args.gdn_state_vtcm
    ):
        parser.error(
            "Typed FP16 GDN state currently requires chunked prefill and DDR state"
        )
    args.capacity = args.capacity or settings.context.max_seq_len
    settings = replace(
        settings, context=replace(settings.context, max_seq_len=args.capacity)
    ).validate()
    if bool(args.router_qdq) != (settings.quantization.router_shared == "w8a16_gptq"):
        parser.error(
            "w8a16_gptq needs --router-qdq; fp16 uses checkpoint constants without --router-qdq"
        )
    if len(set(args.layers)) != len(args.layers) or not 1 <= len(args.layers) <= 4:
        parser.error("Select one to four distinct layers")
    if any(layer not in range(40) for layer in args.layers):
        parser.error("Layer IDs must be 0..39")
    from executorch.examples.qualcomm.oss_scripts.moe.wrappers.binding import (
        load_binding,
    )

    qnn = load_binding(args.qnn_binding)
    import torch
    import executorch.examples.qualcomm.custom_op.deltanet  # noqa: F401
    from executorch.backends.qualcomm._passes.qnn_pass_manager import (
        get_qnn_pass_manager_cls,
    )
    from executorch.backends.qualcomm._passes.tag_quant_io import TagQuantIO
    from executorch.backends.qualcomm.serialization.qc_schema import (
        QcomChipset,
        QnnExecuTorchBackendType,
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
    from executorch.examples.qualcomm.oss_scripts.moe.model.attention import (
        fp16_attention_core,
    )
    from executorch.examples.qualcomm.oss_scripts.moe.wrappers.graph_utils import (
        resize_attention,
    )
    from executorch.examples.qualcomm.oss_scripts.moe.wrappers.prefill import (
        padded_prefill_front,
    )
    from executorch.examples.qualcomm.custom_op.deltanet_support.gdn_state import (
        fp16_gdn_state,
    )
    from executorch.examples.qualcomm.oss_scripts.moe.model.moe_layer import (
        FusedFront,
        MoeFront,
        QuantizedFusedFront,
    )
    from executorch.examples.qualcomm.oss_scripts.moe.quantize.recipe import (
        validate_router_graph,
        validate_router_source,
    )

    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    models, examples, router_calibration = {}, {}, {}
    for layer in args.layers:
        if settings.quantization.attention == "gptq":
            calibration = json.loads(
                (args.prefill_source / f"layer{layer}/GPTQ.json").read_text()
            )
            if (
                calibration.get("source") != "safetensors"
                or calibration.get("method") != "all_projection_gptq"
            ):
                raise ValueError(
                    f"Layer {layer} is missing fresh all-projection GPTQ provenance"
                )
        prefill = torch.export.load(
            args.prefill_source / f"layer{layer}/front-qdq.pt2"
        ).module()
        prefill_inputs = tuple(
            torch.zeros(node.meta["val"].shape)
            for node in prefill.graph.nodes
            if node.op == "placeholder"
        )
        if fp16_state and layer % 4 != 3:
            prefill, prefill_inputs = fp16_gdn_state(
                prefill, prefill_inputs, decode=False
            )
        prefill, prefill_inputs = padded_prefill_front(
            prefill, prefill_inputs, args.capacity, args.prefill_gdn_core
        )
        decode = torch.export.load(
            args.decode_source / f"layer{layer}/decode-qdq.pt2"
        ).module()
        decode_inputs = resize_attention(decode, args.capacity)
        if fp16_state and layer % 4 != 3:
            decode, decode_inputs = fp16_gdn_state(decode, decode_inputs, decode=True)
        if layer % 4 == 3:
            fp16_attention_core(decode, decode_inputs)
        constants = args.prefill_source / f"layer{layer}/moe-constants.npz"
        if args.router_qdq:
            router_calibration[layer] = validate_router_source(
                args.router_qdq / f"layer{layer}",
                settings.quantization.router_gptq_damping,
            )
            router = torch.export.load(
                args.router_qdq / f"layer{layer}/router-ar128-qdq.pt2"
            ).module()
            decode_router = torch.export.load(
                args.router_qdq / f"layer{layer}/router-ar1-qdq.pt2"
            ).module()
            validate_router_graph(router)
            validate_router_graph(decode_router)
            decoder = QuantizedFusedFront(decode, decode_router)
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

    from executorch.examples.qualcomm.custom_op.deltanet_support.package import (
        deltanet_package,
    )

    package = deltanet_package(
        args.op_package_host,
        args.op_package_device,
        fp16_state=fp16_state,
        state_vtcm=args.gdn_state_vtcm,
    )
    specs = generate_qnn_executorch_compiler_spec(
        QcomChipset[settings.target.soc_model],
        generate_htp_compiler_spec(use_fp16=False, use_weight_sharing=True),
        shared_buffer=True,
        op_package_options=package,
    )
    jobs = get_qnn_pass_manager_cls(
        QnnExecuTorchBackendType.kHtpBackend
    ).get_capture_program_passes()
    state_shapes = {(1, 8192, 4), (1, 32, 128, 128)}
    for length in (1, 128, args.capacity, args.capacity - 1):
        state_shapes.update({(1, 2, 256, length), (1, 2, length, 256)})

    def tag(node):
        value = node.meta.get("val")
        if (
            fp16_state
            and isinstance(value, torch.Tensor)
            and tuple(value.shape) == (1, 32, 128, 128)
        ):
            return None
        if (
            isinstance(value, torch.Tensor)
            and tuple(value.shape) in state_shapes
            and (
                node.op == "placeholder"
                or any(user.op == "output" for user in node.users)
            )
        ):
            return {"uint16": torch.uint16}[settings.cache.kv_dtype]

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
    if len(contexts) != 1:
        raise ValueError("Shared methods must refer to one context")
    from executorch.examples.qualcomm.oss_scripts.moe.utils import (
        validate_qnn_io_memory,
    )

    report = dict(
        **validate_qnn_io_memory(args.output / next(iter(contexts.values()))),
        target=settings.target.to_dict(),
        compile_settings=settings.compile_settings(),
        model_configuration=settings.to_dict(),
        layers=args.layers,
        capacity=args.capacity,
        router="w8a16_gptq" if args.router_qdq else "fp16",
        router_calibration=router_calibration,
        weight_sharing=True,
        prefill_core=args.prefill_gdn_core,
        gdn_state_dtype=settings.cache.gdn_state_dtype,
        gdn_state_vtcm=args.gdn_state_vtcm,
        prefill_valid_length=True,
        prefill_history=args.capacity,
        io_order="QNN input_N_/output_N_ name IDs; resolve against runtime tensor metadata",
        graphs=graphs,
        contexts={v: k for k, v in contexts.items()},
    )
    (args.output / "GRAPHS.json").write_text(json.dumps(report, indent=2) + "\n")
    print("SHARED_CONTEXT_READY", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
