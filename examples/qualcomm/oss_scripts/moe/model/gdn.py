"""Audit calibrated GDN grids without changing weights or running a device."""

import argparse
import hashlib
import json
import operator
from pathlib import Path


def file_record(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.stat()
    return dict(
        path=str(path.resolve()),
        sha256=digest.hexdigest(),
        bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
    )


def grid_record(node):
    target = str(node.target)
    if target not in (
        "quantized_decomposed.quantize_per_tensor.default",
        "quantized_decomposed.dequantize_per_tensor.default",
    ):
        raise ValueError(f"Expected calibrated per-tensor grid at {node}: {target}")
    scale, zero, minimum, maximum, dtype = node.args[1:]
    if not isinstance(scale, (int, float)) or not isinstance(zero, int):
        raise ValueError(f"Expected scalar grid constants at {node}")
    tensor = node.meta.get("val")
    return dict(
        node=node.name,
        target=target,
        shape=list(tensor.shape),
        logical_dtype=str(tensor.dtype),
        scale=float(scale),
        torch_zero_point=zero,
        qnn_offset=-zero,
        qmin=minimum,
        qmax=maximum,
        storage_dtype=str(dtype),
        clip_min=(minimum - zero) * scale,
        clip_max=(maximum - zero) * scale,
        granularity="per_tensor",
    )


def layer_record(path):
    import torch

    module = torch.export.load(path).module()
    cores = [
        node
        for node in module.graph.nodes
        if str(node.target) == "qnn_custom_ops.deltanet_core.default"
    ]
    if len(cores) != 1:
        raise ValueError(
            f"Expected one calibrated GDN core in {path}, got {len(cores)}"
        )
    core = cores[0]
    names = ("query", "key", "value", "decay", "beta", "initial_state")
    inputs = {name: grid_record(node) for name, node in zip(names, core.args)}
    outputs = {}
    for user in core.users:
        if user.target != operator.getitem or user.args[1] not in (0, 1):
            raise ValueError(f"Unexpected GDN output: {user}")
        quantizers = [
            node
            for node in user.users
            if str(node.target) == "quantized_decomposed.quantize_per_tensor.default"
        ]
        if len(quantizers) != 1:
            raise ValueError(f"Expected one output grid after {user}")
        outputs[("attention", "final_state")[user.args[1]]] = grid_record(quantizers[0])
    state_in, state_out = inputs["initial_state"], outputs["final_state"]
    keys = ("scale", "torch_zero_point", "qmin", "qmax", "storage_dtype")
    return dict(
        source=file_record(path),
        inputs=inputs,
        outputs=outputs,
        state_grid_shared=all(state_in[key] == state_out[key] for key in keys),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Directory containing calibrated layerN/front-qdq.pt2",
    )
    parser.add_argument(
        "--layers", type=int, nargs="+", default=[i for i in range(40) if i % 4 != 3]
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qnn-binding", type=Path)
    args = parser.parse_args()
    from executorch.examples.qualcomm.oss_scripts.moe.wrappers.binding import (
        load_binding,
    )

    load_binding(args.qnn_binding)
    import torch
    import executorch.examples.qualcomm.custom_op.deltanet  # noqa: F401

    torch.set_num_threads(2)
    report = dict(
        version=1,
        scope="Calibrated FP32-reference QDQ graphs, before export-only core replacement",
        source=str(args.source.resolve()),
        layers={},
    )
    for layer in args.layers:
        if layer not in range(40) or layer % 4 == 3:
            parser.error("Select GDN layers in 0..39 (excluding every fourth layer)")
        record = layer_record(args.source / f"layer{layer}/front-qdq.pt2")
        report["layers"][str(layer)] = record
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            "GDN_GRID_READY",
            layer,
            record["inputs"]["initial_state"]["scale"],
            flush=True,
        )


if __name__ == "__main__":
    main()
