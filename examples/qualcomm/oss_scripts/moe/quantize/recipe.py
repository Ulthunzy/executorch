import json
import math


def validate_router_source(directory, damping):
    report = json.loads((directory / "REPORT.json").read_text())
    if (
        report.get("method") != "fixed_grid_gptq"
        or report.get("weight_bits") != 8
        or report.get("activation_bits") != 16
        or not math.isclose(report.get("damping", float("nan")), damping)
    ):
        raise ValueError(
            f"{directory}: Router/shared calibration differs from --model-config; "
            "run prepare_router.py with the selected recipe first"
        )
    return report


def validate_router_graph(module):
    import torch

    weights = [
        node
        for node in module.graph.nodes
        if node.target == torch.ops.quantized_decomposed.dequantize_per_channel.default
    ]
    if len(weights) != 5 or any(
        node.args[4:7] != (-127, 127, torch.int8) for node in weights
    ):
        raise ValueError(
            "The w8a16_gptq recipe requires five calibrated INT8 projection grids"
        )
