"""Explicit affine conversions around an experimental FP32 GDN core."""

import torch

from executorch.backends.qualcomm.builders import op_to  # noqa: F401
from executorch.backends.qualcomm.builders.node_visitor_manager import (
    _node_visitor_dict,
    register_node_visitor,
)


boundary_library = torch.library.Library("moe_gdn_precision", "DEF")
boundary_library.define("float32_boundary(Tensor value) -> Tensor")


@torch.library.impl(boundary_library, "float32_boundary", "CompositeExplicitAutograd")
def float32_boundary_impl(value):
    if value.dtype != torch.float32:
        raise ValueError("GDN precision boundaries require logical FP32 tensors")
    return value.clone()


@register_node_visitor
class Float32Boundary(_node_visitor_dict["aten._to_copy.default"]):
    target = ["moe_gdn_precision.float32_boundary.default"]


float32_boundary = torch.ops.moe_gdn_precision.float32_boundary.default
