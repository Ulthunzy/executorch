"""QNN registration for valid-length causal Attention."""

import torch
from executorch.examples.qualcomm.custom_op.htp_op_package.package import op_package


def attention_package(host, device):
    import executorch.examples.qualcomm.custom_op.attention  # noqa: F401

    return op_package(
        host,
        device,
        [(torch.ops.qnn_custom_ops.attention_valid.default, "AttentionValid")],
    )
