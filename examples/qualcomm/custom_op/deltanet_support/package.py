"""QNN package registration shared by DeltaNet model exporters."""

import torch
from executorch.examples.qualcomm.custom_op.htp_op_package.package import op_package


def deltanet_package(
    host,
    device,
    *,
    fp16_state=False,
    state_vtcm=False,
    valid_length=False,
):
    import executorch.examples.qualcomm.custom_op.deltanet  # noqa: F401

    operators = [
        (
            torch.ops.qnn_custom_ops.deltanet_core.default,
            "DeltaNetCoreVtcm" if state_vtcm else "DeltaNetCore",
        ),
    ]
    if fp16_state:
        operators.append(
            (
                torch.ops.qnn_custom_ops.deltanet_core_fp16_state.default,
                "DeltaNetCoreFp16State",
            )
        )
    if valid_length:
        operators.append(
            (
                torch.ops.qnn_custom_ops.deltanet_core_valid.default,
                "DeltaNetCoreValidVtcm" if state_vtcm else "DeltaNetCoreValid",
            )
        )
    return op_package(host, device, operators)
