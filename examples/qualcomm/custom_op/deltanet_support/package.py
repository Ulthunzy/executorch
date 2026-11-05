"""QNN package registration shared by DeltaNet model exporters."""

from pathlib import Path

import torch
from executorch.backends.qualcomm.serialization.qc_schema import (
    QnnExecuTorchOpPackageInfo,
    QnnExecuTorchOpPackageOptions,
    QnnExecuTorchOpPackagePlatform,
    QnnExecuTorchOpPackageTarget,
)


def deltanet_package(
    host, device, *, fp16_state=False, state_vtcm=False, valid_length=False
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
    locations = (
        (
            str(Path(host).resolve()),
            QnnExecuTorchOpPackageTarget.CPU,
            QnnExecuTorchOpPackagePlatform.X86_64,
        ),
        (
            device,
            QnnExecuTorchOpPackageTarget.HTP,
            QnnExecuTorchOpPackagePlatform.AARCH64_ANDROID,
        ),
    )
    return QnnExecuTorchOpPackageOptions(
        op_package_infos=[
            QnnExecuTorchOpPackageInfo(
                op_package_name="DeltaNetOpPackage",
                interface_provider="DeltaNetOpPackageInterfaceProvider",
                custom_op_name=str(operator),
                qnn_op_type_name=name,
                op_package_path=path,
                target=target,
                platform=platform,
            )
            for operator, name in operators
            for path, target, platform in locations
        ]
    )
