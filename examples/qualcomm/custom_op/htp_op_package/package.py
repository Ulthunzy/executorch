"""Registration for the shared HTP operator bundle."""

from pathlib import Path

from executorch.backends.qualcomm.serialization.qc_schema import (
    QnnExecuTorchOpPackageInfo,
    QnnExecuTorchOpPackageOptions,
    QnnExecuTorchOpPackagePlatform,
    QnnExecuTorchOpPackageTarget,
)


def op_package(host, device, operators):
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
