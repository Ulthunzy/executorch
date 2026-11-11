# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import getpass
import subprocess
from pathlib import Path

from executorch.backends.qualcomm.serialization.qc_schema import (
    _soc_info_table,
    QcomChipset,
)
from executorch.examples.qualcomm.custom_op.deltanet import setup_op_package

DEFAULT_OP_PACKAGE_DIR = (
    Path(__file__).resolve().parents[2]
    / "custom_op/deltanet_op_package_htp/DeltaNetOpPackage"
)
DEVICE_WORKSPACE = f"/data/local/tmp/{getpass.getuser()}/executorch/static_llm"


def setup_gdn_op_package(args):
    arch = _soc_info_table[QcomChipset[args.soc_model]].htp_info.htp_arch
    if arch < 79:
        raise ValueError("Qwen3.5 requires an HTP v79 or newer target")
    reduction_flags = (
        "-DDELTANET_U16_FULL_PRECISION_REDUCTION=0 "
        "-DDELTANET_U16_SPLIT_REDUCTION=1 "
        "-DDELTANET_U16_KARATSUBA_REDUCTION=0"
    )
    subprocess.run(
        [
            "make",
            f"X86_CPPFLAGS={reduction_flags}",
            f"HEXAGON_CPPFLAGS={reduction_flags}",
            "htp_x86",
            "htp_aarch64",
            f"htp_v{arch}",
        ],
        cwd=args.gdn_op_package_dir,
        check=True,
    )
    package_args = argparse.Namespace(
        op_package_dir=args.gdn_op_package_dir,
        build_op_package=False,
        soc_model=args.soc_model,
        workspace=DEVICE_WORKSPACE,
    )
    options, paths = setup_op_package(package_args, state_vtcm=True)
    if arch == 81:
        from executorch.examples.qualcomm.custom_op.attention_support.package import (
            attention_package,
        )
        from executorch.examples.qualcomm.custom_op.deltanet_support.chunk import (
            chunk_package,
        )
        from executorch.examples.qualcomm.custom_op.deltanet_support.decode_gate import (
            decode_gate_package,
        )
        from executorch.examples.qualcomm.custom_op.deltanet_support.state_update import (
            state_update_package,
        )

        host = (
            Path(args.gdn_op_package_dir)
            / "build/x86_64-linux-clang/libQnnDeltaNetOpPackage.so"
        )
        device = f"{DEVICE_WORKSPACE}/libQnnDeltaNetOpPackage_HTP.so"
        for package in (
            chunk_package,
            decode_gate_package,
            state_update_package,
            attention_package,
        ):
            options.op_package_infos.extend(package(host, device).op_package_infos)
    required_paths = [
        *paths,
        str(
            Path(args.gdn_op_package_dir)
            / "build/x86_64-linux-clang/libQnnDeltaNetOpPackage.so"
        ),
    ]
    missing = [path for path in required_paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing DeltaNet op package libraries:\n" + "\n".join(missing)
        )
    return options, paths


def qwen35_prefill_capacity(max_seq_len, max_context_len):
    capacity = max(256, ((max_seq_len + 127) // 128) * 128)
    if max_context_len < capacity:
        raise ValueError(
            f"Qwen3.5 AR128 requires max_context_len >= {capacity} "
            f"for max_seq_len={max_seq_len}; received {max_context_len}."
        )
    return capacity


def is_qwen35_model(decoder_model):
    return decoder_model.startswith(("qwen3_5", "lincal3_5"))
