#!/usr/bin/env bash
set -euo pipefail
moe_root=$(cd "$(dirname "$0")/.." && pwd)
task_repo=$(cd "$moe_root/../../../../.." && pwd)
task_build=${MOE_BUILD_DIR:-"$task_repo/build/moe/qwen35-35B-A3B"}
mkdir -p "$task_build"
task_build=$(cd "$task_build" && pwd)
: "${QNN_SDK_ROOT:?Set QNN_SDK_ROOT}"
: "${HEXAGON_SDK_ROOT:?Set HEXAGON_SDK_ROOT}"
: "${HEXAGON_TOOLS_ROOT:?Set HEXAGON_TOOLS_ROOT}"
: "${ANDROID_NDK_ROOT:?Set ANDROID_NDK_ROOT}"
task_arch=$(PYTHONPATH="$task_repo/..${PYTHONPATH:+:$PYTHONPATH}" \
  "${MOE_PYTHON:-python}" "$moe_root/workflow/target.py" \
  --soc-model "${MOE_SOC_MODEL:-SM8975}")
make -C "$task_repo/examples/qualcomm/custom_op/deltanet_op_package_htp/DeltaNetOpPackage" \
  -f Makefile -f "$moe_root/native/gdn_arch.mk" WORK="$task_build/op-package" MOE_DSP_ARCH="$task_arch" htp_x86 "htp_v$task_arch"
