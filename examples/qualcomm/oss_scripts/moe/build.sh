#!/usr/bin/env bash
set -euo pipefail
moe_root=$(cd "$(dirname "$0")" && pwd -P)
task_repo=$(cd "$moe_root/../../../.." && pwd -P)
task_build=${MOE_BUILD_DIR:-"$task_repo/build/moe/qwen35-35B-A3B"}
mkdir -p "$task_build"
task_build=$(cd "$task_build" && pwd -P)
exec {task_build_lock}>"$task_build/.build.lock"
if ! flock -n "$task_build_lock"; then
  echo "Another build is using $task_build" >&2
  exit 1
fi
: "${QNN_SDK_ROOT:?Set QNN_SDK_ROOT}"
: "${HEXAGON_SDK_ROOT:?Set HEXAGON_SDK_ROOT}"
: "${HEXAGON_TOOLS_ROOT:?Set HEXAGON_TOOLS_ROOT}"
: "${ANDROID_NDK_ROOT:?Set ANDROID_NDK_ROOT}"
task_soc=${MOE_SOC_MODEL:-SM8975}
task_arch=$(PYTHONPATH="$task_repo/..${PYTHONPATH:+:$PYTHONPATH}" "${MOE_PYTHON:-python}" "$moe_root/target.py" --soc-model "$task_soc")
case "${MOE_NATIVE_PROFILE:-0}" in 0|1) ;; *) echo "MOE_NATIVE_PROFILE must be 0 or 1" >&2; exit 2 ;; esac
case "${MOE_BUILD_NATIVE:-1}" in 0|1) ;; *) echo "MOE_BUILD_NATIVE must be 0 or 1" >&2; exit 2 ;; esac
export MOE_SOC_MODEL="$task_soc" MOE_DSP_ARCH="$task_arch"
rm -f "$task_build/TARGET.json"
if [[ "${MOE_BUILD_NATIVE:-1}" == 1 ]]; then
  MOE_BUILD_LOCK_FD=$task_build_lock bash "$moe_root/runner/native/build.sh" --arch "$task_arch" --output "$task_build"
fi
cmake -S "$moe_root" -B "$task_build/cmake" -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 -DANDROID_STL=c++_static \
  -DCMAKE_BUILD_TYPE=Release -DQNN_SDK_ROOT="$QNN_SDK_ROOT" -DHEXAGON_SDK_ROOT="$HEXAGON_SDK_ROOT" \
  -DMOE_NATIVE_BUILD="$task_build" -DMOE_DSP_ARCH="$task_arch" \
  -DMOE_NATIVE_PROFILE="${MOE_NATIVE_PROFILE:-0}" -DMOE_BUILD_TOOLS=ON \
  -DEXECUTORCH_ANDROID_BUILD="${EXECUTORCH_ANDROID_BUILD:-$task_repo/build-android}" \
  -DCMAKE_RUNTIME_OUTPUT_DIRECTORY="$task_build" -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="$task_build"
cmake --build "$task_build/cmake" -j "${MOE_BUILD_JOBS:-3}"
"${MOE_PYTHON:-python}" "$moe_root/build_manifest.py" runtime --output "$task_build" \
  --arch "$task_arch" --soc "$task_soc" --profile "${MOE_NATIVE_PROFILE:-0}" --sdk "$QNN_SDK_ROOT"

for legacy in libmoe_decode.so libqwen35_shared.so libqwen35_moe_jni.so; do
  rm -f "$task_build/$legacy"
done

echo "Built MoE runners in $task_build"
