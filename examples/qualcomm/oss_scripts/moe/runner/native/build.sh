#!/usr/bin/env bash
set -euo pipefail
native_root=$(cd "$(dirname "$0")" && pwd)
task_repo=$(cd "$native_root/../../../../../.." && pwd)
task_arch=85
task_build=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --arch) task_arch=${2#v}; shift 2 ;;
    --output) task_build=$2; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
case "$task_arch" in 79|81|85) ;; *) echo "Unsupported HTP architecture: $task_arch" >&2; exit 2 ;; esac
: "${task_build:?Pass --output for the operator build directory}"
: "${QNN_SDK_ROOT:?Set QNN_SDK_ROOT}"
: "${HEXAGON_SDK_ROOT:?Set HEXAGON_SDK_ROOT}"
: "${HEXAGON_TOOLS_ROOT:?Set HEXAGON_TOOLS_ROOT}"
: "${ANDROID_NDK_ROOT:?Set ANDROID_NDK_ROOT}"
mkdir -p "$task_build"
task_build=$(cd "$task_build" && pwd)
if [[ -z "${MOE_BUILD_LOCK_FD:-}" ]]; then
  exec {MOE_BUILD_LOCK_FD}>"$task_build/.build.lock"
fi
if ! flock -n "$MOE_BUILD_LOCK_FD"; then
  echo "Another build is using $task_build" >&2
  exit 1
fi
case "${MOE_NATIVE_PROFILE:-0}" in 0|1) ;; *) echo "MOE_NATIVE_PROFILE must be 0 or 1" >&2; exit 2 ;; esac
rm -f "$task_build/TARGET.json" "$task_build/NATIVE_TARGET.json"
ggml_root=${LLAMA_CPP_ROOT:-"$task_repo/../llama.cpp"}
ggml_htp="$ggml_root/ggml/src/ggml-hexagon/htp"
if [[ ! -f "$ggml_htp/hvx-sigmoid.h" ]]; then
  echo "Set LLAMA_CPP_ROOT to the Qualcomm llama.cpp checkout containing hvx-sigmoid.h" >&2
  exit 1
fi

qaic="$HEXAGON_SDK_ROOT/ipc/fastrpc/qaic/Ubuntu/qaic"
compiler="$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android28-clang++"
native_flags=(-O3 -fPIC -mhmx -mhvx -mhvx-length=128B -mv"$task_arch" -DQCOM_HTP_ARCH="$task_arch"
  -I"$QNN_SDK_ROOT/include/QNN" -I"$ggml_htp" -I"$ggml_htp/.."
  -I"$HEXAGON_SDK_ROOT/incs" -I"$HEXAGON_SDK_ROOT/incs/stddef"
  -I"$HEXAGON_SDK_ROOT/rtos/qurt/computev$task_arch/include/qurt"
  -I"$HEXAGON_SDK_ROOT/rtos/qurt/computev$task_arch/include/posix"
  -Drestrict=__restrict -DSM_EXPERT_BITS=2 -DSM_MOE_PROFILE=${MOE_NATIVE_PROFILE:-0}
  -DSM_MOE_STAGE_READY=0 -DSM_MOE_ACCUMULATE=0 -DSM_MOE_ROUTE_VTCM=1
  -DSM_MOE_HMX_BATCH=8 -DPREPARE_DISABLED -DUSE_OS_QURT -DNATIVE_RPC)

objects=()
for stage in prefill decode; do
  generated="$task_build/native-$stage"
  kernel_root="$task_repo/examples/qualcomm/custom_op/lowbit"
  interface="et_${stage}_rpc"
  native_sources=("$native_root/$stage/integer_rpc_impl.cpp"
    "$native_root/$stage/bank_mapping.cpp" "$kernel_root/$stage/ffn.cpp")
  if [[ "$stage" == decode ]]; then
    native_sources+=("$native_root/decode/bank_mapper.cpp")
  fi
  mkdir -p "$generated"
  "$qaic" -mdll -o "$generated" -I"$HEXAGON_SDK_ROOT/incs" \
    -I"$HEXAGON_SDK_ROOT/incs/stddef" "$native_root/$stage/$interface.idl"
  "$HEXAGON_TOOLS_ROOT/bin/hexagon-clang" "${native_flags[@]}" -I"$generated" \
    -c "$generated/${interface}_skel.c" -o "$generated/${interface}_skel.o"
  objects+=("$generated/${interface}_skel.o")
  for source in "${native_sources[@]}"; do
    object="$generated/$(basename "${source%.cpp}").o"
    "$HEXAGON_TOOLS_ROOT/bin/hexagon-clang++" "${native_flags[@]}" \
      -I"$generated" -std=c++17 -c "$source" -o "$object"
    objects+=("$object")
  done
  "${compiler%++}" -O3 -fPIC -I"$HEXAGON_SDK_ROOT/incs" \
    -I"$HEXAGON_SDK_ROOT/incs/stddef" -c "$generated/${interface}_stub.c" \
    -o "$generated/${interface}_stub.o"
done
mkdir -p "$task_build/native-common"
for source in "$kernel_root/N2Utils.cpp" "$native_root/worker_start.cpp"; do
  object="$task_build/native-common/$(basename "${source%.cpp}").o"
  "$HEXAGON_TOOLS_ROOT/bin/hexagon-clang++" "${native_flags[@]}" \
    -std=c++17 -c "$source" -o "$object"
  objects+=("$object")
done
op_root="$task_repo/examples/qualcomm/custom_op/htp_op_package"
make -C "$op_root" WORK="$task_build/op-package" htp_x86 \
  "NATIVE_OBJECTS=${objects[*]}" \
  "$task_build/op-package/hexagon-v$task_arch/libet_qnnop.so"
cp "$task_build/op-package/hexagon-v$task_arch/libet_qnnop.so" "$task_build/libet_qnnop.so"
"${MOE_PYTHON:-python}" "$native_root/../../build_manifest.py" native --output "$task_build" \
  --arch "$task_arch" --profile "${MOE_NATIVE_PROFILE:-0}" --sdk "$QNN_SDK_ROOT"
