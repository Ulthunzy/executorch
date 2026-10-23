#!/usr/bin/env bash
set -euo pipefail

test_root=$(cd "$(dirname "$0")" && pwd)
moe_root=$(cd "$test_root/../.." && pwd)
task_repo=$(cd "$moe_root/../../../../.." && pwd)
task_build=${MOE_BUILD_DIR:-"$task_repo/build/moe/qwen35-35B-A3B"}
test_build="$task_build/lock-tests"
mkdir -p "$test_build"
test_build=$(cd "$test_build" && pwd)
"${HOST_CXX:-c++}" -std=c++17 -O2 -pthread "$moe_root/native/worker_start.cpp" \
  "$test_root/worker_start_test.cpp" -o "$test_build/worker_start_test"
timeout 10 "$test_build/worker_start_test"

device=${1:-}
if [[ -z "$device" ]]; then
  exit 0
fi
: "${QNN_SDK_ROOT:?Set QNN_SDK_ROOT}"
: "${HEXAGON_SDK_ROOT:?Set HEXAGON_SDK_ROOT}"
: "${HEXAGON_TOOLS_ROOT:?Set HEXAGON_TOOLS_ROOT}"
: "${ANDROID_NDK_ROOT:?Set ANDROID_NDK_ROOT}"
ggml_root=${LLAMA_CPP_ROOT:-"$task_repo/../llama.cpp"}
ggml_htp="$ggml_root/ggml/src/ggml-hexagon/htp"
flags=(-O3 -fPIC -std=c++17 -mhmx -mhvx -mhvx-length=128B -mv85
  -I"$QNN_SDK_ROOT/include/QNN" -I"$ggml_htp" -I"$ggml_htp/.."
  -I"$HEXAGON_SDK_ROOT/incs" -I"$HEXAGON_SDK_ROOT/incs/stddef"
  -I"$HEXAGON_SDK_ROOT/rtos/qurt/computev85/include/qurt"
  -I"$HEXAGON_SDK_ROOT/rtos/qurt/computev85/include/posix"
  -Drestrict=__restrict -DSM_EXPERT_BITS=2 -DSM_MOE_PROFILE=0
  -DSM_MOE_STAGE_READY=0 -DSM_MOE_ACCUMULATE=0 -DSM_MOE_ROUTE_VTCM=1
  -DSM_MOE_HMX_BATCH=8 -DNATIVE_RPC -DPREPARE_DISABLED -DUSE_OS_QURT)
adb_command=(adb -s "$device")

for stage in prefill decode; do
  generated="$task_build/native-$stage"
  if [[ ! -f "$generated/integer_rpc_skel.o" || ! -f "$generated/integer_rpc_stub.o" ]]; then
    echo "Run build.sh first; missing generated RPC objects in $generated" >&2
    exit 1
  fi
  sources=("$moe_root/native/$stage/integer_rpc_impl.cpp"
    "$moe_root/native/$stage/IntegerMoE.cpp" "$moe_root/native/N2Utils.cpp"
    "$moe_root/native/worker_start.cpp" "$test_root/lock_failure_shim.cpp")
  ar=128
  hvx_skip=0
  if [[ "$stage" == decode ]]; then
    sources+=("$moe_root/native/decode/bank_mapper.cpp")
    ar=1
    hvx_skip=1
  fi
  probe="$test_build/$stage-probe"
  "$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android28-clang++" \
    -O2 -std=c++17 -static-libstdc++ -DMOE_TEST_AR="$ar" -I"$generated" \
    -I"$HEXAGON_SDK_ROOT/incs" -I"$HEXAGON_SDK_ROOT/incs/stddef" \
    "$test_root/lock_failure_probe.cpp" "$generated/integer_rpc_stub.o" \
    -L"$HEXAGON_SDK_ROOT/ipc/fastrpc/remote/ship/android_aarch64" -lcdsprpc -o "$probe"
  for kind in hvx hmx; do
    directory="$test_build/$stage-$kind"
    mkdir -p "$directory"
    extra=(-DMOE_TEST_HVX_SKIP="$hvx_skip" -Wl,--wrap=qurt_hvx_lock)
    if [[ "$kind" == hmx ]]; then
      extra=(-DMOE_TEST_HMX_FAILURE -Wl,--wrap=compute_resource_hmx_lock)
    fi
    "$HEXAGON_TOOLS_ROOT/bin/hexagon-clang++" "${flags[@]}" "${extra[@]}" \
      -I"$generated" -shared -nostdlib++ "${sources[@]}" "$generated/integer_rpc_skel.o" \
      -Wl,-Bstatic -lc++ -lc++abi -Wl,-Bdynamic -o "$directory/libinteger_rpc_skel.so"
    cp "$probe" "$directory/lock_failure_probe"
    remote="/data/local/tmp/qwen35-moe-lock-tests/$stage-$kind"
    "${adb_command[@]}" shell "mkdir -p $remote"
    "${adb_command[@]}" push "$directory/libinteger_rpc_skel.so" "$directory/lock_failure_probe" "$remote/"
    "${adb_command[@]}" shell "cd $remote && LD_LIBRARY_PATH=$remote:/vendor/lib64:/odm/lib64 ADSP_LIBRARY_PATH='$remote;/odm/lib64/rfsa;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/dsp' timeout 20 ./lock_failure_probe" \
      | tee "$directory/DEVICE.log"
  done
done
