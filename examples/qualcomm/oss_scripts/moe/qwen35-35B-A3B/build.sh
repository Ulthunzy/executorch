#!/usr/bin/env bash
set -euo pipefail

moe_root=$(cd "$(dirname "$0")" && pwd)
task_repo=$(cd "$moe_root/../../../../.." && pwd)
task_build=${MOE_BUILD_DIR:-"$task_repo/build/moe/qwen35-35B-A3B"}
mkdir -p "$task_build"
task_build=$(cd "$task_build" && pwd)
runtime_build=${EXECUTORCH_ANDROID_BUILD:-"$task_repo/build-android"}
: "${QNN_SDK_ROOT:?Set QNN_SDK_ROOT to the QNN 2.48.1 SDK}"
: "${HEXAGON_SDK_ROOT:?Set HEXAGON_SDK_ROOT}"
: "${HEXAGON_TOOLS_ROOT:?Set HEXAGON_TOOLS_ROOT}"
: "${ANDROID_NDK_ROOT:?Set ANDROID_NDK_ROOT}"
ggml_root=${LLAMA_CPP_ROOT:-"$task_repo/../llama.cpp"}
ggml_htp="$ggml_root/ggml/src/ggml-hexagon/htp"
if [[ ! -f "$ggml_htp/hvx-sigmoid.h" ]]; then
  echo "Set LLAMA_CPP_ROOT to the Qualcomm llama.cpp checkout containing hvx-sigmoid.h" >&2
  exit 1
fi

qaic="$HEXAGON_SDK_ROOT/ipc/fastrpc/qaic/Ubuntu/qaic"
compiler="$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android28-clang++"
native_flags=(-O3 -fPIC -mhmx -mhvx -mhvx-length=128B -mv85
  -I"$QNN_SDK_ROOT/include/QNN" -I"$ggml_htp" -I"$ggml_htp/.."
  -I"$HEXAGON_SDK_ROOT/incs" -I"$HEXAGON_SDK_ROOT/incs/stddef"
  -I"$HEXAGON_SDK_ROOT/rtos/qurt/computev85/include/qurt"
  -I"$HEXAGON_SDK_ROOT/rtos/qurt/computev85/include/posix"
  -Drestrict=__restrict -DSM_EXPERT_BITS=2 -DSM_MOE_PROFILE=0
  -DSM_MOE_STAGE_READY=0 -DSM_MOE_ACCUMULATE=0 -DSM_MOE_ROUTE_VTCM=1
  -DSM_MOE_HMX_BATCH=8 -DPREPARE_DISABLED -DUSE_OS_QURT -DNATIVE_RPC)

for stage in prefill decode; do
  generated="$task_build/native-$stage"
  native_sources=("$moe_root/native/$stage/integer_rpc_impl.cpp"
    "$moe_root/native/$stage/IntegerMoE.cpp" "$moe_root/native/N2Utils.cpp"
    "$moe_root/native/worker_start.cpp")
  if [[ "$stage" == decode ]]; then
    native_sources+=("$moe_root/native/decode/bank_mapper.cpp")
  fi
  mkdir -p "$generated"
  "$qaic" -mdll -o "$generated" -I"$HEXAGON_SDK_ROOT/incs" \
    -I"$HEXAGON_SDK_ROOT/incs/stddef" "$moe_root/native/$stage/integer_rpc.idl"
  "$HEXAGON_TOOLS_ROOT/bin/hexagon-clang" "${native_flags[@]}" -I"$generated" \
    -c "$generated/integer_rpc_skel.c" -o "$generated/integer_rpc_skel.o"
  "$HEXAGON_TOOLS_ROOT/bin/hexagon-clang++" "${native_flags[@]}" -I"$generated" \
    -std=c++17 -shared -nostdlib++ "${native_sources[@]}" \
    "$generated/integer_rpc_skel.o" -Wl,-Bstatic -lc++ -lc++abi -Wl,-Bdynamic \
    -o "$generated/libinteger_rpc_skel.so"

  "${compiler%++}" -O3 -fPIC -I"$HEXAGON_SDK_ROOT/incs" \
    -I"$HEXAGON_SDK_ROOT/incs/stddef" -c "$generated/integer_rpc_stub.c" \
    -o "$generated/integer_rpc_stub.o"
  host_flags=()
  host_output="$task_build/${stage}_handoff_runner"
  if [[ "$stage" == decode ]]; then
    host_flags=(-fPIC -shared -fvisibility=hidden -fvisibility-inlines-hidden
      -DMOE_DECODE_LIBRARY -Wl,-Bsymbolic -Wl,--exclude-libs,ALL
      -Wl,--version-script="$moe_root/runner/decode.exports")
    host_output="$task_build/libmoe_decode.so"
  fi
  "$compiler" -O3 -std=c++17 -static-libstdc++ -DC10_USING_CUSTOM_GENERATED_MACROS \
    -DET_EVENT_TRACER_ENABLED -DET_ENABLE_DEPRECATED_CONSTANT_BUFFER=0 \
    -DET_ENABLE_PROGRAM_VERIFICATION=0 -I"$task_repo/.." \
    -I"$task_repo/runtime/core/portable_type/c10" -I"$runtime_build/include" \
    -I"$generated" -I"$HEXAGON_SDK_ROOT/incs" -I"$HEXAGON_SDK_ROOT/incs/stddef" \
    -I"$QNN_SDK_ROOT/include/QNN" "${host_flags[@]}" "$moe_root/runner/$stage/"*.cpp \
    "$moe_root/runner/"*.cpp "$moe_root/runner/shared/graph_binding.cpp" \
    "$generated/integer_rpc_stub.o" -Wl,--start-group \
    "$runtime_build/libexecutorch.a" "$runtime_build/libexecutorch_core.a" \
    "$runtime_build/extension/module/libextension_module.a" \
    "$runtime_build/extension/data_loader/libextension_data_loader.a" \
    "$runtime_build/extension/tensor/libextension_tensor.a" \
    "$runtime_build/extension/flat_tensor/libextension_flat_tensor.a" \
    "$runtime_build/extension/named_data_map/libextension_named_data_map.a" -Wl,--end-group \
    -L"$HEXAGON_SDK_ROOT/ipc/fastrpc/remote/ship/android_aarch64" \
    -lcdsprpc -ldl -llog -o "$host_output"
done
"$compiler" -O3 -std=c++17 -static-libstdc++ -fPIC -shared -fvisibility=hidden \
  -fvisibility-inlines-hidden -Wl,-Bsymbolic -Wl,--exclude-libs,ALL \
  -I"$QNN_SDK_ROOT/include/QNN" "$moe_root/runner/shared/context_store.cpp" \
  "$moe_root/runner/shared/weight_source.cpp" "$moe_root/runner/shared/graph_binding.cpp" \
  "$moe_root/runner/decode/qnn_front.cpp" -ldl -o "$task_build/libqwen35_shared.so"
"$compiler" -O2 -std=c++17 -static-libstdc++ -I"$QNN_SDK_ROOT/include/QNN" \
  "$moe_root/tools/capture_weights.cpp" "$moe_root/runner/decode/qnn_front.cpp" \
  "$moe_root/runner/shared/graph_binding.cpp" -ldl -o "$task_build/capture_weights"
"$moe_root/build_op_package.sh"
echo "Built runners and V85 Native MoE libraries in $task_build"
