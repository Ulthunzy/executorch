#!/usr/bin/env bash
set -euo pipefail
moe_root=$(cd "$(dirname "$0")/../.." && pwd)
task_repo=$(cd "$moe_root/../../../../.." && pwd)
task_build=${MOE_TEST_BUILD:-$(mktemp -d /tmp/qwen35-host-tests-XXXXXX)}
mkdir -p "$task_build"
: "${QNN_SDK_ROOT:?Set QNN_SDK_ROOT}"
task_cxx=${CXX:-c++}
task_python=${MOE_TEST_PYTHON:-python}
task_runtime=${EXECUTORCH_ANDROID_BUILD:-"$task_repo/build-android"}
"$task_cxx" -std=c++17 -O1 -pthread -I"$QNN_SDK_ROOT/include/QNN" \
  "$moe_root/runner/tests/test_context_store.cpp" -o "$task_build/context"
"$task_build/context"
"$task_cxx" -std=c++17 -O2 "$moe_root/runner/tests/test_cache_transfer.cpp" \
  "$moe_root/runner/cache_transfer.cpp" -o "$task_build/cache"
"$task_build/cache"
"$task_cxx" -std=c++17 -O2 -I"$QNN_SDK_ROOT/include/QNN" \
  "$moe_root/runner/tests/test_cache_encoding.cpp" -o "$task_build/cache-encoding"
"$task_build/cache-encoding"
"$task_cxx" -std=c++17 -O1 -I"$task_runtime/include" \
  "$moe_root/runner/tests/test_chat_terminal.cpp" \
  "$moe_root/runner/chat/terminal.cpp" \
  -o "$task_build/terminal"
"$task_build/terminal"
for task_arch in 79 81 85; do
  "$task_cxx" -std=c++17 -O1 -DC10_USING_CUSTOM_GENERATED_MACROS \
    -DSM_MOE_DSP_ARCH="$task_arch" \
    -I"$task_repo/.." -I"$task_runtime/include" \
    -I"$task_repo/runtime/core/portable_type/c10" \
    -I"$task_repo/third-party/flatbuffers/include" -I"$task_repo/build-x86/schema/include" \
    "$moe_root/runner/tests/test_runtime_settings.cpp" \
    "$moe_root/runner/chat/settings.cpp" "$moe_root/runner/chat/options.cpp" \
    "$moe_root/runner/runtime_utils.cpp" "$moe_root/runner/model_package.cpp" \
    "$task_repo/build-x86/libexecutorch_core.a" -lz -pthread -ldl \
    -o "$task_build/settings-v$task_arch"
  "$task_build/settings-v$task_arch"
done
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$task_repo/..${PYTHONPATH:+:$PYTHONPATH}" \
  "$task_python" -m pytest -q -o addopts= "$moe_root/runner/tests/test_padding.py"
