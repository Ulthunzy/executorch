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
"$task_cxx" -std=c++17 -O1 -I"$task_repo/.." -I"$task_runtime/include" \
  -I"$task_repo/runtime/core/portable_type/c10" \
  "$moe_root/runner/tests/test_runtime_settings.cpp" \
  "$moe_root/runner/chat/settings.cpp" "$moe_root/runner/chat/options.cpp" \
  "$moe_root/runner/runtime_utils.cpp" -o "$task_build/settings"
"$task_build/settings"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$task_repo/..${PYTHONPATH:+:$PYTHONPATH}" \
  "$task_python" -m pytest -q -o addopts= "$moe_root/runner/tests/test_padding.py"
