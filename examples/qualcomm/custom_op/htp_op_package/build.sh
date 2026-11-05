#!/usr/bin/env bash
set -euo pipefail
op_root=$(cd "$(dirname "$0")" && pwd)
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
make -C "$op_root" WORK="$task_build" htp_x86 "htp_v$task_arch"
