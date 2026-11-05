#!/usr/bin/env bash
set -euo pipefail
op_root=$(cd "$(dirname "$0")" && pwd)
exec bash "$op_root/../htp_op_package/build.sh" "$@"
