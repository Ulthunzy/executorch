from pathlib import Path
import shutil
import subprocess

import pytest


def test_decode_rejects_unnegotiated_or_mismatched_cache_abi(tmp_path):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    root = Path(__file__).resolve().parents[1]
    source = (root / "runner/decode/decoder_api.cpp").read_text()
    query = source[source.index('extern "C" size_t qwen35_decode_session_state_size') :]
    query = query[: query.index("\n}\n") + 3]
    opening = source.split('extern "C" int qwen35_decode_open', 1)[1]
    guard = opening.split("try {", 1)[1].split("  if (session || module)", 1)[0]
    harness = tmp_path / "abi.cpp"
    harness.write_text(
        '#include "runner/session_state.h"\n'
        "#include <cassert>\n#include <stdexcept>\n"
        "bool cache_abi_checked = false;\n"
        + query
        + "\nvoid opening() {"
        + guard
        + "\n}\n"
        + r"""
int main() {
  for (size_t caller : {size_t(0), sizeof(Qwen35SessionState) - 16,
                        sizeof(Qwen35SessionState), size_t(1)}) {
    if (caller)
      assert(qwen35_decode_session_state_size(caller) == sizeof(Qwen35SessionState));
    bool rejected = false;
    try { opening(); }
    catch (const std::runtime_error&) { rejected = true; }
    assert(rejected == (caller != sizeof(Qwen35SessionState)));
  }
}
"""
    )
    binary = tmp_path / "abi"
    subprocess.run(
        [compiler, "-std=c++17", "-I", str(root), str(harness), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)
