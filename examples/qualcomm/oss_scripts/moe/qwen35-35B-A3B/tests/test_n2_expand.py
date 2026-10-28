import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("arch", [79, 81, 85])
def test_n2_expand_matches_signed_n4(tmp_path, arch):
    tools = os.environ.get("HEXAGON_TOOLS_ROOT")
    if not tools:
        pytest.skip("HEXAGON_TOOLS_ROOT with SDK libnative is required")
    native = Path(tools) / "libnative"
    if not (native / "include/hvx_hexagon_protos.h").is_file() or not (
        native / "lib/libnative.a"
    ).is_file():
        pytest.skip("SDK libnative headers and library are required")
    compiler = shutil.which("clang++")
    if compiler is None:
        pytest.skip("A host clang++ compiler is required")
    source = Path(__file__).with_name("n2_expand_test.cpp")
    binary = tmp_path / "n2-expand-test"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-D__HVXDBL__",
            "-DUSE_OS_LINUX",
            f"-D__HEXAGON_ARCH__={arch}",
            f"-I{native / 'include'}",
            "-fsanitize=address,undefined",
            "-fno-sanitize-recover=all",
            "-fno-omit-frame-pointer",
            str(source),
            str(source.parent.parent / "native/N2Utils.cpp"),
            f"-L{native / 'lib'}",
            "-Wl,--whole-archive",
            "-lnative",
            "-Wl,--no-whole-archive",
            "-pthread",
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    environment = dict(os.environ)
    environment["ASAN_OPTIONS"] = (
        environment.get("ASAN_OPTIONS", "") + ":detect_leaks=0"
    ).lstrip(":")
    result = subprocess.run(
        [str(binary)],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert f"PASS N2 arch={arch} cases=3329" in result.stdout
