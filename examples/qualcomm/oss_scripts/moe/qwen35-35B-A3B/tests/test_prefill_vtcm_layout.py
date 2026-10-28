import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("arch", [79, 81, 85])
def test_prefill_vtcm_regions_and_resource_bounds(tmp_path, arch):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("A host C++17 compiler is required")
    source = Path(__file__).with_name("prefill_vtcm_layout_test.cpp")
    binary = tmp_path / "layout-test"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            f"-DSM_MOE_DSP_ARCH={arch}",
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    assert f"arch={arch}" in result.stdout
