# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Check isolated teacher forcing and compiler include precedence."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import build_driver
import pytest


@pytest.mark.parametrize("existing_hook", [False, True])
def test_overlay_switches_tokens_without_changing_source(tmp_path, existing_hook):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    include = "executorch/examples/qualcomm/oss_scripts/llama/runner/decoder_runner.h"
    source_root = tmp_path / "source"
    header = source_root / include
    header.parent.mkdir(parents=True)
    declaration = (
        "#ifdef ET_QNN_PPL_EVALUATION\nint32_t next_teacher_forcing_token();\n#endif\n"
    )
    branch = (
        "#ifdef ET_QNN_PPL_EVALUATION\nreturn next_teacher_forcing_token();\n#endif\n"
    )
    original = (
        "#pragma once\n#include <cstdint>\nnamespace example {\n"
        + (declaration if existing_hook else "")
        + "class DecoderRunner {\npublic:\ninline int32_t logits_to_token(int64_t pos) {\n"
        + (branch if existing_hook else "")
        + "return static_cast<int32_t>(pos);\n}\n};\n}\n"
    )
    header.write_text(original)
    overlay = build_driver.prepare_ppl_overlay(header.parents[1], tmp_path / "output")
    assert header.read_text() == original
    assert (overlay / include).read_text().count("next_teacher_forcing_token") == 2
    runner = header.with_name("runner.h")
    runner.write_text(f"#include <{include}>\n")
    entry = tmp_path / "entry.cpp"
    entry.write_text(
        f"#include <{runner.relative_to(source_root)}>\n"
        "namespace example { int32_t next_teacher_forcing_token() { return 73; } }\n"
        "int main() { return example::DecoderRunner().logits_to_token(27) == EXPECTED ? 0 : 1; }\n"
    )
    for enabled, expected in ((False, 27), (True, 73)):
        binary = tmp_path / f"driver-{enabled}"
        command = [compiler, "-std=c++17", "-I" + str(overlay), "-I" + str(source_root)]
        if enabled:
            command.append("-DET_QNN_PPL_EVALUATION")
        subprocess.run(
            command + [f"-DEXPECTED={expected}", str(entry), "-o", str(binary)],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run([str(binary)], check=True)


@pytest.mark.parametrize("ppl", [False, True])
def test_build_uses_canonical_sources_and_private_includes(tmp_path, monkeypatch, ppl):
    build = tmp_path / "build"
    target = (
        build / "examples/qualcomm/oss_scripts/llama/CMakeFiles/qnn_llama_runner.dir"
    )
    target.mkdir(parents=True)
    prefix = "CMakeFiles/qnn_llama_runner.dir/"
    (target / "link.txt").write_text(
        f"/toolchain/clang++ {prefix}qnn_llama_runner.cpp.o {prefix}runner/decoder_runner.cpp.o -o runner\n"
    )
    (target / "flags.make").write_text(
        "CXX_DEFINES = -DORIGINAL\nCXX_INCLUDES = -I/original\nCXX_FLAGS = -O2\n"
    )
    output = tmp_path / "output"
    argv = ["build_driver.py", "--build-folder", str(build), "--output", str(output)]
    if ppl:
        argv += ["--define", "ET_QNN_PPL_EVALUATION"]
    monkeypatch.setattr("sys.argv", argv)
    monkeypatch.setattr(build_driver.subprocess, "run", lambda *args, **kwargs: None)
    build_driver.main()
    commands = json.loads((output / "commands.json").read_text())["compile"]
    llama = Path(build_driver.__file__).resolve().parents[2]
    assert {command[-1] for command in commands} == {
        str(llama / "model_validation/benchmarks/driver.cpp"),
        str(llama / "runner/decoder_runner.cpp"),
    }
    for command in commands:
        includes = [flag for flag in command if flag.startswith("-I")]
        assert includes == (["-I" + str(output / "include_overlay")] if ppl else []) + [
            "-I/original"
        ]
    assert (output / "include_overlay").exists() == ppl


def test_current_decoder_includes_can_use_overlay():
    runner = Path(build_driver.__file__).resolve().parents[2] / "runner"
    references = []
    for source in runner.rglob("*"):
        if source.suffix in {".h", ".cpp"}:
            references.extend(
                re.findall(
                    r'#include\s+[<"]([^>"\n]*decoder_runner\.h)[>"]',
                    source.read_text(),
                )
            )
    assert references
    assert set(references) == {
        "executorch/examples/qualcomm/oss_scripts/llama/runner/decoder_runner.h"
    }


def test_incomplete_hook_is_rejected(tmp_path):
    runner = tmp_path / "runner"
    runner.mkdir()
    (runner / "decoder_runner.h").write_text("int32_t next_teacher_forcing_token();\n")
    with pytest.raises(ValueError, match="Incomplete teacher-forcing hook"):
        build_driver.prepare_ppl_overlay(tmp_path, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_runner_output_parameter_names_match_implementation():
    runner = Path(build_driver.__file__).resolve().parents[2] / "runner"
    for filename, pattern in (
        ("runner.h", r"explicit Runner\((.*?)\);"),
        ("runner.cpp", r"Runner::Runner\((.*?)\)\s*:"),
    ):
        signature = re.search(pattern, (runner / filename).read_text(), re.S).group(1)
        assert re.findall(r"const std::string& (\w+_path)", signature) == [
            "model_path",
            "tokenizer_path",
            "dump_logits_path",
            "performance_output_path",
        ]
