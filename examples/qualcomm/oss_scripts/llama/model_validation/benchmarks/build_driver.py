# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Build an isolated test driver from an existing Makefiles QNN runner build."""

import argparse
import hashlib
import json
import re
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def prepare_ppl_overlay(llama_dir, output):
    """Add teacher forcing only to a private copy of the current decoder header."""
    header = llama_dir / "runner/decoder_runner.h"
    contents = header.read_text()
    hook = "next_teacher_forcing_token"
    if hook in contents:
        if not re.search(
            r"int32_t\s+next_teacher_forcing_token\s*\(\s*\)\s*;", contents
        ) or not re.search(
            r"return\s+next_teacher_forcing_token\s*\(\s*\)\s*;", contents
        ):
            raise ValueError(f"Incomplete teacher-forcing hook in {header}")
    else:
        namespace = "namespace example {\n"
        method = re.compile(r"(\binline\s+int32_t\s+logits_to_token\s*\([^)]*\)\s*\{)")
        if contents.count(namespace) != 1 or len(method.findall(contents)) != 1:
            raise ValueError(f"Unsupported decoder header layout: {header}")
        contents = contents.replace(
            namespace,
            namespace + "#ifdef ET_QNN_PPL_EVALUATION\n"
            "int32_t next_teacher_forcing_token();\n#endif\n",
            1,
        )
        contents = method.sub(
            r"\1\n#ifdef ET_QNN_PPL_EVALUATION\n"
            "    return next_teacher_forcing_token();\n#endif",
            contents,
            count=1,
        )
    overlay = output / "include_overlay"
    destination = (
        overlay
        / "executorch/examples/qualcomm/oss_scripts/llama/runner/decoder_runner.h"
    )
    if destination.resolve() == header.resolve():
        raise ValueError("The PPL overlay must be separate from the production header")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(contents)
    return overlay


def main():
    """Build the test entry point without modifying existing runner objects."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-folder", type=Path, default=Path("build-android"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--entry-point", type=Path)
    parser.add_argument("--define", action="append", default=[])
    args = parser.parse_args()
    source_dir = Path(__file__).resolve().parent
    entry = (args.entry_point or source_dir / "driver.cpp").resolve()
    if not entry.is_file():
        parser.error(f"Missing entry point: {entry}")
    build = args.build_folder.resolve()
    link_dir = build / "examples/qualcomm/oss_scripts/llama"
    target = link_dir / "CMakeFiles/qnn_llama_runner.dir"
    if not (target / "link.txt").is_file():
        parser.error(
            "Build qnn_llama_runner with CMake Unix Makefiles first; see README.md"
        )
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    output = (args.output or build / "self_identity_eval").resolve()
    output.mkdir(parents=True, exist_ok=True)
    link = shlex.split((target / "link.txt").read_text())
    flags = {}
    for line in (target / "flags.make").read_text().splitlines():
        if line.startswith("CXX_") and " = " in line:
            key, value = line.split(" = ", 1)
            flags[key] = shlex.split(value)
    compiler = [link[0]] + [
        arg for arg in link[1:] if arg.startswith(("--target=", "--sysroot="))
    ]
    defines = flags["CXX_DEFINES"].copy()
    for name in args.define:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            parser.error("--define accepts a macro name, not arbitrary compiler flags")
        defines.append("-D" + name)
    includes = flags["CXX_INCLUDES"].copy()
    if any(
        re.match(r"-DET_QNN_PPL_EVALUATION(?:=|$)", flag)
        for flag in defines + flags["CXX_FLAGS"]
    ):
        try:
            overlay = prepare_ppl_overlay(source_dir.parents[1], output)
        except ValueError as error:
            parser.error(str(error))
        includes.insert(0, "-I" + str(overlay))
    base = compiler + defines + includes + flags["CXX_FLAGS"]
    commands, replacements = [], {}
    prefix = "CMakeFiles/qnn_llama_runner.dir/"
    for arg in link:
        if not arg.startswith(prefix) or not arg.endswith(".cpp.o"):
            continue
        relative = arg.removeprefix(prefix).removesuffix(".o")
        source = source_dir.parents[1] / relative
        if relative == "qnn_llama_runner.cpp":
            source = entry
        elif not relative.startswith("runner/"):
            continue
        obj = output / (source.stem + ".o")
        replacements[arg] = str(obj)
        commands.append(base + ["-o", str(obj), "-c", str(source)])
    if prefix + "qnn_llama_runner.cpp.o" not in replacements:
        parser.error("Unsupported link layout: qnn_llama_runner entry point not found")

    def compile_one(command):
        print("COMPILE", command[-1], flush=True)
        subprocess.run(command, cwd=link_dir, check=True)

    # Recompile the QNN runner sources together with the driver to match headers.
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(compile_one, commands))
    binary = output / "identity_eval_driver"
    link[link.index("-o") + 1] = str(binary)
    linked = []
    index = 0
    while index < len(link):
        arg = link[index]
        if arg == "-Xlinker" and link[index + 1].startswith("--dependency-file="):
            index += 2
            continue
        linked.append(replacements.get(arg, arg))
        index += 1
    subprocess.run(linked, cwd=link_dir, check=True)
    subprocess.run(
        [str(Path(link[0]).with_name("llvm-strip")), str(binary)], check=True
    )
    (output / "commands.json").write_text(
        json.dumps(
            {"cwd": str(link_dir), "compile": commands, "link": linked}, indent=2
        )
    )
    (output / "driver_source.sha256").write_text(
        hashlib.sha256(entry.read_bytes()).hexdigest() + "\n"
    )
    print(binary)


if __name__ == "__main__":
    main()
