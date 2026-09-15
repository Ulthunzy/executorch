# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Check explicit Buck source lists; this is not a substitute for a Buck build."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def target(path, name):
    """Inspect literal target fields without executing internal Buck macros."""
    tree = ast.parse(path.read_text())
    for call in ast.walk(tree):
        if isinstance(call, ast.Call):
            fields = {keyword.arg: keyword.value for keyword in call.keywords}
            if (
                isinstance(fields.get("name"), ast.Constant)
                and fields["name"].value == name
            ):
                return fields
    raise AssertionError(f"Missing target {name} in {path}")


def test_buck_text_encoder_sources_and_runner_dependency():
    """Keep Java/native encoder registration and its runner dependency together."""
    android = ROOT / "extension/android"
    java = target(android / "BUCK", "executorch_llama")
    sources = ast.literal_eval(java["srcs"])
    encoder = "executorch_android/src/main/java/org/pytorch/executorch/extension/textencoder/TextEncoderModule.java"
    assert encoder in sources
    assert ast.literal_eval(java["pure_kotlin"]) is False
    assert all((android / source).is_file() for source in sources)

    native = target(android / "jni/BUCK", "executorch_llama_jni")
    sources = ast.literal_eval(native["srcs"])
    assert "jni_layer_textencoder.cpp" in sources
    assert all((android / "jni" / source).is_file() for source in sources)
    assert (
        "//xplat/executorch/extension/llm/runner:runner_lib_static"
        in ast.literal_eval(native["deps"])
    )
