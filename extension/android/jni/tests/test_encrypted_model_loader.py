# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import subprocess
from pathlib import Path

import pytest

AESGCM = pytest.importorskip("cryptography.hazmat.primitives.ciphers.aead").AESGCM
ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture(scope="module")
def binary(tmp_path_factory):
    """Build the real loader with host OpenSSL and memory-safety instrumentation."""
    build = Path(os.environ.get("EXECUTORCH_HOST_BUILD", ROOT / "build-x86"))
    core = build / "libexecutorch_core.a"
    compiler = shutil.which("c++")
    if compiler is None or not core.exists():
        pytest.skip(
            "Build the host runtime core and install OpenSSL development headers"
        )
    output = tmp_path_factory.mktemp("encrypted-loader") / "test"
    command = [
        compiler,
        "-std=c++17",
        "-O1",
        "-g",
        "-fsanitize=address,undefined",
        "-fno-omit-frame-pointer",
        "-no-pie",
        "-DC10_USING_CUSTOM_GENERATED_MACROS",
        f"-I{ROOT.parent}",
        f"-I{build / 'include'}",
        f"-I{build / 'include/executorch/runtime/core/portable_type/c10'}",
        str(ROOT / "extension/android/jni/encrypted_model_loader.cpp"),
        str(Path(__file__).with_name("encrypted_model_loader_test.cpp")),
        str(core),
        "-lcrypto",
        "-lpthread",
        "-o",
        str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return output


@pytest.mark.parametrize(
    "case",
    [
        "plain",
        "missing-file",
        "legacy",
        "legacy-override",
        "etenc128",
        "etenc256",
        "large",
        "wrong-key",
        "missing-key",
        "invalid-key",
        "corrupt-nonce",
        "corrupt-tag",
        "corrupt-payload",
        "corrupt-legacy-tag",
        "corrupt-legacy-key",
        "truncated-header",
        "truncated-legacy",
        "truncated-etenc",
        "truncated-payload",
        "empty-payload",
    ],
)
def test_encrypted_loader(binary, tmp_path, case):
    """Expose exact plaintext only after complete container authentication."""
    # Test keys/nonces only; fixtures never contain production model material.
    key = bytes(range(16 if case == "etenc128" else 32))
    nonce = bytes(range(12))
    plaintext = bytes(range(256)) * (8193 if case == "large" else 2)
    sealed = AESGCM(key).encrypt(nonce, plaintext, None)
    legacy = b"\xff" * 8 + key + nonce + sealed
    etenc = b"ETENC1\0\0" + nonce + sealed[-16:] + sealed[:-16]
    payload, key_arg, error = etenc, key.hex(), 0
    if case.startswith("legacy") or "legacy" in case:
        payload, key_arg = legacy, ""
    if case == "legacy-override":
        payload = legacy[:8] + b"\0" * 32 + legacy[40:]
        key_arg = key.hex()
    if case == "plain":
        payload = plaintext
    elif case == "missing-file":
        error = 0x22
    elif case == "wrong-key":
        key_arg, error = "ff" * 32, 0x23
    elif case in ("missing-key", "invalid-key"):
        key_arg, error = ("" if case == "missing-key" else "zz" * 16), 0x12
    elif case.startswith("corrupt-"):
        offset = {
            "corrupt-nonce": 8,
            "corrupt-tag": 20,
            "corrupt-payload": 36,
            "corrupt-legacy-tag": -1,
            "corrupt-legacy-key": 8,
        }[case]
        payload = bytearray(payload)
        payload[offset] ^= 1
        error = 0x23
    elif case.startswith("truncated-"):
        length = {
            "truncated-header": 5,
            "truncated-legacy": 60,
            "truncated-etenc": 30,
        }.get(case, -1)
        payload, error = payload[:length], 0x23
    elif case == "empty-payload":
        tag = AESGCM(key).encrypt(nonce, b"", None)
        payload, error = b"ETENC1\0\0" + nonce + tag, 0x23
    path = tmp_path / "model.pte"
    if case != "missing-file":
        path.write_bytes(payload)
    expected = tmp_path / "expected.bin"
    expected.write_bytes(plaintext)
    result = subprocess.run(
        [
            str(binary),
            str(path),
            key_arg,
            str(error),
            "plain" if case == "plain" else str(expected),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
