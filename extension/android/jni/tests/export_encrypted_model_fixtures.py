# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Wrap a tiny synthetic PTE for QnnEncryptedModelTest; never use these test keys in production."""

import argparse
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def main():
    """Wrap synthetic PTE bytes with public test keys for device regression tests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pte", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plaintext = args.pte.read_bytes()
    args.output.mkdir(parents=True, exist_ok=True)
    for size in (16, 32):
        key = bytes(range(size))
        nonce = bytes(range(12))
        sealed = AESGCM(key).encrypt(nonce, plaintext, None)
        wrapped = b"ETENC1\0\0" + nonce + sealed[-16:] + sealed[:-16]
        (args.output / f"etenc{size * 8}.pte").write_bytes(wrapped)
        if size == 32:
            legacy = b"\xff" * 8 + key + nonce + sealed
            (args.output / "legacy.pte").write_bytes(legacy)
            damaged = bytearray(wrapped)
            damaged[20] ^= 1
            (args.output / "damaged.pte").write_bytes(damaged)
            (args.output / "truncated.pte").write_bytes(wrapped[:30])


if __name__ == "__main__":
    main()
