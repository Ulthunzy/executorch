# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Invalidate Make outputs when flags or toolchains change."""

import os
import sys
import tempfile
from pathlib import Path


def main():
    path = Path(sys.argv[1])
    config = os.environ[sys.argv[2]] + "\n" + " ".join(sys.argv[3:]) + "\n"
    if path.is_file() and path.read_text() == config:
        return
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as file:
        file.write(config)
        temporary = Path(file.name)
    temporary.replace(path)


if __name__ == "__main__":
    main()
