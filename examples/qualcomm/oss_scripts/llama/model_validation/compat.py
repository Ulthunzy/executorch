# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Load legacy standalone tools without importing the ExecuTorch package."""

import importlib.util
import sys
from pathlib import Path


def forward(namespace, relative_path):
    """Execute canonical code in the caller's namespace, preserving import patches."""
    target = Path(__file__).resolve().parent / relative_path
    directory = str(target.parent)
    if directory not in sys.path:
        sys.path.insert(0, directory)
    spec = importlib.util.spec_from_file_location(namespace["__name__"], target)
    namespace.update(
        __file__=str(target),
        __cached__=spec.cached,
        __loader__=spec.loader,
        __spec__=spec,
        __package__=spec.parent,
    )
    exec(spec.loader.get_code(spec.name), namespace)
