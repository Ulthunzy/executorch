# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compatibility entry point; implementation lives in model_validation/calibration."""

import runpy as _runpy
from pathlib import Path as _Path

_runpy.run_path(
    str(_Path(__file__).resolve().parents[1] / "model_validation/compat.py")
)["forward"](globals(), "calibration/validate_think_calibration.py")
