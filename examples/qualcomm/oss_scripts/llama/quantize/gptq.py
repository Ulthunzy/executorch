# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from executorch.backends.qualcomm.quantizer.metrics import (
    linear_error_metrics,
    log_quantization,
)


@torch.no_grad()
def gptq_fixed_grid(
    weight: torch.Tensor,
    calibration_inputs: torch.Tensor,
    scales: torch.Tensor,
    *,
    group_size: int = 32,
    quant_min: int = -7,
    quant_max: int = 7,
    damping: float = 0.01,
    block_size: int = 128,
    report: dict | None = None,
    label: str = "fixed_grid_gptq",
) -> torch.Tensor:
    """Return signed codes using GPTQ compensation on fixed symmetric grids.

    Weight is [out, in], inputs are [samples, in], and scales are [out, groups].
    Keeping column order preserves the existing QNN block layout. Use only
    calibration inputs; the caller selects CPU/CUDA through their device.
    Algorithm reference: https://github.com/IST-DASLab/gptq .
    """
    if weight.ndim != 2 or calibration_inputs.ndim != 2:
        raise ValueError("weight and calibration_inputs must be matrices")
    rows, columns = weight.shape
    if group_size <= 0 or block_size <= 0 or columns % group_size:
        raise ValueError("positive group/block sizes and complete groups required")
    if calibration_inputs.shape[0] == 0 or calibration_inputs.shape[1] != columns:
        raise ValueError("calibration input shape must match weight columns")
    if scales.shape != (rows, columns // group_size):
        raise ValueError("scale shape must match the weight groups")
    if not -128 <= quant_min < 0 < quant_max <= 127 or damping <= 0:
        raise ValueError("signed int8 code range and positive damping required")
    if not all(torch.isfinite(v).all() for v in (weight, calibration_inputs, scales)):
        raise ValueError("weight, inputs and scales must be finite")
    if not (scales > 0).all():
        raise ValueError("scales must be positive")

    device = calibration_inputs.device
    w = weight.to(device=device, dtype=torch.float32).clone()
    x = calibration_inputs.float()
    scale_grid = scales.to(device=device, dtype=torch.float32)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        hessian = x.T @ x / x.shape[0]
        diagonal = torch.arange(columns, device=device)
        dead = hessian.diag() == 0
        hessian[diagonal[dead], diagonal[dead]] = 1
        w[:, dead] = 0
        hessian[diagonal, diagonal] += damping * hessian.diag().mean()
        inverse_factor = torch.linalg.cholesky(
            torch.cholesky_inverse(torch.linalg.cholesky(hessian)), upper=True
        )
        result = torch.empty_like(w, dtype=torch.int8)
        for begin in range(0, columns, block_size):
            end = min(begin + block_size, columns)
            pending = w[:, begin:end].clone()
            errors = torch.zeros_like(pending)
            factor = inverse_factor[begin:end, begin:end]
            for index in range(end - begin):
                scale = scale_grid[:, (begin + index) // group_size]
                codes = torch.round(pending[:, index] / scale).clamp(
                    quant_min, quant_max
                )
                result[:, begin + index] = codes.to(torch.int8)
                error = (pending[:, index] - codes * scale) / factor[index, index]
                errors[:, index] = error
                pending[:, index:] -= error[:, None] * factor[index, index:][None, :]
            w[:, end:] -= errors @ inverse_factor[begin:end, end:]
        if report is not None:
            reconstructed = result.float() * scale_grid.repeat_interleave(
                group_size, -1
            )
            metrics = linear_error_metrics(weight.to(device).float(), reconstructed, x)
            report.update(
                output_error=metrics,
                error_scope="weight_only_projection_on_calibration_inputs",
                stop_reason="all_columns_quantized",
                processed_columns=columns,
                blocks=(columns + block_size - 1) // block_size,
            )
            log_quantization(label, **report)
        return result.to(weight.device)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
