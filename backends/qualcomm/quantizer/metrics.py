"""Numerical diagnostics for quantized projections and reconstruction."""

import json
import math

import torch


def error_metrics(signal, error, elements):
    signal, error = float(signal), float(error)
    if not all(math.isfinite(v) and v >= 0 for v in (signal, error)):
        raise FloatingPointError("Quantization produced nonfinite signal or error")
    if elements <= 0:
        raise ValueError("Quantization error requires nonempty samples")
    return dict(
        mse=error / elements,
        nmse=error / max(signal, 1e-30),
        sqnr_db=10 * math.log10(signal / error) if signal > 0 and error > 0 else None,
        exact=error == 0,
        zero_signal=signal == 0,
    )


def log_quantization(label, **values):
    print(
        "QUANTIZATION_METRIC "
        + json.dumps(dict(module=label, **values), allow_nan=False),
        flush=True,
    )


@torch.no_grad()
def linear_error_metrics(weight, reconstructed, inputs):
    signal = torch.zeros((), dtype=torch.float64, device=inputs.device)
    error = torch.zeros_like(signal)
    weight = weight.float()
    difference = reconstructed.float() - weight
    for rows in inputs.split(128):
        reference = rows.float() @ weight.T
        delta = rows.float() @ difference.T
        signal += reference.double().square().sum()
        error += delta.double().square().sum()
    return error_metrics(signal, error, inputs.shape[0] * weight.shape[0])


@torch.no_grad()
def hessian_error_metrics(
    weight, reconstructed, hessian, *, scales=None, group_size=None
):
    signal = torch.zeros((), dtype=torch.float64, device=weight.device)
    error = torch.zeros_like(signal)
    for start in range(0, len(weight), 128):
        original = weight[start : start + 128].float()
        value = reconstructed[start : start + 128].float()
        if scales is not None:
            value = value * scales[start : start + 128].repeat_interleave(
                group_size, -1
            )
        delta = value - original
        if hessian.ndim == 3:
            h = hessian[start : start + 128]
            signal += (original[:, None] @ h @ original[:, :, None]).double().sum()
            error += (delta[:, None] @ h @ delta[:, :, None]).double().sum()
        else:
            signal += ((original @ hessian) * original).double().sum()
            error += ((delta @ hessian) * delta).double().sum()
    # H = X.T @ X / samples: these sums are already averaged over input rows.
    if not torch.isfinite(signal) or not torch.isfinite(error):
        raise FloatingPointError("Quantization projection error is nonfinite")
    return error_metrics(signal.clamp_min(0), error.clamp_min(0), len(weight))
