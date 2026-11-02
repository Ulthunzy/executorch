"""Resolve the host device used for calibration and weight quantization."""

from contextlib import contextmanager


def resolve_device(requested="auto"):
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError("calibration_device must be auto, cpu or cuda")
    if requested == "cpu":
        return "cpu"
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if requested == "cuda":
        raise ValueError(
            "CUDA calibration was requested but CUDA is unavailable; "
            "use --calibration-device cpu or auto"
        )
    return "cpu"


@contextmanager
def reference_backend(device):
    if device != "cpu":
        yield
        return
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as model

    previous = (
        model.FusedRMSNormGated,
        model.causal_conv1d_fn,
        model.causal_conv1d_update,
        model.chunk_gated_delta_rule,
        model.fused_recurrent_gated_delta_rule,
    )
    # Decoder construction otherwise selects installed CUDA kernels even on a CPU host.
    model.FusedRMSNormGated = None
    model.causal_conv1d_fn = None
    model.causal_conv1d_update = None
    model.chunk_gated_delta_rule = None
    model.fused_recurrent_gated_delta_rule = None
    try:
        yield
    finally:
        (
            model.FusedRMSNormGated,
            model.causal_conv1d_fn,
            model.causal_conv1d_update,
            model.chunk_gated_delta_rule,
            model.fused_recurrent_gated_delta_rule,
        ) = previous
