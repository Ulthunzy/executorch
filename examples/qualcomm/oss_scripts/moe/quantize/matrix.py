"""Fixed-grid GPTQ for the Linear/Conv weights actually consumed by QNN."""

import torch
import torch.nn.functional as F

from executorch.examples.qualcomm.oss_scripts.llama.quantize.qwen35 import tensor
from executorch.backends.qualcomm.quantizer.metrics import (
    hessian_error_metrics,
    log_quantization,
)

TARGETS = {torch.ops.aten.linear.default, torch.ops.aten.conv2d.default}


def projection_path(node):
    stack = node.meta.get("nn_module_stack", {})
    if not stack:
        raise ValueError(f"Missing projection provenance: {node.name}")
    return next(reversed(stack.values()))[0]


def original_weights(graph):
    weights = {}
    for node in graph.graph.nodes:
        if node.target not in TARGETS:
            continue
        path = projection_path(node)
        if path in weights:
            raise ValueError(f"Repeated projection needs explicit sharing: {path}")
        weights[path] = tensor(graph, node.args[1]).detach().cpu().float().clone()
    return weights


class HessianCapture(torch.fx.Interpreter):
    def __init__(self, module):
        super().__init__(module)
        self.hessians = {}
        self.counts = {}

    def run_node(self, node):
        if node.target in TARGETS:
            path = projection_path(node)
            x = self.env[node.args[0]].detach().float()
            if node.target == torch.ops.aten.linear.default:
                x = x.reshape(-1, x.shape[-1])
                h = x.T @ x
                count = len(x)
            else:
                weight = self.env[node.args[1]]
                stride = node.args[3] if len(node.args) > 3 else (1, 1)
                padding = node.args[4] if len(node.args) > 4 else (0, 0)
                dilation = node.args[5] if len(node.args) > 5 else (1, 1)
                groups = node.args[6] if len(node.args) > 6 else 1
                patches = F.unfold(x, weight.shape[-2:], dilation, padding, stride)
                if groups == 1:
                    x = patches.transpose(1, 2).reshape(-1, patches.shape[1])
                    h = x.T @ x
                    count = len(x)
                elif groups == weight.shape[0] and weight.shape[1] == 1:
                    x = patches.reshape(x.shape[0], groups, -1, patches.shape[-1])
                    x = x.permute(1, 0, 3, 2).reshape(groups, -1, weight[0].numel())
                    h = x.transpose(1, 2).bmm(x)
                    count = x.shape[1]
                else:
                    raise ValueError(f"Unsupported grouped convolution: {path}")
            if path in self.hessians:
                self.hessians[path].add_(h)
            else:
                self.hessians[path] = h
            self.counts[path] = self.counts.get(path, 0) + count
        return super().run_node(node)


@torch.no_grad()
def codes_from_hessian(weight, hessian, scales, group, qmin, qmax, damping):
    w = weight.float().clone()
    h = hessian.float().clone()
    if h.ndim == 3:
        # Each depthwise filter has its own four-dimensional input space.
        w = w[:, None, :]
        grid = scales[:, None, :]
    else:
        h = h[None]
        w = w[None]
        grid = scales[None]
    columns = w.shape[-1]
    diagonal = torch.arange(columns, device=w.device)
    dead = h.diagonal(dim1=-2, dim2=-1) == 0
    h[:, diagonal, diagonal] = torch.where(dead, 1.0, h[:, diagonal, diagonal])
    w.masked_fill_(dead[:, None, :], 0)
    h[:, diagonal, diagonal] += damping * h.diagonal(dim1=-2, dim2=-1).mean(-1)[:, None]
    inverse = torch.linalg.cholesky(
        torch.cholesky_inverse(torch.linalg.cholesky(h)), upper=True
    )
    result = torch.empty_like(w, dtype=torch.int8)
    for begin in range(0, columns, 128):
        end = min(begin + 128, columns)
        pending = w[:, :, begin:end].clone()
        errors = torch.empty_like(pending)
        factor = inverse[:, begin:end, begin:end]
        for index in range(end - begin):
            scale = grid[:, :, (begin + index) // group]
            code = (pending[:, :, index] / scale).round().clamp(qmin, qmax)
            result[:, :, begin + index] = code.to(torch.int8)
            error = (pending[:, :, index] - code * scale) / factor[
                :, index, index, None
            ]
            errors[:, :, index] = error
            pending[:, :, index:] -= error[:, :, None] * factor[:, None, index, index:]
        w[:, :, end:] -= errors.bmm(inverse[:, begin:end, end:])
    return result.reshape_as(weight)


@torch.no_grad()
def apply_gptq(module, weights, capture, damping=0.01):
    details = []
    for node in module.graph.nodes:
        if node.target not in TARGETS:
            continue
        path = projection_path(node)
        weight = (
            weights[path].flatten(1).to(next(iter(capture.hessians.values())).device)
        )
        dq = node.args[1]
        codes = tensor(module, dq.args[0])
        if dq.target == torch.ops.torchao.dequantize_affine.default:
            scales = tensor(module, dq.args[2])
            zeros = tensor(module, dq.args[3])
            group = dq.args[1][1]
            qmin, qmax = dq.args[-2:]
            current = scales.reshape(weight.shape[0], -1)
            base = current.amax(-1, keepdim=True) / 16
            effective = base * (current / base).round().clamp(1, 16)
        elif dq.target == torch.ops.quantized_decomposed.dequantize_per_channel.default:
            scales = tensor(module, dq.args[1])
            zeros = tensor(module, dq.args[2])
            group = weight.shape[1]
            qmin, qmax = dq.args[4:6]
            effective = scales.reshape(-1, 1)
        else:
            raise ValueError(
                f"Projection is missing a supported weight grid: {path}, {dq.target}"
            )
        if zeros.any() or not (effective > 0).all():
            raise ValueError(f"Expected a positive symmetric weight grid: {path}")
        h = capture.hessians.pop(path) / capture.counts[path]
        optimized = codes_from_hessian(weight, h, effective, group, qmin, qmax, damping)
        metrics = hessian_error_metrics(
            weight, optimized, h, scales=effective, group_size=group
        )
        changed = float((codes.reshape_as(optimized) != optimized).float().mean())
        codes.copy_(optimized.reshape_as(codes))
        scales.copy_(effective.reshape_as(scales))
        details.append(
            dict(
                module=path,
                method="fixed_grid_gptq",
                shape=list(weight.shape),
                bits=4 if qmax < 8 else 8,
                samples=capture.counts[path],
                damping=damping,
                changed_code_fraction=changed,
                output_error=metrics,
                error_scope="weight_only_projection_on_calibration_inputs",
                stop_reason="all_columns_quantized",
                processed_columns=weight.shape[-1],
            )
        )
        print("GPTQ_PROJECTION", details[-1], flush=True)
        log_quantization(
            path,
            **{key: value for key, value in details[-1].items() if key != "module"},
        )
    if len(details) != len(weights) or capture.hessians:
        raise ValueError("GPTQ projection coverage is incomplete")
    return details


@torch.no_grad()
def quantize_vision_positions(visual, device, damping=0.01):
    # The fixed-resolution position lookup is a sparse four-neighbour linear map.
    # Its Hessian has off-diagonal terms; one-hot embedding RTN is not sufficient.
    _, height, width = visual.static_grid_thw[0].tolist()
    side = visual.num_grid_per_side
    h = torch.linspace(0, side - 1, height, device=device)
    w = torch.linspace(0, side - 1, width, device=device)
    hf, wf = h.long(), w.long()
    hc, wc = (hf + 1).clamp_max(side - 1), (wf + 1).clamp_max(side - 1)
    dh, dw = h - hf, w - wf
    indices = [
        (hf[:, None] * side + wf).flatten(),
        (hf[:, None] * side + wc).flatten(),
        (hc[:, None] * side + wf).flatten(),
        (hc[:, None] * side + wc).flatten(),
    ]
    coefficients = [
        ((1 - dh)[:, None] * (1 - dw)).flatten(),
        ((1 - dh)[:, None] * dw).flatten(),
        (dh[:, None] * (1 - dw)).flatten(),
        (dh[:, None] * dw).flatten(),
    ]
    x = torch.zeros(height * width, side * side, device=device)
    rows = torch.arange(len(x), device=device)
    for ids, values in zip(indices, coefficients):
        x.index_put_((rows, ids), values, accumulate=True)
    weight = visual.pos_embed.weight.detach().T.contiguous().to(device)
    scale = (weight.abs().amax(-1, keepdim=True) / 127).clamp_min(1e-12)
    hessian = x.T @ x / len(x)
    codes = codes_from_hessian(
        weight, hessian, scale, weight.shape[1], -127, 127, damping
    )
    quantized = (codes.float() * scale).T
    before = x @ weight.T
    after = x @ quantized
    error = float((before - after).double().square().sum())
    signal = float(before.double().square().sum())
    visual.pos_embed.weight.copy_(quantized.to(visual.pos_embed.weight.device))
    visual.static_pos_embeds.copy_(
        visual.fast_pos_embed_interpolate(visual.static_grid_thw)
    )
    return dict(
        module="visual.pos_embed",
        method="fixed_grid_gptq",
        bits=8,
        hessian="actual bilinear position interpolation",
        samples=len(x),
        shape=list(weight.shape),
        damping=damping,
        output_sqnr_db=float(10 * torch.log10(torch.tensor(signal / error))),
    )
