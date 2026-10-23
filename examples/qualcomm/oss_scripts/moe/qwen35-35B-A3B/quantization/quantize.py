import torch
import torch.nn.functional as F


def a8(value, group):
    shape = value.shape
    x = value.reshape(*shape[:-1], shape[-1] // group, group)
    scale = x.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 127
    code = torch.trunc(x / scale + torch.where(x < 0, -0.5, 0.5))
    return code.clamp(-127, 127).mul(scale).reshape(shape)


def nearest(value, levels):
    boundaries = (levels[:-1] + levels[1:]) / 2
    return levels[torch.bucketize(value.contiguous(), boundaries)]


def choose_scales(weight, inputs, group, levels):
    b, n, k = weight.shape
    w = weight.reshape(b, n, k // group, group)
    importance = inputs.square().sum(1).reshape(b, 1, k // group, group)
    maximum = w.abs().amax(-1).clamp_min(1e-12)
    best = torch.full_like(maximum, float("inf"))
    result = maximum / levels.abs().max()
    for ratio in (
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
        0.95,
        1.0,
        1.1,
        1.2,
    ):
        scale = (maximum * ratio / levels.abs().max()).clamp_min(1e-12)
        code = nearest(w / scale[..., None], levels)
        error = ((w - code * scale[..., None]).square() * importance).sum(-1)
        improve = error < best
        result = torch.where(improve, scale, result)
        best = torch.minimum(best, error)
    return result


def dequantize(codes, scales, group):
    return codes.float() * scales.repeat_interleave(group, -1)


@torch.no_grad()
def quantize(weight, inputs, group, levels, use_gptq=True, damping=0.1):
    scales = choose_scales(weight, inputs, group, levels)
    if not use_gptq:
        return (
            nearest(weight / scales.repeat_interleave(group, -1), levels).to(
                torch.int8
            ),
            scales,
        )
    b, n, k = weight.shape
    w = weight.float().clone()
    h = inputs.transpose(1, 2).bmm(inputs) / inputs.shape[1]
    diagonal = torch.arange(k, device=w.device)
    h[:, diagonal, diagonal] += (
        damping * h.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-10)[:, None]
    )
    inverse = torch.linalg.cholesky(
        torch.cholesky_inverse(torch.linalg.cholesky(h)), upper=True
    )
    codes = torch.empty_like(w, dtype=torch.int8)
    for start in range(0, k, 128):
        end = min(start + 128, k)
        pending = w[:, :, start:end].clone()
        errors = torch.empty_like(pending)
        factor = inverse[:, start:end, start:end]
        for column in range(end - start):
            scale = scales[:, :, (start + column) // group]
            code = nearest(pending[:, :, column] / scale, levels)
            codes[:, :, start + column] = code.to(torch.int8)
            error = (pending[:, :, column] - code * scale) / factor[
                :, column, column, None
            ]
            errors[:, :, column] = error
            pending[:, :, column:] -= (
                error[:, :, None] * factor[:, None, column, column:]
            )
        w[:, :, end:] -= errors.bmm(inverse[:, start:end, end:])
    return codes, scales


@torch.no_grad()
def reconstruct_down(weight, actual_middle, target, ridge=0.1):
    h = actual_middle.transpose(1, 2).bmm(actual_middle)
    diagonal = torch.arange(h.shape[-1], device=h.device)
    h[:, diagonal, diagonal] += (
        ridge * h.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-8)[:, None]
    )
    error = target - actual_middle.bmm(weight.transpose(1, 2))
    correction = torch.linalg.solve(h, actual_middle.transpose(1, 2).bmm(error))
    return weight + correction.transpose(1, 2)


def middle(inputs, gu):
    gate, up = inputs.bmm(gu.transpose(1, 2)).chunk(2, -1)
    return a8(F.silu(gate) * up, 512)


def joint_scales(
    qgu, sgu, qd, sd, inputs, target, validation=None, weights=None, steps=100
):
    with torch.enable_grad():
        log_gu = torch.zeros_like(sgu, requires_grad=True)
        log_down = torch.zeros_like(sd, requires_grad=True)
        optimizer = torch.optim.Adam([log_gu, log_down], lr=0.02)
        x = a8(inputs, 1024)
        if weights is None:
            weights = torch.ones(inputs.shape[:2], device=inputs.device)
        weights = weights / weights.sum(1, keepdim=True)
        if validation is None:
            validation = (inputs[:, -32:], target[:, -32:])
            weights = weights.clone()
            weights[:, -32:] = 0
        energy = target.square().mean((1, 2)).clamp_min(1e-12)
        best = torch.full_like(energy, float("inf"))
        best_gu, best_down = sgu.clone(), sd.clone()
        for step in range(steps):
            gu = dequantize(qgu, sgu * log_gu.exp(), 1024)
            down = dequantize(qd, sd * log_down.exp(), 512)
            g, u = x.bmm(gu.transpose(1, 2)).chunk(2, -1)
            mid = F.silu(g) * u
            mid = mid + (a8(mid, 512) - mid).detach()
            prediction = mid.bmm(down.transpose(1, 2))
            error = (prediction - target).square().mean(-1) / energy[:, None]
            with torch.no_grad():
                predicted_validation = middle(a8(validation[0], 1024), gu).bmm(
                    down.transpose(1, 2)
                )
                validation_error = (
                    (predicted_validation - validation[1]).square().mean((1, 2))
                )
            improved = validation_error < best
            best_gu = torch.where(
                improved[:, None, None], sgu * log_gu.detach().exp(), best_gu
            )
            best_down = torch.where(
                improved[:, None, None], sd * log_down.detach().exp(), best_down
            )
            best = torch.minimum(best, validation_error)
            loss = (error * weights).sum(1).mean() + 0.01 * (
                log_gu.square().mean() + log_down.square().mean()
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                log_gu.clamp_(-0.7, 0.7)
                log_down.clamp_(-0.7, 0.7)
    return best_gu.detach(), best_down.detach()
