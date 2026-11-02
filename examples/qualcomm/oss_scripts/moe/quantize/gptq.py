import torch
import torch.nn.functional as F

from executorch.backends.qualcomm.quantizer.metrics import (
    error_metrics,
    log_quantization,
)


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
    qgu,
    sgu,
    qd,
    sd,
    inputs,
    target,
    validation=None,
    weights=None,
    steps=100,
    *,
    min_steps=20,
    patience=10,
    min_delta=1e-4,
    target_nmse=None,
    validation_weights=None,
    report=None,
    label="expert_reconstruction",
):
    if any(type(v) is not int or v < 0 for v in (steps, min_steps, patience)):
        raise ValueError(
            "Reconstruction step limits and patience must be nonnegative integers"
        )
    if not 0 <= min_delta < 1 or (
        target_nmse is not None and not 0 <= target_nmse < float("inf")
    ):
        raise ValueError(
            "Reconstruction needs a finite relative min_delta and target NMSE"
        )
    with torch.enable_grad():
        log_gu = torch.zeros_like(sgu, requires_grad=True)
        log_down = torch.zeros_like(sd, requires_grad=True)
        optimizer = torch.optim.Adam([log_gu, log_down], lr=0.02)
        x = a8(inputs, 1024)
        if weights is None:
            weights = torch.ones(inputs.shape[:2], device=inputs.device)
        if validation is None:
            if inputs.shape[1] < 2:
                raise ValueError(
                    "Reconstruction requires separate fitting and selection rows"
                )
            heldout = min(32, max(1, inputs.shape[1] // 4))
            validation = (inputs[:, -heldout:], target[:, -heldout:])
            weights = weights.clone()
            weights[:, -heldout:] = 0
        if validation_weights is None:
            validation_weights = torch.ones(
                validation[0].shape[:2], device=inputs.device
            )
        for values in (weights, validation_weights):
            if (
                not torch.isfinite(values).all()
                or (values < 0).any()
                or (values.sum(1) <= 0).any()
            ):
                raise ValueError("Reconstruction needs finite positive sample weights")
        weights = weights / weights.sum(1, keepdim=True)
        validation_weights = validation_weights / validation_weights.sum(
            1, keepdim=True
        )
        energy = (target.square().mean(-1) * weights).sum(1).clamp_min(1e-12)
        selection_energy = (validation[1].square().mean(-1) * validation_weights).sum(1)
        validation_x = a8(validation[0], 1024)
        with torch.no_grad():
            prediction = middle(validation_x, dequantize(qgu, sgu, 1024)).bmm(
                dequantize(qd, sd, 512).transpose(1, 2)
            )
            best = (
                (prediction - validation[1]).square().mean(-1) * validation_weights
            ).sum(1)
            initial = best.clone()
        if not torch.isfinite(best).all() or not torch.isfinite(selection_energy).all():
            raise FloatingPointError(f"{label}: nonfinite initial selection error")
        best_gu, best_down = sgu.clone(), sd.clone()
        active = torch.ones_like(best, dtype=torch.bool)
        stalled = torch.zeros_like(best, dtype=torch.int64)
        stopped = torch.full_like(stalled, steps)
        best_step = torch.zeros_like(stalled)
        reasons = ["max_steps" if steps else "disabled"] * len(best)
        monitored = best / selection_energy.clamp_min(1e-12)
        log_quantization(
            label,
            step=0,
            max_steps=steps,
            selection_mse=float(best.mean()),
            selection_nmse=float(monitored.mean()),
        )
        for step in range(1, steps + 1):
            gu = dequantize(qgu, sgu * log_gu.exp(), 1024)
            down = dequantize(qd, sd * log_down.exp(), 512)
            g, u = x.bmm(gu.transpose(1, 2)).chunk(2, -1)
            mid = F.silu(g) * u
            mid = mid + (a8(mid, 512) - mid).detach()
            prediction = mid.bmm(down.transpose(1, 2))
            error = (prediction - target).square().mean(-1) / energy[:, None]
            per_expert = (error * weights).sum(1) + 0.01 * (
                log_gu.square().mean((1, 2)) + log_down.square().mean((1, 2))
            )
            loss = per_expert[active].mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"{label}: nonfinite reconstruction loss at step {step}"
                )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                log_gu.clamp_(-0.7, 0.7)
                log_down.clamp_(-0.7, 0.7)
                current_gu, current_down = sgu * log_gu.exp(), sd * log_down.exp()
                predicted_validation = middle(
                    validation_x, dequantize(qgu, current_gu, 1024)
                ).bmm(dequantize(qd, current_down, 512).transpose(1, 2))
                validation_error = (
                    (predicted_validation - validation[1]).square().mean(-1)
                    * validation_weights
                ).sum(1)
                if not torch.isfinite(validation_error).all():
                    raise FloatingPointError(
                        f"{label}: nonfinite selection error at step {step}"
                    )
                improved = active & (validation_error < best)
                best_gu = torch.where(improved[:, None, None], current_gu, best_gu)
                best_down = torch.where(
                    improved[:, None, None], current_down, best_down
                )
                best_step = torch.where(improved, step, best_step)
                best = torch.where(improved, validation_error, best)
                normalized = best / selection_energy.clamp_min(1e-12)
                significant = active & (
                    (monitored - normalized)
                    > min_delta * monitored.abs().clamp_min(1e-12)
                )
                monitored = torch.where(significant, normalized, monitored)
                stalled = torch.where(significant, 0, stalled + active.long())
                reached = (
                    torch.zeros_like(active)
                    if target_nmse is None
                    else normalized <= target_nmse
                )
                plateau = (
                    (stalled >= patience) if patience else torch.zeros_like(active)
                )
                done = active & (reached | plateau) & (step >= min(min_steps, steps))
                stopped = torch.where(done, step, stopped)
                for index in done.nonzero().flatten().tolist():
                    reasons[index] = "target_nmse" if reached[index] else "plateau"
                active &= ~done
                if step == 1 or step % 10 == 0 or not active.any() or step == steps:
                    stats = error_metrics(selection_energy.mean(), best.mean(), 1)
                    log_quantization(
                        label,
                        step=step,
                        max_steps=steps,
                        active_experts=int(active.sum()),
                        train_nmse=float((error * weights).sum(1).mean()),
                        selection=stats,
                    )
                if not active.any():
                    break
        details = [
            dict(
                steps=int(stopped[index]),
                best_step=int(best_step[index]),
                stop_reason=reasons[index],
                initial_selection=error_metrics(
                    selection_energy[index], initial[index], 1
                ),
                best_selection=error_metrics(selection_energy[index], best[index], 1),
            )
            for index in range(len(best))
        ]
        if report is not None:
            report.update(
                experts=details,
                max_steps=steps,
                min_steps=min_steps,
                patience=patience,
                min_delta=min_delta,
                target_nmse=target_nmse,
            )
        for index, details_row in enumerate(details):
            log_quantization(f"{label}/{index}", **details_row)
    return best_gu.detach(), best_down.detach()
