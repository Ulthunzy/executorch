# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""AR128 and warm-cache AR1 calibration with fixed-grid out-projection GPTQ."""

import copy
import gc
import logging

import torch

from executorch.backends.qualcomm.export_utils import make_quantizer
from executorch.examples.qualcomm.oss_scripts.llama.decoder_constants import (
    TEXT_DECODER,
    VISION_ENCODER,
)
from executorch.examples.qualcomm.oss_scripts.llama.inference.decoder import (
    _build_mrope_positions,
)
from executorch.examples.qualcomm.oss_scripts.llama.quantize.generation import (
    DQ,
    position_input_index,
    Q,
    remove_export_checks,
    specialize_decode,
)
from executorch.examples.qualcomm.oss_scripts.llama.quantize.gptq import gptq_fixed_grid
from torchao.quantization.pt2e.observer import MinMaxObserver
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e


def move_graph(module, device):
    remove_export_checks(module)
    module.to(device)
    for node in module.graph.nodes:
        if node.target == torch.ops.aten.to.device:
            node.args = (node.args[0], torch.device(device), *node.args[2:])
        elif "device" in node.kwargs:
            node.kwargs = {**node.kwargs, "device": torch.device(device)}
    module.recompile()
    return module


def tensor(module, node):
    try:
        return module.get_buffer(node.target)
    except AttributeError:
        return module.get_parameter(node.target)


@torch.no_grad()
def stabilize_gdn_weight_scales(module):
    """Requantize tiny GDN weight encodings that saturate on QNN HTP."""
    minimum = torch.finfo(torch.float16).tiny
    changed = 0
    for node in module.graph.nodes:
        path = next(
            reversed(node.meta.get("nn_module_stack", {"": ("", "")}).values())
        )[0]
        if node.target != torch.ops.aten.conv2d.default or not path.endswith(
            (
                ".attention.in_proj_qvab.conv",
                ".attention.in_proj_kz.conv",
                ".attention.causal_conv2d",
            )
        ):
            continue
        weight = node.args[1]
        if weight.target == torch.ops.torchao.dequantize_affine.default:
            codes = tensor(module, weight.args[0])
            scales = tensor(module, weight.args[2])
            zero = tensor(module, weight.args[3])
            shape = (scales.shape[0], scales.shape[1], -1)
            broadcast = (scales.shape[0], scales.shape[1], 1)
            quant_min, quant_max = weight.args[-2:]
        elif (
            weight.target
            == torch.ops.quantized_decomposed.dequantize_per_channel.default
        ):
            codes = tensor(module, weight.args[0])
            scales = tensor(module, weight.args[1])
            zero = tensor(module, weight.args[2])
            assert weight.args[3] == 0
            shape = codes.shape
            broadcast = (scales.numel(),) + (1,) * (codes.ndim - 1)
            quant_min, quant_max = weight.args[4:6]
        else:
            continue
        small = scales < minimum
        if not small.any():
            continue
        assert not zero.any(), "GDN weights must use symmetric quantization"
        safe = scales.clamp_min(minimum)
        # Preserve the represented weights: raising scales alone amplifies the
        # tiny channels rather than fixing their encoding.
        values = codes.reshape(shape).float() * scales.reshape(broadcast)
        updated = (values / safe.reshape(broadcast)).round().clamp(quant_min, quant_max)
        codes.copy_(updated.reshape_as(codes).to(codes.dtype))
        scales.copy_(safe)
        changed += int(small.sum())
    logging.info("Stabilized %d tiny GDN weight scales for QNN HTP", changed)
    return changed


def protect_linear_bias(module):
    """Ensure INT32 biases fit activation_scale * per-channel weight_scale."""
    changed = 0
    for node in module.graph.nodes:
        if node.target != torch.ops.aten.linear.default or len(node.args) < 3:
            continue
        activation, weight, bias = node.args[:3]
        if (
            not isinstance(bias, torch.fx.Node)
            or weight.op != "call_module"
            or bias.op != "call_module"
        ):
            continue
        observer = module.get_submodule(weight.target)
        if observer.qscheme != torch.per_channel_symmetric:
            continue
        activation_scale = module.get_submodule(activation.target).calculate_qparams()[
            0
        ]
        required = tensor(module, bias.args[0]).detach().abs().double() / (
            (2**31 - 512) * activation_scale.double()
        )
        current = observer.calculate_qparams()[0]
        changed += int((required > current.double()).sum())
        extent = (required * ((observer.quant_max - observer.quant_min) / 2)).to(
            observer.min_val.dtype
        )
        observer.min_val.copy_(torch.minimum(observer.min_val, -extent))
        observer.max_val.copy_(torch.maximum(observer.max_val, extent))
    logging.info("Protected %d visual bias channels from INT32 overflow", changed)
    return changed


def calibration_samples(loaders, embedding, image_token_id):
    vision_loader = loaders.get(VISION_ENCODER)
    vision_batches = iter(vision_loader) if vision_loader is not None else None
    samples = []
    for text in loaders[TEXT_DECODER]:
        assert len(text["token_ids"]) == 1
        ids = torch.tensor([text["token_ids"][0]], dtype=torch.int64)
        hidden = embedding(ids).detach().float()
        images = (
            torch.utils._pytree.tree_leaves(next(vision_batches))
            if vision_batches is not None
            else []
        )
        if images:
            features = torch.cat(
                [value.reshape(1, -1, hidden.shape[-1]) for value in images], dim=1
            )
            image_mask = (ids == image_token_id).unsqueeze(-1).expand_as(hidden)
            if int(image_mask.sum()) != features.numel():
                raise ValueError("Image features and decoder image tokens do not match")
            hidden = hidden.masked_scatter(image_mask, features)
        elif (ids == image_token_id).any():
            raise ValueError("Image token has no paired calibrated encoder output")
        positions = _build_mrope_positions(ids, ids.shape[1], image_token_id)
        samples.append(
            {"hidden": hidden, "positions": positions, "image": bool(images)}
        )
    if vision_batches is not None and next(vision_batches, None) is not None:
        raise ValueError("Unpaired encoder calibration samples")
    return samples


def attention_masks(inputs, past, device, sliding_window=None):
    ar = inputs[0].shape[1]
    row = torch.arange(ar, device=device)[:, None]
    masks = []
    for index, mask in enumerate(inputs[1 : position_input_index(inputs)]):
        history = mask.shape[-1] - ar
        column = torch.arange(mask.shape[-1], device=device)[None, :] - history
        visible = (column >= -min(past, history)) & (column <= row)
        if index > 0:
            if sliding_window is None:
                raise ValueError("SWA calibration requires a sliding window")
            visible &= row - column < sliding_window
        masks.append(torch.where(visible, 0.0, -255.0)[None, None])
    return masks


def update_states(states, outputs):
    result = []
    assert len(states) == len(outputs)
    for old, new in zip(states, outputs):
        if old.shape == new.shape:
            result.append(new)
        else:
            axes = [i for i, (a, b) in enumerate(zip(old.shape, new.shape)) if a != b]
            assert len(axes) == 1
            axis = axes[0]
            result.append(
                torch.cat((old, new), axis).narrow(
                    axis, new.shape[axis], old.shape[axis]
                )
            )
    return result


def resize_states(states, inputs, device):
    result = []
    for old, target in zip(states, inputs[position_input_index(inputs) + 1 :]):
        value = torch.zeros_like(target, device=device)
        if old.shape == value.shape:
            value.copy_(old)
        else:
            axes = [i for i, (a, b) in enumerate(zip(old.shape, value.shape)) if a != b]
            assert len(axes) == 1
            axis = axes[0]
            count = min(old.shape[axis], value.shape[axis])
            value.narrow(axis, value.shape[axis] - count, count).copy_(
                old.narrow(axis, old.shape[axis] - count, count)
            )
        result.append(value)
    return result


class ProjectionCapture(torch.fx.Interpreter):
    def __init__(self, module):
        super().__init__(module)
        self.inputs = {}
        self.count = 128
        self.collect = True
        self.projections = {
            node: next(reversed(node.meta["nn_module_stack"].values()))[0]
            for node in module.graph.nodes
            if node.target == torch.ops.aten.conv2d.default
            and next(
                reversed(node.meta.get("nn_module_stack", {"": ("", "")}).values())
            )[0].endswith(".attention.out_proj.conv")
        }

    def run_node(self, node):
        if self.collect and node in self.projections:
            value = self.env[node.args[0]]
            rows = value.permute(0, 2, 3, 1).reshape(-1, value.shape[1])[: self.count]
            self.inputs.setdefault(self.projections[node], []).append(
                rows.detach().cpu()
            )
        return super().run_node(node)


@torch.no_grad()
def calibrate_prefill(
    module, inputs, samples, device, capture=None, sliding_window=None
):
    move_graph(module, device)
    ar = inputs[0].shape[1]
    warm = []
    for ordinal, sample in enumerate(samples):
        hidden, positions = sample["hidden"].to(device), sample["positions"].to(device)
        states = [
            torch.zeros_like(value, device=device)
            for value in inputs[position_input_index(inputs) + 1 :]
        ]
        # Reserve at least sixteen real tokens for AR1 calibration. Only full
        # valid blocks may seed AR1; padded GDN updates are never carried over.
        warm_start = max(0, (hidden.shape[1] - 16) // ar * ar)
        seed = [value.cpu().clone() for value in states]
        for start in range(0, hidden.shape[1], ar):
            count = min(ar, hidden.shape[1] - start)
            chunk = torch.zeros_like(inputs[0], device=device)
            chunk[:, :count] = hidden[:, start : start + count]
            pos = (
                torch.arange(start, start + ar, dtype=torch.int32, device=device)
                .reshape(1, 1, ar)
                .expand(3, 1, ar)
                .clone()
            )
            pos[..., :count] = positions[..., start : start + count]
            values = (
                chunk,
                *attention_masks(inputs, start, device, sliding_window),
                pos,
                *states,
            )
            if capture is not None:
                capture.count = count
                capture.collect = not sample["image"] or not any(
                    not s["image"] for s in samples
                )
                outputs = capture.run(*values)
            else:
                outputs = module(*values)
            if count == ar:
                states = update_states(states, outputs[1:])
                if start + ar == warm_start:
                    seed = [value.cpu().clone() for value in states]
        warm.append((warm_start, seed))
        logging.info(
            "Qwen3.5 AR128 calibration %d/%d: %d tokens, image=%s",
            ordinal + 1,
            len(samples),
            hidden.shape[1],
            sample["image"],
        )
    move_graph(module, "cpu")
    torch.cuda.empty_cache()
    return warm


@torch.no_grad()
def calibrate_decode(module, inputs, samples, warm, device, sliding_window=None):
    move_graph(module, device)
    for ordinal, (sample, (start, states)) in enumerate(zip(samples, warm)):
        states = resize_states([v.to(device) for v in states], inputs, device)
        hidden, positions = sample["hidden"].to(device), sample["positions"].to(device)
        for index in range(start, hidden.shape[1]):
            outputs = module(
                hidden[:, index : index + 1],
                *attention_masks(inputs, index, device, sliding_window),
                positions[..., index : index + 1],
                *states,
            )
            states = update_states(states, outputs[1:])
        logging.info(
            "Qwen3.5 warm AR1 calibration %d/%d: past=%d, tokens=%d",
            ordinal + 1,
            len(samples),
            start,
            hidden.shape[1] - start,
        )
    move_graph(module, "cpu")
    torch.cuda.empty_cache()


def activation_quantizers(module):
    dynamic, result = {}, []
    for node in module.graph.nodes:
        dynamic[node] = node.op == "placeholder" or (
            node.op not in ("get_attr", "output")
            and any(dynamic.get(n, False) for n in node.all_input_nodes)
        )
        if node.target == Q and dynamic[node]:
            result.append(node)
    return result


def observe_activations(module, groups):
    result = torch.fx.GraphModule(module, copy.deepcopy(module.graph))
    remove_export_checks(result)
    for ordinal, node in enumerate(activation_quantizers(result)):
        encoding = tuple(node.args[1:])
        if encoding not in groups:
            scale, zero, lower, upper, _ = encoding
            observer = MinMaxObserver(
                dtype=torch.int32,
                qscheme=(
                    torch.per_tensor_symmetric
                    if lower < 0 and zero == 0
                    else torch.per_tensor_affine
                ),
                quant_min=lower,
                quant_max=upper,
                eps=0.0001 / 65535,
            )
            observer.min_val.fill_((lower - zero) * scale)
            observer.max_val.fill_((upper - zero) * scale)
            groups[encoding] = observer
        name = f"shared_activation_range_{ordinal}"
        result.add_module(name, groups[encoding])
        with result.graph.inserting_before(node):
            observed = result.graph.call_module(name, (node.args[0],))
        for decoded in list(node.users):
            assert decoded.target == DQ
            decoded.replace_all_uses_with(observed)
            result.graph.erase_node(decoded)
        result.graph.erase_node(node)
    # PT2E hoists weight DQ and its following casts/views ahead of all layers.
    # Emit the whole constant preparation chain only when a layer needs it.
    constants, weight_preparations = set(), set()
    for node in result.graph.nodes:
        if node.op == "get_attr":
            constants.add(node)
        if (
            node.op == "call_function"
            and node.all_input_nodes
            and all(source in constants for source in node.all_input_nodes)
        ):
            constants.add(node)
            if node.target in {
                torch.ops.torchao.dequantize_affine.default,
                torch.ops.quantized_decomposed.dequantize_per_channel.default,
                DQ,
            } or any(source in weight_preparations for source in node.all_input_nodes):
                weight_preparations.add(node)
    graph, emitted = torch.fx.Graph(), {}

    def emit(node):
        if node not in emitted:
            for source in node.all_input_nodes:
                emit(source)
            emitted[node] = graph.node_copy(node, emitted.__getitem__)

    for node in result.graph.nodes:
        if node not in weight_preparations:
            emit(node)
    result.graph = graph
    result.graph.eliminate_dead_code()
    result.graph.lint()
    # Rebuilding also drops unused FP32 attributes retained after PT2E folding.
    return torch.fx.GraphModule(result, result.graph)


def apply_activation_ranges(module, groups):
    for node in activation_quantizers(module):
        old = tuple(node.args[1:])
        observer = groups[old]
        if not torch.isfinite(observer.min_val) or not torch.isfinite(observer.max_val):
            raise RuntimeError(f"Non-finite calibration range at {node.name}")
        scale, zero = observer.calculate_qparams()
        new = (scale.item(), zero.item(), *old[2:])
        node.args = (node.args[0], *new)
        for decoded in list(node.users):
            assert decoded.target == DQ
            decoded.args = (decoded.args[0], *new)
    module.recompile()


@torch.no_grad()
def optimize_out_projections(module, weights, capture, device):
    details = []
    for node in module.graph.nodes:
        path = next(
            reversed(node.meta.get("nn_module_stack", {"": ("", "")}).values())
        )[0]
        if node.target != torch.ops.aten.conv2d.default or path not in weights:
            continue
        weight = node.args[1]
        assert weight.target == torch.ops.torchao.dequantize_affine.default
        assert tuple(weight.args[1]) == (1, 32, 1, 1)
        codes, scales = tensor(module, weight.args[0]), tensor(module, weight.args[2])
        original = weights[path]
        rows, columns = original.shape
        current = scales.reshape(rows, columns // 32)
        base = current.amax(-1) / 16
        effective = base[:, None] * torch.round(current / base[:, None]).clamp(1, 16)
        inputs = torch.cat(capture.inputs[path]).to(device)
        encoding = node.args[0].args[1:]
        assert node.args[0].target == DQ
        inputs = DQ(Q(inputs, *encoding), *encoding)
        diagnostics = {}
        optimized = gptq_fixed_grid(
            original.to(device),
            inputs,
            effective.to(device),
            report=diagnostics,
            label=path,
        ).cpu()
        fraction = float((optimized != codes.reshape_as(optimized)).float().mean())
        codes.copy_(optimized.reshape_as(codes))
        scales.copy_(effective.reshape_as(scales))
        details.append(
            {"module": path, "changed_code_fraction": fraction, **diagnostics}
        )
        logging.info(
            "Qwen3.5 fixed-grid GPTQ %s: %.2f%% codes changed", path, fraction * 100
        )
        torch.cuda.empty_cache()
    assert len(details) == len(weights), (len(details), len(weights))
    return details


@torch.no_grad()
def quantize_qwen35(
    module,
    inputs,
    samples,
    recipe,
    device,
    backend,
    soc_model,
    sliding_window=None,
    sliding_layers=(),
):
    previous_matmul = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        remove_export_checks(module)
        weights = {}
        for node in module.graph.nodes:
            path = next(
                reversed(node.meta.get("nn_module_stack", {"": ("", "")}).values())
            )[0]
            if node.target == torch.ops.aten.conv2d.default and path.endswith(
                ".attention.out_proj.conv"
            ):
                weights[path] = tensor(module, node.args[1]).detach().flatten(1)
        quantizer = make_quantizer(backend=backend, soc_model=soc_model)
        quantizer.set_recipe(recipe)
        prepared = prepare_pt2e(module, quantizer)
        capture = ProjectionCapture(prepared)
        calibrate_prefill(prepared, inputs, samples, device, capture, sliding_window)
        quantized = convert_pt2e(prepared)
        quantized.graph.eliminate_dead_code()
        quantized = torch.fx.GraphModule(quantized, quantized.graph)
        program = torch.export.export(quantized, inputs, strict=False)
        quantized = program.module()
        remove_export_checks(quantized)
        stabilize_gdn_weight_scales(quantized)
        gptq = optimize_out_projections(quantized, weights, capture, device)
        del prepared, program, capture, weights
        gc.collect()
        torch.cuda.empty_cache()
        decode, decode_inputs = specialize_decode(quantized, inputs, sliding_layers)
        groups = {}
        observed = observe_activations(quantized, groups)
        warm = calibrate_prefill(
            observed, inputs, samples, device, sliding_window=sliding_window
        )
        del observed
        observed = observe_activations(decode, groups)
        calibrate_decode(observed, decode_inputs, samples, warm, device, sliding_window)
        del observed, warm
        apply_activation_ranges(quantized, groups)
        apply_activation_ranges(decode, groups)
        logging.info(
            "Qwen3.5 calibrated %d shared activation grids on %s", len(groups), device
        )
        return (
            quantized,
            decode,
            decode_inputs,
            {"gptq": gptq, "activation_groups": len(groups)},
        )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_matmul
        torch.backends.cudnn.allow_tf32 = previous_cudnn
