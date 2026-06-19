# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import json
import logging
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e

from executorch.backends.qualcomm._passes.convert_mha_to_sha import ConvertMhaToSha
from executorch.backends.qualcomm._passes.qnn_pass_manager import (
    get_capture_program_passes,
)
from executorch.backends.qualcomm._passes.utils import (
    get_passes_dependency_for_capture_program,
)
from executorch.backends.qualcomm.export_utils import (
    QnnConfig,
    SimpleADB,
    _ptq_calibrate,
    make_quantizer,
    setup_common_args_and_variables,
)
from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
from executorch.backends.qualcomm.serialization.qc_schema import (
    QcomChipset,
    QnnExecuTorchBackendType,
)
from executorch.backends.qualcomm.utils.utils import (
    capture_program,
    convert_linear_to_conv2d,
    generate_gpu_compiler_spec,
    generate_htp_compiler_spec,
    generate_lpai_compiler_spec,
    generate_qnn_executorch_compiler_spec,
    get_qnn_context_binary_alignment,
    get_soc_to_lpai_hw_ver_map,
    to_edge_transform_and_lower_to_qnn,
)
from executorch.examples.models.llama.model_args import ModelArgs
from executorch.examples.models.llama.rope import hf_precompute_freqs_cis
from executorch.examples.qualcomm.oss_scripts.llama.static_llm_quant_recipe import (
    Qwen3_0_6BQuantRecipe,
    Qwen3_5_0_8BQuantRecipe,
)
from executorch.examples.qualcomm.oss_scripts.llama.model.static_llama import (
    LlamaAttention,
    Qwen35DecoderLayer,
    Qwen35RMSNorm,
)
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass


class AttentionOnlyLayer(torch.nn.Module):
    def __init__(self, layer_idx: int, config: ModelArgs):
        super().__init__()
        self.attention = LlamaAttention(
            layer_idx, config, output_new_cache_only=True
        )
        self.attention.prepare_attention_conv()

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        atten_mask: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ):
        return self.attention(
            hidden_states,
            freqs_cos,
            freqs_sin,
            atten_mask,
            k_cache,
            v_cache,
        )


class AttentionStack(torch.nn.Module):
    def __init__(self, config: ModelArgs, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList(
            [AttentionOnlyLayer(index, config) for index in range(num_layers)]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        atten_mask: torch.Tensor,
        *caches,
    ):
        k_caches = caches[: self.num_layers]
        v_caches = caches[self.num_layers :]
        output_k_cache = []
        output_v_cache = []
        for index, layer in enumerate(self.layers):
            hidden_states, k_cache, v_cache = layer(
                hidden_states,
                freqs_cos,
                freqs_sin,
                atten_mask,
                k_caches[index],
                v_caches[index],
            )
            output_k_cache.extend(k_cache)
            output_v_cache.extend(v_cache)
        return hidden_states, output_k_cache, output_v_cache


class MiniQwen35Decoder(torch.nn.Module):
    def __init__(
        self,
        config: ModelArgs,
        return_layer_outputs: bool = False,
        copy_full_attention_inputs: bool = False,
        debug_layer_outputs: bool = False,
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [Qwen35DecoderLayer(index, config, True) for index in range(config.n_layers)]
        )
        self.norm = Qwen35RMSNorm(
            config.dim,
            eps=config.norm_eps,
            add_unit_offset=config.rms_norm_add_unit_offset,
        )
        self.output = torch.nn.Linear(
            config.dim, config.vocab_size, bias=config.output_bias
        )
        self.n_layers = config.n_layers
        self.return_layer_outputs = return_layer_outputs
        self.copy_full_attention_inputs = copy_full_attention_inputs
        self.debug_layer_outputs = debug_layer_outputs

    def prepare_for_qnn(self) -> torch.nn.Module:
        for layer in self.layers:
            if getattr(layer.attention, "prepare_fused_linear_in_proj", None):
                layer.attention.prepare_fused_linear_in_proj()
            if getattr(layer.attention, "prepare_attention_conv", None):
                layer.attention.prepare_attention_conv()
            if getattr(layer.feed_forward, "prepare_feedfoward_conv", None):
                layer.feed_forward.prepare_feedfoward_conv()
        return convert_linear_to_conv2d(self)

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        atten_mask: torch.Tensor,
        *caches,
    ):
        k_caches = caches[: self.n_layers]
        v_caches = caches[self.n_layers :]
        output_k_cache = []
        output_v_cache = []
        layer_outputs = []
        debug_outputs = []
        for index, layer in enumerate(self.layers):
            layer_input = hidden_states
            if self.copy_full_attention_inputs and isinstance(
                layer.attention, LlamaAttention
            ):
                hidden_states = torch.ops.aten._to_copy.default(
                    hidden_states, dtype=hidden_states.dtype
                )
                layer_input = hidden_states

            if self.debug_layer_outputs:
                debug_outputs.append(layer_input)
                attention_norm = layer.attention_norm(hidden_states)
                debug_outputs.append(attention_norm)
                attention_out, k_cache, v_cache = layer.attention(
                    hidden_states=attention_norm,
                    freqs_cos=freqs_cos,
                    freqs_sin=freqs_sin,
                    atten_mask=atten_mask,
                    k_caches=k_caches[index],
                    v_caches=v_caches[index],
                )
                debug_outputs.append(attention_out)
                if layer.post_attention_norm:
                    attention_out = layer.post_attention_norm(attention_out)
                    debug_outputs.append(attention_out)
                hidden_states = (
                    hidden_states + attention_out * layer.residual_multiplier
                    if layer.residual_multiplier is not None
                    else hidden_states + attention_out
                )
                debug_outputs.append(hidden_states)
                ffn_input = (
                    attention_norm if layer.ffn_norm is None else layer.ffn_norm(hidden_states)
                )
                debug_outputs.append(ffn_input)
                ffn_out = layer.feed_forward(ffn_input)
                debug_outputs.append(ffn_out)
                if layer.post_ffn_norm:
                    ffn_out = layer.post_ffn_norm(ffn_out)
                    debug_outputs.append(ffn_out)
                hidden_states = (
                    hidden_states + ffn_out * layer.residual_multiplier
                    if layer.residual_multiplier is not None
                    else hidden_states + ffn_out
                )
                debug_outputs.append(hidden_states)
            else:
                hidden_states, k_cache, v_cache = layer(
                    hidden_states,
                    freqs_cos=freqs_cos,
                    freqs_sin=freqs_sin,
                    atten_mask=atten_mask,
                    k_caches=k_caches[index],
                    v_caches=v_caches[index],
                )
                if self.return_layer_outputs:
                    layer_outputs.append(hidden_states)
            output_k_cache.extend(k_cache)
            output_v_cache.extend(v_cache)
            if self.debug_layer_outputs and self.return_layer_outputs:
                layer_outputs.append(hidden_states)
        logits = self.output(self.norm(hidden_states))
        if self.debug_layer_outputs:
            if self.return_layer_outputs:
                return logits, output_k_cache, output_v_cache, debug_outputs, layer_outputs
            return logits, output_k_cache, output_v_cache, debug_outputs
        if self.return_layer_outputs:
            return logits, output_k_cache, output_v_cache, layer_outputs
        return logits, output_k_cache, output_v_cache


def is_decoder_variant(variant: str) -> bool:
    return variant in {
        "qwen35_decoder",
        "qwen35_decoder_full",
        "qwen35_decoder_lf",
        "qwen35_decoder_lf_no_gate",
        "qwen35_decoder_llf",
    }


def qwen35_layer_types(variant: str, num_layers: int) -> List[str]:
    if variant == "qwen35_decoder_full":
        return ["full_attention"] * num_layers
    if variant in {"qwen35_decoder_lf", "qwen35_decoder_lf_no_gate"}:
        pattern = ["linear_attention", "full_attention"]
        return [pattern[index % len(pattern)] for index in range(num_layers)]
    if variant == "qwen35_decoder_llf":
        pattern = ["linear_attention", "linear_attention", "full_attention"]
        return [pattern[index % len(pattern)] for index in range(num_layers)]
    pattern = [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ]
    return [pattern[index % len(pattern)] for index in range(num_layers)]


def qwen35_linear_conv_dim(config: ModelArgs) -> int:
    key_dim = config.linear_key_head_dim * config.linear_num_key_heads
    value_dim = config.linear_value_head_dim * config.linear_num_value_heads
    return key_dim * 2 + value_dim


def make_config(
    variant: str,
    ar_len: int,
    context_len: int,
    num_layers: int = 1,
    vocab_size: int = 1024,
) -> ModelArgs:
    common = {
        "dim": 1024,
        "n_layers": num_layers,
        "norm_eps": 1e-6,
        "max_batch_size": 1,
        "max_seq_len": context_len,
        "max_context_len": context_len,
        "use_kv_cache": True,
        "use_hf_rope": True,
        "use_qk_norm": True,
        "qk_norm_before_rope": True,
        "attention_qkv_bias": False,
    }
    del ar_len
    if is_decoder_variant(variant):
        return ModelArgs(
            **common,
            hidden_dim=3584,
            vocab_size=vocab_size,
            n_heads=8,
            n_kv_heads=2,
            head_dim=256,
            rope_theta=10000000.0,
            partial_rotary_factor=0.25,
            use_q_gate=variant != "qwen35_decoder_lf_no_gate",
            rms_norm_add_unit_offset=True,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_num_key_heads=16,
            linear_num_value_heads=16,
            layer_types=qwen35_layer_types(variant, num_layers),
        )
    if variant == "qwen3":
        return ModelArgs(
            **common,
            n_heads=16,
            n_kv_heads=8,
            head_dim=128,
            rope_theta=1000000.0,
        )
    if variant == "qwen35":
        return ModelArgs(
            **common,
            n_heads=8,
            n_kv_heads=2,
            head_dim=256,
            rope_theta=10000000.0,
            partial_rotary_factor=0.25,
            use_q_gate=True,
            rms_norm_add_unit_offset=True,
        )
    if variant == "qwen35_no_gate":
        return ModelArgs(
            **common,
            n_heads=8,
            n_kv_heads=2,
            head_dim=256,
            rope_theta=10000000.0,
            partial_rotary_factor=0.25,
            rms_norm_add_unit_offset=True,
        )
    if variant == "qwen35_full_rope":
        return ModelArgs(
            **common,
            n_heads=8,
            n_kv_heads=2,
            head_dim=256,
            rope_theta=10000000.0,
            partial_rotary_factor=1.0,
            use_q_gate=True,
            rms_norm_add_unit_offset=True,
        )
    if variant == "qwen35_repeat2":
        return ModelArgs(
            **common,
            n_heads=8,
            n_kv_heads=4,
            head_dim=256,
            rope_theta=10000000.0,
            partial_rotary_factor=0.25,
            use_q_gate=True,
            rms_norm_add_unit_offset=True,
        )
    raise RuntimeError(f"Unknown variant: {variant}")


def make_model(
    variant: str,
    ar_len: int,
    context_len: int,
    num_layers: int,
    vocab_size: int,
    return_layer_outputs: bool = False,
    copy_full_attention_inputs: bool = False,
    debug_layer_outputs: bool = False,
) -> torch.nn.Module:
    config = make_config(variant, ar_len, context_len, num_layers, vocab_size)
    if is_decoder_variant(variant):
        return (
            MiniQwen35Decoder(
                config,
                return_layer_outputs,
                copy_full_attention_inputs,
                debug_layer_outputs,
            )
            .prepare_for_qnn()
            .eval()
        )
    return AttentionStack(config, num_layers).eval()


def make_output_labels(
    variant: str,
    num_layers: int,
    return_layer_outputs: bool,
    debug_layer_outputs: bool,
) -> List[str]:
    labels = ["logits"]
    labels.extend(f"layer_{index}_k_cache" for index in range(num_layers))
    labels.extend(f"layer_{index}_v_cache" for index in range(num_layers))
    if debug_layer_outputs:
        config = make_config(variant, ar_len=1, context_len=16, num_layers=num_layers)
        for index in range(num_layers):
            layer_type = config.layer_types[index]
            labels.extend(
                [
                    f"layer_{index}_{layer_type}_input",
                    f"layer_{index}_{layer_type}_attention_norm",
                    f"layer_{index}_{layer_type}_attention_out",
                ]
            )
            if config.post_attention_norm:
                labels.append(f"layer_{index}_{layer_type}_post_attention_norm")
            labels.extend(
                [
                    f"layer_{index}_{layer_type}_attention_residual",
                    f"layer_{index}_{layer_type}_ffn_input",
                    f"layer_{index}_{layer_type}_ffn_out",
                ]
            )
            if config.post_ffn_norm:
                labels.append(f"layer_{index}_{layer_type}_post_ffn_norm")
            labels.append(f"layer_{index}_{layer_type}_output")
    if return_layer_outputs:
        labels.extend(f"layer_{index}_output" for index in range(num_layers))
    return labels


def make_inputs(
    variant: str,
    ar_len: int,
    context_len: int,
    num_layers: int,
    vocab_size: int,
    seed: int,
) -> Tuple[torch.Tensor, ...]:
    config = make_config(variant, ar_len, context_len, num_layers, vocab_size)
    torch.manual_seed(seed)
    hidden_states = torch.randn(1, ar_len, config.dim)
    freqs_cos, freqs_sin = hf_precompute_freqs_cis(
        config.head_dim,
        config.max_context_len,
        config.rope_freq_base,
        config.partial_rotary_factor,
    )
    freqs_cos = freqs_cos[:ar_len, : freqs_cos.shape[-1] // 2].contiguous()
    freqs_sin = freqs_sin[:ar_len, : freqs_sin.shape[-1] // 2].contiguous()
    atten_mask = torch.zeros(1, ar_len, config.max_context_len)
    k_caches = []
    v_caches = []
    for index in range(num_layers):
        layer_type = (
            config.layer_types[index]
            if is_decoder_variant(variant)
            else "full_attention"
        )
        if layer_type == "linear_attention":
            k_caches.append(
                torch.randn(
                    1,
                    qwen35_linear_conv_dim(config),
                    config.linear_conv_kernel_dim,
                )
                * 0.01
            )
            v_caches.append(
                torch.randn(
                    1,
                    config.linear_num_value_heads,
                    config.linear_key_head_dim,
                    config.linear_value_head_dim,
                )
                * 0.01
            )
        else:
            k_caches.append(
                torch.randn(
                    1,
                    config.n_kv_heads,
                    config.head_dim,
                    config.max_context_len - ar_len,
                )
                * 0.01
            )
            v_caches.append(
                torch.randn(
                    1,
                    config.n_kv_heads,
                    config.max_context_len - ar_len,
                    config.head_dim,
                )
                * 0.01
            )
    return (hidden_states, freqs_cos, freqs_sin, atten_mask, *k_caches, *v_caches)


def parse_quant_modes(raw: str) -> List[str]:
    result: List[str] = []
    for item in raw.split(","):
        name = item.strip()
        if not name:
            continue
        if name in {"none", "fp16", "float", "recipe"}:
            result.append(name)
        else:
            getattr(QuantDtype, name)
            result.append(name)
    return result


def quant_name(quant_mode: str) -> str:
    return "fp16" if quant_mode in {"none", "float"} else quant_mode


def is_quantized(quant_mode: str) -> bool:
    return quant_mode not in {"none", "fp16", "float"}


def make_recipe(variant: str):
    if variant == "qwen3":
        return Qwen3_0_6BQuantRecipe()
    if variant.startswith("qwen35"):
        return Qwen3_5_0_8BQuantRecipe()
    raise RuntimeError(f"No model quant recipe is mapped for variant: {variant}")


def make_model_quantizer(variant: str, quant_mode: str, qnn_config: QnnConfig):
    if quant_mode == "recipe":
        quantizer = make_quantizer(
            backend=qnn_config.backend,
            soc_model=qnn_config.soc_model,
        )
        quantizer.set_recipe(make_recipe(variant).recipe)
        return quantizer
    return make_quantizer(
        quant_dtype=getattr(QuantDtype, quant_mode),
        backend=qnn_config.backend,
        soc_model=qnn_config.soc_model,
    )


def flatten_outputs(outputs) -> List[torch.Tensor]:
    if isinstance(outputs, torch.Tensor):
        return [outputs]
    flattened = []
    for output in outputs:
        flattened.extend(flatten_outputs(output))
    return flattened


def get_exported_program(edge):
    exported_program = edge.exported_program
    if callable(exported_program) and not exported_program.__class__.__name__.endswith(
        "ExportedProgram"
    ):
        return exported_program()
    return exported_program


def tensor_diff(ref: torch.Tensor, out: torch.Tensor) -> Dict[str, object]:
    diff = (ref.detach().cpu().float() - out.detach().cpu().float()).abs()
    return {
        "shape": list(ref.shape),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "allclose": bool(torch.allclose(ref, out, atol=1e-4, rtol=1e-3)),
    }


def output_sort_key(path: Path):
    match = re.fullmatch(r"output_(\d+)_(\d+)\.raw", path.name)
    if match:
        return int(match.group(1)), int(match.group(2))
    return (0, path.name)


def local_mha2sha_check(
    model: torch.nn.Module,
    sample_input: Tuple[torch.Tensor, ...],
) -> Dict[str, object]:
    with torch.no_grad():
        refs = flatten_outputs(model(*sample_input))
    edge = capture_program(copy.deepcopy(model), sample_input)
    exported_program = get_exported_program(edge)
    graph_module = exported_program.graph_module
    softmax_shapes = [
        list(node.meta["val"].shape)
        for node in graph_module.graph.nodes
        if node.op == "call_function" and "softmax" in str(node.target)
    ]
    ConvertMhaToSha(exported_program)(graph_module)
    sha_nodes = sum(
        1
        for node in graph_module.graph.nodes
        if "_h_" in node.name or node.name.endswith("_sha_concat")
    )
    with torch.no_grad():
        outs = flatten_outputs(exported_program.module()(*sample_input))
    return {
        "softmax_shapes": softmax_shapes,
        "sha_nodes": sha_nodes,
        "outputs": [tensor_diff(ref, out) for ref, out in zip(refs, outs)],
    }


def make_quantized_model(
    model: torch.nn.Module,
    qnn_config: QnnConfig,
    sample_input: Tuple[torch.Tensor, ...],
    dataset: List[Tuple[torch.Tensor, ...]],
    variant: str,
    quant_mode: str,
) -> torch.nn.Module:
    captured_model = torch.export.export(
        copy.deepcopy(model), sample_input, strict=False
    ).module()
    quantizer = make_model_quantizer(variant, quant_mode, qnn_config)
    annotated_model = _ptq_calibrate(captured_model, quantizer, dataset)
    return convert_pt2e(annotated_model)


def local_quantized_mha2sha_check(
    model: torch.nn.Module,
    qnn_config: QnnConfig,
    sample_input: Tuple[torch.Tensor, ...],
    dataset: List[Tuple[torch.Tensor, ...]],
    variant: str,
    quant_mode: str,
) -> Dict[str, object]:
    lower_model = make_quantized_model(
        model, qnn_config, sample_input, dataset, variant, quant_mode
    )
    with torch.no_grad():
        refs = flatten_outputs(lower_model(*sample_input))
    edge = capture_program(copy.deepcopy(lower_model), sample_input)
    exported_program = get_exported_program(edge)
    graph_module = exported_program.graph_module
    softmax_shapes = [
        list(node.meta["val"].shape)
        for node in graph_module.graph.nodes
        if node.op == "call_function" and "softmax" in str(node.target)
    ]
    ConvertMhaToSha(exported_program)(graph_module)
    sha_nodes = sum(
        1
        for node in graph_module.graph.nodes
        if "_h_" in node.name or node.name.endswith("_sha_concat")
    )
    with torch.no_grad():
        outs = flatten_outputs(exported_program.module()(*sample_input))
    return {
        "softmax_shapes": softmax_shapes,
        "sha_nodes": sha_nodes,
        "outputs": [tensor_diff(ref, out) for ref, out in zip(refs, outs)],
    }


def make_compile_spec(
    qnn_config: QnnConfig,
    quantized: bool,
    use_mha2sha: bool,
):
    if qnn_config.backend == QnnExecuTorchBackendType.kLpaiBackend:
        backend_options = generate_lpai_compiler_spec(
            target_env=get_soc_to_lpai_hw_ver_map()[qnn_config.soc_model]
        )
    elif qnn_config.backend == QnnExecuTorchBackendType.kGpuBackend:
        backend_options = generate_gpu_compiler_spec()
    else:
        backend_options = generate_htp_compiler_spec(
            use_fp16=not quantized,
            htp_performance_mode=qnn_config.htp_performance_mode,
        )
    return generate_qnn_executorch_compiler_spec(
        soc_model=getattr(QcomChipset, qnn_config.soc_model),
        backend_options=backend_options,
        online_prepare=qnn_config.online_prepare,
        profile_level=qnn_config.profile_level,
        shared_buffer=qnn_config.shared_buffer,
        dump_intermediate_outputs=qnn_config.dump_intermediate_outputs,
        use_mha2sha=use_mha2sha,
    )


def build_pte(
    model: torch.nn.Module,
    qnn_config: QnnConfig,
    pte_stem: Path,
    dataset: List[Tuple[torch.Tensor, ...]],
    variant: str,
    quant_mode: str,
    use_mha2sha: bool,
    skip_mutable_buffer: bool,
) -> None:
    if qnn_config.pre_gen_pte:
        logging.info("Using pregenerated PTE from %s", qnn_config.pre_gen_pte)
        return

    sample_input = dataset[0]
    compile_spec = make_compile_spec(
        qnn_config, is_quantized(quant_mode), use_mha2sha
    )
    if is_quantized(quant_mode):
        lower_model = make_quantized_model(
            model, qnn_config, sample_input, dataset, variant, quant_mode
        )
    else:
        lower_model = model

    edge_prog_mgr = to_edge_transform_and_lower_to_qnn(
        lower_model,
        sample_input,
        compile_spec,
        passes_job=get_capture_program_passes(),
        dep_table=get_passes_dependency_for_capture_program(),
        skip_node_id_set=qnn_config.skip_delegate_node_ids,
        skip_node_op_set=qnn_config.skip_delegate_node_ops,
        skip_mutable_buffer=skip_mutable_buffer,
    )
    allocate_io = not (qnn_config.shared_buffer or qnn_config.direct_build_folder)
    executorch_config = ExecutorchBackendConfig(
        memory_planning_pass=MemoryPlanningPass(
            alloc_graph_input=allocate_io,
            alloc_graph_output=allocate_io,
        ),
        segment_alignment=get_qnn_context_binary_alignment(),
    )
    exec_prog_mgr = edge_prog_mgr.to_executorch(config=executorch_config)
    with open(f"{pte_stem}.pte", "wb") as file:
        exec_prog_mgr.write_to_file(file)


def parse_runner_avg_ms(output: str) -> Optional[float]:
    matches = re.findall(
        r"(?<!Total\s)\d+\s+inferences?\s+took\s+[0-9.]+\s+ms,\s+avg\s+([0-9.]+)\s+ms",
        output,
    )
    if not matches:
        matches = re.findall(
            r"Total\s+\d+\s+inferences?\s+took\s+[0-9.]+\s+ms,\s+avg\s+([0-9.]+)\s+ms",
            output,
        )
    return None if not matches else float(matches[-1])


def read_outputs(output_dir: Path, refs: Sequence[torch.Tensor]) -> List[Dict[str, object]]:
    output_info = []
    output_files = sorted(output_dir.glob("*.raw"), key=output_sort_key)
    for output_file, ref in zip(output_files, refs):
        expected_numel = ref.numel()
        raw_size = output_file.stat().st_size
        dtype = None
        for candidate in (np.float32, np.float16, np.uint16, np.uint8):
            if raw_size == expected_numel * np.dtype(candidate).itemsize:
                dtype = candidate
                break
        if dtype is None:
            output_info.append(
                {"file": str(output_file), "error": f"unexpected byte size {raw_size}"}
            )
            continue
        data = np.fromfile(output_file, dtype=dtype).reshape(tuple(ref.shape))
        out = torch.from_numpy(data.astype(np.float32))
        output_info.append(
            {
                "file": str(output_file),
                "dtype": np.dtype(dtype).name,
                **tensor_diff(ref, out),
            }
        )
    return output_info


def compare_output_records(
    off_result: Dict[str, object], on_result: Dict[str, object]
) -> Dict[str, object]:
    outputs = []
    output_labels = off_result.get("output_labels", [])
    for index, (off_output, on_output) in enumerate(
        zip(off_result["outputs"], on_result["outputs"])
    ):
        label = output_labels[index] if index < len(output_labels) else None
        off_file = off_output.get("file")
        on_file = on_output.get("file")
        if (
            not off_file
            or not on_file
            or "error" in off_output
            or "error" in on_output
            or off_output.get("shape") != on_output.get("shape")
        ):
            outputs.append(
                {
                    "index": index,
                    "label": label,
                    "off_file": off_file,
                    "on_file": on_file,
                    "error": "missing output, prior read error, or shape mismatch",
                }
            )
            continue

        off_data = np.fromfile(off_file, dtype=np.dtype(off_output["dtype"])).reshape(
            tuple(off_output["shape"])
        )
        on_data = np.fromfile(on_file, dtype=np.dtype(on_output["dtype"])).reshape(
            tuple(on_output["shape"])
        )
        diff = np.abs(off_data.astype(np.float32) - on_data.astype(np.float32))
        outputs.append(
            {
                "index": index,
                "label": label,
                "off_file": off_file,
                "on_file": on_file,
                "shape": off_output["shape"],
                "max_abs_diff": float(diff.max()),
                "mean_abs_diff": float(diff.mean()),
                "p99_abs_diff": float(np.quantile(diff, 0.99)),
            }
        )

    valid_outputs = [item for item in outputs if "max_abs_diff" in item]
    return {
        "time": datetime.now().isoformat(timespec="seconds"),
        "type": "mha2sha_pairwise_compare",
        "variant": off_result["variant"],
        "quant_dtype": off_result["quant_dtype"],
        "quant_mode": off_result["quant_mode"],
        "ar_len": off_result["ar_len"],
        "context_len": off_result["context_len"],
        "num_layers": off_result["num_layers"],
        "skip_mutable_buffer": off_result.get("skip_mutable_buffer", False),
        "return_layer_outputs": off_result.get("return_layer_outputs", False),
        "copy_full_attention_inputs": off_result.get(
            "copy_full_attention_inputs", False
        ),
        "debug_layer_outputs": off_result.get("debug_layer_outputs", False),
        "off_case": off_result["case"],
        "on_case": on_result["case"],
        "max_abs_diff": max(
            (item["max_abs_diff"] for item in valid_outputs), default=None
        ),
        "mean_abs_diff": (
            sum(item["mean_abs_diff"] for item in valid_outputs) / len(valid_outputs)
            if valid_outputs
            else None
        ),
        "outputs": outputs,
    }


def append_record(record_path: Path, result: Dict[str, object]) -> None:
    record_path.parent.mkdir(parents=True, exist_ok=True)
    with record_path.open("a") as file:
        file.write(json.dumps(result, ensure_ascii=False, indent=2))
        file.write("\n\n")


def run_one(
    args,
    qnn_config: Optional[QnnConfig],
    variant: str,
    quant_mode: str,
    use_mha2sha: bool,
) -> Dict[str, object]:
    qname = quant_name(quant_mode)
    mha_name = "mha2sha_on" if use_mha2sha else "mha2sha_off"
    mutable_name = "_skip_mutable_buffer" if args.skip_mutable_buffer else ""
    layer_output_name = "_layer_outputs" if args.return_layer_outputs else ""
    copy_full_name = "_copy_full_inputs" if args.copy_full_attention_inputs else ""
    debug_name = "_debug_outputs" if args.debug_layer_outputs else ""
    case_name = (
        f"{variant}_{qname}_{mha_name}_ar{args.ar_len}_ctx{args.context_len}"
        f"_layers{args.num_layers}{mutable_name}{layer_output_name}{copy_full_name}"
        f"{debug_name}"
    )
    case_dir = Path(args.artifact) / case_name
    if (
        case_dir.exists()
        and not args.keep_artifact
        and (qnn_config is None or not qnn_config.pre_gen_pte)
    ):
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    dataset = [
        make_inputs(
            variant,
            args.ar_len,
            args.context_len,
            args.num_layers,
            args.mini_vocab_size,
            args.seed + index,
        )
        for index in range(args.calibration_samples)
    ]
    sample_input = dataset[0]
    model = make_model(
        variant,
        args.ar_len,
        args.context_len,
        args.num_layers,
        args.mini_vocab_size,
        args.return_layer_outputs,
        args.copy_full_attention_inputs,
        args.debug_layer_outputs,
    )
    with torch.no_grad():
        refs = flatten_outputs(model(*sample_input))
    output_labels = make_output_labels(
        variant,
        args.num_layers,
        args.return_layer_outputs,
        args.debug_layer_outputs,
    )

    result = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "case": case_name,
        "variant": variant,
        "quant_dtype": qname,
        "quant_mode": quant_mode,
        "use_mha2sha": use_mha2sha,
        "skip_mutable_buffer": args.skip_mutable_buffer,
        "ar_len": args.ar_len,
        "context_len": args.context_len,
        "num_layers": args.num_layers,
        "return_layer_outputs": args.return_layer_outputs,
        "copy_full_attention_inputs": args.copy_full_attention_inputs,
        "debug_layer_outputs": args.debug_layer_outputs,
        "output_labels": output_labels,
    }
    if args.local_check or args.local_only:
        result["local_mha2sha_check"] = local_mha2sha_check(model, sample_input)
        if qnn_config is not None and is_quantized(quant_mode):
            result["local_quantized_mha2sha_check"] = local_quantized_mha2sha_check(
                model, qnn_config, sample_input, dataset, variant, quant_mode
            )

    if args.local_only:
        return result

    pte_stem = case_dir / case_name
    assert qnn_config is not None
    build_pte(
        model,
        qnn_config,
        pte_stem,
        dataset,
        variant,
        quant_mode,
        use_mha2sha,
        args.skip_mutable_buffer,
    )
    if args.compile_only:
        return result

    runner_output = {"text": ""}

    def capture_output(proc):
        runner_output["text"] = proc.stdout

    adb = SimpleADB(
        qnn_config=qnn_config,
        pte_path=f"{pte_stem}.pte",
        workspace=f"/data/local/tmp/executorch/{case_name}",
    )
    adb.push(inputs=[sample_input])
    adb.execute(output_callback=capture_output, iteration=args.iteration)
    (case_dir / "runner.log").write_text(runner_output["text"])
    adb.pull(host_output_path=str(case_dir))

    result["runner_avg_ms"] = parse_runner_avg_ms(runner_output["text"])
    result["outputs"] = read_outputs(case_dir / "outputs", refs)
    return result


def main(args) -> None:
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    quant_modes = parse_quant_modes(args.quant_modes)
    mha_modes = [item.strip() for item in args.mha2sha.split(",") if item.strip()]
    needs_qnn_config = not args.local_only or any(
        is_quantized(mode) for mode in quant_modes
    )
    qnn_config = (
        None
        if not needs_qnn_config
        else QnnConfig.load_config(args.config_file if args.config_file else args)
    )
    if os.path.exists(args.artifact) and not args.keep_artifact and not args.pre_gen_pte:
        shutil.rmtree(args.artifact)
    os.makedirs(args.artifact, exist_ok=True)

    record_path = Path(args.record)
    results = []
    result_pairs = {}
    for quant_mode in quant_modes:
        for variant in variants:
            for mode in mha_modes:
                result = run_one(
                    args,
                    qnn_config,
                    variant,
                    quant_mode,
                    use_mha2sha=mode == "on",
                )
                append_record(record_path, result)
                results.append(result)
                pair_key = (
                    variant,
                    quant_mode,
                    args.ar_len,
                    args.context_len,
                    args.num_layers,
                    args.skip_mutable_buffer,
                    args.return_layer_outputs,
                    args.copy_full_attention_inputs,
                    args.debug_layer_outputs,
                )
                result_pairs.setdefault(pair_key, {})[mode] = result

    for pair in result_pairs.values():
        if (
            "off" not in pair
            or "on" not in pair
            or "outputs" not in pair["off"]
            or "outputs" not in pair["on"]
        ):
            continue
        comparison = compare_output_records(pair["off"], pair["on"])
        append_record(record_path, comparison)
        results.append(comparison)

    summary_path = Path(args.artifact) / "summary.json"
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    logging.info("Saved summary to %s", summary_path)
    logging.info("Appended test record to %s", record_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = setup_common_args_and_variables()
    parser.add_argument(
        "-a",
        "--artifact",
        default="./qwen-mha2sha-attention-test",
        type=str,
        help="Folder for generated Qwen attention QNN artifacts.",
    )
    parser.add_argument("--variants", default="qwen3,qwen35", type=str)
    parser.add_argument(
        "--quant_modes",
        "--quant_dtypes",
        default="recipe",
        type=str,
        help="Comma list: recipe, none/fp16, or QuantDtype names such as use_16a4w.",
    )
    parser.add_argument("--mha2sha", default="off,on", type=str, help="Comma list: off,on")
    parser.add_argument("--ar_len", default=1, type=int)
    parser.add_argument("--context_len", default=16, type=int)
    parser.add_argument("--num_layers", default=1, type=int)
    parser.add_argument("--mini_vocab_size", default=1024, type=int)
    parser.add_argument("--calibration_samples", default=4, type=int)
    parser.add_argument("--iteration", default=50, type=int)
    parser.add_argument("--local_check", action="store_true")
    parser.add_argument("--local_only", action="store_true")
    parser.add_argument("--keep_artifact", action="store_true")
    parser.add_argument(
        "--return_layer_outputs",
        action="store_true",
        help="Append each decoder layer hidden state to the model outputs.",
    )
    parser.add_argument(
        "--copy_full_attention_inputs",
        action="store_true",
        help="Insert a simple tensor boundary before full attention layers.",
    )
    parser.add_argument(
        "--debug_layer_outputs",
        action="store_true",
        help="Return intermediate tensors inside each decoder layer for phone-side diffing.",
    )
    parser.add_argument(
        "--skip_mutable_buffer",
        action="store_true",
        help="Do not delegate mutable buffer nodes to QNN during lowering.",
    )
    parser.add_argument(
        "--record",
        default="./qwen_mha2sha_attention_test_record.txt",
        type=str,
        help="Append-only JSONL-style test notes.",
    )

    args = parser.parse_args()
    if args.seed is None:
        args.seed = 666

    try:
        main(args)
    except SystemExit:
        raise
    except Exception as e:
        raise RuntimeError(e)
