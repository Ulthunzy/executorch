"""Load one real MoE layer at a time from safetensors."""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from executorch.examples.models.llama.model_args import ModelArgs
from executorch.examples.models.llama.rope import hf_precompute_freqs_cis
from executorch.examples.qualcomm.oss_scripts.llama.model import Qwen35RMSNorm
from executorch.examples.qualcomm.oss_scripts.llama.model.static_llama import (
    LlamaAttention,
    Qwen35DeltaNet,
)


class Checkpoint:
    def __init__(self, path):
        self.path = Path(path)
        self.mapping = json.loads(
            (self.path / "model.safetensors.index.json").read_text()
        )["weight_map"]
        self.config = json.loads((self.path / "config.json").read_text())["text_config"]

    def tensor(self, name):
        with safe_open(
            self.path / self.mapping[name], framework="pt", device="cpu"
        ) as source:
            return source.get_tensor(name)

    def layer_tensor(self, index, name):
        return self.tensor(f"model.language_model.layers.{index}.{name}")

    def constants(self, index, output):
        mapping = {
            "gate_weight": "mlp.gate.weight",
            "shared_expert_gate_proj_weight": "mlp.shared_expert.gate_proj.weight",
            "shared_expert_up_proj_weight": "mlp.shared_expert.up_proj.weight",
            "shared_expert_down_proj_weight": "mlp.shared_expert.down_proj.weight",
            "shared_expert_gate_weight": "mlp.shared_expert_gate.weight",
        }
        np.savez(
            output,
            **{
                k: self.layer_tensor(index, v).float().numpy()
                for k, v in mapping.items()
            },
        )


class AttentionLayer(torch.nn.Module):
    def __init__(self, checkpoint, index):
        super().__init__()
        c = checkpoint.config
        self.linear = c["layer_types"][index] == "linear_attention"
        args = ModelArgs(
            dim=c["hidden_size"],
            hidden_dim=c["moe_intermediate_size"],
            n_heads=c["num_attention_heads"],
            n_kv_heads=c["num_key_value_heads"],
            head_dim=c["head_dim"],
            n_layers=c["num_hidden_layers"],
            max_seq_len=1024,
            max_context_len=1024,
            norm_eps=c["rms_norm_eps"],
            linear_num_key_heads=c["linear_num_key_heads"],
            linear_num_value_heads=c["linear_num_value_heads"],
            linear_key_head_dim=c["linear_key_head_dim"],
            linear_value_head_dim=c["linear_value_head_dim"],
            linear_conv_kernel_dim=c["linear_conv_kernel_dim"],
            use_hf_rope=True,
            partial_rotary_factor=c["partial_rotary_factor"],
            rope_theta=c["rope_parameters"]["rope_theta"],
            use_qk_norm=True,
            qk_norm_before_rope=True,
            rms_norm_add_unit_offset=True,
            use_q_gate=True,
            use_kv_cache=True,
        )
        self.attention = (
            Qwen35DeltaNet(index, args)
            if self.linear
            else LlamaAttention(index, args, output_new_cache_only=True)
        )
        self.input_norm = Qwen35RMSNorm(c["hidden_size"], eps=c["rms_norm_eps"])
        self.post_norm = Qwen35RMSNorm(c["hidden_size"], eps=c["rms_norm_eps"])
        self.input_norm.weight.data.copy_(
            checkpoint.layer_tensor(index, "input_layernorm.weight").float()
        )
        self.post_norm.weight.data.copy_(
            checkpoint.layer_tensor(index, "post_attention_layernorm.weight").float()
        )
        if self.linear:
            state = {
                name: checkpoint.layer_tensor(index, "linear_attn." + name).float()
                for name in self.attention.state_dict()
            }
        else:
            mapping = {
                "wq": "q_proj",
                "wk": "k_proj",
                "wv": "v_proj",
                "wo": "o_proj",
                "q_norm_fn": "q_norm",
                "k_norm_fn": "k_norm",
            }
            state = {
                name: checkpoint.layer_tensor(
                    index,
                    "self_attn."
                    + mapping[name.split(".")[0]]
                    + "."
                    + name.split(".", 1)[1],
                ).float()
                for name in self.attention.state_dict()
            }
        self.attention.load_state_dict(state, strict=True)
        if self.linear:
            self.attention.prepare_fused_kz_linear_in_proj()
            self.attention.prepare_attention_conv()


class AttentionFront(torch.nn.Module):
    def __init__(self, checkpoint, index):
        super().__init__()
        self.layers = torch.nn.ModuleList([AttentionLayer(checkpoint, index)])

    def forward(self, *inputs):
        layer = self.layers[0]
        h = inputs[0]
        norm = layer.input_norm(h)
        if layer.linear:
            y, first, second = layer.attention(
                norm, None, None, None, inputs[1], inputs[2]
            )
        else:
            y, first, second = layer.attention(norm, *inputs[1:])
        residual = h + y
        return layer.post_norm(residual), residual, first[0], second[0]


def rope_tables(config, length=2048):
    cos, sin = hf_precompute_freqs_cis(
        config["head_dim"],
        length,
        config["rope_parameters"]["rope_theta"],
        config["partial_rotary_factor"],
    )
    return cos[:, :32].contiguous(), sin[:, :32].contiguous()


def attention_inputs(hidden, linear, cos, sin, start=0, first=None, second=None):
    device = hidden.device
    ar = hidden.shape[1]
    if first is None:
        first = torch.zeros(
            (1, 8192, 4) if linear else (1, 2, 256, 1024 - ar), device=device
        )
        second = torch.zeros(
            (1, 32, 128, 128) if linear else (1, 2, 1024 - ar, 256), device=device
        )
    if linear:
        return hidden, first, second
    mask = torch.full((1, 1, ar, 1024), -255.0, device=device)
    if start:
        mask[..., 1024 - ar - min(start, 1024 - ar) : 1024 - ar] = 0
    mask[..., 1024 - ar :] = torch.triu(
        torch.full((ar, ar), -255.0, device=device), diagonal=1
    )
    return (
        hidden,
        cos[start : start + ar].to(device).contiguous(),
        sin[start : start + ar].to(device).contiguous(),
        mask,
        first,
        second,
    )


def run_attention(module, hidden, linear, cos, sin, sequences=None):
    for sequence in range(len(hidden)) if sequences is None else sequences:
        first = second = None
        for start in range(0, hidden.shape[1], 128):
            args = attention_inputs(
                hidden[sequence : sequence + 1, start : start + 128].contiguous(),
                linear,
                cos,
                sin,
                start,
                first,
                second,
            )
            result = module(*args)
            if linear:
                first, second = result[2:]
            else:
                old_first, old_second = args[-2:]
                first = torch.cat((old_first[..., 128:], result[2]), -1)
                second = torch.cat((old_second[:, :, 128:], result[3]), 2)
            yield sequence, start, args, result


class Head(torch.nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        self.norm = Qwen35RMSNorm(2048, eps=checkpoint.config["rms_norm_eps"])
        self.norm.weight.data.copy_(
            checkpoint.tensor("model.language_model.norm.weight").float()
        )
        self.output = torch.nn.Linear(2048, 248320, bias=False)
        self.output.weight.data.copy_(checkpoint.tensor("lm_head.weight").float())

    def forward(self, hidden):
        return self.output(self.norm(hidden))
