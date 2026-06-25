import argparse
import json
import os
from typing import Dict

import torch
from safetensors.torch import load_file
from executorch.examples.models.checkpoint import get_mapped_key

# Map Lincal-VL's HF text decoder weights to ExecuTorch's static Llama-style
# decoder naming. Vision weights are intentionally ignored here because the
# Qualcomm multimodal flow compiles the vision encoder separately.
_LINCAL_VL_TO_META = {
    # Lincal-VL checkpoints wrap the text decoder under `model.language_model`.
    "model.language_model.embed_tokens.weight": "tok_embeddings.weight",
    "model.language_model.norm.weight": "norm.weight",
    # Keep legacy Qwen-style keys for compatibility with other checkpoints.
    "model.embed_tokens.weight": "tok_embeddings.weight",
    "model.norm.weight": "norm.weight",
    "lm_head.weight": "output.weight",
    "model.language_model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
    "model.language_model.layers.{}.self_attn.q_proj.bias": "layers.{}.attention.wq.bias",
    "model.language_model.layers.{}.self_attn.q_norm.weight": "layers.{}.attention.q_norm_fn.weight",
    "model.language_model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
    "model.language_model.layers.{}.self_attn.k_proj.bias": "layers.{}.attention.wk.bias",
    "model.language_model.layers.{}.self_attn.k_norm.weight": "layers.{}.attention.k_norm_fn.weight",
    "model.language_model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
    "model.language_model.layers.{}.self_attn.v_proj.bias": "layers.{}.attention.wv.bias",
    "model.language_model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
    "model.language_model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
    "model.language_model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
    "model.language_model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
    "model.language_model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
    "model.language_model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
    "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
    "model.layers.{}.self_attn.q_proj.bias": "layers.{}.attention.wq.bias",
    "model.layers.{}.self_attn.q_norm.weight": "layers.{}.attention.q_norm_fn.weight",
    "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
    "model.layers.{}.self_attn.k_proj.bias": "layers.{}.attention.wk.bias",
    "model.layers.{}.self_attn.k_norm.weight": "layers.{}.attention.k_norm_fn.weight",
    "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
    "model.layers.{}.self_attn.v_proj.bias": "layers.{}.attention.wv.bias",
    "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
    "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
    "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
    "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
    "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
    "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
}


def load_checkpoint_from_safetensors(input_dir: str) -> Dict[str, torch.Tensor]:
    index_path = os.path.join(input_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            index = json.load(f)
        checkpoint_shards = sorted(set(index["weight_map"].values()))

        shard_to_keys = {}
        for weight_name, shard in index["weight_map"].items():
            shard_to_keys.setdefault(shard, []).append(weight_name)

        merged_state_dict = {}
        for shard in checkpoint_shards:
            shard_data = load_file(os.path.join(input_dir, shard))
            for weight_name in shard_to_keys[shard]:
                merged_state_dict[weight_name] = shard_data[weight_name]
        return merged_state_dict

    model_path = os.path.join(input_dir, "model.safetensors")
    if os.path.exists(model_path):
        return load_file(model_path)

    raise FileNotFoundError(f"Could not find safetensors checkpoint in {input_dir}")


def lincal_vl_hf_to_meta(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    converted_state_dict = {}
    for key, value in state_dict.items():
        try:
            new_key = get_mapped_key(key, _LINCAL_VL_TO_META)
            converted_state_dict[new_key] = value
        except Exception:
            # Keep only text decoder weights for the static LLM path.
            pass

    if "lm_head.weight" not in state_dict and "tok_embeddings.weight" in converted_state_dict:
        converted_state_dict["output.weight"] = converted_state_dict[
            "tok_embeddings.weight"
        ]

    return converted_state_dict


def convert_weights(input_dir: str, output_file: str) -> None:
    print("Loading checkpoint...")
    sd = load_checkpoint_from_safetensors(input_dir)
    print("Converting checkpoint...")
    converted_sd = lincal_vl_hf_to_meta(sd)
    if not converted_sd:
        raise RuntimeError(
            "No Lincal-VL text decoder weights were converted. "
            "Please verify the checkpoint key mapping."
        )
    print("Saving checkpoint...")
    torch.save(converted_sd, output_file)
    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Lincal-VL text decoder weights to Meta format."
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Path to directory containing safetensors checkpoint files.",
    )
    parser.add_argument("output", type=str, help="Path to the output checkpoint")
    args = parser.parse_args()
    convert_weights(args.input_dir, args.output)


if __name__ == "__main__":
    main()
