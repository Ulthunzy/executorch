import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open


def convert_weights(input_dir: str, output_file: str) -> None:
    root = Path(input_dir)
    with (root / "config.json").open() as f:
        config = json.load(f)
    with (root / "model.safetensors.index.json").open() as f:
        weight_map = json.load(f)["weight_map"]

    if config["model_type"] != "llama":
        raise ValueError("Expected a Llama model")
    if config.get("tie_word_embeddings", False):
        raise ValueError("Expected separate embedding and output weights")

    dim = config["hidden_size"]
    hidden_dim = config["intermediate_size"]
    vocab = config["vocab_size"]
    heads = config["num_attention_heads"]
    kv_heads = config["num_key_value_heads"]
    head_dim = config.get("head_dim", dim // heads)

    mapping = {
        "model.embed_tokens.weight": ("tok_embeddings.weight", (vocab, dim)),
        "model.norm.weight": ("norm.weight", (dim,)),
        "lm_head.weight": ("output.weight", (vocab, dim)),
    }
    layer_mapping = {
        "self_attn.q_proj.weight": ("attention.wq.weight", (heads * head_dim, dim)),
        "self_attn.k_proj.weight": ("attention.wk.weight", (kv_heads * head_dim, dim)),
        "self_attn.v_proj.weight": ("attention.wv.weight", (kv_heads * head_dim, dim)),
        "self_attn.o_proj.weight": ("attention.wo.weight", (dim, heads * head_dim)),
        "mlp.gate_proj.weight": ("feed_forward.w1.weight", (hidden_dim, dim)),
        "mlp.down_proj.weight": ("feed_forward.w2.weight", (dim, hidden_dim)),
        "mlp.up_proj.weight": ("feed_forward.w3.weight", (hidden_dim, dim)),
        "input_layernorm.weight": ("attention_norm.weight", (dim,)),
        "post_attention_layernorm.weight": ("ffn_norm.weight", (dim,)),
    }

    for i in range(config["num_hidden_layers"]):
        for source, (target, shape) in layer_mapping.items():
            mapping[f"model.layers.{i}.{source}"] = (f"layers.{i}.{target}", shape)

    missing = set(mapping) - set(weight_map)
    unexpected = set(weight_map) - set(mapping)
    if missing or unexpected:
        raise ValueError(
            f"Missing keys: {sorted(missing)}; "
            f"unexpected keys: {sorted(unexpected)}"
        )

    converted = {}
    for shard in sorted(set(weight_map.values())):
        with safe_open(root / shard, framework="pt", device="cpu") as f:
            for source, filename in weight_map.items():
                if filename != shard:
                    continue
                target, expected_shape = mapping[source]
                tensor = f.get_tensor(source)
                if tuple(tensor.shape) != expected_shape:
                    raise ValueError(
                        f"{source}: got {tuple(tensor.shape)}, "
                        f"expected {expected_shape}"
                    )
                converted[target] = tensor

    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, output)
    print(f"Saved {len(converted)} tensors to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output_file")
    args = parser.parse_args()
    convert_weights(args.input_dir, args.output_file)
