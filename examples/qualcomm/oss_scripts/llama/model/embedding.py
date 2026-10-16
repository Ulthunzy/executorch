# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.qualcomm.quantizer.quant_recipe import QuantRecipe
from executorch.examples.models.llama.source_transformation.quantize import (
    EmbeddingQuantHandler,
)
from torch import nn


class TokenEmbedding(nn.Module):
    def __init__(
        self,
        input_embedding_module,
        max_batch_size: int,
        ar_len: int,
        vocab_size: int,
        dim: int,
        use_i64_token: bool,
    ):
        super().__init__()
        self.input_embedding_module = input_embedding_module
        self.max_batch_size = max_batch_size
        self.ar_len = ar_len
        self.vocab_size = vocab_size
        self.dim = dim
        self.use_i64_token = use_i64_token

    def get_quantization_transform(self, recipe: QuantRecipe):
        graph = torch.export.export(self, self.get_example_input(), strict=True).graph
        for node in graph.nodes:
            if node.target != torch.ops.aten.embedding.default:
                continue
            config = recipe.get_node_quant_config(node)
            if config is None:
                continue
            block_size = config.block_size
            if (
                block_size is None
                or len(block_size) != 2
                or block_size[0] != 1
                or block_size[1] <= 0
                or self.dim % block_size[1] != 0
            ):
                raise ValueError(
                    "Packed embedding requires block_size=(1, group_size) "
                    "with group_size dividing the embedding dimension."
                )
            bitwidth = (config.weight.quant_max - config.weight.quant_min).bit_length()
            if bitwidth not in (2, 4, 8):
                raise ValueError(f"Unsupported packed embedding bitwidth: {bitwidth}")
            return lambda model: EmbeddingQuantHandler(
                model,
                bitwidth=bitwidth,
                group_size=block_size[1],
                packed=bitwidth in (2, 4),
            ).quantized_model()
        return None

    def get_example_input(self):
        return (
            torch.randint(
                self.vocab_size,
                (self.max_batch_size, self.ar_len),
                dtype=torch.int64 if self.use_i64_token else torch.int32,
            ),
        )

    def forward(self, tokens):
        return self.input_embedding_module(tokens)
