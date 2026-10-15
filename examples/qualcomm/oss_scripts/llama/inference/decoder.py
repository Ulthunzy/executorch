# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import torch
from executorch.examples.qualcomm.oss_scripts.llama.masking_utils import AttentionMask


def _build_mrope_positions(
    input_ids: torch.Tensor,
    max_context_len: int,
    image_token_id: int,
) -> torch.Tensor:
    token_ids = input_ids[0].tolist()
    positions = []
    rope_pos = 0
    index = 0
    while index < len(token_ids):
        if token_ids[index] != image_token_id:
            next_index = index
            while (
                next_index < len(token_ids) and token_ids[next_index] != image_token_id
            ):
                next_index += 1
            for pos in range(rope_pos, rope_pos + next_index - index):
                positions.append([pos, pos, pos])
            rope_pos += next_index - index
            index = next_index
            continue

        next_index = index
        while next_index < len(token_ids) and token_ids[next_index] == image_token_id:
            next_index += 1
        image_token_count = next_index - index
        grid_size = math.isqrt(image_token_count)
        if grid_size * grid_size != image_token_count:
            raise ValueError(
                f"Qwen3.5 image token count ({image_token_count}) must form a square grid."
            )
        for height in range(grid_size):
            for width in range(grid_size):
                positions.append([rope_pos, rope_pos + height, rope_pos + width])
        rope_pos += grid_size
        index = next_index

    position_delta = rope_pos - len(token_ids)
    while len(positions) < max_context_len:
        pos = len(positions) + position_delta
        positions.append([pos, pos, pos])

    position_tensor = torch.tensor(
        positions[:max_context_len], dtype=torch.int32
    ).T.unsqueeze(1)
    return position_tensor.expand(-1, input_ids.size(0), -1).contiguous()


def merge_modality_embeddings(
    input_ids: torch.Tensor,
    hidden_states: Tuple[torch.Tensor, ...],
    tok_embedding: torch.nn.Module,
    token_id: int,
) -> torch.Tensor:
    """Merge vision/audio hidden states into token embeddings at the special token positions."""
    special_image_mask = (input_ids == token_id).unsqueeze(-1)
    inputs_embeds = tok_embedding(input_ids)
    special_image_mask = special_image_mask.expand_as(inputs_embeds).to(
        inputs_embeds.device
    )
    image_hidden_states = torch.cat(hidden_states, dim=1).to(
        inputs_embeds.device, inputs_embeds.dtype
    )
    return inputs_embeds.masked_scatter(special_image_mask, image_hidden_states)


@dataclass
class DecoderInputs:
    pos_ids: torch.Tensor
    atten_mask: Union[AttentionMask, Tuple[torch.Tensor, ...]]
    k_caches: List[torch.Tensor]
    v_caches: List[torch.Tensor]
    input_ids: Optional[torch.Tensor] = None
    inputs_embeds: Optional[torch.Tensor] = None

    def __iter__(self):
        # Yield order must match the compiled model's input signature:
        # (input_ids | inputs_embeds, *atten_mask, [pos_ids], *k_caches, *v_caches)
        yield self.input_ids if self.input_ids is not None else self.inputs_embeds
        yield from self.atten_mask
        yield self.pos_ids
        yield from self.k_caches
        yield from self.v_caches

    def __post_init__(self):
        if (self.input_ids is None) == (self.inputs_embeds is None):
            raise ValueError("Exactly one of input_ids or inputs_embeds must be set.")


class DecoderInference:
    """
    Single forward pass execution for LLM/MLLM decoders.

    Holds the execution context (pos_ids, KV cache placeholders, default
    attention mask) and handles multimodal embedding merging.

    tok_embedding is NOT stored here — it is passed at call time so callers
    always use the live (possibly re-prepared) module rather than a stale
    reference captured at construction.
    """

    def __init__(
        self,
        get_example_inputs: Callable,
        audio_token_id: Optional[int] = None,
        image_token_id: Optional[int] = None,
        max_context_len: int = 1024,
        max_batch_size: int = 1,
        use_i64_token: bool = False,
    ):
        self.audio_token_id = audio_token_id
        self.image_token_id = image_token_id
        self.max_context_len = max_context_len
        self.max_batch_size = max_batch_size
        self.pos_ids = (
            torch.arange(max_context_len, dtype=torch.int32)
            .unsqueeze(0)
            .expand(max_batch_size, -1)
            .contiguous()
        )
        self._dtype = torch.int64 if use_i64_token else torch.int32
        # Always require the with-KV-cache variant of get_example_inputs so
        # pos_ids and KV cache placeholders are available.  This is required
        # for encoding override: scale/zp collected on the CALIBRATE graph
        # must be copied to the DECODE graph, and both graphs share the same
        # input signature when KV caches are present.
        example_inputs = get_example_inputs()
        assert len(example_inputs) == 5, (
            f"DecoderInference requires get_example_inputs to return "
            f"(tokens, attn_mask, pos_ids, k_caches, v_caches) (5 elements); "
            f"got {len(example_inputs)}.  Pass the with KV-cache version."
        )
        _, self._default_attn_mask, example_pos_ids, self._k_caches, self._v_caches = (
            example_inputs
        )
        self._uses_mrope = example_pos_ids.dim() == 3 and image_token_id is not None

    def to(self, device: Union[str, torch.device]) -> "DecoderInference":
        self.pos_ids = self.pos_ids.to(device)
        self._k_caches = [cache.to(device) for cache in self._k_caches]
        self._v_caches = [cache.to(device) for cache in self._v_caches]
        return self

    def get_inputs(
        self,
        input_ids: torch.Tensor,
        attn_mask: Optional[AttentionMask] = None,
        hidden_states: Tuple[torch.Tensor, ...] = (),
        tok_embedding: Optional[torch.fx.GraphModule] = None,
    ) -> "DecoderInputs":
        if attn_mask is None:
            attn_mask = self._default_attn_mask
        device = self.pos_ids.device
        input_ids = input_ids.to(device)
        raw_input_ids: Optional[torch.Tensor] = None
        inputs_embeds: Optional[torch.Tensor] = None
        if all(
            (
                hidden_states,
                tok_embedding,
                self.audio_token_id or self.image_token_id,
            )
        ):
            inputs_embeds = merge_modality_embeddings(
                input_ids,
                hidden_states,
                tok_embedding,
                self.audio_token_id or self.image_token_id,
            )
        elif tok_embedding is not None:
            inputs_embeds = tok_embedding(input_ids)
        else:
            raw_input_ids = input_ids
        pos_ids = (
            _build_mrope_positions(
                input_ids,
                self.max_context_len,
                self.image_token_id,
            )
            if self._uses_mrope
            else self.pos_ids
        )
        return DecoderInputs(
            input_ids=raw_input_ids,
            inputs_embeds=inputs_embeds,
            pos_ids=pos_ids.to(device),
            atten_mask=tuple(mask.to(device) for mask in attn_mask),
            k_caches=self._k_caches,
            v_caches=self._v_caches,
        )

    @torch.no_grad()
    def predict_step(
        self,
        module: Union[torch.nn.Module, torch.fx.GraphModule],
        input_ids: torch.Tensor,
        attn_mask: Optional[AttentionMask] = None,
        hidden_states: Tuple[torch.Tensor, ...] = (),
        tok_embedding: Optional[torch.fx.GraphModule] = None,
    ) -> torch.Tensor:
        """Single forward pass through the decoder; returns logits."""
        logits, *_ = module(
            *self.get_inputs(input_ids, attn_mask, hidden_states, tok_embedding)
        )
        return logits

    def make_forward_fn(self) -> Callable:
        k = self._k_caches
        v = self._v_caches
        pos = self.pos_ids

        def _forward(
            model: Union[torch.nn.Module, torch.fx.GraphModule],
            model_inputs: dict,
        ) -> torch.Tensor:
            input_ids = model_inputs["input_ids"]
            attn_masks = model_inputs["attention_mask"]
            out = model(input_ids, *attn_masks, pos, *k, *v)
            return out[0] if isinstance(out, (tuple, list)) else out

        return _forward
