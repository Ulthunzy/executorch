# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional

import torch
from executorch.examples.models.model_base import EagerModelBase
from PIL import Image


def _load_qwen2_5_vl_transformers():
    try:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except ImportError as exc:
        raise ImportError(
            "Qwen2.5-VL support requires a transformers build that ships "
            "Qwen2_5_VLForConditionalGeneration."
        ) from exc

    return AutoProcessor, Qwen2_5_VLForConditionalGeneration


def _clone_tensor_tree(inputs: Mapping[str, Any]) -> Dict[str, Any]:
    cloned: Dict[str, Any] = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            cloned[key] = value.clone()
        else:
            cloned[key] = value
    return cloned


@dataclass
class Qwen2_5VLPrefillInputs:
    input_ids: torch.Tensor
    attention_mask: Optional[torch.Tensor]
    token_embeddings: torch.Tensor
    image_embeddings: Optional[torch.Tensor]
    video_embeddings: Optional[torch.Tensor]
    position_ids: Optional[torch.Tensor]
    rope_deltas: Optional[torch.Tensor]
    raw_model_inputs: Dict[str, Any]


class Qwen2_5VLVisionEncoder(torch.nn.Module):
    """
    Thin wrapper over the Hugging Face image feature path.

    This intentionally keeps the extra `image_grid_thw` input explicit because
    Qwen2.5-VL depends on it for multimodal position handling.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        return self.model.get_image_features(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )


class Qwen2_5VLVideoEncoder(torch.nn.Module):
    """
    Thin wrapper over the Hugging Face video feature path.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        pixel_values_videos: torch.Tensor,
        video_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        return self.model.get_video_features(
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
        )


class Qwen2_5VLTokenEmbedding(torch.nn.Module):
    """
    Exposes the text token embedding layer as an explicit module.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.embedding = model.get_input_embeddings()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)


class Qwen2_5VLTextDecoder(torch.nn.Module):
    """
    Decoder-only wrapper for the text generation stage.

    The important detail is that the decoder cache contains both textual and
    visual prefix tokens after prefill; the vision tower itself is not
    autoregressively cached.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        rope_deltas: Optional[torch.Tensor] = None,
        use_cache: bool = True,
    ) -> Any:
        return self.model(
            input_ids=None,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
            rope_deltas=rope_deltas,
            use_cache=use_cache,
            return_dict=True,
        )


class Qwen2_5VLAdapter(torch.nn.Module):
    """
    Architecture-faithful adapter for Qwen2.5-VL.

    This mirrors the Hugging Face model layout closely enough to inspect or
    export the component boundaries we care about in ExecuTorch:
    - vision encoder
    - token embedding
    - text decoder

    It does not try to hide the extra multimodal metadata that Qwen2.5-VL
    requires.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        *,
        device_map: str = "cpu",
        torch_dtype: Optional[torch.dtype] = None,
        processor_kwargs: Optional[Mapping[str, Any]] = None,
        model_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__()
        AutoProcessor, Qwen2_5_VLForConditionalGeneration = (
            _load_qwen2_5_vl_transformers()
        )

        resolved_model_kwargs = dict(model_kwargs or {})
        resolved_model_kwargs.setdefault("device_map", device_map)
        if torch_dtype is not None:
            resolved_model_kwargs["torch_dtype"] = torch_dtype

        self.model_name = model_name
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            **resolved_model_kwargs,
        )
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            **dict(processor_kwargs or {}),
        )

        self.vision_encoder = Qwen2_5VLVisionEncoder(self.model)
        self.video_encoder = Qwen2_5VLVideoEncoder(self.model)
        self.token_embedding = Qwen2_5VLTokenEmbedding(self.model)
        self.text_decoder = Qwen2_5VLTextDecoder(self.model)

    def prepare_inputs(
        self,
        *,
        messages: Optional[Iterable[Mapping[str, Any]]] = None,
        images: Optional[Any] = None,
        videos: Optional[Any] = None,
        prompt: str = "Describe the image.",
        add_generation_prompt: bool = True,
    ) -> Dict[str, Any]:
        """
        Prepare processor outputs using the HF chat template path.

        If `messages` is omitted, a single-image prompt is synthesized.
        """

        resolved_messages = list(messages) if messages is not None else [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            resolved_messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        model_inputs = self.processor(
            text=[text],
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt",
        )
        return _clone_tensor_tree(model_inputs)

    def build_position_inputs(
        self,
        model_inputs: Mapping[str, Any],
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not hasattr(self.model, "get_rope_index"):
            return None, None

        position_ids, rope_deltas = self.model.get_rope_index(
            input_ids=model_inputs["input_ids"],
            image_grid_thw=model_inputs.get("image_grid_thw"),
            video_grid_thw=model_inputs.get("video_grid_thw"),
            attention_mask=model_inputs.get("attention_mask"),
        )
        return position_ids, rope_deltas

    def inspect_prefill_inputs(
        self,
        model_inputs: Mapping[str, Any],
    ) -> Qwen2_5VLPrefillInputs:
        inputs = _clone_tensor_tree(model_inputs)
        image_embeddings = None
        video_embeddings = None

        if "pixel_values" in inputs and "image_grid_thw" in inputs:
            image_embeddings = self.vision_encoder(
                inputs["pixel_values"],
                inputs["image_grid_thw"],
            )

        if "pixel_values_videos" in inputs and "video_grid_thw" in inputs:
            video_embeddings = self.video_encoder(
                inputs["pixel_values_videos"],
                inputs["video_grid_thw"],
            )

        position_ids, rope_deltas = self.build_position_inputs(inputs)
        token_embeddings = self.token_embedding(inputs["input_ids"])

        return Qwen2_5VLPrefillInputs(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            token_embeddings=token_embeddings,
            image_embeddings=image_embeddings,
            video_embeddings=video_embeddings,
            position_ids=position_ids,
            rope_deltas=rope_deltas,
            raw_model_inputs=inputs,
        )

    def prefill(self, model_inputs: Mapping[str, Any]) -> Any:
        """
        Run the full multimodal prefill path once.

        The returned `past_key_values` belongs to the text decoder and already
        contains the visual prefix contribution because the decoder saw the
        combined multimodal sequence during prefill.
        """

        inputs = _clone_tensor_tree(model_inputs)
        return self.model(
            **inputs,
            use_cache=True,
            return_dict=True,
        )

    def decode_one_token(
        self,
        token_id: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        past_key_values: Any,
        rope_deltas: Optional[torch.Tensor] = None,
    ) -> Any:
        token_embeddings = self.token_embedding(token_id)
        return self.text_decoder(
            inputs_embeds=token_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            rope_deltas=rope_deltas,
            use_cache=True,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
    ) -> Any:
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            use_cache=True,
            return_dict=True,
        )


class Qwen2_5VLModel(EagerModelBase):
    """
    Minimal eager wrapper so the adapter can live alongside the other examples.

    The synthetic example image keeps this example self-contained and avoids
    downloading demo assets.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        *,
        device_map: str = "cpu",
        torch_dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.adapter = Qwen2_5VLAdapter(
            model_name=model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
        )
        self._example_inputs: Optional[tuple[Any, ...]] = None

    def get_eager_model(self) -> torch.nn.Module:
        return self.adapter

    def get_example_inputs(self):
        if self._example_inputs is not None:
            return self._example_inputs

        blank = Image.new("RGB", (448, 448), color=(0, 0, 0))
        model_inputs = self.adapter.prepare_inputs(images=[blank])
        self._example_inputs = (
            model_inputs["input_ids"],
            model_inputs.get("attention_mask"),
            model_inputs.get("pixel_values"),
            model_inputs.get("image_grid_thw"),
            model_inputs.get("pixel_values_videos"),
            model_inputs.get("video_grid_thw"),
        )
        return self._example_inputs
