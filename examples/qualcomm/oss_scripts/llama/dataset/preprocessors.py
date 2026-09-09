# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from executorch.examples.qualcomm.oss_scripts.llama import LLMModelConfig
from executorch.examples.qualcomm.oss_scripts.llama.dataset.loaders import (
    load_audio_file,
)
from executorch.examples.qualcomm.oss_scripts.llama.decoder_constants import (
    AUDIO_ENCODER,
    VISION_ENCODER,
)
from executorch.examples.qualcomm.oss_scripts.llama.encoder.encoder_config import (
    AudioModalityConfig,
    VisionModalityConfig,
)
from executorch.examples.qualcomm.oss_scripts.llama.tokenizer import (
    ALM_SPECIAL_TOKENS,
    AUDIO_TOKEN,
    IMG_TOKEN,
    VLM_SPECIAL_TOKENS,
)

from transformers import AutoProcessor
from transformers.image_utils import load_image
from PIL import Image


def _qwen_vl_image_to_patches(
    image: Image.Image,
    img_resized_h: int,
    img_resized_w: int,
    patch_size: int = 16,
    temporal_patch_size: int = 2,
    merge_size: int = 2,
) -> torch.Tensor:
    image = image.convert("RGB").resize(
        (img_resized_w, img_resized_h), Image.Resampling.BICUBIC
    )
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - 0.5) / 0.5
    array = np.transpose(array, (2, 0, 1))

    patches = np.expand_dims(np.expand_dims(array, axis=0), axis=1)
    remainder = patches.shape[1] % temporal_patch_size
    if remainder:
        repeats = np.repeat(patches[:, -1:], temporal_patch_size - remainder, axis=1)
        patches = np.concatenate([patches, repeats], axis=1)

    batch_size = patches.shape[0]
    grid_t = patches.shape[1] // temporal_patch_size
    channels = patches.shape[2]
    grid_h = img_resized_h // patch_size
    grid_w = img_resized_w // patch_size
    patches = patches.reshape(
        batch_size,
        grid_t,
        temporal_patch_size,
        channels,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )
    patches = patches.transpose(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
    patches = patches.reshape(
        batch_size * grid_t * grid_h * grid_w,
        channels * temporal_patch_size * patch_size * patch_size,
    )
    return torch.from_numpy(patches).to(torch.float32)


class ModalityPreprocessor:
    """Wraps AutoProcessor for one model repo; caches the processor across calls."""

    def __init__(self, repo_id: str):
        self._processor = AutoProcessor.from_pretrained(repo_id)

    def preprocess_audio(
        self,
        config: AudioModalityConfig,
        prompt: str,
        wav: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        # Pad to fixed length so input_features has shape [1, n_bins, input_dim]
        hop_length = self._processor.audio_processor.melspec_kwargs["hop_length"]
        target_len = (config.n_bins * 2 - 1) * hop_length
        pad = target_len - wav.shape[-1]
        if pad > 0:
            wav = F.pad(wav, (0, pad))
        elif pad < 0:
            suggested = (wav.shape[-1] // hop_length + 1) // 2
            raise ValueError(
                f"Audio length ({wav.shape[-1]} samples) exceeds target ({target_len} samples) "
                f"derived from n_bins={config.n_bins}. Set n_bins >= {suggested} in the config."
            )
        return (self._processor(prompt, wav, return_tensors="pt").input_features,)

    def preprocess_images(
        self,
        config: VisionModalityConfig,
        prompt: str,
        images: list,
    ) -> List[Tuple[torch.Tensor, ...]]:
        """Applies the vision processor to all images → list of (pixel_values,) per patch."""
        encoder_name = config.encoder_class.__name__
        if encoder_name in {"Qwen3VLVisionEncoder", "Qwen35VLVisionEncoder"}:
            patches = [
                _qwen_vl_image_to_patches(
                    image,
                    config.img_resized_h,
                    config.img_resized_w,
                )
                for image in images
            ]
            if encoder_name == "Qwen35VLVisionEncoder":
                return [(patch.unsqueeze(0).contiguous(),) for patch in patches]
            return [(patch,) for patch in patches]

        if encoder_name == "Qwen2_5VLVisionEncoder":
            resized_images = [
                image.resize(
                    (config.img_resized_w, config.img_resized_h),
                    Image.Resampling.BICUBIC,
                )
                for image in images
            ]
            pixel_values = self._processor.image_processor.preprocess(
                images=resized_images,
                return_tensors="pt",
                do_resize=False,
            )["pixel_values"]
            return [(pixel_values,)]

        pixel_values = self._processor(
            text=prompt,
            images=images,
            return_tensors="pt",
            crop_to_patches=False,
            size={"height": config.img_resized_h, "width": config.img_resized_w},
        ).pixel_values
        assert pixel_values.dim() in (
            4,
            5,
        ), f"Unsupported pixel_values dim={pixel_values.dim()}; expected 4D or 5D."
        # HTP Prepare failed when pixel_values has 5D dimension
        if pixel_values.dim() == 5:
            pixel_values = pixel_values.squeeze(0)  # (N, C, H, W)
        return [(pixel_values[i][None, ...],) for i in range(len(pixel_values))]


def preprocess_encoder_inputs(
    config: LLMModelConfig,
    decoder_model: str,
    audio_paths: List[str],
    image_paths: List[str],
) -> Dict[str, List[Tuple[torch.Tensor, ...]]]:
    """
    Preprocess raw encoder inputs for runtime inference.

    Separated from EncoderDatasetBuilder (calibration-only) because this is
    a runtime inference concern, not a calibration data concern.
    """
    preprocessor = ModalityPreprocessor(config.repo_id)
    result = {}
    for modality, paths in (
        (AUDIO_ENCODER, audio_paths),
        (VISION_ENCODER, image_paths),
    ):
        if not hasattr(config, modality) or not paths:
            continue
        modality_config = getattr(config, modality)
        if issubclass(modality_config, AudioModalityConfig):
            token = ALM_SPECIAL_TOKENS[decoder_model][AUDIO_TOKEN]
            normalized_prompt = token * len(paths)
            result[modality] = [
                preprocessor.preprocess_audio(
                    modality_config,
                    normalized_prompt,
                    load_audio_file(p, config.repo_id),
                )
                for p in paths
            ]
        elif issubclass(modality_config, VisionModalityConfig):
            token = VLM_SPECIAL_TOKENS[decoder_model][IMG_TOKEN]
            normalized_prompt = token * len(paths)
            result[modality] = preprocessor.preprocess_images(
                modality_config, normalized_prompt, [load_image(p) for p in paths]
            )
    return result
