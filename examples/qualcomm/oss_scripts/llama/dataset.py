# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import io
import os
from typing import Callable, Dict, List, Optional

import numpy as np
import requests

import numpy as np
import torch
from executorch.examples.qualcomm.oss_scripts.llama import LLMModelConfig
from executorch.examples.qualcomm.oss_scripts.llama.decoder_constants import (
    AUDIO_ENCODER,
    TEXT_DECODER,
    TEXT_ENCODER,
    TOK_EMBEDDING,
    VISION_ENCODER,
)
from executorch.examples.qualcomm.oss_scripts.llama.encoder.encoder_config import (
    AudioModalityConfig,
    MultiModalityConfig,
    VisionModalityConfig,
)
from executorch.examples.qualcomm.oss_scripts.llama.tokenizer import TokenizerWrapper
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoProcessor
from transformers.image_utils import load_image


def _qwen_vl_image_to_patches(
    image: Image.Image,
    img_resized_h: int,
    img_resized_w: int,
    patch_size: int,
    temporal_patch_size: int,
    merge_size: int,
) -> torch.Tensor:
    image = image.convert("RGB").resize(
        (img_resized_w, img_resized_h), Image.Resampling.BICUBIC
    )
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - 0.5) / 0.5
    array = np.transpose(array, (2, 0, 1))

    patches = np.expand_dims(np.expand_dims(array, axis=0), axis=1)
    if patches.shape[1] % temporal_patch_size != 0:
        repeats = np.repeat(
            patches[:, -1:],
            temporal_patch_size - (patches.shape[1] % temporal_patch_size),
            axis=1,
        )
        patches = np.concatenate([patches, repeats], axis=1)

    batch_size = patches.shape[0]
    grid_t = patches.shape[1] // temporal_patch_size
    channel = patches.shape[2]
    grid_h = img_resized_h // patch_size
    grid_w = img_resized_w // patch_size

    patches = patches.reshape(
        batch_size,
        grid_t,
        temporal_patch_size,
        channel,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )
    patches = patches.transpose(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
    flatten_patches = patches.reshape(
        batch_size * grid_t * grid_h * grid_w,
        channel * temporal_patch_size * patch_size * patch_size,
    )
    return torch.from_numpy(flatten_patches).to(torch.float32)


class DatasetBuilder:
    def __init__(
        self,
        control_args: argparse.Namespace,
        config: LLMModelConfig,
        tokenizer_wrapper: TokenizerWrapper,
    ):
        # Directory path where processed data artifacts will be saved
        self.control_args = control_args
        self.config = config
        self.tokenizer_wrapper = tokenizer_wrapper

        self.artifact = control_args.artifact
        self.repo_id = config.repo_id

    def _build_audio_dataset(
        self, config: AudioModalityConfig, prompt: str, files_path: List[str]
    ):
        """
        This will process audio using the HuggingFace processor and save
        the processed audio for runtime evaluation.

        Args:
            config (AudioModalityConfig): containing audio URL
            prompt (str): Text prompt to be processed alongside the audio

        Returns:
            tuple of audio feature tensors
        """
        try:
            import soundfile
        except ImportError:
            raise ImportError(
                "Please install the `soundfile` package via `pip install soundfile` for audio data loading"
            )

        dataset = []
        processor = AutoProcessor.from_pretrained(self.repo_id)
        for audio_path in files_path:
            if isinstance(audio_path, str) and audio_path.startswith(
                ("http://", "https://")
            ):
                resp = requests.get(audio_path, timeout=60)
                resp.raise_for_status()
                data = io.BytesIO(resp.content)
                wav, sr = soundfile.read(data, always_2d=False)
            else:
                if not os.path.exists(audio_path):
                    try:
                        audio_path = hf_hub_download(
                            repo_id=self.config.repo_id, filename=audio_path
                        )
                    except Exception:
                        raise FileNotFoundError(
                            f"Audio file {audio_path} not found locally or in HuggingFace repository {self.config.repo_id}."
                        )
                wav, sr = soundfile.read(audio_path, always_2d=False)
            wav = torch.from_numpy(wav).float().unsqueeze(0)  # [1, T]

            # Pad to fixed length so input_features has shape [1, n_bins, input_dim]
            hop_length = processor.audio_processor.melspec_kwargs["hop_length"]
            target_raw_length = (config.n_bins * 2 - 1) * hop_length
            pad_size = target_raw_length - wav.shape[-1]
            if pad_size > 0:
                wav = torch.nn.functional.pad(wav, (0, pad_size))
            elif pad_size < 0:
                suggested_n_bins = (wav.shape[-1] // hop_length + 1) // 2
                raise ValueError(
                    f"Audio length ({wav.shape[-1]} samples) exceeds target ({target_raw_length} samples) "
                    f"derived from n_bins={config.n_bins}. Set n_bins >= {suggested_n_bins} in the config to avoid information loss."
                )

            # Process audio with text prompt using HuggingFace processor
            input_features = processor(prompt, wav, return_tensors="pt").input_features
            dataset.append((input_features,))

        return dataset

    def _build_vision_dataset(
        self, config: VisionModalityConfig, prompt: str, files_path: List[str]
    ):
        """
        This will processes images using the HuggingFace processor and saves
        the processed pixel values for runtime evaluation.

        Args:
            config (VisionModalityConfig): containing image URL and resize parameters
            prompt (str): Text prompt
            files_path (List[str]): List of file paths for images. Each path can be either a URL or a local file path.

        Returns:
            tuple of pixel values tensors
        """

        images = [load_image(image_path) for image_path in files_path]
        if self.control_args.decoder_model in {
            "qwen3_vl-2b_instruct",
            "lincal3_vl-2b_instruct",
        }:
            return [
                (
                    _qwen_vl_image_to_patches(
                        image,
                        config.img_resized_h,
                        config.img_resized_w,
                        patch_size=16,
                        temporal_patch_size=2,
                        merge_size=2,
                    ),
                )
                for image in images
            ]
        if config.encoder_class.__name__ == "Qwen35VLVisionEncoder":
            return self._build_qwen35_vl_vision_dataset(config, images)

        # Process image with text prompt using HuggingFace processor
        # Some HF processors (e.g. InternVL3) need to pass text arg or it will cause error and process failed
        processor = AutoProcessor.from_pretrained(self.repo_id)
        if self.control_args.decoder_model in {
            "qwen2_5_vl-3b_instruct",
            "lincal_vl-3b_instruct",
        }:
            resized_images = [
                image.resize(
                    (config.img_resized_w, config.img_resized_h),
                    Image.Resampling.BICUBIC,
                )
                for image in images
            ]
            pixel_values = processor.image_processor.preprocess(
                images=resized_images,
                return_tensors="pt",
                do_resize=False,
            )["pixel_values"]
            return [(pixel_values,)]

        pixel_values = processor(
            text=prompt,
            images=images,
            return_tensors="pt",
            crop_to_patches=False,
            size={
                "height": config.img_resized_h,
                "width": config.img_resized_w,
            },
        ).pixel_values

        assert pixel_values.dim() in (4, 5), (
            f"Unsupported pixel_values dim={pixel_values.dim()}); "
            f"expected 5D (1,N,C,H,W) or 4D (N,C,H,W)."
        )

        # HTP Prepare failed when pixel_values has 5D dimension, so we squeeze the batch dimension here.
        if pixel_values.dim() == 5:
            pixel_values = pixel_values.squeeze(0)  # (N, C, H, W)

        # save image file for runtime evaluation
        return [(pixel_values[i][None, ...],) for i in range(len(pixel_values))]

    def _build_qwen35_vl_vision_dataset(
        self, config: VisionModalityConfig, images: List[Image.Image]
    ):
        patches = []
        patch_size = 16
        temporal_patch_size = 2
        merge_size = 2
        image_mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)[:, None, None]
        image_std = np.array([0.5, 0.5, 0.5], dtype=np.float32)[:, None, None]

        for image in images:
            image = image.convert("RGB").resize(
                (config.img_resized_w, config.img_resized_h),
                resample=Image.Resampling.BICUBIC,
            )
            image = np.asarray(image, dtype=np.float32) / 255.0
            image = np.transpose(image, (2, 0, 1))
            image = (image - image_mean) / image_std

            image = np.expand_dims(image, axis=0)
            image = np.expand_dims(image, axis=1)
            remainder = image.shape[1] % temporal_patch_size
            if remainder:
                repeats = np.repeat(
                    image[:, -1:],
                    temporal_patch_size - remainder,
                    axis=1,
                )
                image = np.concatenate([image, repeats], axis=1)

            batch_size = image.shape[0]
            grid_t = image.shape[1] // temporal_patch_size
            channels = image.shape[2]
            grid_h = config.img_resized_h // patch_size
            grid_w = config.img_resized_w // patch_size
            image = image.reshape(
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
            image = image.transpose(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
            image = image.reshape(
                batch_size * grid_t * grid_h * grid_w,
                channels * temporal_patch_size * patch_size * patch_size,
            )
            patches.append((torch.from_numpy(image).unsqueeze(0).contiguous(),))

        return patches

    def _build_dataset_for_encoder(
        self,
        config: MultiModalityConfig,
        prompt: str,
        files_path: List[str],
    ) -> Optional[tuple]:
        if issubclass(config, AudioModalityConfig):
            return self._build_audio_dataset(config, prompt, files_path)
        elif issubclass(config, VisionModalityConfig):
            return self._build_vision_dataset(config, prompt, files_path)
        else:
            raise NotImplementedError(
                f"Dataset building for {config} is not yet supported. "
            )

    def prepare_calibration_dataset(
        self,
        prompts: List[str],
        chat_template: Callable,
    ):
        # 1. Initialize data
        # Shape convention: (num_samples, num_turns).
        # Currently, user prompt calibration is one-shot per prompt (num_samples = 1).
        calibration_data: Dict[str, List[List]] = {
            # Encoders / embeddings: initialize an empty turn list for each prompt.
            AUDIO_ENCODER: [[] for _ in range(len(prompts))],
            TEXT_ENCODER: [[] for _ in range(len(prompts))],
            VISION_ENCODER: [[] for _ in range(len(prompts))],
            TOK_EMBEDDING: [[] for _ in range(len(prompts))],
            # Decoder targets: one string per prompt.
            TEXT_DECODER: ["" for _ in range(len(prompts))],
        }

        # 2. Prepare messages for multi-turn conversation
        messages = self.tokenizer_wrapper.prepare_messages(prompts)

        # 3. build dataset by modality
        is_multimodal = any(
            [
                hasattr(self.config, AUDIO_ENCODER),
                hasattr(self.config, VISION_ENCODER),
            ]
        )
        for turn_idx, message in enumerate(messages):
            prompt = message["text"]

            # 3.1. Apply chat template formatting if available (for instruction-tuned/reasoning models)
            prompt = (
                self.tokenizer_wrapper.apply_prompt_template(
                    chat_template, prompt, self.control_args.system_prompt
                )
                if chat_template is not None
                else prompt
            )

            # 3.2 Build calibration datasets for each available encoder modality
            for modality in [AUDIO_ENCODER, TEXT_ENCODER, VISION_ENCODER]:
                if not hasattr(self.config, modality) or not message["files_path"]:
                    continue

                data = self._build_dataset_for_encoder(
                    getattr(self.config, modality),
                    prompt,
                    message["files_path"],
                )
                calibration_data[modality][turn_idx] = data

            # 3.3. Expand multimodal tokens in prompt for decoder
            prompt = (
                self.tokenizer_wrapper.prepare_multimodal_prompt(prompt)
                if is_multimodal
                else prompt
            )

            # Add prompt to decoder calibration data
            calibration_data[TEXT_DECODER][turn_idx] = prompt

        return calibration_data
