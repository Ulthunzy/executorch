# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from dataclasses import dataclass

from executorch.examples.qualcomm.oss_scripts.llama.encoder.encoder_quant_recipe import (
    EncoderQuantRecipe,
    GraniteSpeechEncoderQuantRecipe,
    InternVL3EncoderQuantRecipe,
    Qwen2_5_VL_Encoder_QuantRecipe,
    Qwen35VLEncoderQuantRecipe,
    SmolVLMEncoderQuantRecipe,
)
from executorch.examples.qualcomm.oss_scripts.llama.model.audio_encoder import (
    GraniteSpeechCTCEncoderWrapper,
)
from executorch.examples.qualcomm.oss_scripts.llama.model.vision_encoder import (
    Idefics3VisionEncoder,
    InternVL3VisionEncoder,
    Qwen2_5VLVisionEncoder,
    Qwen35VLVisionEncoder,
    Qwen3VLVisionEncoder,
)


@dataclass(init=False, frozen=True)
class MultiModalityConfig(ABC):
    """
    Base config class for late fusion modality encoders.

    Attributes:
        encoder_class: The encoder class that implements the modality processing.
        quant_recipe: Quantization recipe for optimizing the encoder.
        num_sharding: Specify the number of splits by inserting the fallback custom op. The graph will be split evenly by layers. Only larger encoder needs sharding, e.g., GraniteSpeech with 24 layers. For smaller encoders like SmolVLM with 12 layers, sharding is not necessary.
    """

    encoder_class: type
    quant_recipe: EncoderQuantRecipe
    num_sharding: int = 1

    @abstractmethod
    def create_encoder(self, config):
        pass


@dataclass(init=False, frozen=True)
class AudioModalityConfig(MultiModalityConfig):
    """
    Base config for audio modality encoders.

    Attributes:
        audio_seq_len: Number of audio tokens in the sequence.
    """

    audio_seq_len: int
    n_bins: int

    def create_encoder(self, config):
        return self.encoder_class(config, n_bins=self.n_bins)


@dataclass(init=False, frozen=True)
class VisionModalityConfig(MultiModalityConfig):
    """
    Base config for vision modality encoders.

    Attributes:
        encoder_class: The vision encoder class implementation.
        quant_recipe: Quantization recipe for the vision encoder.
        img_seq_len: Number of image tokens/patches in the sequence.
        img_resized_h: Target height for image resizing (pixels).
        img_resized_w: Target width for image resizing (pixels).
        use_fp16: Skip PTQ and compile floating point vision operators on HTP.
    """

    img_seq_len: int
    img_resized_h: int
    img_resized_w: int
    use_fp16: bool = False

    def create_encoder(self, config):
        return self.encoder_class(
            config, img_resized_h=self.img_resized_h, img_resized_w=self.img_resized_w
        )


@dataclass(init=False, frozen=True)
class GraniteSpeechEncoder(AudioModalityConfig):
    """
    Config for GraniteSpeech audio encoder.
    """

    encoder_class = GraniteSpeechCTCEncoderWrapper
    audio_seq_len = 171
    n_bins = 844
    quant_recipe = GraniteSpeechEncoderQuantRecipe
    num_sharding = 8


@dataclass(init=False, frozen=True)
class SmolVLMEncoder(VisionModalityConfig):
    """
    Config for SmolVLM vision encoder.
    """

    encoder_class = Idefics3VisionEncoder
    img_seq_len = 64
    img_resized_h = 512
    img_resized_w = 512
    quant_recipe = SmolVLMEncoderQuantRecipe


@dataclass(init=False, frozen=True)
class InternVL3Encoder(VisionModalityConfig):
    """
    Config for InternVL3 vision encoder.
    """

    encoder_class = InternVL3VisionEncoder
    img_seq_len = 256
    img_resized_h = 448
    img_resized_w = 448
    quant_recipe = InternVL3EncoderQuantRecipe


@dataclass(init=False, frozen=True)
class Qwen2_5VL3BEncoder(VisionModalityConfig):
    """
    Config for Qwen2.5-VL 3B vision encoder.
    """

    encoder_class = Qwen2_5VLVisionEncoder
    img_seq_len = 256
    img_resized_h = 448
    img_resized_w = 448
    img_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    quant_recipe = Qwen2_5_VL_Encoder_QuantRecipe


@dataclass(init=False, frozen=True)
class Qwen3VL2BEncoder(VisionModalityConfig):
    """
    Config for Qwen3-VL 2B vision encoder.
    """

    encoder_class = Qwen3VLVisionEncoder
    img_seq_len = 196
    img_resized_h = 448
    img_resized_w = 448
    img_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    quant_recipe = Qwen2_5_VL_Encoder_QuantRecipe


@dataclass(init=False, frozen=True)
class Qwen35VLEncoder(VisionModalityConfig):
    """
    Config for Qwen3.5-VL vision encoder.
    """

    encoder_class = Qwen35VLVisionEncoder
    img_seq_len = 196
    img_resized_h = 448
    img_resized_w = 448
    img_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    quant_recipe = Qwen35VLEncoderQuantRecipe
    use_fp16 = False
