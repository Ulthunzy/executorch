# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
import torch.nn.functional as F

from executorch.examples.qualcomm.utils import replace_module_with_custom_class
from torch import nn
from executorch.examples.qualcomm.oss_scripts.llama.model import ROTARY_EMB_REGISTRY

from transformers.models.idefics3.modeling_idefics3 import (
    BaseModelOutput,
    Idefics3Config,
    Idefics3Connector,
    Idefics3Encoder,
    Idefics3PreTrainedModel,
    Idefics3VisionConfig,
    Idefics3VisionEmbeddings,
)
from transformers.models.internvl.modeling_internvl import (
    InternVLConfig,
    InternVLMultiModalProjector,
    InternVLVisionModel,
)
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLMLP,
    Qwen2_5_VisionTransformerPretrainedModel,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    apply_rotary_pos_emb_vision as apply_qwen3_vl_rotary_pos_emb,
    BaseModelOutputWithDeepstackFeatures,
    Qwen3VLVisionConfig,
    Qwen3VLVisionMLP,
    Qwen3VLVisionModel,
)


# Custom implementation based on `transformers/models/idefics3/modeling_idefics3/Idefics3VisionEmbeddings.py` (Transformers v5.0.0rc1)
#
# Qualcomm optimization:
# Precompute and register positional IDs as a buffer to avoid computation during forward passes.
class CustomIdefics3VisionEmbeddings(Idefics3VisionEmbeddings):
    def __init__(self, config: Idefics3VisionConfig):
        super().__init__(config)
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches_per_side = self.image_size // self.patch_size
        self.num_patches = self.num_patches_per_side**2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

        # ========================== Qualcomm Changed: Precompute position ids ==========================
        S = self.num_patches_per_side
        H = self.image_size // self.patch_size
        W = self.image_size // self.patch_size

        # Assume batch size of 1 for precomputation
        B = 1

        # Create boundaries for bucketize
        boundaries = torch.arange(1.0 / S, 1.0, 1.0 / S)  # [S-1]

        # Assume full attention mask (all True) for precomputation
        patch_attention_mask = torch.ones((B, H, W), dtype=torch.bool)

        # Calculate nb_h and nb_w (number of True values in first column/row)
        nb_h = patch_attention_mask[:, :, 0].sum(dim=1)  # [B]
        nb_w = patch_attention_mask[:, 0, :].sum(dim=1)  # [B]
        nb_h_clamped = torch.clamp(nb_h, min=1).float()  # [B]
        nb_w_clamped = torch.clamp(nb_w, min=1).float()  # [B]

        # Create indices
        h_idx = torch.arange(H, dtype=torch.float32).unsqueeze(0)  # [1, H]
        w_idx = torch.arange(W, dtype=torch.float32).unsqueeze(0)  # [1, W]

        # Calculate fractional positions
        frac_h = (h_idx / nb_h_clamped.unsqueeze(1)) * (1.0 - 1e-6)  # [B, H]
        frac_w = (w_idx / nb_w_clamped.unsqueeze(1)) * (1.0 - 1e-6)  # [B, W]

        # Bucketize to get position indices
        bucket_h = torch.bucketize(frac_h, boundaries, right=True)  # [B, H]
        bucket_w = torch.bucketize(frac_w, boundaries, right=True)  # [B, W]

        # Create position grid: pos = h * S + w
        pos_grid = bucket_h.unsqueeze(2) * S + bucket_w.unsqueeze(1)  # [B, H, W]
        pos_full = pos_grid.reshape(B, H * W)  # [B, H*W]

        # Apply attention mask
        mask_flat = patch_attention_mask.view(B, H * W)  # [B, H*W]
        position_ids = torch.where(
            mask_flat, pos_full, torch.zeros_like(pos_full)
        )  # [B, H*W]

        # Register the precomputed position_ids
        self.register_buffer("position_ids", position_ids, persistent=False)
        # ===============================================================================================

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        # pixel_values: [B, 3, max_im_h, max_im_w]
        B, _, max_im_h, max_im_w = pixel_values.shape

        # 1) patch embedding: [B, C, H, W] -> [B, H*W, C]
        patch_embeds = self.patch_embedding(pixel_values)  # [B, C, H, W]
        embeddings = patch_embeds.flatten(2).transpose(1, 2)

        # ========================== Qualcomm Changed: Use the precomputed position_ids ==========================
        position_ids = self.position_ids.to(pixel_values.device)
        # ========================================================================================================

        # Expand to match batch size if needed
        if B > 1 and position_ids.size(0) == 1:
            position_ids = position_ids.expand(B, -1)

        # Add positional embedding
        embeddings = embeddings + self.position_embedding(position_ids)

        return embeddings


# Custom implementation based on `transformers/models/idefics3/modeling_idefics3/Idefics3VisionTransformer.py` (Transformers v5.0.0rc1)
#
# Qualcomm changes:
# Assume the image is non-empty and skip attention mask propagation to the encoder
class CustomIdefics3VisionTransformer(Idefics3PreTrainedModel):
    config: Idefics3VisionConfig

    def __init__(self, config: Idefics3VisionConfig):
        super().__init__(config)
        self.embeddings = Idefics3VisionEmbeddings(config)
        self.encoder = Idefics3Encoder(config)
        self.post_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def forward(self, pixel_values):
        hidden_states = self.embeddings(pixel_values=pixel_values)
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)

        return BaseModelOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


# Custom implementation based on `transformers/models/idefics3/modeling_idefics3.py` (Transformers v5.0.0rc1).
#
# Qualcomm optimization:
# - Dynamic shape support is removed; computations are now static for efficiency.
# - After image preprocessing, we assume all patch images are valid (non-empty),
#   so attention masks are no longer required in the Vision Transformer.
class Idefics3VisionEncoder(Idefics3PreTrainedModel):
    def __init__(
        self, config: Idefics3Config, img_resized_h: int = 512, img_resized_w: int = 512
    ):
        super().__init__(config)
        self.vision_model = CustomIdefics3VisionTransformer._from_config(
            config.vision_config
        )
        self.connector = Idefics3Connector(config)
        self.config = config
        self.img_resized_h = img_resized_h
        self.img_resized_w = img_resized_w

        replace_module_with_custom_class(
            self.vision_model,
            target_class=Idefics3VisionEmbeddings,
            custom_class=CustomIdefics3VisionEmbeddings,
            strict=True,
            extra_custom_kwargs={"config": config.vision_config},
        )

    def preprocess(self, pixel_values: Tuple[torch.FloatTensor]) -> Tuple[torch.Tensor]:
        # HTP Prepare failed when pixel_values has 5D dimension, so we squeeze the batch dimension here.
        pixel_values = pixel_values[0]
        return (pixel_values.squeeze(0),)

    def get_example_inputs(self):
        # pixel values - use config dimensions instead of hardcoded values
        return (
            torch.randn(
                (1, 3, self.img_resized_h, self.img_resized_w), dtype=torch.float32
            ),
        )

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_attention_mask: torch.LongTensor = None,
    ):
        """
        Encodes images into continuous embeddings that can be forwarded to the language model.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input images.
            pixel_attention_mask (`torch.LongTensor`, *optional*):
                The attention mask indicating padded regions in the image.
        """
        pixel_values = pixel_values[None, ...]
        batch_size, num_images, num_channels, height, width = pixel_values.shape
        pixel_values = pixel_values.view(
            batch_size * num_images, *pixel_values.shape[2:]
        )

        # ========================== Qualcomm Changed ==========================
        # Since dynamic shapes are unsupported, we assume all patches are valid and don't need `patch_attention_mask`.

        # # Remove padding images - padding images are full 0.
        # nb_values_per_image = pixel_values.shape[1:].numel()
        # real_images_inds = (pixel_values == 0.0).sum(dim=(-1, -2, -3)) != nb_values_per_image
        # pixel_values = pixel_values[real_images_inds].contiguous()

        # # Handle the vision attention mask
        # if pixel_attention_mask is None:
        #     pixel_attention_mask = torch.ones(
        #         size=(pixel_values.size(0), pixel_values.size(2), pixel_values.size(3)),
        #         dtype=torch.bool,
        #         device=pixel_values.device,
        #     )
        # else:
        #     # Remove padding images from the mask
        #     pixel_attention_mask = pixel_attention_mask.view(batch_size * num_images, *pixel_attention_mask.shape[2:])
        #     pixel_attention_mask = pixel_attention_mask[real_images_inds].contiguous()

        # patch_size = self.config.vision_config.patch_size
        # patches_subgrid = pixel_attention_mask.unfold(dimension=1, size=patch_size, step=patch_size)
        # patches_subgrid = patches_subgrid.unfold(dimension=2, size=patch_size, step=patch_size)
        # patch_attention_mask = (patches_subgrid.sum(dim=(-1, -2)) > 0).bool()
        # ======================================================================

        # Get sequence from the vision encoder
        image_hidden_states = self.vision_model(pixel_values=pixel_values)
        image_hidden_states.last_hidden_state

        # Modality projection & resampling
        image_hidden_states = self.connector(image_hidden_states.last_hidden_state)
        return image_hidden_states


# Copy from transformers/models/internvl/modeling_internvl.py (Transformers v5.0.0rc1).
class InternVL3VisionEncoder(torch.nn.Module):
    def __init__(
        self, config: InternVLConfig, img_resized_h: int = 448, img_resized_w: int = 448
    ):
        super(InternVL3VisionEncoder, self).__init__()
        self.vision_tower = InternVLVisionModel(config.vision_config)
        self.multi_modal_projector = InternVLMultiModalProjector(config)
        self.config = config
        self.img_resized_h = img_resized_h
        self.img_resized_w = img_resized_w

    def preprocess(self, pixel_values: Tuple[torch.FloatTensor]) -> Tuple[torch.Tensor]:
        return pixel_values

    def get_example_inputs(self):
        # pixel values - use config dimensions instead of hardcoded values
        return (
            torch.randn(
                (1, 3, self.img_resized_h, self.img_resized_w), dtype=torch.float32
            ),
        )

    def pixel_shuffle(self, vision_features: torch.Tensor, scale_factor: float = 0.5):
        """Perform pixel shuffle downsampling on vision features.

        Args:
            vision_features (`torch.Tensor`):
                Input tensor of shape (batch_size, width, height, channels).
            scale_factor (`float`, *optional*, defaults to `0.5`):
                Factor by which to downsample. Default is 0.5, which halves the dimensions.

        Returns:
            vision_features (`torch.Tensor`):
                Downsampled tensor of shape (batch_size, height*scale_factor, width*scale_factor, channels/(scale_factor^2)).
        """
        batch_size, width, height, channels = vision_features.size()

        if height % scale_factor != 0 or width % scale_factor != 0:
            raise ValueError(
                "Height and width must be divisible by scale_factor for proper downsampling."
            )

        # Reshape to allow downsampling
        vision_features = vision_features.view(
            batch_size, width, int(height * scale_factor), int(channels / scale_factor)
        )
        # Permute dimensions to align downsampled axis correctly
        vision_features = vision_features.permute(0, 2, 1, 3).contiguous()

        # Reshape to achieve final downsampled dimensions
        vision_features = vision_features.view(
            batch_size,
            int(height * scale_factor),
            int(width * scale_factor),
            int(channels / (scale_factor**2)),
        )

        # Swap height and width back for proper orientation
        vision_features = vision_features.permute(0, 2, 1, 3).contiguous()

        return vision_features

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        vision_feature_layer=-1,
        vision_feature_select_strategy="default",
    ):
        """
        Obtains image last hidden states from the vision tower and apply multimodal projection.

        Args:
            pixel_values (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`)
               The tensors corresponding to the input images.
            vision_feature_layer (`int` or `list[int]`):
                Layer index or list of layer indices to extract features from.
        Returns:
            vision_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`.
        """
        vision_feature_layer = (
            vision_feature_layer
            if vision_feature_layer is not None
            else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        downsample_ratio = self.config.downsample_ratio
        if vision_feature_layer == -1:
            vision_features = self.vision_tower(
                pixel_values=pixel_values
            ).last_hidden_state
        else:
            vision_features = self.vision_model(
                pixel_values=pixel_values
            ).hidden_states[vision_feature_layer]

        if vision_feature_select_strategy == "default":
            vision_features = vision_features[:, 1:, :]

        # Calculate dimensions based on vision features
        channels = vision_features.shape[1]
        feature_size = int(channels**0.5)
        batch_size = vision_features.shape[0]

        # Reshape tensor to spatial dimensions
        vision_features = vision_features.reshape(
            batch_size, feature_size, feature_size, -1
        )

        # Apply downsampling using pixel shuffle
        vision_features = self.pixel_shuffle(
            vision_features, scale_factor=downsample_ratio
        )

        # Reshape tensor to prepare for projection
        vision_features = vision_features.reshape(
            batch_size, -1, vision_features.shape[-1]
        )

        # Project features through multi-modal projector
        vision_features = self.multi_modal_projector(vision_features)
        return vision_features


def _rotate_half_vision(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class StaticQwen35VLVisionAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim**-0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states)
            .reshape(seq_length, 3, self.num_heads, self.head_dim)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        cos, sin = position_embeddings
        cos = cos.unsqueeze(1).to(dtype=query_states.dtype)
        sin = sin.unsqueeze(1).to(dtype=query_states.dtype)
        query_states = query_states * cos + _rotate_half_vision(query_states) * sin
        key_states = key_states * cos + _rotate_half_vision(key_states) * sin

        query_states = query_states.transpose(0, 1)
        key_states = key_states.transpose(0, 1)
        value_states = value_states.transpose(0, 1)
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1))
        attn_weights = attn_weights * self.scaling
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(0, 1).reshape(seq_length, self.dim)
        return self.proj(attn_output).contiguous()


class StaticQwen35VLVisionBlock(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = StaticQwen35VLVisionAttention(config)
        self.mlp = Qwen3VLVisionMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            position_embeddings=position_embeddings,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class StaticQwen35VLVisionModel(Qwen3VLVisionModel):
    def __init__(self, config, grid_thw: torch.Tensor):
        super().__init__(config)
        proj = self.patch_embed.proj
        self.patch_embed_weight_flat = nn.Parameter(
            proj.weight.detach().flatten(1).clone()
        )
        self.patch_embed_bias = nn.Parameter(proj.bias.detach().clone())

        self.blocks = nn.ModuleList(
            [StaticQwen35VLVisionBlock(config) for _ in range(config.depth)]
        )

        static_grid_thw = grid_thw.to(dtype=torch.int64)
        static_rotary_pos_emb = super().rot_pos_emb(static_grid_thw).detach()
        static_pos_embeds = super().fast_pos_embed_interpolate(static_grid_thw).detach()
        self.register_buffer("static_grid_thw", static_grid_thw, persistent=False)
        self.register_buffer(
            "static_rotary_pos_emb", static_rotary_pos_emb, persistent=False
        )
        self.register_buffer("static_pos_embeds", static_pos_embeds, persistent=False)

        def _sync_static_vision_buffers(module, _incompatible_keys):
            with torch.no_grad():
                module.patch_embed_weight_flat.copy_(
                    module.patch_embed.proj.weight.flatten(1)
                )
                module.patch_embed_bias.copy_(module.patch_embed.proj.bias)
                module.static_pos_embeds.copy_(
                    Qwen3VLVisionModel.fast_pos_embed_interpolate(
                        module, module.static_grid_thw
                    )
                )

        self.register_load_state_dict_post_hook(_sync_static_vision_buffers)

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor = None):
        del grid_thw
        target_dtype = self.patch_embed_weight_flat.dtype
        hidden_states = F.linear(
            hidden_states.to(dtype=target_dtype),
            self.patch_embed_weight_flat,
            self.patch_embed_bias,
        )
        hidden_states = hidden_states + self.static_pos_embeds.to(
            dtype=hidden_states.dtype, device=hidden_states.device
        )

        rotary_pos_emb = self.static_rotary_pos_emb.to(hidden_states.device)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (
            emb.cos().to(dtype=hidden_states.dtype),
            emb.sin().to(dtype=hidden_states.dtype),
        )

        for blk in self.blocks:
            hidden_states = blk(
                hidden_states,
                position_embeddings=position_embeddings,
            )

        return self.merger(hidden_states)


class Qwen35VLVisionEncoder(torch.nn.Module):
    def __init__(self, config, img_resized_h: int = 448, img_resized_w: int = 448):
        super().__init__()
        patch_size = config.vision_config.patch_size
        self.image_grid_thw = torch.tensor(
            [[1, img_resized_h // patch_size, img_resized_w // patch_size]],
            dtype=torch.int64,
        )
        self.visual = StaticQwen35VLVisionModel._from_config(
            config.vision_config,
            grid_thw=self.image_grid_thw,
        )
        self.config = config
        self.img_resized_h = img_resized_h
        self.img_resized_w = img_resized_w

        temporal_patch_size = config.vision_config.temporal_patch_size
        self.image_seq_len = int(
            self.image_grid_thw.prod(-1).item()
            // (config.vision_config.spatial_merge_size**2)
        )
        self.input_seq_len = int(self.image_grid_thw.prod(-1).item())
        self.input_feature_dim = (
            config.vision_config.in_channels
            * temporal_patch_size
            * patch_size
            * patch_size
        )

    def preprocess(self, pixel_values: Tuple[torch.FloatTensor]) -> Tuple[torch.Tensor]:
        return pixel_values

    def get_example_inputs(self):
        return (
            torch.randn(
                (1, self.input_seq_len, self.input_feature_dim),
                dtype=torch.float32,
            ),
        )

    def forward(self, pixel_values: torch.FloatTensor):
        if pixel_values.dim() == 3 and pixel_values.shape[0] == 1:
            pixel_values = pixel_values.squeeze(0)
        image_embeds = self.visual(
            pixel_values,
            grid_thw=self.image_grid_thw.to(pixel_values.device),
        )
        image_embeds = image_embeds[: self.image_seq_len]
        return image_embeds.unsqueeze(0)


class StaticQwen2_5VLVisionAttention(nn.Module):
    def __init__(
        self,
        config,
        full_seq_len: int,
        window_seq_len: int,
    ) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.full_seq_len = full_seq_len
        self.window_seq_len = window_seq_len
        self.q_proj = nn.Linear(self.dim, self.dim, bias=True)
        self.k_proj = nn.Linear(self.dim, self.dim, bias=True)
        self.v_proj = nn.Linear(self.dim, self.dim, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim**-0.5
        self.apply_rope_emb = ROTARY_EMB_REGISTRY["default"]
        self.register_load_state_dict_pre_hook(self._split_qkv_state_dict)

    @staticmethod
    def _split_qkv_tensor(qkv_tensor: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if qkv_tensor.shape[0] % 3 != 0:
            raise ValueError(
                f"Expected fused QKV tensor's leading dimension to be divisible by 3, got {tuple(qkv_tensor.shape)}"
            )
        return torch.chunk(qkv_tensor, 3, dim=0)

    def _split_qkv_state_dict(
        self,
        module,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        del module, local_metadata, strict, missing_keys, unexpected_keys, error_msgs

        qkv_weight_key = f"{prefix}qkv.weight"
        if qkv_weight_key in state_dict:
            q_weight, k_weight, v_weight = self._split_qkv_tensor(
                state_dict.pop(qkv_weight_key)
            )
            state_dict[f"{prefix}q_proj.weight"] = q_weight
            state_dict[f"{prefix}k_proj.weight"] = k_weight
            state_dict[f"{prefix}v_proj.weight"] = v_weight

        qkv_bias_key = f"{prefix}qkv.bias"
        if qkv_bias_key in state_dict:
            q_bias, k_bias, v_bias = self._split_qkv_tensor(
                state_dict.pop(qkv_bias_key)
            )
            state_dict[f"{prefix}q_proj.bias"] = q_bias
            state_dict[f"{prefix}k_proj.bias"] = k_bias
            state_dict[f"{prefix}v_proj.bias"] = v_bias

    def _attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> torch.Tensor:
        attn_weights = torch.matmul(query_states, key_states) * self.scaling
        attn_weights = torch.softmax(attn_weights, dim=-1)
        return torch.matmul(attn_weights, value_states)

    def _window_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> torch.Tensor:
        # Run attention independently within each window chunk to match
        # HuggingFace's eager attention path with `cu_window_seqlens`.
        #
        # Shapes:
        #   query_states: [heads, seq, head_dim]
        #   key_states:   [heads, seq, head_dim]
        #   value_states: [heads, seq, head_dim]
        seq_len = query_states.shape[1]
        if self.window_seq_len <= 0 or seq_len % self.window_seq_len != 0:
            raise ValueError(
                f"Expected seq_len to be a positive multiple of window_seq_len, got seq_len={seq_len}, window_seq_len={self.window_seq_len}"
            )

        num_windows = seq_len // self.window_seq_len
        q = query_states.reshape(
            self.num_heads, num_windows, self.window_seq_len, self.head_dim
        ).permute(1, 0, 2, 3)
        k = key_states.reshape(
            self.num_heads, num_windows, self.window_seq_len, self.head_dim
        ).permute(1, 0, 2, 3)
        v = value_states.reshape(
            self.num_heads, num_windows, self.window_seq_len, self.head_dim
        ).permute(1, 0, 2, 3)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        return attn_output.permute(1, 0, 2, 3).reshape(
            self.num_heads, seq_len, self.head_dim
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        full_attention: bool,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states = self.q_proj(hidden_states).reshape(
            seq_length, self.num_heads, self.head_dim
        )
        key_states = self.k_proj(hidden_states).reshape(
            seq_length, self.num_heads, self.head_dim
        )
        value_states = self.v_proj(hidden_states).reshape(
            seq_length, self.num_heads, self.head_dim
        )
        cos, sin = position_embeddings
        query_states = query_states.transpose(0, 1).contiguous()
        key_states = key_states.transpose(0, 1).contiguous()
        value_states = value_states.transpose(0, 1).contiguous()

        query_states = self.apply_rope_emb(query_states.unsqueeze(0), cos, sin).squeeze(0)
        key_states = self.apply_rope_emb(key_states.unsqueeze(0), cos, sin).squeeze(0)

        if full_attention:
            attn_output = self._attention(
                query_states,
                key_states.transpose(-2, -1),
                value_states,
            )
        else:
            attn_output = self._window_attention(query_states, key_states, value_states)

        attn_output = attn_output.transpose(0, 1).reshape(seq_length, self.dim)

        attn_output = self.proj(attn_output).contiguous()
        return attn_output


class StaticQwen2_5VLVisionBlock(nn.Module):
    def __init__(
        self,
        config,
        full_seq_len: int,
        window_seq_len: int,
    ) -> None:
        super().__init__()
        self.norm1 = Qwen2RMSNorm(config.hidden_size, eps=1e-6)
        self.norm2 = Qwen2RMSNorm(config.hidden_size, eps=1e-6)
        self.attn = StaticQwen2_5VLVisionAttention(
            config,
            full_seq_len=full_seq_len,
            window_seq_len=window_seq_len,
        )
        self.mlp = Qwen2_5_VLMLP(config, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        full_attention: bool,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            position_embeddings=position_embeddings,
            full_attention=full_attention,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class StaticQwen2_5VisionTransformerPretrainedModel(
    Qwen2_5_VisionTransformerPretrainedModel
):
    def __init__(self, config, grid_thw: torch.Tensor):
        super().__init__(config)
        # Flattened patch-embed weights (Conv3d -> Linear equivalent).
        #
        # We keep the original Conv3d module so HF checkpoints can still load via
        # `strict=False`, but use a 2D Parameter in forward to avoid introducing
        # 5D tensors in the exported/quantized graph.
        proj = self.patch_embed.proj
        self.patch_embed_weight_flat = nn.Parameter(
            proj.weight.detach().flatten(1).clone()
        )

        def _sync_patch_embed_weight_flat(module, _incompatible_keys):
            with torch.no_grad():
                module.patch_embed_weight_flat.copy_(
                    module.patch_embed.proj.weight.flatten(1)
                )

        self.register_load_state_dict_post_hook(_sync_patch_embed_weight_flat)
        static_grid_thw = grid_thw.to(dtype=torch.int64)
        static_rotary_pos_emb = super().rot_pos_emb(static_grid_thw)
        static_window_index, static_cu_window_seqlens = super().get_window_index(
            static_grid_thw
        )
        static_cu_window_seqlens = torch.unique_consecutive(
            torch.tensor(static_cu_window_seqlens, dtype=torch.int32)
        )
        static_cu_seqlens = torch.repeat_interleave(
            static_grid_thw[:, 1] * static_grid_thw[:, 2],
            static_grid_thw[:, 0],
        ).cumsum(dim=0, dtype=torch.int32)
        static_cu_seqlens = F.pad(static_cu_seqlens, (1, 0), value=0)
        static_reverse_indices = torch.argsort(static_window_index)
        window_seq_len = int(static_cu_window_seqlens[1].item())
        full_seq_len = int(static_cu_seqlens[-1].item())

        self.blocks = nn.ModuleList(
            [
                StaticQwen2_5VLVisionBlock(
                    config,
                    full_seq_len=full_seq_len,
                    window_seq_len=window_seq_len,
                )
                for _ in range(config.depth)
            ]
        )

        self.register_buffer("static_grid_thw", static_grid_thw, persistent=False)
        self.register_buffer(
            "static_rotary_pos_emb", static_rotary_pos_emb, persistent=False
        )
        self.register_buffer("static_window_index", static_window_index, persistent=False)
        self.register_buffer(
            "static_cu_window_seqlens", static_cu_window_seqlens, persistent=False
        )
        self.register_buffer("static_cu_seqlens", static_cu_seqlens, persistent=False)
        self.register_buffer(
            "static_reverse_indices", static_reverse_indices, persistent=False
        )

    def forward(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor = None, **kwargs
    ) -> torch.Tensor:
        # Qwen2.5-VL patch embedding is implemented as a Conv3d over a single
        # patch volume after reshaping `hidden_states` from [S, C*T*P*P] to
        # [S, C, T, P, P]. In the AOT+PTQ flow, mixing this 5D conv3d with the
        # rest of the vision transformer can trigger QNN internal `*InputSlicePad`
        # failures. Since kernel == stride and the output spatial/temporal dims
        # are 1, this operation is equivalent to a Linear on the flattened patch.
        target_dtype = self.patch_embed_weight_flat.dtype
        hidden_states = F.linear(
            hidden_states.to(dtype=target_dtype),
            self.patch_embed_weight_flat,
            bias=None,
        )
        rotary_pos_emb = self.static_rotary_pos_emb.to(hidden_states.device)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )
        window_index = self.static_window_index.to(hidden_states.device)
        hidden_states = hidden_states[window_index, :, :].reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )
        rotary_pos_emb = rotary_pos_emb[window_index, :, :].reshape(seq_len, -1)
        # Keep position embedding dtype aligned with the model's hidden_states
        # to avoid implicit type promotion (e.g. bf16 weights + fp32 cos/sin
        # -> fp32 activations) which can later break matmul/linear ops.
        position_embeddings = (
            rotary_pos_emb.cos().to(dtype=hidden_states.dtype),
            rotary_pos_emb.sin().to(dtype=hidden_states.dtype),
        )

        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                position_embeddings=position_embeddings,
                full_attention=layer_num in self.fullatt_block_indexes,
            )

        hidden_states = self.merger(hidden_states)
        reverse_indices = self.static_reverse_indices.to(hidden_states.device)
        hidden_states = hidden_states[reverse_indices, :]

        return hidden_states


class Qwen2_5VLVisionEncoder(torch.nn.Module):
    def __init__(self, config, img_resized_h: int = 448, img_resized_w: int = 448):
        super().__init__()
        patch_size = config.vision_config.patch_size
        self.image_grid_thw = torch.tensor(
            [[1, img_resized_h // patch_size, img_resized_w // patch_size]],
            dtype=torch.int64,
        )
        self.visual = StaticQwen2_5VisionTransformerPretrainedModel._from_config(
            config.vision_config,
            grid_thw=self.image_grid_thw,
        )
        self.config = config
        self.img_resized_h = img_resized_h
        self.img_resized_w = img_resized_w

        temporal_patch_size = config.vision_config.temporal_patch_size
        self.image_seq_len = int(
            self.image_grid_thw.prod(-1).item()
            // (config.vision_config.spatial_merge_size**2)
        )
        self.input_seq_len = int(self.image_grid_thw.prod(-1).item())
        self.input_feature_dim = (
            config.vision_config.in_channels
            * temporal_patch_size
            * patch_size
            * patch_size
        )

    def preprocess(self, pixel_values: Tuple[torch.FloatTensor]) -> Tuple[torch.Tensor]:
        if len(pixel_values) != 1:
            return pixel_values

        patches = pixel_values[0]
        if patches.dim() == 3 and patches.shape[0] == 1:
            patches = patches.squeeze(0)
        return (patches,)

    def get_example_inputs(self):
        return (
            torch.randn(
                (self.input_seq_len, self.input_feature_dim),
                dtype=torch.float32,
            ),
        )

    def forward(self, pixel_values: torch.FloatTensor):
        image_embeds = self.visual(
            pixel_values,
            grid_thw=self.image_grid_thw.to(pixel_values.device),
        )
        image_embeds = image_embeds[: self.image_seq_len]
        return image_embeds.unsqueeze(0)


class StaticQwen3VLVisionAttention(nn.Module):
    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim**-0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states)
            .reshape(seq_length, 3, self.num_heads, self.head_dim)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = apply_qwen3_vl_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        query_states = query_states.transpose(0, 1).contiguous()
        key_states = key_states.transpose(0, 1).contiguous()
        value_states = value_states.transpose(0, 1).contiguous()

        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1))
        attn_weights = attn_weights * self.scaling
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(0, 1).reshape(seq_length, self.dim)

        return self.proj(attn_output).contiguous()


class StaticQwen3VLVisionBlock(nn.Module):
    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = StaticQwen3VLVisionAttention(config)
        self.mlp = Qwen3VLVisionMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            position_embeddings=position_embeddings,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class StaticQwen3VLVisionModel(Qwen3VLVisionModel):
    def __init__(self, config: Qwen3VLVisionConfig, grid_thw: torch.Tensor):
        super().__init__(config)
        proj = self.patch_embed.proj
        self.patch_embed_weight_flat = nn.Parameter(
            proj.weight.detach().flatten(1).clone()
        )
        self.patch_embed_bias_flat = nn.Parameter(proj.bias.detach().clone())

        def _sync_patch_embed_params(module, _incompatible_keys):
            with torch.no_grad():
                module.patch_embed_weight_flat.copy_(
                    module.patch_embed.proj.weight.flatten(1)
                )
                module.patch_embed_bias_flat.copy_(module.patch_embed.proj.bias)

        self.register_load_state_dict_post_hook(_sync_patch_embed_params)
        static_grid_thw = grid_thw.to(dtype=torch.int64)
        static_pos_embeds = super().fast_pos_embed_interpolate(static_grid_thw)
        static_pos_embeds = static_pos_embeds.detach().to(torch.float32)
        static_rotary_pos_emb = super().rot_pos_emb(static_grid_thw).detach()

        self.blocks = nn.ModuleList(
            [StaticQwen3VLVisionBlock(config) for _ in range(config.depth)]
        )
        self.register_buffer("static_grid_thw", static_grid_thw, persistent=False)
        self.register_buffer("static_pos_embeds", static_pos_embeds, persistent=False)
        self.register_buffer(
            "static_rotary_pos_emb", static_rotary_pos_emb, persistent=False
        )

    def forward(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor = None, **kwargs
    ) -> BaseModelOutputWithDeepstackFeatures:
        del grid_thw, kwargs
        target_dtype = self.patch_embed_weight_flat.dtype
        hidden_states = F.linear(
            hidden_states.to(dtype=target_dtype),
            self.patch_embed_weight_flat,
            self.patch_embed_bias_flat,
        )
        hidden_states = hidden_states + self.static_pos_embeds.to(
            device=hidden_states.device, dtype=hidden_states.dtype
        )

        seq_len, _ = hidden_states.size()
        rotary_pos_emb = self.static_rotary_pos_emb.to(hidden_states.device)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (
            emb.cos().to(dtype=hidden_states.dtype),
            emb.sin().to(dtype=hidden_states.dtype),
        )

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                position_embeddings=position_embeddings,
            )
            if layer_num in self.deepstack_visual_indexes:
                deepstack_feature = self.deepstack_merger_list[
                    self.deepstack_visual_indexes.index(layer_num)
                ](hidden_states)
                deepstack_feature_lists.append(deepstack_feature)

        merged_hidden_states = self.merger(hidden_states)
        return BaseModelOutputWithDeepstackFeatures(
            last_hidden_state=hidden_states,
            pooler_output=merged_hidden_states,
            deepstack_features=deepstack_feature_lists,
        )


class Qwen3VLVisionEncoder(torch.nn.Module):
    def __init__(self, config, img_resized_h: int = 448, img_resized_w: int = 448):
        super().__init__()
        patch_size = config.vision_config.patch_size
        self.image_grid_thw = torch.tensor(
            [[1, img_resized_h // patch_size, img_resized_w // patch_size]],
            dtype=torch.int64,
        )
        self.visual = StaticQwen3VLVisionModel._from_config(
            config.vision_config,
            grid_thw=self.image_grid_thw,
        )
        self.config = config
        self.img_resized_h = img_resized_h
        self.img_resized_w = img_resized_w

        temporal_patch_size = config.vision_config.temporal_patch_size
        self.image_seq_len = int(
            self.image_grid_thw.prod(-1).item()
            // (config.vision_config.spatial_merge_size**2)
        )
        self.input_seq_len = int(self.image_grid_thw.prod(-1).item())
        self.input_feature_dim = (
            config.vision_config.in_channels
            * temporal_patch_size
            * patch_size
            * patch_size
        )

    def preprocess(self, pixel_values: Tuple[torch.FloatTensor]) -> Tuple[torch.Tensor]:
        if len(pixel_values) != 1:
            return pixel_values

        patches = pixel_values[0]
        if patches.dim() == 3 and patches.shape[0] == 1:
            patches = patches.squeeze(0)
        return (patches,)

    def get_example_inputs(self):
        return (
            torch.randn(
                (self.input_seq_len, self.input_feature_dim),
                dtype=torch.float32,
            ),
        )

    def forward(self, pixel_values: torch.FloatTensor):
        image_outputs = self.visual(
            pixel_values,
            grid_thw=self.image_grid_thw.to(pixel_values.device),
        )
        image_embeds = image_outputs.pooler_output[: self.image_seq_len]
        return image_embeds.unsqueeze(0)
