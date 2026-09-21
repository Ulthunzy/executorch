# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import copy
import gc
import inspect
import json
import logging
import math
import os
import re
import time
import types

from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import torch

from executorch.backends.qualcomm._passes import FoldQDQ, I64toI32, TagQuantIO
from executorch.backends.qualcomm._passes.build_quant_io import BuildQuantIo
from executorch.backends.qualcomm._passes.qnn_pass_manager import (
    get_qnn_pass_manager_cls,
)
from executorch.backends.qualcomm.builders.utils import is_graph_output
from executorch.backends.qualcomm.export_utils import make_quantizer
from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype

from executorch.backends.qualcomm.utils.constants import (
    QCOM_PASS_ACTIVATE_KEY,
    QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY,
)
from executorch.backends.qualcomm.utils.utils import (
    convert_linear_to_conv2d,
    to_edge_transform_and_lower_to_qnn,
    update_spill_fill_size,
)
from executorch.devtools.backend_debug import print_delegation_info
from executorch.examples.models.llama.hf_download import (
    download_and_convert_hf_checkpoint,
)
from executorch.examples.models.llama.source_transformation.quantize import (
    get_quant_embedding_transform,
)
from executorch.examples.qualcomm.oss_scripts.llama import (
    LLM_VARIANT_ARCHS,
    LLMModelConfig,
)
from executorch.examples.qualcomm.oss_scripts.llama.dataset import (
    DataConfig,
    DatasetBuilder,
    ModalityEncoderDataset,
)
from executorch.examples.qualcomm.oss_scripts.llama.decoder_constants import (
    AUDIO_ENCODER,
    DECODE_QDQ_FILENAME,
    DECODER_GRAPH_NAMES,
    TEXT_DECODER,
    TEXT_ENCODER,
    TOK_EMBEDDING,
    TOK_EMBEDDING_GRAPH_NAMES,
    VISION_ENCODER,
)
from executorch.examples.qualcomm.oss_scripts.llama.encoder.encoder_config import (
    GraniteSpeechEncoder,
)
from executorch.examples.qualcomm.oss_scripts.llama.encoder.encoder_quant_recipe import (
    EncoderQuantRecipe,
)
from executorch.examples.qualcomm.oss_scripts.llama.evaluator.lm_eval_adapter import (
    run_lm_eval,
)
from executorch.examples.qualcomm.oss_scripts.llama.inference import (
    DecoderInference,
    EncoderInference,
)
from executorch.examples.qualcomm.oss_scripts.llama.mix_precision_analyzer import (
    PerLayerSqnrAnalyzer,
    save_suggest_recipes,
)
from executorch.examples.qualcomm.oss_scripts.llama.model.embedding import (
    TokenEmbedding,
)
from executorch.examples.qualcomm.oss_scripts.llama.model.static_llama import (
    LlamaModel,
    ModelArgs,
    MultiScopeAwareLlamaModel,
    MultiScopeAwareLlamaModel_swa,
    MultiScopeAwareLlamaModelWithoutEmbedding,
    MultiScopeAwareLlamaModelWithoutEmbedding_swa,
    Qwen35ModelWithoutEmbedding,
    Qwen35ModelWithoutEmbedding_swa,
)
from executorch.examples.qualcomm.oss_scripts.llama.quantize import (
    PTQStrategy,
    QATStrategy,
)
from executorch.examples.qualcomm.oss_scripts.llama.static_llm_quant_recipe import (
    StaticLLMQuantRecipe,
)
from executorch.examples.qualcomm.oss_scripts.llama.tokenizer import TokenizerWrapper
from executorch.examples.qualcomm.oss_scripts.llama.train.config import TrainingArgs
from executorch.examples.qualcomm.oss_scripts.llama.wrappers.base_component import (
    Component,
    get_model_specific_kwargs,
    log_info,
    Mode,
    process_model_args,
    Processor,
    Request,
)
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
from executorch.extension.llm.custom_ops import model_sharding
from executorch.extension.llm.export.builder import DType
from torch.utils.data import DataLoader
from torchao.quantization.pt2e import MinMaxObserver, move_exported_model_to_train
from torchao.quantization.pt2e.quantize_pt2e import (
    convert_pt2e,
    prepare_pt2e,
    prepare_qat_pt2e,
)
from transformers import AutoConfig, AutoModel, AutoModelForSpeechSeq2Seq


def _release_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


_HF_VLM_MODEL_CACHE: Dict[str, Any] = {}


def _is_attention_mask_placeholder(node, expected_shapes):
    names = (str(node.name), str(node.target))
    if not any("atten_mask" in name or "attention_mask" in name for name in names):
        return False

    val = node.meta.get("val")
    return val is not None and tuple(val.size()) in expected_shapes


def _get_cached_qwen2_5_vl_for_conditional_generation(repo_id: str):
    model = _HF_VLM_MODEL_CACHE.get(repo_id)
    if model is not None:
        return model

    from transformers import Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        repo_id, _attn_implementation="eager"
    ).eval()
    _HF_VLM_MODEL_CACHE[repo_id] = model
    return model


def _load_local_model_config(
    params_path: Optional[str],
) -> Optional[Dict[str, Any]]:
    if params_path is None:
        return None
    with open(params_path) as f:
        return json.load(f)


def _hf_config_path(*candidates: Optional[str]) -> Optional[str]:
    """Return the HF config.json that ships with the weights, if any."""
    for candidate in candidates:
        if not candidate:
            continue
        path = os.path.join(os.path.expanduser(candidate), "config.json")
        if os.path.isfile(path):
            return path
    return None


def _full_attention_layers(layer_types: Optional[List[str]]) -> Optional[List[int]]:
    if not layer_types:
        return None
    return [index for index, kind in enumerate(layer_types) if kind != "sliding_attention"]


# HF config 字段 -> params/ModelArgs 字段。只映射语义完全一致的键。
_HF_MODEL_ARG_FIELDS: Dict[str, str] = {
    "num_hidden_layers": "n_layers",
    "hidden_size": "dim",
    "intermediate_size": "hidden_dim",
    "num_attention_heads": "n_heads",
    "num_key_value_heads": "n_kv_heads",
    "head_dim": "head_dim",
    "vocab_size": "vocab_size",
    "rms_norm_eps": "norm_eps",
    "rope_theta": "rope_theta",
    "sliding_window": "sliding_window",
    "layer_types": "layer_types",
}


def _same_value(current: Any, expected: Any) -> bool:
    if isinstance(current, (int, float)) and isinstance(expected, (int, float)):
        return math.isclose(float(current), float(expected), rel_tol=1e-9, abs_tol=1e-12)
    return current == expected


def _sync_model_args_from_hf_config(
    model_args: Dict[str, Any],
    *,
    sources: Tuple[Optional[str], ...],
) -> Dict[str, Any]:
    """Reconcile params with the weights' own HF config, HF wins on mismatch.

    The params JSON (--params or the registry template) is a snapshot of one
    weight version and can lag behind the checkpoint -- e.g. layer_types of a
    new release still carrying the previous sliding/full layout. Architecture
    keys that the HF config defines follow the weights, so mismatch is reported
    and the HF value wins. Params stay authoritative for keys HF does not carry.
    """
    path = _hf_config_path(*sources)
    if path is None:
        logging.warning(
            "No HF config.json found next to the weights (%s); model args stay as "
            "declared by the params file. Pass --source_safetensors <weights dir> "
            "to sync them with the checkpoint.",
            ", ".join(str(source) for source in sources if source) or "<none>",
        )
        return model_args

    with open(path) as f:
        hf_config = json.load(f)
    mismatched, synced = [], []
    for hf_key, arg_key in _HF_MODEL_ARG_FIELDS.items():
        expected = hf_config.get(hf_key)
        if expected is None or arg_key not in model_args:
            continue
        expected = list(expected) if isinstance(expected, list) else expected
        current = model_args[arg_key]
        if _same_value(current, expected):
            continue
        mismatched.append((hf_key, current, expected))
        model_args[arg_key] = expected
        synced.append(arg_key)

    layer_types = model_args.get("layer_types")
    if layer_types is not None and model_args.get("n_layers") not in (None, len(layer_types)):
        raise ValueError(
            f"layer_types in {path} has {len(layer_types)} entries but "
            f"n_layers is {model_args['n_layers']}"
        )

    if mismatched:
        for hf_key, current, expected in mismatched:
            if hf_key == "layer_types":
                current = _full_attention_layers(current)
                expected = _full_attention_layers(expected)
            logging.warning(
                "params declares %s=%s but %s declares %s; using the HF config value.",
                _HF_MODEL_ARG_FIELDS[hf_key],
                current,
                path,
                expected,
            )
        logging.info(
            "Synced %d model arg(s) from %s: %s.",
            len(synced),
            path,
            ", ".join(synced),
        )
    else:
        logging.info("Model args match %s.", path)
    return model_args


def _is_qwen35_decoder_model(decoder_model: str) -> bool:
    return decoder_model.startswith(("qwen3_5", "lincal3_5"))


def _is_lincal35_decoder_model(decoder_model: str) -> bool:
    return decoder_model.startswith("lincal3_5")


def _load_safetensor_tensors(repo_id: str, keys: List[str]) -> Dict[str, torch.Tensor]:
    from safetensors import safe_open

    repo_path = os.path.expanduser(repo_id)
    index_path = os.path.join(repo_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        missing = [key for key in keys if key not in weight_map]
        if missing:
            raise KeyError(f"Missing safetensors keys in {repo_id}: {missing}")

        shard_to_keys: Dict[str, List[str]] = {}
        for key in keys:
            shard_to_keys.setdefault(weight_map[key], []).append(key)

        tensors = {}
        for shard, shard_keys in shard_to_keys.items():
            with safe_open(
                os.path.join(repo_path, shard), framework="pt", device="cpu"
            ) as f:
                for key in shard_keys:
                    tensors[key] = f.get_tensor(key)
        return tensors

    model_path = os.path.join(repo_path, "model.safetensors")
    if os.path.exists(model_path):
        with safe_open(model_path, framework="pt", device="cpu") as f:
            available = set(f.keys())
            missing = [key for key in keys if key not in available]
            if missing:
                raise KeyError(f"Missing safetensors keys in {model_path}: {missing}")
            return {key: f.get_tensor(key) for key in keys}

    raise FileNotFoundError(f"Could not find safetensors checkpoint in {repo_id}")


def _load_safetensor_state_dict_by_prefix(
    repo_id: str, prefix: str, strip_prefix: str = ""
) -> Dict[str, torch.Tensor]:
    repo_path = os.path.expanduser(repo_id)
    index_path = os.path.join(repo_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            keys = [key for key in json.load(f)["weight_map"] if key.startswith(prefix)]
    else:
        from safetensors import safe_open

        model_path = os.path.join(repo_path, "model.safetensors")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Could not find safetensors checkpoint in {repo_id}"
            )
        with safe_open(model_path, framework="pt", device="cpu") as f:
            keys = [key for key in f.keys() if key.startswith(prefix)]

    tensors = _load_safetensor_tensors(repo_id, keys)
    return {
        key.removeprefix(strip_prefix): value.to(torch.float32)
        for key, value in tensors.items()
    }


def _load_lincal35_input_embedding(repo_id: str) -> torch.nn.Embedding:
    weight = _load_safetensor_tensors(
        repo_id, ["model.language_model.embed_tokens.weight"]
    )["model.language_model.embed_tokens.weight"].to(torch.float32)
    return torch.nn.Embedding.from_pretrained(weight, freeze=True)


def _flatten_model_args_for_static_llama(
    model_config: Dict[str, Any]
) -> Dict[str, Any]:
    if "text_config" not in model_config:
        flattened = dict(model_config)
        flattened.pop("vision_config", None)
        return flattened

    text_config = model_config["text_config"]
    text_model_type = text_config.get("model_type")
    is_qwen35_text = text_model_type == "qwen3_5_text"
    rope_parameters = text_config.get("rope_parameters") or {}
    rope_theta = text_config.get(
        "rope_theta", rope_parameters.get("rope_theta", model_config.get("rope_theta"))
    )
    flattened = {
        "dim": text_config["hidden_size"],
        "n_layers": text_config["num_hidden_layers"],
        "n_heads": text_config["num_attention_heads"],
        "n_kv_heads": text_config.get(
            "num_key_value_heads", text_config["num_attention_heads"]
        ),
        "vocab_size": text_config["vocab_size"],
        "hidden_dim": text_config["intermediate_size"],
        "head_dim": text_config.get(
            "head_dim", text_config["hidden_size"] // text_config["num_attention_heads"]
        ),
        "norm_eps": text_config.get("rms_norm_eps", 1e-5),
        "rope_theta": rope_theta,
        "attention_qkv_bias": text_config.get(
            "attention_qkv_bias",
            text_config.get(
                "attention_bias", model_config.get("attention_qkv_bias", True)
            ),
        ),
        "use_qk_norm": text_config.get(
            "use_qk_norm", model_config.get("use_qk_norm", is_qwen35_text)
        ),
        "qk_norm_before_rope": text_config.get(
            "qk_norm_before_rope",
            model_config.get("qk_norm_before_rope", is_qwen35_text),
        ),
        "use_hf_rope": True,
        "use_scaled_rope": False,
        "bos_idx": text_config.get("bos_token_id", model_config.get("bos_token_id", 1)),
        "eos_idx": text_config.get("eos_token_id", model_config.get("eos_token_id", 3)),
    }

    layer_types = text_config.get("layer_types") or model_config.get("layer_types")
    if layer_types is not None:
        flattened["layer_types"] = layer_types

    partial_rotary_factor = text_config.get(
        "partial_rotary_factor",
        rope_parameters.get(
            "partial_rotary_factor", model_config.get("partial_rotary_factor")
        ),
    )
    if partial_rotary_factor is not None:
        flattened["partial_rotary_factor"] = partial_rotary_factor

    mrope_section = text_config.get(
        "mrope_section",
        rope_parameters.get("mrope_section", model_config.get("mrope_section")),
    )
    if mrope_section is not None:
        flattened["mrope_section"] = mrope_section

    for key in (
        "rms_norm_add_unit_offset",
        "linear_conv_kernel_dim",
        "linear_key_head_dim",
        "linear_value_head_dim",
        "linear_num_key_heads",
        "linear_num_value_heads",
    ):
        if key in text_config:
            flattened[key] = text_config[key]
        elif key in model_config:
            flattened[key] = model_config[key]

    if is_qwen35_text:
        flattened["rms_norm_add_unit_offset"] = text_config.get(
            "rms_norm_add_unit_offset",
            model_config.get("rms_norm_add_unit_offset", True),
        )
        flattened["attention_type"] = text_config.get(
            "attention_type", model_config.get("attention_type", "mha")
        )
        flattened["use_q_gate"] = text_config.get(
            "attn_output_gate", model_config.get("use_q_gate", True)
        )

    use_sliding_window = bool(
        text_config.get("use_sliding_window")
        or model_config.get("use_sliding_window")
        or text_config.get("sliding_window")
        or (layer_types is not None and "sliding_attention" in layer_types)
    )
    if use_sliding_window:
        flattened["sliding_window"] = text_config.get(
            "sliding_window", model_config.get("sliding_window")
        )
        flattened["local_rope_theta"] = text_config.get(
            "local_rope_theta",
            model_config.get("local_rope_theta", rope_theta),
        )

    return flattened


def _overlay_local_vision_config(
    auto_model, local_config: Optional[Dict[str, Any]]
) -> None:
    if local_config is None:
        return

    vision_config = local_config.get("vision_config")
    if (
        vision_config is None
        or getattr(auto_model.config, "vision_config", None) is None
    ):
        return

    for key, value in vision_config.items():
        setattr(auto_model.config.vision_config, key, value)

    depth = getattr(auto_model.config.vision_config, "depth", None)
    fullatt_block_indexes = getattr(
        auto_model.config.vision_config, "fullatt_block_indexes", None
    )
    if depth is not None and fullatt_block_indexes is not None:
        auto_model.config.vision_config.fullatt_block_indexes = [
            idx for idx in fullatt_block_indexes if idx < depth
        ]


def _get_vision_num_layers(auto_model) -> int:
    vision_config = auto_model.config.vision_config
    num_layers = getattr(vision_config, "num_hidden_layers", None)
    if num_layers is not None:
        return num_layers
    return getattr(vision_config, "depth")


def is_node_src_start_with_name(node: torch.fx.Node, kv_cache_prefix: str) -> bool:
    """
    Return True if any NodeSource in node.meta['from_node'] has a name
    starting with `kv_cache_prefix`. Used to identify K/V cache nodes by their
    "k_" or "v_" name prefix in the traced graph.
    """

    def has_source_name_prefix(
        node_src: torch.fx.traceback.NodeSource, kv_cache_prefix: str
    ) -> bool:

        name = getattr(node_src, "name", None)
        if isinstance(name, str) and name.startswith(kv_cache_prefix):
            return True

        children = getattr(node_src, "from_node", None)
        if not children:
            return False

        for src in children:
            if has_source_name_prefix(src, kv_cache_prefix):
                return True

        return False

    node_srcs = node.meta.get("from_node", None)
    if not node_srcs:
        return False

    return any(
        has_source_name_prefix(node_src, kv_cache_prefix) for node_src in node_srcs
    )


class TextDecoder(Component):

    def __init__(
        self,
        control_args: argparse.Namespace,
        config: LLMModelConfig,
        mode: Mode,
        apply_embedding: bool = False,
    ):
        self.control_args = control_args
        self.config = config
        self.mode = mode
        self.pass_manager_cls = get_qnn_pass_manager_cls()
        self.passes_job = self.pass_manager_cls.get_capture_program_passes()
        self.dep_table = (
            self.pass_manager_cls.get_passes_dependency_for_capture_program()
        )
        self.meta = {}
        recipe_cls = (
            self.config.qat_recipe
            if control_args.qat and self.config.qat_recipe
            else self.config.quant_recipe
        )
        self.quant_recipe: StaticLLMQuantRecipe = (
            recipe_cls(mode == Mode.CALIBRATE) if recipe_cls else None
        )

        # For multimodal embedding
        self.apply_embedding = apply_embedding
        self.tok_embedding_passes_job = (
            self.pass_manager_cls.get_capture_program_passes()
            if apply_embedding
            else None
        )
        self.tok_embedding_dep_table = (
            self.pass_manager_cls.get_passes_dependency_for_capture_program()
            if apply_embedding
            else None
        )

        # load static llama model args
        if control_args.decoder_model == "gemma4-e2b":
            from executorch.examples.models.gemma4.text_decoder.gemma4_config import (
                Gemma4Config,
            )

            params_path = config.params_path
            self.gemma4_config = Gemma4Config.from_json(params_path)
            self.gemma4_config.use_kv_cache = True
            self.gemma4_config.max_batch_size = 1
            self.gemma4_config.max_seq_len = control_args.max_seq_len
            self.gemma4_config.max_context_len = control_args.max_context_len
            self.model_args = None
        else:
            params_path = (
                config.params_path
                if control_args.params is None
                else control_args.params
            )
            with open(params_path) as f:
                model_args_dict = _flatten_model_args_for_static_llama(json.load(f))
                model_args_dict = _sync_model_args_from_hf_config(
                    model_args_dict,
                    # --checkpoint 的权重不一定来自 repo_id，只有直接下载时才信任它。
                    sources=(getattr(control_args, "source_safetensors", None),)
                    + ((config.repo_id,) if control_args.checkpoint is None else ()),
                )
                self.model_args = process_model_args(
                    control_args,
                    ModelArgs(**model_args_dict),
                    self.quant_recipe,
                    config,
                    mode,
                )
        # prepare instance
        self.tok_embedding, self.decoder = self._prepare_model()

        # check if sharding required
        if self.decoder and self.config.num_sharding > 1:
            layer_prefix_offsets = None
            if self.control_args.decoder_model == "gemma4-e2b":
                n_self = self.gemma4_config.num_self_decoder_layers
                layer_prefix_offsets = {
                    "model.self_decoder.layers": 0,
                    "model.cross_decoder.layers": n_self,
                }
            SplitGraph, setting = model_sharding.get_split_graph_pass(
                self.meta["get_n_layers"],
                shares=self.config.num_sharding,
                pattern=self._get_sharding_get_pattern(),
                layer_prefix_offsets=layer_prefix_offsets,
            )
            self.passes_job[SplitGraph] = setting
            self.dep_table[SplitGraph] = [FoldQDQ]
            self.dep_table[TagQuantIO] = [SplitGraph]

        self._decoder_inference = (
            DecoderInference(
                get_example_inputs=self.get_example_inputs,
                audio_token_id=self.meta.get("audio_token_id", None),
                image_token_id=self.meta.get("image_token_id", None),
                max_context_len=self.meta["get_max_context_len"],
                max_batch_size=self.meta["get_max_batch_size"],
                use_i64_token=self.control_args.embedding_quantize is not None,
            )
            if self.decoder is not None
            else None
        )

    def _get_sharding_get_pattern(self):
        if self.control_args.decoder_model == "gemma4-e2b":
            prefixes = [
                "model.cross_decoder.layers",
                "model.self_decoder.layers",
            ]
        else:
            prefixes = [
                "layers",
            ]
        prefix_alt = "|".join(re.escape(p) for p in prefixes)
        return rf"^(?:{prefix_alt})\.(\d+)"

    def _prepare_model(self):  # noqa: C901
        if (instance := self._get_model_instance()) is None:
            return None, None
        tok_embedding, decoder = instance
        # load parameters for HF models
        if self.control_args.checkpoint is None:
            if self.control_args.decoder_model == "gemma4-e2b":
                from executorch.examples.qualcomm.oss_scripts.gemma4.text_decoder.convert_weights import (
                    remap_keys,
                )
                from huggingface_hub import snapshot_download

                state_dict = self.config.convert_weights.__func__(
                    snapshot_download(repo_id=self.config.repo_id),
                    self.gemma4_config,
                    torch.float32,
                )
                state_dict = remap_keys(state_dict)
            else:
                checkpoint = download_and_convert_hf_checkpoint(
                    self.config.repo_id,
                    self.config.convert_weights.__func__,
                )
                state_dict = torch.load(
                    checkpoint, weights_only=True, map_location="cpu", mmap=True
                )
                if self.control_args.decoder_model in {
                    "gemma-2b",
                    "gemma2-2b",
                    "gemma3-1b",
                }:
                    for k, v in state_dict.items():
                        if "norm" not in k:
                            continue
                        # Llama does x.to(float16) * w whilst Gemma3 is (x * w).to(float16)
                        # See https://github.com/huggingface/transformers/pull/29402
                        state_dict[k] = v.float() + torch.ones(
                            v.shape, dtype=torch.float32
                        )
                for k, v in state_dict.items():
                    if "tok_embeddings.weight" == k:
                        state_dict[k] = (
                            v.float() * self.model_args.embedding_scale_factor
                        )
        else:
            state_dict = torch.load(
                self.control_args.checkpoint,
                weights_only=True,
                map_location="cpu",
                mmap=True,
            )
            if "model" in state_dict:
                state_dict = state_dict["model"]

            if self.control_args.decoder_model == "stories260k":
                state_dict = {
                    k.replace("_orig_mod.", ""): v for k, v in state_dict.items()
                }

        # change to HF weight to improve the performance of RoPE in HTP backend.
        if self.config.transform_weight:

            def permute(w, heads, partial_rotary_dim):
                dim_0 = w.size(0)
                dim_1 = w.size(1)
                transformed_weight = (
                    w.view(
                        heads, -1, dim_0 // heads // 2 // partial_rotary_dim, 2, dim_1
                    )
                    .transpose(2, 3)
                    .reshape(dim_0, dim_1)
                )
                return transformed_weight

            # TODO: handle cases where input size isn't divisible.
            partial_rotary_dim = int(1 // self.model_args.partial_rotary_factor)
            for layer_i in range(decoder.n_layers):
                state_dict[f"layers.{layer_i}.attention.wq.weight"] = permute(
                    state_dict[f"layers.{layer_i}.attention.wq.weight"],
                    decoder.n_heads,
                    partial_rotary_dim,
                )
                state_dict[f"layers.{layer_i}.attention.wk.weight"] = permute(
                    state_dict[f"layers.{layer_i}.attention.wk.weight"],
                    decoder.n_kv_heads,
                    partial_rotary_dim,
                )

        decoder_state_keys = set(decoder.state_dict().keys())
        if set(state_dict.keys()) - decoder_state_keys:
            state_dict = {
                key: value
                for key, value in state_dict.items()
                if key in decoder_state_keys
            }
        decoder.load_state_dict(state_dict, strict=True, assign=True)

        if any([self.config.r1, self.config.r2]):
            raise RuntimeError(
                "SpinQuant (r1/r2) is no longer supported: the "
                "torchao.prototype.spinquant module has been deleted."
            )

        # perform model transformation
        for layer in decoder.layers:
            if getattr(layer.attention, "prepare_fused_linear_in_proj", None):
                layer.attention.prepare_fused_linear_in_proj()
            if getattr(layer.attention, "prepare_attention_conv", None):
                layer.attention.prepare_attention_conv()
            if getattr(layer.feed_forward, "prepare_feedforward_conv", None):
                layer.feed_forward.prepare_feedforward_conv()

        decoder = convert_linear_to_conv2d(decoder)

        # check dtype override
        if self.control_args.dtype_override is not None:
            dtype_override = DType[self.control_args.dtype_override]
            decoder = decoder.to(dtype_override.to_torch_dtype())

        # check embedding fallback
        if self.control_args.embedding_quantize:
            self.passes_job[I64toI32][QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY][
                "skip_node"
            ] = {"tokens"}
            if self.apply_embedding:
                self.tok_embedding_passes_job[I64toI32][
                    QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY
                ]["skip_node"] = {"tokens"}

        if tok_embedding is not None:
            tok_embedding = tok_embedding.eval()

        return tok_embedding, decoder.eval()

    def _get_model_instance(self) -> LlamaModel:
        if self.mode == Mode.PREFILL and self.control_args.model_mode == "kv":
            return None
        use_i64_token = self.control_args.embedding_quantize is not None

        # get embedding model
        tok_embedding = None
        if self.apply_embedding:
            if hasattr(self.config, AUDIO_ENCODER):
                auto_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    self.config.repo_id, _attn_implementation="eager"
                )
            elif self.control_args.decoder_model in {
                "qwen2_5_vl-3b_instruct",
                "lincal_vl-3b_instruct",
            }:
                auto_model = _get_cached_qwen2_5_vl_for_conditional_generation(
                    self.config.repo_id
                )
            elif _is_lincal35_decoder_model(self.control_args.decoder_model):
                auto_model = None
                input_embeddings = _load_lincal35_input_embedding(self.config.repo_id)
            else:
                auto_model = AutoModel.from_pretrained(
                    self.config.repo_id, _attn_implementation="eager"
                )
                input_embeddings = auto_model.get_input_embeddings().to(torch.float32)
            if auto_model is not None:
                input_embeddings = auto_model.get_input_embeddings().to(torch.float32)
            tok_embedding = TokenEmbedding(
                input_embeddings,
                self.model_args.max_batch_size,
                self.model_args.ar_len,
                self.model_args.vocab_size,
                self.model_args.dim,
                use_i64_token,
            )
        # get decoder model
        if self.control_args.decoder_model in {"gemma-2b", "gemma3-1b"}:
            # For gemma, we have preprocessed the weight of rmsnorm
            self.model_args.norm_type = "rmsnorm"

        if self.control_args.decoder_model == "gemma4-e2b":
            from executorch.examples.qualcomm.oss_scripts.gemma4.model_wrapper import (
                Gemma4TextModelWrapper,
            )

            ar_len = (
                self.control_args.prefill_ar_len
                if self.mode == Mode.PREFILL
                else (
                    self.control_args.max_context_len
                    if self.mode == Mode.CALIBRATE
                    else 1
                )
            )
            extra_kwargs = {
                # 32 is the sentinel for "unquantized KV IO"; get_kv_io_bit_width()
                # returns it too when the recipe has no default_quant_dtype.
                "kv_io_bit_width": (
                    self.quant_recipe.get_kv_io_bit_width() if self.quant_recipe else 32
                ),
            }
            decoder: Gemma4TextModelWrapper = Gemma4TextModelWrapper(
                self.gemma4_config,
                ar_len=ar_len,
                output_new_cache_only=True,
                output_cache=True,
                use_i64_token=use_i64_token,
                enable_masked_softmax=False,
                **extra_kwargs,
            )
        else:
            decoder_cls = LLM_VARIANT_ARCHS.get(
                self.control_args.decoder_model, LlamaModel
            )
            if getattr(self.control_args, "enable_true_swa", False) and inspect.isclass(
                decoder_cls
            ):
                if issubclass(decoder_cls, MultiScopeAwareLlamaModel):
                    decoder_cls = MultiScopeAwareLlamaModel_swa
                elif issubclass(decoder_cls, MultiScopeAwareLlamaModelWithoutEmbedding):
                    decoder_cls = MultiScopeAwareLlamaModelWithoutEmbedding_swa
                elif issubclass(decoder_cls, Qwen35ModelWithoutEmbedding) and (
                    getattr(self.model_args, "sliding_window", None) is not None
                    and "sliding_attention" in (self.model_args.layer_types or [])
                ):
                    decoder_cls = Qwen35ModelWithoutEmbedding_swa

            decoder: LlamaModel = decoder_cls(
                self.model_args,
                ar_len=self.model_args.ar_len,
                output_new_cache_only=True,
                output_cache=True,
                use_i64_token=use_i64_token,
                **get_model_specific_kwargs(self.control_args, self.config),
            )

        self.meta = decoder.get_metadata()
        # get example input
        self.example_input = decoder.get_example_inputs()
        self.get_example_inputs = decoder.get_example_inputs
        self.export_input = (
            self.example_input[0],  # tokens or hidden_states
            *self.example_input[1],  # attn_mask
            *((self.example_input[2],) if decoder.use_kv_cache else []),  # pos_ids
            *(self.example_input[3] if decoder.use_kv_cache else []),  # k_caches
            *(self.example_input[4] if decoder.use_kv_cache else []),  # v_caches
        )
        self.io_shape = {
            # logit output
            (
                decoder.max_batch_size,
                decoder.ar_len,
                decoder.vocab_size,
            ),
        }
        # shape of k caches and v caches
        if self.control_args.decoder_model == "gemma4-e2b":
            # Gemma 4 has per-layer head_dim: sliding=256, full=512
            kv_head_dims = {
                self.meta["get_head_dim"],
                self.meta["get_global_head_dim"],
            }
            self.kv_cache_shape = set()
            for head_dim in kv_head_dims:
                self.kv_cache_shape.add((head_dim, self.meta["get_max_context_len"]))
                self.kv_cache_shape.add((self.meta["get_max_context_len"], head_dim))
                self.kv_cache_shape.add((head_dim, self.meta["get_ar_len"]))
                self.kv_cache_shape.add((self.meta["get_ar_len"], head_dim))
        else:
            self.kv_cache_shape = {
                # single head, kv input
                (self.meta["get_head_dim"], self.meta["get_max_context_len"]),
                (self.meta["get_max_context_len"], self.meta["get_head_dim"]),
                # single head, kv output
                (self.meta["get_head_dim"], self.meta["get_ar_len"]),
                (self.meta["get_ar_len"], self.meta["get_head_dim"]),
            }

        cache_len = self.meta["get_max_context_len"] - self.meta["get_ar_len"]
        for head_dim in {
            shape[0]
            for shape in self.kv_cache_shape
            if shape[1] == self.meta["get_max_context_len"]
        }:
            self.kv_cache_shape |= {
                (head_dim, cache_len),
                (cache_len, head_dim),
            }

        self.cache_io_shape = set()
        if decoder.use_kv_cache:
            self.cache_io_shape = {
                tuple(cache.shape)
                for cache in (*self.example_input[3], *self.example_input[4])
            }

        if getattr(self.control_args, "enable_true_swa", False) and (
            "get_sliding_window" in self.meta
        ):
            max_seq_len = self.meta.get(
                "get_max_seq_len", self.meta["get_max_context_len"]
            )
            window_context_len = (
                min(max_seq_len, self.meta["get_sliding_window"])
                + self.meta["get_ar_len"]
                - 1
            )
            window_cache_len = max(0, window_context_len - self.meta["get_ar_len"])
            self.kv_cache_shape |= {
                (self.meta["get_head_dim"], window_cache_len),
                (window_cache_len, self.meta["get_head_dim"]),
                (self.meta["get_head_dim"], window_context_len),
                (window_context_len, self.meta["get_head_dim"]),
            }

        if self.apply_embedding:
            self.tok_embedding_export_input = (
                tok_embedding.get_example_input()
            )  # tokens

        return tok_embedding, decoder

    @property
    def attn_mask(self):
        """Attention mask for this decoder graph, used as a schema for dataset construction."""
        return self.example_input[1]

    def _save_logits_quant_attrs(self):
        for node in self.decoder.graph.nodes:
            if node.op == "output":
                for output_node in node.args[0]:
                    if (
                        output_node.target
                        == torch.ops.quantized_decomposed.dequantize_per_tensor.default
                    ):
                        source_node = output_node.args[0].args[0]
                        if source_node.meta["val"].size() in self.io_shape:
                            self.meta["get_logits_scale"] = output_node.args[1]
                            self.meta["get_logits_zero_point"] = output_node.args[2]
                            break

    def _save_output_kv_cache_quant_attrs(self):
        kv_idx = 0
        for node in self.decoder.graph.nodes:
            if not is_graph_output(node):
                continue
            cache_output_node = node.args[0].args[0]
            if cache_output_node.meta["val"].size()[-2:] in self.kv_cache_shape:
                # [QCOM_SCALE, QCOM_ZERO_POINT, QCOM_QUANT_MIN, QCOM_QUANT_MAX, QCOM_DTYPE]
                # This meta is for attention sink feature
                self.meta[f"get_kv_output_{kv_idx}_quant_attr"] = [
                    node.args[1],
                    node.args[2],
                    node.args[3],
                    node.args[4],
                    str(node.args[5]),
                ]
                kv_idx += 1

    def _save_input_kv_cache_quant_attrs(self):
        kv_idx = 0
        quantize_targets = {
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.quantize_per_tensor.tensor,
        }
        for node in self.decoder.graph.nodes:
            if node.op != "placeholder" or "val" not in node.meta:
                continue
            if tuple(node.meta["val"].size()) not in self.cache_io_shape:
                continue
            quant_node = next(
                (user for user in node.users if user.target in quantize_targets),
                None,
            )
            if quant_node is None:
                continue
            self.meta[f"get_kv_input_{kv_idx}_scale"] = quant_node.args[1]
            self.meta[f"get_kv_input_{kv_idx}_zero_point"] = quant_node.args[2]
            kv_idx += 1
        if kv_idx == 0:
            logging.warning("No KV cache input qparams were saved from the graph.")
        else:
            logging.info(
                "Saved %d KV cache input qparams from the converted graph.",
                kv_idx,
            )

    def _collect_input_kv_cache_qparams(self):
        quantize_targets = {
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.quantize_per_tensor.tensor,
        }
        cache_input_qparams = []
        for node in self.decoder.graph.nodes:
            if node.op != "placeholder" or "val" not in node.meta:
                continue
            if tuple(node.meta["val"].size()) not in self.cache_io_shape:
                continue
            quant_node = next(
                (user for user in node.users if user.target in quantize_targets),
                None,
            )
            if quant_node is None:
                continue
            cache_input_qparams.append(
                (
                    quant_node.args[1],
                    quant_node.args[2],
                    quant_node.args[3],
                    quant_node.args[4],
                    quant_node.args[5],
                )
            )
        return cache_input_qparams

    def _find_qdq_node(self, node, targets, visited):
        if not isinstance(node, torch.fx.Node):
            return None
        if node in visited:
            return None
        visited.add(node)
        if node.target in targets:
            return node
        for arg in node.args:
            if isinstance(arg, torch.fx.Node):
                qdq_node = self._find_qdq_node(arg, targets, visited)
                if qdq_node is not None:
                    return qdq_node
            if isinstance(arg, (list, tuple)):
                for item in arg:
                    qdq_node = self._find_qdq_node(item, targets, visited)
                    if qdq_node is not None:
                        return qdq_node
        return None

    def _apply_qparams(self, qdq_node, qparams):
        old_args = qdq_node.args
        new_args = (
            old_args[0],
            qparams[0],
            qparams[1],
            qparams[2],
            qparams[3],
            qparams[4],
        )
        if old_args == new_args:
            return False
        qdq_node.args = new_args
        return True

    def _align_output_kv_cache_qparams(self, cache_input_qparams, source: str):
        quantize_targets = {
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.quantize_per_tensor.tensor,
        }
        dequantize_targets = {
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.tensor,
        }
        changed = 0
        visited = 0
        for node in self.decoder.graph.nodes:
            if node.op != "output":
                continue
            outputs = node.args[0]
            for output_index, output_node in enumerate(outputs[1:]):
                if output_index >= len(cache_input_qparams):
                    break
                dequant_node = self._find_qdq_node(
                    output_node, dequantize_targets, set()
                )
                if dequant_node is None:
                    continue
                visited += 1
                qparams = cache_input_qparams[output_index]
                quant_node = self._find_qdq_node(
                    dequant_node.args[0], quantize_targets, set()
                )
                if quant_node is not None and self._apply_qparams(quant_node, qparams):
                    changed += 1
                if self._apply_qparams(dequant_node, qparams):
                    changed += 1
            break

        if changed > 0:
            self.decoder.graph.lint()
            self.decoder.recompile()
            logging.info(
                "Aligned %d KV cache output qparams to %s qparams.",
                changed,
                source,
            )
        else:
            logging.info(
                "KV cache qparam alignment against %s visited %d outputs, no changes needed.",
                source,
                visited,
            )

    def _align_input_kv_cache_qparams(self, cache_input_qparams, source: str):
        quantize_targets = {
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.quantize_per_tensor.tensor,
        }
        changed = 0
        visited = 0
        kv_idx = 0
        for node in self.decoder.graph.nodes:
            if node.op != "placeholder" or "val" not in node.meta:
                continue
            if tuple(node.meta["val"].size()) not in self.cache_io_shape:
                continue
            if kv_idx >= len(cache_input_qparams):
                break
            quant_node = next(
                (user for user in node.users if user.target in quantize_targets),
                None,
            )
            visited += 1
            if quant_node is not None and self._apply_qparams(
                quant_node, cache_input_qparams[kv_idx]
            ):
                changed += 1
            kv_idx += 1

        if changed > 0:
            self.decoder.graph.lint()
            self.decoder.recompile()
            logging.info(
                "Aligned %d KV cache input qparams to %s qparams.",
                changed,
                source,
            )
        else:
            logging.info(
                "KV cache input qparam alignment against %s visited %d inputs, no changes needed.",
                source,
                visited,
            )

    def _align_kv_cache_io_qparams(self):
        cache_input_qparams = self._collect_input_kv_cache_qparams()
        expected_cache_qparams = 2 * self.meta["get_n_layers"]
        if len(cache_input_qparams) != expected_cache_qparams:
            logging.warning(
                "Expected %d cache input qparams, found %d; skip KV cache qparam alignment.",
                expected_cache_qparams,
                len(cache_input_qparams),
            )
            return

        self._align_output_kv_cache_qparams(cache_input_qparams, "matching input")

    def _tag_ios(self, node, fixed_point_type):  # noqa: C901
        max_context_len = self.meta["get_max_context_len"]
        max_seq_len = self.meta.get("get_max_seq_len", max_context_len)
        atten_mask_shape = {
            (
                self.meta["get_max_batch_size"],
                1,  # num_heads=1: the mask broadcasts across all heads.
                self.meta["get_ar_len"],
                max_context_len,
            ),
        }
        if max_context_len != max_seq_len:
            atten_mask_shape.add(
                (
                    self.meta["get_max_batch_size"],
                    1,
                    self.meta["get_ar_len"],
                    max_seq_len,
                )
            )

        if getattr(self.control_args, "enable_true_swa", False) and (
            "get_sliding_window" in self.meta
        ):
            window_context_len = (
                min(max_seq_len, self.meta["get_sliding_window"])
                + self.meta["get_ar_len"]
                - 1
            )
            window_cache_len = max(0, window_context_len - self.meta["get_ar_len"])
            self.kv_cache_shape |= {
                (self.meta["get_head_dim"], window_cache_len),
                (window_cache_len, self.meta["get_head_dim"]),
                (self.meta["get_head_dim"], window_context_len),
                (window_context_len, self.meta["get_head_dim"]),
            }
            atten_mask_shape.add(
                (
                    self.meta["get_max_batch_size"],
                    1,
                    self.meta["get_ar_len"],
                    window_context_len,
                )
            )

        freq_shape = {
            (self.meta["get_ar_len"], self.meta["get_head_dim"] // 2),
        }
        if self.control_args.decoder_model == "gemma4-e2b":
            freq_shape.add(
                (self.meta["get_ar_len"], self.meta["get_global_head_dim"] // 2)
            )

        freq_op = {exir_ops.edge.aten.select.int, exir_ops.edge.aten.select_copy.int}
        quant_io_type = None

        if node.op == "placeholder":
            users = [
                user
                for user in node.users
                if not (isinstance(user.target, str) and user.target == "_guards_fn")
            ]
            node_shape = tuple(node.meta["val"].size())
            if (
                _is_qwen35_decoder_model(self.control_args.decoder_model)
                and node_shape in self.cache_io_shape
            ):
                quant_io_type = fixed_point_type["kv_type"]
            elif (
                len(users) > 0
                and "args_" in node.name
                and users[0].meta["val"].size()[-2:] in self.kv_cache_shape
            ):
                quant_io_type = fixed_point_type["kv_type"]
            elif node_shape in self.io_shape:
                quant_io_type = fixed_point_type["io_type"]
            elif _is_attention_mask_placeholder(node, atten_mask_shape):
                quant_io_type = fixed_point_type["io_type"]

        if is_graph_output(node):
            node_shape = tuple(node.meta["val"].size())
            if (
                _is_qwen35_decoder_model(self.control_args.decoder_model)
                and node_shape in self.cache_io_shape
            ):
                quant_io_type = fixed_point_type["kv_type"]
            elif node.meta["val"].size()[-2:] in self.kv_cache_shape:
                quant_io_type = fixed_point_type["kv_type"]
            elif node_shape in self.io_shape:
                quant_io_type = fixed_point_type["io_type"]

        # tag sharding io
        if exir_ops.edge.llama.fallback.default in [
            u.target for u in list(node.users.keys())
        ] + [node.target]:
            quant_io_type = fixed_point_type["io_type"]

        # tag select op as quantized tensors for freq_sin and freq_cos. It is caused by sharding
        if (
            not _is_qwen35_decoder_model(self.control_args.decoder_model)
            and node.target in freq_op
            and node.meta["val"].size() in freq_shape
        ):
            quant_io_type = fixed_point_type["io_type"]

        # Matched against stack_trace source text, so this is coupled to the
        # variable names in model forward: renaming per_layer_inputs or
        # changing how it is sliced silently drops the tag.
        if (
            self.control_args.decoder_model == "gemma4-e2b"
            and "stack_trace" in node.meta
            and (
                "per_layer_inputs[n_self:]" in node.meta["stack_trace"]
                or "per_layer_inputs[i]" in node.meta["stack_trace"]
            )
        ):
            quant_io_type = fixed_point_type["io_type"]

        # Tag the donor K/V that YOCO shared layers consume.
        #
        # Matched on full_k/full_v are the local variable names in model forward.
        # Renaming them drops the tag.
        if self.control_args.decoder_model == "gemma4-e2b" and (
            is_node_src_start_with_name(node, "full_k")
            or is_node_src_start_with_name(node, "full_v")
        ):
            quant_io_type = fixed_point_type["io_type"]
        return quant_io_type

    def _quant_recipe_suggestion(
        self,
        fp32_gm: torch.fx.GraphModule,
        qdq_gm: torch.fx.GraphModule,
        text_dataloader: DataLoader,
        recipe: StaticLLMQuantRecipe,
    ):
        """
        Compare fp32 vs QDQ intermediate outputs and write SQNR reports.

        fp32_gm: Fp32 exported GraphModule (before prepare_pt2e).
        qdq_gm: QDQ GraphModule (after convert_pt2e).

        Output files:
          ``{model_name}_quantization_error.csv``: per-group statistics
          ``{model_name}_suggest_recipe.py``: Python script containing quantization recipe classes
        based on the suggested quant recipe overrides.
        """
        model_name = self.control_args.decoder_model
        report = PerLayerSqnrAnalyzer(
            model_name=model_name,
            num_layers=self.meta["get_n_layers"],
            fp32_gm=fp32_gm,
            qdq_gm=qdq_gm,
            analysis_recipe=recipe,
        ).analyze(
            self._decoder_inference,
            text_dataloader,
        )
        report.save_analysis_summary()
        suggest_recipe_overrides = report.suggest_recipe_overrides()
        save_suggest_recipes(report, suggest_recipe_overrides)

    def _auto_tune_calibration_threads(self):
        """Find the optimal thread count for calibration via quick microbenchmark.

        The default thread count is often higher than the calibration graph can
        use efficiently. Run a few observer-free forwards and keep the fastest
        setting for the PTQ calibration pass.
        """
        # Use sched_getaffinity when available — it respects cgroup/taskset
        # constraints (e.g. containers), unlike os.cpu_count() which returns
        # the host total regardless of pinning.
        available = (
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else (os.cpu_count() or 1)
        )
        baseline = min(torch.get_num_threads(), available)
        # Sample fractions of the thread ceiling from low through the
        # bandwidth-saturation knee up to the current default.
        fractions = (1 / 8, 1 / 4, 3 / 8, 1 / 2, 2 / 3, 3 / 4, 1.0)
        candidates = sorted(
            {1, baseline} | {max(1, round(baseline * f)) for f in fractions}
        )
        original = torch.get_num_threads()
        best_threads, best_time = original, float("inf")
        try:
            for n_threads in candidates:
                torch.set_num_threads(n_threads)
                try:
                    with torch.no_grad():
                        self.decoder(*self.export_input)  # warmup
                        t0 = time.perf_counter()
                        for _ in range(3):
                            self.decoder(*self.export_input)
                        elapsed = time.perf_counter() - t0
                    if elapsed < best_time:
                        best_threads, best_time = n_threads, elapsed
                except Exception:
                    logging.debug("Auto-tune: threads=%d failed, skipping", n_threads)
                    continue
        finally:
            torch.set_num_threads(original)
        if best_time == float("inf"):
            logging.warning(
                "Auto-tune: all candidates %s failed, falling back to %d threads",
                candidates,
                baseline,
            )
            return baseline
        logging.info(
            "Auto-tune calibration threads: tested %s, best=%d (%.1fms/fwd)",
            candidates,
            best_threads,
            best_time / 3 * 1000,
        )
        return best_threads

    @log_info
    def quantize(self, request: Request):  # noqa: C901
        if self.quant_recipe is None:
            return

        if self.decoder is None or (
            self.apply_embedding and self.tok_embedding is None
        ):
            return

        data = request.method_data[TEXT_DECODER]
        # check bit width graph io
        fixed_point_type = {"kv_type": torch.float32, "io_type": torch.float32}
        if data.skip_quantize:
            # already init as float32
            return
        else:
            if self.quant_recipe.get_kv_io_bit_width() == 8:
                fixed_point_type["kv_type"] = torch.uint8
            elif self.quant_recipe.get_kv_io_bit_width() == 16:
                fixed_point_type["kv_type"] = torch.uint16
            else:
                raise RuntimeError(
                    f"unknown kv io bit width {self.quant_recipe.get_kv_io_bit_width()}"
                )

            if self.quant_recipe.get_logits_output_bit_width() == 16:
                fixed_point_type["io_type"] = torch.uint16
            else:
                raise RuntimeError(
                    f"unknown logits io bit width {self.quant_recipe.get_logits_output_bit_width()}"
                )

        # embedding fallback and quantization
        if self.control_args.embedding_quantize:
            self.decoder = get_quant_embedding_transform(
                embedding_quantize=self.control_args.embedding_quantize
            )(self.decoder)
            if self.apply_embedding:
                self.tok_embedding = get_quant_embedding_transform(
                    embedding_quantize=self.control_args.embedding_quantize
                )(self.tok_embedding)

        data = request.method_data[TEXT_DECODER]

        quantizer = make_quantizer(backend=data.backend, soc_model=data.soc_model)
        quantizer.set_recipe(self.quant_recipe.recipe)
        tok_embedding_quantizer = make_quantizer(
            quant_dtype=QuantDtype.use_16a8w,
            per_channel_conv=True,
            per_channel_linear=True,
            act_observer=MinMaxObserver,
            backend=data.backend,
            soc_model=data.soc_model,
        )

        use_qat = self.control_args.qat and self.mode == Mode.CALIBRATE
        with torch.no_grad():
            graph_module = None
            self.decoder = torch.export.export(
                self.decoder, self.export_input, strict=True
            ).module()
            if self.mode == Mode.CALIBRATE and (
                self.control_args.quant_recipe_suggestion or use_qat
            ):
                graph_module = copy.deepcopy(self.decoder)
            if self.apply_embedding:
                self.tok_embedding = torch.export.export(
                    self.tok_embedding,
                    self.tok_embedding.get_example_input(),
                    strict=True,
                ).module()

            if (
                self.control_args.verbose
                and self.mode == Mode.CALIBRATE
                and not self.apply_embedding
            ):
                run_lm_eval(
                    module=self.decoder,
                    get_example_inputs=self.get_example_inputs,
                    tokenizer=data.tokenizer,
                    max_seq_length=self.meta["get_max_context_len"],
                    tasks=self.control_args.eval_tasks,
                    use_i64_token=self.control_args.embedding_quantize is not None,
                    num_fewshot=self.control_args.eval_num_fewshot,
                    limit=self.control_args.eval_limit,
                    max_batch_size=self.meta["get_max_batch_size"],
                    event_name="export_tasks",
                )

            calibration_threads = None
            calibration_device = self.control_args.calibration_device
            if (
                self.mode == Mode.CALIBRATE
                and not use_qat
                and calibration_device == "cpu"
            ):
                calibration_threads = getattr(
                    self.control_args, "calibration_num_threads", 0
                )
                if calibration_threads == 0:
                    calibration_threads = self._auto_tune_calibration_threads()

            if use_qat:
                self.decoder = prepare_qat_pt2e(self.decoder, quantizer)
                move_exported_model_to_train(self.decoder)
            else:
                self.decoder = prepare_pt2e(self.decoder, quantizer)
            if self.apply_embedding:
                self.tok_embedding = prepare_pt2e(
                    self.tok_embedding, tok_embedding_quantizer
                )

            if self.mode == Mode.CALIBRATE:
                calibration_dataloaders = {
                    AUDIO_ENCODER: request.method_data[
                        AUDIO_ENCODER
                    ].quantization_data.intermediate_outputs,
                    VISION_ENCODER: request.method_data[
                        VISION_ENCODER
                    ].quantization_data.intermediate_outputs,
                    TEXT_DECODER: data.quantization_data.calib_loader,
                }

                if use_qat:
                    training_args = TrainingArgs.from_yaml(
                        self.control_args.train_config
                    )
                    training_args.lr_config = self.control_args.lr_config
                    frozen = (
                        [".*"]
                        if self.control_args.freeze_all_params
                        # freeze_all_params: CI-only flag to verify QAT vs PTQ accuracy difference
                        # by disabling weight updates and only updating scale/zero_point.
                        else getattr(self.quant_recipe, "frozen_param_patterns", None)
                    )
                    QATStrategy(
                        inference=self._decoder_inference,
                        module=self.decoder,
                        tok_embedding=self.tok_embedding,
                        seq_mse_candidates=self.config.seq_mse_candidates,
                    ).quantize(
                        calib_loader=calibration_dataloaders,
                        training_args=training_args,
                        teacher=graph_module,
                        train_loader=data.quantization_data.train_loader,
                        val_loader=data.quantization_data.val_loader,
                        frozen_param_patterns=frozen,
                    )
                    logging.info("QAT training complete")
                else:
                    original_threads = torch.get_num_threads()
                    if calibration_threads is not None:
                        torch.set_num_threads(calibration_threads)
                        logging.info(
                            "Calibration using %d threads (was %d)",
                            calibration_threads,
                            original_threads,
                        )
                    try:
                        PTQStrategy(
                            inference=self._decoder_inference,
                            module=self.decoder,
                            tok_embedding=self.tok_embedding,
                            seq_mse_candidates=self.config.seq_mse_candidates,
                            device=calibration_device,
                        ).quantize(calib_loader=calibration_dataloaders)
                    finally:
                        torch.set_num_threads(original_threads)
                    logging.info("Calibration complete")
            else:
                # one dummy inference to remove affine observer
                # error happened in convert_pt2e
                self.decoder(*self.export_input)

            self.decoder = convert_pt2e(self.decoder)
            if _is_qwen35_decoder_model(self.control_args.decoder_model):
                self._align_kv_cache_io_qparams()

            if (
                self.mode == Mode.CALIBRATE
                and self.control_args.quant_recipe_suggestion
            ):
                self._quant_recipe_suggestion(
                    graph_module,
                    self.decoder,
                    calibration_dataloaders[TEXT_DECODER],
                    self.quant_recipe.recipe,
                )

            # FP32 model used as QAT teacher or quant-recipe-suggestion reference; release after use.
            del graph_module
            gc.collect()

            if self.apply_embedding:
                self.tok_embedding = convert_pt2e(self.tok_embedding)

            if (
                self.control_args.verbose
                and self.mode == Mode.CALIBRATE
                and not self.apply_embedding
            ):
                run_lm_eval(
                    module=self.decoder,
                    get_example_inputs=self.get_example_inputs,
                    tokenizer=data.tokenizer,
                    max_seq_length=self.meta["get_max_context_len"],
                    tasks=self.control_args.eval_tasks,
                    use_i64_token=self.control_args.embedding_quantize is not None,
                    num_fewshot=self.control_args.eval_num_fewshot,
                    limit=self.control_args.eval_limit,
                    max_batch_size=self.meta["get_max_batch_size"],
                    event_name="convert_pt2e_tasks",
                )

        # save logit's quantization attributes to meta
        self._save_logits_quant_attrs()

        # save KV cache input quantization attributes for qzero initialization
        if _is_qwen35_decoder_model(self.control_args.decoder_model):
            self._save_input_kv_cache_quant_attrs()

        # save output KV cache's quantization attributes to meta for attention sink
        self._save_output_kv_cache_quant_attrs()

        # setup quantized IO
        self.passes_job[TagQuantIO][QCOM_PASS_ACTIVATE_KEY] = True
        self.passes_job[TagQuantIO][QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY][
            "get_quant_io_dtype_fn"
        ] = partial(self._tag_ios, fixed_point_type=fixed_point_type)
        if self.tok_embedding_passes_job is not None:
            self.tok_embedding_passes_job[TagQuantIO][QCOM_PASS_ACTIVATE_KEY] = True
            self.tok_embedding_passes_job[TagQuantIO][
                QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY
            ]["get_quant_io_dtype_fn"] = partial(
                self._tag_ios, fixed_point_type=fixed_point_type
            )


class HybridTextDecoder(Component):
    @log_info
    def __init__(
        self,
        control_args: argparse.Namespace,
        config: LLMModelConfig,
        apply_embedding: bool = False,
    ):
        self.decode = TextDecoder(
            control_args,
            config,
            Mode.DECODE,
            apply_embedding=apply_embedding,
        )
        self.prefill = TextDecoder(
            control_args,
            config,
            Mode.PREFILL,
            apply_embedding=apply_embedding,
        )
        # Full AR sequence with KV cache; used only for quantization.
        # Scales/zp collected here are propagated to decode and prefill graphs via _encoding_override.
        self.calibration_prefill = TextDecoder(
            control_args,
            config,
            Mode.CALIBRATE,
            apply_embedding=apply_embedding,
        )

        self.control_args = control_args
        self.config = config
        self.set_next(self.decode).set_next(self.prefill)

        self.apply_embedding = apply_embedding

    def _encoding_override(  # noqa: C901
        self,
        quantized_model,
        unquantized_model,
        override_kv_cache,
        allow_non_isomorphic=False,
    ):
        pbq_target = {
            torch.ops.torchao.dequantize_affine,
            torch.ops.torchao.quantize_affine,
        }
        pcq_target = {
            torch.ops.quantized_decomposed.dequantize_per_channel.default,
            torch.ops.quantized_decomposed.quantize_per_channel.default,
        }
        ptq_target = {
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
        }
        qdq_target = pbq_target | pcq_target | ptq_target

        def compare_nodes(quantized_node, unquantized_node):
            def info(node):
                return node.name + (
                    str(node.meta["nn_module_stack"].values())
                    if node.op == "call_function"
                    else ""
                )

            assert info(quantized_node) == info(
                unquantized_node
            ), f"found unmatched order for ops: {quantized_node} vs {unquantized_node}"

        def resolve_param_target(node):
            return (
                node
                if node.op == "call_function" and node.target not in qdq_target
                else resolve_param_target(list(node.users)[0])
            )

        def activation_override(quantized_node, unquantized_node):
            quantized_qdq_users = {}
            for user in list(quantized_node.users):
                if user.target in qdq_target:
                    quantized_qdq_users.setdefault(user.target, []).append(user)

            target_occurrence = {}
            for unquantized_user in list(unquantized_node.users):
                if unquantized_user.target not in qdq_target:
                    continue
                occurrence = target_occurrence.get(unquantized_user.target, 0)
                candidates = quantized_qdq_users.get(unquantized_user.target, [])
                if occurrence >= len(candidates):
                    raise RuntimeError(
                        "Quantization encoding propagation found no matching Q/DQ "
                        f"user for {unquantized_user} from {unquantized_node}."
                    )
                quantized_user = candidates[occurrence]
                target_occurrence[unquantized_user.target] = occurrence + 1
                unquantized_user.args = (
                    unquantized_user.args[0],
                    *quantized_user.args[1:],
                )
                activation_override(quantized_user, unquantized_user)

        def parameter_override(quantized_node, unquantized_node):
            # Some parameters need to be iterated over to retrieve attributes such as static_llama.tok_embedding.weight
            def _get_attr(graph_module: torch.fx.GraphModule, target: str) -> Any:
                attr: Any = graph_module
                for target_atom in target.split("."):
                    attr = getattr(attr, target_atom)
                return attr

            def _set_attr(
                graph_module: torch.fx.GraphModule, target: str, replacement: Any
            ) -> Any:
                attr: Any = graph_module
                target_list = target.split(".")
                for target_atom in target_list[:-1]:
                    attr = getattr(attr, target_atom)
                setattr(attr, target_list[-1], replacement)

            _set_attr(
                unquantized_model,
                unquantized_node.target,
                _get_attr(quantized_model, quantized_node.target),
            )
            # scale / zero point are part of op's attributes
            if any(user.target in ptq_target for user in unquantized_node.users):
                activation_override(quantized_node, unquantized_node)

        def semantic_key(node):
            module_stack = tuple(
                str(value) for value in node.meta.get("nn_module_stack", {}).values()
            )
            return (
                node.op,
                str(node.target),
                module_stack,
                node.meta.get("stack_trace", ""),
            )

        def match_semantic_nodes(quantized_nodes, unquantized_nodes, kind):
            candidates = {}
            for node in quantized_nodes:
                candidates.setdefault(semantic_key(node), []).append(node)

            occurrences = {}
            pairs = []
            for node in unquantized_nodes:
                key = semantic_key(node)
                occurrence = occurrences.get(key, 0)
                matches = candidates.get(key, [])
                if occurrence >= len(matches):
                    raise RuntimeError(
                        f"Unable to propagate {kind} encoding for node {node.name}: "
                        "the calibration graph has no matching source operation."
                    )
                pairs.append((matches[occurrence], node))
                occurrences[key] = occurrence + 1
            return pairs

        # copy encoding for hybrid mode
        parameters = [
            {
                n: resolve_param_target(n)
                for n in model.graph.nodes
                if n.op == "get_attr"
            }
            for model in (quantized_model, unquantized_model)
        ]
        activations = [
            [
                n
                for n in model.graph.nodes
                if n.target not in qdq_target
                and n.op in {"call_function", "placeholder"}
            ]
            for model in (quantized_model, unquantized_model)
        ]
        if allow_non_isomorphic:
            activation_pairs = match_semantic_nodes(
                activations[0], activations[1], "activation"
            )
            quantized_parameters = {node.target: node for node in parameters[0].keys()}
            parameter_pairs = []
            for node in parameters[1].keys():
                if node.target not in quantized_parameters:
                    raise RuntimeError(
                        f"Unable to propagate parameter encoding for {node.target}: "
                        "the calibration graph has no matching parameter."
                    )
                parameter_pairs.append((quantized_parameters[node.target], node))
        else:
            if len(activations[0]) != len(activations[1]):
                raise RuntimeError(
                    "Encoding propagation requires isomorphic graphs, but activation "
                    f"counts differ ({len(activations[0])} vs {len(activations[1])})."
                )
            if len(parameters[0]) != len(parameters[1]):
                raise RuntimeError(
                    "Encoding propagation requires isomorphic graphs, but parameter "
                    f"counts differ ({len(parameters[0])} vs {len(parameters[1])})."
                )
            activation_pairs = list(zip(*activations))
            parameter_pairs = list(zip(*[p.keys() for p in parameters]))
            for act_quantized, act_unquantized in activation_pairs:
                compare_nodes(act_quantized, act_unquantized)
            for op_quantized, op_unquantized in zip(*[p.values() for p in parameters]):
                compare_nodes(op_quantized, op_unquantized)

        for act_quantized, act_unquantized in activation_pairs:
            activation_override(act_quantized, act_unquantized)

        for param_quantized, param_unquantized in parameter_pairs:
            parameter_override(param_quantized, param_unquantized)

        if override_kv_cache:
            k_input_cache_nodes = []
            v_input_cache_nodes = []
            for node in unquantized_model.graph.nodes:
                if node.op != "placeholder":
                    continue

                if "args_" in node.name:
                    args_idx = int(node.name.split("_")[-1])

                    n_cache_layers = (
                        self.decode.meta["get_n_self_layers"]
                        if self.control_args.decoder_model == "gemma4-e2b"
                        else self.decode.meta["get_n_layers"]
                    )
                    if args_idx >= n_cache_layers:
                        v_input_cache_nodes.append(node)
                    else:
                        k_input_cache_nodes.append(node)

            if not k_input_cache_nodes or not v_input_cache_nodes:
                raise RuntimeError(
                    "KV cache input detection failed. This likely means the model naming "
                    "does not match expected prefixes."
                )

            k_output_cache_nodes = []
            v_output_cache_nodes = []
            for node in quantized_model.graph.nodes:
                if not is_graph_output(node):
                    continue
                cache_output_node = node.args[0].args[0]
                if is_node_src_start_with_name(cache_output_node, kv_cache_prefix="k_"):
                    k_output_cache_nodes.append(cache_output_node)
                elif is_node_src_start_with_name(
                    cache_output_node, kv_cache_prefix="v_"
                ):
                    v_output_cache_nodes.append(cache_output_node)

            if not k_output_cache_nodes or not v_output_cache_nodes:
                raise RuntimeError(
                    "KV cache detection failed. This likely means the model naming "
                    "does not match expected prefixes."
                )

            for input_k_cache_node, output_k_cache_node in zip(
                k_input_cache_nodes, k_output_cache_nodes
            ):
                activation_override(output_k_cache_node, input_k_cache_node)
            for input_v_cache_node, output_v_cache_node in zip(
                v_input_cache_nodes, v_output_cache_nodes
            ):
                activation_override(output_v_cache_node, input_v_cache_node)

        unquantized_model.recompile()

    def quantize(self, request: Request):
        if request.method_data[TEXT_DECODER].skip_quantize:
            return
        self.calibration_prefill.quantize(request)

    @log_info
    def compile(self, request: Request):  # noqa: C901
        # perform encoding override for models to compile
        # ---
        # theoretically decode & prefill model should share the same encoding
        # given that they are using the identical calibration dataset.
        #
        #
        # here we use a mechanism to make sure the encoding align correctly and
        # save AoT quantization time as well.
        # ---
        if not request.method_data[TEXT_DECODER].skip_quantize:
            self._encoding_override(
                quantized_model=self.calibration_prefill.decoder,
                unquantized_model=self.decode.decoder,
                override_kv_cache=True,
                allow_non_isomorphic=_is_qwen35_decoder_model(
                    self.control_args.decoder_model
                ),
            )

            if self.apply_embedding:
                self._encoding_override(
                    quantized_model=self.calibration_prefill.tok_embedding,
                    unquantized_model=self.decode.tok_embedding,
                    override_kv_cache=False,
                )

            # For hybrid mode, override encoding of prefill model.
            if (
                self.prefill.decoder is not None
                and self.prefill.meta["get_use_kv_cache"]
            ):
                if _is_qwen35_decoder_model(self.control_args.decoder_model):
                    self._encoding_override(
                        quantized_model=self.calibration_prefill.decoder,
                        unquantized_model=self.prefill.decoder,
                        override_kv_cache=False,
                        allow_non_isomorphic=True,
                    )
                    decode_cache_qparams = self.decode._collect_input_kv_cache_qparams()
                    expected_cache_qparams = 2 * self.decode.meta["get_n_layers"]
                    if len(decode_cache_qparams) == expected_cache_qparams:
                        self.prefill._align_input_kv_cache_qparams(
                            decode_cache_qparams, "decode input"
                        )
                        self.prefill._align_output_kv_cache_qparams(
                            decode_cache_qparams, "decode input"
                        )
                    else:
                        logging.warning(
                            "Expected %d decode cache input qparams, found %d; "
                            "skip qwen3.5-family hybrid cross-graph KV qparam alignment.",
                            expected_cache_qparams,
                            len(decode_cache_qparams),
                        )
                else:
                    self._encoding_override(
                        quantized_model=self.decode.decoder,
                        unquantized_model=self.prefill.decoder,
                        override_kv_cache=True,
                    )

                if self.apply_embedding:
                    self._encoding_override(
                        quantized_model=self.decode.tok_embedding,
                        unquantized_model=self.prefill.tok_embedding,
                        override_kv_cache=False,
                    )

            if not self.control_args.skip_qdq_save:
                # SQNR evaluates full sequences. Keep the historical filename for compatibility.
                qdq_ep = torch.export.export(
                    self.calibration_prefill.decoder,
                    self.calibration_prefill.export_input,
                    strict=True,
                )
                qdq_ep_path = f"{self.decode.control_args.artifact}/{DECODE_QDQ_FILENAME}"
                torch.export.save(qdq_ep, qdq_ep_path)
                del qdq_ep
                logging.info(f"QDQ EP saved to {qdq_ep_path}")
            else:
                logging.info("Skipping full-sequence QDQ evaluation artifact")

            self.decode._save_logits_quant_attrs()
            if _is_qwen35_decoder_model(self.control_args.decoder_model):
                self.decode._save_input_kv_cache_quant_attrs()
            self.decode._save_output_kv_cache_quant_attrs()

        # calibration_prefill is only used for encoding override
        # free it once encoding override is complete.
        del self.calibration_prefill
        _release_memory()

        # prepare lowering tok_embedding if applicable
        if self.apply_embedding:
            tok_embedding_data = request.method_data[TOK_EMBEDDING]
            models = [
                d for d in [self.decode, self.prefill] if d.tok_embedding is not None
            ]
            tok_embedding_example_inputs = [
                m.tok_embedding_export_input for m in models if m is not None
            ]  # tokens
            tok_embedding_graph_names = TOK_EMBEDDING_GRAPH_NAMES[: len(models)]

        # prepare lowering decoder
        data = request.method_data[TEXT_DECODER]
        models = [d for d in [self.decode, self.prefill] if d.decoder is not None]
        example_inputs = [m.export_input for m in models if m is not None]
        graph_names = DECODER_GRAPH_NAMES[: len(models)]

        # start lowering
        if self.apply_embedding:
            tok_embedding_edge_prog_mgr = to_edge_transform_and_lower_to_qnn(
                module=dict(
                    zip(
                        tok_embedding_graph_names,
                        [model.tok_embedding for model in models],
                    )
                ),
                inputs=dict(
                    zip(tok_embedding_graph_names, tok_embedding_example_inputs)
                ),
                compiler_specs=dict(
                    zip(tok_embedding_graph_names, tok_embedding_data.compile_spec)
                ),
                dep_table=dict(
                    zip(
                        tok_embedding_graph_names,
                        [model.tok_embedding_dep_table for model in models],
                    )
                ),
                passes_job=dict(
                    zip(
                        tok_embedding_graph_names,
                        [model.tok_embedding_passes_job for model in models],
                    )
                ),
            )
            if self.control_args.verbose:
                for ep in tok_embedding_edge_prog_mgr._edge_programs.values():
                    print_delegation_info(ep.graph_module)

            executorch_config = ExecutorchBackendConfig(
                # For shared buffer, user must pass the memory address
                # which is allocated by RPC memory to executor runner
                memory_planning_pass=MemoryPlanningPass(
                    alloc_graph_input=False,
                    alloc_graph_output=False,
                ),
                passes=[BuildQuantIo()],
            )
            tok_embedding_exec_prog_mgr = tok_embedding_edge_prog_mgr.to_executorch(
                executorch_config
            )
            data = request.method_data[TOK_EMBEDDING]
            with open(
                f"{self.control_args.artifact}/{data.pte_filename}.pte", "wb"
            ) as file:
                tok_embedding_exec_prog_mgr.write_to_file(file)
            del tok_embedding_edge_prog_mgr, tok_embedding_exec_prog_mgr
            if not self.control_args.optrace:
                for model in models:
                    model.tok_embedding = None
                    model.tok_embedding_export_input = None
            _release_memory()

        # decoder lowering
        data = request.method_data[TEXT_DECODER]
        edge_prog_mgr = to_edge_transform_and_lower_to_qnn(
            module=dict(zip(graph_names, [model.decoder for model in models])),
            inputs=dict(zip(graph_names, example_inputs)),
            compiler_specs=dict(zip(graph_names, data.compile_spec)),
            constant_methods={**self.decode.meta},
            dep_table=dict(zip(graph_names, [model.dep_table for model in models])),
            passes_job=dict(zip(graph_names, [model.passes_job for model in models])),
            skip_node_op_set={"llama.fallback.default"},
        )

        if self.config.num_sharding > 1:
            for graph_name in graph_names:
                update_spill_fill_size(edge_prog_mgr.exported_program(graph_name))

        if self.control_args.verbose:
            for ep in edge_prog_mgr._edge_programs.values():
                print_delegation_info(ep.graph_module)

        executorch_config = ExecutorchBackendConfig(
            # For shared buffer, user must pass the memory address
            # which is allocated by RPC memory to executor runner
            memory_planning_pass=MemoryPlanningPass(
                alloc_graph_input=False,
                alloc_graph_output=False,
            ),
            passes=[BuildQuantIo()],
        )
        exec_prog_mgr = edge_prog_mgr.to_executorch(executorch_config)
        data = request.method_data[TEXT_DECODER]
        with open(
            f"{self.control_args.artifact}/{data.pte_filename}.pte", "wb"
        ) as file:
            exec_prog_mgr.write_to_file(file)
        del edge_prog_mgr, exec_prog_mgr
        _release_memory()


class Modality(Component):
    def __init__(
        self, control_args: argparse.Namespace, config: LLMModelConfig, modality
    ):
        self.control_args = control_args
        self.model = None
        self.modality = modality
        repo_id = config.repo_id
        self.preprocess = lambda data: data
        params_path = (
            config.params_path if control_args.params is None else control_args.params
        )
        local_model_config = _load_local_model_config(params_path)

        if config := getattr(config, modality, None):
            if modality == AUDIO_ENCODER:
                auto_model = AutoModelForSpeechSeq2Seq.from_pretrained(repo_id)
                self.num_layers = auto_model.config.encoder_config.num_layers
                self.ctx_size = auto_model.config.encoder_config.context_size
                backbone = auto_model
            elif modality == TEXT_ENCODER:
                raise NotImplementedError(f"{modality} is under development")
            elif modality == VISION_ENCODER:
                if control_args.decoder_model == "lincal_vl-3b_instruct":
                    auto_model = _get_cached_qwen2_5_vl_for_conditional_generation(
                        repo_id
                    )
                    backbone = auto_model.model
                    _overlay_local_vision_config(auto_model, local_model_config)
                    vision_config = auto_model.config.vision_config
                    auto_model.config.dtype = torch.float32
                    vision_config.dtype = torch.float32
                    self.num_layers = _get_vision_num_layers(auto_model)
                    auto_model = auto_model.to(torch.float32).eval()
                    self.model = config().create_encoder(auto_model.config).eval()
                    self.model.load_state_dict(
                        backbone.state_dict(), strict=False
                    )  # set strict to false to simplify parameter loading for non-text models
                else:
                    if _is_lincal35_decoder_model(control_args.decoder_model):
                        auto_config = AutoConfig.from_pretrained(repo_id)
                        config_holder = types.SimpleNamespace(config=auto_config)
                        _overlay_local_vision_config(config_holder, local_model_config)
                        vision_config = auto_config.vision_config
                        auto_config.dtype = torch.float32
                        vision_config.dtype = torch.float32
                        self.num_layers = _get_vision_num_layers(config_holder)
                        self.model = config().create_encoder(auto_config).eval()
                        self.model.load_state_dict(
                            _load_safetensor_state_dict_by_prefix(
                                repo_id, "model.visual.", strip_prefix="model."
                            ),
                            strict=False,
                        )
                    else:
                        auto_model = AutoModel.from_pretrained(
                            repo_id, _attn_implementation="eager"
                        )
                        backbone = auto_model
                        _overlay_local_vision_config(auto_model, local_model_config)
                        vision_config = auto_model.config.vision_config
                        auto_model.config.dtype = torch.float32
                        vision_config.dtype = torch.float32
                        self.num_layers = _get_vision_num_layers(auto_model)
                        auto_model = auto_model.to(torch.float32).eval()
                        self.model = config().create_encoder(auto_model.config).eval()
                        self.model.load_state_dict(
                            backbone.state_dict(), strict=False
                        )  # set strict to false to simplify parameter loading for non-text models
            else:
                raise NotImplementedError(f"Find no {modality}")

            if self.model is None:
                auto_model = auto_model.to(torch.float32).eval()
                self.model = config().create_encoder(auto_model.config).eval()
                self.model.load_state_dict(
                    backbone.state_dict(), strict=False
                )  # set strict to false to simplify parameter loading for non-text models
            self.model = self.model.to(torch.float32).eval()
            self.example_input = self.model.get_example_inputs()
            self.preprocess = getattr(self.model, "preprocess", lambda data: data)

            # set quant recipe
            self.quant_recipe: EncoderQuantRecipe = (
                config.quant_recipe(True) if config.quant_recipe else None
            )

            # metadata
            self.config = config

            self._encoder_inference = EncoderInference()

        self.pass_manager_cls = get_qnn_pass_manager_cls()
        self.passes_job = self.pass_manager_cls.get_capture_program_passes()
        self.dep_table = (
            self.pass_manager_cls.get_passes_dependency_for_capture_program()
        )

    def _tag_ios(self, node, fixed_point_type):
        quant_io_type = None

        # tag sharding io
        if exir_ops.edge.llama.fallback.default in [
            u.target for u in list(node.users.keys())
        ] + [node.target]:
            quant_io_type = fixed_point_type["io_type"]

        # GraniteSpeech: tag _to_copy op as quantized tensors for attn dist. It is caused by sharding
        if (
            issubclass(self.config, GraniteSpeechEncoder)
            and node.target == exir_ops.edge.aten._to_copy.default
            and node.meta["val"].size() == (self.ctx_size, self.ctx_size)
        ):
            quant_io_type = torch.int32

        return quant_io_type

    def _get_sharding_get_pattern(self):
        prefixes = [
            "encoder.layers",
            "visual.blocks",
            "vision_tower.encoder.layer",
            "vision_model.encoder.layers",
        ]
        prefix_alt = "|".join(re.escape(p) for p in prefixes)
        return rf"^(?:{prefix_alt})\.(\d+)"

    def compile(self, request: Request):
        if self.model is None:
            return

        request_data = request.method_data[self.modality]
        # check if sharding required
        if self.config.num_sharding > 1:
            SplitGraph, setting = model_sharding.get_split_graph_pass(
                self.num_layers,
                shares=self.config.num_sharding,
                pattern=self._get_sharding_get_pattern(),
            )
            self.passes_job[SplitGraph] = setting
            self.dep_table[SplitGraph] = [FoldQDQ]
            self.dep_table[TagQuantIO] = [SplitGraph]

            if not request_data.skip_quantize:
                fixed_point_type = {"io_type": torch.float32}

                # setup quantized IO
                self.passes_job[TagQuantIO][QCOM_PASS_ACTIVATE_KEY] = True
                self.passes_job[TagQuantIO][QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY][
                    "get_quant_io_dtype_fn"
                ] = partial(self._tag_ios, fixed_point_type=fixed_point_type)

        edge_prog_mgr = to_edge_transform_and_lower_to_qnn(
            module=self.model,
            inputs=self.example_input,
            compiler_specs=request_data.compile_spec,
            dep_table=self.dep_table,
            passes_job=self.passes_job,
            skip_node_op_set={"llama.fallback.default"},
        )

        if self.config.num_sharding > 1:
            update_spill_fill_size(edge_prog_mgr.exported_program())
        if self.control_args.verbose:
            print_delegation_info(edge_prog_mgr.exported_program().graph_module)

        exec_prog_mgr = edge_prog_mgr.to_executorch(
            ExecutorchBackendConfig(passes=[BuildQuantIo()])
        )
        data = request.method_data[self.modality]
        with open(
            f"{self.control_args.artifact}/{data.pte_filename}.pte", "wb"
        ) as file:
            exec_prog_mgr.write_to_file(file)
        del edge_prog_mgr, exec_prog_mgr
        if not self.control_args.optrace:
            self.model = None
            self.example_input = None
        _release_memory()

    def _calibrate(self, model, calibration_datasets):
        outputs = []
        for batch in calibration_datasets:
            outputs_each_batch = [
                self._encoder_inference.predict_step(model, data)
                for data in batch["inputs"]
            ]
            outputs.append(outputs_each_batch)
        return DataLoader(
            ModalityEncoderDataset(outputs),
            batch_size=self.control_args.batch_size,
            shuffle=False,
            drop_last=self.control_args.batch_size > 1,
        )

    def quantize(self, request: Request):
        if self.model is None:
            return

        request_data = request.method_data[self.modality]
        calibration_datasets = request_data.quantization_data.calib_loader

        with torch.no_grad():
            self.model = torch.export.export(self.model, self.example_input).module()

            if request_data.skip_quantize:
                logging.info(f"skipping encoder quantization for {self.modality}")
                intermediate_outputs = self._calibrate(self.model, calibration_datasets)
                request_data.quantization_data.intermediate_outputs = (
                    intermediate_outputs
                )
                return

            quantizer = make_quantizer(
                backend=request_data.backend, soc_model=request_data.soc_model
            )
            quantizer.set_recipe(self.quant_recipe.recipe)
            self.model = prepare_pt2e(self.model, quantizer)

            # start calibration
            intermediate_outputs = self._calibrate(self.model, calibration_datasets)
            request_data.quantization_data.intermediate_outputs = intermediate_outputs

            self.model = convert_pt2e(self.model)

            # QDQ inference
            if self.control_args.verbose:
                qdq_intermediate_outputs = self._calibrate(
                    self.model, calibration_datasets
                )
                request_data.quantization_data.qdq_intermediate_outputs = (
                    qdq_intermediate_outputs
                )


class MultiModalManager(Component):
    def __init__(self, control_args: argparse.Namespace, config: LLMModelConfig):
        self.control_args = control_args
        self.config = config
        self.audio_encoder = Modality(
            control_args,
            config,
            AUDIO_ENCODER,
        )
        self.text_encoder = Modality(
            control_args,
            config,
            TEXT_ENCODER,
        )
        self.vision_encoder = Modality(
            control_args,
            config,
            VISION_ENCODER,
        )
        self.text_decoder = HybridTextDecoder(
            control_args,
            config,
            apply_embedding=self.audio_encoder.model or self.vision_encoder.model,
        )
        self._modalities = [
            AUDIO_ENCODER,
            TEXT_ENCODER,
            VISION_ENCODER,
            TOK_EMBEDDING,
            TEXT_DECODER,
        ]
        # build dependency chain
        self.set_next(self.vision_encoder).set_next(self.audio_encoder).set_next(
            self.text_decoder
        )

    def process(self, request: Request) -> Request:
        Processor.process(self, request)

    @log_info
    def compile(
        self,
        compile_specs: Dict[str, List[CompileSpec]],
        pte_filenames: Dict[str, str],
        skip_quantize: Dict[str, bool],
    ):
        compile_request = Request(
            inspect.currentframe().f_code.co_name,
            {
                m: Request.Data(
                    compile_spec=compile_specs[m],
                    pte_filename=pte_filenames[m],
                    skip_quantize=skip_quantize[m] if m in skip_quantize else False,
                )
                for m in self._modalities
            },
        )
        self.process(compile_request)

    @log_info
    def quantize(
        self,
        tokenizer_wrapper: TokenizerWrapper,
        skip_quantize: Dict[str, bool],
        backend,
        soc_model,
    ):
        data_config = DataConfig.from_args(self.control_args)
        dataset_builder = DatasetBuilder(
            data_config=data_config,
            llm_config=self.config,
            tokenizer_wrapper=tokenizer_wrapper,
            attn_mask=self.text_decoder.calibration_prefill.attn_mask,
        )
        if self.control_args.qat:
            calib_loader, train_loader, val_loader = (
                dataset_builder.build_qat_dataloaders()
            )
        else:
            calib_loader = dataset_builder.build_calib_dataloaders()
            train_loader = dict.fromkeys(calib_loader)
            val_loader = dict.fromkeys(calib_loader)

        quantize_request = Request(
            inspect.currentframe().f_code.co_name,
            {
                m: Request.Data(
                    quantization_data=Request.QuantizationData(
                        calib_loader=calib_loader[m],
                        train_loader=train_loader[m],
                        val_loader=val_loader[m],
                    ),
                    skip_quantize=skip_quantize.get(m, False),
                    tokenizer=tokenizer_wrapper.tokenizer,
                    backend=backend,
                    soc_model=soc_model,
                )
                for m in self._modalities
            },
        )
        self.process(quantize_request)
        quantize_request = None
        _release_memory()
