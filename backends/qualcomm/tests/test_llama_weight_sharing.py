# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import Mock

import pytest
from executorch.backends.qualcomm.serialization.qc_schema_serialize import (
    flatbuffer_to_option,
)
from executorch.examples.qualcomm.oss_scripts.llama import (
    Lincal_1_7B,
    llama,
    LLMModelConfig,
    SUPPORTED_LLM_MODELS,
)
from executorch.examples.qualcomm.oss_scripts.llama.decoder_constants import (
    AUDIO_ENCODER,
    DECODER_GRAPH_NAMES,
    TEXT_DECODER,
    TEXT_ENCODER,
    TOK_EMBEDDING,
    TOK_EMBEDDING_GRAPH_NAMES,
    VISION_ENCODER,
)


class NoWeightSharingLincal(type(Lincal_1_7B)):
    use_weight_sharing = False


def test_model_config_enables_weight_sharing_by_default():
    assert LLMModelConfig.use_weight_sharing is True
    assert SUPPORTED_LLM_MODELS["lincal-1_7b"].use_weight_sharing is True
    assert all(
        isinstance(config.use_weight_sharing, bool)
        for config in SUPPORTED_LLM_MODELS.values()
    )
    assert "use_weight_sharing" in str(SUPPORTED_LLM_MODELS["lincal-1_7b"])


@pytest.mark.parametrize(
    "config", [Lincal_1_7B, NoWeightSharingLincal()], ids=["enabled", "disabled"]
)
@pytest.mark.parametrize("model_mode", ["kv", "hybrid", "lookahead"])
@pytest.mark.parametrize("is_multimodal", [False, True])
def test_model_weight_sharing_reaches_compile_specs(
    config, model_mode, is_multimodal, monkeypatch, tmp_path
):
    args = llama._build_parser().parse_args(
        [
            "--decoder_model",
            "lincal-1_7b",
            "--soc_model",
            "SM8750",
            "--backend",
            "htp",
            "--prompt",
            "hello",
            "--model_mode",
            model_mode,
            "--prefill_ar_len",
            "4",
            "--artifact",
            str(tmp_path),
        ]
    )
    args.use_fp16 = False
    manager = Mock()
    monkeypatch.setattr(llama, "MultiModalManager", Mock(return_value=manager))

    llama.compile(args, config, {}, None, is_multimodal)

    specs = manager.compile.call_args.kwargs["compile_specs"]
    assert len(specs[TEXT_DECODER]) == len(DECODER_GRAPH_NAMES)
    components = [TEXT_DECODER]
    if is_multimodal:
        assert len(specs[TOK_EMBEDDING]) == len(TOK_EMBEDDING_GRAPH_NAMES)
        components.append(TOK_EMBEDDING)
        for modality in (AUDIO_ENCODER, TEXT_ENCODER, VISION_ENCODER):
            options = flatbuffer_to_option(specs[modality][0].value)
            assert options.backend_options.htp_options.use_weight_sharing is False
    else:
        assert specs[TOK_EMBEDDING] is None
    for modality in components:
        for graph_specs in specs[modality]:
            options = flatbuffer_to_option(graph_specs[0].value)
            assert (
                options.backend_options.htp_options.use_weight_sharing
                is config.use_weight_sharing
            )
