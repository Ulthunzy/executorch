# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from types import SimpleNamespace

import pytest
import torch
from datasets import IterableDataset
from executorch.examples.qualcomm.oss_scripts.llama.dataset import loaders
from executorch.examples.qualcomm.oss_scripts.llama.dataset.builders import (
    DatasetBuilder,
)
from executorch.examples.qualcomm.oss_scripts.llama.dataset.config import DataConfig
from executorch.examples.qualcomm.oss_scripts.llama.dataset.datasets import LLMDataset
from executorch.examples.qualcomm.oss_scripts.llama.decoder_constants import (
    TEXT_DECODER,
)
from executorch.examples.qualcomm.oss_scripts.llama.llama import (
    _build_parser,
    export_llama,
)
from executorch.examples.qualcomm.oss_scripts.llama.masking_utils import (
    AttentionMask,
    CausalAttentionMask,
)


class TextTokenizer:
    eos_id = 0

    def encode(self, text, *, bos, eos):
        assert not bos and not eos
        return list(map(ord, text))


@pytest.fixture
def hub_sources(monkeypatch):
    calls = []

    def load(repo, **kwargs):
        calls.append((repo, kwargs))
        text = "中文校准" if repo == "BAAI/CCI3-HQ" else "English calibration"
        return IterableDataset.from_generator(
            lambda: ({"text": text} for _ in range(30))
        )

    monkeypatch.setattr(loaders, "hf_load_dataset", load)
    return calls


def test_mixed_sources_stream_train_split(hub_sources):
    sequences = loaders.load_text_calibration_datasets(
        ["cci3-hq", "wikitext-2"], [3, 2], TextTokenizer(), 8, seed=17
    )
    assert len(sequences) == 5
    assert all(len(seq) == 8 for seq in sequences)
    assert sum(any(token > 127 for token in seq) for seq in sequences) == 3
    assert hub_sources == [
        ("BAAI/CCI3-HQ", {"name": None, "split": "train", "streaming": True}),
        (
            "Salesforce/wikitext",
            {"name": "wikitext-2-raw-v1", "split": "train", "streaming": True},
        ),
    ]
    assert sequences == loaders.load_text_calibration_datasets(
        ["cci3-hq", "wikitext-2"], [3, 2], TextTokenizer(), 8, seed=17
    )


def test_broadcast_sample_count(hub_sources):
    result = loaders.load_text_calibration_datasets(
        ["cci3-hq", "wikitext-2"], [2], TextTokenizer(), 8
    )
    assert len(result) == 4


@pytest.mark.parametrize("counts", [[], [0], [-1], [1, 2, 3]])
def test_invalid_counts_before_network(counts, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Invalid arguments must fail before contacting the Hub")

    monkeypatch.setattr(loaders, "hf_load_dataset", unexpected)
    with pytest.raises(ValueError):
        loaders.load_text_calibration_datasets(
            ["cci3-hq", "wikitext-2"], counts, TextTokenizer(), 8
        )


def test_local_jsonl_empty_rows_and_eos(tmp_path, caplog):
    path = tmp_path / "corpus.jsonl"
    path.write_text("\n".join(json.dumps({"text": s}) for s in ["", None, "ab", "cd"]))
    result = loaders.load_text_calibration_datasets(
        [str(path)], [3], TextTokenizer(), 8
    )
    assert len(result) == 1
    assert sorted(result[0]) == [0, 0, 97, 98, 99, 100]
    assert "only 1/3" in caplog.text


def test_chat_dataset_not_silently_accepted(monkeypatch):
    monkeypatch.setattr(
        loaders,
        "hf_load_dataset",
        lambda *a, **kw: IterableDataset.from_generator(
            lambda: iter([{"messages": []}])
        ),
    )
    with pytest.raises(ValueError, match="no usable calibration text"):
        loaders.load_text_calibration_datasets(["cci3-hq"], [1], TextTokenizer(), 8)


def test_gated_error_not_silently_dropping_source(monkeypatch):
    def denied(*args, **kwargs):
        raise PermissionError("dataset access requires authorization")

    monkeypatch.setattr(loaders, "hf_load_dataset", denied)
    with pytest.raises(PermissionError, match="authorization"):
        loaders.load_text_calibration_datasets(["cci3-hq"], [1], TextTokenizer(), 8)


def test_raw_text_only_calibration_dataloader(hub_sources):
    args = _build_parser().parse_args(
        [
            "--decoder_model",
            "lincal-1_7b",
            "--prompt",
            "hello",
            "--max_context_len",
            "8",
            "--calib_text_datasets",
            "cci3-hq",
            "wikitext-2",
            "--calib_text_samples",
            "2",
            "1",
            "--seed",
            "17",
        ]
    )
    cfg = DataConfig.from_args(args)
    assert cfg.seed == 17
    builder = DatasetBuilder(
        cfg,
        SimpleNamespace(),
        SimpleNamespace(tokenizer=TextTokenizer()),
        AttentionMask([CausalAttentionMask(1, 8, 8)]),
    )
    batches = list(builder.build_calib_dataloaders()[TEXT_DECODER])
    assert len(batches) == 3
    assert all(batch["input_ids"].shape == (1, 8) for batch in batches)
    assert all(batch["input_ids"].dtype == torch.int32 for batch in batches)


def test_cli_invalid_sample_count():
    args = _build_parser().parse_args(
        [
            "--decoder_model",
            "lincal-1_7b",
            "--prompt",
            "hello",
            "--calib_text_datasets",
            "cci3-hq",
            "--calib_text_samples",
            "0",
        ]
    )
    with pytest.raises(ValueError, match="positive counts"):
        export_llama(args)


def test_raw_text_qat_calibration(hub_sources, monkeypatch):
    cfg = DataConfig(
        max_context_len=8,
        token_dtype=torch.int32,
        calib_text_datasets=["cci3-hq", "wikitext-2"],
        calib_text_samples=[1],
        train_tasks=["test-task"],
    )
    builder = DatasetBuilder(
        cfg,
        SimpleNamespace(),
        SimpleNamespace(tokenizer=TextTokenizer()),
        AttentionMask([CausalAttentionMask(1, 8, 8)]),
    )
    monkeypatch.setattr(
        builder._text_decoder_dataset_builder,
        "from_lm_eval",
        lambda *args: LLMDataset([[1, 2, 3, 4]]),
    )
    calib, train, val = builder.build_qat_dataloaders()
    assert len(list(calib[TEXT_DECODER])) == 2
    assert len(list(train[TEXT_DECODER])) == 1
    assert val[TEXT_DECODER] is None


def test_raw_text_rejects_multimodal():
    with pytest.raises(ValueError, match="text-only"):
        DatasetBuilder(
            DataConfig(max_context_len=8, calib_text_datasets=["cci3-hq"]),
            SimpleNamespace(vision_encoder=object()),
            SimpleNamespace(tokenizer=TextTokenizer()),
            AttentionMask([CausalAttentionMask(1, 8, 8)]),
        )
