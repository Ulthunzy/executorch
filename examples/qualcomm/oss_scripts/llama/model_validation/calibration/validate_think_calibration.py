# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Verify a curated bundle through the actual ExecuTorch calibration loader."""

import argparse
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from build_think_calibration import file_hash, format_chatml, read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--max-context-len", type=int, default=4096)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise ValueError("Choose a new validation output filename")

    import torch
    from pytorch_tokenizers import get_tokenizer
    from transformers import AutoTokenizer

    from executorch.examples.qualcomm.oss_scripts.llama.dataset.builders import (
        DatasetBuilder,
    )
    from executorch.examples.qualcomm.oss_scripts.llama.dataset.config import DataConfig
    from executorch.examples.qualcomm.oss_scripts.llama.dataset.loaders import (
        load_text_calibration_datasets,
    )
    from executorch.examples.qualcomm.oss_scripts.llama.decoder_constants import (
        TEXT_DECODER,
    )
    from executorch.examples.qualcomm.oss_scripts.llama.masking_utils import (
        AttentionMask,
        CausalAttentionMask,
    )

    torch.set_num_threads(4)
    manifest = read_json(args.data_dir / "manifest.json")
    for name, expected_hash in manifest["source_files"].items():
        if file_hash(args.data_dir / name) != expected_hash:
            raise ValueError(f"Bundle file changed: {name}")
    if file_hash(args.tokenizer / "tokenizer.json") != manifest["tokenizer_sha256"]:
        raise ValueError("Bundle was prepared with a different tokenizer")
    tokenizer = get_tokenizer(
        str(args.tokenizer / "tokenizer.json"),
        str(args.tokenizer / "tokenizer_config.json"),
    )
    hf = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True, trust_remote_code=False
    )
    path = args.data_dir / "calibration.chatml.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    messages = read_json(args.data_dir / "messages.json")
    assert len(rows) == len(messages) == len({row["id"] for row in rows})
    expected = []
    for row, sample in zip(rows, messages):
        assert row["text"] == format_chatml(sample["messages"]), row["id"]
        ids = tokenizer.encode(row["text"], bos=False, eos=False)
        assert ids == hf.encode(row["text"], add_special_tokens=False), row["id"]
        expected.append(ids)
    loaded = load_text_calibration_datasets(
        [str(path)],
        [len(rows)],
        tokenizer,
        args.max_context_len,
        seed=42,
        preserve_sources=[str(path)],
    )
    assert Counter(map(tuple, loaded)) == Counter(map(tuple, expected))
    counts = {}
    for marker in ("<think>", "</think>", "<|im_end|>"):
        ids = tokenizer.encode(marker, bos=False, eos=False)
        assert len(ids) == 1, marker
        counts[marker] = sum(seq.count(ids[0]) for seq in loaded)
        assert counts[marker] == sum(row["text"].count(marker) for row in rows)
    assert counts["<think>"] == counts["</think>"]

    builder = DatasetBuilder(
        DataConfig(
            max_context_len=args.max_context_len,
            token_dtype=torch.int32,
            calib_text_datasets=[str(path)],
            calib_text_samples=[len(rows)],
            calib_text_preserve_sources=[str(path)],
        ),
        SimpleNamespace(),
        SimpleNamespace(tokenizer=tokenizer),
        AttentionMask(
            [CausalAttentionMask(1, args.max_context_len, args.max_context_len)]
        ),
    )
    loader = builder.build_calib_dataloaders()[TEXT_DECODER]
    assert len(loader) == len(rows)
    probes = {0, max(range(len(loaded)), key=lambda i: len(loaded[i]))}
    probes.add(
        next(
            i
            for i, seq in enumerate(loaded)
            if seq.count(hf.convert_tokens_to_ids("<think>")) == 2
        )
    )
    for index in probes:
        batch = loader.collate_fn([loader.dataset[index]])
        ids = batch["token_ids"][0]
        assert ids == loaded[index]
        assert batch["input_ids"][0, : len(ids)].tolist() == ids
        assert batch["input_ids"].shape == (1, args.max_context_len)
        assert batch["labels"] is None
    result = {
        "status": "passed",
        "conversations": len(rows),
        "tokens": sum(map(len, loaded)),
        "max_conversation_tokens": max(map(len, loaded)),
        "special_token_counts": counts,
        "all_rows_token_exact": True,
        "hf_and_export_tokenizers_match": True,
        "collator_probes": sorted(probes),
        "max_context_len": args.max_context_len,
        "calibration_data_sha256": file_hash(path),
        "loader_sha256": file_hash(Path(__file__).resolve().parents[2] / "dataset/loaders.py"),
        "note": "Validated data loading/collation only; no model calibration, export or device inference was performed.",
    }
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
