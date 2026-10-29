"""Check calibrated weights against their original safetensors checkpoint."""

import hashlib
import json
from pathlib import Path
from executorch.examples.qualcomm.oss_scripts.moe.checkpoint import input_stamp


def checkpoint_identity(model):
    model = Path(model)
    index = json.loads((model / "model.safetensors.index.json").read_text())
    if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict):
        raise ValueError("Invalid safetensors checkpoint index")
    paths = [model / "config.json", model / "model.safetensors.index.json"]
    paths.extend(
        model / name
        for name in (
            "tokenizer.json",
            "tokenizer_config.json",
            "generation_config.json",
        )
        if (model / name).is_file()
    )
    paths.extend(model / name for name in sorted(set(index["weight_map"].values())))
    return {str(path): input_stamp(path) for path in paths}


def validate_checkpoint_source(model, metadata):
    index = (model / "model.safetensors.index.json").read_bytes()
    if (
        not isinstance(metadata, dict)
        or metadata.get("index_sha256") != hashlib.sha256(index).hexdigest()
    ):
        raise ValueError("Checkpoint index changed")
    shards = metadata.get("safetensors", {})
    if not isinstance(shards, dict):
        raise ValueError("Checkpoint shard provenance is incomplete")
    if set(shards) != set(json.loads(index)["weight_map"].values()):
        raise ValueError("Checkpoint shard provenance is incomplete")
    for name, recorded in shards.items():
        stat = (model / name).stat()
        if (stat.st_size, stat.st_mtime_ns) != (
            recorded["bytes"],
            recorded["mtime_ns"],
        ):
            raise ValueError(f"Checkpoint shard changed: {name}")
