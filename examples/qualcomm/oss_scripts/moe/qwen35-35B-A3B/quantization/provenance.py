"""Check calibrated weights against their original safetensors checkpoint."""

import hashlib
import json


def validate_checkpoint_source(model, metadata):
    index = (model / "model.safetensors.index.json").read_bytes()
    if metadata.get("index_sha256") != hashlib.sha256(index).hexdigest():
        raise ValueError("Checkpoint index changed")
    shards = metadata.get("safetensors", {})
    if set(shards) != set(json.loads(index)["weight_map"].values()):
        raise ValueError("Checkpoint shard provenance is incomplete")
    for name, recorded in shards.items():
        stat = (model / name).stat()
        if (stat.st_size, stat.st_mtime_ns) != (
            recorded["bytes"],
            recorded["mtime_ns"],
        ):
            raise ValueError(f"Checkpoint shard changed: {name}")
