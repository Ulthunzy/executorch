"""Optional SM8850 image startup-cache assets; no model weights are changed."""

import hashlib
import json
from pathlib import Path
import struct

TABLE_NAME = "cold-experts-image.i32"
MANIFEST_NAME = "cold-experts-image.json"
DEFAULT_PROFILE = Path(__file__).resolve().parents[1] / "configs/sm8850_image_cold.json"
SOURCE_IDENTITIES = (
    (
        "examples/models/dinov2/dog.jpg",
        "f3f87bb8ab3c26c7ecfd3ac60421d7f32b0503d1d6c5baf8bac42ed93d86351a",
    ),
    (
        "third-party/ao/examples/sam2_amg_server/dog.jpg",
        "bf76876b90e3ebd521f9882b9177ba8f33e80cb7ec09c630f179b122edd125e1",
    ),
    (
        "extension/llm/apple/ExecuTorchLLM/__tests__/resources/IMG_0005.jpg",
        "539b954ca2ba011124d8d7130bb9217ce648dd87cc055f2c04c351e78e641c04",
    ),
)
STRATEGIES = ("token_frequency", "image_presence_then_token_frequency")


def validate_metadata(metadata):
    if (
        metadata.get("version") != 1
        or metadata.get("target") != "SM8850"
        or metadata.get("shape") != [40, 64]
        or metadata.get("num_experts") != 256
        or metadata.get("route_shape") != [2, 40, 128, 8]
        or metadata.get("benchmark_excluded") is not True
        or tuple(
            (source["path"], source["sha256"]) for source in metadata.get("sources", [])
        )
        != SOURCE_IDENTITIES
    ):
        raise ValueError("Invalid independent-image cold profile provenance")
    model = metadata.get("model", {})
    if (
        model.get("layers"),
        model.get("hidden_size"),
        model.get("top_k"),
        model.get("expert_bits"),
    ) != (40, 2048, 8, 2):
        raise ValueError("Image cold profile requires the40-layer W2/Top8 model")
    loo = metadata.get("loo", {})
    if set(loo) != set(STRATEGIES):
        raise ValueError("Image cold profile requires both predefined LOO strategies")
    for strategy in STRATEGIES:
        folds = loo[strategy]["fold_unique_hits"]
        if (
            len(folds) != 3
            or any(not 0 <= value <= 2560 for value in folds)
            or sum(folds) != loo[strategy]["total_unique_hits"]
        ):
            raise ValueError("Invalid image cold LOO totals")
    winner = max(STRATEGIES, key=lambda strategy: loo[strategy]["total_unique_hits"])
    if metadata.get("selected_strategy") != winner:
        raise ValueError("Image cold profile does not match its64-slot LOO selection")


def validate_table(data, metadata):
    validate_metadata(metadata)
    if len(data) != 40 * 64 * 4 or hashlib.sha256(data).hexdigest() != metadata.get(
        "table_sha256"
    ):
        raise ValueError("Image cold table size/checksum mismatch")
    ids = struct.unpack("<2560i", data)
    for layer in range(40):
        row = ids[layer * 64 : (layer + 1) * 64]
        if len(set(row)) != 64 or any(expert < 0 or expert >= 256 for expert in row):
            raise ValueError(
                "Image cold table requires64 distinct valid experts per layer"
            )


def load_profile(path=DEFAULT_PROFILE):
    profile = json.loads(Path(path).read_text())
    rows = profile.pop("experts")
    if len(rows) != 40 or any(len(row) != 64 for row in rows):
        raise ValueError("Image cold profile requires40x64 ranked expert IDs")
    data = struct.pack("<2560i", *(expert for row in rows for expert in row))
    validate_table(data, profile)
    return data, profile


def image_cold_assets(root, soc_model, sidecar_root=None):
    if soc_model != "SM8850":
        return ()
    roots = [Path(root)]
    if sidecar_root is not None and Path(sidecar_root) != roots[0]:
        roots.append(Path(sidecar_root))
    for directory in roots:
        table, manifest = directory / TABLE_NAME, directory / MANIFEST_NAME
        if not table.exists() and not manifest.exists():
            continue
        if not table.is_file() or not manifest.is_file():
            raise ValueError("Image cold startup requires both table and manifest")
        validate_table(table.read_bytes(), json.loads(manifest.read_text()))
        return table, manifest
    return ()


def materialize_image_cold_profile(root, profile=DEFAULT_PROFILE, *, overwrite=False):
    root = Path(root)
    data, metadata = load_profile(profile)
    contents = {
        TABLE_NAME: data,
        MANIFEST_NAME: (
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
        ).encode(),
    }
    root.mkdir(parents=True, exist_ok=True)
    for name, content in contents.items():
        target = root / name
        if target.exists() and target.read_bytes() != content and not overwrite:
            raise FileExistsError("Preserve existing image cold asset: " + str(target))
    for name, content in contents.items():
        target = root / name
        if target.exists() and target.read_bytes() == content:
            continue
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_bytes(content)
        temporary.replace(target)
    return root / TABLE_NAME, root / MANIFEST_NAME


def ensure_image_cold_profile(root):
    existing = image_cold_assets(root, "SM8850")
    if existing:
        data, _ = load_profile()
        if existing[0].read_bytes() != data:
            raise ValueError(
                "Image cold assets differ from the current repository profile; materialize them again explicitly"
            )
        return existing
    return materialize_image_cold_profile(root)
