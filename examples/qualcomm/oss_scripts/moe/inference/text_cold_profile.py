"""Optional SM8850 startup-cache assets from independent natural texts."""

import hashlib
import json
from pathlib import Path
import struct

TABLE_NAME = "cold-experts-text.i32"
MANIFEST_NAME = "cold-experts-text.json"
DEFAULT_PROFILE = (
    Path(__file__).resolve().parents[1] / "configs/qwen35_35b_a3b/sm8850_text_cold.json"
)
SOURCE_IDENTITIES = (
    (
        "text0_zh_kitchen.txt",
        "fb7a83d4b37085af47e3aab4f3d063b0f43fcbf1fecc1c01e53c107b41dcd4d3",
        "a32a82e5ac23c47d1ddaad02659b4265f41e1cbcaee5939a83c7c1d774b0b71f",
    ),
    (
        "text1_en_garden.txt",
        "ea0b135822814ba12bde26886f6474672221834f61fbf5c40f45ebcf006b1aa9",
        "7d37fcd7bd7b6be999280ea338d8206d683a4468225fcbd75f4d71bb5518b3cc",
    ),
    (
        "text2_zh_geometry.txt",
        "376a9dad54abec96e44c417cc09b21509a934045c4c7b7e347701c09c087bb75",
        "7f1ea397e5a30f2ec1b9bc119823e795f8c2cb011d3cb8510ed8913383eecc8f",
    ),
)
STRATEGIES = ("token_frequency", "sample_presence_then_token_frequency")


def validate_metadata(metadata):
    sources = metadata.get("sources", [])
    if (
        metadata.get("version") != 1
        or metadata.get("target") != "SM8850"
        or metadata.get("modality") != "text"
        or metadata.get("shape") != [40, 64]
        or metadata.get("num_experts") != 256
        or metadata.get("route_shape") != [2, 40, 128, 8]
        or metadata.get("benchmark_excluded") is not True
        or metadata.get("independent_heldout_excluded") is not True
        or tuple(
            (source["filename"], source["sha256"], source["prompt_utf8_sha256"])
            for source in sources
        )
        != SOURCE_IDENTITIES
        or any(
            (
                source.get("valid_tokens"),
                source.get("extra_space_tokens"),
                source.get("inactive_padding_tokens"),
            )
            != (256, 0, 0)
            for source in sources
        )
    ):
        raise ValueError("Invalid independent-natural-text cold profile provenance")
    model = metadata.get("model", {})
    if (
        model.get("layers"),
        model.get("hidden_size"),
        model.get("top_k"),
        model.get("expert_bits"),
    ) != (40, 2048, 8, 2):
        raise ValueError("Text cold profile requires the 40-layer W2/Top8 model")
    loo = metadata.get("loo", {})
    if set(loo) != set(STRATEGIES):
        raise ValueError("Text cold profile requires both predefined LOO strategies")
    for strategy in STRATEGIES:
        folds = loo[strategy]["fold_unique_hits"]
        if (
            len(folds) != 3
            or any(type(value) is not int or not 0 <= value <= 2560 for value in folds)
            or sum(folds) != loo[strategy]["total_unique_hits"]
        ):
            raise ValueError("Invalid text cold LOO totals")
    winner = max(STRATEGIES, key=lambda strategy: loo[strategy]["total_unique_hits"])
    if metadata.get("selected_strategy") != winner:
        raise ValueError("Text cold profile does not match its 64-slot LOO selection")


def validate_table(data, metadata):
    validate_metadata(metadata)
    if len(data) != 40 * 64 * 4 or hashlib.sha256(data).hexdigest() != metadata.get(
        "table_sha256"
    ):
        raise ValueError("Text cold table size/checksum mismatch")
    ids = struct.unpack("<2560i", data)
    for layer in range(40):
        row = ids[layer * 64 : (layer + 1) * 64]
        if len(set(row)) != 64 or any(expert < 0 or expert >= 256 for expert in row):
            raise ValueError(
                "Text cold table requires 64 distinct valid experts per layer"
            )


def load_profile(path=DEFAULT_PROFILE):
    profile = json.loads(Path(path).read_text())
    rows = profile.pop("experts")
    if len(rows) != 40 or any(len(row) != 64 for row in rows):
        raise ValueError("Text cold profile requires 40x64 ranked expert IDs")
    for source in profile["sources"]:
        content = source.pop("content")
        if (
            not content.endswith("\n")
            or hashlib.sha256(content.encode()).hexdigest() != source["sha256"]
            or hashlib.sha256(content[:-1].encode()).hexdigest()
            != source["prompt_utf8_sha256"]
        ):
            raise ValueError("Independent natural calibration text content changed")
    data = struct.pack("<2560i", *(expert for row in rows for expert in row))
    validate_table(data, profile)
    return data, profile


def text_cold_assets(root, soc_model, sidecar_root=None):
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
            raise ValueError("Text cold startup requires both table and manifest")
        validate_table(table.read_bytes(), json.loads(manifest.read_text()))
        return table, manifest
    return ()


def materialize_text_cold_profile(root, profile=DEFAULT_PROFILE, *, overwrite=False):
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
            raise FileExistsError("Preserve existing text cold asset: " + str(target))
    for name, content in contents.items():
        target = root / name
        if target.exists() and target.read_bytes() == content:
            continue
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_bytes(content)
        temporary.replace(target)
    return root / TABLE_NAME, root / MANIFEST_NAME


def ensure_text_cold_profile(root):
    existing = text_cold_assets(root, "SM8850")
    if existing:
        data, _ = load_profile()
        if existing[0].read_bytes() != data:
            raise ValueError(
                "Text cold assets differ from the current repository profile; materialize them again explicitly"
            )
        return existing
    return materialize_text_cold_profile(root)
