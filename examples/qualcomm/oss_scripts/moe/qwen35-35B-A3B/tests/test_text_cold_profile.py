import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from workflow.image_cold_profile import (  # noqa: E402
    image_cold_assets,
    materialize_image_cold_profile,
)
from workflow.model_package import PackageReader, create_package  # noqa: E402
from workflow.text_cold_profile import (  # noqa: E402
    DEFAULT_PROFILE,
    MANIFEST_NAME,
    TABLE_NAME,
    ensure_text_cold_profile,
    load_profile,
    materialize_text_cold_profile,
    text_cold_assets,
    validate_table,
)


def test_materialize_preserves_original_model_table_and_image_profile(tmp_path):
    model = tmp_path / "model.pte"
    original = tmp_path / "cold-experts.i32"
    model.write_bytes(b"old model")
    original.write_bytes(b"old cold table")
    image = materialize_image_cold_profile(tmp_path)
    before_image = [path.read_bytes() for path in image]
    table, manifest = materialize_text_cold_profile(tmp_path)
    assert len(table.read_bytes()) == 10240
    assert hashlib.sha256(table.read_bytes()).hexdigest() == (
        "a737c9bf3ae84ccd12c70cd467f4b042f24b95f35308b28f334d18d1d0220a83"
    )
    metadata = json.loads(manifest.read_text())
    assert manifest.stat().st_size < 4096
    assert metadata["loo"]["token_frequency"]["total_unique_hits"] == 6655
    assert (
        metadata["loo"]["sample_presence_then_token_frequency"]["total_unique_hits"]
        == 6985
    )
    assert all("content" not in source for source in metadata["sources"])
    assert ensure_text_cold_profile(tmp_path) == (table, manifest)
    assert model.read_bytes() == b"old model"
    assert original.read_bytes() == b"old cold table"
    assert [path.read_bytes() for path in image] == before_image


def test_old_package_sidecars_and_architecture_fallback(tmp_path):
    from workflow.config import parse_args

    package = tmp_path / "old-model.pte"
    create_package(package, {"MODEL_CONFIG.json": ROOT / "configs/model_sm8850.json"})
    before = package.read_bytes()
    assets = materialize_text_cold_profile(tmp_path)
    args = parse_args(
        ["--config", str(ROOT / "configs/sm8850.json"), "--weights-path", str(package)]
    )
    assert args.artifacts != tmp_path
    assert (
        text_cold_assets(args.artifacts, "SM8850", args.image_cold_profile_dir)
        == assets
    )
    for soc in ("SM8750", "SM8975"):
        assert text_cold_assets(args.artifacts, soc, tmp_path) == ()
    assets[1].unlink()
    with pytest.raises(ValueError, match="both table and manifest"):
        text_cold_assets(args.artifacts, "SM8850", tmp_path)
    assert text_cold_assets(args.artifacts, "SM8750", tmp_path) == ()
    assert package.read_bytes() == before


def test_source_content_sha_zero_padding_and_loo_winner_are_validated(tmp_path):
    profile = json.loads(DEFAULT_PROFILE.read_text())
    changed = tmp_path / "profile.json"
    profile["sources"][0]["content"] += "benchmark"
    changed.write_text(json.dumps(profile))
    with pytest.raises(ValueError, match="content changed"):
        load_profile(changed)
    data, metadata = load_profile()
    for field, value in (("extra_space_tokens", 1), ("sha256", "0" * 64)):
        invalid = json.loads(json.dumps(metadata))
        invalid["sources"][0][field] = value
        with pytest.raises(ValueError, match="provenance"):
            validate_table(data, invalid)
    invalid = dict(metadata, selected_strategy="token_frequency")
    with pytest.raises(ValueError, match="LOO selection"):
        validate_table(data, invalid)


def test_corruption_duplicate_ids_and_stale_table_are_rejected(tmp_path):
    data, metadata = load_profile()
    with pytest.raises(ValueError, match="checksum"):
        validate_table(data[:-4], metadata)
    ids = list(struct.unpack("<2560i", data))
    ids[0] = ids[1]
    duplicate = struct.pack("<2560i", *ids)
    invalid = dict(metadata, table_sha256=hashlib.sha256(duplicate).hexdigest())
    with pytest.raises(ValueError, match="distinct"):
        validate_table(duplicate, invalid)
    assets = materialize_text_cold_profile(tmp_path)
    different = struct.pack(
        "<2560i", *((value + 1) % 256 for value in struct.unpack("<2560i", data))
    )
    assets[0].write_bytes(different)
    assets[1].write_text(
        json.dumps(dict(metadata, table_sha256=hashlib.sha256(different).hexdigest()))
    )
    with pytest.raises(ValueError, match="current repository profile"):
        ensure_text_cold_profile(tmp_path)


def test_pack_export_adds_text_pair_and_native_links_without_changing_image(
    tmp_path, monkeypatch
):
    import workflow.model_package as package

    for group in (*[f"group{i}" for i in range(10)], "head"):
        (tmp_path / "shared-model" / group).mkdir(parents=True)
    config = SimpleNamespace(
        output=tmp_path,
        components=("text", "vision"),
        phases=(),
        model=tmp_path / "source",
        model_settings=SimpleNamespace(
            target=SimpleNamespace(soc_model="SM8850"),
            quantization=SimpleNamespace(embedding="int4_g32"),
        ),
    )
    captured = {}

    def capture(destination, files, links, plans):
        captured.update(files=files, links=links)
        return captured

    monkeypatch.setattr(package, "create_package", capture)
    package.pack_export(config)
    for name in (TABLE_NAME, MANIFEST_NAME):
        assert name in captured["files"]
        assert captured["links"]["model/prefill/" + name] == "../../" + name
    assert "cold-experts-image.i32" in captured["files"]
    assert image_cold_assets(tmp_path, "SM8850")
    selected = (TABLE_NAME, MANIFEST_NAME)
    mini = tmp_path / "mini.pte"
    create_package(
        mini,
        {name: captured["files"][name] for name in selected},
        {"model/prefill/" + name: "../../" + name for name in selected},
    )
    extracted = PackageReader(mini).extract(tmp_path / "cache")
    assert (extracted / "model/prefill" / TABLE_NAME).read_bytes() == (
        tmp_path / TABLE_NAME
    ).read_bytes()
    assert ensure_text_cold_profile(extracted)
    config.components = ("text",)
    package.pack_export(config)
    assert TABLE_NAME in captured["files"]
    assert "cold-experts-image.i32" not in captured["files"]
    config.model_settings.target.soc_model = "SM8750"
    package.pack_export(config)
    assert TABLE_NAME not in captured["files"]


def test_rebuilt_provenance_for_same_table_can_be_repacked(tmp_path):
    profile = json.loads(DEFAULT_PROFILE.read_text())
    profile.pop("calibration_evidence_sha256")
    profile["calibration_collection_sha256"] = "1" * 64
    rebuilt = tmp_path / "rebuilt.json"
    rebuilt.write_text(json.dumps(profile))
    assets = materialize_text_cold_profile(tmp_path / "assets", rebuilt)
    assert ensure_text_cold_profile(tmp_path / "assets") == assets


def test_deploy_cleans_text_targets_before_push_and_when_optional_absent(
    tmp_path, monkeypatch
):
    from workflow.config import parse_args
    from workflow.device import DevicePaths
    import workflow.deployment as deployment

    assets = materialize_text_cold_profile(tmp_path)
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    args = parse_args(["--config", str(ROOT / "configs/sm8850.json")])
    args.artifacts = extracted
    args.image_cold_profile_dir = tmp_path
    args.shared_model = None
    args.deploy_weights = False
    args.vision_artifacts = None
    args.local_prefill_pte = None
    args.local_decode_pte = None
    paths = DevicePaths("/remote/isolated", "/remote/shared")
    calls = []

    class Device:
        def adb(self, *items):
            calls.append(tuple(map(str, items)))
            return ""

    monkeypatch.setattr(deployment, "load_prefill_bundle", lambda args: None)
    monkeypatch.setattr(deployment, "runtime_assets", lambda args: tmp_path)
    monkeypatch.setattr(deployment, "deploy_embedding", lambda *args: None)
    deployment.deploy_artifacts(args, Device(), paths, tmp_path, None)
    removals = [
        index
        for index, call in enumerate(calls)
        if call[0] == "shell"
        and "rm -f" in call[1]
        and TABLE_NAME in call[1]
        and MANIFEST_NAME in call[1]
    ]
    pushes = [
        calls.index(("push", str(path), paths.prefill + "/" + path.name))
        for path in assets
    ]
    assert len(removals) == 1 and removals[0] < min(pushes)
    for path in assets:
        path.unlink()
    calls.clear()
    deployment.deploy_artifacts(args, Device(), paths, tmp_path, None)
    assert any(
        call[0] == "shell" and "rm -f" in call[1] and TABLE_NAME in call[1]
        for call in calls
    )
    assert not any(call[0] == "push" and TABLE_NAME in call[-1] for call in calls)


def test_rebuild_checks_input_tokens_and_chunk_layer_routes_with_deterministic_tie(
    tmp_path, monkeypatch
):
    spec = importlib.util.spec_from_file_location(
        "rebuild_text_cold", ROOT / "tools/text_cold_profile.py"
    )
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    profile = json.loads(DEFAULT_PROFILE.read_text())
    tokens = bytes(256 * 8)
    token_sha = hashlib.sha256(tokens).hexdigest()
    for source in profile["sources"]:
        source["expected_input_sha256"] = token_sha
    local_profile = tmp_path / "profile.json"
    local_profile.write_text(json.dumps(profile))
    monkeypatch.setattr(tool, "DEFAULT_PROFILE", local_profile)
    runs = []
    for index, (filename, digest, prompt_sha) in enumerate(tool.SOURCE_IDENTITIES):
        values = []
        for chunk in range(2):
            for layer in range(40):
                for token in range(128):
                    values.extend(
                        (layer * 80 + index * 8 + rank) % 256 for rank in range(8)
                    )
        directory = tmp_path / "collection" / f"text{index}"
        directory.mkdir(parents=True)
        data = struct.pack("<81920i", *values)
        (directory / "prefill-all-routes.i32").write_bytes(data)
        (directory / "tokens.i64").write_bytes(tokens)
        runs.append(
            {
                "label": f"text{index}",
                "source": {
                    "sha256": digest,
                    "filename": filename,
                    "prompt_utf8_sha256": prompt_sha,
                    "prompt": profile["sources"][index]["content"][:-1],
                    "valid_tokens": 256,
                    "real_space_tokens_added": 0,
                    "inactive_padding_tokens": 0,
                },
                "route_sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    collection = tmp_path / "COLLECTION.json"
    collection.write_text(json.dumps({"runs": runs}))
    rebuilt = tool.rebuild_profile(collection)
    assert rebuilt["selected_strategy"] == "token_frequency"
    assert rebuilt["experts"][0] == list(range(64))
    assert rebuilt["experts"][1] == list(range(80, 104)) + list(range(40))
    assert rebuilt["benchmark_excluded"] and rebuilt["independent_heldout_excluded"]
    (tmp_path / "collection/text0/tokens.i64").write_bytes(b"wrong")
    with pytest.raises(ValueError, match="checksum"):
        tool.rebuild_profile(collection)
