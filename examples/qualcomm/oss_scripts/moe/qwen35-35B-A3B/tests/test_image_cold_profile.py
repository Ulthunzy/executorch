import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from workflow.image_cold_profile import (  # noqa: E402
    DEFAULT_PROFILE,
    MANIFEST_NAME,
    SOURCE_IDENTITIES,
    TABLE_NAME,
    ensure_image_cold_profile,
    image_cold_assets,
    load_profile,
    materialize_image_cold_profile,
    validate_table,
)
from workflow.model_package import PackageReader, create_package  # noqa: E402


def test_profile_materializes_without_touching_old_model_or_text_table(tmp_path):
    old = tmp_path / "model.pte"
    text = tmp_path / "cold-experts.i32"
    old.write_bytes(b"original model")
    text.write_bytes(b"original text profile")
    assets = materialize_image_cold_profile(tmp_path)
    assert assets[0].stat().st_size == 10240
    assert (
        hashlib.sha256(assets[0].read_bytes()).hexdigest()
        == "cd884a2a8198198654bb8fae35126508a37b66071d0ffb0d64834d21021e639a"
    )
    assert assets[1].stat().st_size < 4096
    manifest = json.loads(assets[1].read_text())
    assert "experts" not in manifest and "collection" not in manifest
    assert manifest["loo"]["token_frequency"]["total_unique_hits"] == 7366
    assert (
        manifest["loo"]["image_presence_then_token_frequency"]["total_unique_hits"]
        == 7409
    )
    assert ensure_image_cold_profile(tmp_path) == assets
    assert old.read_bytes() == b"original model"
    assert text.read_bytes() == b"original text profile"


def test_optional_assets_support_old_package_sidecars_and_other_soc_fallback(tmp_path):
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    assert image_cold_assets(extracted, "SM8850") == ()
    assets = materialize_image_cold_profile(tmp_path)
    assert image_cold_assets(extracted, "SM8850", tmp_path) == assets
    assert image_cold_assets(extracted, "SM8750", tmp_path) == ()
    assert image_cold_assets(extracted, "SM8975", tmp_path) == ()
    assets[1].unlink()
    with pytest.raises(ValueError, match="both table and manifest"):
        image_cold_assets(extracted, "SM8850", tmp_path)
    assert image_cold_assets(extracted, "SM8750", tmp_path) == ()


def test_profile_rejects_corrupt_table_and_unapproved_calibration_source(tmp_path):
    data, metadata = load_profile()
    with pytest.raises(ValueError, match="checksum"):
        validate_table(b"bad" + data[3:], metadata)
    wrong = json.loads(json.dumps(metadata))
    wrong["sources"][0]["path"] = "test.jpg"
    with pytest.raises(ValueError, match="provenance"):
        validate_table(data, wrong)
    ids = list(struct.unpack("<2560i", data))
    ids[0] = ids[1]
    duplicate = struct.pack("<2560i", *ids)
    wrong = dict(metadata, table_sha256=hashlib.sha256(duplicate).hexdigest())
    with pytest.raises(ValueError, match="distinct"):
        validate_table(duplicate, wrong)
    assets = materialize_image_cold_profile(tmp_path)
    changed_ids = [(expert + 1) % 256 for expert in struct.unpack("<2560i", data)]
    changed_data = struct.pack("<2560i", *changed_ids)
    wrong = dict(metadata, table_sha256=hashlib.sha256(changed_data).hexdigest())
    assets[0].write_bytes(changed_data)
    assets[1].write_text(json.dumps(wrong))
    with pytest.raises(ValueError, match="current repository profile"):
        ensure_image_cold_profile(tmp_path)


def test_rebuilt_provenance_for_identical_table_can_be_repacked(tmp_path):
    profile = json.loads(DEFAULT_PROFILE.read_text())
    profile.pop("calibration_evidence_sha256")
    profile["calibration_collection_sha256"] = "1" * 64
    source = tmp_path / "rebuilt-profile.json"
    source.write_text(json.dumps(profile))
    assets = materialize_image_cold_profile(tmp_path / "assets", source)
    assert ensure_image_cold_profile(tmp_path / "assets") == assets
    package = tmp_path / "rebuilt.pte"
    create_package(package, {path.name: path for path in assets})
    extracted = PackageReader(package).extract(tmp_path / "cache")
    assert ensure_image_cold_profile(extracted)


@pytest.mark.parametrize("arch", [None, 79, 81, 85])
def test_actual_image_selection_and_legacy_architecture_fallback(tmp_path, arch):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    command = [
        compiler,
        "-std=c++17",
        str(ROOT / "tests/cold_start_test.cpp"),
        "-o",
        str(tmp_path / "selection"),
    ]
    if arch is not None:
        command.append(f"-DSM_MOE_DSP_ARCH={arch}")
    subprocess.run(command, check=True, capture_output=True)
    subprocess.run(
        [str(tmp_path / "selection"), str(tmp_path / "assets")],
        check=True,
        capture_output=True,
    )


def test_pack_export_embeds_optional_assets_and_native_asset_links(
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
    mini = tmp_path / "mini.pte"
    create_package(
        mini,
        {name: captured["files"][name] for name in (TABLE_NAME, MANIFEST_NAME)},
        {
            "model/prefill/" + name: "../../" + name
            for name in (TABLE_NAME, MANIFEST_NAME)
        },
    )
    reader = PackageReader(mini)
    extracted = reader.extract(tmp_path / "cache")
    assert (extracted / "model/prefill" / TABLE_NAME).read_bytes() == (
        tmp_path / TABLE_NAME
    ).read_bytes()
    assert image_cold_assets(extracted, "SM8850")
    config.model_settings.target.soc_model = "SM8750"
    package.pack_export(config)
    assert TABLE_NAME not in captured["files"]


def test_rebuild_uses_chunk_layer_layout_and_deterministic_loo_tie(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "rebuild_image_cold", ROOT / "tools/image_cold_profile.py"
    )
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    runs = []
    for index, (relative, digest) in enumerate(SOURCE_IDENTITIES):
        values = []
        for chunk in range(2):
            for layer in range(40):
                for token in range(128):
                    values.extend(
                        (layer * 80 + index * 8 + rank) % 256 for rank in range(8)
                    )
        directory = tmp_path / "collection" / f"image{index}"
        directory.mkdir(parents=True)
        data = struct.pack("<81920i", *values)
        (directory / "prefill-all-routes.i32").write_bytes(data)
        runs.append(
            {
                "label": f"image{index}",
                "source": {"sha256": digest, "repository_path": relative},
                "route_sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    collection = tmp_path / "COLLECTION.json"
    collection.write_text(json.dumps({"runs": runs}))
    profile = tool.rebuild_profile(collection)
    assert profile["selected_strategy"] == "token_frequency"
    assert profile["experts"][0] == list(range(64))
    assert profile["experts"][1] == list(range(80, 104)) + list(range(40))
    assert profile["benchmark_excluded"]
    bad = runs.copy()
    bad[0] = dict(bad[0], source={"sha256": "benchmark", "repository_path": "test.jpg"})
    collection.write_text(json.dumps({"runs": bad}))
    with pytest.raises(ValueError, match="approved images"):
        tool.rebuild_profile(collection)


def test_old_pte_adjacent_sidecars_survive_python_extraction(tmp_path):
    from workflow.config import parse_args

    package = tmp_path / "old-model.pte"
    create_package(package, {"MODEL_CONFIG.json": ROOT / "configs/model_sm8850.json"})
    original_sha = hashlib.sha256(package.read_bytes()).hexdigest()
    assets = materialize_image_cold_profile(tmp_path)
    args = parse_args(
        ["--config", str(ROOT / "configs/sm8850.json"), "--weights-path", str(package)]
    )
    assert args.artifacts != tmp_path
    assert args.image_cold_profile_dir == tmp_path
    assert (
        image_cold_assets(args.artifacts, "SM8850", args.image_cold_profile_dir)
        == assets
    )
    assert hashlib.sha256(package.read_bytes()).hexdigest() == original_sha


def test_deployment_unlinks_optional_targets_before_push_and_diagnostic_uses_actual_image(
    tmp_path, monkeypatch
):
    from workflow.config import parse_args
    from workflow.device import DevicePaths
    import workflow.deployment as deployment

    assets = materialize_image_cold_profile(tmp_path)
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    args = parse_args(["--config", str(ROOT / "configs/sm8850.json"), "--with-vision"])
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
    table_push = calls.index(("push", str(assets[0]), paths.prefill + "/" + TABLE_NAME))
    manifest_push = calls.index(
        ("push", str(assets[1]), paths.prefill + "/" + MANIFEST_NAME)
    )
    removals = [
        index
        for index, call in enumerate(calls)
        if call[0] == "shell"
        and "rm -f" in call[1]
        and TABLE_NAME in call[1]
        and MANIFEST_NAME in call[1]
    ]
    assert len(removals) == 1 and removals[0] < min(table_push, manifest_push)
    text = deployment.build_command(args, paths, {"eos_token_id": [248046]}, None)
    image = deployment.build_command(args, paths, {"eos_token_id": [248046]}, object())
    assert "PREFILL_INITIAL_IMAGE=0" in text
    assert "PREFILL_INITIAL_IMAGE=1" in image
