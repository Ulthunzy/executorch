import errno
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from workflow import reuse_vision as module  # noqa: E402


@pytest.fixture
def artifacts(tmp_path):
    model, source = tmp_path / "model", tmp_path / "source"
    model.mkdir()
    source.mkdir()
    config = dict(
        image_token_id=248056,
        vision_config=dict(
            patch_size=16, spatial_merge_size=2, in_channels=3, temporal_patch_size=2
        ),
    )
    (model / "config.json").write_text(json.dumps(config))
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            dict(
                version=1,
                target=dict(soc_model="SM8850"),
                quantization=dict(vision="w8a16_gptq", matrix_gptq_damping=0.01),
            )
        )
    )
    metadata = dict(
        version=1,
        target=dict(soc_model="SM8850", dsp_arch="v81"),
        precision="W8A16",
        image_height=448,
        image_width=448,
        image_token_id=248056,
        input_shape=[1, 784, 1536],
        output_shape=[1, 196, 2048],
        model_config_sha256=hashlib.sha256(
            (model / "config.json").read_bytes()
        ).hexdigest(),
        gptq=[
            dict(module=name, method="fixed_grid_gptq", bits=8, damping=0.01)
            for name in ("visual.pos_embed", "visual.blocks.0.attn.qkv")
        ],
    )
    (source / "VISION.json").write_text(json.dumps(metadata))
    (source / "vision.bin").write_bytes(b"synthetic-context")
    (source / "vision.pte").write_bytes(b"synthetic-pte")
    return source, model, recipe, metadata


def validate(artifacts, height=448, width=448):
    source, model, recipe, _ = artifacts
    return module.validate_vision_source(source, model, recipe, height, width)


def reuse(artifacts, output):
    source, model, recipe, _ = artifacts
    return module.reuse_vision(source, output, model, recipe, 448, 448)


def test_real_metadata_validation_and_immutable_hardlinks(artifacts, tmp_path):
    source, _, _, metadata = artifacts
    original = {name: (source / name).read_bytes() for name in module.VISION_FILES}
    assert validate(artifacts) == metadata
    output = tmp_path / "output"
    output.mkdir()
    record = reuse(artifacts, output)
    assert json.loads((output / "VISION_REUSE.json").read_text()) == record
    for name, data in original.items():
        assert (source / name).read_bytes() == (output / name).read_bytes() == data
        assert (source / name).samefile(output / name)
        stamp = record["files"][name]
        assert stamp["method"] == "hardlink"
        for key in ("source", "destination"):
            assert stamp[key]["bytes"] == len(data)
            assert stamp[key]["sha256"] == hashlib.sha256(data).hexdigest()


def test_cross_filesystem_copy2_fallback(artifacts, tmp_path, monkeypatch):
    def cross_device(*args):
        raise OSError(errno.EXDEV, "synthetic cross-device link")

    monkeypatch.setattr(module.os, "link", cross_device)
    output = tmp_path / "output"
    record = reuse(artifacts, output)
    for name in module.VISION_FILES:
        assert record["files"][name]["method"] == "copy2"
        assert not (artifacts[0] / name).samefile(output / name)
        assert (artifacts[0] / name).read_bytes() == (output / name).read_bytes()


@pytest.mark.parametrize("location", ["same", "inside", "parent"])
def test_rejects_overlapping_directories(artifacts, location):
    source = artifacts[0]
    output = {"same": source, "inside": source / "nested", "parent": source.parent}[
        location
    ]
    with pytest.raises(ValueError, match="non-nested"):
        reuse(artifacts, output)


@pytest.mark.parametrize("name", module.VISION_FILES)
@pytest.mark.parametrize("missing", [False, True])
def test_requires_three_nonempty_files(artifacts, name, missing):
    path = artifacts[0] / name
    if missing:
        path.unlink()
    else:
        path.write_bytes(b"")
    with pytest.raises(FileNotFoundError if missing else ValueError):
        validate(artifacts)


@pytest.mark.parametrize("change", ["target", "arch", "hash", "shape", "precision"])
def test_rejects_incompatible_metadata(artifacts, change):
    source, _, _, metadata = artifacts
    if change == "target":
        metadata["target"] = dict(soc_model="SM8750", dsp_arch="v79")
    elif change == "arch":
        metadata["target"]["dsp_arch"] = "v85"
    elif change == "hash":
        metadata["model_config_sha256"] = "0" * 64
    elif change == "shape":
        metadata["output_shape"] = [1, 195, 2048]
    else:
        metadata["precision"] = "FP16"
    (source / "VISION.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        validate(artifacts)


def test_requires_exact_requested_dimensions(artifacts):
    with pytest.raises(ValueError, match="dimensions"):
        validate(artifacts, height=224)
    with pytest.raises(ValueError, match="integers"):
        validate(artifacts, width=448.0)


@pytest.mark.parametrize(
    "change", ["empty", "method", "bits", "position", "damping", "malformed"]
)
def test_gptq_recipe_requires_consistent_fixed_grids(artifacts, change):
    source, _, _, metadata = artifacts
    reports = metadata["gptq"]
    if change == "empty":
        reports.clear()
    elif change == "method":
        reports[1]["method"] = "rtn"
    elif change == "bits":
        reports[1]["bits"] = 4
    elif change == "position":
        reports.pop(0)
    elif change == "damping":
        reports[0]["damping"] = 0.02
    else:
        reports[0] = None
    (source / "VISION.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="GPTQ"):
        validate(artifacts)


def test_plain_w8a16_can_reuse_non_gptq_vision(artifacts):
    source, _, recipe, metadata = artifacts
    settings = json.loads(recipe.read_text())
    settings["quantization"]["vision"] = "w8a16"
    recipe.write_text(json.dumps(settings))
    metadata["gptq"] = []
    (source / "VISION.json").write_text(json.dumps(metadata))
    assert validate(artifacts) == metadata


def test_validation_uses_resolved_settings_object(artifacts):
    source, model, recipe, metadata = artifacts
    settings = module.load_model_config(recipe)
    data = json.loads(recipe.read_text())
    data["target"]["soc_model"] = "SM8975"
    recipe.write_text(json.dumps(data))
    assert module.validate_vision_source(source, model, settings, 448, 448) == metadata
    with pytest.raises(ValueError, match="Artifact target"):
        validate(artifacts)


def test_refuses_to_replace_existing_payloads(artifacts, tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    existing = output / "vision.bin"
    existing.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        reuse(artifacts, output)
    assert existing.read_bytes() == b"keep"
    assert not (output / "VISION.json").exists()


def test_failed_copy_publishes_no_partial_artifacts(artifacts, tmp_path, monkeypatch):
    original = {
        name: (artifacts[0] / name).read_bytes() for name in module.VISION_FILES
    }
    copy2 = module.shutil.copy2

    def cross_device(*args):
        raise OSError(errno.EXDEV, "synthetic cross-device link")

    def fail_copy(source, destination):
        if source.name == "vision.bin":
            destination.write_bytes(b"partial")
            raise OSError(errno.EIO, "synthetic read failure")
        return copy2(source, destination)

    monkeypatch.setattr(module.os, "link", cross_device)
    monkeypatch.setattr(module.shutil, "copy2", fail_copy)
    output = tmp_path / "output"
    with pytest.raises(OSError, match="read failure"):
        reuse(artifacts, output)
    assert not list(output.iterdir())
    assert original == {
        name: (artifacts[0] / name).read_bytes() for name in module.VISION_FILES
    }


def test_direct_cli_needs_no_qnn_binding_or_model_weights(artifacts, tmp_path):
    source, model, recipe, _ = artifacts
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "workflow/reuse_vision.py"),
            "--source",
            str(source),
            "--output",
            str(tmp_path / "cli-output"),
            "--model",
            str(model),
            "--model-config",
            str(recipe),
            "--image-height",
            "448",
            "--image-width",
            "448",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "REUSED_VISION" in result.stdout
