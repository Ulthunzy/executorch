import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from test_export_workflow import make_config, write
from workflow.export_config import load_export_config
from workflow.export_pipeline import ExportPipeline


def vision_config(tmp_path):
    source = tmp_path / "vision-source"
    config_path = make_config(
        tmp_path,
        components=["vision"],
        vision_source=str(source),
        images=[],
        format="qnn",
        soc_model="SM8850",
    )
    data = json.loads(config_path.read_text())
    model = Path(data["model"])
    config = json.loads((model / "config.json").read_text())
    config.update(
        image_token_id=248056,
        vision_config=dict(
            patch_size=16, spatial_merge_size=2, in_channels=3, temporal_patch_size=2
        ),
    )
    write(model / "config.json", json.dumps(config).encode())
    metadata = dict(
        version=1,
        target=dict(soc_model="SM8850", dsp_arch="v81"),
        model_config_sha256=hashlib.sha256(
            (model / "config.json").read_bytes()
        ).hexdigest(),
        precision="W8A16",
        image_height=448,
        image_width=448,
        image_token_id=248056,
        input_shape=[1, 784, 1536],
        output_shape=[1, 196, 2048],
        gptq=[
            dict(
                module="visual.pos_embed",
                method="fixed_grid_gptq",
                bits=8,
                damping=0.01,
            )
        ],
    )
    write(source / "VISION.json", json.dumps(metadata).encode())
    write(source / "vision.bin")
    write(source / "vision.pte")
    return config_path, source


def test_reused_vision_needs_no_checkpoint_shards_or_calibration_images(tmp_path):
    path, source = vision_config(tmp_path)
    config = load_export_config(path)
    (config.model / "model.safetensors.index.json").unlink()
    pipeline = ExportPipeline(config)
    pipeline.check()
    assert pipeline.checkpoint_shards() == []
    assert len(pipeline.stages) == 1
    stage = pipeline.stages[0]
    assert stage.command[2].endswith("workflow/reuse_vision.py")
    assert (
        "--calib-images" not in stage.command
        and "--calibration-device" not in stage.command
    )
    assert stage.command[stage.command.index("--source") + 1] == str(source)
    assert any(p.name == "VISION_REUSE.json" for p in stage.outputs)
    assert not config.output.exists()


@pytest.mark.parametrize("missing", ["vision.bin", "vision.pte", "VISION.json"])
def test_reuse_vision_missing_input_fails_preflight(tmp_path, missing):
    path, source = vision_config(tmp_path)
    (source / missing).unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        ExportPipeline(load_export_config(path)).check()


def test_components_override_preserves_configured_vision_source(tmp_path):
    path, source = vision_config(tmp_path)
    data = json.loads(path.read_text())
    data["components"] = ["text"]
    path.write_text(json.dumps(data))
    default = load_export_config(path)
    selected = load_export_config(path, components=["text", "vision"])
    assert default.components == ("text",)
    assert (
        selected.components == ("text", "vision") and selected.vision_source == source
    )
    assert selected.source == "calibrated"
