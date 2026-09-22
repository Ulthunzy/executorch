"""Export lineage binds source, calibration, PT2 and PTE content."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from model_validation.common.lineage import write_export_lineage
from model_validation.ppl.artifacts import digest, read_json, tree_identity


def test_export_lineage_is_complete_and_content_bound(tmp_path):
    source = tmp_path / "safetensors"
    source.mkdir()
    (source / "model.safetensors").write_bytes(b"hf")
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint")
    calibration = tmp_path / "calibration.json"
    calibration.write_text("[]", encoding="utf-8")
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("{}", encoding="utf-8")
    pt2 = tmp_path / "decode_qdq.pt2"
    pt2.write_bytes(b"pt2")
    pte = tmp_path / "model.pte"
    pte.write_bytes(b"pte")
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    args = SimpleNamespace(
        artifact=str(artifact),
        source_safetensors=str(source),
        checkpoint=str(checkpoint),
        params=None,
        calib_samples=[str(calibration)],
        calib_text_datasets=None,
        calib_text_preserve_sources=None,
        calib_tasks=None,
        calib_hf_dataset=None,
        soc_model="SM8750",
        backend="qnn",
        quantization_mode="8a4w",
    )
    code_body = {
        "commit": "a" * 40,
        "dirty": False,
        "tracked_diff_sha256": "b" * 64,
        "untracked_sources": {},
    }
    code = {**code_body, "sha256": digest(code_body)}
    with patch("model_validation.common.lineage._git_identity", return_value=code):
        output = write_export_lineage(
            args,
            pte_path=pte,
            pt2_path=pt2,
            runtime_tokenizer_path=tokenizer,
            qnn_sdk_build_id="2.49.40.260810",
            repo=tmp_path,
        )
    lineage = read_json(output)
    saved_hash = lineage.pop("lineage_sha256")
    assert saved_hash == digest(lineage)
    assert lineage["complete"] is True
    assert lineage["configuration_sha256"] == digest(lineage["configuration"])
    assert (
        lineage["sources"]["safetensors"]["sha256"] == tree_identity(source)["sha256"]
    )
    assert lineage["outputs"]["pt2"]["sha256"] == tree_identity(pt2)["sha256"]
    assert lineage["outputs"]["pte"]["sha256"] == tree_identity(pte)["sha256"]


def test_export_lineage_rejects_unbound_or_missing_quantization_inputs(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.safetensors").write_bytes(b"hf")
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint")
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("{}", encoding="utf-8")
    pte = tmp_path / "model.pte"
    pte.write_bytes(b"pte")
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    args = SimpleNamespace(
        artifact=str(artifact),
        source_safetensors=str(source),
        checkpoint=str(checkpoint),
        params=None,
        calib_samples=None,
        calib_text_datasets=["wikitext-2"],
        calib_text_preserve_sources=None,
        calib_tasks=None,
        calib_hf_dataset=None,
        qat=False,
        soc_model="SM8750",
        backend="qnn",
    )
    code_body = {
        "commit": "a" * 40,
        "dirty": False,
        "tracked_diff_sha256": "b" * 64,
        "untracked_sources": {},
    }
    code = {**code_body, "sha256": digest(code_body)}
    with patch("model_validation.common.lineage._git_identity", return_value=code):
        output = write_export_lineage(
            args,
            pte_path=pte,
            pt2_path=None,
            runtime_tokenizer_path=tokenizer,
            qnn_sdk_build_id="2.49",
            repo=tmp_path,
        )
    lineage = read_json(output)
    assert lineage["complete"] is False
    assert lineage["outputs"]["pt2"] is None
    assert lineage["sources"]["calibration"][0]["kind"] == "external"
