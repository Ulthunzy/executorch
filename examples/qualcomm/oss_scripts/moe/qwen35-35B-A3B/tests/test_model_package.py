import json
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from workflow.checkpoint import StageCheckpoint, artifact_stamps  # noqa: E402
from workflow.model_package import PackageReader, create_package  # noqa: E402


def test_stage_cache_detects_same_size_corruption_with_unchanged_mtime(tmp_path):
    root = tmp_path / "output"
    payload = root / "weights.bin"
    with StageCheckpoint(root, {"source": 1}, "STATE.json") as cache:
        payload.write_bytes(b"x" * (1024 * 1024 + 5))
        cache.update("weights", status="complete", outputs=artifact_stamps((payload,)))
    before = payload.stat()
    with payload.open("r+b") as f:
        f.write(b"bad")
    os.utime(payload, ns=(before.st_atime_ns, before.st_mtime_ns))
    with StageCheckpoint(root, {"source": 1}, "STATE.json") as cache:
        assert not cache.reusable("weights", (payload,))


def test_stage_cache_recreates_invalid_state_and_protects_source(tmp_path):
    root = tmp_path / "output"
    root.mkdir()
    (root / "STATE.json").write_text("{bad")
    (root / "partial.bin").write_bytes(b"old")
    source = tmp_path / "source.safetensors"
    source.write_bytes(b"source")
    with StageCheckpoint(root, {"source": 1}, "STATE.json", (source,)):
        assert not (root / "partial.bin").exists()
    assert source.read_bytes() == b"source"
    with pytest.raises(ValueError, match="protected input"):
        StageCheckpoint(tmp_path, {}, "STATE.json", (source,))


def test_stage_cache_lock_survives_workspace_reset(tmp_path):
    with StageCheckpoint(tmp_path / "output", {}, "STATE.json"):
        with pytest.raises(RuntimeError, match="already using"):
            with StageCheckpoint(tmp_path / "output", {}, "STATE.json"):
                pass


def test_package_streams_assets_and_repairs_missing_or_corrupt_cache(
    tmp_path, monkeypatch
):
    from executorch.exir._serialize._cord import FileBackedData

    def no_materialization(*args):
        raise AssertionError("Resource was loaded as one large byte buffer")

    monkeypatch.setattr(FileBackedData, "to_bytes", no_materialization)
    weight = tmp_path / "weights.bin"
    with weight.open("wb") as f:
        f.truncate(17 * 1024 * 1024 + 3)
    destination = tmp_path / "model.pte"
    create_package(
        destination,
        {"shared/weights.bin": weight},
        {"model/weight.bin": "../shared/weights.bin"},
    )
    with destination.open("rb") as f:
        assert f.read(8)[4:6] == b"ET"
    reader = PackageReader(destination)
    directory = reader.extract()
    copy = directory / "shared/weights.bin"
    assert copy.stat().st_size == weight.stat().st_size
    stamp = copy.stat().st_mtime_ns
    assert reader.extract() == directory
    assert copy.stat().st_mtime_ns == stamp
    copy.write_bytes(b"corrupt")
    reader.extract()
    assert copy.stat().st_size == weight.stat().st_size
    copy.unlink()
    reader.extract()
    assert copy.exists()
    assert (directory / "model/weight.bin").resolve() == copy


def test_package_rejects_corrupt_payload_and_unsafe_paths(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"correct weights")
    path = tmp_path / "model.pte"
    create_package(path, {"weights.bin": source})
    reader = PackageReader(path)
    with path.open("r+b") as f:
        f.seek(reader.entries["weights.bin"][0])
        f.write(b"X")
    with pytest.raises(ValueError, match="Corrupt model resource"):
        PackageReader(path).extract()
    with pytest.raises(ValueError, match="Unsafe model resource"):
        create_package(path, {"../outside.bin": source})
    with path.open("r+b") as f:
        f.truncate(80)
    with pytest.raises(ValueError, match="truncated"):
        PackageReader(path)


def test_package_keeps_separated_head_without_export_only_inputs(tmp_path):
    import hashlib
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "head_weights", ROOT / "export/weights.py"
    )
    weights = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(weights)

    source = tmp_path / "raw-head.bin"
    payload = bytes(range(256)) * 64 + b"context-tail"
    source.write_bytes(payload)
    requests = tmp_path / "requests.txt"
    requests.write_text("4096 4096\n12288 4096\n")
    head = tmp_path / "shared-model/head"
    report = weights.separate_context(source, requests, head)
    package = tmp_path / "model.pte"
    names = ("graph.qnn", "weights.bin", "WEIGHTS.json")
    manifest = create_package(
        package, {f"shared-model/head/{name}": head / name for name in names}
    )
    source.unlink()

    reader = PackageReader(package)
    extracted = reader.extract()
    assert reader.manifest == manifest
    assert not (extracted / "head/HEAD.json").exists()
    assert not (extracted / "head/head.bin").exists()
    metadata = json.loads((extracted / "shared-model/head/WEIGHTS.json").read_text())
    assert metadata == json.loads(json.dumps(report))
    assert metadata["source_sha256"] == hashlib.sha256(payload).hexdigest()
    for name in names:
        path = extracted / "shared-model/head" / name
        info = manifest["files"][f"shared-model/head/{name}"]
        assert path.stat().st_size == info["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == info["sha256"]


def test_quantization_cache_reuses_weights_and_repairs_the_first_bad_layer(
    tmp_path, monkeypatch
):
    import hashlib
    import subprocess
    from quantization.prepare_model import main

    model = tmp_path / "source"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "model.safetensors.index.json").write_text('{"weight_map":{}}')
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"text":"calibration"}\n')
    recipe = tmp_path / "recipe.json"
    recipe.write_text((ROOT / "configs/model.json").read_text())
    output = tmp_path / "calibration"
    calls = []

    def run(command, **kwargs):
        if "--stage" in command:
            name = command[command.index("--stage") + 1]
        elif any(str(x).endswith("prepare_embedding.py") for x in command):
            name = "embedding"
        elif any(str(x).endswith("prepare_router.py") for x in command):
            name = "router"
        else:
            name = "experts"
        calls.append(name)
        paths = {
            "data": [
                "DATA.json",
                "INPUTS-fp32.pt",
                "embedding.bf16",
                "assets/cos.f32",
                "assets/sin.f32",
            ],
            "embedding": ["embedding.int4", "embedding.json"],
            "embedding-inputs": ["INPUTS.pt", "EMBEDDING_INPUTS.json"],
            "attention": [
                f"layers/layer0/{n}"
                for n in (
                    "GPTQ.json",
                    "front-qdq.pt2",
                    "decode-qdq.pt2",
                    "moe-constants.npz",
                    "front-output.pt",
                )
            ],
            "router": [
                f"router/layer0/{n}"
                for n in ("REPORT.json", "router-ar128-qdq.pt2", "router-ar1-qdq.pt2")
            ],
            "experts": [
                "banks/layer0/BANK.json",
                "banks/layer0/experts.n2",
                "indices/layer0/bank-index.bin",
                "sequential/layer0.pt",
            ],
        }
        for name_ in paths[name]:
            p = output / name_
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"generated")
        if name == "data":
            (output / "DATA.json").write_text(
                json.dumps(
                    dict(
                        index_sha256=hashlib.sha256(
                            (model / "model.safetensors.index.json").read_bytes()
                        ).hexdigest(),
                        corpus_sha256=hashlib.sha256(corpus.read_bytes()).hexdigest(),
                        safetensors={},
                    )
                )
            )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", run)
    args = [
        "--model",
        str(model),
        "--output",
        str(output),
        "--corpus",
        str(corpus),
        "--model-config",
        str(recipe),
        "--last-layer",
        "0",
    ]
    main(args)
    assert len(calls) == 6
    calls.clear()
    main(args)
    assert not calls
    data = json.loads(recipe.read_text())
    data["generation"]["temperature"] = 0.7
    recipe.write_text(json.dumps(data))
    main(args)
    assert not calls
    (output / "layers/layer0/front-qdq.pt2").write_bytes(b"corrupt")
    main(args)
    assert calls == ["attention", "router", "experts"]
    calls.clear()
    (model / "tokenizer.json").write_text('{"version": "changed"}')
    main(args)
    assert len(calls) == 6
