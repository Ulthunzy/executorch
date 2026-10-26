import json
from pathlib import Path
import struct
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from workflow.model_config import load_model_config  # noqa: E402
from workflow.config import parse_args  # noqa: E402
from workflow.export_config import load_export_config  # noqa: E402
from workflow.export_pipeline import ExportPipeline  # noqa: E402


def write(path, data=b"fixture"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def make_config(tmp_path, **changes):
    model = tmp_path / "model"
    write(
        model / "config.json",
        json.dumps(
            dict(
                text_config=dict(
                    num_hidden_layers=40,
                    hidden_size=2048,
                    num_experts=256,
                    num_experts_per_tok=8,
                )
            )
        ).encode(),
    )
    write(model / "model.safetensors.index.json", b'{"weight_map": {}}')
    numerical = json.loads((ROOT / "configs/model.json").read_text())
    numerical["context"].update(max_seq_len=1024, max_prefill_tokens=128)
    model_config = tmp_path / "model-config.json"
    write(model_config, json.dumps(numerical).encode())
    source = tmp_path / "calibrated"
    write(source / "MODEL_CONFIG.json", json.dumps(numerical).encode())
    for name in (
        "QUANTIZATION_COMPLETE.json",
        "cold-experts.i32",
        "embedding.int4",
        "head/head-qdq.pt2",
        "head/GPTQ.json",
    ):
        write(source / name)
    for layer in range(40):
        for name in ("front-qdq.pt2", "decode-qdq.pt2", "moe-constants.npz"):
            write(source / f"layers/layer{layer}" / name)
        for name in ("router-ar128-qdq.pt2", "router-ar1-qdq.pt2", "REPORT.json"):
            write(source / f"router/layer{layer}" / name)
        for name in ("BANK.json", "experts.n2"):
            write(source / f"banks/layer{layer}" / name)
        write(source / f"indices/layer{layer}/bank-index.bin")
    sdk = tmp_path / "sdk"
    for name in (
        "include/QNN/QnnInterface.h",
        "lib/x86_64-linux-clang/libQnnSystem.so",
        "lib/x86_64-linux-clang/libQnnHtp.so",
    ):
        write(sdk / name)
    build = tmp_path / "build"
    for name in (
        "capture_weights",
        "op-package/x86_64-linux-clang/libQnnDeltaNetOpPackage.so",
        "op-package/hexagon-v85/libQnnDeltaNetOpPackage_HTP.so",
    ):
        write(build / name)
    image = tmp_path / "image.jpg"
    write(image)
    data = dict(
        version=1,
        model_config=str(model_config),
        model=str(model),
        source="calibrated",
        calibration_dir=str(source),
        output=str(tmp_path / "export"),
        components=["text"],
        phases=["prefill", "decode"],
        format="pte",
        build=False,
        build_dir=str(build),
        qnn_sdk_root=str(sdk),
        device="test-serial",
        runtime_device_dir="/data/local/tmp/runtime",
        images=[str(image)],
    )
    data.update(changes)
    path = tmp_path / "export.json"
    write(path, json.dumps(data).encode())
    return path


@pytest.mark.parametrize(
    "updates",
    [
        {"components": ["text", "text"]},
        {"components": ["typo"]},
        {"format": "onnx"},
        {"format": "qnn", "phases": ["decode"]},
        {"build": "false"},
        {"expert_batch": True},
        {"unknown": 1},
        {"remote_root": "/data/local/tmp/export;rm"},
        {"remote_root": None},
        {"device": []},
        {"model_config": None},
        {"op_package_device": None},
    ],
)
def test_invalid_settings_rejected_before_side_effects(tmp_path, updates):
    path = make_config(tmp_path, **updates)
    with pytest.raises(ValueError):
        load_export_config(path)
    assert not (tmp_path / "export").exists()


def test_component_and_format_selection_avoid_unneeded_device_steps(tmp_path):
    path = make_config(
        tmp_path,
        components=["vision"],
        format="qnn",
        device=None,
        runtime_device_dir=None,
        build=True,
    )
    pipeline = ExportPipeline(load_export_config(path))
    pipeline.check()
    assert [s.name for s in pipeline.stages] == ["vision"]
    assert pipeline.stages[0].cleanup[0].name == "vision.pte"
    assert not (tmp_path / "export").exists()


def test_decode_selection_preserves_shared_compile_but_only_packs_decode(tmp_path):
    pipeline = ExportPipeline(
        load_export_config(make_config(tmp_path, phases=["decode"]))
    )
    assert len([s for s in pipeline.stages if s.name.startswith("qnn-group")]) == 10
    assert pipeline.stages[-1].name == "decode"
    assert not any(s.name.startswith("prefill-") for s in pipeline.stages)
    for stage in pipeline.stages:
        if stage.name.startswith("qnn-group"):
            start = stage.command.index("--layers") + 1
            end = stage.command.index("--output")
            assert len(stage.command[start:end]) == 4


def child_runner(pipeline, calls, fail=None, real_pack=False):
    def run(command, **kwargs):
        stage = next(s for s in pipeline.stages if s.command == command)
        calls.append(stage.name)
        if stage.name == fail:
            return subprocess.CompletedProcess(command, 37)
        if real_pack and stage.name in ("prefill-128", "decode"):
            from export.program import main as pack_plan

            pack_plan(list(command[3:]))
            return subprocess.CompletedProcess(command, 0)
        for path in stage.outputs:
            if path.name == "GRAPHS.json":
                group = int(path.parent.name.removeprefix("group"))
                report = dict(
                    contexts={"context0.bin": "fixture"},
                    capacity=1024,
                    weight_sharing=True,
                    layers=list(range(group * 4, group * 4 + 4)),
                    router="w8a16_gptq",
                    compile_settings=pipeline.settings.compile_settings(),
                )
                write(path, json.dumps(report).encode())
            elif path.name == "WEIGHTS.json":
                write(path, b'{"source_sha256": "fixture"}')
            elif path.suffix == ".io":
                layer = int(path.stem.split("_")[0].removeprefix("layer"))
                if path.stem.endswith("_decode"):
                    inputs, outputs = (6 if layer % 4 == 3 else 3), 7
                elif path.stem.endswith("_prefill_attention"):
                    inputs, outputs = (6 if layer % 4 == 3 else 4), 4
                else:
                    inputs, outputs = 1, 3
                write(
                    path,
                    struct.pack(
                        "<" + "I" * (2 + inputs + outputs),
                        inputs,
                        outputs,
                        *range(inputs),
                        *range(outputs),
                    ),
                )
            elif path.name == "cache-encoding.bin":
                write(path, b"Q36CACHE" + struct.pack("<II", 1, 40) + b"\0" * 1280)
            else:
                write(path)
        for path in stage.cleanup:
            write(path, b"owned temporary")
        return subprocess.CompletedProcess(command, 0)

    return run


def test_failed_stage_resumes_without_repeating_completed_groups(tmp_path, monkeypatch):
    config = load_export_config(make_config(tmp_path))
    pipeline = ExportPipeline(config)
    calls = []
    monkeypatch.setattr(
        subprocess, "run", child_runner(pipeline, calls, fail="qnn-group5")
    )
    with pytest.raises(RuntimeError, match="qnn-group5 exited"):
        pipeline.run()
    assert not (config.output / "EXPORT.json").exists()
    state = json.loads((config.output / "EXPORT_STATE.json").read_text())
    assert state["stages"]["qnn-group5"]["status"] == "failed"
    assert not (config.output / "qnn-shared/group0/methods.pte").exists()
    monkeypatch.setattr(subprocess, "run", child_runner(pipeline, calls))
    pipeline.run()
    assert calls.count("head") == 1
    assert calls.count("qnn-group0") == 1
    assert calls.count("qnn-group5") == 2
    count = len(calls)
    pipeline.run()
    assert len(calls) == count
    assert (config.output / "banks").is_symlink()
    assert (config.output / "banks").resolve() == config.calibration_dir / "banks"
    effective = parse_args(["--config", str(config.output / "device.json")])
    assert effective.decode_capacity == 1024
    assert effective.model_config == config.output / "MODEL_CONFIG.json"
    assert effective.local_decode_pte == config.output / "plans/decode.pte"
    assert effective.local_prefill_pte is None
    monkeypatch.setattr(subprocess, "run", child_runner(pipeline, calls, fail="head"))
    with pytest.raises(RuntimeError, match="head exited"):
        pipeline.run(resume=False)
    assert not (config.output / "EXPORT.json").exists()


def test_recipe_and_existing_output_are_not_overwritten(tmp_path):
    config = load_export_config(make_config(tmp_path))
    config.output.mkdir()
    write(config.output / "user-file", b"keep")
    pipeline = ExportPipeline(config)
    with pytest.raises(ValueError, match="not an empty export workspace"):
        pipeline.run()
    assert (config.output / "user-file").read_bytes() == b"keep"
    source = config.calibration_dir / "MODEL_CONFIG.json"
    old = json.loads(source.read_text())
    old["cache"]["kv_dtype"] = "uint8"
    source.write_text(json.dumps(old))
    with pytest.raises(ValueError, match="recipes differ"):
        pipeline.check()


def test_raw_preparation_is_selected_and_cannot_overwrite_calibration(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    write(corpus, b'{"text":"calibration fixture"}\n')
    config = load_export_config(
        make_config(tmp_path, source="safetensors", corpus=str(corpus))
    )
    pipeline = ExportPipeline(config)
    pipeline.check()
    assert pipeline.stages[0].name == "quantize"
    assert "--corpus" in pipeline.stages[0].command
    with pytest.raises(ValueError, match="new calibration_dir"):
        pipeline.run()


def test_resume_rejects_changed_inputs(tmp_path, monkeypatch):
    pipeline = ExportPipeline(load_export_config(make_config(tmp_path)))
    calls = []
    monkeypatch.setattr(
        subprocess, "run", child_runner(pipeline, calls, fail="qnn-group1")
    )
    with pytest.raises(RuntimeError):
        pipeline.run()
    write(
        pipeline.config.calibration_dir / "layers/layer0/front-qdq.pt2",
        b"changed calibration",
    )
    count = len(calls)
    with pytest.raises(ValueError, match="changed"):
        pipeline.run()
    assert len(calls) == count


def test_pte_plans_really_serialize_both_phases(tmp_path, monkeypatch):
    from executorch.exir._serialize._program import deserialize_pte_binary

    config = load_export_config(make_config(tmp_path))
    pipeline = ExportPipeline(config)
    monkeypatch.setattr(subprocess, "run", child_runner(pipeline, [], real_pack=True))
    pipeline.run()
    for phase, filename in (("prefill", "prefill-128"), ("decode", "decode")):
        path = config.output / f"plans/{filename}.pte"
        program = deserialize_pte_binary(path.read_bytes()).program
        plan = program.execution_plan[0]
        assert plan.name == f"{phase}_forward"
        assert len(plan.delegates) == 122
        expected = "Qwen36PrefillStage" if phase == "prefill" else "Qwen36DecodeStage"
        assert {d.id for d in plan.delegates} == {expected}
        metadata = json.loads(path.with_suffix(".json").read_text())
        assert metadata["weight_mode"] == "external_shared"
        assert (
            metadata["compile_settings"]
            == load_model_config(config.model_config).compile_settings()
        )


def test_shared_weight_layout_round_trips_the_opaque_context(tmp_path):
    from export.weights import separate_context

    source = tmp_path / "context.bin"
    original = bytes(range(256)) * 64
    source.write_bytes(original)
    requests = tmp_path / "requests.txt"
    requests.write_text("4096 8192\n4096 8192\n")
    output = tmp_path / "external"
    report = separate_context(source, requests, output)
    image = bytearray(len(original))
    for filename, sections in [
        ("graph.qnn", report["graph_sections"]),
        ("weights.bin", report["weight_sections"]),
    ]:
        data = (output / filename).read_bytes()
        for offset, size, file_offset in sections:
            image[offset : offset + size] = data[file_offset : file_offset + size]
    assert image == original
    assert (output / "weights.bin").read_bytes() == original[4096:12288]
    requests.write_text("1 8192\n")
    with pytest.raises(ValueError, match="page-aligned"):
        separate_context(source, requests, tmp_path / "invalid")


def test_internal_stage_does_not_shadow_quantization_package():
    command = (
        "import sys;"
        f"sys.path[:0]=[{str(ROOT)!r},{str(ROOT / 'export')!r}];"
        "import quantization.checkpoint;"
        f"assert quantization.checkpoint.__file__=={str(ROOT / 'quantization/checkpoint.py')!r}"
    )
    subprocess.run([sys.executable, "-c", command], check=True, capture_output=True)
