"""Validate complete GPTQ outputs independently of later export stages."""

import hashlib
import importlib.metadata
import json
from pathlib import Path

from executorch.examples.qualcomm.oss_scripts.moe.checkpoint import (
    artifact_stamps,
    input_stamp,
)
from executorch.examples.qualcomm.oss_scripts.moe.quantize.provenance import (
    checkpoint_identity,
    validate_checkpoint_source,
)

# Bump when calibration math or packed weight formats change.
QUANTIZATION_VERSION = 1


def prepared_artifacts(root, layers=40):
    names = [
        "MODEL_CONFIG.json",
        "DATA.json",
        "QUANTIZATION_COMPLETE.json",
        "cold-experts.i32",
        "embedding.int4",
        "head/head-qdq.pt2",
        "head/GPTQ.json",
    ]
    for layer in range(layers):
        names.extend(
            f"layers/layer{layer}/{name}"
            for name in (
                "front-qdq.pt2",
                "decode-qdq.pt2",
                "moe-constants.npz",
                "GPTQ.json",
            )
        )
        names.extend(
            f"router/layer{layer}/{name}"
            for name in ("router-ar128-qdq.pt2", "router-ar1-qdq.pt2", "REPORT.json")
        )
        names.extend(
            f"banks/layer{layer}/{name}" for name in ("BANK.json", "experts.n2")
        )
        names.append(f"indices/layer{layer}/bank-index.bin")
    return tuple(Path(root) / name for name in names)


def reusable_preparation(root, model, settings, corpus, qnn_binding=None):
    from executorch.examples.qualcomm.oss_scripts.moe.model.qwen35 import (
        load_model_config,
    )

    root = Path(root)
    try:
        complete = json.loads((root / "QUANTIZATION_COMPLETE.json").read_text())
        if not isinstance(complete, dict):
            return False
        if complete.get("format_version", 1) != QUANTIZATION_VERSION:
            return False
        if (
            complete.get("layers") != 40
            or complete.get("all_expert_gu_down_gptq") is not True
        ):
            return False
        prepared = load_model_config(root / "MODEL_CONFIG.json")
        recorded_settings = complete.get("model_configuration")
        prepared_settings = prepared.to_dict()
        if isinstance(recorded_settings, dict) and isinstance(
            recorded_settings.get("context"), dict
        ):
            recorded_settings = {
                **recorded_settings,
                "context": {
                    k: v
                    for k, v in recorded_settings["context"].items()
                    if k != "max_prefill_tokens"
                },
            }
        if not isinstance(recorded_settings, dict) or any(
            recorded_settings.get(key) != prepared_settings[key]
            for key in ("quantization", "cache", "context", "target")
        ):
            return False
        if (
            prepared.compile_settings() != settings.compile_settings()
            or prepared.quantization != settings.quantization
            or prepared.target != settings.target
        ):
            return False
        data = json.loads((root / "DATA.json").read_text())
        validate_checkpoint_source(model, data)
        if (
            data["corpus_sha256"]
            != hashlib.sha256(Path(corpus).read_bytes()).hexdigest()
        ):
            return False
        state = {}
        if "provenance" not in complete or "artifacts" not in complete:
            state = json.loads((root / "PREPARE_STATE.json").read_text())
            if not isinstance(state, dict):
                return False
        provenance = (
            complete["provenance"] if "provenance" in complete else state["identity"]
        )
        if (
            not isinstance(provenance, dict)
            or not isinstance(provenance.get("packages"), dict)
            or not isinstance(provenance.get("checkpoint"), dict)
        ):
            return False
        if set(provenance["packages"]) != {"torch", "torchao", "transformers"}:
            return False
        for name, version in provenance["packages"].items():
            if importlib.metadata.version(name) != version:
                return False
        recorded = provenance["checkpoint"]
        indexes = [
            Path(name)
            for name in recorded
            if Path(name).name == "model.safetensors.index.json"
        ]
        if len(indexes) != 1:
            return False
        origin = indexes[0].parent
        expected_source = {
            str(Path(name).relative_to(origin)): info for name, info in recorded.items()
        }
        current_source = {
            str(Path(name).relative_to(model)): info
            for name, info in checkpoint_identity(model).items()
        }
        if expected_source != current_source:
            return False
        if "qnn_binding" not in provenance and (root / "PREPARE_STATE.json").is_file():
            state = json.loads((root / "PREPARE_STATE.json").read_text())
            if not isinstance(state, dict) or not isinstance(
                state.get("identity"), dict
            ):
                return False
        if "qnn_binding" in provenance:
            recorded_binding = provenance["qnn_binding"]
        else:
            previous_identity = state.get("identity", {})
            if not isinstance(previous_identity, dict):
                return False
            recorded_binding = previous_identity.get("qnn_binding")
        if recorded_binding != (input_stamp(qnn_binding) if qnn_binding else None):
            return False
        paths = tuple(
            path
            for path in prepared_artifacts(root)
            if path.name != "QUANTIZATION_COMPLETE.json"
        )
        actual = {
            str(Path(name).relative_to(root)): stamp
            for name, stamp in artifact_stamps(paths).items()
        }
        expected = complete.get("artifacts")
        if expected is None:
            if not isinstance(state, dict) or not isinstance(state.get("stages"), dict):
                return False
            if any(
                not isinstance(stage, dict)
                or (
                    stage.get("status") == "complete"
                    and not isinstance(stage.get("outputs"), dict)
                )
                for stage in state["stages"].values()
            ):
                return False
            recorded = {
                name: stamp
                for stage in state["stages"].values()
                if stage.get("status") == "complete"
                for name, stamp in stage["outputs"].items()
            }
            expected = {
                str(path.relative_to(root)): recorded.get(str(path)) for path in paths
            }
            for name in ("MODEL_CONFIG.json", "cold-experts.i32"):
                expected[name] = actual[name]
        return actual == expected
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        importlib.metadata.PackageNotFoundError,
    ):
        return False
