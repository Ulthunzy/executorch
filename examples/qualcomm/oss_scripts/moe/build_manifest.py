"""Provenance for the paired ARM and DSP runtime libraries."""

import argparse
import hashlib
import json
from pathlib import Path

NATIVE_FILES = (
    "native-prefill/et_prefill_rpc.h",
    "native-prefill/et_prefill_rpc_stub.o",
    "native-decode/et_decode_rpc.h",
    "native-decode/et_decode_rpc_stub.o",
    "libet_qnnop.so",
)
RUNTIME_FILES = (
    "qwen35_moe_runner",
    "prefill_handoff_runner",
    "libet_infer.so",
    "libet_qnnop.so",
)


def artifact_hashes(root, names):
    result = {}
    for name in names:
        path = root / name
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"Missing or empty build artifact: {path}")
        with path.open("rb") as stream:
            result[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result


def validate_runtime_build(root, metadata):
    if metadata.get("version", 1) < 2:
        return
    expected = metadata.get("artifacts", {})
    actual = artifact_hashes(root, RUNTIME_FILES)
    if actual != expected or artifact_hashes(root, ("NATIVE_TARGET.json",))[
        "NATIVE_TARGET.json"
    ] != metadata.get("native_build_sha256"):
        raise ValueError(f"Runtime build artifacts changed; rebuild {root}")


def record_build(args):
    root = args.output.resolve()
    metadata = dict(
        version=1,
        dsp_arch=f"v{args.arch}",
        native_profile=args.profile,
        prefill_mapped_io=2,
        qnn_sdk=str(args.sdk.resolve()),
        artifacts=artifact_hashes(root, NATIVE_FILES),
    )
    name = "NATIVE_TARGET.json"
    if args.kind == "runtime":
        native = json.loads((root / name).read_text())
        if native != metadata:
            raise ValueError("Native build does not match the requested runtime")
        metadata = dict(
            version=2,
            target=dict(soc_model=args.soc, dsp_arch=f"v{args.arch}"),
            chat_controls=1,
            runtime_layout="two_libraries",
            native_profile=args.profile,
            idle_power_management=bool(args.idle_power_management),
            native_build_sha256=artifact_hashes(root, (name,))[name],
            artifacts=artifact_hashes(root, RUNTIME_FILES),
        )
        name = "TARGET.json"
    temporary = root / (name + ".tmp")
    temporary.write_text(json.dumps(metadata, indent=2) + "\n")
    temporary.replace(root / name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("native", "runtime"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arch", type=int, choices=(79, 81, 85), required=True)
    parser.add_argument("--profile", type=int, choices=(0, 1), default=0)
    parser.add_argument("--idle-power-management", type=int, choices=(0, 1), default=0)
    parser.add_argument("--sdk", type=Path, required=True)
    parser.add_argument("--soc", choices=("SM8750", "SM8850", "SM8975"))
    args = parser.parse_args()
    if (
        args.kind == "runtime"
        and args.soc != {79: "SM8750", 81: "SM8850", 85: "SM8975"}[args.arch]
    ):
        parser.error("--soc must match --arch for runtime builds")
    record_build(args)
