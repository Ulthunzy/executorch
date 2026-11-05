import json
import os
from pathlib import Path
import subprocess
import tempfile


def validate_qnn_io_memory(context):
    sdk = Path(os.environ["QNN_SDK_ROOT"])
    utility = sdk / "bin/x86_64-linux-clang/qnn-context-binary-utility"
    with tempfile.TemporaryDirectory(prefix="moe-qnn-io-") as temporary:
        metadata = Path(temporary) / "context.json"
        subprocess.run(
            [
                str(utility),
                f"--context_binary={context}",
                f"--json_file={metadata}",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        graphs = json.loads(metadata.read_text())["info"]["graphs"]
    count = 0
    for graph in graphs:
        info = graph["info"]
        for tensor in (*info["graphInputs"], *info["graphOutputs"]):
            if tensor["info"]["memType"] != "QNN_TENSORMEMTYPE_MEMHANDLE":
                raise ValueError(
                    f"{context}: QNN IO was compiled as RAW; rebuild the "
                    "Qualcomm host bindings before exporting shared-buffer graphs"
                )
            count += 1
    if not count:
        raise ValueError(f"{context}: no graph IO metadata")
    return dict(io_memory_type="memhandle", io_tensor_count=count)


def runtime_assets(args):
    if args.shared_model:
        manifest = args.shared_model / "group0/GRAPHS.json"
        if manifest.is_file() and json.loads(manifest.read_text()).get(
            "prefill_valid_length"
        ):
            return args.shared_model
    return args.artifacts / "assets"


def validate_rope(directory, capacity):
    cos = directory / "cos.f32"
    sin = directory / "sin.f32"
    if (
        not cos.is_file()
        or not sin.is_file()
        or cos.stat().st_size != sin.stat().st_size
        or cos.stat().st_size % (32 * 4)
        or cos.stat().st_size < capacity * 32 * 4
    ):
        raise ValueError(f"{directory}: RoPE tables do not cover {capacity} positions")
