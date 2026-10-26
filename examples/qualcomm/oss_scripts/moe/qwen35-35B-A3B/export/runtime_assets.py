"""Build cache encodings and rotary tables from the compiled shared model."""

import json
import os
from pathlib import Path
import subprocess


def build_runtime_assets(output, checkpoint, capacity):
    from quantization.checkpoint import rope_tables

    sdk = Path(os.environ["QNN_SDK_ROOT"])
    source = Path(__file__).resolve().parents[1] / "tools/cache_metadata.cpp"
    tool = output / "cache_metadata"
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-O2",
            "-I" + str(sdk / "include/QNN"),
            str(source),
            "-ldl",
            "-o",
            str(tool),
        ],
        check=True,
    )
    subprocess.run(
        [
            str(tool.resolve()),
            str(sdk),
            str(output),
            str(output / "cache-encoding.bin"),
        ],
        check=True,
    )
    config = json.loads((checkpoint / "config.json").read_text())["text_config"]
    cos, sin = rope_tables(config, max(2048, capacity))
    cos.numpy().tofile(output / "cos.f32")
    sin.numpy().tofile(output / "sin.f32")
    (output / "RUNTIME.json").write_text(
        json.dumps(
            dict(
                capacity=capacity,
                prefill_history=capacity,
                prefill_ar_len=128,
                prefill_valid_length=True,
                rope_rows=len(cos),
                model=str(checkpoint),
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--capacity", type=int, required=True)
    args = parser.parse_args()
    build_runtime_assets(args.output, args.model, args.capacity)
