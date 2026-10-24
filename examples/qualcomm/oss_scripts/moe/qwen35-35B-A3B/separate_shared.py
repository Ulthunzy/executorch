import argparse
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import sys
from dataclasses import replace

from model_config import add_model_config_argument, load_model_config
from workflow.device import AndroidDevice


def main():
    parser = argparse.ArgumentParser(
        description="Capture SDK weight ranges and separate shared contexts, one group at a time"
    )
    add_model_config_argument(parser)
    parser.add_argument("--max-seq-len", type=int, choices=[1152, 2048])
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--head-context", type=Path)
    parser.add_argument("--groups", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--runtime-device-dir", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument(
        "--remote-root", default="/data/local/tmp/qwen35-weight-capture"
    )
    args = parser.parse_args()
    settings = load_model_config(args.model_config)
    if args.max_seq_len is not None:
        settings = replace(
            settings, context=replace(settings.context, max_seq_len=args.max_seq_len)
        ).validate()
    if (
        not args.groups
        or len(set(args.groups)) != len(args.groups)
        or any(g not in range(10) for g in args.groups)
    ):
        parser.error("Choose distinct group IDs 0..9")
    if set(args.groups) == set(range(10)) and args.head_context is None:
        parser.error("Full-model separation also needs --head-context")
    for path in [args.remote_root, args.runtime_device_dir]:
        if (
            not re.fullmatch(r"/[A-Za-z0-9_./-]+", path)
            or ".." in PurePosixPath(path).parts
        ):
            parser.error("Device paths must be absolute without shell syntax or ..")
    if not args.remote_root.startswith("/data/local/tmp/"):
        parser.error("Use a dedicated --remote-root under /data/local/tmp")
    sources = []
    for group in args.groups:
        directory = args.source_dir / f"group{group}"
        manifest = json.loads((directory / "GRAPHS.json").read_text())
        if len(manifest["contexts"]) != 1 or not manifest["weight_sharing"]:
            parser.error(f"Group {group} must contain a single shared context")
        if manifest.get("compile_settings") != settings.compile_settings():
            parser.error(f"Group {group} differs from --model-config")
        sources.append((f"group{group}", directory / next(iter(manifest["contexts"]))))
    if args.head_context:
        sources.append(("head", args.head_context))
    device = AndroidDevice(args.device)
    remote = args.remote_root
    device.adb("shell", f"mkdir -p {remote}")
    device.adb("push", args.build_dir / "capture_weights", remote + "/capture_weights")
    package = args.build_dir / "op-package/hexagon-v85/libQnnDeltaNetOpPackage_HTP.so"
    device.adb("push", package, remote + "/libQnnDeltaNetOpPackage_HTP.so")
    for name, source in sources:
        destination = args.output / name
        destination.mkdir(parents=True, exist_ok=True)
        device.adb("push", source, remote + "/context.bin")
        command = "env " + shlex.quote("LD_LIBRARY_PATH=" + args.runtime_device_dir)
        command += " " + shlex.quote(
            "ADSP_LIBRARY_PATH="
            + args.runtime_device_dir
            + ";/odm/lib64/rfsa;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/dsp"
        )
        command += f" timeout 120 {remote}/capture_weights {remote}/context.bin {remote}/libQnnDeltaNetOpPackage_HTP.so {remote}/requests.txt"
        (destination / "capture.log").write_text(device.adb("shell", command))
        requests = destination / "requests.txt"
        device.adb("pull", remote + "/requests.txt", requests)
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("pack_shared_weights.py")),
                "--context",
                str(source),
                "--requests",
                str(requests),
                "--output",
                str(destination),
            ],
            check=True,
        )
        print("SEPARATED", name, flush=True)


if __name__ == "__main__":
    main()
