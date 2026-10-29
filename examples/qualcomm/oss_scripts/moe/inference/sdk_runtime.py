"""Deploy the selected SDK's Android runtime and target-specific DSP libraries."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex

from executorch.examples.qualcomm.oss_scripts.moe.inference.device import AndroidDevice
from executorch.examples.qualcomm.oss_scripts.moe.target import (
    SOC_MODELS,
    TargetConfig,
)


def runtime_files(sdk, target):
    android = sdk / "lib/aarch64-android"
    dsp = sdk / f"lib/hexagon-{target.dsp_arch}/unsigned"
    arch = target.dsp_arch.upper()
    required = [
        android / "libQnnHtp.so",
        android / "libQnnSystem.so",
        android / "libQnnHtpPrepare.so",
        android / f"libQnnHtp{arch}Stub.so",
        dsp / f"libQnnHtp{arch}Skel.so",
    ]
    required.extend(
        path
        for path in (
            dsp / f"libQnnHtp{arch}.so",
            dsp / f"libqnnhtp{target.dsp_arch}.cat",
        )
        if path.is_file()
    )
    return required


def deploy_runtime(sdk, target, device, remote, manifest):
    sdk = Path(sdk).resolve()
    if (
        not re.fullmatch(r"/data/local/tmp/[A-Za-z0-9_./-]+", remote)
        or ".." in Path(remote).parts
    ):
        raise ValueError("Use an absolute runtime directory under /data/local/tmp")
    files = runtime_files(sdk, target)
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing SDK runtime libraries:\n" + "\n".join(missing))
    device.check_target(target)
    device.adb("shell", "mkdir -p " + shlex.quote(remote))
    device.adb("push", "--sync", *files, remote + "/")
    metadata = dict(
        target=target.to_dict(),
        sdk=str(sdk),
        device=device.serial,
        remote=remote,
        libraries={
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files
        },
    )
    manifest = Path(manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", type=Path, required=True)
    parser.add_argument("--soc-model", choices=SOC_MODELS, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--remote", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    deploy_runtime(
        args.sdk,
        TargetConfig.resolve(args.soc_model),
        AndroidDevice(args.device),
        args.remote,
        args.manifest,
    )


if __name__ == "__main__":
    main()
