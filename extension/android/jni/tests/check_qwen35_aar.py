"""Verify Qwen35 Java APIs, native architectures and Android ELF alignment."""

import argparse
import io
from pathlib import Path
import struct
import zipfile


def check_elf(aar, entry):
    name = Path(entry).name
    with aar.open(entry) as file:
        header = file.read(64)
        if len(header) != 64 or header[:4] != b"\x7fELF" or header[5] != 1:
            raise ValueError(f"Invalid native ELF header: {name}")
        identity = (header[4], struct.unpack_from("<H", header, 18)[0])
        if identity == (1, 164):
            return identity
        if identity != (2, 183):
            raise ValueError(f"Invalid native ELF architecture: {name}")
        offset = struct.unpack_from("<Q", header, 32)[0]
        size, count = struct.unpack_from("<HH", header, 54)
        if (
            size != 56
            or not count
            or offset + size * count > aar.getinfo(entry).file_size
        ):
            raise ValueError(f"Invalid ELF program headers: {name}")
        file.seek(offset)
        loads = 0
        for _ in range(count):
            program = file.read(size)
            if len(program) != size:
                raise ValueError(f"Truncated ELF program header: {name}")
            kind, _, position, address, _, _, _, alignment = struct.unpack(
                "<IIQQQQQQ", program
            )
            if kind == 1:
                loads += 1
                if alignment < 16384 or position % 16384 != address % 16384:
                    raise ValueError(
                        f"Android ELF LOAD segment is not 16 KB aligned: {name} "
                        f"(alignment={alignment:#x}, offset={position:#x}, "
                        f"address={address:#x})"
                    )
        if not loads:
            raise ValueError(f"Missing ELF LOAD segments: {name}")
    return identity


def check_aar(path, modules, arch="81"):
    required = {
        "dense": {
            "libexecutorch.so": (2, 183),
            "libqnn_executorch_backend.so": (2, 183),
            "libQnnDeltaNetOpPackage.so": (2, 183),
            "libQnnDeltaNetOpPackage_HTP.so": (1, 164),
        },
        "moe": {"libet_infer.so": (2, 183), "libet_qnnop.so": (1, 164)},
    }
    with zipfile.ZipFile(path) as aar:
        libraries = {
            Path(name).name: name
            for name in aar.namelist()
            if name.startswith("jni/arm64-v8a/") and name.endswith(".so")
        }
        expected = {
            "libQnnHtp.so": (2, 183),
            "libQnnSystem.so": (2, 183),
            "libQnnHtpPrepare.so": (2, 183),
            f"libQnnHtpV{arch}Stub.so": (2, 183),
            f"libQnnHtpV{arch}Skel.so": (1, 164),
        }
        for module in modules:
            expected.update(required[module])
        missing = sorted(set(expected) - libraries.keys())
        if missing:
            raise ValueError("Missing native libraries: " + ", ".join(missing))
        identities = {name: check_elf(aar, entry) for name, entry in libraries.items()}
        for name, identity in expected.items():
            if identities[name] != identity:
                raise ValueError(f"Invalid native ELF architecture: {name}")
        for other in {"79", "81", "85"} - {arch}:
            if any(f"V{other}" in name for name in libraries):
                raise ValueError(f"Unexpected V{other} runtime in V{arch} AAR")
        with zipfile.ZipFile(io.BytesIO(aar.read("classes.jar"))) as jar:
            for module in modules:
                name = "Qwen35DenseModule" if module == "dense" else "Qwen35MoeModule"
                entry = "org/pytorch/executorch/extension/llm/" + name + ".class"
                if entry not in jar.namelist():
                    raise ValueError(f"Missing Java API: {name}")
    return sorted(libraries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aar", type=Path)
    parser.add_argument("--modules", nargs="+", choices=("dense", "moe"), required=True)
    parser.add_argument("--arch", choices=("79", "81", "85"), default="81")
    args = parser.parse_args()
    try:
        libraries = check_aar(args.aar, args.modules, args.arch)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        parser.exit(1, f"Invalid Qwen35 AAR: {error}\n")
    print(
        f"QWEN35_AAR_OK modules={','.join(args.modules)} arch=v{args.arch} libraries={len(libraries)}"
    )
