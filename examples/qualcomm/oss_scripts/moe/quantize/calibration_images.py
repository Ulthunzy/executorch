"""Pinned COCO image subsets from the official Ultralytics dataset releases."""

import argparse
import fcntl
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import urllib.request
import zipfile

from executorch.examples.qualcomm.oss_scripts.moe.checkpoint import (
    remove_path,
    write_json,
)


DATASETS = {
    "coco128": (
        128,
        "61e5e3028863d8ffc3b81d6a514603954889f0edd5e4b44c4ce60b2da99aeb8e",
    ),
    "coco8": (8, "54c67fe9ef88313e021ec0e92b73c200167bb0a86633e8df8658d832cca828c9"),
}
RELEASE = "https://github.com/ultralytics/assets/releases/download/v0.0.0/"


def dataset_images(manifest):
    manifest = Path(manifest)
    metadata = json.loads(manifest.read_text())
    if not isinstance(metadata, dict) or metadata.get("dataset") not in DATASETS:
        raise ValueError("Invalid COCO calibration manifest")
    count, archive_sha = DATASETS[metadata["dataset"]]
    entries = metadata["images"]
    if (
        metadata["version"] != 1
        or metadata["archive_sha256"] != archive_sha
        or not isinstance(entries, dict)
        or len(entries) != count
    ):
        raise ValueError("Invalid COCO calibration manifest")
    paths = []
    for name, expected in entries.items():
        if not isinstance(name, str) or not isinstance(expected, dict):
            raise ValueError("Invalid calibration image entry")
        relative = PurePosixPath(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not relative.parts
            or relative.parts[0] != "images"
        ):
            raise ValueError("Invalid calibration image path")
        path = manifest.parent / name
        if (
            not path.is_file()
            or path.stat().st_size != expected.get("bytes")
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected.get("sha256")
        ):
            raise ValueError(f"Missing or corrupt calibration image: {path}")
        paths.append(path)
    return paths


def prepare_dataset(name, cache):
    count, expected_sha = DATASETS[name]
    cache = Path(cache).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    root = cache / name
    manifest = root / "DATASET.json"
    with (cache / f".{name}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            if json.loads(manifest.read_text())["dataset"] != name:
                raise ValueError("Calibration dataset name mismatch")
            paths = dataset_images(manifest)
            print("VISION_DATA_REUSED", name, len(paths), root, flush=True)
            return manifest
        except (OSError, ValueError, KeyError, TypeError):
            pass
        archive = cache / f"{name}.zip"
        with tempfile.TemporaryDirectory(prefix=f".{name}-", dir=cache) as temporary:
            temporary = Path(temporary)
            if (
                not archive.is_file()
                or hashlib.sha256(archive.read_bytes()).hexdigest() != expected_sha
            ):
                download = temporary / "download.zip"
                print("VISION_DATA_DOWNLOAD", RELEASE + name + ".zip", flush=True)
                with urllib.request.urlopen(
                    RELEASE + name + ".zip", timeout=60
                ) as response, download.open("wb") as output:
                    shutil.copyfileobj(response, output)
                if hashlib.sha256(download.read_bytes()).hexdigest() != expected_sha:
                    raise ValueError("COCO archive checksum mismatch")
                download.replace(archive)
            staging = temporary / name
            staging.mkdir()
            entries = {}
            from PIL import Image

            with zipfile.ZipFile(archive) as package:
                for item in package.infolist():
                    path = PurePosixPath(item.filename)
                    if (
                        item.is_dir()
                        or len(path.parts) < 4
                        or path.parts[:2] != (name, "images")
                    ):
                        continue
                    if path.is_absolute() or ".." in path.parts:
                        raise ValueError("Invalid COCO archive path")
                    if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                        continue
                    if item.file_size > 32 * 1024 * 1024 or len(entries) >= count:
                        raise ValueError("Unexpected COCO image payload")
                    relative = Path(*path.parts[1:])
                    destination = staging / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    payload = package.read(item)
                    destination.write_bytes(payload)
                    with Image.open(destination) as image:
                        image.verify()
                    entries[relative.as_posix()] = dict(
                        bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()
                    )
            if len(entries) != count:
                raise ValueError(f"Expected {count} COCO images, found {len(entries)}")
            write_json(
                staging / "DATASET.json",
                dict(
                    version=1,
                    dataset=name,
                    source=RELEASE + name + ".zip",
                    archive_sha256=expected_sha,
                    images=dict(sorted(entries.items())),
                ),
            )
            remove_path(root)
            staging.replace(root)
        print("VISION_DATA_READY", name, count, root, flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    prepare_dataset(args.dataset, args.cache)
