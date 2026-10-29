"""Validate and reuse immutable vision artifacts without running an exporter."""

import argparse
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile


from executorch.examples.qualcomm.oss_scripts.moe.model.qwen35 import (
    load_model_config,
    ModelConfig,
)  # noqa: E402
from executorch.examples.qualcomm.oss_scripts.moe.inference.encoder import (
    vision_metadata,
)  # noqa: E402

VISION_FILES = ("VISION.json", "vision.bin", "vision.pte")


def validate_vision_source(source, model, model_config, image_height, image_width):
    source, model = Path(source), Path(model)
    if not source.is_dir():
        raise NotADirectoryError(source)
    for name in VISION_FILES:
        path = source / name
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size == 0:
            raise ValueError(f"Vision artifact must be nonempty: {path}")
    if type(image_height) is not int or type(image_width) is not int:
        raise ValueError("Image dimensions must be integers")
    metadata = vision_metadata(model, source)
    if (
        type(metadata["image_height"]) is not int
        or type(metadata["image_width"]) is not int
        or metadata["image_height"] != image_height
        or metadata["image_width"] != image_width
    ):
        raise ValueError("Vision artifact image dimensions differ from the request")
    settings = (
        model_config.validate()
        if isinstance(model_config, ModelConfig)
        else load_model_config(model_config)
    )
    settings.target.check_manifest(metadata)
    if settings.quantization.vision.endswith("_gptq"):
        reports = metadata.get("gptq")
        if not isinstance(reports, list) or not reports:
            raise ValueError("Vision recipe requires fixed-grid GPTQ metadata")
        for report in reports:
            if (
                not isinstance(report, dict)
                or report.get("method") != "fixed_grid_gptq"
                or type(report.get("bits")) is not int
                or report["bits"] != 8
                or not isinstance(report.get("module"), str)
                or not report["module"]
                or type(report.get("damping")) not in (int, float)
                or not math.isclose(
                    report["damping"], settings.quantization.matrix_gptq_damping
                )
            ):
                raise ValueError(
                    "Vision GPTQ metadata differs from the selected recipe"
                )
        if not any(report["module"] == "visual.pos_embed" for report in reports):
            raise ValueError("Vision GPTQ metadata is missing visual.pos_embed")
    return metadata


def file_stamp(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return dict(bytes=path.stat().st_size, sha256=digest.hexdigest())


def reuse_vision(source, output, model, model_config, image_height, image_width):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError(
            "Vision source and output must be separate, non-nested directories"
        )
    metadata = validate_vision_source(
        source, model, model_config, image_height, image_width
    )
    names = (*VISION_FILES, "VISION_REUSE.json")
    for name in names:
        path = output / name
        if path.exists() or path.is_symlink():
            raise FileExistsError(
                f"Refusing to replace existing vision artifact: {path}"
            )
    output.mkdir(parents=True, exist_ok=True)
    record = dict(
        version=1,
        source=str(source),
        output=str(output),
        model=str(Path(model).resolve()),
        model_config=str(Path(model_config).resolve()),
        model_config_sha256=file_stamp(Path(model_config))["sha256"],
        image_height=image_height,
        image_width=image_width,
        files={},
    )
    published = []
    try:
        with tempfile.TemporaryDirectory(prefix=".vision-reuse-", dir=output) as tmp:
            staging = Path(tmp)
            for name in VISION_FILES:
                original, destination = source / name, staging / name
                before = file_stamp(original)
                try:
                    os.link(original.resolve(strict=True), destination)
                    method = "hardlink"
                except OSError as error:
                    if error.errno != errno.EXDEV:
                        raise
                    shutil.copy2(original, destination)
                    method = "copy2"
                after = file_stamp(destination)
                if before != after:
                    raise ValueError(f"Vision source changed while reusing {original}")
                record["files"][name] = dict(
                    source=dict(path=str(original), **before),
                    destination=dict(path=str(output / name), **after),
                    method=method,
                )
            if (
                validate_vision_source(
                    staging, model, model_config, image_height, image_width
                )
                != metadata
            ):
                raise ValueError("Vision metadata changed while reusing the source")
            (staging / "VISION_REUSE.json").write_text(
                json.dumps(record, indent=2) + "\n"
            )
            for name in names:
                path = output / name
                if path.exists() or path.is_symlink():
                    raise FileExistsError(path)
            for name in names:
                path = output / name
                (staging / name).replace(path)
                published.append(path)
    except BaseException:
        for path in reversed(published):
            path.unlink()
        raise
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--image-height", type=int, required=True)
    parser.add_argument("--image-width", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        record = reuse_vision(
            args.source,
            args.output,
            args.model,
            args.model_config,
            args.image_height,
            args.image_width,
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print("REUSED_VISION", record["output"], flush=True)
    return record


if __name__ == "__main__":
    main()
