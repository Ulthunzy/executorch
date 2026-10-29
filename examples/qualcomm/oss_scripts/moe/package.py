"""One standard PTE with file-backed named data for every model resource."""

import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import zlib

from executorch.examples.qualcomm.oss_scripts.moe.checkpoint import StageCheckpoint

MANIFEST_KEY = "__qwen35_package_manifest__"
CHUNK_BYTES = 8 * 1024 * 1024


def relative_name(name):
    path = PurePosixPath(name)
    if (
        not name
        or name.startswith("EXTRACT_STATE.json")
        or path.is_absolute()
        or str(path) != name
        or ".." in path.parts
        or "\\" in name
        or "\0" in name
    ):
        raise ValueError(f"Unsafe model resource name: {name!r}")
    return path


def file_digest(path):
    digest = hashlib.sha256()
    crc = 0
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_BYTES):
            digest.update(chunk)
            crc = zlib.crc32(chunk, crc)
    return dict(bytes=path.stat().st_size, sha256=digest.hexdigest(), crc32=crc)


def create_package(destination, files, links=None, plans=()):
    from executorch.exir._serialize._cord import FileBackedData
    from executorch.exir._serialize._named_data_store import NamedDataStoreOutput
    from executorch.exir._serialize._program import (
        PTEFile,
        deserialize_pte_binary,
        serialize_pte_binary,
    )
    from executorch.exir._serialize.data_serializer import DataEntry
    from executorch.exir.schema import Program, Buffer, SubsegmentOffsets

    destination = Path(destination)
    files = dict(sorted((name, Path(path)) for name, path in files.items()))
    for name, path in files.items():
        relative_name(name)
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing model resource: {path}")
    if MANIFEST_KEY in files:
        raise ValueError("Reserved model resource name")
    snapshots = {
        name: (path.stat().st_size, path.stat().st_mtime_ns)
        for name, path in files.items()
    }
    assets = {name: file_digest(path) for name, path in files.items()}
    manifest = dict(version=1, files=assets, links=links or {})
    identity = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    manifest["id"] = identity
    program = Program(0, [], [Buffer(b"")], [], [], SubsegmentOffsets(0, []))
    # Only the small host plans are deserialized. Multi-GiB resources stay on disk.
    for path in plans:
        current = deserialize_pte_binary(Path(path).read_bytes()).program
        for plan in current.execution_plan:
            plan = copy.deepcopy(plan)
            for delegate in plan.delegates:
                delegate.processed.index += len(program.backend_delegate_data)
            program.execution_plan.append(plan)
        program.backend_delegate_data.extend(current.backend_delegate_data)
    buffers = []
    entries = {}
    unique = {}
    for name, path in files.items():
        key = (assets[name]["bytes"], assets[name]["sha256"])
        if key not in unique:
            unique[key] = len(buffers)
            buffers.append(FileBackedData(str(path)))
        entries[name] = DataEntry(unique[key], 4096, None)
    entries[MANIFEST_KEY] = DataEntry(len(buffers), 4096, None)
    buffers.append(json.dumps(manifest, sort_keys=True).encode())
    payload = PTEFile(program, named_data=NamedDataStoreOutput(buffers, entries, {}))
    temporary = destination.with_name(destination.name + ".tmp")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("wb") as stream:
            serialize_pte_binary(payload, segment_alignment=4096).write_to_file(stream)
            stream.flush()
            os.fsync(stream.fileno())
        if snapshots != {
            name: (p.stat().st_size, p.stat().st_mtime_ns) for name, p in files.items()
        }:
            raise RuntimeError("Model resources changed while packing")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return manifest


class PackageReader:
    def __init__(self, path):
        from executorch.exir._serialize._program import _get_extended_header
        from executorch.exir._serialize._flatbuffer_program import (
            _flatbuffer_to_program,
        )

        self.path = Path(path).resolve()
        size = self.path.stat().st_size
        with self.path.open("rb") as stream:
            head = stream.read(64)
            header = _get_extended_header(head)
            if (
                header is None
                or not header.is_valid()
                or not 64 <= header.program_size <= 16 * 1024 * 1024
                or header.segment_base_offset > size
                or header.segment_data_size > size - header.segment_base_offset
            ):
                raise ValueError("Invalid or truncated model PTE")
            stream.seek(0)
            program = _flatbuffer_to_program(stream.read(header.program_size))
            self.entries = {}
            for item in program.named_data or ():
                if item.key in self.entries or not 0 <= item.segment_index < len(
                    program.segments
                ):
                    raise ValueError("Invalid named data table")
                segment = program.segments[item.segment_index]
                offset = header.segment_base_offset + segment.offset
                if offset > size or segment.size > size - offset:
                    raise ValueError("Truncated model resource")
                self.entries[item.key] = (offset, segment.size)
            if MANIFEST_KEY not in self.entries:
                raise ValueError("PTE is not a complete MoE model package")
            offset, length = self.entries[MANIFEST_KEY]
            if length > 8 * 1024 * 1024:
                raise ValueError("Oversized model manifest")
            stream.seek(offset)
            self.manifest = json.loads(stream.read(length))
        manifest = dict(self.manifest)
        identity = manifest.pop("id")
        if (
            manifest["version"] != 1
            or hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
            != identity
            or set(self.entries) != set(manifest["files"]) | {MANIFEST_KEY}
        ):
            raise ValueError("Model manifest does not match PTE named data")
        for name, info in manifest["files"].items():
            relative_name(name)
            if self.entries[name][1] != info["bytes"]:
                raise ValueError(f"Incorrect resource length: {name}")

    def extract(self, cache=None):
        base = Path(cache) if cache else self.path.parent / ".qwen35-model-cache"
        target = base / self.manifest["id"]
        identity = dict(package=self.manifest["id"])
        if target.is_symlink():
            raise ValueError("Model cache root cannot be a symlink")
        with StageCheckpoint(target, identity, "EXTRACT_STATE.json") as state:
            with self.path.open("rb") as source:
                for name, info in self.manifest["files"].items():
                    output = target / name
                    previous = state.state["stages"].get(name, {})
                    if output.is_file() and not output.is_symlink():
                        current = output.stat()
                        if (
                            previous
                            == dict(bytes=current.st_size, mtime_ns=current.st_mtime_ns)
                            and current.st_size == info["bytes"]
                        ):
                            continue
                    for parent in output.parents:
                        if parent == target:
                            break
                        if parent.is_symlink():
                            raise ValueError("Symlink in model cache path")
                    output.parent.mkdir(parents=True, exist_ok=True)
                    temporary = output.with_name(output.name + ".tmp")
                    source.seek(self.entries[name][0])
                    remaining = info["bytes"]
                    digest = hashlib.sha256()
                    crc = 0
                    try:
                        with temporary.open("wb") as stream:
                            while remaining:
                                chunk = source.read(min(remaining, CHUNK_BYTES))
                                if not chunk:
                                    raise ValueError(
                                        f"Truncated model resource: {name}"
                                    )
                                remaining -= len(chunk)
                                digest.update(chunk)
                                crc = zlib.crc32(chunk, crc)
                                stream.write(chunk)
                            stream.flush()
                            os.fsync(stream.fileno())
                        if crc != info["crc32"] or digest.hexdigest() != info["sha256"]:
                            raise ValueError(f"Corrupt model resource: {name}")
                        temporary.replace(output)
                    finally:
                        temporary.unlink(missing_ok=True)
                    current = output.stat()
                    state.update(
                        name, bytes=current.st_size, mtime_ns=current.st_mtime_ns
                    )
            for name, link in self.manifest["links"].items():
                relative_name(name)
                output = target / name
                resolved = (output.parent / link).resolve()
                if not resolved.is_relative_to(target) or resolved == target:
                    raise ValueError(f"Unsafe resource link: {name}")
                output.parent.mkdir(parents=True, exist_ok=True)
                if output.is_symlink():
                    if output.resolve() == resolved:
                        continue
                    output.unlink()
                if output.exists():
                    raise ValueError(f"Resource link conflicts with a file: {name}")
                output.symlink_to(link)
        return target


def pack_export(config):
    out = config.output
    files = {"MODEL_CONFIG.json": out / "MODEL_CONFIG.json"}
    image_cold = ()
    if (
        config.model_settings.target.soc_model == "SM8850"
        and "vision" in config.components
    ):
        from executorch.examples.qualcomm.oss_scripts.moe.inference.image_cold_profile import (
            ensure_image_cold_profile,
        )

        image_cold = ensure_image_cold_profile(out)
    if (out / "device.json").is_file():
        files["device.json"] = out / "device.json"
    links = {}
    plans = []
    text = "text" in config.components
    if text:
        text_cold = ()
        if config.model_settings.target.soc_model == "SM8850":
            from executorch.examples.qualcomm.oss_scripts.moe.inference.text_cold_profile import (
                ensure_text_cold_profile,
            )

            text_cold = ensure_text_cold_profile(out)
        for name in ("tokenizer.json", "config.json", "generation_config.json"):
            files[name] = config.model / name
        for name in ("tokenizer_config.json", "preprocessor_config.json"):
            if (config.model / name).is_file():
                files[name] = config.model / name
        for phase in config.phases:
            for suffix in ("pte", "json"):
                files[f"plans/{phase}.{suffix}"] = out / f"plans/{phase}.{suffix}"
            plans.append(out / f"plans/{phase}.pte")
        if "prefill" in config.phases:
            files["plans/PLANS.json"] = out / "plans/PLANS.json"
        for group in (*[f"group{i}" for i in range(10)], "head"):
            for path in (out / "shared-model" / group).iterdir():
                if path.is_file() and (
                    path.name in ("graph.qnn", "weights.bin")
                    or path.suffix in (".json", ".io")
                ):
                    files[f"shared-model/{group}/{path.name}"] = path
        for path in (out / "shared-model").iterdir():
            if path.is_file():
                files[f"shared-model/{path.name}"] = path
        for name in ("cache-encoding.bin", "cos.f32", "sin.f32"):
            files[f"assets/{name}"] = out / "assets" / name
        for layer in range(40):
            for name in ("BANK.json", "experts.n2"):
                files[f"banks/layer{layer}/{name}"] = out / f"banks/layer{layer}/{name}"
            files[f"indices/layer{layer}/bank-index.bin"] = (
                out / f"indices/layer{layer}/bank-index.bin"
            )
        embedding = (
            "embedding.int4"
            if config.model_settings.quantization.embedding == "int4_g32"
            else "embedding.bf16"
        )
        for name in (embedding, "cold-experts.i32"):
            files[name] = out / name
        for asset in (*image_cold, *text_cold):
            files[asset.name] = asset
            links[f"model/prefill/{asset.name}"] = "../../" + asset.name
        for stage in ("prefill", "decode"):
            for name, source in (
                ("indices", "indices"),
                (embedding, embedding),
                ("compact" if stage == "prefill" else "banks", "banks"),
            ):
                links[f"model/{stage}/{name}"] = f"../../{source}"
        links.update(
            {
                "model/prefill/cold-experts.i32": "../../cold-experts.i32",
                "model/prefill/plans": "../../plans",
                "model/decode/decode.pte": "../../plans/decode.pte",
                "model/decode/cache-encoding.bin": "../../assets/cache-encoding.bin",
                "model/decode/prefill-logits.f32": "../prefill/pte-0-logits.f32",
                "model/MODEL_CONFIG.json": "../MODEL_CONFIG.json",
                "model/model-config.json": "../config.json",
                "model/base-cos.f32": "../shared-model/cos.f32",
                "model/base-sin.f32": "../shared-model/sin.f32",
            }
        )
        for name in (
            "prefill-routes.i32",
            "prefill-history-ids.i32",
            "prefill-history-stamps.u64",
        ):
            links[f"model/decode/{name}"] = f"../prefill/{name}"
        for layer in range(40):
            for cache in range(2):
                name = f"layer{layer}-cache{cache}.u16"
                links[f"model/decode/cache-prefill/{name}"] = (
                    f"../../prefill/pte-0-{name}"
                )
    if "vision" in config.components:
        for name in ("vision.bin", "VISION.json"):
            files[f"vision/{name}"] = out / "vision" / name
        links["model/prefill/vision.bin"] = "../../vision/vision.bin"
        links["model/VISION.json"] = "../vision/VISION.json"
    return create_package(out / "model.pte", files, links, plans)
