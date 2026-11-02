"""Quantize independent layer windows using original-model calibration activations."""

import argparse
import csv
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import threading
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from executorch.examples.qualcomm.oss_scripts.moe.model.qwen35 import (
    add_model_config_argument,
    load_model_config,
)
from executorch.examples.qualcomm.oss_scripts.moe.quantize.execution import (
    resolve_device,
)
from executorch.examples.qualcomm.oss_scripts.moe.quantize.provenance import (
    checkpoint_identity,
    validate_checkpoint_source,
)
from executorch.examples.qualcomm.oss_scripts.moe.checkpoint import (
    StageCheckpoint,
    artifact_stamps,
    remove_path,
    input_stamp,
    workspace_lock,
)


def publish_progress(**event):
    descriptor = os.environ.get("ET_EXPORT_PROGRESS_FD")
    if descriptor is not None:
        payload = (json.dumps(event, allow_nan=False) + "\n").encode()
        if len(payload) > 4096:
            raise ValueError("Export progress event exceeds the atomic pipe limit")
        os.write(int(descriptor), payload)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_config_argument(parser)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--qnn-binding", type=Path)
    parser.add_argument("--last-layer", type=int, default=39)
    parser.add_argument(
        "--expert-batch", type=int, choices=[1, 2, 4, 8, 16, 32], default=8
    )
    parser.add_argument("--layer-batch", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    args.device = resolve_device(args.device)
    if args.device == "cpu":
        args.layer_batch = 1
    print(
        "QUANTIZATION_EXECUTION",
        json.dumps(dict(device=args.device, layer_batch=args.layer_batch)),
        flush=True,
    )
    settings = load_model_config(args.model_config)
    if not 0 <= args.last_layer < 40:
        parser.error("Choose last layer 0..39")
    if (
        settings.quantization.attention != "gptq"
        or not settings.quantization.expert_gptq_only
    ):
        parser.error("Use the gptq.json profile: attention=gptq, expert_gptq_only=true")
    args.output = args.output.resolve()
    args.model = args.model.resolve()
    args.corpus = args.corpus.resolve()
    from executorch.examples.qualcomm.oss_scripts.moe.quantize.cache import (
        reusable_preparation,
        QUANTIZATION_VERSION,
    )

    with workspace_lock(args.output):
        if args.last_layer == 39 and reusable_preparation(
            args.output, args.model, settings, args.corpus, args.qnn_binding
        ):
            print("QUANTIZATION_REUSED", args.output, flush=True)
            for layer in range(40):
                publish_progress(ready=f"layer{layer}-router")
            publish_progress(ready="head")
            return

    sources = sorted(
        [
            *ROOT.joinpath("quantize").glob("prepare*.py"),
            ROOT / "quantize/gptq.py",
            ROOT / "quantize/reference.py",
            ROOT / "quantize/execution.py",
            *ROOT.parent.joinpath("llama/model").glob("*.py"),
            *ROOT.parent.joinpath("llama/quantize").glob("*.py"),
            ROOT.parent / "llama/static_llm_quant_recipe.py",
            ROOT.parent / "llama/gdn.py",
            ROOT.parents[1] / "custom_op/deltanet.py",
            *ROOT.parents[1].joinpath("custom_op/deltanet_support").glob("*.py"),
            *ROOT.parents[3].joinpath("backends/qualcomm/quantizer").rglob("*.py"),
            *ROOT.parents[3].joinpath("backends/qualcomm/_passes").rglob("*.py"),
            ROOT.parents[3] / "backends/qualcomm/builders/op_custom_op.py",
        ]
    )
    packages = {}
    for name in ("torch", "torchao", "transformers"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    identity = dict(
        version=1,
        quantization_version=QUANTIZATION_VERSION,
        quantization_recipe=asdict(settings.quantization),
        target=settings.target.to_dict(),
        packages=packages,
        numerical=settings.compile_settings(),
        corpus_sha256=hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
        checkpoint=checkpoint_identity(args.model),
        qnn_binding=input_stamp(args.qnn_binding) if args.qnn_binding else None,
        calibration_device=args.device,
        code_sha256=hashlib.sha256(
            b"".join(str(p).encode() + p.read_bytes() for p in sources if p.is_file())
        ).hexdigest(),
    )
    with StageCheckpoint(
        args.output,
        identity,
        "PREPARE_STATE.json",
        protected=(
            ROOT,
            args.model,
            args.corpus,
            *([args.qnn_binding] if args.qnn_binding else []),
        ),
    ) as cache:
        prepare(args, settings, cache)


def prepare(args, settings, cache):
    (args.output / "logs").mkdir(exist_ok=True)
    configuration = args.output / "MODEL_CONFIG.json"
    configuration.write_text(
        json.dumps(settings.to_dict(), ensure_ascii=False, indent=2) + "\n"
    )
    prefix = [sys.executable, "-u"]
    common = [
        "--model-config",
        str(configuration),
        "--model",
        str(args.model),
        "--output",
        str(args.output),
        "--device",
        args.device,
    ]
    binding = ["--qnn-binding", str(args.qnn_binding)] if args.qnn_binding else []
    commands = (
        json.loads((args.output / "COMMANDS.json").read_text())
        if (args.output / "COMMANDS.json").is_file()
        else []
    )

    lock = threading.RLock()
    cancelled = threading.Event()
    active = {}
    failure = []

    def abort(error):
        with lock:
            if not failure:
                failure.append(error)
            cancelled.set()
            children = tuple(active.values())
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()

    def run(name, command, outputs, *, force=False, cache_names=None):
        outputs = tuple(args.output / path for path in outputs)
        command = list(map(str, command))
        if cache_names is not None and len(cache_names) != len(outputs):
            raise ValueError("Reference checkpoint names must match layer outputs")
        records = (
            [(key, (path,)) for key, path in zip(cache_names, outputs)]
            if cache_names is not None
            else [(name, outputs)]
        )
        try:
            with lock:
                if cancelled.is_set():
                    raise RuntimeError(
                        "Preparation cancelled after another stage failed"
                    )
                if not force and all(cache.valid(key, paths) for key, paths in records):
                    print("STAGE_REUSED", name, flush=True)
                    if name.endswith("-router") or name == "head":
                        publish_progress(ready=name)
                    return True
                for path in outputs:
                    remove_path(path)
                (args.output / "QUANTIZATION_COMPLETE.json").unlink(missing_ok=True)
                commands.append(dict(stage=name, command=command))
                (args.output / "COMMANDS.json").write_text(
                    json.dumps(commands, indent=2) + "\n"
                )
                started_ns = time.monotonic_ns()
                print("STAGE_START", name, flush=True)
                for key, _ in records:
                    cache.update(key, status="running", started_ns=started_ns)
                result = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    pass_fds=(cache.lock.fileno(),),
                    text=True,
                )
                active[name] = result
            with (args.output / "logs" / f"{name}.log").open("w") as log:
                with result.stdout:
                    for line in result.stdout:
                        log.write(line)
                        log.flush()
                        if line.startswith("QUANTIZATION_METRIC "):
                            with lock:
                                print(line, end="", flush=True)
                                publish_progress(metric=line.rstrip())
                result.wait()
            row = dict(
                stage=name,
                seconds=round((time.monotonic_ns() - started_ns) / 1e9, 3),
                exit_code=result.returncode,
            )
            with lock:
                active.pop(name, None)
                with (args.output / "STAGES.csv").open("a") as file:
                    writer = csv.DictWriter(file, row)
                    if file.tell() == 0:
                        writer.writeheader()
                    writer.writerow(row)
                print("STAGE_DONE", json.dumps(row), flush=True)
                if result.returncode:
                    print(
                        (args.output / "logs" / f"{name}.log").read_text()[-6000:],
                        flush=True,
                    )
                    raise subprocess.CalledProcessError(result.returncode, command)
                for key, paths in records:
                    cache.update(
                        key,
                        status="complete",
                        outputs=artifact_stamps(paths),
                        started_ns=started_ns,
                        seconds=row["seconds"],
                    )
                if name.endswith("-router") or name == "head":
                    publish_progress(ready=name)
            return False
        except BaseException as error:
            with lock:
                for key, _ in records:
                    cache.update(key, status="failed", error=str(error))
            abort(error)
            raise

    raw = ROOT / "quantize/prepare_safetensors.py"
    data_reused = run(
        "data",
        prefix + [raw, *common, *binding, "--stage", "data", "--corpus", args.corpus],
        (
            "DATA.json",
            "INPUTS-fp32.pt",
            "embedding.bf16",
            "assets/cos.f32",
            "assets/sin.f32",
        ),
    )
    data = json.loads((args.output / "DATA.json").read_text())
    validate_checkpoint_source(args.model, data)
    if data["corpus_sha256"] != hashlib.sha256(args.corpus.read_bytes()).hexdigest():
        raise ValueError("Calibration corpus changed")

    def prepare_embedding():
        reused = run(
            "embedding",
            prefix
            + [
                ROOT / "quantize/prepare_embedding.py",
                "--model-config",
                configuration,
                "--source",
                args.output / "embedding.bf16",
                "--calibration-inputs",
                args.output / "INPUTS-fp32.pt",
                "--output",
                args.output / "embedding.int4",
            ],
            ("embedding.json", "embedding.int4"),
            force=not data_reused,
        )
        run(
            "embedding-inputs",
            prefix + [raw, *common, *binding, "--stage", "embedding-inputs"],
            ("EMBEDDING_INPUTS.json", "INPUTS.pt"),
            force=not data_reused or not reused,
        )

    (args.output / "layers").mkdir(exist_ok=True)
    for name in ["cos.f32", "sin.f32"]:
        target = args.output / "layers" / name
        if not target.exists():
            target.symlink_to(args.output / "assets" / name)

    def prepare_layer(layer, force):
        directory = args.output / f"layers/layer{layer}"
        reused = run(
            f"layer{layer}-attention",
            prefix + [raw, *common, *binding, "--stage", "attention", "--layer", layer],
            tuple(
                f"layers/layer{layer}/{name}"
                for name in (
                    "GPTQ.json",
                    "front-qdq.pt2",
                    "decode-qdq.pt2",
                    "moe-constants.npz",
                    "front-output.pt",
                )
            ),
            force=force,
        )
        force = force or not reused
        reused = run(
            f"layer{layer}-router",
            prefix
            + [
                ROOT / "quantize/prepare_router.py",
                "--worker",
                "--model-config",
                configuration,
                "--source-dir",
                args.output / "layers",
                "--sequential",
                args.output / "layer-output",
                "--inputs",
                args.output / "INPUTS-fp32.pt",
                "--normalized-inputs",
                args.output / f"reference/layer{layer}.pt",
                "--output",
                args.output / "router",
                "--layers",
                layer,
                "--device",
                args.device,
                *binding,
            ],
            tuple(
                f"router/layer{layer}/{name}"
                for name in (
                    "REPORT.json",
                    "router-ar128-qdq.pt2",
                    "router-ar1-qdq.pt2",
                )
            ),
            force=force,
        )
        force = force or not reused
        run(
            f"layer{layer}-experts",
            prefix
            + [
                ROOT / "quantize/prepare_experts.py",
                *common,
                "--source-dir",
                args.output / "layers",
                "--inputs",
                args.output / "INPUTS-fp32.pt",
                "--front-output",
                directory / "front-output.pt",
                "--router-qdq",
                args.output / "router",
                "--first-layer",
                layer,
                "--last-layer",
                layer,
                "--batch",
                args.expert_batch,
            ],
            (
                f"banks/layer{layer}/BANK.json",
                f"banks/layer{layer}/experts.n2",
                f"indices/layer{layer}/bank-index.bin",
                f"layer-output/layer{layer}.pt",
            ),
            force=force,
        )

    # CPU execution has no lookahead or background embedding work.
    pool = (
        ThreadPoolExecutor(max_workers=args.layer_batch)
        if args.device == "cuda"
        else None
    )
    embedding_pool = ThreadPoolExecutor(max_workers=1) if pool is not None else None
    pending = []
    embedding_future = None

    try:
        if embedding_pool is not None:
            embedding_future = embedding_pool.submit(prepare_embedding)
        else:
            prepare_embedding()
        reference_dirty = not data_reused
        for first in range(0, args.last_layer + 1, args.layer_batch):
            while len(pending) > args.layer_batch:
                done, remaining = wait(pending, return_when=FIRST_COMPLETED)
                pending = list(remaining)
                for future in done:
                    future.result()
            last = min(first + args.layer_batch - 1, args.last_layer)
            reused = run(
                f"reference-{first}-{last}",
                prefix
                + [
                    raw,
                    *common,
                    *binding,
                    "--stage",
                    "reference",
                    "--first-layer",
                    first,
                    "--last-layer",
                    last,
                ],
                tuple(f"reference/layer{layer}.pt" for layer in range(first, last + 1)),
                force=reference_dirty,
                cache_names=tuple(
                    f"reference-layer{layer}" for layer in range(first, last + 1)
                ),
            )
            reference_dirty = reference_dirty or not reused
            for layer in range(first, last + 1):
                if pool is not None:
                    pending.append(pool.submit(prepare_layer, layer, reference_dirty))
                else:
                    prepare_layer(layer, reference_dirty)
        if args.last_layer == 39:
            command = prefix + [raw, *common, *binding, "--stage", "head"]
            outputs = ("head/GPTQ.json", "head/head-qdq.pt2")
            if pool is not None:
                pending.append(
                    pool.submit(run, "head", command, outputs, force=reference_dirty)
                )
            else:
                run("head", command, outputs, force=reference_dirty)
        for future in pending:
            future.result()
        if embedding_future is not None:
            embedding_future.result()
    except BaseException as error:
        abort(error)
        for future in pending:
            future.cancel()
        if embedding_future is not None:
            embedding_future.cancel()
        raise failure[0]
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        if embedding_pool is not None:
            embedding_pool.shutdown(wait=True, cancel_futures=True)

    if args.last_layer == 39:
        import numpy as np

        candidates = []
        for layer in range(40):
            bank = json.loads(
                (args.output / f"banks/layer{layer}/BANK.json").read_text()
            )
            if not all(d["selected_method"] == 3 for d in bank["details"]):
                raise ValueError("Non-GPTQ expert found")
            candidates.append(
                [
                    d["expert"]
                    for d in sorted(
                        bank["details"],
                        key=lambda d: (-d["fit_routed_tokens"], d["expert"]),
                    )[:64]
                ]
            )
        np.asarray(candidates, dtype="<i4").tofile(args.output / "cold-experts.i32")
        from executorch.examples.qualcomm.oss_scripts.moe.quantize.cache import (
            prepared_artifacts,
            QUANTIZATION_VERSION,
        )

        artifacts = artifact_stamps(
            tuple(
                path
                for path in prepared_artifacts(args.output)
                if path.name != "QUANTIZATION_COMPLETE.json"
            )
        )
        (args.output / "QUANTIZATION_COMPLETE.json").write_text(
            json.dumps(
                dict(
                    format_version=QUANTIZATION_VERSION,
                    layers=40,
                    experts=10240,
                    calibration_source="original_model",
                    calibration_dtype=settings.quantization.calibration_dtype,
                    all_expert_gu_down_gptq=True,
                    old_qdq_used=False,
                    model_configuration=settings.to_dict(),
                    provenance={
                        key: cache.identity[key]
                        for key in (
                            "packages",
                            "checkpoint",
                            "corpus_sha256",
                            "qnn_binding",
                        )
                    },
                    artifacts={
                        str(Path(name).relative_to(args.output)): info
                        for name, info in artifacts.items()
                    },
                ),
                indent=2,
            )
            + "\n"
        )
    print("RAW_GPTQ_READY", args.last_layer + 1, flush=True)


if __name__ == "__main__":
    main()
