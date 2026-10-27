"""Serial safetensors -> GPTQ QDQ/banks; QNN compilation remains a separate step."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from workflow.model_config import add_model_config_argument, load_model_config
from quantization.provenance import validate_checkpoint_source
from workflow.checkpoint import (
    StageCheckpoint,
    artifact_stamps,
    remove_path,
)


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
    args = parser.parse_args(argv)
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
    index = json.loads((args.model / "model.safetensors.index.json").read_text())
    from workflow.export_pipeline import file_stamp

    sources = sorted(
        [
            *ROOT.joinpath("quantization").glob("prepare*.py"),
            ROOT / "quantization/quantize.py",
            *ROOT.parents[1].joinpath("llama/model").glob("*.py"),
            *ROOT.parents[1].joinpath("llama/quantize").glob("*.py"),
            ROOT.parents[1] / "llama/static_llm_quant_recipe.py",
            ROOT.parents[1] / "llama/gdn.py",
            ROOT.parents[2] / "custom_op/deltanet.py",
            *ROOT.parents[4].joinpath("backends/qualcomm/quantizer").rglob("*.py"),
            *ROOT.parents[4].joinpath("backends/qualcomm/_passes").rglob("*.py"),
            ROOT.parents[4] / "backends/qualcomm/builders/op_custom_op.py",
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
        packages=packages,
        numerical=settings.compile_settings(),
        corpus_sha256=hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
        checkpoint={
            str(path): file_stamp(path)
            for path in [
                args.model / "config.json",
                args.model / "model.safetensors.index.json",
                *(
                    args.model / name
                    for name in (
                        "tokenizer.json",
                        "tokenizer_config.json",
                        "generation_config.json",
                    )
                    if (args.model / name).is_file()
                ),
                *(
                    args.model / name
                    for name in sorted(set(index["weight_map"].values()))
                ),
            ]
        },
        qnn_binding=file_stamp(args.qnn_binding) if args.qnn_binding else None,
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
    ]
    binding = ["--qnn-binding", str(args.qnn_binding)] if args.qnn_binding else []
    commands = (
        json.loads((args.output / "COMMANDS.json").read_text())
        if (args.output / "COMMANDS.json").is_file()
        else []
    )

    def run(name, command, outputs):
        outputs = tuple(args.output / path for path in outputs)
        if cache.reusable(name, outputs):
            print("STAGE_REUSED", name, flush=True)
            return
        for path in outputs:
            remove_path(path)
        (args.output / "QUANTIZATION_COMPLETE.json").unlink(missing_ok=True)
        command = list(map(str, command))
        commands.append(dict(stage=name, command=command))
        (args.output / "COMMANDS.json").write_text(
            json.dumps(commands, indent=2) + "\n"
        )
        print("STAGE_START", name, flush=True)
        started = time.monotonic()
        cache.update(name, status="running")
        with (args.output / "logs" / f"{name}.log").open("w") as log:
            result = subprocess.run(
                ["/usr/bin/time", "-v", *command],
                stdout=log,
                stderr=subprocess.STDOUT,
                pass_fds=(cache.lock.fileno(),),
            )
        row = dict(
            stage=name,
            seconds=round(time.monotonic() - started, 3),
            exit_code=result.returncode,
        )
        with (args.output / "STAGES.csv").open("a") as file:
            writer = csv.DictWriter(file, row)
            writer.writeheader() if file.tell() == 0 else None
            writer.writerow(row)
        print("STAGE_DONE", json.dumps(row), flush=True)
        if result.returncode:
            cache.update(name, status="failed", exit_code=result.returncode)
            print(
                (args.output / "logs" / f"{name}.log").read_text()[-6000:], flush=True
            )
            raise subprocess.CalledProcessError(result.returncode, command)
        stamps = artifact_stamps(outputs)
        cache.update(name, status="complete", outputs=stamps)

    raw = ROOT / "quantization/prepare_safetensors.py"
    run(
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
    run(
        "embedding",
        prefix
        + [
            ROOT / "quantization/prepare_embedding.py",
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
    )
    run(
        "embedding-inputs",
        prefix + [raw, *common, *binding, "--stage", "embedding-inputs"],
        ("EMBEDDING_INPUTS.json", "INPUTS.pt"),
    )
    (args.output / "layers").mkdir(exist_ok=True)
    for name in ["cos.f32", "sin.f32"]:
        target = args.output / "layers" / name
        if not target.exists():
            target.symlink_to(args.output / "assets" / name)
    for layer in range(args.last_layer + 1):
        directory = args.output / f"layers/layer{layer}"
        run(
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
        )
        run(
            f"layer{layer}-router",
            prefix
            + [
                ROOT / "quantization/prepare_router.py",
                "--model-config",
                configuration,
                "--source-dir",
                args.output / "layers",
                "--sequential",
                args.output / "sequential",
                "--inputs",
                args.output / "INPUTS.pt",
                "--normalized-inputs",
                directory / "front-output.pt",
                "--output",
                args.output / "router",
                "--layers",
                layer,
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
        )
        run(
            f"layer{layer}-experts",
            prefix
            + [
                ROOT / "quantization/prepare_experts.py",
                *common,
                "--source-dir",
                args.output / "layers",
                "--inputs",
                args.output / "INPUTS.pt",
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
                f"sequential/layer{layer}.pt",
            ),
        )
    if args.last_layer == 39:
        run(
            "head",
            prefix + [raw, *common, *binding, "--stage", "head"],
            ("head/GPTQ.json", "head/head-qdq.pt2"),
        )
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
        (args.output / "QUANTIZATION_COMPLETE.json").write_text(
            json.dumps(
                dict(
                    layers=40,
                    experts=10240,
                    all_expert_gu_down_gptq=True,
                    old_qdq_used=False,
                    model_configuration=settings.to_dict(),
                ),
                indent=2,
            )
            + "\n"
        )
    print("RAW_GPTQ_READY", args.last_layer + 1, flush=True)


if __name__ == "__main__":
    main()
