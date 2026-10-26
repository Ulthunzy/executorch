"""Serial safetensors -> GPTQ QDQ/banks; QNN compilation remains a separate step."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from workflow.model_config import add_model_config_argument, load_model_config


def main():
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
    args = parser.parse_args()
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
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "logs").mkdir(exist_ok=True)
    configuration = args.output / "MODEL_CONFIG.json"
    if (
        configuration.is_file()
        and json.loads(configuration.read_text()) != settings.to_dict()
    ):
        parser.error(
            "Output contains a different model configuration; use another directory"
        )
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

    def run(name, command, sentinel):
        if sentinel.is_file():
            print("STAGE_REUSED", name, flush=True)
            return
        command = list(map(str, command))
        commands.append(dict(stage=name, command=command))
        (args.output / "COMMANDS.json").write_text(
            json.dumps(commands, indent=2) + "\n"
        )
        print("STAGE_START", name, flush=True)
        started = time.monotonic()
        with (args.output / "logs" / f"{name}.log").open("w") as log:
            result = subprocess.run(
                ["/usr/bin/time", "-v", *command], stdout=log, stderr=subprocess.STDOUT
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
            print(
                (args.output / "logs" / f"{name}.log").read_text()[-6000:], flush=True
            )
            raise subprocess.CalledProcessError(result.returncode, command)
        if not sentinel.is_file():
            raise RuntimeError(f"{name} did not produce {sentinel}")

    raw = ROOT / "quantization/prepare_safetensors.py"
    run(
        "data",
        prefix + [raw, *common, *binding, "--stage", "data", "--corpus", args.corpus],
        args.output / "DATA.json",
    )
    data = json.loads((args.output / "DATA.json").read_text())
    if (
        data["index_sha256"]
        != hashlib.sha256(
            (args.model / "model.safetensors.index.json").read_bytes()
        ).hexdigest()
    ):
        raise ValueError("Checkpoint index changed")
    if data["corpus_sha256"] != hashlib.sha256(args.corpus.read_bytes()).hexdigest():
        raise ValueError("Calibration corpus changed")
    for name, metadata in data["safetensors"].items():
        stat = (args.model / name).stat()
        if (stat.st_size, stat.st_mtime_ns) != (
            metadata["bytes"],
            metadata["mtime_ns"],
        ):
            raise ValueError(f"Checkpoint shard changed: {name}")
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
            args.output / "INPUTS.pt",
            "--output",
            args.output / "embedding.int4",
        ],
        args.output / "embedding.json",
    )
    run(
        "embedding-inputs",
        prefix + [raw, *common, *binding, "--stage", "embedding-inputs"],
        args.output / "EMBEDDING_INPUTS.json",
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
            directory / "GPTQ.json",
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
            args.output / f"router/layer{layer}/REPORT.json",
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
            args.output / f"banks/layer{layer}/BANK.json",
        )
    if args.last_layer == 39:
        run(
            "head",
            prefix + [raw, *common, *binding, "--stage", "head"],
            args.output / "head/GPTQ.json",
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
