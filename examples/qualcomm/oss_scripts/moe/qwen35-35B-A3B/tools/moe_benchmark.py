"""Run the 2048-context, AR128, W2 matrix against the supplied SM8975 rates."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = {
    "text": {
        256: ((159.66, 175.40), (10.84, 12.86)),
        512: ((272.56, 301.47), (10.91, 12.89)),
        1024: ((352.44, 332.09), (10.02, 11.60)),
    },
    "image": {
        256: ((91.03, 119.20), (11.30, 13.58)),
        512: ((165.33, 167.76), (10.76, 12.81)),
        1024: ((265.63, 251.90), (9.81, 12.19)),
    },
}


def read_measurement(path, tokens, modality="text"):
    if modality not in REFERENCE:
        raise ValueError(f"Unsupported modality: {modality}")
    text = path.read_text(errors="replace")
    matches = re.findall(r"CHAT_STATS turn=\d+ (\{[^\n]+\})", text)
    if len(matches) != 1:
        raise ValueError(f"Expected one CHAT_STATS record in {path}")
    stats = json.loads(matches[0])
    if (stats["prompt_tokens"], stats["generated_tokens"]) != (tokens, tokens):
        raise ValueError(f"Truncated or mismatched token counts: {stats}")
    if stats["context_capacity"] != 2048:
        raise ValueError("Comparison requires 2048 context capacity")
    if stats["stop_reason"] != "max_new_tokens":
        raise ValueError(f"Incomplete generation: {stats['stop_reason']}")
    rates = {name: stats[f"{name}_token_per_sec"] for name in ("prefill", "decode")}
    prefill = re.findall(
        r"PREFILL_TOTAL tokens=(\d+)[^\n]* tokens_per_second=([\d.]+)", text
    )
    if len(prefill) != 1 or int(prefill[0][0]) != tokens:
        raise ValueError(f"Expected one pure LLM prefill measurement in {path}")
    if modality == "text":
        rates["prefill"] = float(prefill[0][1])
    if any(not 0 < rate < float("inf") for rate in rates.values()):
        raise ValueError(f"Invalid throughput in {path}")
    return rates


def summarize(rows, repeats):
    summary = []
    for modality, cases in REFERENCE.items():
        for tokens, reference in cases.items():
            for index, slots in enumerate((32, 64)):
                measured = [
                    row["rates"]
                    for row in rows
                    if (row["modality"], row["tokens"], row["slots"])
                    == (modality, tokens, slots)
                ]
                rates = (
                    {
                        name: statistics.median(row[name] for row in measured)
                        for name in ("prefill", "decode")
                    }
                    if measured
                    else None
                )
                baseline = dict(prefill=reference[0][index], decode=reference[1][index])
                complete = len(measured) == repeats
                summary.append(
                    dict(
                        modality=modality,
                        tokens=tokens,
                        slots=slots,
                        measured_rounds=len(measured),
                        complete=complete,
                        median_token_per_sec=rates,
                        sm8975_token_per_sec=baseline,
                        speedup=(
                            {name: rates[name] / baseline[name] for name in rates}
                            if rates
                            else None
                        ),
                        exceeds_sm8975=complete
                        and all(rates[name] > baseline[name] for name in baseline),
                    )
                )
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/sm8850.json")
    parser.add_argument("--weights-path", type=Path, required=True)
    parser.add_argument(
        "--model-config", type=Path, default=ROOT / "configs/model_sm8850.json"
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--io-workers", type=int, choices=[4, 8, 16])
    parser.add_argument("--prefill-io-workers", type=int, choices=[4, 8, 16])
    parser.add_argument("--read-chunk-kib", type=int, choices=[64, 128, 256, 512])
    parser.add_argument("--text-prompt", default="请介绍注意力机制的原理和应用。")
    parser.add_argument("--image-prompt", default="请详细描述这张图片。")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--run", action="store_true", help="Execute; otherwise save commands only"
    )
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("repeats must be positive")
    recipe = json.loads(args.model_config.read_text())
    if (
        recipe["target"]["soc_model"] != "SM8850"
        or recipe["context"]["max_seq_len"] != 2048
        or recipe["context"]["prefill_ar_len"] != 128
        or recipe["quantization"]["expert_bits"] != 2
    ):
        parser.error("Use the SM8850 2048-context AR128 W2 recipe")
    if not args.image.is_file():
        parser.error(f"Image does not exist: {args.image}")
    args.output.mkdir(parents=True, exist_ok=True)
    commands, measurements = [], []
    report_path = args.output / "RESULTS.json"
    if report_path.exists():
        parser.error("Use a new output directory to preserve the previous measurements")
    report = dict(
        protocol="SM8850, 2048 context, AR128, W2, Top8, input=output tokens, EOS ignored",
        reference="User-provided SM8975 screenshot; prompts, clocks and temperature unknown",
        prefill_scope="Text: LLM loop; image: LLM plus image preprocessing and encoder",
        image_sha256=hashlib.sha256(args.image.read_bytes()).hexdigest(),
        model_config=recipe,
        memory="Not measured; throughput comparison does not establish memory equivalence",
        measurements=measurements,
        commands=commands,
        all_throughput_targets_passed=False,
    )
    for round_index in range(args.repeats):
        for modality in REFERENCE:
            for tokens in REFERENCE[modality]:
                for slots in (32, 64) if round_index % 2 == 0 else (64, 32):
                    label = f"{modality}-{tokens}-slots{slots}-r{round_index}"
                    output = args.output / label
                    command = [
                        sys.executable,
                        str(ROOT / "run_chat.py"),
                        "--config",
                        str(args.config.resolve()),
                        "--weights-path",
                        str(args.weights_path.resolve()),
                        "--model-config",
                        str(args.model_config.resolve()),
                        "--cache-experts",
                        str(slots),
                        "--prompt-tokens",
                        str(tokens),
                        "--max-new-tokens",
                        str(tokens),
                        "--temperature",
                        "0",
                        "--ignore-eos",
                        "--no-enable-thinking",
                        "--no-interactive",
                        "--no-stream",
                        "--output",
                        str(output.resolve()),
                        "--prompt",
                        args.image_prompt if modality == "image" else args.text_prompt,
                    ]
                    for flag, value in (
                        ("--build-dir", args.build_dir),
                        ("--io-workers", args.io_workers),
                        ("--prefill-io-workers", args.prefill_io_workers),
                        ("--read-chunk-kib", args.read_chunk_kib),
                    ):
                        if value is not None:
                            command += [flag, str(value)]
                    command += (
                        ["--image-path", str(args.image.resolve())]
                        if modality == "image"
                        else ["--no-with-vision"]
                    )
                    commands.append(dict(label=label, command=command))
                    report_path.write_text(
                        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
                    )
                    if not args.run:
                        continue
                    with (args.output / f"{label}.log").open("w") as log:
                        subprocess.run(
                            command, stdout=log, stderr=subprocess.STDOUT, check=True
                        )
                    logs = list(output.glob("*/runner.log"))
                    if len(logs) != 1:
                        raise ValueError(f"Expected one native runner log in {output}")
                    run = json.loads((logs[0].parent / "RUN.json").read_text())
                    configuration = run["model_configuration"]
                    if (
                        run["exit_code"] != 0
                        or configuration["target"]["soc_model"] != "SM8850"
                        or configuration["schedule"]["cache_experts"] != slots
                        or configuration["quantization"]["expert_bits"] != 2
                        or configuration["context"]["prefill_ar_len"] != 128
                        or configuration["context"]["max_seq_len"] != 2048
                        or (
                            args.io_workers is not None
                            and configuration["schedule"]["io_workers"]
                            != args.io_workers
                        )
                        or (
                            args.read_chunk_kib is not None
                            and configuration["schedule"]["read_chunk_kib"]
                            != args.read_chunk_kib
                        )
                        or (
                            args.prefill_io_workers is not None
                            and run["prefill_io_workers"] != args.prefill_io_workers
                        )
                    ):
                        raise ValueError(
                            f"Runtime configuration differs from protocol: {run}"
                        )
                    measurements.append(
                        dict(
                            modality=modality,
                            tokens=tokens,
                            slots=slots,
                            round=round_index,
                            log=str(logs[0]),
                            rates=read_measurement(logs[0], tokens, modality),
                        )
                    )
                    report["summary"] = summarize(measurements, args.repeats)
                    report["all_throughput_targets_passed"] = all(
                        row["exceeds_sm8975"] for row in report["summary"]
                    )
                    report_path.write_text(
                        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
                    )
    print(report_path)


if __name__ == "__main__":
    main()
