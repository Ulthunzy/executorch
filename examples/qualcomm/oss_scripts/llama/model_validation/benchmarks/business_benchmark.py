"""Run and score one shared business case set on HF, PT2, or PTE."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from benchmark import resolve_model, run as run_pte  # noqa: E402
from business_scoring import summarize_business, validate_business_cases  # noqa: E402
from hf_benchmark import run as run_hf  # noqa: E402
from ppl.artifacts import tree_identity  # noqa: E402
from pt2_benchmark import run as run_pt2  # noqa: E402
from report import digest, file_hash, read_json  # noqa: E402


def run(args):
    cases = read_json(args.cases)
    validate_business_cases(cases)
    if args.ids:
        unknown = set(args.ids) - {session["id"] for session in cases["sessions"]}
        if unknown:
            raise ValueError(f"Unknown session IDs: {sorted(unknown)}")
    evaluation = {
        "kind": "business",
        "schema_version": 1,
        "cases_sha256": digest(cases),
        "scorer_hashes": {
            name: file_hash(HERE / name)
            for name in (
                "business_benchmark.py",
                "business_scoring.py",
                "report.py",
                "sample_scoring.py",
            )
        },
        "model_sha256": tree_identity(
            resolve_model(args.model) if args.engine == "pte" else args.model.resolve()
        )["sha256"],
    }
    if args.engine == "hf":
        run_hf(args, report_fn=summarize_business, evaluation=evaluation)
    elif args.engine == "pt2":
        if args.tokenizer is None:
            raise ValueError("PT2 business evaluation requires --tokenizer")
        run_pt2(args, report_fn=summarize_business, evaluation=evaluation)
    else:
        if not args.device_lib_dir or args.driver is None:
            raise ValueError(
                "PTE business evaluation requires --device-lib-dir and --driver"
            )
        run_pte(
            args,
            cases=cases,
            report_fn=summarize_business,
            evaluation=evaluation,
            max_new_tokens=args.max_new_tokens,
            thinking_disabled=args.disable_thinking,
            repeat_last_n=args.repeat_last_n,
            canonical_inputs_path=args.canonical_inputs,
            repair_think_eos=False,
            enable_think_budget=False,
            think_budget=512,
        )


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    execution = commands.add_parser("run")
    execution.add_argument("--engine", choices=("hf", "pt2", "pte"), required=True)
    execution.add_argument("--model", type=Path, required=True)
    execution.add_argument("--tokenizer", type=Path)
    execution.add_argument("--cases", type=Path, required=True)
    execution.add_argument("--output", type=Path, required=True)
    execution.add_argument("--device", required=True)
    execution.add_argument("--seq-len", type=int, default=4096)
    execution.add_argument("--max-new-tokens", type=int, default=-1)
    execution.add_argument("--ids", nargs="+")
    execution.add_argument("--repetition-penalty", type=float, default=1.0)
    execution.add_argument("--repeat-last-n", type=int, default=128)
    execution.add_argument("--disable-thinking", action="store_true")
    execution.add_argument("--resume", action="store_true")
    execution.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    execution.add_argument("--attention", choices=("sdpa", "eager"), default="sdpa")
    execution.add_argument("--num-threads", type=int, default=4)
    execution.add_argument(
        "--pt2-layout",
        choices=("lincal-full-sequence",),
        default="lincal-full-sequence",
    )
    execution.add_argument("--device-lib-dir")
    execution.add_argument("--device-model")
    execution.add_argument("--driver", type=Path)
    execution.add_argument("--label")
    execution.add_argument("--mode", choices=("kv", "hybrid"), default="hybrid")
    execution.add_argument(
        "--decoder", choices=("lincal", "qwen3", "qwen2_5"), default="lincal"
    )
    execution.add_argument("--timeout", type=int, default=600)
    execution.add_argument("--canonical-inputs", type=Path)
    report = commands.add_parser("summarize")
    report.add_argument("output", type=Path)
    report.add_argument("--reviews", type=Path)
    return result


def main():
    args = parser().parse_args()
    try:
        if args.command == "summarize":
            print(
                json.dumps(
                    summarize_business(args.output, args.reviews),
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            run(args)
    except (
        ValueError,
        OSError,
        KeyError,
        ImportError,
        subprocess.SubprocessError,
    ) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
