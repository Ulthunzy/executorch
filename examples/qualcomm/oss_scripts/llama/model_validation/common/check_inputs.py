"""Freeze incremental ChatML inputs; optionally compare the actual device tokenizer."""

import argparse
import json
import shlex
import subprocess
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))
from common.output import atomic_directory  # noqa: E402
from benchmark import push_verified  # noqa: E402
from hf_benchmark import chatml_prompt  # noqa: E402
from report import (
    digest,
    file_hash,
    read_json,
    validate_cases,
    write_json,
)  # noqa: E402


def canonical_inputs(tokenizer, cases):
    validate_cases(cases)
    rows = []
    for thinking in (True, False):
        for session in cases["sessions"]:
            for index, turn in enumerate(session["turns"]):
                if "formatted_prompt" in turn:
                    raise ValueError(
                        "This ChatML check does not override preformatted prompts"
                    )
                prompt = chatml_prompt(
                    tokenizer, turn["prompt"], bool(index), not thinking
                )
                rows.append(
                    {
                        "id": f"think{int(thinking)}-{turn['id']}",
                        "prompt": prompt,
                        "input_ids": tokenizer.encode(prompt, add_special_tokens=False),
                    }
                )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--cases", type=Path, default=ROOT / "benchmarks/cases.json")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--serial")
    parser.add_argument("--driver", type=Path)
    parser.add_argument(
        "--device-tokenizer",
        type=Path,
        help="Local tokenizer.json to test on the device",
    )
    parser.add_argument("--device-lib-dir")
    args = parser.parse_args()
    if bool(args.serial) != bool(args.driver) or bool(args.serial) != bool(
        args.device_lib_dir
    ):
        parser.error(
            "Device checking requires --serial, --driver and --device-lib-dir together"
        )
    if args.device_tokenizer and not args.serial:
        parser.error("--device-tokenizer requires a device check")
    if args.output.exists():
        parser.error(
            "Choose a new output directory; existing frozen inputs are never overwritten"
        )
    with atomic_directory(args.output) as working:
        args.output = working
        result = freeze(args)
    if result["native_input_ids_equal"] is False:
        parser.exit(
            1,
            "Native tokenization differs; inspect validation.json. Canonical IDs remain available for an explicitly controlled-input run.\n",
        )


def freeze(args):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True, trust_remote_code=False
    )
    cases = read_json(args.cases)
    canonical = canonical_inputs(tokenizer, cases)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "canonical.json", canonical)
    prompts = [{"id": row["id"], "prompt": row["prompt"]} for row in canonical]
    write_json(args.output / "prompts.json", prompts)
    result = {
        "schema_version": 1,
        "cases_hash": digest(cases),
        "canonical_hash": file_hash(args.output / "canonical.json"),
        "tokenizer_files": {
            p.name: file_hash(p)
            for p in args.tokenizer.iterdir()
            if p.suffix in {".json", ".jinja"}
        },
        "prompts": len(canonical),
        "device_checked": False,
        "native_input_ids_equal": None,
        "note": "Incremental prompts only; each engine keeps its own generated history.",
    }
    if args.serial:
        remote = "/data/local/tmp/executorch-validation-inputs/" + uuid.uuid4().hex
        adb = ["adb", "-s", args.serial]
        subprocess.run(adb + ["shell", "mkdir -p " + shlex.quote(remote)], check=True)
        native_tokenizer = args.device_tokenizer or args.tokenizer / "tokenizer.json"
        for path, name in (
            (args.driver, "driver"),
            (native_tokenizer, "tokenizer.json"),
            (args.output / "prompts.json", "prompts.json"),
        ):
            push_verified(args.serial, path, remote + "/" + name)
        command = (
            f"chmod 755 {shlex.quote(remote + '/driver')} && "
            f"export LD_LIBRARY_PATH={shlex.quote(args.device_lib_dir)} "
            f"ADSP_LIBRARY_PATH={shlex.quote(args.device_lib_dir)} && "
            + shlex.join(
                [
                    remote + "/driver",
                    "--tokenize",
                    remote + "/tokenizer.json",
                    remote + "/prompts.json",
                    remote + "/native.json",
                ]
            )
        )
        subprocess.run(adb + ["shell", command], check=True)
        subprocess.run(
            adb + ["pull", remote + "/native.json", str(args.output / "native.json")],
            check=True,
        )
        actual = read_json(args.output / "native.json")
        expected = [
            {"id": row["id"], "input_ids": row["input_ids"]} for row in canonical
        ]
        result.update(
            device_checked=True,
            native_input_ids_equal=actual == expected,
            driver_hash=file_hash(args.driver),
            native_tokenizer_hash=file_hash(native_tokenizer),
            remote=remote,
            serial=args.serial,
            mismatches=[
                row["id"]
                for i, row in enumerate(expected)
                if i >= len(actual) or actual[i] != row
            ],
            actual_count=len(actual),
        )
    write_json(args.output / "validation.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
