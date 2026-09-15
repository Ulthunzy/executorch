# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Reusable two-turn self-identity benchmark for ChatML QNN models."""

import argparse
import csv
import json
import math
import shlex
import subprocess
import sys
import time
import uuid
from pathlib import Path, PurePosixPath

from report import (
    digest,
    file_hash,
    read_json,
    summarize,
    validate_cases,
    write_json,
)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common.output import run_lock  # noqa: E402


def resolve_model(path):
    """Resolve a PTE file without guessing among multiple exported models."""
    path = path.resolve()
    if path.is_dir():
        candidates = sorted(path.glob("*.pte"))
        if len(candidates) != 1:
            raise ValueError(
                "Model directory must have exactly one PTE; otherwise specify the file"
            )
        path = candidates[0]
    if not path.is_file():
        raise ValueError(f"Model not found: {path}")
    return path


def device_path(value):
    """Restrict device locations to explicit Android test directories."""
    path = PurePosixPath(value)
    if (
        not path.is_absolute()
        or ".." in path.parts
        or not str(path).startswith("/data/local/tmp/")
    ):
        raise ValueError("Device paths must be inside /data/local/tmp/, without '..'")
    return str(path)


def adb(serial, *args, timeout=60):
    """Run ADB against the selected device with a bounded timeout."""
    return subprocess.run(
        ["adb", "-s", serial, *map(str, args)],
        check=True,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=timeout,
    ).stdout


def remote_hash(serial, path):
    """Read the SHA256 of a file on the device."""
    return adb(serial, "shell", "sha256sum " + shlex.quote(path), timeout=180).split()[
        0
    ]


def push_verified(serial, source, destination):
    """Push one artifact and verify its device-side contents."""
    adb(serial, "push", source, destination, timeout=600)
    if remote_hash(serial, destination) != file_hash(source):
        raise ValueError(f"Device file hash mismatch: {destination}")


def completed_session(root, session):
    """Skip only sessions with a complete, unchanged set of result records."""
    marker = root / "complete.json"
    if not marker.is_file():
        return False
    saved = read_json(marker)
    expected = {turn["id"] + ".json" for turn in session["turns"]}
    if set(saved["records"]) != expected:
        raise ValueError(f"Invalid completion marker: {root}")
    if any(
        not (root / name).is_file() or file_hash(root / name) != checksum
        for name, checksum in saved["records"].items()
    ):
        raise ValueError(f"Completed records changed: {root}")
    return True


def session_input(session, thinking_disabled=False, canonical_inputs=None):
    """Prepare incremental ChatML without injecting identity answers."""
    turns = [
        {
            key: turn[key]
            for key in ("id", "turn", "prompt", "formatted_prompt")
            if key in turn
        }
        for turn in session["turns"]
    ]
    if thinking_disabled:
        for index, turn in enumerate(turns):
            if "formatted_prompt" in turn:
                raise ValueError(
                    "--disable-thinking cannot override a preformatted prompt"
                )
            turn["formatted_prompt"] = (
                ("\n" if index else "")
                + "<|im_start|>user\n"
                + turn["prompt"]
                + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            )
    if canonical_inputs is not None:
        for index, turn in enumerate(turns):
            formatted = turn.get(
                "formatted_prompt",
                (
                    ("\n" if index else "")
                    + "<|im_start|>user\n"
                    + turn["prompt"]
                    + "<|im_end|>\n<|im_start|>assistant\n"
                ),
            )
            key = f"think{int(not thinking_disabled)}-{turn['id']}"
            expected = canonical_inputs[key]
            if expected["prompt"] != formatted:
                raise ValueError(f"Canonical prompt mismatch: {key}")
            ids = expected["input_ids"]
            if not ids or any(type(value) is not int or value < 0 for value in ids):
                raise ValueError(f"Invalid canonical token IDs: {key}")
            turn["formatted_prompt"] = formatted
            turn["input_ids"] = ids
    return {"id": session["id"], "group": session["group"], "turns": turns}


def run_session(
    args, root, remote, session, *, thinking_disabled=False, canonical_inputs=None
):
    """Run a whole KV-continuous session, preserving interrupted attempts."""
    sid = session["id"]
    output = root / "sessions" / sid
    if completed_session(output, session):
        print("SKIP", sid, flush=True)
        return True
    if output.exists():
        archive = root / "attempts" / (sid + "-" + uuid.uuid4().hex[:12])
        archive.parent.mkdir(exist_ok=True)
        output.rename(archive)
        print("Archived incomplete session:", archive, flush=True)
    output.mkdir(parents=True)
    write_json(
        output / "input.json",
        session_input(session, thinking_disabled, canonical_inputs),
    )
    destination = remote + "/sessions/" + sid + "-" + uuid.uuid4().hex[:12]
    adb(args.device, "shell", "mkdir -p " + shlex.quote(destination))
    adb(args.device, "push", output / "input.json", destination + "/input.json")
    command = (
        "cd "
        + shlex.quote(args.device_lib_dir)
        + " && export LD_LIBRARY_PATH="
        + shlex.quote(args.device_lib_dir)
        + " ADSP_LIBRARY_PATH="
        + shlex.quote(args.device_lib_dir)
        + " && "
        + shlex.join(
            [
                "timeout",
                "-k",
                "10",
                str(args.timeout),
                remote + "/driver",
                remote + "/config.json",
                destination + "/input.json",
                destination,
            ]
        )
        + " 2>&1"
    )
    (output / "command.txt").write_text(
        shlex.join(["adb", "-s", args.device, "shell", command]) + "\n",
        encoding="utf-8",
    )
    (output / "battery.txt").write_text(
        adb(args.device, "shell", "dumpsys battery"), encoding="utf-8"
    )
    print(time.strftime("%H:%M:%S"), "SESSION", sid, session["group"], flush=True)
    failure = None
    with (output / "session.log").open("w", encoding="utf-8") as logfile:
        try:
            process = subprocess.run(
                ["adb", "-s", args.device, "shell", command],
                stdout=logfile,
                stderr=subprocess.STDOUT,
                timeout=args.timeout + 30,
            )
            if process.returncode:
                failure = (
                    "timeout" if process.returncode in (124, 137) else "process_error"
                )
        except subprocess.TimeoutExpired:
            failure = "transport_timeout"
    # Pull files instead of parsing JSON mixed with native stdout/stderr.
    adb(args.device, "pull", destination + "/.", output, timeout=180)
    valid = True
    for turn in session["turns"]:
        path = output / (turn["id"] + ".json")
        try:
            record = read_json(path)
            for key, value in (
                ("id", turn["id"]),
                ("session_id", sid),
                ("turn", turn["turn"]),
                ("prompt", turn["prompt"]),
            ):
                if record.get(key) != value:
                    raise ValueError(f"Mismatched {key} in {path}")
        except (FileNotFoundError, json.JSONDecodeError):
            if path.exists():
                path.rename(path.with_suffix(".json.partial"))
            partial = output / (turn["id"] + ".txt")
            record = {
                **turn,
                "session_id": sid,
                "error_code": None,
                "stats": None,
                "response": (
                    partial.read_text(errors="replace") if partial.exists() else ""
                ),
                "infrastructure_error": failure or "missing_result",
            }
            write_json(path, record)
            valid = False
        print(
            "DONE",
            turn["id"],
            "error",
            record["error_code"],
            "tokens",
            (record.get("stats") or {}).get("generated_tokens"),
            flush=True,
        )
    if valid and not failure:
        write_json(
            output / "complete.json",
            {
                "records": {
                    turn["id"] + ".json": file_hash(output / (turn["id"] + ".json"))
                    for turn in session["turns"]
                },
            },
        )
    else:
        print(
            "Incomplete session retained; --resume retries both turns:", sid, flush=True
        )
    return valid and not failure


def run(args, **options):
    with run_lock(args.output):
        return run_locked(args, **options)


def run_locked(
    args,
    *,
    cases=None,
    report_fn=summarize,
    evaluation=None,
    max_new_tokens=-1,
    thinking_disabled=False,
    repeat_last_n=None,
    canonical_inputs_path=None,
    repair_think_eos=False,
    enable_think_budget=False,
    think_budget=512,
):
    """Validate and deploy artifacts, run sessions, and generate reports."""
    if type(think_budget) is not int or not 1 <= think_budget <= 2147483647:
        raise ValueError("think-budget must be a positive 32-bit integer")
    if repeat_last_n is not None and (
        type(repeat_last_n) is not int or not 1 <= repeat_last_n <= 2147483647
    ):
        raise ValueError("repeat-last-n must be a positive 32-bit integer")
    args.device_lib_dir = device_path(args.device_lib_dir)
    if args.device_model:
        args.device_model = device_path(args.device_model)
    model = resolve_model(args.model)
    tokenizer = (args.tokenizer or model.parent / "tokenizer.json").resolve()
    driver = args.driver.resolve()
    for path in (tokenizer, driver):
        if not path.is_file():
            raise ValueError(
                f"Missing {path}; see model_validation/benchmarks/README.md"
            )
    if args.seq_len < 2 or args.timeout < 1:
        raise ValueError("seq-len must be >=2; timeout must be positive")
    if not math.isfinite(args.repetition_penalty) or args.repetition_penalty < 1:
        raise ValueError("repetition-penalty must be finite and >=1")
    if evaluation is None and (
        not args.expected_english.strip() or not args.expected_chinese.strip()
    ):
        raise ValueError("Expected names cannot be empty")
    if max_new_tokens == 0 or max_new_tokens < -1:
        raise ValueError("max-new-tokens must be -1 or positive")
    if cases is None:
        cases = read_json(args.cases)
    validate_cases(cases)
    canonical_inputs = None
    if canonical_inputs_path is not None:
        records = read_json(canonical_inputs_path)
        canonical_inputs = {record["id"]: record for record in records}
        if len(canonical_inputs) != len(records):
            raise ValueError("Duplicate canonical prompt IDs")
        for session in cases["sessions"]:
            session_input(session, thinking_disabled, canonical_inputs)
    if thinking_disabled:
        if evaluation is not None:
            raise ValueError(
                "--disable-thinking is only supported by the identity suite"
            )
        for session in cases["sessions"]:
            session_input(session, True)
    if args.ids:
        unknown = set(args.ids) - {session["id"] for session in cases["sessions"]}
        if unknown:
            raise ValueError(f"Unknown session IDs: {sorted(unknown)}")
        cases["sessions"] = [s for s in cases["sessions"] if s["id"] in args.ids]
    if adb(args.device, "get-state").strip() != "device":
        raise ValueError("ADB device not ready")
    libdir = shlex.quote(args.device_lib_dir)
    libraries = adb(
        args.device,
        "shell",
        f'for f in {libdir}/*.so; do if test -f "$f"; then sha256sum "$f"; fi; done',
        timeout=180,
    )
    if (
        "libQnnHtp.so" not in libraries
        or "libqnn_executorch_backend.so" not in libraries
    ):
        raise ValueError(
            "--device-lib-dir must contain the matching QNN and ExecuTorch libraries"
        )
    config = {
        "decoder": args.decoder,
        "mode": args.mode,
        "seq_len": args.seq_len,
        "temperature": 0,
        "repetition_penalty": args.repetition_penalty,
        "ignore_eos": False,
        "max_new_tokens": max_new_tokens,
        "prompt_format": "chatml-preformatted-v1" if evaluation else "chatml-v1",
        "thinking_disabled": thinking_disabled,
    }
    if repeat_last_n is not None:
        config["repeat_last_n"] = repeat_last_n
    if repair_think_eos or enable_think_budget:
        config.update(
            repair_think_eos=repair_think_eos,
            enable_think_budget=enable_think_budget,
            think_budget=think_budget,
        )
    print("Hashing model/tokenizer/driver...", flush=True)
    identity = {
        "cases_hash": digest(cases),
        "model_hash": file_hash(model),
        "tokenizer_hash": file_hash(tokenizer),
        "driver_hash": file_hash(driver),
        "runtime_libraries": libraries.strip(),
        "device": args.device,
        "device_build": adb(
            args.device, "shell", "getprop ro.build.fingerprint"
        ).strip(),
        "config": config,
        "expected": (
            {}
            if evaluation
            else {
                "english": args.expected_english,
                "chinese": args.expected_chinese,
            }
        ),
    }
    if evaluation is not None:
        identity["evaluation"] = evaluation
    if canonical_inputs_path is not None:
        identity["canonical_inputs_hash"] = file_hash(canonical_inputs_path)
    root = args.output.resolve()
    if root.exists():
        if not args.resume or not (root / "manifest.json").exists():
            raise ValueError(
                "Output exists. Use a new directory, or --resume for the same run"
            )
        manifest = read_json(root / "manifest.json")
        if manifest["identity"] != identity:
            raise ValueError(
                "Resume refused: model, cases, device, runtime or configuration changed"
            )
        if read_json(root / "cases.json") != cases:
            raise ValueError("Resume refused: saved question set changed")
        if manifest["device_lib_dir"] != args.device_lib_dir:
            raise ValueError("Resume refused: runtime path changed")
        if (
            evaluation is None
            and (root / "summary.json").is_file()
            and all(
                completed_session(root / "sessions" / s["id"], s)
                for s in cases["sessions"]
            )
        ):
            from common.audit import audit_run

            audit_run(root)
            print("SKIP completed run", root)
            return
    else:
        root.mkdir(parents=True)
        manifest = {
            "schema_version": 1,
            "identity": identity,
            "model_label": args.label or model.parent.name,
            "model_path": str(model),
            "tokenizer_path": str(tokenizer),
            "device_lib_dir": args.device_lib_dir,
            "remote": "/data/local/tmp/executorch-identity-eval/" + uuid.uuid4().hex,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        write_json(root / "manifest.json", manifest)
        write_json(root / "cases.json", cases)
    remote = device_path(manifest["remote"])
    adb(args.device, "shell", "mkdir -p " + shlex.quote(remote))
    if args.device_model:
        if remote_hash(args.device, args.device_model) != identity["model_hash"]:
            raise ValueError("--device-model does not match the local PTE")
        remote_model = args.device_model
    else:
        remote_model = remote + "/model.pte"
        exists = adb(
            args.device,
            "shell",
            "if test -f " + shlex.quote(remote_model) + "; then echo yes; fi",
        ).strip()
        if (
            exists != "yes"
            or remote_hash(args.device, remote_model) != identity["model_hash"]
        ):
            push_verified(args.device, model, remote_model)
    push_verified(args.device, tokenizer, remote + "/tokenizer.json")
    push_verified(args.device, driver, remote + "/driver")
    adb(args.device, "shell", "chmod 755 " + shlex.quote(remote + "/driver"))
    if (
        evaluation is not None
        or thinking_disabled
        or repeat_last_n is not None
        or canonical_inputs is not None
        or repair_think_eos
        or enable_think_budget
    ):
        capabilities = json.loads(
            adb(
                args.device,
                "shell",
                "LD_LIBRARY_PATH="
                + shlex.quote(args.device_lib_dir)
                + " "
                + shlex.quote(remote + "/driver")
                + " --capabilities",
            )
        )
        if capabilities.get("protocol_version", 0) < 2:
            raise ValueError(
                "Rebuild the test driver: formatted prompts require protocol v2"
            )
        if thinking_disabled and not capabilities.get("formatted_prompt", False):
            raise ValueError("The test driver does not support preformatted prompts")
        if repeat_last_n is not None and not capabilities.get("repeat_last_n", False):
            raise ValueError("The test driver does not support repeat-last-n")
        if canonical_inputs is not None and not capabilities.get(
            "tokenized_prompt", False
        ):
            raise ValueError("The test driver does not support canonical token IDs")
        if (repair_think_eos or enable_think_budget) and not capabilities.get(
            "thinking_controls", False
        ):
            raise ValueError("Rebuild the test driver to enable thinking controls")
    write_json(
        root / "device_config.json",
        {
            **config,
            "model": remote_model,
            "tokenizer": remote + "/tokenizer.json",
        },
    )
    adb(args.device, "push", root / "device_config.json", remote + "/config.json")
    failed = []
    try:
        for session in cases["sessions"]:
            if session.get("skip_reason"):
                print("SKIP", session["id"], session["skip_reason"], flush=True)
                continue
            if not run_session(
                args,
                root,
                remote,
                session,
                thinking_disabled=thinking_disabled,
                canonical_inputs=canonical_inputs,
            ):
                failed.append(session["id"])
    finally:
        summary = report_fn(root)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        print("Report:", root / "REPORT.md", flush=True)
    if failed:
        raise ValueError(
            f"Incomplete sessions: {', '.join(failed)}; use --resume to retry"
        )


def compare(runs, output):
    """Combine summaries while flagging differences beyond the model weights."""
    if output.exists():
        raise ValueError("Comparison output already exists; choose a new filename")
    fields = [
        "run",
        "model",
        "planned",
        "attempted",
        "reviewed",
        "normal",
        "identity_correct",
        "normal_rate",
        "identity_correct_rate",
        "eos",
        "repetition_flags",
        "both_turns_normal",
        "decode_token_per_sec_median",
        "comparison_caveats",
    ]
    first = read_json(runs[0] / "manifest.json")["identity"]
    rows = []
    for root in runs:
        identity = read_json(root / "manifest.json")["identity"]
        summary = read_json(root / "summary.json")
        if summary["run_fingerprint"] != digest(identity):
            raise ValueError(f"Stale summary: {root}")
        if digest(read_json(root / "cases.json")) != identity["cases_hash"]:
            raise ValueError(f"Question set changed: {root}")
        for relative, checksum in summary["source_hashes"].items():
            path = root / relative
            if (file_hash(path) if path.exists() else None) != checksum:
                raise ValueError(f"Results changed; run summarize again: {path}")
        changed = [
            key
            for key in first
            if key != "model_hash" and first[key] != identity.get(key)
        ]
        rows.append(
            {
                **summary,
                "run": str(root),
                "comparison_caveats": ",".join(changed),
            }
        )
    with output.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(output)


def add_device_arguments(run_parser):
    """Share deployment and generation flags across benchmark front ends."""
    run_parser.add_argument("--model", type=Path, required=True)
    run_parser.add_argument("--tokenizer", type=Path)
    run_parser.add_argument("--device", required=True)
    run_parser.add_argument("--device-lib-dir", required=True)
    run_parser.add_argument(
        "--device-model", help="Reuse a device PTE after SHA256 verification"
    )
    run_parser.add_argument(
        "--driver",
        type=Path,
        default=Path("build-android/self_identity_eval/identity_eval_driver"),
    )
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--label")
    run_parser.add_argument("--mode", choices=("kv", "hybrid"), default="hybrid")
    run_parser.add_argument(
        "--decoder", choices=("lincal", "qwen3", "qwen2_5"), default="lincal"
    )
    run_parser.add_argument("--seq-len", type=int, default=4096)
    run_parser.add_argument("--repetition-penalty", type=float, default=1.0)
    run_parser.add_argument(
        "--ids", nargs="+", help="Session IDs for a smoke test, e.g. I01"
    )
    run_parser.add_argument(
        "--timeout", type=int, default=600, help="Seconds per session"
    )
    run_parser.add_argument("--resume", action="store_true")


def main():
    """Dispatch device execution, offline reporting, or cross-run comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser(
        "run", help="Run serial device sessions, then summarize"
    )
    add_device_arguments(run_parser)
    run_parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Use the non-thinking ChatML prefix for every turn; requires a protocol-v2 driver.",
    )
    run_parser.add_argument(
        "--repeat-last-n",
        type=int,
        help="Override the repetition window; requires a driver with repeat_last_n support. Omitted: keep the driver's default.",
    )
    run_parser.add_argument("--expected-english", default="lu xiaoxi")
    run_parser.add_argument(
        "--repair-think-eos", action=argparse.BooleanOptionalAction, default=False
    )
    run_parser.add_argument(
        "--enable-think-budget", action=argparse.BooleanOptionalAction, default=False
    )
    run_parser.add_argument("--think-budget", type=int, default=512)
    run_parser.add_argument("--expected-chinese", default="陆小兮")
    run_parser.add_argument("--cases", type=Path, default=HERE / "cases.json")
    run_parser.add_argument(
        "--canonical-inputs",
        type=Path,
        help="Frozen id/prompt/input_ids records; bypass native input tokenization.",
    )
    report_parser = commands.add_parser(
        "summarize", help="Rebuild reports without running inference"
    )
    report_parser.add_argument("output", type=Path)
    report_parser.add_argument("--reviews", type=Path)
    compare_parser = commands.add_parser(
        "compare", help="Combine run summaries into a CSV"
    )
    compare_parser.add_argument("runs", type=Path, nargs="+")
    compare_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "run":
            run(
                args,
                thinking_disabled=args.disable_thinking,
                repeat_last_n=args.repeat_last_n,
                canonical_inputs_path=args.canonical_inputs,
                repair_think_eos=args.repair_think_eos,
                enable_think_budget=args.enable_think_budget,
                think_budget=args.think_budget,
            )
        elif args.command == "summarize":
            print(
                json.dumps(
                    summarize(args.output, args.reviews), ensure_ascii=False, indent=2
                )
            )
        else:
            compare(args.runs, args.output)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
