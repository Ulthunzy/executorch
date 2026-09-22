"""Config-driven sequential jobs with explicit dependencies and guarded resume."""

import fcntl
import hashlib
import itertools
import json
import os
import re
import signal
import shlex
import string
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from common.registry import REPO, ROOT, TOOLS


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def write_json(path, value):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def expand(value, variables):
    if not isinstance(value, (str, int, float, bool)):
        raise ValueError("Arguments and variable values must be scalar values")
    try:
        return string.Template(str(value)).substitute(variables)
    except (KeyError, ValueError) as error:
        raise ValueError(f"Invalid template {value!r}: {error}") from error


def resolve_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO / path).resolve()


@dataclass(frozen=True)
class Job:
    id: str
    stage: str
    tool: str
    argv: list
    inputs: list
    outputs: list
    resume_args: list
    dependencies: list
    resource: str


def load_plan(path):
    config = json.loads(path.read_text(encoding="utf-8"))
    allowed = {"schema_version", "variables", "output_root", "steps", "description"}
    if set(config) - allowed or config.get("schema_version") != 1:
        raise ValueError(
            "Expected schema_version=1 and only documented configuration keys"
        )
    variables = {
        "repo": str(REPO),
        "validation": str(ROOT),
        "config_dir": str(path.resolve().parent),
    }
    for key, value in config.get("variables", {}).items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) or key in {
            *variables,
            "output",
            "job",
        }:
            raise ValueError(f"Invalid/reserved variable: {key}")
        variables[key] = expand(value, variables)
    output = resolve_path(expand(config["output_root"], variables))
    if output == REPO or output in REPO.parents or output.is_relative_to(ROOT):
        raise ValueError(
            "Use a dedicated result directory, outside model_validation source files"
        )
    variables["output"] = str(output)
    steps = config.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("steps must be a nonempty ordered list")
    jobs, stages, job_ids = [], {}, set()
    step_keys = {
        "id",
        "tool",
        "args",
        "inputs",
        "outputs",
        "depends_on",
        "matrix",
        "append_when",
        "resume_args",
        "description",
        "resource",
    }
    for step in steps:
        if set(step) - step_keys:
            raise ValueError(f"Unknown step keys: {set(step) - step_keys}")
        stage, tool = step["id"], step["tool"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", stage) or stage in stages:
            raise ValueError(f"Invalid/duplicate stage: {stage}")
        if tool not in TOOLS:
            raise ValueError(f"Unknown tool: {tool}")
        dependencies = step.get("depends_on", [])
        if not isinstance(dependencies, list) or any(
            name not in stages for name in dependencies
        ):
            raise ValueError(f"{stage}: depends_on must name earlier stages")
        for key in ("args", "inputs", "outputs", "resume_args", "append_when"):
            if key in step and not isinstance(step[key], list):
                raise ValueError(f"{stage}: {key} must be a list")
        if not step.get("outputs"):
            raise ValueError(f"{stage}: declare output files used to verify completion")
        matrix = step.get("matrix", {})
        if not isinstance(matrix, dict):
            raise ValueError(f"{stage}: matrix must be an object")
        for key, values in matrix.items():
            if key in {*variables, "job"} or not re.fullmatch(
                r"[A-Za-z_][A-Za-z0-9_]*", key
            ):
                raise ValueError(f"Invalid/shadowed matrix variable: {key}")
            if (
                not isinstance(values, list)
                or not values
                or len({str(v) for v in values}) != len(values)
            ):
                raise ValueError(f"{stage}: matrix values must be nonempty and unique")
        for condition in step.get("append_when", []):
            if (
                set(condition) != {"parameter", "equals", "args"}
                or condition["parameter"] not in matrix
                or not isinstance(condition["args"], list)
            ):
                raise ValueError(f"{stage}: invalid append_when condition")
        stages[stage] = []
        for values in itertools.product(*matrix.values()):
            combination = dict(zip(matrix, values))
            suffix = "".join(f"-{key}-{value}" for key, value in combination.items())
            job_id = stage + suffix
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", job_id) or job_id in job_ids:
                raise ValueError(f"Invalid/duplicate expanded job ID: {job_id}")
            job_ids.add(job_id)
            scope = {**variables, **combination, "job": job_id}
            argv = [sys.executable, str(TOOLS[tool].script)]
            argv.extend(expand(arg, scope) for arg in step.get("args", []))
            for condition in step.get("append_when", []):
                if combination[condition["parameter"]] == condition["equals"]:
                    argv.extend(expand(arg, scope) for arg in condition["args"])
            inputs = [
                resolve_path(expand(arg, scope)) for arg in step.get("inputs", [])
            ]
            outputs = [resolve_path(expand(arg, scope)) for arg in step["outputs"]]
            if any(
                not item.is_relative_to(output) or item == output for item in outputs
            ):
                raise ValueError(
                    f"{stage}: output markers must be files under output_root"
                )
            jobs.append(
                Job(
                    job_id,
                    stage,
                    tool,
                    argv,
                    inputs,
                    outputs,
                    [expand(arg, scope) for arg in step.get("resume_args", [])],
                    [job for name in dependencies for job in stages[name]],
                    expand(step.get("resource", ""), scope),
                )
            )
            stages[stage].append(job_id)
    markers = [item for job in jobs for item in job.outputs]
    if len(set(markers)) != len(markers):
        raise ValueError(
            "Two jobs declare the same output; include ${job} in matrix output paths"
        )
    return config, output, jobs


def fingerprints(paths):
    result = {}
    for path in paths:
        if path.is_dir():
            files = sorted(
                p
                for p in path.rglob("*")
                if p.is_file()
                and p.suffix
                in {
                    ".safetensors",
                    ".json",
                    ".jinja",
                    ".pt2",
                    ".pte",
                    ".txt",
                    ".jsonl",
                    ".u64",
                    ".log",
                    ".csv",
                    ".md",
                    ".yaml",
                    ".yml",
                }
            )
            if not files:
                raise ValueError(f"No supported input files in {path}")
            result[str(path)] = {str(p.relative_to(path)): file_hash(p) for p in files}
        elif path.is_file():
            result[str(path)] = file_hash(path)
        else:
            raise ValueError(f"Missing declared input/output: {path}")
    return result


@contextmanager
def resource_lock(resource):
    if not resource:
        yield
        return
    directory = (
        Path(tempfile.gettempdir()) / f"executorch-validation-locks-{os.getuid()}"
    )
    directory.mkdir(mode=0o700, exist_ok=True)
    with (directory / (digest(resource) + ".lock")).open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(
                f"Resource busy in another validation workflow: {resource}"
            ) from error
        yield


def run_workflow(args):
    config, output, all_jobs = load_plan(args.config)
    stages = {job.stage for job in all_jobs}
    requested = stages if args.all else set(args.stage or [])
    if not requested or requested - stages:
        raise ValueError(
            f"Choose --stage from {sorted(stages)}, or explicitly use --all"
        )
    selected = {job.id for job in all_jobs if job.stage in requested}
    if args.with_dependencies:
        for job in reversed(all_jobs):
            if job.id in selected:
                selected.update(job.dependencies)
    jobs = [job for job in all_jobs if job.id in selected]
    for job in jobs:
        print(f"[{job.id}] {shlex.join(job.argv)}", flush=True)
    if args.dry_run:
        print(f"DRY RUN: {len(jobs)} jobs; output={output}; no files written")
        return
    if output.exists() and not args.resume:
        raise ValueError(
            f"Output already exists: {output}; use --resume or a new output_root"
        )
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".workflow.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"Another workflow is using {output}") from error
        state_path = output / "workflow.json"
        identity = {"config": config, "python": sys.executable, "repo": str(REPO)}
        state = {"schema_version": 1, "identity": identity, "jobs": {}}
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state["identity"] != identity:
                raise ValueError(
                    "Workflow configuration changed; use a new output_root"
                )
        elif args.resume and any(p.name != ".workflow.lock" for p in output.iterdir()):
            raise ValueError("Cannot resume a directory without workflow.json")
        write_json(state_path, state)
        logs = output / "workflow-logs"
        logs.mkdir(exist_ok=True)
        sources = {
            str(p.relative_to(ROOT)): file_hash(p)
            for p in sorted(ROOT.rglob("*"))
            if p.is_file() and p.suffix in {".py", ".cpp", ".h"}
        }
        by_id = {job.id: job for job in all_jobs}

        def signature_for(job):
            return digest(
                {
                    "argv": job.argv,
                    "inputs": fingerprints(job.inputs),
                    "sources": sources,
                    "script": file_hash(Path(job.argv[1])),
                }
            )

        def require_completed(job_id):
            prerequisite = by_id[job_id]
            done = state["jobs"].get(job_id, {})
            if done.get("status") != "complete":
                raise ValueError(
                    f"Selected stage requires {job_id}; use --with-dependencies"
                )
            if signature_for(prerequisite) != done["signature"]:
                raise ValueError(f"Dependency input/code changed: {job_id}")
            if fingerprints(prerequisite.outputs) != done["outputs"]:
                raise ValueError(f"Dependency outputs changed: {job_id}")
            audit_outputs(prerequisite)
            for parent in prerequisite.dependencies:
                require_completed(parent)

        def audit_outputs(job):
            from common.audit import audit_run
            from reporting.quality_gate import audit_report

            for path in job.outputs:
                if path.name == "summary.json":
                    audit_run(path.parent)
                elif path.name == "gate.json":
                    audit_report(path.parent)

        workflow_exit_code = 0
        for job in jobs:
            for dependency in job.dependencies:
                require_completed(dependency)
            signature = signature_for(job)
            previous = state["jobs"].get(job.id)
            if previous:
                if previous.get("device_state") and not args.device_idle_confirmed:
                    raise ValueError(
                        f"{job.id}: previous device state is unknown. Check the phone, then pass --device-idle-confirmed to resume."
                    )
                if previous["signature"] != signature:
                    raise ValueError(
                        f"{job.id}: input/code changed; use a new output_root"
                    )
                if previous["status"] == "complete":
                    if fingerprints(job.outputs) != previous["outputs"]:
                        raise ValueError(f"{job.id}: completed outputs changed")
                    audit_outputs(job)
                    workflow_exit_code = max(
                        workflow_exit_code, previous.get("exit_code", 0)
                    )
                    print("SKIP", job.id, flush=True)
                    continue
            command = job.argv + (job.resume_args if args.resume else [])
            record = {"signature": signature, "command": command, "status": "running"}
            state["jobs"][job.id] = record
            write_json(state_path, state)
            try:
                with resource_lock(job.resource), (logs / (job.id + ".log")).open(
                    "a", encoding="utf-8"
                ) as log:
                    with subprocess.Popen(
                        command,
                        cwd=REPO,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        errors="replace",
                        start_new_session=True,
                    ) as process:
                        try:
                            for line in process.stdout:
                                print(line, end="", flush=True)
                                log.write(line)
                                log.flush()
                            exit_code = process.wait()
                            if exit_code not in TOOLS[job.tool].accepted_exit_codes:
                                raise subprocess.CalledProcessError(exit_code, command)
                        except BaseException:
                            try:
                                os.killpg(process.pid, signal.SIGTERM)
                            except ProcessLookupError:
                                pass
                            try:
                                process.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                try:
                                    os.killpg(process.pid, signal.SIGKILL)
                                except ProcessLookupError:
                                    pass
                                process.wait()
                            try:
                                os.killpg(process.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                            raise
                if signature_for(job) != signature:
                    raise ValueError(f"{job.id}: input/script changed during execution")
                audit_outputs(job)
                record.update(
                    status="complete",
                    exit_code=exit_code,
                    outputs=fingerprints(job.outputs),
                )
                workflow_exit_code = max(workflow_exit_code, exit_code)
            except BaseException as error:
                record.update(status="failed", error=str(error))
                if job.resource.startswith("adb:"):
                    record["device_state"] = (
                        "unknown_after_failure_check_remote_driver_before_restart"
                    )
                    print(
                        "Device state is unknown after interruption/failure. Check the remote driver before starting another device job.",
                        file=sys.stderr,
                        flush=True,
                    )
                write_json(state_path, state)
                raise
            write_json(state_path, state)
        print(f"Workflow finished: {output}")
        return workflow_exit_code
