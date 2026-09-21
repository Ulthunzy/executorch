"""Content-bound provenance for one SafeTensors -> PT2 -> PTE export."""

import hashlib
import os
import subprocess
from pathlib import Path

from ..ppl.artifacts import digest, file_hash, tree_identity, write_json


RUNTIME_ONLY_ARGS = {
    "artifact",
    "build_folder",
    "compile_only",
    "device",
    "enable_x86_64",
    "eval_limit",
    "eval_methods",
    "eval_num_fewshot",
    "eval_tasks",
    "host",
    "ignore_eos",
    "ip",
    "port",
    "pre_gen_pte",
    "prompt",
    "repair_think_eos",
    "repetition_penalty",
    "skip_push",
    "system_prompt",
    "temperature",
}


def _json_value(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return str(value)


def _local_identity(value, *, required=False):
    if not value:
        if required:
            raise ValueError("A required lineage source was not provided")
        return None
    path = Path(value).expanduser().resolve()
    if not path.exists():
        if required:
            raise ValueError(f"Lineage source does not exist: {path}")
        return {"kind": "external", "value": str(value)}
    return {"kind": "local", **tree_identity(path)}


def _calibration_sources(args):
    result = []
    for option in (
        "calib_samples",
        "calib_text_datasets",
        "calib_text_preserve_sources",
    ):
        for value in getattr(args, option, None) or []:
            result.append({"option": option, **_local_identity(value)})
    for option in ("calib_tasks", "calib_hf_dataset"):
        value = getattr(args, option, None)
        if not value:
            continue
        values = value if isinstance(value, list) else [value]
        result.extend(
            {"option": option, "kind": "external", "value": item} for item in values
        )
    return result


def _training_sources(args):
    result = []
    for option in ("train_tasks", "train_hf_dataset"):
        value = getattr(args, option, None)
        if not value:
            continue
        values = value if isinstance(value, list) else [value]
        result.extend(
            {"option": option, "kind": "external", "value": item} for item in values
        )
    return result


def _git_identity(repo):
    repo = Path(repo).resolve()

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=repo, check=False, capture_output=True
        )

    commit_result = git("rev-parse", "HEAD")
    commit = commit_result.stdout.decode(errors="replace").strip()
    if commit_result.returncode or len(commit) != 40:
        return None
    diff = git("diff", "--binary", "HEAD", "--")
    untracked = git("ls-files", "--others", "--exclude-standard", "-z")
    status = git("status", "--porcelain=v1", "--untracked-files=all", "-z")
    if any(result.returncode for result in (diff, untracked, status)):
        return None
    source_suffixes = {
        ".bzl",
        ".c",
        ".cc",
        ".cpp",
        ".cxx",
        ".h",
        ".hpp",
        ".py",
        ".sh",
    }
    untracked_sources = {}
    for raw in untracked.stdout.split(b"\0"):
        if not raw:
            continue
        relative = raw.decode(errors="surrogateescape")
        path = repo / relative
        if path.is_file() and (
            path.suffix in source_suffixes or path.name == "CMakeLists.txt"
        ):
            untracked_sources[relative] = file_hash(path)
    body = {
        "commit": commit,
        "dirty": bool(status.stdout),
        "tracked_diff_sha256": hashlib.sha256(diff.stdout).hexdigest(),
        "untracked_sources": untracked_sources,
    }
    return {**body, "sha256": digest(body)}


def write_export_lineage(
    args,
    *,
    pte_path,
    pt2_path,
    runtime_tokenizer_path,
    qnn_sdk_build_id,
    repo,
):
    """Write an immutable lineage manifest after both deployment artifacts exist."""
    pte = _local_identity(pte_path, required=True)
    pt2 = _local_identity(pt2_path)
    safetensors = _local_identity(getattr(args, "source_safetensors", None))
    checkpoint = _local_identity(getattr(args, "checkpoint", None))
    runtime_tokenizer = _local_identity(runtime_tokenizer_path)
    calibration = _calibration_sources(args)
    code = _git_identity(repo)
    qat = bool(getattr(args, "qat", False))
    training = _training_sources(args)
    train_config = _local_identity(getattr(args, "train_config", None)) if qat else None
    lr_config = _local_identity(getattr(args, "lr_config", None)) if qat else None
    configuration = {
        key: _json_value(value)
        for key, value in sorted(vars(args).items())
        if key not in RUNTIME_ONLY_ARGS
    }
    body = {
        "schema": "executorch.quantization.lineage.v1",
        "complete": all(
            (
                safetensors is not None and safetensors.get("kind") == "local",
                checkpoint is not None and checkpoint.get("kind") == "local",
                runtime_tokenizer is not None
                and runtime_tokenizer.get("kind") == "local",
                pt2 is not None and pt2.get("kind") == "local",
                calibration
                and all(item.get("kind") == "local" for item in calibration),
                not qat
                or (
                    training
                    and all(item.get("kind") == "local" for item in training)
                    and train_config is not None
                    and train_config.get("kind") == "local"
                    and lr_config is not None
                    and lr_config.get("kind") == "local"
                ),
                qnn_sdk_build_id,
                code,
            )
        ),
        "sources": {
            "safetensors": safetensors,
            "checkpoint": checkpoint,
            "params": _local_identity(getattr(args, "params", None)),
            "runtime_tokenizer": runtime_tokenizer,
            "calibration": calibration,
            "training": training,
            "train_config": train_config,
            "lr_config": lr_config,
        },
        "configuration": configuration,
        "configuration_sha256": digest(configuration),
        "outputs": {"pt2": pt2, "pte": pte},
        "environment": {
            "code": code,
            "code_commit": code.get("commit") if code else None,
            "qnn_sdk_build_id": qnn_sdk_build_id,
            "qnn_sdk_root": os.environ.get("QNN_SDK_ROOT"),
            "soc_model": getattr(args, "soc_model", None),
            "backend": getattr(args, "backend", None),
        },
    }
    body["lineage_sha256"] = digest(body)
    output = Path(args.artifact).resolve() / "export_lineage.json"
    write_json(output, body)
    return output
