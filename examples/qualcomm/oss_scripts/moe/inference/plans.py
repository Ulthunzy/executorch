import json
import hashlib

from executorch.examples.qualcomm.oss_scripts.moe.model.qwen35 import (
    shared_model_signature,
)
from executorch.examples.qualcomm.oss_scripts.moe.utils import runtime_assets


def check_plan(path, args, phase, expected_tokens=None):
    if not path.is_file():
        raise FileNotFoundError(f"{path}: missing exported PTE; run export.py first")
    metadata = json.loads(path.with_suffix(".json").read_text())
    args.model_settings.target.check_manifest(metadata)
    if metadata["phase"] != phase or metadata["layers"] != 40:
        raise ValueError(f"{path}: wrong phase or layer count")
    if not args.model_settings.matches_compile_settings(
        metadata.get("compile_settings")
    ):
        raise ValueError(
            f"{path}: model/quantization/KV/context configuration differs; run export.py with the selected --model-config"
        )
    shared = metadata["weight_mode"] == "external_shared"
    if shared != bool(args.shared_model):
        raise ValueError(f"{path}: weight mode differs from the selected artifacts")
    if shared and metadata.get("shared_signature") != shared_model_signature(
        args.shared_model
    ):
        raise ValueError(
            f"{path}: shared context/IO metadata differs; run export.py for the selected model"
        )
    if (
        phase == "decode"
        and metadata.get("cache_encoding_sha256")
        != hashlib.sha256(
            (runtime_assets(args) / "cache-encoding.bin").read_bytes()
        ).hexdigest()
    ):
        raise ValueError(
            f"{path}: prefill cache encodings differ; run export.py for matching decode PTEs"
        )
    if phase == "prefill":
        limit = metadata.get("plan_tokens")
        if (
            metadata.get("dynamic_sequence") is not True
            or metadata.get("ar_len") != 128
            or type(limit) is not int
            or not 128 <= limit <= 1024
            or limit % 128
        ):
            raise ValueError(
                f"{path}: invalid dynamic prefill workspace; re-export the single PTE"
            )
        if metadata.get("prefill_order") != args.prefill_order:
            raise ValueError(
                f"{path}: prefill order differs; run export.py for the selected schedule"
            )
        expected = (
            expected_tokens
            if expected_tokens is not None
            else (args.prompt_tokens if args.prefill_order == "layer_major" else 128)
        )
        if expected is not None and (
            expected < 128
            or expected > args.model_settings.context.max_seq_len
            or expected % 128
        ):
            raise ValueError(
                f"{path}: plan length does not match the prompt and prefill order"
            )
    return metadata


def select_plans(args):
    """Select existing export artifacts; inference never serializes programs."""
    phases = ("decode",) if args.runner == "chat" else ("prefill", "decode")
    plans = {}
    for phase in phases:
        current = args.local_decode_pte if phase == "decode" else args.local_prefill_pte
        if current is None:
            filename = f"{phase}.pte"
            current = args.artifacts / "plans" / filename
        plans[phase] = check_plan(current, args, phase)
        if phase == "decode":
            args.local_decode_pte = current
        else:
            args.local_prefill_pte = current
    return plans


def load_prefill_bundle(args):
    if args.runner != "chat":
        return None
    directory = args.artifacts / "plans"
    manifest = directory / "PLANS.json"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"{manifest}: missing exported plan bundle; run export.py first"
        )
    metadata = json.loads(manifest.read_text())
    if (
        metadata.get("version") not in {2, 3}
        or metadata.get("ar_len") != 128
        or metadata.get("capacity") != args.decode_capacity
        or metadata.get("prefill_order") != args.prefill_order
        or metadata.get("prefill_pte") != "prefill.pte"
    ):
        raise ValueError(
            f"{manifest}: context or schedule differs; export matching PTEs with export.py"
        )
    args.model_settings.target.check_manifest(metadata["model_configuration"])
    expected_limit = (
        metadata.get("workspace_tokens")
        if metadata["version"] == 3
        else metadata.get("max_prefill_tokens", 1024)
    )
    if metadata["version"] == 2 and args.prefill_order == "chunk_major":
        expected_limit = 128
    if (
        type(expected_limit) is not int
        or not 128 <= expected_limit <= 1024
        or expected_limit % 128
        or (args.prefill_order == "chunk_major" and expected_limit != 128)
    ):
        raise ValueError(f"{manifest}: invalid prefill workspace bound")
    plan = check_plan(
        directory / "prefill.pte", args, "prefill", expected_tokens=expected_limit
    )
    if plan["plan_tokens"] != expected_limit:
        raise ValueError(
            f"{manifest}: prefill workspace bound differs from the schedule"
        )
    return directory
