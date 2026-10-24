import json
import hashlib

from model_config import shared_model_signature


def check_plan(path, args, phase):
    if not path.is_file():
        raise FileNotFoundError(path)
    metadata = json.loads(path.with_suffix(".json").read_text())
    if metadata["phase"] != phase or metadata["layers"] != 40:
        raise ValueError(f"{path}: wrong phase or layer count")
    if metadata.get("compile_settings") != args.model_settings.compile_settings():
        raise ValueError(
            f"{path}: model/quantization/KV/context configuration differs; repack with the selected --model-config"
        )
    shared = metadata["weight_mode"] == "external_shared"
    if shared != bool(args.shared_model):
        raise ValueError(f"{path}: weight mode differs from the selected artifacts")
    if shared and metadata.get("shared_signature") != shared_model_signature(
        args.shared_model
    ):
        raise ValueError(
            f"{path}: shared context/IO metadata differs; repack for the selected model"
        )
    if (
        phase == "decode"
        and metadata.get("cache_encoding_sha256")
        != hashlib.sha256(
            (args.artifacts / "assets/cache-encoding.bin").read_bytes()
        ).hexdigest()
    ):
        raise ValueError(
            f"{path}: prefill cache encodings differ; repack the decode plan"
        )
    if phase == "prefill":
        if metadata.get("prefill_order") != args.prefill_order:
            raise ValueError(f"{path}: prefill order differs; repack the plan")
        expected = args.prompt_tokens if args.prefill_order == "layer_major" else 128
        if expected is not None and metadata["plan_tokens"] != expected:
            raise ValueError(
                f"{path}: plan length does not match the prompt and prefill order"
            )
    return metadata


def prepare_plans(args, run_dir):
    # Shared plans contain stage metadata; creating one does not compile QNN
    # or duplicate weights and lets the requested prefill order select the loop.
    if args.shared_model:
        from pack_pte import main as pack_plan

        for phase, current in (
            ("prefill", args.local_prefill_pte),
            ("decode", args.local_decode_pte),
        ):
            if current is not None:
                continue
            path = run_dir / "plans" / f"{phase}.pte"
            command = [
                "--quiet",
                "--model-config",
                str(args.model_config),
                "--phase",
                phase,
                "--source-dir",
                str(args.artifacts),
                "--shared-model",
                str(args.shared_model),
                "--head",
                str(args.shared_model / "head/graph.qnn"),
                "--output",
                str(path),
                "--max-seq-len",
                str(args.decode_capacity),
                "--prefill-order",
                args.prefill_order,
                "--prefill-tokens",
                str(args.prompt_tokens),
            ]
            if phase == "decode":
                command += [
                    "--cache-encoding",
                    str(args.artifacts / "assets/cache-encoding.bin"),
                ]
            pack_plan(command)
            if phase == "prefill":
                args.local_prefill_pte = path
            else:
                args.local_decode_pte = path
    plans = {}
    for phase, path in (
        ("prefill", args.local_prefill_pte),
        ("decode", args.local_decode_pte),
    ):
        if path is not None:
            plans[phase] = check_plan(path, args, phase)
    if args.prompt_tokens > 128 and "prefill" not in plans:
        raise ValueError(
            "Multi-block prefill needs a local plan/JSON sidecar or shared model so the configured schedule can be verified"
        )
    return plans
