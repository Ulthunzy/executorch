from datetime import datetime
import hashlib
import json

from .device import AndroidDevice, DevicePaths
from .deployment import build_command, deploy_artifacts
from .prompts import prepare_prompt
from .plans import prepare_plans
from .results import collect_results, stream_tokens, verify_results


def inference(args):
    vision = None
    if args.image_path:
        from multimodal import vision_metadata

        vision = vision_metadata(args.model, args.vision_artifacts)
        if args.model_settings.quantization.vision.endswith("_gptq") and (
            not vision.get("gptq")
            or not all(d.get("method") == "fixed_grid_gptq" for d in vision["gptq"])
            or not any(d.get("module") == "visual.pos_embed" for d in vision["gptq"])
        ):
            raise ValueError(
                "The selected vision recipe requires GPTQ projections and position embeddings"
            )
        if not args.image_path.is_file():
            raise FileNotFoundError(args.image_path)
    if args.check:
        print(f"Local artifacts and optimized runners ready: {args.artifacts}")
        return
    import numpy as np
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    messages, ids = prepare_prompt(args, tokenizer, np, vision)
    args.prompt_tokens = len(ids)
    if not 1 <= args.max_new_tokens <= args.decode_capacity + 1 - len(ids):
        raise ValueError("Generation exceeds the configured context capacity")
    teacher = None
    if args.teacher_tokens:
        teacher = np.fromfile(args.teacher_tokens, dtype="<i8")
        if (
            len(teacher) < args.max_new_tokens - 1
            or (teacher < 0).any()
            or (teacher >= len(tokenizer)).any()
        ):
            raise ValueError(
                "Teacher token file is too short or contains invalid token IDs"
            )
    generation = json.loads((args.model / "generation_config.json").read_text())
    label = (
        f"w2a8-fused-c{args.decode_capacity}-p{args.prompt_tokens}"
        f"-cache{args.cache_experts}"
    )
    run_dir = args.output / (datetime.now().strftime("%Y%m%d-%H%M%S-%f") + "-" + label)
    run_dir.mkdir(parents=True)
    np.asarray(ids, dtype="<i8").tofile(run_dir / "tokens.i64")
    (run_dir / "PROMPT.txt").write_text(tokenizer.decode(ids), encoding="utf-8")
    record = dict(
        model_configuration=args.model_settings.to_dict(),
        input_mode=(
            "token_file"
            if args.prefill_token_file
            else ("image" if vision is not None else "text")
        ),
        template_enable_thinking=(
            None if args.prefill_token_file else args.enable_thinking
        ),
        enable_thinking=args.enable_thinking,
        system_prompt=args.system_prompt,
        prefill_order=args.prefill_order,
        shared_model=str(args.shared_model) if args.shared_model else None,
        model=str(args.model),
        input_ids=ids,
        messages=messages,
        prompt_tokens=len(ids),
        prefill_ar_len=128,
        decode_context_capacity=args.decode_capacity,
        max_new_tokens=args.max_new_tokens,
        expert_bits=2,
        activation_bits=8,
        expert_bank_variant="reconstructed",
        weight_codebook=[-7, -2, 2, 7],
        cache_experts=args.cache_experts,
        cache_policy=f"lru{args.cache_experts}",
        front_variant="fused",
        decode_execution="same_process_library",
        vision_residency="model_lifetime" if vision is not None else None,
        state_swap=args.state_swap,
        embedding_bits=4 if args.embedding_table else 16,
        decode_context_load_mode=args.decode_context_load_mode,
        kv_append=args.kv_append_only,
        teacher_tokens=(str(args.teacher_tokens) if args.teacher_tokens else None),
        prefill_token_file=(
            str(args.prefill_token_file) if args.prefill_token_file else None
        ),
        ignore_eos=args.ignore_eos,
        eos_token_ids=generation["eos_token_id"],
        serial=args.device,
        stream_runner_sha256=hashlib.sha256(
            (args.build_dir / "libmoe_decode.so").read_bytes()
        ).hexdigest(),
        tokens_sha256=hashlib.sha256((run_dir / "tokens.i64").read_bytes()).hexdigest(),
    )
    if vision is not None:
        from multimodal import prepare_image

        record["vision"] = prepare_image(args, ids, vision, run_dir, np)
    print("RESULTS", run_dir, flush=True)
    record["plans"] = prepare_plans(args, run_dir)
    device = AndroidDevice(args.device)
    paths = DevicePaths(
        args.remote_root + "/" + label, args.remote_root + "/shared-model"
    )
    deploy_artifacts(args, device, paths, run_dir, vision)
    command = build_command(args, paths, generation, vision)
    record.update(command=command, before=device.snapshot())
    (run_dir / "INPUT.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n"
    )
    record["exit_code"] = None
    try:
        exit_code, generated, timings = stream_tokens(
            device, command, run_dir, tokenizer, args.text_only
        )
        record["exit_code"] = exit_code
        record["after"] = device.snapshot()
        if exit_code:
            raise RuntimeError(
                f"Runner exited with {exit_code}; see {run_dir}/runner.log"
            )
        (run_dir / "TOKEN_TIMINGS.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in timings)
        )
        collect_results(args, device, paths, run_dir, vision)
        return verify_results(
            args, run_dir, tokenizer, generated, teacher, generation["eos_token_id"]
        )
    except (Exception, KeyboardInterrupt) as error:
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (run_dir / "RUN.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n"
        )
