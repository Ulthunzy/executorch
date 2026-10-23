import argparse
from datetime import datetime
import hashlib
import itertools
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import struct
import subprocess
import time

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]


def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/sm8975.json"
    )
    selection, _ = config_parser.parse_known_args()
    defaults = json.loads(selection.config.read_text())
    parser = argparse.ArgumentParser(
        parents=[config_parser],
        description="Qwen3.5-35B-A3B streaming MoE chat on SM8975/V85",
    )
    parser.add_argument("--device", default=defaults["device"])
    parser.add_argument("--model", type=Path, default=Path(defaults["model"]))
    parser.add_argument("--artifacts", type=Path, default=REPO / defaults["artifacts"])
    parser.add_argument(
        "--build-dir", type=Path, default=REPO / "build/moe/qwen35-35B-A3B"
    )
    parser.add_argument(
        "-a",
        "--output",
        type=Path,
        default=REPO / "llama-output/qwen35-35B-A3B",
    )
    parser.add_argument(
        "--prefill-pte",
        default=defaults["prefill_pte"],
        help="Existing full prefill PTE on the device; sequence plans must match the prompt length",
    )
    parser.add_argument(
        "--decode-pte",
        default=defaults["decode_pte"],
        help="Existing full AR1 PTE path on the device",
    )
    parser.add_argument("--remote-weights", default=defaults["remote_weights"])
    parser.add_argument("--remote-root", default=defaults["remote_root"])
    parser.add_argument("--prompt", default="什么是注意力机制,你用python来实现一下")
    parser.add_argument(
        "--prompt-tokens", type=int, choices=list(range(128, 1025, 128))
    )
    parser.add_argument("--image-path", type=Path)
    parser.add_argument(
        "--vision-artifacts",
        type=Path,
        default=REPO
        / defaults.get("vision_artifacts", defaults["artifacts"] + "/vision"),
    )
    parser.add_argument(
        "--decode-capacity", type=int, choices=[1152, 2048], default=1152
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--prefill-token-file", type=Path)
    parser.add_argument("--teacher-tokens", type=Path)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--save-all-logits", action="store_true")
    parser.add_argument("--trace-layers", action="store_true")
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--io-workers", type=int, choices=[4, 8, 16], default=8)
    parser.add_argument(
        "--read-chunk-kib",
        type=int,
        choices=[64, 128, 256, 512, 1024],
        default=256,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check local artifacts and build outputs without ADB or inference",
    )
    args = parser.parse_args()
    if args.prompt_tokens is None and not args.image_path:
        args.prompt_tokens = 128
    if args.image_path and args.prompt.count("<image>") > 1:
        parser.error("Only one image is supported")
    if (
        not 1
        <= args.max_new_tokens
        <= args.decode_capacity + 1 - (args.prompt_tokens or 0)
    ):
        parser.error("--max-new-tokens exceeds the configured context capacity")
    for value in [
        args.prefill_pte,
        args.decode_pte,
        args.remote_root,
        args.remote_weights,
    ]:
        if (
            not re.fullmatch(r"/[A-Za-z0-9_./-]+", value)
            or ".." in PurePosixPath(value).parts
        ):
            parser.error(
                "Device paths must be absolute paths without whitespace, shell syntax or .. components"
            )
    if not args.remote_root.startswith("/data/local/tmp/"):
        parser.error(
            "--remote-root must be a dedicated directory under /data/local/tmp"
        )
    return args


def check_artifacts(args):
    required = [
        args.build_dir / "prefill_handoff_runner",
        args.build_dir / "libmoe_decode.so",
        args.build_dir / "native-prefill/libinteger_rpc_skel.so",
        args.build_dir / "native-decode/libinteger_rpc_skel.so",
        args.build_dir / "op-package/hexagon-v85/libQnnDeltaNetOpPackage_HTP.so",
        args.model / "config.json",
        args.model / "generation_config.json",
        args.artifacts / "cold-experts.i32",
        *[
            args.artifacts / "assets" / name
            for name in ["cache-encoding.bin", "cos.f32", "sin.f32"]
        ],
    ]
    for layer in range(40):
        required.extend(
            [
                args.artifacts / f"banks/layer{layer}/BANK.json",
                args.artifacts / f"indices/layer{layer}/bank-index.bin",
            ]
        )
        directory = (
            args.artifacts / f"assets/context{args.decode_capacity}-fused"
            if layer % 4 == 3
            else args.artifacts / "assets/gdn-fused"
        ) / f"layer{layer}"
        required.extend([directory / "attention.bin", directory / "io.bin"])
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing artifacts/build outputs:\n" + "\n".join(missing)
        )
    text = json.loads((args.model / "config.json").read_text())["text_config"]
    if (
        text["num_hidden_layers"],
        text["hidden_size"],
        text["num_experts"],
        text["num_experts_per_tok"],
    ) != (40, 2048, 256, 8):
        raise ValueError(
            "This runner requires the 40-layer Qwen3.5-35B-A3B MoE configuration"
        )
    if (args.artifacts / "cold-experts.i32").stat().st_size != 40 * 64 * 4:
        raise ValueError("Expected 64 cold startup experts per layer")
    for layer in range(40):
        bank = json.loads(
            (args.artifacts / f"banks/layer{layer}/BANK.json").read_text()
        )
        if bank["codebook"] != [-7, -2, 2, 7] or not bank["sequential"]:
            raise ValueError(
                f"Layer {layer} requires sequentially reconstructed W2 weights"
            )
        with (args.artifacts / f"indices/layer{layer}/bank-index.bin").open(
            "rb"
        ) as handle:
            if struct.unpack("<8I", handle.read(32))[-1] != 7:
                raise ValueError(f"Layer {layer}: incompatible expert index version")
        directory = (
            args.artifacts / f"assets/context{args.decode_capacity}-fused"
            if layer % 4 == 3
            else args.artifacts / "assets/gdn-fused"
        ) / f"layer{layer}"
        raw = (directory / "io.bin").read_bytes()
        if len(raw) % 4:
            raise ValueError(f"Layer {layer}: malformed graph IO order")
        order = struct.unpack("<" + "I" * (len(raw) // 4), raw)
        inputs = 6 if layer % 4 == 3 else 3
        if (
            len(order) != 2 + inputs + 7
            or order[:2] != (inputs, 7)
            or sorted(order[2 : 2 + inputs]) != list(range(inputs))
            or sorted(order[2 + inputs :]) != list(range(7))
        ):
            raise ValueError(
                f"Layer {layer}: expected fused Attention/Router/shared IO"
            )


def prepare_prompt(args, tokenizer, np, vision=None):
    if args.prefill_token_file:
        ids = np.fromfile(args.prefill_token_file, dtype="<i8").tolist()
        if (
            not ids
            or (args.prompt_tokens is not None and len(ids) != args.prompt_tokens)
            or not 128 <= len(ids) <= 1024
            or len(ids) % 128
            or min(ids) < 0
            or max(ids) >= len(tokenizer)
        ):
            raise ValueError(
                "The token file must contain exactly --prompt-tokens valid int64 token IDs"
            )
        return [], ids
    if args.image_path:
        from multimodal import image_prompt

        messages, ids, _ = image_prompt(
            tokenizer, args.prompt, vision["output_shape"][1], args.prompt_tokens
        )
        return messages, ids
    base = (
        "你是一个有帮助的人工智能助手。请使用简洁、准确、自然的中文回答用户的问题。"
        "不要编造信息；如果不确定，请明确说明。请直接回答问题，无需展示思考过程。"
        "你可以回答日常生活、语言学习、数学计算和科学知识方面的问题。"
        "请注意理解问题的具体要求，区分已知事实和个人推测，不要编造自己的身份或经历。"
    )
    phrases = [
        "回答时语气友好。",
        "请不要重复用户的问题。",
        "回答长度控制在三句话以内。",
        "如果是计算问题，请给出正确结果。",
        "这是一次中文对话测试。",
        "现在开始。",
        "请优先保证事实准确。",
        "请不要输出无关内容。",
        "不需要列出过长的清单。",
        "请用完整的句子表达。",
    ]
    if args.prompt_tokens == 128:
        for count in range(len(phrases) + 1):
            for extra in itertools.combinations(phrases, count):
                messages = [
                    {"role": "system", "content": base + "".join(extra)},
                    {"role": "user", "content": args.prompt},
                ]
                ids = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    return_dict=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                if len(ids) == 128:
                    return messages, ids
    else:
        document = (
            "注意力机制根据查询和键之间的匹配程度计算权重，再用这些权重汇总值向量。"
            "Transformer中的多头注意力在不同子空间处理输入，位置编码表示序列中的相对或绝对位置。"
            "自回归模型使用因果掩码，只允许当前位置关注当前及之前的信息。"
            "预填充阶段可以并行处理一批输入；生成阶段每次处理上一个输出token，并更新已有缓存。"
            "键值缓存避免重新计算历史token的投影，但较长序列需要更多内存。"
            "混合专家模型为每个token选择一组专家，路由分数决定各专家输出的加权比例。"
            "内存缓存中的专家可以复用；缺失的权重从外存读入后再参与计算。"
            "最近最少使用策略根据同一层历史访问顺序选择被替换的专家，当前调用使用的专家需要保留。"
            "异步读取可以和已有计算重叠，但实际收益取决于命中率、存储吞吐、调度和内存复制。"
            "评估性能时应分别记录初始化、预填充和逐token生成，并明确输入长度与生成步数。"
        )
        context_ids = tokenizer.encode(document * 12, add_special_tokens=False)
        for length in range(args.prompt_tokens - 200, args.prompt_tokens):
            context = tokenizer.decode(
                context_ids[:length], clean_up_tokenization_spaces=False
            )
            messages = [
                {
                    "role": "system",
                    "content": base + "以下是参考资料：\n" + context,
                },
                {"role": "user", "content": args.prompt},
            ]
            ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            if len(ids) == args.prompt_tokens:
                return messages, ids
    raise ValueError(
        "Cannot construct the exact benchmark prompt length; use --prefill-token-file for arbitrary input"
    )


def main():
    args = parse_args()
    check_artifacts(args)
    vision = None
    if args.image_path:
        from multimodal import vision_metadata

        vision = vision_metadata(args.model, args.vision_artifacts)
        if not args.image_path.is_file():
            raise FileNotFoundError(args.image_path)
    if args.check:
        print(f"Local artifacts and optimized runners ready: {args.artifacts}")
        return
    import numpy as np
    from transformers import AutoTokenizer

    def adb(*items):
        return subprocess.run(
            ["adb", "-s", args.device, *map(str, items)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    def snapshot():
        thermal = adb("shell", "dumpsys thermalservice")
        match = re.search(r"Thermal Status:\s*(\d+)", thermal)
        return dict(
            thermal_status=int(match.group(1)) if match else None,
            meminfo=adb("shell", "cat /proc/meminfo"),
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    messages, ids = prepare_prompt(args, tokenizer, np, vision)
    args.prompt_tokens = len(ids)
    if not 1 <= args.max_new_tokens <= args.decode_capacity + 1 - len(ids):
        raise ValueError("Generation exceeds the configured context capacity")
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
    label = f"w2a8-fused-c{args.decode_capacity}-p{args.prompt_tokens}-cache64"
    run_dir = args.output / (datetime.now().strftime("%Y%m%d-%H%M%S-%f") + "-" + label)
    run_dir.mkdir(parents=True)
    np.asarray(ids, dtype="<i8").tofile(run_dir / "tokens.i64")
    (run_dir / "PROMPT.txt").write_text(tokenizer.decode(ids), encoding="utf-8")
    record = dict(
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
        cache_experts=64,
        cache_policy="lru64",
        front_variant="fused",
        decode_execution="same_process_library",
        vision_residency="model_lifetime" if vision is not None else None,
        state_swap=True,
        kv_append=True,
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
    remote = args.remote_root + "/" + label
    prefill_root, decode_root = remote + "/prefill", remote + "/decode"
    common = [
        "embedding.bf16",
        "indices",
        "cos.f32",
        "sin.f32",
        "libQnnHtp.so",
        "libQnnSystem.so",
        "libQnnHtpV85Stub.so",
        "libQnnHtpV85Skel.so",
        "libQnnHtpPrepare.so",
    ]
    commands = [
        "set -e",
        f"mkdir -p {decode_root}/cache-prefill {prefill_root}",
    ]
    for target, program in [
        (prefill_root, args.prefill_pte),
        (decode_root, args.decode_pte),
    ]:
        source = str(PurePosixPath(program).parent)
        commands.append(
            f"ln -sfn {program} {target}/"
            + ("prefill.pte" if target == prefill_root else "decode.pte")
        )
        for name in common:
            commands.append(f"ln -sfn {source}/{name} {target}/{name}")
        commands.append(f"ln -sfn {args.remote_weights}/indices {target}/indices")
    commands.extend(
        [
            f"ln -sfn {args.remote_weights}/banks {prefill_root}/compact",
            f"ln -sfn {args.remote_weights}/banks {decode_root}/banks",
            f"ln -sfn {prefill_root}/pte-0-logits.f32 {decode_root}/prefill-logits.f32",
        ]
    )
    for layer in range(40):
        for c in range(2):
            commands.append(
                f"ln -sfn {prefill_root}/pte-0-layer{layer}-cache{c}.u16 {decode_root}/cache-prefill/layer{layer}-cache{c}.u16"
            )
    for name in [
        "prefill-history-ids.i32",
        "prefill-history-stamps.u64",
        "prefill-routes.i32",
    ]:
        commands.append(f"ln -sfn {prefill_root}/{name} {decode_root}/{name}")
    adb("shell", "\n".join(commands))
    adb("push", run_dir / "tokens.i64", prefill_root + "/tokens.i64")
    if vision is not None:
        adb("push", args.vision_artifacts / "vision.bin", prefill_root + "/vision.bin")
        adb("push", run_dir / "pixels.f32", prefill_root + "/pixels.f32")
    adb(
        "push",
        args.artifacts / "cold-experts.i32",
        prefill_root + "/cold-experts.i32",
    )
    for target, stage in [(prefill_root, "prefill"), (decode_root, "decode")]:
        adb(
            "push",
            args.build_dir
            / ("prefill_handoff_runner" if stage == "prefill" else "libmoe_decode.so"),
            target
            + (
                "/prefill_handoff_runner" if stage == "prefill" else "/libmoe_decode.so"
            ),
        )
        adb(
            "push",
            args.build_dir / f"native-{stage}/libinteger_rpc_skel.so",
            target
            + (
                "/libinteger_rpc_skel.so"
                if stage == "prefill"
                else "/libinteger_decode_rpc_skel.so"
            ),
        )
        for name in ["cos.f32", "sin.f32"]:
            adb("shell", "rm " + target + "/" + name)
            local = (
                run_dir / name
                if vision is not None
                else args.artifacts / "assets" / name
            )
            adb("push", local, target + "/" + name)
    if args.teacher_tokens:
        adb("push", args.teacher_tokens, decode_root + "/teacher.i64")
    adb("shell", "rm -f " + decode_root + "/libQnnDeltaNetOpPackage_HTP.so")
    adb(
        "push",
        args.build_dir / "op-package/hexagon-v85/libQnnDeltaNetOpPackage_HTP.so",
        decode_root + "/libQnnDeltaNetOpPackage_HTP.so",
    )
    adb(
        "push",
        args.artifacts / "assets/cache-encoding.bin",
        decode_root + "/cache-encoding.bin",
    )
    for layer in range(40):
        full_attention = layer % 4 == 3
        local = (
            args.artifacts / f"assets/context{args.decode_capacity}-fused"
            if full_attention
            else args.artifacts / "assets/gdn-fused"
        ) / f"layer{layer}"
        target = decode_root + (
            f"/layer{layer}" if full_attention else f"/overrides/layer{layer}"
        )
        adb("shell", "mkdir -p " + target)
        adb("push", local / "attention.bin", local / "io.bin", target + "/")

    eos = ",".join(map(str, generation["eos_token_id"]))
    environment = dict(
        NATIVE_MAPPED_IO="1",
        NATIVE_RESIDUAL="1",
        REUSE_EXPERTS="1",
        ASYNC_EXPERT_IO="1",
        ASYNC_DECODE_IO="0",
        EXPERT_IO_WORKERS=str(args.io_workers),
        STARTUP_EXPERT_PREFETCH="overlap",
        STARTUP_IO_WORKERS="4",
        SKIP_EXPERT_ZERO="1",
        EXPERT_BITS="2",
        VISION_INPUT="1" if vision is not None else "0",
        LD_LIBRARY_PATH=prefill_root,
        ADSP_LIBRARY_PATH=prefill_root
        + ";"
        + decode_root
        + ";/odm/lib64/rfsa;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/dsp",
        FRONT_OVERRIDE=decode_root + "/overrides",
        DECODE_TRACE_LAYERS="1" if args.trace_layers else "0",
        KV_APPEND_ONLY="1",
        EXPERT_STAGE_READY="0",
        GDN_STATE_SWAP="1",
        SAVE_ALL_LOGITS="1" if args.save_all_logits else "0",
        DECODE_CONTEXT_CAPACITY=str(args.decode_capacity),
        PREFILL_TOKENS=str(args.prompt_tokens),
        EXPERT_READ_CHUNK_KIB=str(args.read_chunk_kib),
        DECODE_EXPERT_IO_WORKERS=str(args.io_workers),
        OVERLAP_STATE_FEEDBACK="1",
        MAILBOX_BACKOFF_US="0",
        MAILBOX_POLL_DELAY_NS="0",
        NATIVE_MAILBOX="1",
        HIT_FIRST_EXPERTS="1",
        EXPERT_READY_IO="1",
        HANDOFF_DECODE_ROOT=decode_root,
        HANDOFF_MAX_TOKENS=str(args.max_new_tokens),
        HANDOFF_EOS_IDS=eos,
        INHERIT_EXPERT_CACHE="1",
        SEED_PREFILL_LRU="0",
        RESIDENT_PREFAULT="0",
    )
    if args.ignore_eos:
        environment["BENCH_IGNORE_EOS"] = "1"
    if args.teacher_tokens:
        environment["BENCH_TOKENS_PATH"] = decode_root + "/teacher.i64"
    command = "cd " + shlex.quote(prefill_root) + " && env "
    command += " ".join(
        shlex.quote(key + "=" + value) for key, value in environment.items()
    )
    command += f" timeout {max(240, args.max_new_tokens * 2 + 30)} ./prefill_handoff_runner {shlex.quote(prefill_root)} 1 0"
    record.update(command=command, before=snapshot())
    (run_dir / "INPUT.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n"
    )
    generated, timings = [], []
    displayed, previous = "", None
    started = time.perf_counter()
    with (run_dir / "runner.log").open("w") as log:
        with subprocess.Popen(
            ["adb", "-s", args.device, "shell", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        ) as process:
            for line in process.stdout:
                arrival = time.perf_counter()
                log.write(line)
                if line.startswith(("PREFILL_TOKEN ", "TOKEN ")):
                    event = json.loads(line.split(" ", 1)[1])
                    if event["index"] == 0 and generated:
                        if event["id"] != generated[0]:
                            raise RuntimeError("Handoff first token mismatch")
                        continue
                    if event["index"] != len(generated):
                        raise RuntimeError("Token stream order mismatch")
                    generated.append(event["id"])
                    row = dict(
                        event,
                        host_ms=(arrival - started) * 1000,
                        host_gap_ms=(
                            None if previous is None else (arrival - previous) * 1000
                        ),
                    )
                    timings.append(row)
                    previous = arrival
                    text = tokenizer.decode(
                        generated,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    ).rstrip("\ufffd")
                    if not text.startswith(displayed):
                        raise RuntimeError("Decoded stream prefix changed")
                    delta, displayed = text[len(displayed) :], text
                    if args.text_only:
                        print(delta, end="", flush=True)
                    else:
                        print(
                            f"TOKEN {event['index']:03d} gap_ms={row['host_gap_ms']} "
                            + json.dumps(delta, ensure_ascii=False),
                            flush=True,
                        )
                elif line.startswith(
                    (
                        "PTE_RESULT ",
                        "VISION_ENCODE ",
                        "MODEL_RESIDENT ",
                        "PREFILL_TOTAL ",
                        "PTE_READY ",
                        "STARTUP_COLD_PRELOAD ",
                        "DECODE_READY ",
                        "EXPERT_HANDOFF_",
                        "EXPERT_HISTORY_",
                        "PREFILL_REUSE_TOTAL ",
                        "EXPERT_RESIDENT_",
                        "EXPERT_CACHE_READY ",
                        "EXPERT_IO_TOTAL ",
                        "DECODE_RESULT ",
                        "DECODE_LOOP ",
                        "DECODE_STOP ",
                        "FATAL ",
                    )
                ):
                    print(line.rstrip(), flush=True)
            exit_code = process.wait()
    record.update(after=snapshot(), exit_code=exit_code)
    (run_dir / "RUN.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n"
    )
    if exit_code:
        raise RuntimeError(f"Runner exited with {exit_code}; see {run_dir}/runner.log")
    if vision is not None:
        adb("pull", prefill_root + "/image-features.f32", run_dir)
    (run_dir / "TOKEN_TIMINGS.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in timings)
    )
    decode_files = [
        "input.i64",
        "output.i64",
        "hidden.f32",
        "logits.f32",
        *[f"layer{i}-cache{c}.u16" for i in range(40) for c in range(2)],
    ]
    adb(
        "pull",
        *[decode_root + "/decode-pte-0-" + suffix for suffix in decode_files],
        run_dir,
    )
    for suffix in decode_files:
        (run_dir / ("decode-pte-0-" + suffix)).rename(run_dir / suffix)
    history_files = [
        "prefill-history-ids.i32",
        "prefill-history-stamps.u64",
        "prefill-routes.i32",
        "prefill-all-routes.i32",
        "prefill-chunk-seeds.i32",
        "prefill-chunk-stamps.u64",
    ]
    adb("pull", *[prefill_root + "/" + name for name in history_files], run_dir)
    adb(
        "pull",
        *[
            decode_root + "/" + name
            for name in [
                "cache-final-ids.i32",
                "cache-seed-ids.i32",
                "reused-ids.i32",
                "reused-stamps.u64",
                "hot-masks.u32",
            ]
        ],
        run_dir,
    )
    (run_dir / "prefill").mkdir()
    prefill_files = [
        "hidden.f32",
        "logits.f32",
        *[f"layer{i}-cache{c}.u16" for i in range(40) for c in range(2)],
    ]
    adb(
        "pull",
        *[prefill_root + "/pte-0-" + suffix for suffix in prefill_files],
        run_dir / "prefill",
    )
    for suffix in prefill_files:
        (run_dir / "prefill" / ("pte-0-" + suffix)).rename(run_dir / "prefill" / suffix)
    inputs = np.fromfile(run_dir / "input.i64", dtype="<i8")
    outputs = np.fromfile(run_dir / "output.i64", dtype="<i8")
    if (
        len(inputs) != len(outputs)
        or len(inputs) != len(generated) - 1
        or generated[1:] != outputs.tolist()
    ):
        raise RuntimeError("Generated token counts disagree with device outputs")
    if args.teacher_tokens:
        if not np.array_equal(inputs, teacher[: len(inputs)]):
            raise RuntimeError("Teacher forcing input mismatch")
    elif len(inputs) and (
        inputs[0] != generated[0] or not np.array_equal(inputs[1:], outputs[:-1])
    ):
        raise RuntimeError("Greedy decode feedback mismatch")
    end = next(
        (i for i, token in enumerate(generated) if token in generation["eos_token_id"]),
        None,
    )
    if not args.ignore_eos and end is not None and end != len(generated) - 1:
        raise RuntimeError("Generation continued after EOS")
    answer = dict(
        answer=tokenizer.decode(
            generated if end is None else generated[:end],
            skip_special_tokens=True,
        ),
        input_ids=inputs.tolist(),
        output_ids=outputs.tolist(),
        eos_index=end,
        greedy_feedback_verified=not bool(args.teacher_tokens),
        teacher_forcing_verified=bool(args.teacher_tokens),
        generated_tokens=len(generated),
    )
    (run_dir / "ANSWER.json").write_text(
        json.dumps(answer, ensure_ascii=False, indent=2) + "\n"
    )
    print("\nANSWER " + answer["answer"], flush=True)
    print("SAVED", run_dir, flush=True)


if __name__ == "__main__":
    main()
