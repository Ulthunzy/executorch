from pathlib import PurePosixPath
import shlex

from .artifacts import decode_layer_dir


def deploy_artifacts(args, device, paths, run_dir, vision):
    adb = device.adb
    prefill_root, decode_root = paths.prefill, paths.decode
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
        source = args.runtime_device_dir or str(PurePosixPath(program).parent)
        local = (
            args.local_prefill_pte if target == prefill_root else args.local_decode_pte
        )
        filename = "prefill.pte" if target == prefill_root else "decode.pte"
        if local is None:
            commands.append(f"ln -sfn {program} {target}/{filename}")
        else:
            commands.append(f"rm -f {target}/{filename}")
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
    for local, target in (
        (args.local_prefill_pte, prefill_root + "/prefill.pte"),
        (args.local_decode_pte, decode_root + "/decode.pte"),
    ):
        if local is not None:
            adb("push", local, target)
    if args.embedding_table:
        embedding_remote = args.remote_root + "/embedding.int4"
        adb("push", args.embedding_table, embedding_remote)
        for target in (prefill_root, decode_root):
            adb("shell", f"ln -sfn {embedding_remote} {target}/embedding.int4")
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
        if args.shared_model:
            continue
        full_attention = layer % 4 == 3
        local = decode_layer_dir(args, layer)
        target = decode_root + (
            f"/layer{layer}" if full_attention else f"/overrides/layer{layer}"
        )
        adb("shell", "mkdir -p " + target)
        adb("push", local / "attention.bin", local / "io.bin", target + "/")

    if args.shared_model:
        shared_remote = paths.shared_model
        adb("shell", "mkdir -p " + shared_remote)
        adb("push", args.build_dir / "libqwen35_shared.so", prefill_root + "/")
        adb("shell", "mkdir -p " + shared_remote + "/head")
        adb(
            "push",
            args.shared_model / "head/graph.qnn",
            args.shared_model / "head/weights.bin",
            shared_remote + "/head/",
        )
        for group in range(10):
            local = args.shared_model / f"group{group}"
            target = shared_remote + f"/group{group}"
            adb("shell", "mkdir -p " + target)
            adb(
                "push",
                local / "graph.qnn",
                local / "weights.bin",
                local / "GRAPHS.json",
                *sorted(local.glob("*.io")),
                target + "/",
            )


def build_command(args, paths, generation, vision):
    prefill_root, decode_root = paths.prefill, paths.decode
    shared_remote = paths.shared_model
    eos = ",".join(map(str, generation["eos_token_id"]))
    environment = dict(
        EMBEDDING_BITS="4" if args.embedding_table else "16",
        NATIVE_MAPPED_IO="1",
        NATIVE_RESIDUAL="1",
        REUSE_EXPERTS="1",
        EXPERT_CACHE_SLOTS=str(args.cache_experts),
        ASYNC_EXPERT_IO="1" if args.prefill_async_io else "0",
        ASYNC_DECODE_IO="0",
        EXPERT_IO_WORKERS=str(args.io_workers),
        STARTUP_EXPERT_PREFETCH=args.startup_prefetch,
        STARTUP_IO_WORKERS=str(args.startup_io_workers),
        SKIP_EXPERT_ZERO="1",
        EXPERT_BITS=str(args.model_settings.quantization.expert_bits),
        VISION_INPUT="1" if vision is not None else "0",
        LD_LIBRARY_PATH=prefill_root,
        ADSP_LIBRARY_PATH=prefill_root
        + ";"
        + decode_root
        + ";/odm/lib64/rfsa;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/dsp",
        FRONT_OVERRIDE=decode_root + "/overrides",
        DECODE_TRACE_LAYERS="1" if args.trace_layers else "0",
        KV_APPEND_ONLY="1" if args.kv_append_only else "0",
        EXPERT_STAGE_READY="0",
        GDN_STATE_SWAP="1" if args.state_swap else "0",
        SAVE_ALL_LOGITS="1" if args.save_all_logits else "0",
        DECODE_CONTEXT_CAPACITY=str(args.decode_capacity),
        PREFILL_TOKENS=str(args.prompt_tokens),
        EXPERT_READ_CHUNK_KIB=str(args.read_chunk_kib),
        DECODE_EXPERT_IO_WORKERS=str(args.io_workers),
        OVERLAP_STATE_FEEDBACK="1" if args.overlap_feedback else "0",
        MAILBOX_BACKOFF_US="0",
        MAILBOX_POLL_DELAY_NS="0",
        NATIVE_MAILBOX="1",
        HIT_FIRST_EXPERTS="1" if args.decode_hit_first else "0",
        EXPERT_READY_IO="1" if args.decode_ready_io else "0",
        HANDOFF_DECODE_ROOT=decode_root,
        HANDOFF_MAX_TOKENS=str(args.max_new_tokens),
        HANDOFF_EOS_IDS=eos,
        INHERIT_EXPERT_CACHE="1",
        SEED_PREFILL_LRU="0",
        RESIDENT_PREFAULT="0",
    )
    if args.shared_model:
        environment["SHARED_MODEL_ROOT"] = shared_remote
        environment["SHARED_OP_PACKAGE"] = (
            decode_root + "/libQnnDeltaNetOpPackage_HTP.so"
        )
        environment.pop("FRONT_OVERRIDE")
    if args.decode_context_load_mode == "serial":
        environment["QNN_SERIAL_CONTEXTS"] = "1"
    if args.ignore_eos:
        environment["BENCH_IGNORE_EOS"] = "1"
    if args.teacher_tokens:
        environment["BENCH_TOKENS_PATH"] = decode_root + "/teacher.i64"
    command = "cd " + shlex.quote(prefill_root) + " && env "
    command += " ".join(
        shlex.quote(key + "=" + value) for key, value in environment.items()
    )
    command += f" timeout {max(240, args.max_new_tokens * 2 + 30)} ./prefill_handoff_runner {shlex.quote(prefill_root)} 1 0"
    return command
