from pathlib import PurePosixPath
import hashlib
import shlex
import subprocess

from executorch.examples.qualcomm.oss_scripts.moe.inference.artifacts import (
    decode_layer_dir,
)
from executorch.examples.qualcomm.oss_scripts.moe.inference.image_cold_profile import (
    image_cold_assets,
    MANIFEST_NAME,
    TABLE_NAME,
)
from executorch.examples.qualcomm.oss_scripts.moe.inference.plans import (
    load_prefill_bundle,
)
from executorch.examples.qualcomm.oss_scripts.moe.utils import runtime_assets
from executorch.examples.qualcomm.oss_scripts.moe.inference.text_cold_profile import (
    text_cold_assets,
    MANIFEST_NAME as TEXT_MANIFEST_NAME,
    TABLE_NAME as TEXT_TABLE_NAME,
)


def verify_device_file(device, source, remote):
    with source.open("rb") as stream:
        expected = hashlib.file_digest(stream, "sha256").hexdigest()
    try:
        actual = device.adb("shell", "sha256sum " + shlex.quote(remote)).split()
    except subprocess.CalledProcessError as error:
        raise FileNotFoundError(
            f"Cannot verify deployed file {remote}; deploy the selected weights first"
        ) from error
    if not actual or actual[0] != expected:
        raise ValueError(
            f"Existing device file {remote} differs from the selected artifacts"
        )


def deploy_embedding(args, device, prefill_root, decode_root):
    name = "embedding.int4" if args.embedding_table else "embedding.bf16"
    source = args.embedding_table or args.artifacts / name
    remote = args.remote_root + "/" + name
    if args.deploy_weights:
        device.adb("push", "--sync", source, remote)
    else:
        verify_device_file(device, source, remote)
    for target in (prefill_root, decode_root):
        device.adb("shell", f"ln -sfn {remote} {target}/{name}")


def deploy_vision(args, device, profile_root, prefill_root):
    source = args.vision_artifacts / "vision.bin"
    target = prefill_root + "/vision.bin"
    if args.runtime_vision_bin:
        verify_device_file(device, source, args.runtime_vision_bin)
        device.adb(
            "shell",
            "ln -sfn "
            + shlex.quote(args.runtime_vision_bin)
            + " "
            + shlex.quote(target),
        )
    else:
        device.adb("push", source, target)
    device.adb(
        "push", args.vision_artifacts / "VISION.json", profile_root + "/VISION.json"
    )


def deploy_shared_model(args, device, shared_remote, prefill_root):
    groups = [
        [args.shared_model / "head" / name for name in ("graph.qnn", "weights.bin")]
    ]
    for group in range(10):
        directory = args.shared_model / f"group{group}"
        groups.append(
            [directory / name for name in ("graph.qnn", "weights.bin", "GRAPHS.json")]
            + sorted(directory.glob("*.io"))
        )
    if not args.deploy_weights:
        for files in groups:
            for source in files:
                relative = source.relative_to(args.shared_model).as_posix()
                verify_device_file(device, source, shared_remote + "/" + relative)
    else:
        device.adb("shell", "mkdir -p " + shared_remote)
        for files in groups:
            target = shared_remote + "/" + files[0].parent.name
            device.adb("shell", "mkdir -p " + target)
            device.adb("push", "--sync", *files, target + "/")


def deploy_artifacts(args, device, paths, run_dir, vision):
    chat_plans = load_prefill_bundle(args)
    if args.deploy_weights:
        device.adb(
            "shell",
            "mkdir -p "
            + args.remote_weights
            + "/banks "
            + args.remote_weights
            + "/indices",
        )
        for name in ("banks", "indices"):
            device.adb(
                "push",
                "--sync",
                str(args.artifacts / name) + "/.",
                args.remote_weights + "/" + name + "/",
            )
    adb = device.adb
    prefill_root, decode_root = paths.prefill, paths.decode
    common = [
        "libQnnHtp.so",
        "libQnnSystem.so",
        f"libQnnHtp{args.dsp_arch.upper()}Stub.so",
        f"libQnnHtp{args.dsp_arch.upper()}Skel.so",
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
        if target == decode_root or args.runner == "diagnostic":
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
    if args.runner == "chat":
        adb("push", "--sync", args.build_dir / "qwen35_moe_runner", prefill_root + "/")
    if chat_plans is not None:
        adb("shell", "mkdir -p " + prefill_root + "/plans")
        adb("push", str(chat_plans) + "/.", prefill_root + "/plans/")
    adb("push", args.model / "tokenizer.json", args.remote_root + "/tokenizer.json")
    profile_root = str(PurePosixPath(prefill_root).parent)
    for name in ("cos", "sin"):
        adb(
            "push",
            runtime_assets(args) / f"{name}.f32",
            f"{profile_root}/base-{name}.f32",
        )
    adb("push", args.model / "config.json", profile_root + "/model-config.json")
    if (
        args.with_vision
        and args.vision_artifacts
        and (args.vision_artifacts / "vision.bin").is_file()
    ):
        deploy_vision(args, device, profile_root, prefill_root)
    else:
        adb("shell", "rm -f " + prefill_root + "/vision.bin")
    for local, target in (
        (args.local_prefill_pte, prefill_root + "/prefill.pte"),
        (args.local_decode_pte, decode_root + "/decode.pte"),
    ):
        if local is not None:
            adb("push", local, target)
    deploy_embedding(args, device, prefill_root, decode_root)
    if (run_dir / "tokens.i64").is_file():
        adb("push", run_dir / "tokens.i64", prefill_root + "/tokens.i64")
    if vision is not None:
        adb("push", run_dir / "pixels.f32", prefill_root + "/pixels.f32")
    adb(
        "push",
        args.artifacts / "cold-experts.i32",
        prefill_root + "/cold-experts.i32",
    )
    optional_cold = image_cold_assets(
        args.artifacts,
        args.model_settings.target.soc_model,
        args.image_cold_profile_dir,
    )
    adb(
        "shell",
        "rm -f "
        + shlex.quote(prefill_root + "/" + TABLE_NAME)
        + " "
        + shlex.quote(prefill_root + "/" + MANIFEST_NAME),
    )
    if optional_cold:
        for local in optional_cold:
            adb("push", local, prefill_root + "/" + local.name)
    optional_text_cold = text_cold_assets(
        args.artifacts,
        args.model_settings.target.soc_model,
        args.image_cold_profile_dir,
    )
    adb(
        "shell",
        "rm -f "
        + shlex.quote(prefill_root + "/" + TEXT_TABLE_NAME)
        + " "
        + shlex.quote(prefill_root + "/" + TEXT_MANIFEST_NAME),
    )
    for local in optional_text_cold:
        adb("push", local, prefill_root + "/" + local.name)
    for target, stage in [(prefill_root, "prefill"), (decode_root, "decode")]:
        adb("push", "--sync", args.build_dir / "libet_infer.so", target + "/")
        adb("push", "--sync", args.build_dir / "libet_qnnop.so", target + "/")
        if stage == "prefill":
            adb("push", args.build_dir / "prefill_handoff_runner", target + "/")
        for name in ["cos.f32", "sin.f32"]:
            local = (
                run_dir / name if vision is not None else runtime_assets(args) / name
            )
            adb("push", local, target + "/" + name)
    if args.teacher_tokens:
        adb("push", args.teacher_tokens, decode_root + "/teacher.i64")
    adb(
        "push",
        runtime_assets(args) / "cache-encoding.bin",
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
        deploy_shared_model(args, device, paths.shared_model, prefill_root)


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
        PREFILL_INITIAL_IMAGE="1" if vision is not None else "0",
        LD_LIBRARY_PATH=prefill_root,
        ADSP_LIBRARY_PATH=prefill_root
        + ";"
        + decode_root
        + ";/odm/lib64/rfsa;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/dsp",
        FRONT_OVERRIDE=decode_root + "/overrides",
        DECODE_TRACE_LAYERS="1" if args.trace_layers else "0",
        MOE_PROFILE_TOTAL="1" if args.profile else "0",
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
        environment["SHARED_OP_PACKAGE"] = decode_root + "/libet_qnnop.so"
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
