import argparse
from dataclasses import replace
import json
from pathlib import Path, PurePosixPath
import re

from executorch.examples.qualcomm.oss_scripts.moe.model.qwen35 import (
    DEFAULT_MODEL_CONFIG,
    load_model_config,
)
from executorch.examples.qualcomm.oss_scripts.moe.target import SOC_MODELS, TargetConfig

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[3]


def parse_args(argv=None):
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/qwen35_35b_a3b/sm8975.json"
    )
    config_parser.add_argument("--model-config", type=Path)
    config_parser.add_argument(
        "--weights-path", "--artifacts", dest="weights_path", type=Path
    )
    selection, _ = config_parser.parse_known_args(argv)
    defaults = json.loads(selection.config.read_text())
    image_cold_profile_dir = None
    package_manifest = None
    if selection.weights_path:
        weights = selection.weights_path.resolve()
        packaged = weights.is_file()
        if packaged:
            from executorch.examples.qualcomm.oss_scripts.moe.package import (
                PackageReader,
            )

            image_cold_profile_dir = weights.parent
            package = PackageReader(weights)
            weights = package.extract()
            package_manifest = package.manifest
            selection.weights_path = weights
        exported_profile = weights / "device.json"
        if exported_profile.is_file():
            defaults.update(json.loads(exported_profile.read_text()))
        if packaged:
            defaults["model"] = str(weights)
        defaults.update(
            artifacts=str(weights),
            shared_model=str(
                weights
                / (
                    "shared-model"
                    if (weights / "shared-model").is_dir()
                    else "separated"
                )
            ),
            embedding_table=str(weights / "embedding.int4"),
            vision_artifacts=str(weights / "vision"),
            local_prefill_pte=None,
            local_decode_pte=(
                str(weights / "plans/decode.pte")
                if (weights / "plans/decode.pte").is_file()
                else None
            ),
        )
        if (weights / "MODEL_CONFIG.json").is_file():
            defaults["model_config"] = str(weights / "MODEL_CONFIG.json")
    if defaults.get("model_family", "qwen35") != "qwen35":
        config_parser.error("Qwen3.5 runtime requires model_family=qwen35")
    model_configuration = selection.model_config or Path(
        defaults.get("model_config", DEFAULT_MODEL_CONFIG)
    )
    if selection.model_config is None and not model_configuration.is_absolute():
        model_configuration = REPO / model_configuration
    config_parser.set_defaults(model_config=model_configuration)
    try:
        settings = load_model_config(model_configuration)
    except (ValueError, OSError) as error:
        config_parser.error(str(error))
    schedule = settings.schedule
    parser = argparse.ArgumentParser(
        parents=[config_parser],
        description="Qwen3.5-35B-A3B Native MoE inference",
    )
    parser.add_argument("--device", default=defaults["device"])
    parser.add_argument(
        "--soc-model",
        "--soc_model",
        choices=SOC_MODELS,
        default=(
            settings.target.soc_model
            if selection.weights_path or selection.model_config
            else defaults.get("soc_model", settings.target.soc_model)
        ),
    )
    parser.add_argument("--model", type=Path, default=Path(defaults["model"]))
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=REPO / defaults.get("build_dir", "build/moe/qwen35-35B-A3B"),
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
    parser.add_argument(
        "--embedding-table",
        type=Path,
        default=(
            REPO / defaults["embedding_table"]
            if settings.quantization.embedding == "int4_g32"
            and defaults.get("embedding_table")
            else None
        ),
        help="Packed group-32 INT4 table; recipe is selected by --model-config",
    )
    parser.add_argument(
        "--decode-context-load-mode",
        choices=["auto", "async", "serial"],
        default=schedule.context_load_mode,
        help="auto uses the artifact manifest; serial enables per-context IO memory estimation",
    )
    parser.add_argument(
        "--shared-model",
        type=Path,
        default=(
            REPO / defaults["shared_model"] if defaults.get("shared_model") else None
        ),
        help="Separated shared contexts for all 40 layers",
    )
    parser.add_argument("--prompt", default="什么是注意力机制,你用python来实现一下")
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        choices=range(128, settings.context.max_seq_len, 128),
    )
    parser.add_argument("--image-path", "--image_path", type=Path)
    parser.add_argument(
        "--with-vision",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Keep the vision encoder available for initial or later /image turns",
    )
    parser.add_argument(
        "--temperature", "--temp", type=float, default=settings.generation.temperature
    )
    parser.add_argument(
        "--interactive",
        action=argparse.BooleanOptionalAction,
        default=settings.generation.interactive,
    )
    parser.add_argument(
        "--stream",
        action=argparse.BooleanOptionalAction,
        default=settings.generation.stream,
    )
    parser.add_argument(
        "--runner",
        choices=["chat", "diagnostic"],
        default=None,
        help="Device tokenizer/session by default; diagnostic retains raw dumps and teacher-forcing",
    )
    parser.add_argument(
        "--deploy-weights", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--vision-artifacts",
        type=Path,
        default=REPO
        / defaults.get("vision_artifacts", defaults["artifacts"] + "/vision"),
    )
    parser.add_argument(
        "--runtime-vision-bin",
        help="Reuse this existing device vision.bin after checking its SHA256",
    )
    parser.add_argument(
        "--max-seq-len",
        "--decode-capacity",
        dest="decode_capacity",
        type=int,
        default=settings.context.max_seq_len,
    )
    parser.add_argument(
        "--kv-dtype", choices=["uint16"], default=settings.cache.kv_dtype
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=settings.generation.max_new_tokens
    )
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=settings.generation.enable_thinking,
    )
    parser.add_argument("--system-prompt", default=settings.generation.system_prompt)
    parser.add_argument(
        "--prefill-order",
        choices=["layer_major", "chunk_major"],
        default=schedule.prefill_order,
        help="chunk_major is deprecated and redirects to the unified layer_major scheduler",
    )
    parser.add_argument(
        "--prefill-async-io",
        action=argparse.BooleanOptionalAction,
        default=schedule.prefill_async_io,
    )
    parser.add_argument(
        "--startup-prefetch",
        choices=["none", "sync", "overlap"],
        default=schedule.startup_prefetch,
    )
    parser.add_argument(
        "--startup-io-workers", type=int, default=schedule.startup_io_workers
    )
    parser.add_argument(
        "--decode-hit-first",
        action=argparse.BooleanOptionalAction,
        default=schedule.decode_hit_first,
    )
    parser.add_argument(
        "--decode-ready-io",
        action=argparse.BooleanOptionalAction,
        default=schedule.decode_ready_io,
    )
    parser.add_argument(
        "--overlap-feedback",
        action=argparse.BooleanOptionalAction,
        default=schedule.overlap_feedback,
    )
    parser.add_argument(
        "--kv-append-only",
        action=argparse.BooleanOptionalAction,
        default=settings.cache.append_only,
    )
    parser.add_argument(
        "--state-swap",
        action=argparse.BooleanOptionalAction,
        default=settings.cache.state_swap,
    )
    parser.add_argument(
        "--local-prefill-pte",
        type=Path,
        default=(
            Path(defaults["local_prefill_pte"])
            if defaults.get("local_prefill_pte")
            else None
        ),
        help="Local packed plan; its JSON sidecar is validated and deployed",
    )
    parser.add_argument(
        "--local-decode-pte",
        type=Path,
        default=(
            Path(defaults["local_decode_pte"])
            if defaults.get("local_decode_pte")
            else None
        ),
        help="Local packed decode plan and JSON sidecar",
    )
    parser.add_argument(
        "--runtime-device-dir",
        default=defaults.get("runtime_device_dir"),
        help="Device directory containing QNN runtime libraries",
    )
    parser.add_argument(
        "--cache-experts",
        type=int,
        choices=[16, 32, 64],
        default=defaults.get("cache_experts", schedule.cache_experts),
        help="Resident LRU expert slots per layer, shared by prefill and decode",
    )
    parser.add_argument("--prefill-token-file", type=Path)
    parser.add_argument("--teacher-tokens", type=Path)
    parser.add_argument(
        "--ignore-eos",
        action=argparse.BooleanOptionalAction,
        default=settings.generation.ignore_eos,
    )
    parser.add_argument("--save-all-logits", action="store_true")
    parser.add_argument("--trace-layers", action="store_true")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Report graph/IO and cumulative DSP timings; requires a MOE_NATIVE_PROFILE=1 build",
    )
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument(
        "--io-workers", type=int, choices=[4, 8, 16], default=schedule.io_workers
    )
    parser.add_argument(
        "--prefill-io-workers",
        type=int,
        choices=[4, 8, 16],
        default=defaults.get("prefill_io_workers"),
        help="Override chat prefill readers independently of decode",
    )
    parser.add_argument(
        "--read-chunk-kib",
        type=int,
        choices=[64, 128, 256, 512, 1024],
        default=defaults.get("read_chunk_kib", schedule.read_chunk_kib),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check local artifacts and build outputs without ADB or inference",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print effective model settings without loading artifacts or using ADB",
    )
    args = parser.parse_args(argv)
    args.image_cold_profile_dir = image_cold_profile_dir
    args.package_manifest = package_manifest
    if args.runner is None:
        args.runner = (
            "diagnostic"
            if (
                args.prefill_token_file
                or args.teacher_tokens
                or args.save_all_logits
                or args.trace_layers
            )
            else "chat"
        )
    if args.image_path and args.with_vision is False:
        parser.error("--image-path conflicts with --no-with-vision")
    args.with_vision = (
        bool(args.image_path) if args.with_vision is None else args.with_vision
    )
    if args.runtime_vision_bin and not args.with_vision:
        parser.error("--runtime-vision-bin requires --with-vision or --image-path")
    args.artifacts = (
        selection.weights_path.resolve()
        if selection.weights_path
        else REPO / defaults["artifacts"]
    )
    try:
        target = TargetConfig.resolve(args.soc_model)
    except ValueError as error:
        parser.error(str(error))
    args.dsp_arch = target.dsp_arch
    args.model_settings = replace(
        settings,
        target=target,
        context=replace(settings.context, max_seq_len=args.decode_capacity),
        cache=replace(
            settings.cache,
            kv_dtype=args.kv_dtype,
            append_only=args.kv_append_only,
            state_swap=args.state_swap,
        ),
        generation=replace(
            settings.generation,
            enable_thinking=args.enable_thinking,
            system_prompt=args.system_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            ignore_eos=args.ignore_eos,
            interactive=args.interactive,
            stream=args.stream,
        ),
        schedule=replace(
            schedule,
            prefill_order=args.prefill_order,
            cache_experts=args.cache_experts,
            prefill_async_io=args.prefill_async_io,
            startup_prefetch=args.startup_prefetch,
            startup_io_workers=args.startup_io_workers,
            decode_hit_first=args.decode_hit_first,
            decode_ready_io=args.decode_ready_io,
            overlap_feedback=args.overlap_feedback,
            io_workers=args.io_workers,
            read_chunk_kib=args.read_chunk_kib,
            context_load_mode=args.decode_context_load_mode,
        ),
    )
    try:
        args.model_settings = args.model_settings.validate()
        args.prefill_order = args.model_settings.schedule.prefill_order
    except ValueError as error:
        parser.error(str(error))
    if args.embedding_table and settings.quantization.embedding != "int4_g32":
        parser.error("--embedding-table requires the int4_g32 embedding recipe")
    if args.image_path and args.prompt.count("<image>") > 1:
        parser.error("Only one image is supported")
    if args.runner == "diagnostic" and (args.temperature != 0 or args.interactive):
        parser.error("Temperature sampling and multi-turn require --runner chat")
    if args.runner == "chat" and (
        args.prefill_token_file
        or args.teacher_tokens
        or args.save_all_logits
        or args.trace_layers
    ):
        parser.error(
            "Raw token files, teacher-forcing and tensor dumps require --runner diagnostic"
        )
    if args.runner == "chat" and not args.shared_model:
        parser.error("Device chat requires separated shared weights")
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
        *([args.runtime_device_dir] if args.runtime_device_dir else []),
        *([args.runtime_vision_bin] if args.runtime_vision_bin else []),
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
