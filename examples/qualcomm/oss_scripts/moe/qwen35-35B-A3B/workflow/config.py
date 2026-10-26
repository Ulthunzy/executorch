import argparse
from dataclasses import replace
import json
from pathlib import Path, PurePosixPath
import re

from model_config import add_model_config_argument, load_model_config

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[4]


def parse_args(argv=None):
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/sm8975.json"
    )
    add_model_config_argument(config_parser)
    selection, _ = config_parser.parse_known_args(argv)
    defaults = json.loads(selection.config.read_text())
    try:
        settings = load_model_config(selection.model_config)
    except (ValueError, OSError) as error:
        config_parser.error(str(error))
    schedule = settings.schedule
    parser = argparse.ArgumentParser(
        parents=[config_parser],
        description="Qwen3.5-35B-A3B streaming MoE chat on SM8975/V85",
    )
    parser.add_argument("--device", default=defaults["device"])
    parser.add_argument("--model", type=Path, default=Path(defaults["model"]))
    parser.add_argument("--artifacts", type=Path, default=REPO / defaults["artifacts"])
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
        choices=range(128, settings.context.max_prefill_tokens + 1, 128),
    )
    parser.add_argument("--image-path", type=Path)
    parser.add_argument(
        "--vision-artifacts",
        type=Path,
        default=REPO
        / defaults.get("vision_artifacts", defaults["artifacts"] + "/vision"),
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
        help="Local packed plan; its JSON sidecar is validated and deployed",
    )
    parser.add_argument(
        "--local-decode-pte",
        type=Path,
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
        choices=[32, 64],
        default=schedule.cache_experts,
        help="Resident LRU expert slots per layer, shared by prefill and decode",
    )
    parser.add_argument("--prefill-token-file", type=Path)
    parser.add_argument("--teacher-tokens", type=Path)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--save-all-logits", action="store_true")
    parser.add_argument("--trace-layers", action="store_true")
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument(
        "--io-workers", type=int, choices=[4, 8, 16], default=schedule.io_workers
    )
    parser.add_argument(
        "--read-chunk-kib",
        type=int,
        choices=[64, 128, 256, 512, 1024],
        default=schedule.read_chunk_kib,
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
    args.model_settings = replace(
        settings,
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
        args.model_settings.validate()
    except ValueError as error:
        parser.error(str(error))
    if args.embedding_table and settings.quantization.embedding != "int4_g32":
        parser.error("--embedding-table requires the int4_g32 embedding recipe")
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
        *([args.runtime_device_dir] if args.runtime_device_dir else []),
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
