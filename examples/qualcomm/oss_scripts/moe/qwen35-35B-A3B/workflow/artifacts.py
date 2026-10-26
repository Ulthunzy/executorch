import json
import struct
import hashlib

from .plans import check_plan
from .runtime_assets import runtime_assets, validate_rope


def decode_layer_dir(args, layer):
    directory = (
        f"assets/context{args.decode_capacity}-fused"
        if layer % 4 == 3
        else "assets/gdn-fused"
    )
    return args.artifacts / directory / f"layer{layer}"


def check_io_order(path, inputs, outputs):
    raw = path.read_bytes()
    if len(raw) != 4 * (2 + inputs + outputs):
        raise ValueError(f"{path}: malformed graph IO order")
    order = struct.unpack("<" + "I" * (len(raw) // 4), raw)
    if (
        order[:2] != (inputs, outputs)
        or sorted(order[2 : 2 + inputs]) != list(range(inputs))
        or sorted(order[2 + inputs :]) != list(range(outputs))
    ):
        raise ValueError(f"{path}: incompatible graph IO order")


def check_artifacts(args):
    if (
        args.model_settings.quantization.embedding == "int4_g32"
        and args.embedding_table is None
    ):
        raise ValueError(
            "The int4_g32 recipe requires --embedding-table; select bf16 in --model-config to use the BF16 table"
        )
    for phase, path in (
        ("prefill", args.local_prefill_pte),
        ("decode", args.local_decode_pte),
    ):
        if path is not None:
            check_plan(path, args, phase)
    if args.decode_context_load_mode == "auto":
        manifest = args.artifacts / "ASSETS.json"
        args.decode_context_load_mode = (
            json.loads(manifest.read_text()).get("decode_context_load_mode", "async")
            if manifest.is_file()
            else "async"
        )
    if args.decode_context_load_mode not in ("async", "serial"):
        raise ValueError("Unsupported decode context loading mode in ASSETS.json")
    if args.shared_model and args.decode_context_load_mode == "serial":
        raise ValueError(
            "Shared contexts require batched initialization; serial loading is unsupported"
        )
    target_file = args.build_dir / "TARGET.json"
    build_metadata = (
        json.loads(target_file.read_text()) if target_file.is_file() else {}
    )
    args.model_settings.target.check_manifest(build_metadata)
    if args.runner == "chat" and build_metadata.get("chat_controls") != 1:
        raise ValueError(
            "Rebuild the native chat runner with this version of build.sh (the unified export builds it automatically)"
        )
    required = [
        args.build_dir / "qwen35_moe_runner",
        args.build_dir / "prefill_handoff_runner",
        args.build_dir / "libmoe_decode.so",
        args.build_dir / "native-prefill/libinteger_rpc_skel.so",
        args.build_dir / "native-decode/libinteger_rpc_skel.so",
        args.build_dir
        / f"op-package/hexagon-{args.dsp_arch}/libQnnDeltaNetOpPackage_HTP.so",
        args.model / "config.json",
        args.model / "generation_config.json",
        args.model / "tokenizer.json",
        args.artifacts / "cold-experts.i32",
        *[
            runtime_assets(args) / name
            for name in ["cache-encoding.bin", "cos.f32", "sin.f32"]
        ],
    ]
    if args.shared_model:
        required.extend(
            [
                args.build_dir / "libqwen35_shared.so",
                args.shared_model / "head/graph.qnn",
                args.shared_model / "head/weights.bin",
                args.shared_model / "head/WEIGHTS.json",
            ]
        )
        for group in range(10):
            required.extend(
                args.shared_model / f"group{group}" / name
                for name in ("graph.qnn", "weights.bin", "GRAPHS.json")
            )
            for layer in range(group * 4, group * 4 + 4):
                required.extend(
                    args.shared_model / f"group{group}" / f"layer{layer}_{method}.io"
                    for method in ("prefill_attention", "prefill_router", "decode")
                )
    if args.embedding_table:
        required.append(args.embedding_table)
    else:
        required.append(args.artifacts / "embedding.bf16")
    for layer in range(40):
        required.extend(
            [
                args.artifacts / f"banks/layer{layer}/BANK.json",
                args.artifacts / f"indices/layer{layer}/bank-index.bin",
            ]
        )
        if args.deploy_weights:
            required.append(args.artifacts / f"banks/layer{layer}/experts.n2")
        if not args.shared_model:
            directory = decode_layer_dir(args, layer)
            required.extend([directory / "attention.bin", directory / "io.bin"])
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing artifacts/build outputs:\n" + "\n".join(missing)
        )
    validate_rope(runtime_assets(args), args.decode_capacity)
    if args.shared_model:
        covered = set()
        for group in range(10):
            manifest = json.loads(
                (args.shared_model / f"group{group}/GRAPHS.json").read_text()
            )
            args.model_settings.target.check_manifest(manifest)
            if args.runner == "chat" and not manifest.get("prefill_valid_length"):
                raise ValueError(
                    "Re-export the shared graphs for native chat: AR128 valid-length input is required"
                )
            expected = list(range(group * 4, group * 4 + 4))
            if (
                manifest["layers"] != expected
                or manifest["capacity"] != args.decode_capacity
                or not manifest["weight_sharing"]
                or manifest.get("prefill_core") != "chunked_mixed_fp16"
                or manifest["router"] != args.model_settings.quantization.router_shared
            ):
                raise ValueError(f"Shared group {group}: layer/capacity mismatch")
            if (
                "compile_settings" in manifest
                or args.model_settings.quantization.attention == "gptq"
            ) and manifest.get(
                "compile_settings"
            ) != args.model_settings.compile_settings():
                raise ValueError(
                    f"Shared group {group}: quantization/KV/context differs from --model-config"
                )
            for layer in expected:
                inputs = 6 if layer % 4 == 3 else 3
                for method, counts in (
                    (
                        "prefill_attention",
                        (
                            inputs
                            + (
                                manifest.get("prefill_valid_length", False)
                                and layer % 4 != 3
                            ),
                            4,
                        ),
                    ),
                    ("prefill_router", (1, 3)),
                    ("decode", (inputs, 7)),
                ):
                    path = args.shared_model / f"group{group}/layer{layer}_{method}.io"
                    check_io_order(path, *counts)
            covered.update(manifest["layers"])
        if covered != set(range(40)):
            raise ValueError("Shared model must contain all 40 layers")
        if args.model_settings.quantization.head == "w4a16_gptq":
            head = json.loads((args.artifacts / "head/HEAD.json").read_text())
            args.model_settings.target.check_manifest(head)
            provenance = json.loads(
                (args.shared_model / "head/WEIGHTS.json").read_text()
            )
            with (args.artifacts / "head/head.bin").open("rb") as file:
                digest = hashlib.file_digest(file, "sha256").hexdigest()
            if (
                head["quantization"]["method"] != "all_projection_gptq"
                or digest != provenance["source_sha256"]
            ):
                raise ValueError("Shared head does not match the calibrated GPTQ head")
    if args.embedding_table:
        with args.embedding_table.open("rb") as handle:
            header = handle.read(24)
        if (
            header != struct.pack("<8s4I", b"Q35EMB4", 1, 248320, 2048, 32)
            or args.embedding_table.stat().st_size != 317849664
        ):
            raise ValueError("Expected group-32 INT4 embedding for 248320 x 2048")
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
        if args.model_settings.quantization.expert_gptq_only and (
            not bank.get("quantization_config", {}).get("gptq_only")
            or len(bank.get("details", [])) != 256
            or any(d["selected_method"] != 3 for d in bank["details"])
        ):
            raise ValueError(
                f"Layer {layer}: the selected recipe requires GPTQ for every GU/down matrix"
            )
        if (
            bank["codebook"] != [-7, -2, 2, 7]
            or not bank["sequential"]
            or (bank.get("bits"), bank.get("gu_group"), bank.get("down_group"))
            != (2, 1024, 512)
        ):
            raise ValueError(
                f"Layer {layer} requires sequentially reconstructed W2 weights"
            )
        with (args.artifacts / f"indices/layer{layer}/bank-index.bin").open(
            "rb"
        ) as handle:
            if struct.unpack("<8I", handle.read(32)) != (
                0x4E32424B,
                2,
                256,
                2048,
                512,
                1024,
                512,
                7,
            ):
                raise ValueError(f"Layer {layer}: incompatible expert index version")
        if args.shared_model:
            continue
        directory = decode_layer_dir(args, layer)
        metadata = directory / "IO.json"
        if metadata.is_file():
            info = json.loads(metadata.read_text())
            if (
                info.get("router_quantization")
                != args.model_settings.quantization.router_shared
            ):
                raise ValueError(
                    f"Layer {layer}: Router/shared recipe differs from the compiled graph"
                )
            if layer % 4 == 3 and info.get("context") != args.decode_capacity:
                raise ValueError(
                    f"Layer {layer}: compiled context differs from --max-seq-len"
                )
            if "compile_settings" in info:
                expected = args.model_settings.compile_settings()
                if any(
                    info["compile_settings"][key] != expected[key]
                    for key in ("quantization", "cache")
                ):
                    raise ValueError(
                        f"Layer {layer}: compiled quantization/KV interface mismatch"
                    )
        inputs = 6 if layer % 4 == 3 else 3
        check_io_order(directory / "io.bin", inputs, 7)
