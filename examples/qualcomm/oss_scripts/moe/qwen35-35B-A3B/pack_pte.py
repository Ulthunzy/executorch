import argparse
import hashlib
import json
from pathlib import Path
import struct

from dataclasses import replace
from model_config import (
    add_model_config_argument,
    load_model_config,
    shared_model_signature,
)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Pack calibrated per-layer QNN binaries into a complete 40-layer MoE PTE plan"
    )
    add_model_config_argument(parser)
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress per-stage packing output"
    )
    parser.add_argument("--max-seq-len", type=int)
    parser.add_argument("--prefill-order", choices=["layer_major", "chunk_major"])
    parser.add_argument("--phase", choices=["prefill", "decode"], required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--cache-encoding", type=Path, help="Required for decode")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--prefill-tokens",
        type=int,
        choices=range(128, 1025, 128),
        default=128,
        help="Host plan sequence length; QNN and Native kernels remain AR128",
    )
    parser.add_argument(
        "--shared-model", type=Path, help="Separated four-layer context groups"
    )
    args = parser.parse_args(argv)
    settings = load_model_config(args.model_config)
    settings = replace(
        settings,
        context=replace(
            settings.context,
            max_seq_len=args.max_seq_len or settings.context.max_seq_len,
        ),
        schedule=replace(
            settings.schedule,
            prefill_order=args.prefill_order or settings.schedule.prefill_order,
        ),
    ).validate()
    if args.phase == "decode" and args.cache_encoding is None:
        parser.error("Decode packing requires --cache-encoding")
    if args.shared_model:
        for group in range(10):
            manifest = json.loads(
                (args.shared_model / f"group{group}/GRAPHS.json").read_text()
            )
            if (
                manifest["capacity"] != settings.context.max_seq_len
                or manifest["router"] != settings.quantization.router_shared
            ):
                parser.error(
                    f"Shared group {group} does not match --model-config/--max-seq-len"
                )
            if (
                "compile_settings" in manifest
                or settings.quantization.attention == "gptq"
            ) and manifest.get("compile_settings") != settings.compile_settings():
                parser.error(
                    f"Shared group {group}: compiled quantization/KV/context mismatch"
                )
    from executorch.exir._serialize._program import (
        PTEFile,
        serialize_pte_binary,
    )
    from executorch.exir.scalar_type import ScalarType
    from executorch.exir.schema import (
        AllocationDetails,
        BackendDelegate,
        BackendDelegateDataReference,
        BackendDelegateInlineData,
        Buffer,
        Chain,
        ContainerMetadata,
        DataLocation,
        DelegateCall,
        EValue,
        ExecutionPlan,
        Instruction,
        Program,
        SubsegmentOffsets,
        Tensor,
        TensorShapeDynamism,
    )

    decode = args.phase == "decode"
    ar_len = (
        1
        if decode
        else (
            args.prefill_tokens
            if settings.schedule.prefill_order == "layer_major"
            else settings.context.prefill_ar_len
        )
    )
    hidden_bytes = ar_len * 2048 * 4
    route_bytes = max(4096, ar_len * 8 * 4)
    offsets = dict(
        chain=0,
        norm=hidden_bytes,
        residual=2 * hidden_bytes,
        shared=3 * hidden_bytes,
        ids=4 * hidden_bytes,
        scores=4 * hidden_bytes + route_bytes,
        logits=4 * hidden_bytes + 2 * route_bytes,
    )
    values, instructions, delegates, blobs, records = [], [], [], [], []

    def tensor(shape, slot=None, dtype=ScalarType.FLOAT):
        allocation = AllocationDetails(1, offsets[slot], 0) if slot else None
        value = Tensor(
            dtype,
            0,
            shape,
            list(range(len(shape))),
            False,
            0,
            0,
            allocation,
            TensorShapeDynamism.STATIC,
        )
        values.append(EValue(value))
        return len(values) - 1

    def stage(kind, layer, arguments, primary=None, auxiliary=None):
        if args.shared_model:
            primary = args.cache_encoding if decode and kind == 0 else None
            auxiliary = (
                (args.shared_model / f"group{layer // 4}" / f"layer{layer}_decode.io")
                if decode and kind == 1
                else None
            )
        data = primary.read_bytes() if primary else b""
        aux = auxiliary.read_bytes() if auxiliary else b""
        offset = (128 + len(data) + 127) & ~127
        magic = b"Q36DECO1" if decode else b"Q36PREF1"
        header = struct.pack(
            "<8sIIIIQQQ",
            magic,
            3 if args.shared_model else (2 if ar_len > 128 else 1),
            kind,
            layer,
            ar_len if args.shared_model or ar_len > 128 else 0,
            len(data),
            offset,
            len(aux),
        )
        blob = header.ljust(128, b"\0") + data
        if aux:
            blob = blob.ljust(offset, b"\0") + aux
        index = len(delegates)
        blobs.append(BackendDelegateInlineData(blob))
        backend = "Qwen36DecodeStage" if decode else "Qwen36PrefillStage"
        delegates.append(
            BackendDelegate(
                backend,
                BackendDelegateDataReference(DataLocation.INLINE, index),
                [],
            )
        )
        instructions.append(Instruction(DelegateCall(index, arguments)))
        records.append(
            dict(
                kind=kind,
                layer=layer,
                binary=str(primary) if primary else None,
                sha256=hashlib.sha256(data).hexdigest() if data else None,
            )
        )
        if not args.quiet:
            print("STAGE", index, kind, layer, len(blob), flush=True)

    tokens = tensor([1, ar_len], dtype=ScalarType.LONG)
    position = tensor([1], dtype=ScalarType.LONG) if decode else None
    hidden = tensor([1, ar_len, 2048], "chain")
    stage(
        0,
        0,
        [tokens, position, hidden] if decode else [tokens, hidden],
        args.cache_encoding if decode else None,
    )
    for layer in range(40):
        norm, residual = tensor([1, ar_len, 2048], "norm"), tensor(
            [1, ar_len, 2048], "residual"
        )
        directory = args.source_dir / f"layer{layer}"
        if decode:
            attention, metadata = (
                directory / "attention.bin",
                directory / "io.bin",
            )
        else:
            attention = (
                args.source_dir / f"cold/layer{layer}" if layer % 4 == 3 else directory
            ) / "forward_0.bin"
            metadata = directory / "forward_0.bin" if layer % 4 == 3 else None
        stage(1, layer, [hidden, norm, residual], attention, metadata)
        ids = tensor([1, ar_len, 8], "ids", ScalarType.INT)
        scores, shared = tensor([1, ar_len, 8], "scores"), tensor(
            [1, ar_len, 2048], "shared"
        )
        stage(2, layer, [norm, ids, scores, shared], directory / "moe-front.bin")
        next_hidden = tensor([1, ar_len, 2048], "chain")
        stage(3, layer, [norm, ids, scores, shared, residual, next_hidden])
        hidden = next_hidden
    logits = tensor([1, 1, 248320], "logits")
    stage(4, 0, [hidden, logits], args.head)
    assert len(instructions) == 122
    plan = ExecutionPlan(
        "decode_forward" if decode else "prefill_forward",
        ContainerMetadata("", ""),
        values,
        [tokens, position] if decode else [tokens],
        [logits, hidden],
        [Chain([], [], instructions, None)],
        [],
        delegates,
        [0, offsets["logits"] + 248320 * 4],
    )
    program = Program(0, [plan], [Buffer(b"")], blobs, [], SubsegmentOffsets(0, []))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as file:
        serialize_pte_binary(
            PTEFile(program),
            extract_delegate_segments=True,
            delegate_alignment=128,
        ).write_to_file(file)
    args.output.with_suffix(".json").write_text(
        json.dumps(
            dict(
                phase=args.phase,
                ar_len=1 if decode else 128,
                plan_tokens=ar_len,
                requested_prefill_tokens=args.prefill_tokens,
                prefill_order=settings.schedule.prefill_order,
                compile_settings=settings.compile_settings(),
                shared_signature=(
                    shared_model_signature(args.shared_model)
                    if args.shared_model
                    else None
                ),
                cache_encoding_sha256=(
                    hashlib.sha256(args.cache_encoding.read_bytes()).hexdigest()
                    if decode
                    else None
                ),
                layers=40,
                weight_mode=(
                    "external_shared" if args.shared_model else "embedded_independent"
                ),
                shared_model=str(args.shared_model) if args.shared_model else None,
                stages=records,
            ),
            indent=2,
        )
        + "\n"
    )
    print("EXPORTED", args.output, args.output.stat().st_size, flush=True)


if __name__ == "__main__":
    main()
