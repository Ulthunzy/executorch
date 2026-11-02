"""Evaluate heldout teacher-forced device outputs against the original model."""

import argparse
import json
import math
from pathlib import Path

from executorch.examples.qualcomm.oss_scripts.moe.quantize.execution import (
    resolve_device,
)


def reference_hidden(artifacts):
    import torch
    from executorch.examples.qualcomm.oss_scripts.moe.model.qwen35 import (
        load_model_config,
    )
    from executorch.examples.qualcomm.oss_scripts.moe.quantize.reference import (
        reference_output,
    )

    if (artifacts / "reference/layer39.pt").is_file():
        dtype = load_model_config(
            artifacts / "MODEL_CONFIG.json"
        ).quantization.calibration_dtype
        return reference_output(artifacts, 39, dtype)["all_output"].float(), dtype
    return (
        torch.load(
            artifacts / "layers/layer39/reference.pt", weights_only=True, mmap=True
        )["all_output"].float(),
        "float32",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--sequence", type=int, choices=[8, 9], default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    args.device = resolve_device(args.device)
    import numpy as np
    import torch
    import torch.nn.functional as F
    from executorch.examples.qualcomm.oss_scripts.moe.quantize.checkpoint import (
        Checkpoint,
        Head,
    )

    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    inputs = torch.load(args.artifacts / "INPUTS.pt", weights_only=True, mmap=True)
    ids = inputs["heldout_ids"][args.sequence - 8].numpy()
    prompt = np.fromfile(args.run / "tokens.i64", "<i8")
    prefix = len(prompt)
    consumed = np.fromfile(args.run / "input.i64", "<i8")
    count = len(consumed) + 1
    if not np.array_equal(prompt, ids[:prefix]) or not np.array_equal(
        consumed, ids[prefix : prefix + len(consumed)]
    ):
        raise ValueError("Device inputs are not the selected heldout sequence")
    targets = ids[prefix : prefix + count]
    if len(targets) != count:
        raise ValueError("Teacher run exceeds the heldout sequence")
    golden, dtype_name = reference_hidden(args.artifacts)
    golden = golden[args.sequence, prefix - 1 : prefix - 1 + count]
    reference_dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[dtype_name]
    prefill_hidden = np.fromfile(args.run / "prefill/hidden.f32", "<f4").reshape(
        -1, 2048
    )[-1:]
    decode_hidden = np.fromfile(args.run / "hidden.f32", "<f4").reshape(-1, 2048)
    hidden = np.concatenate((prefill_hidden, decode_hidden))
    if hidden.shape != (count, 2048):
        raise ValueError("Missing device hidden rows")
    vocab = 248320
    first = np.fromfile(args.run / "prefill/logits.f32", "<f4").reshape(-1, vocab)[-1:]
    logits = (
        np.memmap(
            args.run / "logits.f32", mode="r", dtype="<f4", shape=(count - 1, vocab)
        )
        if count > 1
        else np.empty((0, vocab), dtype="<f4")
    )
    head = (
        Head(Checkpoint(args.model))
        .eval()
        .to(device=args.device, dtype=reference_dtype)
    )
    native_loss = reference_loss = 0.0
    top1_match = 0
    for start in range(0, count, 32):
        end = min(start + 32, count)
        native = (
            np.concatenate((first, logits[: end - 1]))[start:end]
            if start == 0
            else logits[start - 1 : end - 1]
        )
        native = torch.from_numpy(np.array(native, copy=True)).to(args.device)
        reference = (
            head(golden[start:end, None].to(device=args.device, dtype=reference_dtype))
            .reshape(-1, vocab)
            .float()
        )
        target = torch.tensor(targets[start:end], device=args.device)
        if not torch.isfinite(native).all() or not torch.isfinite(reference).all():
            raise ValueError("Nonfinite logits")
        native_loss += float(F.cross_entropy(native, target, reduction="sum"))
        reference_loss += float(F.cross_entropy(reference, target, reduction="sum"))
        top1_match += int((native.argmax(-1) == reference.argmax(-1)).sum())

    def sqnr(reference, actual):
        a = reference.double()
        b = actual.double()
        signal = a.square().sum()
        noise = (a - b).square().sum()
        return float(10 * torch.log10(signal / noise))

    actual = torch.from_numpy(hidden)
    normalized_gold = F.rms_norm(golden, (2048,), eps=1e-6)
    normalized_actual = F.rms_norm(actual, (2048,), eps=1e-6)
    raw_sqnr = sqnr(golden, actual)
    normalized_sqnr = sqnr(normalized_gold, normalized_actual)
    token_curve = []
    for start in range(0, count, 128):
        end = min(start + 128, count)
        token_curve.append(
            dict(
                first_target_position=prefix + start,
                last_target_position=prefix + end - 1,
                hidden_sqnr_db=sqnr(golden[start:end], actual[start:end]),
                normalized_hidden_sqnr_db=sqnr(
                    normalized_gold[start:end], normalized_actual[start:end]
                ),
            )
        )
    report = dict(
        heldout_sequence=args.sequence,
        prefix_tokens=prefix,
        predicted_tokens=count,
        reference_ppl=math.exp(reference_loss / count),
        reference_dtype=dtype_name,
        reference_device=args.device,
        device_ppl=math.exp(native_loss / count),
        hidden_sqnr_db=raw_sqnr,
        normalized_hidden_sqnr_db=normalized_sqnr,
        top1_agreement=top1_match / count,
        token_curve=token_curve,
        reference=f"Original safetensors {dtype_name} chain on the identical heldout token IDs",
        fitted_on_heldout=False,
        device_run=str(args.run),
    )
    report["bf16_ppl" if dtype_name == "bfloat16" else "fp32_ppl"] = report[
        "reference_ppl"
    ]
    (args.run / "QUALITY.json").write_text(json.dumps(report, indent=2) + "\n")
    print("DEVICE_QUALITY", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
