import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(
        description="Group-32 INT4 embedding with the diagonal one-hot GPTQ objective"
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--calibration-inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    shape = (248320, 2048)
    if args.source.resolve() == args.output.resolve():
        parser.error("Source and output must differ")
    if args.source.stat().st_size != np.prod(shape) * 2:
        parser.error("Expected a 248320 x 2048 BF16 table")
    data = torch.load(args.calibration_inputs, weights_only=True, mmap=True)
    tokens = data["tokens"].numpy()
    if tokens.shape != (8, 1024) or tokens.min() < 0 or tokens.max() >= shape[0]:
        parser.error("Expected eight 1024-token calibration sequences")
    counts = np.bincount(tokens.reshape(-1), minlength=shape[0])
    heldout = data["heldout_ids"].numpy()
    heldout_counts = np.bincount(heldout.reshape(-1), minlength=shape[0])
    source = np.memmap(args.source, mode="r", dtype="<u2", shape=shape)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    signal = error = train_signal = train_error = test_signal = test_error = 0.0
    with args.output.open("wb") as handle:
        handle.write(struct.pack("<8s4I", b"Q35EMB4", 1, *shape, 32).ljust(64, b"\0"))
        for start in range(0, shape[0], 1024):
            values = (source[start : start + 1024].astype(np.uint32) << 16).view(
                np.float32
            )
            values = values.reshape(-1, 64, 32)
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite embedding weights")
            bias = values.min(-1).astype("<f2")
            scale = ((values.max(-1) - bias.astype(np.float32)) / 15).astype("<f2")
            scale = np.maximum(scale, np.finfo(np.float16).tiny)
            # For lookup inputs H=diag(token frequencies). There are no
            # off-diagonal GPTQ corrections; the optimum on each fixed grid is RTN.
            codes = np.clip(
                np.rint((values - bias[..., None]) / scale[..., None]), 0, 15
            ).astype(np.uint8)
            restored = codes * scale[..., None].astype(np.float32) + bias[
                ..., None
            ].astype(np.float32)
            row_signal = np.square(values, dtype=np.float64).sum((1, 2))
            row_error = np.square(values - restored, dtype=np.float64).sum((1, 2))
            signal += row_signal.sum()
            error += row_error.sum()
            frequency = counts[start : start + len(values)]
            train_signal += row_signal @ frequency
            train_error += row_error @ frequency
            frequency = heldout_counts[start : start + len(values)]
            test_signal += row_signal @ frequency
            test_error += row_error @ frequency
            blocks = np.empty((*values.shape[:-1], 20), dtype=np.uint8)
            blocks[..., :16] = codes[..., ::2] | (codes[..., 1::2] << 4)
            blocks[..., 16:18] = scale[..., None].view(np.uint8)
            blocks[..., 18:20] = bias[..., None].view(np.uint8)
            handle.write(blocks.tobytes())
    result = dict(
        method="diagonal_one_hot_gptq_equivalent_to_rtn",
        hessian="diag(calibration_token_counts / 8192)",
        off_diagonal_compensation=False,
        unseen_rows="same min/max fixed grid and nearest rounding",
        calibration_tokens=int(counts.sum()),
        calibration_unique_tokens=int(np.count_nonzero(counts)),
        calibration_sha256=hashlib.sha256(tokens.tobytes()).hexdigest(),
        heldout_used_for_fitting=False,
        group_size=32,
        bits=4,
        shape=shape,
        source_bytes=args.source.stat().st_size,
        packed_bytes=args.output.stat().st_size,
        weight_sqnr_db=float(10 * np.log10(signal / error)),
        calibration_lookup_sqnr_db=float(10 * np.log10(train_signal / train_error)),
        heldout_lookup_sqnr_db=float(10 * np.log10(test_signal / test_error)),
    )
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
