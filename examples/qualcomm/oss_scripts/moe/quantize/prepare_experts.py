from pathlib import Path
import argparse
import gc
import json
import struct
import time

from executorch.examples.qualcomm.oss_scripts.moe.model.qwen35 import (
    add_model_config_argument,
    load_model_config,
)

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

from executorch.examples.qualcomm.custom_op import deltanet  # noqa: F401
from executorch.examples.qualcomm.oss_scripts.llama.model import chunk_gdn  # noqa: F401
from executorch.examples.qualcomm.oss_scripts.llama.quantize.qwen35 import (
    move_graph,
)
from executorch.examples.qualcomm.oss_scripts.moe.quantize.gptq import (
    a8,
    dequantize,
    joint_scales,
    middle,
    quantize,
)


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct W2 experts with sequential layer calibration"
    )
    add_model_config_argument(parser)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Per-layer AR128 front-qdq.pt2/constants and cos/sin tables",
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        required=True,
        help="Eight calibration and two heldout 1024-token embedding sequences",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--last-layer", type=int, default=39)
    parser.add_argument("--first-layer", type=int, default=0)
    parser.add_argument("--front-output", type=Path)
    parser.add_argument("--router-qdq", type=Path)
    parser.add_argument("--batch", type=int, choices=[1, 2, 4, 8, 16, 32], default=8)
    parser.add_argument("--joint-steps", type=int)
    parser.add_argument("--gptq-damping", type=float)
    args = parser.parse_args()
    settings = load_model_config(args.model_config)
    args.joint_steps = (
        settings.quantization.reconstruction_steps
        if args.joint_steps is None
        else args.joint_steps
    )
    args.gptq_damping = (
        settings.quantization.expert_gptq_damping
        if args.gptq_damping is None
        else args.gptq_damping
    )
    if (
        args.joint_steps < 0
        or not np.isfinite(args.gptq_damping)
        or args.gptq_damping <= 0
    ):
        parser.error(
            "Expected nonnegative reconstruction steps and positive GPTQ damping"
        )
    expert_settings = dict(
        bits=2,
        activation_bits=8,
        gu_group=1024,
        down_group=512,
        damping=args.gptq_damping,
        joint_steps=args.joint_steps,
        gptq_only=settings.quantization.expert_gptq_only,
        router_shared="w8a16_gptq" if args.router_qdq else "fp16",
    )
    if not 0 <= args.first_layer <= args.last_layer < 40:
        parser.error("--last-layer must be between 0 and 39")
    if args.front_output and args.first_layer != args.last_layer:
        parser.error("--front-output requires a single layer")
    ROOT = args.output
    REFERENCE = args.source_dir
    MODEL = args.model
    MAPPING = json.loads((MODEL / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    ROOT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    levels = torch.tensor([-7.0, -2.0, 2.0, 7.0], device="cuda")
    fit_sequences = [0, 1, 2, 4, 5, 6, 7]
    data = torch.load(args.inputs, weights_only=True, mmap=True)
    if tuple(data["calibration"].shape) != (8, 1024, 2048) or tuple(
        data["heldout"].shape
    ) != (2, 1024, 2048):
        raise ValueError(
            "Expected eight calibration and two heldout FP32 embedding sequences"
        )
    hidden = torch.cat((data["calibration"], data["heldout"])).cuda()
    if args.first_layer:
        hidden = torch.load(
            ROOT / f"sequential/layer{args.first_layer-1}.pt",
            weights_only=True,
            mmap=True,
        )["hidden"].cuda()
    cos = torch.from_numpy(
        np.fromfile(REFERENCE / "cos.f32", "<f4").reshape(-1, 32)
    ).cuda()
    sin = torch.from_numpy(
        np.fromfile(REFERENCE / "sin.f32", "<f4").reshape(-1, 32)
    ).cuda()

    def pack(codes, scales):
        n, k = codes.shape
        indices = torch.bucketize(
            codes.float().contiguous(), (levels[:-1] + levels[1:]) / 2
        )
        assert torch.equal(levels[indices], codes.float())
        c = (
            indices.reshape(n // 32, 32, k // 128, 4, 4, 2, 4)
            .permute(0, 2, 3, 4, 1, 6, 5)
            .contiguous()
            .to(torch.uint8)
        )
        pairs = ((c[..., 0] & 3) | ((c[..., 1] & 3) << 2)).reshape(-1, 2)
        packed = (pairs[:, 0] | (pairs[:, 1] << 4)).cpu().numpy().tobytes()
        sums = codes.reshape(n // 32, 32, k // 128, 128).sum(-1).permute(0, 2, 1)
        grids = scales.reshape(n // 32, 32, -1).transpose(1, 2).contiguous()
        return (
            packed,
            grids.cpu().numpy().astype("<f4").tobytes(),
            sums.cpu().numpy().astype("<i2").tobytes(),
        )

    def attention(layer, x):
        module = move_graph(
            torch.export.load(REFERENCE / f"layer{layer}/front-qdq.pt2").module(),
            "cuda",
        )
        normalized, residual = torch.empty_like(x), torch.empty_like(x)
        for sequence in range(len(x)):
            if layer % 4 != 3:
                first = torch.zeros(1, 8192, 4, device="cuda")
                second = torch.zeros(1, 32, 128, 128, device="cuda")
            else:
                first = torch.zeros(1, 2, 256, 896, device="cuda")
                second = torch.zeros(1, 2, 896, 256, device="cuda")
            for start in range(0, 1024, 128):
                inputs = x[sequence : sequence + 1, start : start + 128].contiguous()
                if layer % 4 != 3:
                    result = module(inputs, first, second)
                    first, second = result[2:]
                else:
                    mask = torch.full((1, 1, 128, 1024), -255.0, device="cuda")
                    if start:
                        mask[..., 896 - start : 896] = 0
                    mask[..., 896:] = torch.triu(
                        torch.full((128, 128), -255.0, device="cuda"),
                        diagonal=1,
                    )
                    result = module(
                        inputs,
                        cos[start : start + 128].contiguous(),
                        sin[start : start + 128].contiguous(),
                        mask,
                        first,
                        second,
                    )
                    first = torch.cat((first[..., 128:], result[2]), -1)
                    second = torch.cat((second[:, :, 128:], result[3]), 2)
                normalized[sequence, start : start + 128] = result[0][0]
                residual[sequence, start : start + 128] = result[1][0]
        del module
        return normalized, residual

    def samples(x, ids, expert, maximum, support):
        selected = x[(ids == expert).any(-1)]
        count = len(selected)
        if count > maximum:
            selected = selected[
                torch.linspace(0, count - 1, maximum, device="cuda").round().long()
            ]
        global_rows = x[
            torch.linspace(0, len(x) - 1, support, device="cuda").round().long()
        ]
        if not count:
            return (
                global_rows,
                torch.full((support,), 1 / support, device="cuda"),
                count,
            )
        weights = torch.cat(
            (
                torch.full((len(selected),), 0.65 / len(selected), device="cuda"),
                torch.full((support,), 0.35 / support, device="cuda"),
            )
        )
        return torch.cat((selected, global_rows)), weights, count

    for layer in range(args.first_layer, args.last_layer + 1):
        checkpoint = ROOT / f"sequential/layer{layer}.pt"
        directory = ROOT / f"banks/layer{layer}"
        if checkpoint.is_file() and (directory / "BANK.json").is_file():
            if (
                json.loads((directory / "BANK.json").read_text()).get(
                    "quantization_config"
                )
                != expert_settings
            ):
                raise ValueError(
                    f"Layer {layer}: existing bank has different or unrecorded quantization settings; use a new output directory"
                )
            hidden = torch.load(checkpoint, weights_only=True, mmap=True)[
                "hidden"
            ].cuda()
            print("REUSED", layer, flush=True)
            continue
        started = time.monotonic()
        if args.front_output:
            front = torch.load(args.front_output, weights_only=True, mmap=True)
            norm, residual = front["norm"].cuda(), front["residual"].cuda()
        else:
            norm, residual = attention(layer, hidden)
        print(
            "FRONT_READY",
            layer,
            round(time.monotonic() - started, 2),
            flush=True,
        )
        constants = np.load(REFERENCE / f"layer{layer}/moe-constants.npz")
        weights = {
            k: torch.from_numpy(constants[k]).cuda().half() for k in constants.files
        }
        x = norm.reshape(-1, 2048)
        score, ids = (
            F.linear(x.half(), weights["gate_weight"]).float().softmax(-1).topk(8, -1)
        )
        score = score / score.sum(-1, keepdim=True)
        shared = F.linear(
            F.silu(F.linear(x.half(), weights["shared_expert_gate_proj_weight"]))
            * F.linear(x.half(), weights["shared_expert_up_proj_weight"]),
            weights["shared_expert_down_proj_weight"],
        )
        shared = (
            shared * F.linear(x.half(), weights["shared_expert_gate_weight"]).sigmoid()
        ).float()
        if args.router_qdq:
            from executorch.examples.qualcomm.oss_scripts.llama.quantize.generation import (
                remove_export_checks,
            )

            router = move_graph(
                torch.export.load(
                    args.router_qdq / f"layer{layer}/router-ar128-qdq.pt2"
                ).module(),
                "cuda",
            )
            remove_export_checks(router)
            ids, score, shared = router(norm)
            ids = ids.reshape(-1, 8).long()
            score = score.reshape(-1, 8)
            shared = shared.reshape(-1, 2048)
            del router
        fit_x = norm[fit_sequences].reshape(-1, 2048)
        fit_ids = ids.reshape(10, 1024, 8)[fit_sequences].reshape(-1, 8)
        val_x = norm[3]
        val_ids = ids.reshape(10, 1024, 8)[3]
        routed = torch.zeros_like(x)
        names = [
            f"model.language_model.layers.{layer}.mlp.experts." + n
            for n in ("gate_up_proj", "down_proj")
        ]
        directory.mkdir(parents=True, exist_ok=True)
        offsets, lengths, details = [], [], []
        temporary = directory / "experts.n2.partial"
        with safe_open(str(MODEL / MAPPING[names[0]]), framework="pt") as gf, safe_open(
            str(MODEL / MAPPING[names[1]]), framework="pt"
        ) as df, temporary.open("wb") as file:
            file.write(bytes(4096))
            for start in range(0, 256, args.batch):
                experts = list(range(start, min(start + args.batch, 256)))
                wgu = torch.stack(
                    [
                        gf.get_slice(names[0])[e : e + 1][0].cuda().float()
                        for e in experts
                    ]
                )
                wd = torch.stack(
                    [
                        df.get_slice(names[1])[e : e + 1][0].cuda().float()
                        for e in experts
                    ]
                )
                train_rows = [samples(fit_x, fit_ids, e, 256, 128) for e in experts]
                val_rows = [samples(val_x, val_ids, e, 64, 64) for e in experts]
                width = max(len(v[0]) for v in train_rows)
                val_width = max(len(v[0]) for v in val_rows)
                cal = torch.stack(
                    [F.pad(v[0], (0, 0, 0, width - len(v[0]))) for v in train_rows]
                )
                row_weights = torch.stack(
                    [F.pad(v[1], (0, width - len(v[1]))) for v in train_rows]
                )
                validation = torch.stack(
                    [F.pad(v[0], (0, 0, 0, val_width - len(v[0]))) for v in val_rows]
                )
                g, u = cal.bmm(wgu.transpose(1, 2)).chunk(2, -1)
                target = (F.silu(g) * u).bmm(wd.transpose(1, 2))
                vg, vu = validation.bmm(wgu.transpose(1, 2)).chunk(2, -1)
                val_target = (F.silu(vg) * vu).bmm(wd.transpose(1, 2))
                actual_x = a8(cal, 1024)
                weighted_x = actual_x * row_weights.sqrt()[:, :, None]
                best = torch.full((len(experts),), float("inf"), device="cuda")
                chosen = [
                    torch.empty_like(wgu, dtype=torch.int8),
                    torch.empty_like(wgu[:, :, :2]),
                    torch.empty_like(wd, dtype=torch.int8),
                    torch.empty_like(wd[:, :, :1]),
                ]
                method = torch.zeros(len(experts), dtype=torch.int64, device="cuda")
                methods = (
                    [True] if settings.quantization.expert_gptq_only else [False, True]
                )
                for gu_gptq in methods:
                    qgu, sgu = quantize(
                        wgu,
                        weighted_x,
                        1024,
                        levels,
                        gu_gptq,
                        damping=args.gptq_damping,
                    )
                    gu = dequantize(qgu, sgu, 1024)
                    mid = middle(actual_x, gu)
                    for down_gptq in methods:
                        qd, sd = quantize(
                            wd,
                            mid * row_weights.sqrt()[:, :, None],
                            512,
                            levels,
                            down_gptq,
                            damping=args.gptq_damping,
                        )
                        down = dequantize(qd, sd, 512)
                        prediction = middle(a8(validation, 1024), gu).bmm(
                            down.transpose(1, 2)
                        )
                        error = (prediction - val_target).square().mean((1, 2))
                        improve = error < best
                        chosen = [
                            torch.where(improve[:, None, None], value, previous)
                            for value, previous in zip((qgu, sgu, qd, sd), chosen)
                        ]
                        method = torch.where(
                            improve, int(gu_gptq) * 2 + int(down_gptq), method
                        )
                        best = torch.minimum(best, error)
                qgu, sgu, qd, sd = chosen
                sgu, sd = joint_scales(
                    qgu,
                    sgu,
                    qd,
                    sd,
                    cal,
                    target,
                    validation=(validation, val_target),
                    weights=row_weights,
                    steps=args.joint_steps,
                )
                gu, down = dequantize(qgu, sgu, 1024), dequantize(qd, sd, 512)
                for b, expert in enumerate(experts):
                    payload = [*pack(qgu[b], sgu[b]), *pack(qd[b], sd[b])]
                    position = file.tell()
                    cursor, parts = 128, []
                    for part in payload:
                        parts.append(cursor)
                        cursor += len(part)
                    file.write(
                        struct.pack(
                            "<12I",
                            0x4E324D45,
                            512,
                            512,
                            cursor,
                            *parts,
                            expert,
                            layer,
                        ).ljust(128, b"\0")
                    )
                    for part in payload:
                        file.write(part)
                    file.write(bytes(-file.tell() % 128))
                    offsets.append(position)
                    lengths.append(cursor)
                    token, rank = torch.where(ids == expert)
                    predicted = middle(a8(x[token][None], 1024), gu[b : b + 1]).bmm(
                        down[b : b + 1].transpose(1, 2)
                    )[0]
                    routed.index_add_(0, token, predicted * score[token, rank, None])
                    details.append(
                        dict(
                            expert=expert,
                            fit_routed_tokens=train_rows[b][2],
                            selection_routed_tokens=val_rows[b][2],
                            selected_method=int(method[b]),
                        )
                    )
                if start % 64 == 0:
                    print(
                        "PROGRESS",
                        layer,
                        start,
                        round(time.monotonic() - started, 2),
                        flush=True,
                    )
            file.write(bytes(-file.tell() % 4096))
            file.seek(0)
            file.write(struct.pack("<8I", 0x4E32424B, 2, 256, 2048, 512, 1024, 512, 7))
            file.write(np.asarray(offsets, dtype="<u8").tobytes())
        bank = directory / "experts.n2"
        temporary.replace(bank)
        index = ROOT / f"indices/layer{layer}"
        index.mkdir(parents=True, exist_ok=True)
        with bank.open("rb") as file:
            (index / "bank-index.bin").write_bytes(
                file.read(4096) + np.asarray(lengths, dtype="<u8").tobytes()
            )
        hidden = residual + (routed + shared).reshape_as(residual)
        reference = torch.load(
            REFERENCE / f"layer{layer}/reference.pt",
            weights_only=True,
            mmap=True,
        )["test"]["output"].cuda()
        actual = hidden[8:9]
        normalized_gold = (
            reference / reference.square().mean(-1, keepdim=True).add(1e-6).sqrt()
        )
        normalized_actual = (
            actual / actual.square().mean(-1, keepdim=True).add(1e-6).sqrt()
        )
        sqnr = float(
            10
            * torch.log10(
                reference.double().square().sum()
                / (reference.double() - actual.double()).square().sum()
            )
        )
        normalized_sqnr = float(
            10
            * torch.log10(
                normalized_gold.double().square().sum()
                / (normalized_gold.double() - normalized_actual.double()).square().sum()
            )
        )
        report = dict(
            quantization_config=expert_settings,
            layer=layer,
            codebook=[-7, -2, 2, 7],
            bits=2,
            gu_group=1024,
            down_group=512,
            sequential=True,
            calibration_tokens=8192,
            fitting_tokens=7168,
            selection_tokens=1024,
            heldout_used_for_fitting=False,
            full_expert_scale_reconstruction=True,
            heldout_hidden_sqnr_db=sqnr,
            heldout_normalized_hidden_sqnr_db=normalized_sqnr,
            bytes=bank.stat().st_size,
            details=details,
            seconds=time.monotonic() - started,
            cuda_peak_mib=torch.cuda.max_memory_allocated() / 2**20,
            expert_batch_size=args.batch,
        )
        checkpoint.parent.mkdir(exist_ok=True)
        torch.save(dict(hidden=hidden.cpu()), checkpoint)
        (directory / "BANK.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            "LAYER_READY",
            layer,
            "SQNR",
            sqnr,
            "NORMALIZED",
            normalized_sqnr,
            "SECONDS",
            report["seconds"],
            flush=True,
        )
        del (
            norm,
            residual,
            routed,
            shared,
            x,
            fit_x,
            fit_ids,
            val_x,
            val_ids,
            reference,
            weights,
        )
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
