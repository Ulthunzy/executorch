#!/usr/bin/env python3
# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export the isolated core with one encoding for initial and final state."""

import argparse
import json
from pathlib import Path
import sys


def main():
    early_parser = argparse.ArgumentParser(add_help=False)
    early_parser.add_argument("--qnn-binding", type=Path)
    early_args, _ = early_parser.parse_known_args()
    if early_args.qnn_binding:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "export"))
        from binding import load_binding

        load_binding(early_args.qnn_binding)

    import torch
    from executorch.examples.qualcomm.custom_op import deltanet

    parser = deltanet.setup_common_args_and_variables()
    parser.add_argument("--qnn-binding", type=Path)
    parser.add_argument("-a", "--artifact", type=Path, required=True)
    parser.add_argument("--op_package_dir", type=Path, required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--ar_len", type=int, required=True)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--head_k_dim", type=int, default=128)
    parser.add_argument("--head_v_dim", type=int, default=128)
    parser.add_argument("--input_dtype", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument(
        "--quant_dtypes",
        default="use_16a4w",
        choices=("fp16", "use_16a4w", "use_16a8w"),
    )
    parser.add_argument("--variants", choices=("core_only",), default="core_only")
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--warm_up", type=int, default=5)
    parser.add_argument("--calibration_samples", type=int, default=2)
    parser.add_argument("--independent-state-qparams", action="store_true")
    parser.add_argument("--state-vtcm", action="store_true")
    parser.set_defaults(
        build_op_package=False, stream_recurrence=False, state_vtcm=False
    )
    args = parser.parse_args()
    args.seed = 666 if args.seed is None else args.seed
    quant_dtype = deltanet.parse_quant_dtypes(args.quant_dtypes)[0]
    dtype = torch.float16 if args.input_dtype == "fp16" else torch.float32
    scaling = args.head_k_dim**-0.5
    dataset = []
    for index in range(args.calibration_samples):
        inputs = deltanet.make_inputs(
            args.ar_len,
            args.seed + index,
            1,
            args.heads,
            args.head_k_dim,
            args.head_v_dim,
            dtype,
        )
        dataset.append(
            (
                deltanet._l2norm(inputs[0].float(), 1e-6).mul(scaling).to(dtype),
                deltanet._l2norm(inputs[1].float(), 1e-6).to(dtype),
                inputs[2],
                inputs[3].float().exp().to(dtype),
                *inputs[4:],
            )
        )
    model = deltanet.CustomOpDeltaNetCoreOnly().eval()
    with torch.no_grad():
        references = model(*dataset[0])
    args.artifact.mkdir(parents=True, exist_ok=True)
    reference_dir = args.artifact / "reference"
    reference_dir.mkdir(exist_ok=True)
    for index, tensor in enumerate(references):
        tensor.cpu().numpy().tofile(reference_dir / f"output_{index}.raw")
    qnn_config = deltanet.QnnConfig.load_config(args)
    package_options, package_paths = deltanet.setup_op_package(
        args, state_vtcm=args.state_vtcm
    )
    quantizer = (
        None
        if quant_dtype is None
        else deltanet.make_quantizer(
            quant_dtype=quant_dtype,
            custom_annotations=deltanet.deltanet_quant_annotations(
                quant_dtype, share_state_qparams=not args.independent_state_qparams
            ),
            backend=deltanet.get_backend_type(args.backend),
            soc_model=args.soc_model,
        )
    )
    case_name = (
        f"gdn_core_t{args.ar_len}_h{args.heads}_k{args.head_k_dim}_v{args.head_v_dim}"
    )
    case_dir = args.artifact / case_name
    case_dir.mkdir(exist_ok=True)
    pte_base = case_dir / case_name
    deltanet.build_executorch_binary(
        model=model,
        qnn_config=qnn_config,
        file_name=str(pte_base),
        dataset=dataset,
        quant_dtype=quant_dtype,
        custom_quantizer=quantizer,
        op_package_options=package_options,
    )
    adb = deltanet.SimpleADB(
        qnn_config=qnn_config, pte_path=str(pte_base) + ".pte", workspace=args.workspace
    )
    adb.push(inputs=[dataset[0]], files=package_paths)
    adb.extra_cmds += f" --warm_up {args.warm_up}"
    runner_output = []
    adb.execute(
        output_callback=lambda process: runner_output.append(process.stdout),
        iteration=args.iteration,
    )
    (case_dir / "runner.log").write_text(runner_output[0])
    adb.pull(host_output_path=str(case_dir))
    result = {
        "variant": "core_only",
        "share_state_qparams": not args.independent_state_qparams,
        "state_vtcm": args.state_vtcm,
        "quant_dtype": args.quant_dtypes,
        "runner_avg_ms": deltanet.parse_runner_avg_ms(runner_output[0]),
        "outputs": deltanet.read_outputs(case_dir / "outputs", references),
    }
    (args.artifact / "summary.json").write_text(json.dumps([result], indent=2))


if __name__ == "__main__":
    main()
