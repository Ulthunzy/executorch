# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
import re
import shutil
import sys
from collections import OrderedDict
from multiprocessing.connection import Client
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from executorch.backends.qualcomm._passes import ExpandBroadcastTensorShape, FoldQDQ
from executorch.backends.qualcomm._passes.qnn_pass_manager import (
    get_capture_program_passes,
)
from executorch.backends.qualcomm._passes.utils import (
    get_passes_dependency_for_capture_program,
)
from executorch.backends.qualcomm.debugger.utils import generate_optrace
from executorch.backends.qualcomm.export_utils import (
    build_executorch_binary,
    QnnConfig,
    setup_common_args_and_variables,
    SimpleADB,
)
from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
from executorch.backends.qualcomm.utils.constants import (
    QCOM_PASS_ACTIVATE_KEY,
    QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY,
)
from executorch.backends.qualcomm.utils.utils import get_soc_to_chipset_map
from executorch.exir import to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.passes import dead_code_elimination_pass


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + eps)


class OldDeltaNetRule(torch.nn.Module):
    def __init__(self, scaling: float = 0.08838834764831845):
        super().__init__()
        self.scaling = scaling

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        recurrent_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        initial_dtype = query.dtype
        query = _l2norm(query.float()) * self.scaling
        key = _l2norm(key.float())
        value = value.float()
        beta = beta.float()
        last_recurrent_state = recurrent_state[: query.shape[0]].float()
        outputs = []

        for i in range(query.shape[1]):
            q_t = query[:, i]
            k_t = key[:, i]
            v_t = value[:, i]
            g_t = g[:, i].exp().unsqueeze(-1).unsqueeze(-1)
            g_t = g_t.expand_as(last_recurrent_state)
            beta_t = beta[:, i].unsqueeze(-1).expand_as(v_t)

            last_recurrent_state = last_recurrent_state * g_t
            k_for_state = k_t.unsqueeze(-1).expand_as(last_recurrent_state)
            kv_mem = (last_recurrent_state * k_for_state).sum(dim=-2)
            delta = (v_t - kv_mem) * beta_t
            state_update = k_t.unsqueeze(-1).expand(
                k_t.shape[0], k_t.shape[1], k_t.shape[2], delta.shape[-1]
            ) * delta.unsqueeze(-2).expand(
                delta.shape[0], delta.shape[1], k_t.shape[-1], delta.shape[-1]
            )
            last_recurrent_state = last_recurrent_state + state_update
            q_for_state = q_t.unsqueeze(-1).expand_as(last_recurrent_state)
            outputs.append((last_recurrent_state * q_for_state).sum(dim=-2))

        return torch.stack(outputs, dim=1).to(
            initial_dtype
        ), last_recurrent_state.to(initial_dtype)


class NewDeltaNetRule(torch.nn.Module):
    def __init__(self, scaling: float = 0.08838834764831845):
        super().__init__()
        self.scaling = scaling

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        recurrent_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        initial_dtype = query.dtype
        if query.dtype != torch.float32:
            query = query.float()
        if key.dtype != torch.float32:
            key = key.float()
        if value.dtype != torch.float32:
            value = value.float()
        if beta.dtype != torch.float32:
            beta = beta.float()
        last_recurrent_state = recurrent_state[: query.shape[0]]
        if last_recurrent_state.dtype != torch.float32:
            last_recurrent_state = last_recurrent_state.float()
        query = _l2norm(query) * self.scaling
        key = _l2norm(key)
        outputs = []

        for i in range(query.shape[1]):
            q_t = query[:, i]
            k_t = key[:, i]
            v_t = value[:, i]
            g_t = g[:, i].exp().unsqueeze(-1).unsqueeze(-1)
            beta_t = beta[:, i].unsqueeze(-1)

            last_recurrent_state = last_recurrent_state * g_t
            kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
            delta = (v_t - kv_mem) * beta_t
            last_recurrent_state = last_recurrent_state + k_t.unsqueeze(
                -1
            ) * delta.unsqueeze(-2)
            outputs.append((last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2))

        core_attn_out = torch.stack(outputs, dim=1)
        if core_attn_out.dtype != initial_dtype:
            core_attn_out = core_attn_out.to(initial_dtype)
        if last_recurrent_state.dtype != initial_dtype:
            last_recurrent_state = last_recurrent_state.to(initial_dtype)
        return core_attn_out, last_recurrent_state


class RemoveBroadcastExpandForTest(ExportPass):
    def __init__(self):
        super().__init__()
        self.broadcast_op_targets = {
            exir_ops.edge.aten.add.Tensor,
            exir_ops.edge.aten.sub.Tensor,
            exir_ops.edge.aten.mul.Tensor,
            exir_ops.edge.aten.div.Tensor,
        }

    def _broadcastable(self, input_shape, output_shape):
        input_shape = tuple(input_shape)
        output_shape = tuple(output_shape)
        if len(input_shape) > len(output_shape):
            return False
        padded_input_shape = (1,) * (len(output_shape) - len(input_shape)) + input_shape
        return all(
            input_dim == output_dim or input_dim == 1
            for input_dim, output_dim in zip(padded_input_shape, output_shape)
        )

    def _can_fold(self, expand_node, user_node):
        if user_node.target not in self.broadcast_op_targets:
            return False
        source_node = expand_node.args[0]
        if not isinstance(source_node, torch.fx.Node):
            return False
        return self._broadcastable(
            source_node.meta["val"].shape, expand_node.meta["val"].shape
        ) and self._broadcastable(
            source_node.meta["val"].shape, user_node.meta["val"].shape
        )

    def call(self, graph_module: torch.fx.GraphModule):
        modified = False
        for node in list(graph_module.graph.nodes):
            if node.target != exir_ops.edge.aten.expand_copy.default:
                continue
            source_node = node.args[0]
            for user_node in list(node.users.keys()):
                if self._can_fold(node, user_node):
                    user_node.replace_input_with(node, source_node)
                    modified = True
            if not node.users:
                graph_module.graph.erase_node(node)

        if modified:
            dead_code_elimination_pass(graph_module)
            graph_module.recompile()
            graph_module.graph.lint()
        return PassResult(graph_module, modified)


def make_inputs(
    ar_len: int,
    seed: int,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, ...]:
    torch.manual_seed(seed)
    batch, heads, head_k_dim, head_v_dim = 1, 16, 128, 128
    return (
        torch.randn(batch, ar_len, heads, head_k_dim, dtype=dtype),
        torch.randn(batch, ar_len, heads, head_k_dim, dtype=dtype),
        torch.randn(batch, ar_len, heads, head_v_dim, dtype=dtype),
        torch.randn(batch, ar_len, heads, dtype=torch.float32) * 0.01,
        torch.rand(batch, ar_len, heads, dtype=dtype),
        torch.randn(batch, heads, head_k_dim, head_v_dim, dtype=dtype),
    )


def parse_quant_dtypes(raw: str) -> List[Optional[QuantDtype]]:
    result = []
    for item in raw.split(","):
        name = item.strip()
        if not name:
            continue
        if name in {"none", "fp16", "float"}:
            result.append(None)
        else:
            result.append(getattr(QuantDtype, name))
    return result


def quant_name(quant_dtype: Optional[QuantDtype]) -> str:
    return "fp16" if quant_dtype is None else quant_dtype.name


def make_model(variant: str) -> torch.nn.Module:
    if variant in {"old", "old_pass"}:
        return OldDeltaNetRule().eval()
    if variant in {"new", "new_pass"}:
        return NewDeltaNetRule().eval()
    raise RuntimeError(f"Unknown variant: {variant}")


def pass_enabled(variant: str) -> bool:
    return variant in {"old_pass", "new_pass"}


def make_passes_job(enable_remove_broadcast_expand: bool):
    passes_job = get_capture_program_passes()
    if not enable_remove_broadcast_expand:
        return passes_job

    ordered_passes_job = OrderedDict()
    for pass_cls, setting in passes_job.items():
        if pass_cls is ExpandBroadcastTensorShape:
            ordered_passes_job[RemoveBroadcastExpandForTest] = {
                QCOM_PASS_ACTIVATE_KEY: True,
                QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY: {},
            }
        ordered_passes_job[pass_cls] = setting
    return ordered_passes_job


def make_passes_dependency(enable_remove_broadcast_expand: bool):
    if not enable_remove_broadcast_expand:
        return None
    passes_dependency = get_passes_dependency_for_capture_program()
    passes_dependency[RemoveBroadcastExpandForTest] = [FoldQDQ]
    passes_dependency[ExpandBroadcastTensorShape] = [RemoveBroadcastExpandForTest]
    return passes_dependency


def count_expand_copy(model: torch.nn.Module, sample_input: Tuple[torch.Tensor, ...]):
    edge_program = to_edge(torch.export.export(model, sample_input, strict=True))
    graph_module = edge_program.exported_program().graph_module
    before = sum(
        n.target == exir_ops.edge.aten.expand_copy.default
        for n in graph_module.graph.nodes
    )
    graph_module = RemoveBroadcastExpandForTest()(graph_module).graph_module
    after_remove_broadcast_expand = sum(
        n.target == exir_ops.edge.aten.expand_copy.default
        for n in graph_module.graph.nodes
    )
    return {
        "edge_expand_copy": before,
        "edge_expand_copy_after_remove_broadcast_expand": after_remove_broadcast_expand,
    }


def parse_runner_avg_ms(output: str) -> Optional[float]:
    matches = re.findall(
        r"(?<!Total\s)\d+\s+inferences?\s+took\s+[0-9.]+\s+ms,\s+avg\s+([0-9.]+)\s+ms",
        output,
    )
    if not matches:
        matches = re.findall(
            r"Total\s+\d+\s+inferences?\s+took\s+[0-9.]+\s+ms,\s+avg\s+([0-9.]+)\s+ms",
            output,
        )
    return None if not matches else float(matches[-1])


def read_outputs(
    output_dir: Path,
    refs: Sequence[torch.Tensor],
) -> List[Dict[str, object]]:
    output_info = []
    output_files = sorted(output_dir.glob("*.raw"))
    for index, (output_file, ref) in enumerate(zip(output_files, refs)):
        expected_numel = ref.numel()
        raw_size = output_file.stat().st_size
        dtype = None
        for candidate in (np.float32, np.float16, np.uint16, np.uint8):
            if raw_size == expected_numel * np.dtype(candidate).itemsize:
                dtype = candidate
                break
        if dtype is None:
            output_info.append(
                {
                    "file": str(output_file),
                    "error": f"unexpected byte size {raw_size}",
                }
            )
            continue

        data = np.fromfile(output_file, dtype=dtype).reshape(tuple(ref.shape))
        out = torch.from_numpy(data.astype(np.float32))
        ref_float = ref.detach().cpu().float()
        abs_diff = (out - ref_float).abs()
        output_info.append(
            {
                "file": str(output_file),
                "dtype": np.dtype(dtype).name,
                "shape": list(ref.shape),
                "max_abs_diff": float(abs_diff.max()),
                "mean_abs_diff": float(abs_diff.mean()),
            }
        )
    return output_info


def run_one(
    args,
    qnn_config: QnnConfig,
    variant: str,
    quant_dtype: Optional[QuantDtype],
    dataset: List[Tuple[torch.Tensor, ...]],
) -> Dict[str, object]:
    qname = quant_name(quant_dtype)
    model = make_model(variant)
    sample_input = dataset[0]
    with torch.no_grad():
        refs = model(*sample_input)

    case_name = f"deltanet_{variant}_{qname}_ar{args.ar_len}"
    case_dir = Path(args.artifact) / case_name
    if case_dir.exists() and not qnn_config.pre_gen_pte:
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    pte_file = case_dir / case_name
    counts = count_expand_copy(model, sample_input)
    passes_job = make_passes_job(pass_enabled(variant))

    logging.info(
        "Building %s quant=%s remove_broadcast_expand=%s",
        variant,
        qname,
        pass_enabled(variant),
    )
    try:
        build_executorch_binary(
            model=model,
            qnn_config=qnn_config,
            file_name=str(pte_file),
            dataset=dataset,
            quant_dtype=quant_dtype,
            passes_job=passes_job,
            passes_dependency=make_passes_dependency(pass_enabled(variant)),
        )
    except SystemExit:
        if not args.compile_only:
            raise

    result = {
        "case": case_name,
        "variant": variant,
        "quant_dtype": qname,
        "remove_broadcast_expand_pass": pass_enabled(variant),
        **counts,
    }
    if args.compile_only:
        return result

    pte_path = f"{pte_file}.pte"
    adb = SimpleADB(
        qnn_config=qnn_config,
        pte_path=pte_path,
        workspace=f"/data/local/tmp/executorch/{case_name}",
    )

    if args.optrace:
        result["binaries_trace"] = generate_optrace(
            str(case_dir),
            get_soc_to_chipset_map()[qnn_config.soc_model],
            adb,
            pte_path,
            dataset,
        )

    runner_output = {"text": ""}

    def capture_output(proc):
        runner_output["text"] = proc.stdout

    adb.push(inputs=[sample_input])
    adb.execute(output_callback=capture_output, iteration=args.iteration)
    (case_dir / "runner.log").write_text(runner_output["text"])
    adb.pull(host_output_path=str(case_dir))

    result["runner_avg_ms"] = parse_runner_avg_ms(runner_output["text"])
    result["outputs"] = read_outputs(case_dir / "outputs", refs)
    return result


def main(args) -> None:
    qnn_config = QnnConfig.load_config(args.config_file if args.config_file else args)
    if args.optrace and qnn_config.profile_level != 3:
        raise RuntimeError("--optrace requires --profile_level 3")

    if os.path.exists(args.artifact) and not args.keep_artifact and not args.pre_gen_pte:
        shutil.rmtree(args.artifact)
    os.makedirs(args.artifact, exist_ok=True)

    dataset = [
        make_inputs(args.ar_len, args.seed + i)
        for i in range(args.calibration_samples)
    ]

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    quant_dtypes = parse_quant_dtypes(args.quant_dtypes)
    results = []
    for quant_dtype in quant_dtypes:
        for variant in variants:
            results.append(run_one(args, qnn_config, variant, quant_dtype, dataset))

    summary_path = Path(args.artifact) / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    logging.info("Saved summary to %s", summary_path)

    if args.ip and args.port != -1:
        with Client((args.ip, args.port)) as conn:
            conn.send(json.dumps({"results": results}))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = setup_common_args_and_variables()
    parser.add_argument(
        "-a",
        "--artifact",
        default="./qwen35-deltanet-qnn-test",
        type=str,
        help="Folder for generated DeltaNet QNN artifacts.",
    )
    parser.add_argument("--ar_len", default=1, type=int)
    parser.add_argument("--iteration", default=50, type=int)
    parser.add_argument("--calibration_samples", default=4, type=int)
    parser.add_argument("--variants", default="old,new", type=str)
    parser.add_argument("--quant_dtypes", default="use_16a4w", type=str)
    parser.add_argument("--optrace", action="store_true")
    parser.add_argument("--keep_artifact", action="store_true")

    args = parser.parse_args()
    if args.seed is None:
        args.seed = 666

    try:
        main(args)
    except Exception as e:
        if args.ip and args.port != -1:
            with Client((args.ip, args.port)) as conn:
                conn.send(json.dumps({"Error": str(e)}))
        else:
            raise Exception(e)
