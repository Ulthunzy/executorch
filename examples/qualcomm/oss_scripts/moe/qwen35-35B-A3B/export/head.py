import argparse
import hashlib
import json
from pathlib import Path

import sys

# Internal stage scripts also support direct execution for diagnostics.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from workflow.model_config import add_model_config_argument, load_model_config


def main():
    parser = argparse.ArgumentParser(
        description="Compile the newly GPTQ-calibrated shared language head"
    )
    add_model_config_argument(parser)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qnn-binding", type=Path)
    args = parser.parse_args()
    settings = load_model_config(args.model_config)
    report = json.loads(args.source.with_name("GPTQ.json").read_text())
    if (
        settings.quantization.head != "w4a16_gptq"
        or report["method"] != "all_projection_gptq"
        or len(report["projections"]) != 1
        or report["projections"][0]["shape"] != [248320, 2048]
        or report["projections"][0]["bits"] != 4
    ):
        parser.error("A freshly calibrated W4A16 GPTQ head is required")
    from export.binding import load_binding

    qnn = load_binding(args.qnn_binding)
    import torch
    from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset
    from executorch.backends.qualcomm.utils.utils import (
        generate_htp_compiler_spec,
        generate_qnn_executorch_compiler_spec,
        to_edge_transform_and_lower_to_qnn,
    )
    from executorch.exir._serialize._program import deserialize_pte_binary

    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    graph = torch.export.load(args.source).module()
    sample = (torch.zeros(1, 1, 2048),)
    specs = generate_qnn_executorch_compiler_spec(
        QcomChipset[settings.target.soc_model],
        generate_htp_compiler_spec(use_fp16=False),
        shared_buffer=False,
    )
    manager = to_edge_transform_and_lower_to_qnn(graph, sample, specs)
    args.output.mkdir(parents=True, exist_ok=True)
    pte = args.output / "head.pte"
    with pte.open("wb") as file:
        manager.to_executorch().write_to_file(file)
    program = deserialize_pte_binary(pte.read_bytes()).program
    plan = program.execution_plan[0]
    instructions = [i.instr_args for chain in plan.chains for i in chain.instructions]
    if len(instructions) != 1 or type(instructions[0]).__name__ != "DelegateCall":
        raise ValueError("Head must have a single QNN delegate")
    blob = program.backend_delegate_data[plan.delegates[0].processed.index].data
    (args.output / "head.bin").write_bytes(bytes(qnn.StripProtocol(blob)))
    (args.output / "HEAD.json").write_text(
        json.dumps(
            dict(
                target=settings.target.to_dict(),
                source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                quantization=report,
                shape=[1, 1, 2048],
            ),
            indent=2,
        )
        + "\n"
    )
    print("GPTQ_HEAD_EXPORTED", args.output, flush=True)


if __name__ == "__main__":
    main()
