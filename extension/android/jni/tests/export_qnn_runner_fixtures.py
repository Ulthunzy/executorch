# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Small portable PTEs for testing QNN runner control flow without QNN export."""

import argparse
from pathlib import Path

import torch
from executorch.exir import ExecutorchBackendConfig, to_edge
from executorch.exir.passes import MemoryPlanningPass


class DecoderFixture(torch.nn.Module):
    def __init__(self, ar):
        super().__init__()
        self.ar = ar

    def forward(self, tokens, mask, *cache_inputs):
        anchor = tokens.to(torch.float32).sum() * 0
        return (
            torch.zeros(1, self.ar, 16) + anchor,
            torch.zeros(1, 1, 2, self.ar) + anchor,
            torch.zeros(1, 1, self.ar, 2) + anchor,
        )


class TextEncoderFixture(torch.nn.Module):
    def forward(self, ids, mask):
        return ids.to(torch.float32) * mask


def inputs(ar):
    values = (torch.zeros(1, ar, dtype=torch.int64), torch.zeros(1, 1, ar, 16))
    if ar == 16:
        return values
    return (
        *values,
        torch.zeros(1, ar, dtype=torch.int32),
        torch.zeros(1, 1, 2, 16 - ar),
        torch.zeros(1, 1, 16 - ar, 2),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    encoder = to_edge(
        torch.export.export(
            TextEncoderFixture(),
            (torch.zeros(1, 4, dtype=torch.int32), torch.ones(1, 4)),
        )
    ).to_executorch()
    (args.output / "text_encoder.pte").write_bytes(encoder.buffer)
    for ar in (4, 16):
        methods = {
            "prefill_forward": torch.export.export(DecoderFixture(ar), inputs(ar)),
            "kv_forward": torch.export.export(DecoderFixture(1), inputs(1)),
        }
        program = to_edge(methods, constant_methods={"get_n_layers": 1}).to_executorch(
            config=ExecutorchBackendConfig(
                memory_planning_pass=MemoryPlanningPass(
                    alloc_graph_input=False, alloc_graph_output=False
                )
            )
        )
        path = args.output / f"hybrid_ar{ar}.pte"
        path.write_bytes(program.buffer)
        print(path)


if __name__ == "__main__":
    main()
