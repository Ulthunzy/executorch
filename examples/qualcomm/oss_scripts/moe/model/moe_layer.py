from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


class MoeFront(torch.nn.Module):
    def __init__(self, constants: Path):
        super().__init__()
        with np.load(constants) as values:
            self.register_buffer(
                "gate_weight", torch.from_numpy(values["gate_weight"]).float()
            )
            self.register_buffer(
                "shared_expert_gate_proj_weight",
                torch.from_numpy(values["shared_expert_gate_proj_weight"]).float(),
            )
            self.register_buffer(
                "shared_expert_up_proj_weight",
                torch.from_numpy(values["shared_expert_up_proj_weight"]).float(),
            )
            self.register_buffer(
                "shared_expert_down_proj_weight",
                torch.from_numpy(values["shared_expert_down_proj_weight"]).float(),
            )
            self.register_buffer(
                "shared_expert_gate_weight",
                torch.from_numpy(values["shared_expert_gate_weight"]).float(),
            )

    def forward(self, x):
        logits = F.linear(x, self.gate_weight)
        scores, ids = logits.softmax(-1).topk(8, -1)
        scores = scores / scores.sum(-1, keepdim=True)
        gate = F.linear(x, self.shared_expert_gate_proj_weight)
        up = F.linear(x, self.shared_expert_up_proj_weight)
        shared = F.linear(F.silu(gate) * up, self.shared_expert_down_proj_weight)
        shared = shared * F.linear(x, self.shared_expert_gate_weight).sigmoid()
        return ids.to(torch.int32), scores, shared


class FusedFront(torch.nn.Module):
    def __init__(self, attention, constants: Path):
        super().__init__()
        self.attention = attention
        self.moe = MoeFront(constants).half().eval()

    def forward(self, *inputs):
        outputs = self.attention(*inputs)
        ids, scores, shared = self.moe(outputs[0].half())
        return (*outputs, ids, scores.float(), shared.float())


class QuantizedFusedFront(torch.nn.Module):
    def __init__(self, attention, moe):
        super().__init__()
        self.attention = attention
        self.moe = moe

    def forward(self, *inputs):
        outputs = self.attention(*inputs)
        ids, scores, shared = self.moe(outputs[0])
        return (*outputs, ids, scores.float(), shared.float())
