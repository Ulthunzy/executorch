"""Change only the recurrent GDN cache interface after GPTQ calibration."""

import operator

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.passes.fake_tensor_prop import FakeTensorProp


def fp16_gdn_state(module, inputs, *, decode):
    graph = module.graph
    cores = [
        node
        for node in graph.nodes
        if str(node.target) == "qnn_custom_ops.deltanet_core.default"
    ]
    if len(cores) != 1 or len(inputs) != 3:
        raise ValueError("Expected one GDN core with hidden/conv/recurrent-state IO")
    core = cores[0]
    dq = torch.ops.quantized_decomposed.dequantize_per_tensor.default
    q = torch.ops.quantized_decomposed.quantize_per_tensor.default

    def strip_boundary(value, terminal):
        path, pairs = [], 0
        while value is not terminal:
            path.append(value)
            if value.target == dq and value.args[0].target == q:
                path.append(value.args[0])
                value = value.args[0].args[0]
                pairs += 1
            elif value.target in (
                torch.ops.aten.to.dtype,
                torch.ops.aten._to_copy.default,
            ):
                value = value.args[0]
            else:
                raise ValueError(
                    f"Unexpected operation on GDN state boundary: {value.target}"
                )
        if not pairs:
            raise ValueError("Expected calibrated recurrent-state QDQ")
        return path

    initial = [node for node in graph.nodes if node.op == "placeholder"][2]
    input_path = strip_boundary(core.args[5], initial)
    core.args = (*core.args[:5], initial)
    if decode:
        core.target = torch.ops.qnn_custom_ops.deltanet_core_fp16_state.default
    output = next(node for node in graph.nodes if node.op == "output")
    values = list(output.args[0])
    state = next(
        user
        for user in core.users
        if user.target == operator.getitem and user.args[1] == 1
    )
    output_path = strip_boundary(values[3], state)
    for node in {initial, core, state, *input_path, *output_path}:
        for key in ("quant_attrs", "requantize", "quantization_annotation"):
            node.meta.pop(key, None)
    with graph.inserting_before(output):
        values[3] = graph.call_function(
            torch.ops.aten._to_copy.default, (state,), {"dtype": torch.float16}
        )
    output.args = (tuple(values),)
    inputs = (*inputs[:2], inputs[2].half())
    for node in (*output_path, *input_path):
        if not node.users:
            graph.erase_node(node)
    graph.eliminate_dead_code()
    graph.lint()
    module.recompile()
    FakeTensorProp(module, mode=FakeTensorMode(allow_non_fake_inputs=True)).propagate(
        *inputs
    )
    return module, inputs
