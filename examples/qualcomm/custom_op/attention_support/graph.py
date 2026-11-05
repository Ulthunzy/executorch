"""Replace a calibrated causal Attention core with the shared valid-length op."""

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.passes.fake_tensor_prop import FakeTensorProp

from executorch.examples.qualcomm.custom_op import attention  # noqa: F401


def replace_attention(module, inputs, capacity):
    graph = module.graph
    q = torch.ops.quantized_decomposed.quantize_per_tensor.default
    dq = torch.ops.quantized_decomposed.dequantize_per_tensor.default
    matmuls = [
        node for node in graph.nodes if node.target == torch.ops.aten.matmul.default
    ]
    if len(matmuls) != 2:
        raise ValueError("Expected one QK and one probability/value matmul")
    qk, pv = matmuls
    softmax = next(
        node for node in graph.nodes if node.target == torch.ops.aten.softmax.int
    )

    def left_until(node, predicate):
        while isinstance(node, torch.fx.Node):
            if predicate(node):
                return node
            if not node.args:
                break
            node = node.args[0]
        raise ValueError("Unsupported calibrated Attention core")

    def grid(node):
        node = left_until(node, lambda n: n.target == dq)
        if node.args[0].target != q or node.args[3:] != (0, 65535, torch.int32):
            raise ValueError("AttentionValid requires calibrated per-tensor U16 grids")
        return float(node.args[1]), int(node.args[2])

    def constant(value):
        if not isinstance(value, torch.fx.Node):
            return value
        if value.op == "get_attr":
            return module.get_buffer(value.target)
        if value.op != "call_function":
            raise ValueError("Expected a constant Attention scale")
        args = torch.fx.map_arg(value.args, constant)
        kwargs = torch.fx.map_arg(value.kwargs, constant)
        return value.target(*args, **kwargs)

    def cache_pair(node):
        node = left_until(
            node,
            lambda n: n.target == dq
            and n.args[0].target == q
            and n.args[0].args[0].target == torch.ops.aten.cat.default,
        )
        cat = node.args[0].args[0]
        if len(cat.args[0]) != 2:
            raise ValueError("Expected historical and new KV tensors")
        return cat.args[0]

    add = left_until(softmax.args[0], lambda n: n.target == torch.ops.aten.add.Tensor)
    divide = left_until(add.args[0], lambda n: n.target == torch.ops.aten.div.Tensor)
    divisor = constant(divide.args[1])
    divisor = float(divisor.item() if isinstance(divisor, torch.Tensor) else divisor)
    hk, nk = cache_pair(qk.args[1])
    hv, nv = cache_pair(pv.args[1])
    key_grid, value_grid = grid(hk), grid(hv)
    grids = (
        *grid(divide.args[0]),
        1.0 / divisor,
        *grid(add.args[0]),
        *grid(softmax.args[0]),
        *grid(pv.args[0]),
        *key_grid,
        *value_grid,
    )
    placeholders = [node for node in graph.nodes if node.op == "placeholder"]
    key_input = left_until(hk, lambda n: n.op == "placeholder")
    value_input = left_until(hv, lambda n: n.op == "placeholder")
    mask_input = left_until(add.args[1], lambda n: n.op == "placeholder")
    values = list(inputs)
    queries = qk.args[0].meta["val"].shape[-2]
    heads = nk.meta["val"].shape[1]
    dimension = nk.meta["val"].shape[2]
    for node in (key_input, value_input, mask_input):
        tensor = torch.zeros((1, 1, 1, 3), dtype=torch.int32)
        if node is mask_input:
            tensor.reshape(-1)[0] = queries
        values[placeholders.index(node)] = tensor
        node.meta["val"] = tensor
    module.register_buffer(
        "valid_attention_grids",
        torch.tensor(grids, dtype=torch.float32).reshape(1, 1, 1, 13),
    )
    with graph.inserting_before(pv):
        metadata = graph.get_attr("valid_attention_grids")
        result = graph.call_function(
            torch.ops.qnn_custom_ops.attention_valid.default,
            (qk.args[0], key_input, nk, value_input, nv, mask_input, metadata),
        )
    pv.replace_all_uses_with(result)
    graph.eliminate_dead_code()
    graph.lint()
    module.recompile()
    FakeTensorProp(module, mode=FakeTensorMode(allow_non_fake_inputs=True)).propagate(
        *values
    )
    record = dict(
        capacity=capacity,
        heads=heads,
        dimension=dimension,
        descriptor_words=3,
        layout="prefix",
        encodings=[
            dict(scale=scale, offset=-zero, elements=heads * dimension * capacity)
            for scale, zero in (key_grid, value_grid)
        ],
    )
    return module, tuple(values), record
