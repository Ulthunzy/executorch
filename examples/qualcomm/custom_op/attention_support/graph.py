"""Replace a calibrated causal Attention core with the shared valid-length op."""

import torch

from executorch.examples.qualcomm.custom_op import attention  # noqa: F401
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.passes.fake_tensor_prop import FakeTensorProp


def replace_attention_layers(module, inputs, capacity):
    graph = module.graph
    q = torch.ops.quantized_decomposed.quantize_per_tensor.default
    dq = torch.ops.quantized_decomposed.dequantize_per_tensor.default
    softmaxes = [
        node for node in graph.nodes if node.target == torch.ops.aten.softmax.int
    ]
    if not softmaxes:
        raise ValueError("Expected calibrated causal Attention cores")

    def left_until(node, predicate):
        while isinstance(node, torch.fx.Node):
            if predicate(node):
                return node
            if not node.args:
                break
            node = node.args[0]
        raise ValueError("Unsupported calibrated Attention core")

    def grid(node, cache=False):
        node = left_until(node, lambda n: n.target == dq)
        encodings = [(0, 65535, torch.int32)]
        if cache:
            encodings.append((0, 255, torch.uint8))
        if node.args[0].target != q or node.args[3:] not in encodings:
            raise ValueError(
                "AttentionValid requires per-tensor U16 Q and U8/U16 KV grids"
            )
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

    placeholders = [node for node in graph.nodes if node.op == "placeholder"]
    values = list(inputs)
    records = []
    for ordinal, softmax in enumerate(softmaxes):
        add = left_until(
            softmax.args[0], lambda n: n.target == torch.ops.aten.add.Tensor
        )
        divide = left_until(
            add.args[0], lambda n: n.target == torch.ops.aten.div.Tensor
        )
        qk = left_until(
            divide.args[0], lambda n: n.target == torch.ops.aten.matmul.default
        )
        users = list(softmax.users)
        while len(users) == 1 and users[0].target in (q, dq):
            users = list(users[0].users)
        if len(users) != 1 or users[0].target != torch.ops.aten.matmul.default:
            raise ValueError("Expected one probability/value matmul")
        pv = users[0]
        divisor = constant(divide.args[1])
        divisor = float(
            divisor.item() if isinstance(divisor, torch.Tensor) else divisor
        )
        hk, nk = cache_pair(qk.args[1])
        hv, nv = cache_pair(pv.args[1])
        key_grid, value_grid = grid(hk, cache=True), grid(hv, cache=True)
        cache_nodes = [
            left_until(n, lambda x: x.target == dq) for n in (hk, hv, nk, nv)
        ]
        if len({n.args[3:] for n in cache_nodes}) != 1:
            raise ValueError(
                "AttentionValid requires matching historical and new KV bit widths"
            )
        grids = (
            *grid(divide.args[0]),
            1.0 / divisor,
            *grid(add.args[0]),
            *grid(softmax.args[0]),
            *grid(pv.args[0]),
            *key_grid,
            *value_grid,
        )
        key_input = left_until(hk, lambda n: n.op == "placeholder")
        value_input = left_until(hv, lambda n: n.op == "placeholder")
        mask_input = left_until(add.args[1], lambda n: n.op == "placeholder")
        queries = qk.args[0].meta["val"].shape[-2]
        heads = nk.meta["val"].shape[1]
        dimension = nk.meta["val"].shape[2]
        for node in (key_input, value_input, mask_input):
            tensor = torch.zeros((1, 1, 1, 3), dtype=torch.int32)
            if node is mask_input:
                tensor.reshape(-1)[0] = queries
            values[placeholders.index(node)] = tensor
        name = f"valid_attention_grids_{ordinal}"
        module.register_buffer(
            name, torch.tensor(grids, dtype=torch.float32).reshape(1, 1, 1, 13)
        )
        with graph.inserting_before(pv):
            metadata = graph.get_attr(name)
            result = graph.call_function(
                torch.ops.qnn_custom_ops.attention_valid.default,
                (qk.args[0], key_input, nk, value_input, nv, mask_input, metadata),
            )
        pv.replace_all_uses_with(result)
        records.append(
            dict(
                capacity=capacity,
                heads=heads,
                dimension=dimension,
                descriptor_words=3,
                layout="prefix",
                key_input_index=placeholders.index(key_input),
                value_input_index=placeholders.index(value_input),
                encodings=[
                    dict(
                        scale=scale, offset=-zero, elements=heads * dimension * capacity
                    )
                    for scale, zero in (key_grid, value_grid)
                ],
            )
        )
    graph.eliminate_dead_code()
    graph.lint()
    module.recompile()
    FakeTensorProp(module, mode=FakeTensorMode(allow_non_fake_inputs=True)).propagate(
        *values
    )
    return module, tuple(values), records


def replace_attention(module, inputs, capacity):
    if sum(n.target == torch.ops.aten.matmul.default for n in module.graph.nodes) != 2:
        raise ValueError("Expected one QK and one probability/value matmul")
    module, inputs, records = replace_attention_layers(module, inputs, capacity)
    if len(records) != 1:
        raise ValueError("Expected one calibrated Attention core")
    record = records[0]
    del record["key_input_index"], record["value_input_index"]
    return module, inputs, record
