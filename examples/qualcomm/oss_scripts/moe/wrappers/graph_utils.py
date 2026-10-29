import torch
import copy
from torch._subclasses.fake_tensor import FakeTensorMode


def specialize_front_ar1(original, inputs):
    """Specialize one calibrated AR128 front while retaining its Q/DQ grids."""
    from executorch.examples.qualcomm.oss_scripts.llama.quantize.generation import (
        remove_export_checks,
    )

    graph = torch.fx.GraphModule(original, copy.deepcopy(original.graph))
    remove_export_checks(graph)
    values = [inputs[0][:, :1].clone()]
    if len(inputs) == 3:
        values.extend(inputs[1:])
    elif len(inputs) == 6:
        values.extend(
            (
                inputs[1][:1].clone(),
                inputs[2][:1].clone(),
                torch.zeros(1, 1, 1, 1024),
                torch.zeros(1, 2, 256, 1023),
                torch.zeros(1, 2, 1023, 256),
            )
        )
    elif len(inputs) != 1:
        raise ValueError("Expected a head or single-layer attention signature")
    attributes = dict(graph.named_buffers()) | dict(graph.named_parameters())
    mode = FakeTensorMode(allow_non_fake_inputs=True)
    env, arguments = {}, iter(values)
    with mode:
        for node in graph.graph.nodes:
            if node.op == "placeholder":
                value = mode.from_tensor(next(arguments))
            elif node.op == "get_attr":
                value = mode.from_tensor(attributes[node.target])
            elif node.op == "output":
                value = torch.fx.map_arg(node.args[0], env.__getitem__)
            else:
                args = list(node.args)
                if node.target in {
                    torch.ops.aten.reshape.default,
                    torch.ops.aten.view.default,
                    torch.ops.aten.expand.default,
                }:
                    shape = list(args[1])
                    old = args[0].meta["val"]
                    new = env[args[0]]
                    if old.numel() == 128 * new.numel():
                        if len(shape) <= 2:
                            if shape[0] > 0:
                                if shape[0] % 128:
                                    raise ValueError(
                                        f"Cannot specialize {node.name}: {shape}"
                                    )
                                shape[0] //= 128
                        elif 128 in shape:
                            shape[shape.index(128)] = 1
                    args[1] = shape
                elif node.target == torch.ops.aten.slice.Tensor:
                    axis = args[1]
                    old = args[0].meta["val"].shape[axis]
                    new = env[args[0]].shape[axis]
                    if len(args) > 2 and args[2] == -128 and new == old - 127:
                        args[2] = -1
                    if len(args) > 3 and args[3] == old == 128 and new == 1:
                        args[3] = 1
                node.args = tuple(args)
                value = node.target(
                    *torch.fx.map_arg(node.args, env.__getitem__),
                    **torch.fx.map_arg(node.kwargs, env.__getitem__),
                )
            env[node] = value
    for node, value in env.items():
        node.meta["val"] = value
    graph.graph.lint()
    graph.recompile()
    return graph, tuple(values)


def resize_attention(module, capacity):
    module.graph.set_codegen(torch.fx.graph.CodeGen())
    # Shape specialization invalidates ExportedModule's original input guard.
    module._forward_pre_hooks.clear()
    for node in list(module.graph.nodes):
        if "aten._assert_tensor_metadata" in str(node.target) or (
            node.op == "call_module" and node.target == "_guards_fn"
        ):
            assert not node.users
            module.graph.erase_node(node)
    module.recompile()
    nodes = list(module.graph.nodes)
    placeholders = [node for node in nodes if node.op == "placeholder"]
    full_attention = len(placeholders) == 6
    assert len(placeholders) in (3, 6)
    old_capacity = placeholders[3].meta["val"].shape[-1] if full_attention else None
    ar_len = placeholders[0].meta["val"].shape[1]
    inputs = []
    for index, node in enumerate(placeholders):
        shape = list(node.meta["val"].shape)
        if full_attention and index >= 3:
            axis = -1 if index in (3, 4) else -2
            shape[axis] = capacity if index == 3 else capacity - ar_len
        inputs.append(
            torch.ones(shape) if full_attention and index == 1 else torch.zeros(shape)
        )
    attributes = dict(module.named_buffers()) | dict(module.named_parameters())
    mode = FakeTensorMode(allow_non_fake_inputs=True)
    env, values = {}, iter(inputs)
    with mode:
        for node in nodes:
            if node.op == "placeholder":
                value = mode.from_tensor(next(values))
            elif node.op == "get_attr":
                value = mode.from_tensor(attributes[node.target])
            elif node.op == "output":
                value = torch.fx.map_arg(node.args[0], env.__getitem__)
            else:
                arguments = list(node.args)
                if full_attention and node.target in {
                    torch.ops.aten.reshape.default,
                    torch.ops.aten.view.default,
                    torch.ops.aten.expand.default,
                }:
                    if tuple(arguments[0].meta["val"].shape) != tuple(
                        env[arguments[0]].shape
                    ):
                        arguments[1] = [
                            (
                                capacity
                                if x == old_capacity
                                else (
                                    capacity - ar_len
                                    if x == old_capacity - ar_len
                                    else x
                                )
                            )
                            for x in arguments[1]
                        ]
                node.args = tuple(arguments)
                value = node.target(
                    *torch.fx.map_arg(node.args, env.__getitem__),
                    **torch.fx.map_arg(node.kwargs, env.__getitem__),
                )
            env[node] = value
    for node, value in env.items():
        node.meta["val"] = value
    module.graph.lint()
    module.recompile()
    return tuple(inputs)
