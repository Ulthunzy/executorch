import torch
from torch._subclasses.fake_tensor import FakeTensorMode


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
    inputs = []
    for index, node in enumerate(placeholders):
        shape = list(node.meta["val"].shape)
        if full_attention:
            shape = [
                (
                    capacity
                    if x == old_capacity
                    else capacity - 1 if x == old_capacity - 1 else x
                )
                for x in shape
            ]
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
                                else (capacity - 1 if x == old_capacity - 1 else x)
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
