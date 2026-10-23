import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.passes.fake_tensor_prop import FakeTensorProp


def fp16_attention_core(module, inputs):
    from .split_attention_core import split_attention_core

    split_attention_core(module, inputs)
    graph = module.graph
    for node in list(graph.nodes):
        if node.target != torch.ops.aten.matmul.default:
            continue
        with graph.inserting_before(node):
            values = tuple(
                graph.call_function(
                    torch.ops.aten._to_copy.default,
                    (value,),
                    {"dtype": torch.float16},
                )
                for value in node.args
            )
            product = graph.call_function(torch.ops.aten.matmul.default, values)
            converted = graph.call_function(
                torch.ops.aten._to_copy.default,
                (product,),
                {"dtype": torch.float32},
            )
        node.replace_all_uses_with(converted)
    graph.eliminate_dead_code()
    graph.lint()
    module.recompile()
    FakeTensorProp(module, mode=FakeTensorMode(allow_non_fake_inputs=True)).propagate(
        *inputs
    )
    return module
