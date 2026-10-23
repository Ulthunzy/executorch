import torch
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch._subclasses.fake_tensor import FakeTensorMode

Q = torch.ops.quantized_decomposed.quantize_per_tensor.default
DQ = torch.ops.quantized_decomposed.dequantize_per_tensor.default
SHAPES = {
    torch.ops.aten.reshape.default,
    torch.ops.aten.view.default,
    torch.ops.aten.expand.default,
    torch.ops.aten.unsqueeze.default,
}


def split_attention_core(module, inputs):
    graph = module.graph
    matmuls = [
        node for node in graph.nodes if node.target == torch.ops.aten.matmul.default
    ]
    assert len(matmuls) == 2
    for matmul in matmuls:
        left, right = matmul.args
        assert left.target == right.target == DQ
        heads = left.meta["val"].shape[1]
        assert heads == 16
        # Keep projections and Q/K normalization intact; stop at the unexpanded KV cache.
        boundary = right
        while True:
            if (
                boundary.target == DQ
                and boundary.args[0].target == Q
                and boundary.args[0].args[0].target == torch.ops.aten.cat.default
            ):
                break
            assert boundary.target in {Q, DQ} | SHAPES, (
                boundary.name,
                boundary.target,
            )
            boundary = boundary.args[0]
        kv_heads = boundary.meta["val"].shape[1]
        assert heads % kv_heads == 0
        output_q = next(iter(matmul.users))
        assert output_q.target == Q and len(matmul.users) == 1
        outputs = []
        with graph.inserting_before(matmul):

            def slice_dequant(node, index):
                sliced = graph.call_function(
                    torch.ops.aten.slice.Tensor, (node, 1, index, index + 1)
                )
                integer = graph.call_function(Q, (sliced, *node.args[1:]))
                return graph.call_function(
                    DQ, (integer, *node.args[1:]), dict(node.kwargs)
                )

            # Preserve every calibrated grid while skipping the GQA repeat/reshape chain.
            def kv_value(node, index):
                if node is boundary:
                    return slice_dequant(node, index)
                if node.target in SHAPES:
                    return kv_value(node.args[0], index)
                assert node.target in {Q, DQ}
                return graph.call_function(
                    node.target,
                    (kv_value(node.args[0], index), *node.args[1:]),
                    dict(node.kwargs),
                )

            for head in range(heads):
                query = slice_dequant(left, head)
                key_value = kv_value(right, head // (heads // kv_heads))
                value = graph.call_function(
                    torch.ops.aten.matmul.default, (query, key_value)
                )
                integer = graph.call_function(
                    Q, (value, *output_q.args[1:]), dict(output_q.kwargs)
                )
                outputs.append(graph.call_function(DQ, (integer, *output_q.args[1:])))
            concat = graph.call_function(torch.ops.aten.cat.default, (outputs, 1))
            joined = graph.call_function(Q, (concat, *output_q.args[1:]))
        output_q.replace_all_uses_with(joined)
    graph.eliminate_dead_code()
    graph.lint()
    module.recompile()
    FakeTensorProp(module, mode=FakeTensorMode(allow_non_fake_inputs=True)).propagate(
        *inputs
    )
    return module
