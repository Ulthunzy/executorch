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

            def slice_dequant(node, index, sequence=None):
                if sequence is not None:
                    start, end, dimension = sequence
                    source = node.args[0].args[0] if node.args[0].target == Q else None
                    if (
                        source is not None
                        and source.target == torch.ops.aten.cat.default
                        and source.args[1] % node.meta["val"].ndim == dimension
                    ):
                        parts, offset = [], 0
                        for part in source.args[0]:
                            width = part.meta["val"].shape[dimension]
                            first, last = max(start - offset, 0), min(
                                end - offset, width
                            )
                            if first < last:
                                assert part.target == DQ
                                piece = graph.call_function(
                                    torch.ops.aten.slice.Tensor,
                                    (part, dimension, first, last),
                                )
                                codes = graph.call_function(Q, (piece, *part.args[1:]))
                                parts.append(
                                    graph.call_function(DQ, (codes, *part.args[1:]))
                                )
                            offset += width
                        assert parts
                        sliced = (
                            parts[0]
                            if len(parts) == 1
                            else graph.call_function(
                                torch.ops.aten.cat.default, (parts, dimension)
                            )
                        )
                    else:
                        sliced = graph.call_function(
                            torch.ops.aten.slice.Tensor, (node, dimension, start, end)
                        )
                    integer = graph.call_function(Q, (sliced, *node.args[1:]))
                    node = graph.call_function(DQ, (integer, *node.args[1:]))
                sliced = graph.call_function(
                    torch.ops.aten.slice.Tensor, (node, 1, index, index + 1)
                )
                integer = graph.call_function(Q, (sliced, *node.args[1:]))
                return graph.call_function(
                    DQ, (integer, *node.args[1:]), dict(node.kwargs)
                )

            # Preserve every calibrated grid while skipping the GQA repeat/reshape chain.
            def kv_value(node, index, sequence=None):
                if node is boundary:
                    return slice_dequant(node, index, sequence)
                if node.target in SHAPES:
                    return kv_value(node.args[0], index, sequence)
                assert node.target in {Q, DQ}
                return graph.call_function(
                    node.target,
                    (kv_value(node.args[0], index, sequence), *node.args[1:]),
                    dict(node.kwargs),
                )

            for head in range(heads):
                reduction = left.meta["val"].shape[-1] > 4096
                length = (
                    left.meta["val"].shape[-1]
                    if reduction
                    else right.meta["val"].shape[-1]
                )
                step = 2048 if length > 4096 else length
                pieces = []
                for start in range(0, length, step):
                    end = min(start + step, length)
                    # Slice sequence before heads: no full-context KV head
                    # temporary needs to fit in VTCM. Softmax stays global.
                    query = slice_dequant(
                        left,
                        head,
                        (start, end, 3) if reduction and length > 4096 else None,
                    )
                    key_value = kv_value(
                        right,
                        head // (heads // kv_heads),
                        (start, end, 2 if reduction else 3) if length > 4096 else None,
                    )
                    pieces.append(
                        graph.call_function(
                            torch.ops.aten.matmul.default, (query, key_value)
                        )
                    )
                value = pieces[0]
                if reduction:
                    for piece in pieces[1:]:
                        value = graph.call_function(
                            torch.ops.aten.add.Tensor, (value, piece)
                        )
                elif len(pieces) > 1:
                    value = graph.call_function(torch.ops.aten.cat.default, (pieces, 3))
                if len(pieces) > 1:
                    value.meta["fp16_attention_join"] = True
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
