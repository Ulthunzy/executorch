"""AR128 fronts whose padding does not advance recurrent or convolution state."""

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.passes.fake_tensor_prop import FakeTensorProp

from executorch.examples.qualcomm.oss_scripts.moe.wrappers.graph_utils import (
    resize_attention,
)


from executorch.examples.qualcomm.custom_op.deltanet_support.masking import (
    masked_fp32_core,
    masked_u16_core,
)


def padded_prefill_front(module, inputs, capacity, gdn_core="chunked_mixed_fp16"):
    if gdn_core not in (
        "chunked_mixed_fp16",
        "chunk128_hmx",
        "custom_fp16",
        "custom_fp32",
        "custom_u16",
    ):
        raise ValueError("Unsupported prefill GDN implementation")
    from executorch.examples.qualcomm.oss_scripts.llama.quantize.generation import (
        mask_chunk_prefill_padding,
        remove_export_checks,
        replace_prefill_cores,
    )

    remove_export_checks(module)
    if len(inputs) == 6:
        # Keep the complete logical history plus a physical AR128 append area.
        # A partial final block can then fit without dropping historical KV.
        inputs = resize_attention(module, capacity + 128)
        if capacity > 4096:
            from executorch.examples.qualcomm.oss_scripts.moe.model.attention import (
                fp16_attention_core,
            )

            fp16_attention_core(module, inputs)
        elif capacity + 128 <= 4096:
            # Sequence tiling would introduce unquantized partial MatMuls.
            from executorch.examples.qualcomm.oss_scripts.moe.model.split_attention import (
                split_attention_core,
            )

            split_attention_core(module, inputs)
        return module, inputs
    if len(inputs) != 3 or inputs[0].shape[1] != 128:
        raise ValueError("Expected an AR128 GDN or full-attention front")

    graph = module.graph
    placeholders = [node for node in graph.nodes if node.op == "placeholder"]
    with graph.inserting_after(placeholders[-1]):
        index = graph.placeholder("last_token_index")
    index.meta["val"] = torch.tensor([127], dtype=torch.int32)
    inputs = (*inputs, index.meta["val"])
    originals = {
        node: tuple(node.args)
        for node in graph.nodes
        if str(node.target) == "qnn_custom_ops.deltanet_core.default"
    }
    mask_chunk_prefill_padding(module)
    for core in list(graph.nodes):
        if str(core.target) != "qnn_custom_ops.deltanet_core.default":
            continue
        if gdn_core == "custom_u16":
            masked_u16_core(graph, core, originals[core], index)
            continue
        if gdn_core == "custom_fp32":
            masked_fp32_core(graph, core, originals[core], index)
            continue
        active = core.args[4].args[1]
        with graph.inserting_before(core):
            active = graph.call_function(torch.ops.aten.unsqueeze.default, (active, -1))
            masked = []
            for value in core.args[:3]:
                half = graph.call_function(
                    torch.ops.aten._to_copy.default, (value,), {"dtype": torch.float16}
                )
                masked.append(
                    graph.call_function(torch.ops.aten.mul.Tensor, (half, active))
                )
        core.args = (*masked, *core.args[3:])
        if gdn_core == "custom_fp16":
            with graph.inserting_before(core):
                state = graph.call_function(
                    torch.ops.aten._to_copy.default,
                    (core.args[5],),
                    {"dtype": torch.float16},
                )
            core.args = (*core.args[:5], state)
            for output in list(core.users):
                with graph.inserting_after(output):
                    restored = graph.call_function(
                        torch.ops.aten._to_copy.default,
                        (output,),
                        {"dtype": torch.float32},
                    )
                output.replace_all_uses_with(restored)
                restored.args = (output,)
    graph.eliminate_dead_code()
    graph.lint()
    module.recompile()
    FakeTensorProp(module, mode=FakeTensorMode(allow_non_fake_inputs=True)).propagate(
        *inputs
    )
    if gdn_core in ("chunked_mixed_fp16", "chunk128_hmx"):
        replace_prefill_cores(module, fused_chunk=gdn_core == "chunk128_hmx")
    module = torch.export.export(module, inputs, strict=False).module()
    remove_export_checks(module)
    return module, inputs
