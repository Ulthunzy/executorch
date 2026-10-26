#include "shared_io.h"

void SharedIo::bind(
    Graph& graph,
    bool input,
    int index,
    const std::shared_ptr<Buffer>& buffer) {
  auto& tensors = input ? graph.inputs : graph.outputs;
  auto& buffers = input ? graph.inbuf : graph.outbuf;
  if (Graph::size(tensors[index]) > buffer->size)
    throw std::runtime_error("shared IO capacity");
  buffers[index] = buffer;
  graph.bind(tensors[index], buffer);
}

SharedIo::SharedIo(
    Runtime& runtime,
    std::vector<std::unique_ptr<Attention>>& attention,
    std::vector<std::unique_ptr<Graph>>& router,
    const std::shared_ptr<Buffer>& chain) {
  auto allocate = [&](size_t size) {
    auto buffer = std::make_shared<Buffer>(runtime, size);
    bytes += buffer->size;
    return buffer;
  };
  hidden = allocate(2048 * 4);
  residual = allocate(2048 * 4);
  ids = allocate(8 * 4);
  scores = allocate(8 * 4);
  shared = allocate(2048 * 4);
  cos = allocate(32 * 4);
  sin = allocate(32 * 4);
  size_t mask_bytes = 0;
  for (const auto& a : attention)
    if (!a->linear)
      mask_bytes =
          std::max(mask_bytes, Graph::size(a->graph.inputs[a->input[3]]));
  if (mask_bytes)
    mask = allocate(mask_bytes);
  for (size_t i = 0; i < attention.size(); i++) {
    auto& a = *attention[i];
    bind(a.graph, true, a.input[0], chain);
    a.alias = chain;
    // Recurrent and KV buffers remain private to each layer for the full
    // session.
    if (!a.linear) {
      bind(a.graph, true, a.input[1], cos);
      bind(a.graph, true, a.input[2], sin);
      bind(a.graph, true, a.input[3], mask);
    }
    bind(a.graph, false, a.output[0], hidden);
    bind(a.graph, false, a.output[1], residual);
    if (a.output.size() == 7) {
      bind(a.graph, false, a.output[4], ids);
      bind(a.graph, false, a.output[5], scores);
      bind(a.graph, false, a.output[6], shared);
    }
    if (!router[i])
      continue;
    auto& r = *router[i];
    bind(r, true, r.index(true, 0), hidden);
    bind(r, false, r.index(false, 0), ids);
    bind(r, false, r.index(false, 1), scores);
    bind(r, false, r.index(false, 2), shared);
  }
}
