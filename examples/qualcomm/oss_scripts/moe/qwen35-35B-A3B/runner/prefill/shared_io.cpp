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

SharedIo::SharedIo(Runtime& runtime) {
  auto allocate = [&](size_t size) {
    auto buffer = std::make_shared<Buffer>(runtime, size);
    bytes += buffer->size;
    return buffer;
  };
  hidden = allocate(128 * 2048 * 4);
  residual = allocate(128 * 2048 * 4);
  ids = allocate(128 * 8 * 4);
  scores = allocate(128 * 8 * 4);
  shared = allocate(128 * 2048 * 4);
  cos = allocate(128 * 32 * 4);
  sin = allocate(128 * 32 * 4);
  mask = allocate(128 * 1024 * 4);
}

void SharedIo::bind_layer(
    Attention& a,
    Graph& r,
    const std::shared_ptr<Buffer>& chain) {
  bind(a.graph, true, a.input[0], chain);
  a.hidden_alias = chain;
  // Recurrent and KV buffers remain private to each layer for the full session.
  if (!a.linear) {
    bind(a.graph, true, a.input[1], cos);
    bind(a.graph, true, a.input[2], sin);
    bind(a.graph, true, a.input[3], mask);
  }
  bind(a.graph, false, a.output[0], hidden);
  bind(a.graph, false, a.output[1], residual);
  bind(r, true, r.index(true, 0), hidden);
  bind(r, false, r.index(false, 0), ids);
  bind(r, false, r.index(false, 1), scores);
  bind(r, false, r.index(false, 2), shared);
}
