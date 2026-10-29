#include "graph_binding.h"
#include <cstdlib>
#include <stdexcept>

const QnnSystemContext_GraphInfoV1_t* borrow_shared_graph(
    const SharedGraphBinding& binding,
    Qnn_ContextHandle_t* context,
    Qnn_GraphHandle_t* graph) {
  const QnnSystemContext_GraphInfoV1_t* metadata = nullptr;
  if (qwen35_shared_acquire(binding.directory.c_str(), binding.name.c_str(), context, graph, &metadata))
    throw std::runtime_error("borrow shared graph: " + binding.name);
  return metadata;
}

void release_shared_graph(Qnn_ContextHandle_t context) {
  qwen35_shared_release(context);
}

int register_shared_io(
    Qnn_ContextHandle_t context,
    const Qnn_MemDescriptor_t* descriptor,
    Qnn_MemHandle_t* handle) {
  return qwen35_shared_register(context, descriptor, handle);
}

int unregister_shared_io(Qnn_ContextHandle_t context, Qnn_MemHandle_t handle) {
  return qwen35_shared_unregister(context, handle);
}

uint64_t shared_context_count() {
  return getenv("SHARED_MODEL_ROOT") ? qwen35_shared_context_count() : 0;
}
