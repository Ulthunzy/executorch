#include "graph_binding.h"
#include <cstdlib>
#include <dlfcn.h>
#include <stdexcept>

namespace {
struct Api {
  decltype(&qwen35_shared_acquire) acquire;
  decltype(&qwen35_shared_release) release;
  decltype(&qwen35_shared_register) register_io;
  decltype(&qwen35_shared_unregister) unregister_io;
  decltype(&qwen35_shared_context_count) context_count;

  Api() {
    void* library = dlopen("libqwen35_shared.so", RTLD_NOW | RTLD_LOCAL);
    if (!library)
      throw std::runtime_error(dlerror());
    acquire = reinterpret_cast<decltype(acquire)>(
        dlsym(library, "qwen35_shared_acquire"));
    release = reinterpret_cast<decltype(release)>(
        dlsym(library, "qwen35_shared_release"));
    register_io = reinterpret_cast<decltype(register_io)>(
        dlsym(library, "qwen35_shared_register"));
    unregister_io = reinterpret_cast<decltype(unregister_io)>(
        dlsym(library, "qwen35_shared_unregister"));
    context_count = reinterpret_cast<decltype(context_count)>(
        dlsym(library, "qwen35_shared_context_count"));
    if (!acquire || !release || !register_io || !unregister_io ||
        !context_count)
      throw std::runtime_error("shared context API");
  }
};

Api& api() {
  static Api value;
  return value;
}
} // namespace

bool shared_graph_enabled(const std::string& path) {
  return getenv("SHARED_MODEL_ROOT") &&
      (path.find("/layer") != std::string::npos ||
       path.substr(path.find_last_of('/') + 1) == "head.bin");
}

const QnnSystemContext_GraphInfoV1_t* borrow_shared_graph(
    const std::string& path,
    Qnn_ContextHandle_t* context,
    Qnn_GraphHandle_t* graph) {
  std::string directory = getenv("SHARED_MODEL_ROOT"), name;
  const auto start = path.rfind("/layer");
  if (start == std::string::npos) {
    directory += "/head";
  } else {
    size_t used = 0;
    const int layer = std::stoi(path.substr(start + 6), &used);
    if (layer < 0 || layer >= 40 || path[start + 6 + used] != '/')
      throw std::runtime_error("shared layer path");
    const auto file = path.substr(path.find_last_of('/') + 1);
    const std::string method = file == "forward_0.bin" ? "prefill_attention"
        : file == "attention.bin"                      ? "decode"
                                                       : "prefill_router";
    name = "layer" + std::to_string(layer) + "_" + method;
    directory += "/group" + std::to_string(layer / 4);
  }
  const QnnSystemContext_GraphInfoV1_t* metadata = nullptr;
  if (api().acquire(directory.c_str(), name.c_str(), context, graph, &metadata))
    throw std::runtime_error("borrow shared graph: " + name);
  return metadata;
}

void release_shared_graph(Qnn_ContextHandle_t context) {
  api().release(context);
}

int register_shared_io(
    Qnn_ContextHandle_t context,
    const Qnn_MemDescriptor_t* descriptor,
    Qnn_MemHandle_t* handle) {
  return api().register_io(context, descriptor, handle);
}

int unregister_shared_io(Qnn_ContextHandle_t context, Qnn_MemHandle_t handle) {
  return api().unregister_io(context, handle);
}

uint64_t shared_context_count() {
  return getenv("SHARED_MODEL_ROOT") ? api().context_count() : 0;
}
