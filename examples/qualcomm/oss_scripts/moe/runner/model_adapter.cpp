#include "model_adapter.h"
#include <cstdlib>
#include <map>
#include <mutex>
#include <stdexcept>

namespace moe {
namespace {
std::map<std::string, AdapterFactory> adapters;
std::mutex adapter_mutex;
} // namespace
void register_model_adapter(const std::string& family, AdapterFactory factory) {
  if (family.empty() || !factory)
    throw std::invalid_argument(
        "A model family and adapter factory are required");
  std::lock_guard<std::mutex> lock(adapter_mutex);
  const auto inserted = adapters.emplace(family, factory);
  if (!inserted.second && inserted.first->second != factory)
    throw std::invalid_argument("Model adapter already registered: " + family);
}
std::unique_ptr<ModelAdapter> create_model_adapter(const std::string& family) {
  if (family == "qwen35")
    register_qwen35_adapter();
  AdapterFactory factory;
  {
    std::lock_guard<std::mutex> lock(adapter_mutex);
    const auto found = adapters.find(family);
    if (found == adapters.end())
      throw std::invalid_argument("Unsupported model adapter: " + family);
    factory = found->second;
  }
  auto result = factory();
  if (!result)
    throw std::runtime_error("Model adapter factory returned null");
  return result;
}
} // namespace moe

std::optional<SharedGraphBinding> resolve_model_graph(const std::string& path) {
  const char* family = getenv("MOE_MODEL_FAMILY");
  auto model = moe::create_model_adapter(family && *family ? family : "qwen35");
  return model->resolve_graph(path);
}
