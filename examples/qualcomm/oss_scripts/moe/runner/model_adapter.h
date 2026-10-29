#pragma once
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "chat/options.h"
#include "qnn/shared/graph_binding.h"

namespace moe {
struct ModelInfo {
  int layers, hidden_size, vocabulary, experts, top_k;
  uint64_t eos, image_pad;
  std::vector<std::pair<std::string, uint64_t>> special_tokens;
};

class ModelAdapter {
 public:
  virtual ~ModelAdapter() = default;
  virtual const ModelInfo& info() const = 0;
  virtual std::optional<SharedGraphBinding> resolve_graph(
      const std::string& path) const = 0;
  virtual std::string prompt(
      const moe_chat::Options&,
      const std::string& text,
      bool first,
      bool ended,
      bool image) const = 0;
};
using AdapterFactory = std::unique_ptr<ModelAdapter> (*)();
void register_model_adapter(const std::string& family, AdapterFactory factory);
std::unique_ptr<ModelAdapter> create_model_adapter(const std::string& family);
void register_qwen35_adapter();
} // namespace moe

std::optional<SharedGraphBinding> resolve_model_graph(const std::string& path);
