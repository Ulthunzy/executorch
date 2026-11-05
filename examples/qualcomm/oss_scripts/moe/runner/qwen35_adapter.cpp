#include <charconv>
#include <cstdlib>
#include <stdexcept>
#include <string_view>
#include "shared_graphs.h"

std::optional<SharedGraphBinding> resolve_qwen35_graph(
    const std::string& path) {
  const char* root = getenv("SHARED_MODEL_ROOT");
  if (!root || !*root)
    return std::nullopt;
  const std::string_view source(path);
  const auto slash = source.find_last_of('/');
  const auto file = source.substr(slash == source.npos ? 0 : slash + 1);
  if (file != "head.bin" && file != "forward_0.bin" &&
      file != "attention.bin" && file != "moe-front.bin" &&
      file != "vision.bin")
    return std::nullopt;
  std::string directory(root);
  while (!directory.empty() && directory.back() == '/')
    directory.pop_back();
  if (file == "head.bin")
    return SharedGraphBinding{directory + "/head", ""};
  if (file == "vision.bin")
    return SharedGraphBinding{directory + "/vision", ""};
  if (slash == source.npos || slash == 0)
    return std::nullopt;
  const auto previous = source.find_last_of('/', slash - 1);
  const auto parent = source.substr(
      previous == source.npos ? 0 : previous + 1,
      slash - (previous == source.npos ? 0 : previous + 1));
  if (parent.substr(0, 5) != "layer")
    return std::nullopt;
  int layer = -1;
  const auto parsed =
      std::from_chars(parent.data() + 5, parent.data() + parent.size(), layer);
  if (parsed.ec != std::errc{} || parsed.ptr != parent.data() + parent.size() ||
      layer < 0 || layer >= 40)
    throw std::runtime_error("shared Qwen layer path");
  const std::string_view method = file == "forward_0.bin" ? "prefill_attention"
      : file == "attention.bin"                           ? "decode"
                                                          : "prefill_router";
  directory += "/group" + std::to_string(layer / 4);
  return SharedGraphBinding{
      std::move(directory),
      "layer" + std::to_string(layer) + "_" + std::string(method)};
}

#include "model_adapter.h"
namespace moe {
namespace {
class Qwen35Adapter final : public ModelAdapter {
 public:
  std::optional<SharedGraphBinding> resolve_graph(
      const std::string& path) const override {
    return resolve_qwen35_graph(path);
  }
  const ModelInfo& info() const override {
    static const ModelInfo value{
        40,
        2048,
        248320,
        256,
        8,
        248046,
        248056,
        {{"<|im_start|>", 248045},
         {"<|im_end|>", 248046},
         {"<|endoftext|>", 248044},
         {"<think>", 248068},
         {"</think>", 248069}}};
    return value;
  }
  std::string prompt(
      const moe_chat::Options& options,
      const std::string& text,
      bool first,
      bool ended,
      bool image) const override {
    std::string result;
    if (first)
      result = "<|im_start|>system\n" + options.system_prompt + "<|im_end|>\n";
    else
      result = ended ? "\n" : "<|im_end|>\n";
    result += "<|im_start|>user\n";
    if (image)
      result += "<|vision_start|><|image_pad|><|vision_end|>";
    result += text + "<|im_end|>\n<|im_start|>assistant\n";
    result += options.thinking ? "<think>\n" : "<think>\n\n</think>\n\n";
    return result;
  }
};
std::unique_ptr<ModelAdapter> make_qwen35() {
  return std::make_unique<Qwen35Adapter>();
}
} // namespace
void register_qwen35_adapter() {
  register_model_adapter("qwen35", make_qwen35);
}
} // namespace moe
