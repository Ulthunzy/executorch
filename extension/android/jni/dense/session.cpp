#include "session.h"

#include <executorch/backends/qualcomm/runtime/QnnExecuTorch.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/multimodal_runner.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/qwen_image.h>
#include <executorch/extension/llm/runner/stats.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/platform/runtime.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace executorch::android::dense {
namespace {
using extension::Module;
using extension::llm::MultimodalInput;
using runtime::Error;
std::recursive_mutex package_directory_mutex;

void check(Error status, const char* operation) {
  if (status != Error::Ok)
    throw std::runtime_error(
        std::string(operation) + ": ExecuTorch error " +
        std::to_string(static_cast<unsigned>(status)));
}

// This backend option is process-wide; restore it after loading this session.
class PackageDirectory {
 public:
  explicit PackageDirectory(const std::string& directory) {
    if (!lock_.owns_lock())
      throw std::runtime_error(
          "Another dense QNN call is active in this process");
    if (directory.size() >= runtime::kMaxOptionValueLength)
      throw std::invalid_argument("Native library path is too long");
    std::strcpy(previous_.key, QNN_RUNTIME_OP_PACKAGE_DIRECTORY);
    previous_.value = std::array<char, runtime::kMaxOptionValueLength>{};
    check(
        runtime::get_option(QNN_BACKEND, {&previous_, 1}),
        "Get op-package path");
    const auto* old = std::getenv("ADSP_LIBRARY_PATH");
    had_adsp_path_ = old != nullptr;
    if (old)
      adsp_path_ = old;
    const std::string search = directory + (old ? ";" + adsp_path_ : "");
    runtime::BackendOption option = previous_;
    std::array<char, runtime::kMaxOptionValueLength> path{};
    std::memcpy(path.data(), directory.data(), directory.size());
    option.value = path;
    check(
        runtime::set_option(QNN_BACKEND, {&option, 1}), "Set op-package path");
    if (setenv("ADSP_LIBRARY_PATH", search.c_str(), 1) != 0) {
      const auto status = runtime::set_option(QNN_BACKEND, {&previous_, 1});
      (void)status;
      throw std::runtime_error("Cannot set DSP library search path");
    }
  }
  ~PackageDirectory() {
    const auto status = runtime::set_option(QNN_BACKEND, {&previous_, 1});
    (void)status;
    if (had_adsp_path_)
      setenv("ADSP_LIBRARY_PATH", adsp_path_.c_str(), 1);
    else
      unsetenv("ADSP_LIBRARY_PATH");
  }

 private:
  std::unique_lock<std::recursive_mutex> lock_{
      package_directory_mutex,
      std::try_to_lock};
  runtime::BackendOption previous_;
  std::string adsp_path_;
  bool had_adsp_path_ = false;
};

std::unique_ptr<Module> module(const std::string& path) {
  if (!std::ifstream(path).good())
    throw std::invalid_argument("Cannot read model file: " + path);
  return std::make_unique<Module>(
      path, Module::LoadMode::MmapUseMlockIgnoreErrors);
}
} // namespace

Session::Session(Config config) : config_(std::move(config)) {
  runtime::runtime_init();
  if (config_.image_height != config_.image_width ||
      config_.image_height <= 0 || config_.image_height > 4096 ||
      config_.image_height % 32)
    throw std::invalid_argument(
        "This dense runner requires a square image grid divisible by 32");
  for (const char* name :
       {"libQnnDeltaNetOpPackage.so", "libQnnDeltaNetOpPackage_HTP.so"}) {
    if (!std::ifstream(config_.libraries + "/" + name).good())
      throw std::invalid_argument(
          "Missing native op-package: " + config_.libraries + "/" + name);
  }
  // Do not mutate PTEs or require the /data/local/tmp paths embedded at export.
  PackageDirectory packages(config_.libraries);
  auto decoder = module(config_.decoder);
  check(decoder->load(), "Read decoder");
  auto names = decoder->method_names();
  if (!names.ok() || !names->count("kv_forward") ||
      !names->count("prefill_forward"))
    throw std::invalid_argument("Dense JNI requires a Qwen3.5 hybrid PTE");
  auto metadata = decoder->method_meta("prefill_forward");
  if (!metadata.ok())
    throw std::invalid_argument("Cannot read prefill metadata");
  auto embedding_meta = metadata->input_tensor_meta(0);
  auto max_past = decoder->get("get_prefill_max_past");
  if (!embedding_meta.ok() || !max_past.ok() || !max_past->isInt())
    throw std::invalid_argument("Missing prefill capacity metadata");
  const auto shape = embedding_meta->sizes();
  if (embedding_meta->scalar_type() != aten::ScalarType::Float ||
      shape.size() != 3 || shape[0] != 1 || shape[1] <= 0 || shape[2] <= 0)
    throw std::invalid_argument("Decoder input must be FP32 [1, AR, hidden]");
  const int64_t past = max_past->toInt();
  if (past < 0 || past > std::numeric_limits<int32_t>::max() - shape[1])
    throw std::invalid_argument("Invalid prefill capacity metadata");
  const int32_t capacity = static_cast<int32_t>(past + shape[1]);
  if (config_.context_length == 0)
    config_.context_length = capacity;
  if (config_.context_length < 2 || config_.context_length > capacity)
    throw std::invalid_argument(
        "Requested context exceeds compiled prefill capacity");
  runner_ = std::make_unique<example::QNNMultimodalRunner>(
      module(config_.encoder),
      module(config_.embedding),
      std::move(decoder),
      "qwen3_5",
      config_.tokenizer,
      "",
      "",
      0.0f,
      example::QNNMultimodalRunner::kHybrid,
      true);
}

Session::~Session() = default;
void Session::prepare() {
  runner_->prepare_generation();
}

std::string Session::generate(
    const std::string& prompt,
    const std::string& image,
    int count,
    float temperature,
    bool thinking,
    bool ignore_eos,
    const std::function<void(const std::string&)>& callback) {
  if (prompt.empty() || count <= 0 || !std::isfinite(temperature) ||
      temperature < 0)
    throw std::invalid_argument(
        "Prompt, token budget or temperature is invalid");
  PackageDirectory packages(config_.libraries);
  std::vector<MultimodalInput> inputs;
  std::string prefix;
  if (!started_ && !config_.system.empty())
    prefix = "<|im_start|>system\n" + config_.system + "<|im_end|>\n";
  // An interrupted/length-limited assistant still needs its ChatML boundary.
  if (started_)
    prefix += ended_with_eos_ ? "\n" : "<|im_end|>\n";
  prefix += "<|im_start|>user\n";
  if (!image.empty()) {
    auto pixels = example::qwen_image_patches(
        image, config_.image_height, config_.image_width);
    auto meta = runner_->get_encoder_method_meta();
    if (!meta.ok())
      check(meta.error(), "Load vision encoder");
    auto tensor_meta = meta->input_tensor_meta(0);
    if (!tensor_meta.ok() ||
        tensor_meta->scalar_type() != aten::ScalarType::Float ||
        tensor_meta->sizes().size() != 3 || tensor_meta->sizes()[0] != 1 ||
        tensor_meta->sizes()[1] !=
            config_.image_height / 16 * (config_.image_width / 16) ||
        tensor_meta->sizes()[2] != 1536)
      throw std::invalid_argument(
          "Image dimensions do not match the encoder PTE");
    inputs.emplace_back(prefix + "<|vision_start|>");
    inputs.emplace_back(extension::llm::Image(
        std::move(pixels),
        tensor_meta->sizes()[2],
        tensor_meta->sizes()[1],
        1));
    prefix = "<|vision_end|>";
  }
  prefix += prompt + "<|im_end|>\n<|im_start|>assistant\n";
  prefix += thinking ? "<think>\n" : "<think>\n\n</think>\n\n";
  inputs.emplace_back(prefix);
  extension::llm::GenerationConfig generation;
  generation.echo = false;
  generation.max_new_tokens = count;
  generation.seq_len = config_.context_length;
  generation.temperature = temperature;
  generation.ignore_eos = ignore_eos;
  std::string stats;
  bool first_piece = true;
  check(
      runner_->generate(
          inputs,
          generation,
          [&](const std::string& piece) {
            if (piece.empty() || piece == "<|im_end|>" ||
                piece == "<|endoftext|>" || piece == "<|end_of_text|>")
              return;
            if (callback) {
              if (first_piece && thinking)
                callback("<think>\n");
              callback(piece);
            }
            first_piece = false;
          },
          [&](const llm::Stats& value) {
            stats = extension::llm::stats_to_json_string(value);
          }),
      "Dense generation (resetContext is required after interrupted prefill)");
  started_ = true;
  ended_with_eos_ = runner_->reached_eos();
  if (!stats.empty()) {
    stats.pop_back();
    stats +=
        ",\"global_kv_tokens\":" + std::to_string(runner_->context_position()) +
        ",\"context_length\":" + std::to_string(config_.context_length) +
        ",\"pending_tokens\":1,\"enable_thinking\":" +
        (thinking ? "true" : "false") + "}";
  }
  return stats;
}

void Session::stop() {
  runner_->stop();
}
void Session::reset() {
  runner_->reset();
  started_ = false;
  ended_with_eos_ = false;
}
} // namespace executorch::android::dense
