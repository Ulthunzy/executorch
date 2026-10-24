#pragma once

#include "../decode_api.h"
#include "tokenizer.h"
#include "image.h"
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace moe_chat {
struct Options {
  std::string model_root, tokenizer_path, shared_model, image_path;
  std::string prefill_order;
  std::string system_prompt = "你是一个有帮助的人工智能助手。";
  std::vector<std::string> prompts;
  int max_new_tokens = 128, capacity = 1152, cache_experts = 32;
  float temperature = 0;
  bool interactive = false, thinking = false, tokenize_only = false;
  std::string log_path;
};

Options parse_options(int argc, char** argv);
void configure_environment(const Options& options, int prefix_tokens);

class Conversation {
 public:
  explicit Conversation(Options options);
  ~Conversation();
  void prepare();
  int run(void* library);
  void report_error(const std::string& message);
  std::vector<float> initial_image(VisionEncoder* encoder);
  Options options;

 private:
  DeviceTokenizer tokenizer;
  std::unique_ptr<ImageLayout> image_layout;
  std::unique_ptr<RotaryPositions> rotary;
  std::vector<float> first_pixels;
  VisionEncoder* vision = nullptr;
  FILE* console = nullptr;
  std::vector<float> logits;
  std::vector<uint64_t> first_tokens;
  std::vector<int64_t> generated;
  int pending = -1, turns = 0;
  bool ended = false, empty = true;
  int (*open)(const char*, int) = nullptr;
  int (*step)(int64_t, float*, size_t) = nullptr;
  int (*reset)() = nullptr;
  int (*position)() = nullptr;
  void (*close)() = nullptr;
  const char* (*error)() = nullptr;
  int (*set_inputs)(const float*, size_t, const float*, const float*, size_t) =
      nullptr;

  std::vector<uint64_t> encode(const std::string& text);
  std::string prompt(const std::string& text, bool first, bool image = false)
      const;
  std::vector<uint64_t>
  prompt_tokens(const std::string& text, bool first, bool image);
  void consume(int64_t token);
  void generate();
  bool turn(const std::string& text);
};
} // namespace moe_chat
