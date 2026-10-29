#pragma once

#include "../decode_api.h"
#include "../model_adapter.h"
#include "tokenizer.h"
#include "options.h"
#include "image.h"
#include "client.h"
#include "model_session.h"
#include <executorch/extension/llm/runner/stats.h>
#include "../model_package.h"
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace moe_chat {

void register_conversation(const std::string& family);

class Conversation final : public ModelSession {
 public:
  explicit Conversation(Options options, Client* client = nullptr);
  ~Conversation() override;
  void prepare() override;
  int run() override;
  void report_error(const std::string& message) override;
  std::vector<float> initial_image(VisionEncoder* encoder) override;
  void initial_prefill_complete(double milliseconds) override;

 private:
  std::unique_ptr<moe::ModelAdapter> model;
  DeviceTokenizer tokenizer;
  std::unique_ptr<ImageLayout> image_layout;
  std::unique_ptr<RotaryPositions> rotary;
  std::vector<float> first_pixels;
  VisionEncoder* vision = nullptr;
  FILE* console = nullptr;
  std::vector<float> logits;
  std::vector<uint64_t> first_tokens;
  uint64_t last_input_token = 0;
  std::vector<int64_t> generated;
  int pending = -1, turns = 0;
  bool ended = false, empty = true;
  Client* client = nullptr;
  bool needs_reset = false;
  executorch::extension::llm::Stats stats;
  double initial_prompt_ms = 0;
  std::string stop_reason;
  int (*open)(const char*, int) = nullptr;
  int (*step)(int64_t, float*, size_t) = nullptr;
  int (*reset)() = nullptr;
  int (*position)() = nullptr;
  void (*close)() = nullptr;
  const char* (*error)() = nullptr;
  int (*export_state)(Qwen35SessionState*) = nullptr;
  int (*import_state)(const Qwen35SessionState*) = nullptr;
  int (*set_inputs)(const float*, size_t, const float*, const float*, size_t) =
      nullptr;

  std::vector<uint64_t> encode(const std::string& text);
  std::string prompt(const std::string& text, bool first, bool image = false)
      const;
  std::vector<uint64_t>
  prompt_tokens(const std::string& text, bool first, bool image);
  void consume(int64_t token);
  bool prefill(
      const std::vector<uint64_t>& ids,
      const std::vector<float>& features);
  void generate();
  bool turn(std::string text);
  void reset_context();
  void client_turn(const Request& request);
};
} // namespace moe_chat
