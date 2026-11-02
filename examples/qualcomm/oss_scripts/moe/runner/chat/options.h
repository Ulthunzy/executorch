#pragma once

#include "settings.h"
#include "../model_package.h"
#include <memory>
#include <string>
#include <vector>

namespace moe_chat {
struct Options {
  std::string model_family = "qwen35";
  std::string model_root, tokenizer_path, shared_model, image_path;
  std::string prefill_order;
  std::string model_config;
  std::string package_cache, library_directory;
  std::shared_ptr<ModelPackageLock> package_lock;
  RuntimeSettings runtime;
  std::string system_prompt = "你是一个有帮助的人工智能助手。";
  std::vector<std::string> prompts;
  int max_new_tokens = 128, capacity = 1152, cache_experts = 32;
  int prefill_workspace_tokens = 1024;
  int prompt_tokens = 0;
  int prefill_io_workers = 8;
  float temperature = 0;
  bool interactive = false, thinking = true, tokenize_only = false;
  bool ignore_eos = false;
  bool profile = false;
  bool help_requested = false;
  bool stream = true, vision_enabled = true;
  bool preload = false;
  std::string log_path;
};

} // namespace moe_chat

namespace moe_chat {
Options parse_options(int argc, char** argv);
void configure_environment(const Options&, int prefix_tokens, int valid_tokens = -1);
}
