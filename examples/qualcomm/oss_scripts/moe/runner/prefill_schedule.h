#pragma once

#include <stdexcept>

inline int prefill_plan_tokens(int prompt_tokens, int workspace_tokens) {
  if (prompt_tokens < 128 || prompt_tokens % 128 || workspace_tokens < 128 ||
      workspace_tokens > 1024 || workspace_tokens % 128)
    throw std::runtime_error("invalid AR128 prefill workspace");
  return prompt_tokens <= workspace_tokens ? prompt_tokens : 128;
}
