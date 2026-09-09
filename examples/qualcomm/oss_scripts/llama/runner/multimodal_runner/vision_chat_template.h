/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/multimodal_runner.h>
#include <executorch/runtime/platform/log.h>
#include <string>
#include <vector>

inline const std::string IMG_TOKEN = "<image>";

/**
 * Special tokens structure for vision modality
 */
struct VisionSpecialTokens {
  std::string image_token;
  std::string global_img;
  std::string fake_wrap_start;
  std::string fake_wrap_end;
};

/**
 * Get special tokens based on model version
 */
inline VisionSpecialTokens get_special_tokens(
    example::VisionLanguageModel model_version) {
  VisionSpecialTokens tokens;
  tokens.image_token = IMG_TOKEN;

  switch (model_version) {
    case example::VisionLanguageModel::kSmolvlm:
      tokens.global_img = "<global-img>";
      tokens.fake_wrap_start = "<fake_token_around_image>";
      tokens.fake_wrap_end = "<fake_token_around_image>";
      break;
    case example::VisionLanguageModel::kInternvl3:
      tokens.global_img = "";
      tokens.fake_wrap_start = "<img>";
      tokens.fake_wrap_end = "</img>";
      break;
    case example::VisionLanguageModel::kQwen2_5_VL:
    case example::VisionLanguageModel::kLincal3_VL:
      tokens.image_token = "<|image_pad|>";
      tokens.global_img = "";
      tokens.fake_wrap_start = "<|vision_start|>";
      tokens.fake_wrap_end = "<|vision_end|>";
      break;
    case example::VisionLanguageModel::kQwen3_5:
    case example::VisionLanguageModel::kLincal3_5:
      tokens.global_img = "";
      tokens.fake_wrap_start = "<|vision_start|>";
      tokens.fake_wrap_end = "<|vision_end|>";
      break;
    default:
      break;
  }

  return tokens;
}

/**
 * Expand image tokens in prompt with model-specific wrapping tokens
 * Replaces each <image> token with the full format including special wrapper
 * tokens
 */
inline std::string expand_image_tokens(
    const std::string& prompt,
    const VisionSpecialTokens& specials) {
  // Keep IMG_TOKEN as a runtime-only dispatch marker for models whose tokenizer
  // uses a different placeholder token, e.g. Qwen-VL's <|image_pad|>.
  const std::string dispatch_marker =
      specials.image_token == IMG_TOKEN ? "" : IMG_TOKEN;
  const std::string image_prompt = specials.fake_wrap_start +
      specials.global_img + specials.image_token + dispatch_marker +
      specials.fake_wrap_end;

  // Replace the user-facing image marker with the model-specific prompt.
  size_t pos = 0;
  std::string expanded = prompt;
  while ((pos = expanded.find(IMG_TOKEN, pos)) != std::string::npos) {
    expanded.replace(pos, IMG_TOKEN.size(), image_prompt);
    pos += image_prompt.size();
  }
  ET_LOG(Info, "Prompt after expanding image token: %s", expanded.c_str());

  return expanded;
}

/**
 * Format prompt based on model version with multimodal token expansion
 */
inline std::string apply_chat_template(
    const std::string& system_prompt,
    const std::string& prompt,
    example::VisionLanguageModel model_version) {
  std::string formatted_prompt;
  VisionSpecialTokens specials = get_special_tokens(model_version);

  switch (model_version) {
    case example::VisionLanguageModel::kSmolvlm: {
      if (!system_prompt.empty()) {
        formatted_prompt.append(
            "<|start_header_id|>system<|end_header_id|>\n\n");
        formatted_prompt.append(system_prompt);
        formatted_prompt.append("<|eot_id|>");
      }
      formatted_prompt.append("<|im_start|>User:");
      formatted_prompt.append(expand_image_tokens(prompt, specials));
      formatted_prompt.append("<end_of_utterance>\nAssistant:");
      break;
    }
    case example::VisionLanguageModel::kInternvl3: {
      if (!system_prompt.empty()) {
        formatted_prompt.append("<|im_start|>system<|im_end|>\n\n");
        formatted_prompt.append(system_prompt);
        formatted_prompt.append("<|im_end|>");
      }
      formatted_prompt.append("<|im_start|>user:\n");
      formatted_prompt.append(expand_image_tokens(prompt, specials));
      formatted_prompt.append("<|im_end|>assistant\n");
      break;
    }
    case example::VisionLanguageModel::kQwen2_5_VL:
    case example::VisionLanguageModel::kLincal3_VL: {
      if (!system_prompt.empty()) {
        formatted_prompt.append("<|im_start|>system\n");
        formatted_prompt.append(system_prompt);
        formatted_prompt.append("<|im_end|>\n");
      }
      formatted_prompt.append("<|im_start|>user\n");
      std::string expanded_prompt = expand_image_tokens(prompt, specials);
      size_t pos = 0;
      while (
          (pos = expanded_prompt.find("<|vision_end|>", pos)) !=
          std::string::npos) {
        pos += std::string("<|vision_end|>").size();
        if (pos < expanded_prompt.size() && expanded_prompt[pos] != '\n') {
          expanded_prompt.insert(pos, "\n");
        }
      }
      formatted_prompt.append(expanded_prompt);
      formatted_prompt.append("<|im_end|>\n");
      formatted_prompt.append("<|im_start|>assistant\n");
      break;
    }
    case example::VisionLanguageModel::kQwen3_5:
    case example::VisionLanguageModel::kLincal3_5: {
      if (!system_prompt.empty()) {
        formatted_prompt.append("<|im_start|>system\n");
        formatted_prompt.append(system_prompt);
        formatted_prompt.append("<|im_end|>\n");
      }
      formatted_prompt.append("<|im_start|>user\n");
      formatted_prompt.append(expand_image_tokens(prompt, specials));
      formatted_prompt.append("<|im_end|>\n");
      formatted_prompt.append("<|im_start|>assistant\n");
      formatted_prompt.append("<think>\n\n</think>\n\n");
      break;
    }
    default:
      ET_CHECK_MSG(false, "unsupported Vision-Language model version");
      break;
  }
  return formatted_prompt;
}
