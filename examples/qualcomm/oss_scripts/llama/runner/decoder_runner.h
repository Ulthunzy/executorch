/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <atomic>

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/repetition_penalty.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/utils.h>
#include <executorch/extension/llm/sampler/sampler.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

namespace example {
class DecoderRunner {
 public:
  DecoderRunner(
      executorch::extension::Module* module,
      int32_t vocab_size,
      float temperature,
      float repetition_penalty = 1.0f,
      size_t repeat_last_n = RepetitionPenalty::kDefaultWindowSize);

  void begin_turn(const std::vector<uint64_t>& prompt, bool reset) {
    repetition_penalty_.begin_turn(prompt, reset);
  }

  void record_token(uint64_t token) {
    repetition_penalty_.record_token(token);
  }

  void set_stop_flag(const std::atomic<bool>* flag) {
    stop_requested_ = flag;
  }

  bool is_stop_requested() const {
    return stop_requested_ && stop_requested_->load(std::memory_order_relaxed);
  }

  executorch::runtime::Error set_temperature(float temperature);
  /**
   * Run LLM text decoder with inputs to generate next token.
   * @param inputs The inputs to the LLM Module.
   * @return The output of the LLM Module. This will be a tensor of logits.
   */
  executorch::runtime::Result<executorch::aten::Tensor> step(
      const std::string& method_name,
      std::vector<executorch::runtime::EValue>& inputs);

  /**
   * Once KV Cache output data pointer change, need to set
   * the output for specify method name in the module.
   * @return The error code.
   */
  executorch::runtime::Error set_outputs(
      const std::string& method_name,
      std::vector<executorch::aten::Tensor> output_values);

  /**
   * Load the Module for text decode purpose.
   * @return The error code.
   */
  executorch::runtime::Error load(const std::vector<std::string>& method_names);
  /**
   * Check if the required methods in the Module is loaded.
   * @return True if the Module is loaded, false otherwise.
   */
  bool is_method_loaded(const std::vector<std::string>& method_names);

  /**
   * Sample the next token from the logits tensor.
   * @param logits_tensor The logits tensor.
   * @return The next token.
   */
  inline int32_t logits_to_token(
      const executorch::aten::Tensor& logits_tensor,
      int64_t pos) {
    std::byte* logits = logits_tensor.mutable_data_ptr<std::byte>();
    auto num_tokens = logits_tensor.size(1);
    auto vocab_size = logits_tensor.size(2);
    logits_f_.resize(vocab_size);
    std::byte* logits_last = logits;
    // offset to the meaningful logit we want for prefill model.
    executorch::aten::ScalarType logits_dtype = logits_tensor.scalar_type();
    size_t logits_nbytes = getDtypeSize(logits_dtype);
    if (num_tokens > 1) {
      logits_last += pos * vocab_size * logits_nbytes;
    }
    // Preserve legacy sampling when the repetition penalty is disabled.
    for (int i = 0; i < vocab_size; i++) {
      switch (logits_dtype) {
        case executorch::aten::ScalarType::UInt16:
          logits_f_[i] = reinterpret_cast<uint16_t*>(logits_last)[i];
          break;
        case executorch::aten::ScalarType::Byte:
          logits_f_[i] = reinterpret_cast<uint8_t*>(logits_last)[i];
          break;
        case executorch::aten::ScalarType::Float:
          logits_f_[i] = reinterpret_cast<float*>(logits_last)[i];
          break;
        default:
          ET_CHECK_MSG(
              false,
              "The scalar_type %s of logits is not supported",
              executorch::runtime::toString(logits_dtype));
      }
    }
    const bool quantized = logits_dtype != executorch::aten::ScalarType::Float;
    repetition_penalty_.apply(
        logits_f_,
        quantized ? logits_scale_ : 1.0f,
        quantized ? logits_zero_point_ : 0);
    if (quantized && temperature_ > 0 && !repetition_penalty_.enabled()) {
      for (float& logit : logits_f_) {
        logit = (logit - logits_zero_point_) * logits_scale_;
      }
    }
    return sampler_->sample(logits_f_.data());
  }

 protected:
  executorch::extension::Module* module_;
  std::unique_ptr<executorch::extension::llm::Sampler> sampler_;
  RepetitionPenalty repetition_penalty_;
  std::vector<float> logits_f_;
  float logits_scale_{1.0f};
  int64_t logits_zero_point_{0};
  float temperature_{0};
  bool quantized_logits_{false};
  bool has_logits_quantization_{false};
  const std::atomic<bool>* stop_requested_{nullptr};
};
} // namespace example
