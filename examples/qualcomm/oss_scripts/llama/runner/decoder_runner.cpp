/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Given inputs, run a text decoder and return logits.

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/decoder_runner.h>

#include <cmath>
#include <ctime>
using executorch::aten::Tensor;
using executorch::extension::Module;
using executorch::extension::llm::Sampler;
using executorch::llm::kTopp;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::Result;

namespace example {

DecoderRunner::DecoderRunner(
    Module* module,
    int32_t vocab_size,
    float temperature,
    float repetition_penalty)
    : module_(module),
      sampler_(std::make_unique<Sampler>(
          vocab_size,
          temperature,
          kTopp,
          static_cast<unsigned long long>(std::time(nullptr)))),
      repetition_penalty_(repetition_penalty) {
  ET_CHECK_MSG(
      std::isfinite(repetition_penalty) && repetition_penalty >= 1.0f,
      "repetition_penalty must be finite and >= 1.0");
}

Error DecoderRunner::set_outputs(
    const std::string& method_name,
    std::vector<executorch::aten::Tensor> output_values) {
  for (size_t i = 0; i < output_values.size(); ++i) {
    ET_CHECK_OK_OR_RETURN_ERROR(
        module_->set_output(method_name, output_values[i], i));
  }
  return Error::Ok;
}

Error DecoderRunner::load(const std::vector<std::string>& method_names) {
  for (const std::string& method_name : method_names) {
    ET_CHECK_OK_OR_RETURN_ERROR(module_->load_method(method_name));
  }
  if (repetition_penalty_.enabled()) {
    for (const std::string& method_name : method_names) {
      ET_ASSIGN_OR_RETURN(meta, module_->method_meta(method_name));
      ET_ASSIGN_OR_RETURN(output_meta, meta.output_tensor_meta(0));
      if (output_meta.scalar_type() == executorch::aten::ScalarType::Float) {
        continue;
      }
      ET_ASSIGN_OR_RETURN(scale, module_->get("get_logits_scale"));
      ET_ASSIGN_OR_RETURN(zero_point, module_->get("get_logits_zero_point"));
      ET_CHECK_OR_RETURN_ERROR(
          scale.isDouble() && zero_point.isInt(),
          InvalidArgument,
          "Repetition penalty requires valid logits quantization metadata");
      logits_scale_ = static_cast<float>(scale.toDouble());
      logits_zero_point_ = zero_point.toInt();
      ET_CHECK_OR_RETURN_ERROR(
          std::isfinite(logits_scale_) && logits_scale_ > 0,
          InvalidArgument,
          "Invalid logits scale for repetition penalty");
      ET_LOG(
          Info,
          "Repetition penalty: dequantizing logits (scale=%g, zero_point=%lld)",
          logits_scale_,
          static_cast<long long>(logits_zero_point_));
      break;
    }
  }
  return Error::Ok;
}

bool DecoderRunner::is_method_loaded(
    const std::vector<std::string>& method_names) {
  bool method_loaded = true;
  for (const std::string& method_name : method_names) {
    method_loaded &= module_->is_method_loaded(method_name);
  }
  return method_loaded;
}

// This function is functional, meaning it shouldn't modify any state of the
// input. It should be safe to call multiple times with the same inputs. The
// outer loop (call site) is responsible for managing state.
Result<Tensor> DecoderRunner::step(
    const std::string& method_name,
    std::vector<EValue>& inputs) {
  Result<std::vector<EValue>> outputs_res =
      module_->execute(method_name, inputs);
  ET_CHECK_OK_OR_RETURN_ERROR(outputs_res.error());
  ET_CHECK_MSG(
      outputs_res.get()[0].isTensor(),
      "Non Tensor Output returned from executing LLM");

  // Return the logits tensor
  return outputs_res.get()[0].toTensor();
}

} // namespace example
