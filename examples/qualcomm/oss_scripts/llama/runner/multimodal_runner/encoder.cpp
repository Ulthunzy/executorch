/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/encoder.h>
#include <cstring>
#include <fstream>

using executorch::aten::Tensor;
using executorch::aten::ScalarType;
using executorch::extension::Module;
using executorch::extension::TensorPtr;
using executorch::extension::from_blob;
using executorch::extension::llm::Image;
using executorch::runtime::Error;
using executorch::runtime::MethodMeta;
using executorch::runtime::Result;

namespace example {

EncoderRunner::EncoderRunner(executorch::extension::Module* module)
    : module_(module) {}

bool EncoderRunner::is_method_loaded() const {
  return module_->is_method_loaded(kEncoderForwardName);
}

Error EncoderRunner::load() {
  if (is_method_loaded()) {
    return Error::Ok;
  }

  auto load_result = module_->load_method(kEncoderForwardName);
  if (load_result != Error::Ok) {
    ET_LOG(Error, "Failed to load encoder method");
    return load_result;
  }

  // Get image sequence length from output metadata
  Result<MethodMeta> method_meta = module_->method_meta(kEncoderForwardName);
  if (!method_meta.ok()) {
    ET_LOG(Error, "Failed to get encoder method metadata");
    return method_meta.error();
  }

  return Error::Ok;
}

Result<Tensor> EncoderRunner::encode(TensorPtr& image_tensor) {
  ET_CHECK_MSG(is_method_loaded(), "Encoder method not loaded");

  auto tensor_ptr = image_tensor.get();
  ET_LOG(Info, "Encoding tensor with numel: %zu", tensor_ptr->numel());

  std::vector<executorch::runtime::EValue> encoder_inputs;
  encoder_inputs.emplace_back(*tensor_ptr);

  auto encoder_result = module_->forward(encoder_inputs);
  ET_CHECK_OK_OR_RETURN_ERROR(encoder_result.error());

  encoder_output_ = std::move(encoder_result.get());
  ET_LOG(Info, "Encoder execution completed, got hidden states");

  return encoder_output_[0].toTensor();
}

Result<Tensor> EncoderRunner::encode(const Image& image) {
  ET_CHECK_MSG(is_method_loaded(), "Encoder method not loaded");

  Result<MethodMeta> method_meta = module_->method_meta(kEncoderForwardName);
  ET_CHECK_MSG(method_meta.ok(), "Failed to get encoder method metadata");

  auto input_meta = method_meta->input_tensor_meta(0);
  std::vector<int32_t> input_sizes(
      input_meta->sizes().begin(), input_meta->sizes().end());
  const ScalarType input_dtype = input_meta->scalar_type();
  size_t count = 1;
  for (const auto size : input_meta->sizes())
    count *= size;
  ET_CHECK_OR_RETURN_ERROR(
      (image.is_float() && input_dtype == ScalarType::Float &&
       image.get_float_data().size() == count) ||
          (image.is_uint8() && input_dtype == ScalarType::Byte &&
           image.get_uint8_data().size() == count),
      InvalidArgument,
      "Image buffer does not match encoder input metadata");

  TensorPtr tensor;
  if (input_dtype == ScalarType::Float) {
    ET_CHECK_MSG(
        image.is_float(),
        "Encoder expects float input but received a non-float buffer");
    tensor = from_blob(
        const_cast<float*>(image.get_float_data().data()),
        input_sizes,
        ScalarType::Float);
  } else if (input_dtype == ScalarType::Byte) {
    ET_CHECK_MSG(
        image.is_uint8(),
        "Encoder expects uint8 input but received a non-uint8 buffer");
    tensor = from_blob(
        const_cast<uint8_t*>(image.get_uint8_data().data()),
        input_sizes,
        ScalarType::Byte);
  } else {
    ET_CHECK_MSG(
        false,
        "Unsupported encoder input dtype: %d",
        static_cast<int>(input_dtype));
  }

  return encode(tensor);
}

} // namespace example
