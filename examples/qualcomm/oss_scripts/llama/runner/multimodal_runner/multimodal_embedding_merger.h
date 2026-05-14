/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/utils.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>

#include <memory>
#include <vector>

namespace example {

enum class EmbeddingType { kText, kImage };

class MultimodalEmbeddingMerger {
 public:
  explicit MultimodalEmbeddingMerger(int32_t embedding_dim);

  /**
   * @brief Reset the merger state for a new sequence
   */
  void reset();

  // Append embeddings
  void add_embeddings(const TensorStruct<float>& embedding);
  void add_embeddings(const executorch::aten::Tensor& embedding);
  void add_text_embeddings(const TensorStruct<float>& text_embeddings);
  void add_image_embeddings(const executorch::aten::Tensor& image_embeddings);

  TensorStruct<float> get_merged_embeddings();
  TensorStruct<float> merge(
      const std::vector<uint64_t>& input_ids,
      uint64_t image_token_id);
  int32_t get_total_tokens() const noexcept {
    return total_tokens_;
  }

 private:
  // Validates shape [batch size, num tokens, dim] and appends data.
  void append_data(
      const float* data,
      ssize_t ndim,
      const executorch::aten::SizesType* sizes);
  void add_typed_embeddings(
      const executorch::aten::Tensor& embeddings,
      const float* data,
      EmbeddingType type);

  int32_t embedding_dim_;
  int32_t total_tokens_{0};

  std::vector<float> embeddings_;
  std::vector<executorch::aten::TensorImpl::SizesType> embedding_sizes_;

  std::vector<std::vector<float>> text_embedding_buffers_;
  std::vector<int64_t> text_embedding_token_counts_;
  std::vector<std::vector<float>> image_embedding_buffers_;
  std::vector<int64_t> image_embedding_token_counts_;
  std::vector<float> merged_embedding_buffer_;
  std::vector<executorch::aten::TensorImpl::SizesType> merged_embedding_sizes_;
};

} // namespace example
