/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/multimodal_embedding_merger.h>
#include <executorch/runtime/platform/log.h>

#include <cstring>
#include <utility>

namespace example {

MultimodalEmbeddingMerger::MultimodalEmbeddingMerger(int32_t embedding_dim)
    : embedding_dim_(embedding_dim) {
  ET_CHECK_MSG(embedding_dim_ > 0, "Embedding dimension must be positive");
}

void MultimodalEmbeddingMerger::reset() {
  embeddings_.clear();
  text_embedding_buffers_.clear();
  text_embedding_token_counts_.clear();
  image_embedding_buffers_.clear();
  image_embedding_token_counts_.clear();
  merged_embedding_buffer_.clear();
  total_tokens_ = 0;
}

void MultimodalEmbeddingMerger::append_data(
    const float* data,
    ssize_t ndim,
    const executorch::aten::SizesType* sizes) {
  ET_CHECK_MSG(ndim == 3, "Embeddings must be a 3D tensor");
  ET_CHECK_MSG(sizes[0] == 1, "Batch size must be 1");
  const int32_t num_tokens = static_cast<int32_t>(sizes[1]);
  const int32_t dim = static_cast<int32_t>(sizes[2]);
  ET_CHECK_MSG(
      dim == embedding_dim_,
      "Embedding dimension mismatch: expected %d, got %d",
      embedding_dim_,
      dim);
  const size_t num_elements =
      static_cast<size_t>(num_tokens) * static_cast<size_t>(embedding_dim_);
  embeddings_.insert(embeddings_.end(), data, data + num_elements);
  total_tokens_ += num_tokens;
}

void MultimodalEmbeddingMerger::add_typed_embeddings(
    const executorch::aten::Tensor& embeddings,
    const float* data,
    EmbeddingType type) {
  ET_CHECK_MSG(embeddings.dim() == 3, "Embeddings must be a 3D tensor");
  const size_t batch_size = static_cast<size_t>(embeddings.sizes()[0]);
  const size_t num_tokens = static_cast<size_t>(embeddings.sizes()[1]);
  const size_t dim = static_cast<size_t>(embeddings.sizes()[2]);

  ET_CHECK_MSG(batch_size == 1, "Batch size must be 1");
  ET_CHECK_MSG(
      dim == static_cast<size_t>(embedding_dim_),
      "Embedding dimension mismatch: expected %zu, got %zu",
      static_cast<size_t>(embedding_dim_),
      dim);

  const size_t num_elements = num_tokens * dim;
  std::vector<float> buffer(data, data + num_elements);
  if (type == EmbeddingType::kText) {
    text_embedding_buffers_.emplace_back(std::move(buffer));
    text_embedding_token_counts_.push_back(static_cast<int64_t>(num_tokens));
  } else {
    image_embedding_buffers_.emplace_back(std::move(buffer));
    image_embedding_token_counts_.push_back(static_cast<int64_t>(num_tokens));
  }

  ET_LOG(
      Info,
      "Added %s embeddings: num_tokens=%zu",
      type == EmbeddingType::kText ? "text" : "image",
      num_tokens);
}

void MultimodalEmbeddingMerger::add_embeddings(
    const TensorStruct<float>& embedding) {
  ET_CHECK_MSG(embedding.tensor != nullptr, "Embedding tensor cannot be null");
  ET_CHECK_MSG(embedding.data != nullptr, "Embedding data cannot be null");
  append_data(
      embedding.data,
      embedding.tensor->dim(),
      embedding.tensor->sizes().data());
  ET_LOG(Info, "Merged TensorStruct embedding: total_tokens=%d", total_tokens_);
}

void MultimodalEmbeddingMerger::add_embeddings(
    const executorch::aten::Tensor& embedding) {
  append_data(
      embedding.const_data_ptr<float>(),
      embedding.dim(),
      embedding.sizes().data());
  ET_LOG(Info, "Merged Tensor embedding: total_tokens=%d", total_tokens_);
}

void MultimodalEmbeddingMerger::add_text_embeddings(
    const TensorStruct<float>& text_embeddings) {
  ET_CHECK_MSG(
      text_embeddings.tensor != nullptr,
      "Text embeddings tensor cannot be null");
  ET_CHECK_MSG(
      text_embeddings.data != nullptr, "Text embeddings data cannot be null");

  executorch::aten::Tensor tensor_wrapper(text_embeddings.tensor.get());
  add_typed_embeddings(tensor_wrapper, text_embeddings.data, EmbeddingType::kText);
}

void MultimodalEmbeddingMerger::add_image_embeddings(
    const executorch::aten::Tensor& image_embeddings) {
  add_typed_embeddings(
      image_embeddings,
      image_embeddings.const_data_ptr<float>(),
      EmbeddingType::kImage);
}

TensorStruct<float> MultimodalEmbeddingMerger::get_merged_embeddings() {
  ET_CHECK_MSG(
      !embeddings_.empty(),
      "No embeddings to return. Call add_embeddings() first.");

  embedding_sizes_ = {
      static_cast<executorch::aten::TensorImpl::SizesType>(1),
      static_cast<executorch::aten::TensorImpl::SizesType>(total_tokens_),
      static_cast<executorch::aten::TensorImpl::SizesType>(embedding_dim_)};

  TensorStruct<float> result;
  result.data = embeddings_.data();
  result.size = embeddings_.size() * sizeof(float);
  result.tensor = std::make_unique<executorch::aten::TensorImpl>(
      executorch::aten::ScalarType::Float,
      embedding_sizes_.size(),
      embedding_sizes_.data(),
      result.data);

  ET_LOG(
      Info,
      "Get merged embeddings: total_tokens=%d, embedding_dim=%d",
      total_tokens_,
      embedding_dim_);

  return result;
}

TensorStruct<float> MultimodalEmbeddingMerger::merge(
    const std::vector<uint64_t>& input_ids,
    uint64_t image_token_id) {
  ET_CHECK_MSG(!input_ids.empty(), "input_ids cannot be empty");
  ET_CHECK_MSG(
      !text_embedding_buffers_.empty(),
      "No text embeddings added. Call add_text_embeddings() first.");

  total_tokens_ = 0;

  size_t num_placeholder_tokens = 0;
  if (image_token_id != 0) {
    for (uint64_t token_id : input_ids) {
      if (token_id == image_token_id) {
        num_placeholder_tokens++;
      }
    }
  }

  ET_CHECK_MSG(
      num_placeholder_tokens == image_embedding_buffers_.size(),
      "Number of placeholder tokens (%zu) must match number of image embeddings (%zu)",
      num_placeholder_tokens,
      image_embedding_buffers_.size());

  // Calculate total tokens: sum of all text tokens + all image tokens
  for (int64_t count : text_embedding_token_counts_) {
    total_tokens_ += count;
  }
  for (int64_t count : image_embedding_token_counts_) {
    total_tokens_ += count;
  }
  total_tokens_ = total_tokens_ - num_placeholder_tokens;

  const size_t total_elements =
      static_cast<size_t>(total_tokens_) * static_cast<size_t>(embedding_dim_);
  merged_embedding_buffer_.assign(total_elements, 0.0f);

  size_t text_emb_idx = 0;
  size_t text_token_idx = 0;
  size_t image_emb_idx = 0;
  size_t output_offset = 0;

  for (uint64_t token_id : input_ids) {
    while (text_emb_idx < text_embedding_token_counts_.size() &&
           text_token_idx >=
               static_cast<size_t>(
                   text_embedding_token_counts_[text_emb_idx])) {
      text_emb_idx++;
      text_token_idx = 0;
    }

    if (image_token_id != 0 && token_id == image_token_id) {
      ET_CHECK_MSG(
          image_emb_idx < image_embedding_buffers_.size(),
          "Image index out of bounds");

      const std::vector<float>& image_buffer =
          image_embedding_buffers_[image_emb_idx];
      const size_t num_image_tokens =
          static_cast<size_t>(image_embedding_token_counts_[image_emb_idx]);

      const size_t num_elements =
          num_image_tokens * static_cast<size_t>(embedding_dim_);
      std::memcpy(
          merged_embedding_buffer_.data() + output_offset,
          image_buffer.data(),
          num_elements * sizeof(float));

      output_offset += num_elements;
      image_emb_idx++;
      text_token_idx++;
    } else {
      ET_CHECK_MSG(
          text_emb_idx < text_embedding_buffers_.size(),
          "Text embedding index out of bounds");

      const std::vector<float>& text_buffer =
          text_embedding_buffers_[text_emb_idx];
      std::memcpy(
          merged_embedding_buffer_.data() + output_offset,
          text_buffer.data() + text_token_idx * embedding_dim_,
          embedding_dim_ * sizeof(float));

      output_offset += embedding_dim_;
      text_token_idx++;
    }
  }

  ET_CHECK_MSG(
      image_emb_idx == image_embedding_buffers_.size(),
      "Not all image embeddings were used: used %zu, expected %zu",
      image_emb_idx,
      image_embedding_buffers_.size());

  TensorStruct<float> merged_embeddings;
  merged_embeddings.data = merged_embedding_buffer_.data();
  merged_embeddings.size = total_elements * sizeof(float);

  merged_embedding_sizes_ = {
      static_cast<executorch::aten::TensorImpl::SizesType>(1),
      static_cast<executorch::aten::TensorImpl::SizesType>(total_tokens_),
      static_cast<executorch::aten::TensorImpl::SizesType>(embedding_dim_)};

  merged_embeddings.tensor = std::make_unique<executorch::aten::TensorImpl>(
      executorch::aten::ScalarType::Float,
      merged_embedding_sizes_.size(),
      merged_embedding_sizes_.data(),
      merged_embeddings.data);

  ET_LOG(
      Info,
      "Merged embeddings: total_tokens=%d, text=%zu, images=%zu, embedding_dim=%d",
      total_tokens_,
      text_embedding_buffers_.size(),
      image_embedding_buffers_.size(),
      embedding_dim_);

  return merged_embeddings;
}

} // namespace example
