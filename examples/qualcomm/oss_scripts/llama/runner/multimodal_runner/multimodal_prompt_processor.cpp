/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/multimodal_prompt_processor.h>
#include <chrono>
#include <numeric>

using executorch::aten::TensorImpl;
using executorch::runtime::MethodMeta;
using executorch::runtime::Result;
using executorch::runtime::TensorInfo;

namespace example {

MultimodalPromptProcessor::MultimodalPromptProcessor(
    DecoderRunner* decoder_runner,
    KVManager* kv_manager,
    const std::string& method_name,
    Metadata metadata,
    std::unique_ptr<MethodMeta> method_meta,
    std::vector<PrefillBucket> prefill_buckets)
    : PromptProcessor(
          decoder_runner,
          kv_manager,
          method_name,
          {metadata.context_len,
           metadata.num_heads,
           metadata.num_layers,
           metadata.ar_len,
           metadata.vocab_size,
           metadata.use_int64_token,
           metadata.sliding_window,
           metadata.cache_mode},
          std::move(method_meta)),
      metadata_(metadata),
      prefill_buckets_(std::move(prefill_buckets)) {
  // Set input_toks_.size to 0 since we use embeddings instead
  input_toks_.size = 0;
  input_pos_.size =
      metadata_.ar_len * metadata_.position_id_dim * sizeof(int32_t);
  input_embedding_.size =
      metadata_.ar_len * metadata_.embedding_dim * sizeof(float);
};

void MultimodalPromptProcessor::init_io(
    IMemAlloc* buffer_manager,
    Result<MethodMeta> method_meta) {
  size_t idx = 0;
  input_tensors_.reserve(method_meta->num_inputs());
  output_tensors_.reserve(method_meta->num_outputs());

  // [I]: input embedding
  Result<TensorInfo> input_embedding = method_meta->input_tensor_meta(idx++);
  input_embedding_.data =
      reinterpret_cast<float*>(buffer_manager->allocate(input_embedding_.size));
  input_embedding_.tensor = std::make_unique<TensorImpl>(
      input_embedding->scalar_type(),
      input_embedding->sizes().size(),
      const_cast<TensorImpl::SizesType*>(input_embedding->sizes().data()),
      input_embedding_.data,
      const_cast<TensorImpl::DimOrderType*>(
          input_embedding->dim_order().data()));
  input_tensors_.emplace_back(input_embedding_.tensor.get());
  buffer_manager->add_memory_info(
      input_embedding_.data, input_embedding_.size, input_embedding.get());

  // [I]: attention_mask
  Result<TensorInfo> attention_mask = method_meta->input_tensor_meta(idx++);
  attention_mask_.data = buffer_manager->allocate(attention_mask_.size);
  attention_mask_.tensor = std::make_unique<TensorImpl>(
      attention_mask->scalar_type(),
      attention_mask->sizes().size(),
      const_cast<TensorImpl::SizesType*>(attention_mask->sizes().data()),
      attention_mask_.data,
      const_cast<TensorImpl::DimOrderType*>(
          attention_mask->dim_order().data()));
  input_tensors_.emplace_back(attention_mask_.tensor.get());
  buffer_manager->add_memory_info(
      attention_mask_.data, attention_mask_.size, attention_mask.get());

  // [I]: sliding window attention_mask
  if (metadata_.cache_mode == CacheMode::HybridCache) {
    Result<TensorInfo> window_attention_mask =
        method_meta->input_tensor_meta(idx++);
    window_attention_mask_.size = window_attention_mask->nbytes();
    const auto window_mask_sizes = window_attention_mask->sizes();
    window_context_len_swa_ = window_mask_sizes[window_mask_sizes.size() - 1];
    true_swa_enabled_swa_ = window_context_len_swa_ != metadata_.context_len;
    ET_LOG(
        Info,
        "[SWA_DEBUG] multimodal init_io(): window_context_len_swa=%d, context_len=%d, true_swa_enabled=%d",
        window_context_len_swa_,
        metadata_.context_len,
        true_swa_enabled_swa_);
    window_attention_mask_.data =
        buffer_manager->allocate(window_attention_mask_.size);
    window_attention_mask_.tensor = std::make_unique<TensorImpl>(
        window_attention_mask->scalar_type(),
        window_attention_mask->sizes().size(),
        const_cast<TensorImpl::SizesType*>(
            window_attention_mask->sizes().data()),
        window_attention_mask_.data,
        const_cast<TensorImpl::DimOrderType*>(
            window_attention_mask->dim_order().data()));
    input_tensors_.emplace_back(window_attention_mask_.tensor.get());
    buffer_manager->add_memory_info(
        window_attention_mask_.data,
        window_attention_mask_.size,
        window_attention_mask.get());
  }

  if (!is_bert()) {
    // [I]: input_pos
    Result<TensorInfo> input_pos = method_meta->input_tensor_meta(idx++);
    input_pos_.data =
        reinterpret_cast<int32_t*>(buffer_manager->allocate(input_pos_.size));
    input_pos_.tensor = std::make_unique<TensorImpl>(
        input_pos->scalar_type(),
        input_pos->sizes().size(),
        const_cast<TensorImpl::SizesType*>(input_pos->sizes().data()),
        input_pos_.data,
        const_cast<TensorImpl::DimOrderType*>(input_pos->dim_order().data()));
    input_tensors_.emplace_back(input_pos_.tensor.get());
    buffer_manager->add_memory_info(
        input_pos_.data, input_pos_.size, input_pos.get());

    // [I] kv_cache
    size_t index = idx; // bypass input_tokens, atten_mask, input_pos
    for (int cache_group = 0; cache_group < 2; ++cache_group) {
      std::vector<std::unique_ptr<TensorImpl>>& cache =
          (cache_group == 0 ? k_cache_in_ : v_cache_in_);
      std::vector<KVCache> cache_ptrs = (cache_group == 0)
          ? kv_manager_->get_k_cache_()
          : kv_manager_->get_v_cache_();
      for (int layer = 0; layer < metadata_.num_layers; ++layer, ++index) {
        Result<TensorInfo> kv_cache = method_meta->input_tensor_meta(index);

        cache[layer] = std::make_unique<TensorImpl>(
            kv_cache->scalar_type(),
            kv_cache->sizes().size(),
            const_cast<TensorImpl::SizesType*>(kv_cache->sizes().data()),
            cache_ptrs[layer].buffer,
            const_cast<TensorImpl::DimOrderType*>(
                kv_cache->dim_order().data()));
        input_tensors_.emplace_back(cache[layer].get());
        buffer_manager->add_memory_info(
            cache_ptrs[layer].buffer, cache[layer]->nbytes(), kv_cache.get());
      }
    }
  }

  if (has_last_token_index_input(method_meta.get())) {
    const size_t index = method_meta->num_inputs() - 1;
    auto index_meta = method_meta->input_tensor_meta(index);
    last_token_index_.size = index_meta->nbytes();
    last_token_index_.data = reinterpret_cast<int32_t*>(
        buffer_manager->allocate(last_token_index_.size));
    last_token_index_.tensor = std::make_unique<TensorImpl>(
        index_meta->scalar_type(),
        index_meta->sizes().size(),
        const_cast<TensorImpl::SizesType*>(index_meta->sizes().data()),
        last_token_index_.data,
        const_cast<TensorImpl::DimOrderType*>(index_meta->dim_order().data()));
    input_tensors_.emplace_back(last_token_index_.tensor.get());
    buffer_manager->add_memory_info(
        last_token_index_.data, last_token_index_.size, index_meta.get());
  }

  // [O]: logits
  Result<TensorInfo> logits = method_meta->output_tensor_meta(0);
  logits_.data = buffer_manager->allocate(logits_.size);
  logits_.tensor = std::make_unique<TensorImpl>(
      logits->scalar_type(),
      logits->sizes().size(),
      const_cast<TensorImpl::SizesType*>(logits->sizes().data()),
      logits_.data,
      const_cast<TensorImpl::DimOrderType*>(logits->dim_order().data()));
  output_tensors_.emplace_back(logits_.tensor.get());
  buffer_manager->add_memory_info(logits_.data, logits_.size, logits.get());

  // [O] kv_cache
  size_t index = 1;
  for (int cache_group = 0; cache_group < 2; ++cache_group) {
    std::vector<std::unique_ptr<TensorImpl>>& cache =
        (cache_group == 0 ? k_cache_out_ : v_cache_out_);
    std::vector<KVCache> cache_ptrs = (cache_group == 0)
        ? kv_manager_->get_k_cache_()
        : kv_manager_->get_v_cache_();
    for (int layer = 0; layer < metadata_.num_layers; ++layer, ++index) {
      Result<TensorInfo> kv_cache = method_meta->output_tensor_meta(index);
      cache[layer] = std::make_unique<TensorImpl>(
          kv_cache->scalar_type(),
          kv_cache->sizes().size(),
          const_cast<TensorImpl::SizesType*>(kv_cache->sizes().data()),
          cache_ptrs[layer].output_buffer,
          const_cast<TensorImpl::DimOrderType*>(kv_cache->dim_order().data()));
      output_tensors_.emplace_back(cache[layer].get());
      buffer_manager->add_memory_info(
          cache_ptrs[layer].output_buffer,
          cache[layer]->nbytes(),
          kv_cache.get());
    }
  }

  // Prepare the vector of EValue to run inference
  inputs_.reserve(input_tensors_.size());
  for (auto& input_tensor : input_tensors_) {
    inputs_.emplace_back(std::move(input_tensor));
  }
}

// prepare embedding
void MultimodalPromptProcessor::prepare_io(
    const TensorStruct<float>& prompt_embedding,
    int32_t num_prompt_tokens,
    int64_t prompt_pos,
    int64_t start_pos,
    const std::vector<int32_t>* position_ids) {
  if (last_token_index_.size > 0) {
    *last_token_index_.data =
        std::min<int64_t>(metadata_.ar_len, num_prompt_tokens - prompt_pos) - 1;
  }
  for (int i = 0; i < metadata_.ar_len; i++) {
    if (!is_bert()) {
      // Prepare pos data
      for (int dim = 0; dim < metadata_.position_id_dim; ++dim) {
        int32_t pos = static_cast<int32_t>(start_pos + i);
        if (position_ids != nullptr && prompt_pos + i < num_prompt_tokens) {
          pos = (*position_ids)
              [(prompt_pos + i) * metadata_.position_id_dim + dim];
        }
        input_pos_.data[dim * metadata_.ar_len + i] = pos;
      }
    }

    // Prepare input token data
    if (prompt_pos + i < num_prompt_tokens) {
      std::memcpy(
          input_embedding_.data + i * metadata_.embedding_dim,
          prompt_embedding.data + (prompt_pos + i) * metadata_.embedding_dim,
          metadata_.embedding_dim * sizeof(float));
    } else {
      std::memset(
          input_embedding_.data + i * metadata_.embedding_dim,
          0,
          metadata_.embedding_dim * sizeof(float));
    }
  }
}

Result<uint64_t> MultimodalPromptProcessor::prefill(
    const TensorStruct<float>& prompt_embedding,
    int64_t start_pos,
    bool dump_logits,
    AttentionSinkRopeRunner* attention_sink_rope_runner,
    const std::vector<int32_t>* position_ids) {
  if (dump_logits) {
    ET_CHECK_MSG(
        logits_.tensor->sizes()[1] == metadata_.ar_len,
        "Dumping all prompt logits requires an all-token logits PTE");
    prompt_all_logits_.clear();
  }
  int32_t num_prompt_tokens = prompt_embedding.tensor->size(1);
  if (!is_bert()) {
    ET_CHECK_MSG(
        metadata_.can_prefill(
            start_pos, num_prompt_tokens, kv_manager_->max_cache_len()),
        "Prompt exceeds prefill capacity: start=%ld tokens=%d AR=%d context=%d max_past=%d",
        start_pos,
        num_prompt_tokens,
        metadata_.ar_len,
        metadata_.context_len,
        metadata_.max_prefill_past);
  } else {
    ET_CHECK_MSG(
        start_pos == 0, "Bert model doesn't support multi-turn conversation.");
  }

  // store the token
  int64_t cur_token;
  int64_t prompt_pos = 0;
  int64_t pos = start_pos;
  int32_t n_update = metadata_.ar_len;
  int num_iters = 1 + ((num_prompt_tokens - 1) / metadata_.ar_len);
  ET_LOG(
      Info,
      "Prompt Processor: total %d prompt tokens (AR-%d * %d iters)",
      num_prompt_tokens,
      metadata_.ar_len,
      num_iters);

  // Rearrange KV cache first
  if (true_swa_enabled_swa_) {
    kv_manager_->rearrange_cache_swa(metadata_.ar_len);
  } else {
    kv_manager_->rearrange_cache(metadata_.ar_len);
  }
  std::vector<int32_t> attention_map(metadata_.ar_len);
  std::iota(attention_map.begin(), attention_map.end(), -1);
  // Initialize attention mask with current position
  kv_manager_->init_attention_mask(
      attention_mask_.data, attention_map, metadata_.ar_len, pos);
  // Initialize window attention mask with current position
  if (metadata_.cache_mode == CacheMode::HybridCache) {
    if (true_swa_enabled_swa_) {
      kv_manager_->init_attention_mask_swa(
          window_attention_mask_.data,
          attention_map,
          metadata_.ar_len,
          pos,
          window_context_len_swa_,
          metadata_.sliding_window);
    } else {
      kv_manager_->init_attention_mask(
          window_attention_mask_.data,
          attention_map,
          metadata_.ar_len,
          pos,
          metadata_.sliding_window);
    }
  }

  // Initialize the output of the module
  ET_CHECK_MSG(
      decoder_runner_->set_outputs(method_name_, output_tensors_) ==
          executorch::runtime::Error::Ok,
      "Failed to set output tensor for module %s",
      method_name_.c_str());
  const std::string* previous_method = nullptr;
  using Clock = std::chrono::steady_clock;
  std::chrono::duration<double, std::milli> prepare_time{0};
  std::chrono::duration<double, std::milli> execute_time{0};
  std::chrono::duration<double, std::milli> cache_time{0};
  for (int i = 0; i < num_iters; ++i) {
    // Fill in the embedding and position data
    const auto prepare_start = Clock::now();
    prepare_io(
        prompt_embedding, num_prompt_tokens, prompt_pos, pos, position_ids);

    // Run inference
    const std::string* selected_method = &method_name_;
    for (const auto& bucket : prefill_buckets_) {
      if (pos <= bucket.max_past) {
        selected_method = &bucket.method_name;
        break;
      }
    }
    if (*selected_method != method_name_) {
      ET_CHECK(
          decoder_runner_->set_outputs(*selected_method, output_tensors_) ==
          executorch::runtime::Error::Ok);
    }
    if (selected_method != previous_method) {
      ET_LOG(Info, "Prefill pos=%ld method=%s", pos, selected_method->c_str());
      previous_method = selected_method;
    }
    const auto execute_start = Clock::now();
    prepare_time += execute_start - prepare_start;
    const auto step_result = decoder_runner_->step(*selected_method, inputs_);
    const auto cache_start = Clock::now();
    execute_time += cache_start - execute_start;
    ET_CHECK_MSG(
        step_result.ok(),
        "Prefill execution failed for method %s: 0x%x",
        selected_method->c_str(),
        static_cast<unsigned>(step_result.error()));
    if (dump_logits) {
      prompt_all_logits_.insert(
          prompt_all_logits_.end(),
          logits_.data,
          logits_.data +
              metadata_.ar_len * metadata_.vocab_size *
                  logits_.getElementSize());
    }
    // In the last run, offset to the meaningful logits.
    if (i == num_iters - 1) {
      n_update = 1 + ((num_prompt_tokens - 1) % metadata_.ar_len);
    }
    // Update KV Cache with the output results
    if (true_swa_enabled_swa_) {
      kv_manager_->update_cache_swa(metadata_.ar_len, pos, n_update, {});
    } else {
      kv_manager_->update_cache(metadata_.ar_len, pos, n_update, {});
    }

    // Update attention mask with current position
    kv_manager_->update_attention_mask(
        attention_mask_.data, metadata_.ar_len, pos, n_update);
    if (metadata_.cache_mode == CacheMode::HybridCache) {
      if (true_swa_enabled_swa_) {
        kv_manager_->update_attention_mask_swa(
            window_attention_mask_.data,
            attention_map,
            metadata_.ar_len,
            pos,
            n_update,
            window_context_len_swa_,
            metadata_.sliding_window);
      } else {
        kv_manager_->update_attention_mask(
            window_attention_mask_.data,
            metadata_.ar_len,
            pos,
            n_update,
            metadata_.sliding_window);
      }
    }
    cache_time += Clock::now() - cache_start;
    prompt_pos += metadata_.ar_len;
    pos += metadata_.ar_len;
  }
  ET_LOG(
      Info,
      "Prefill timing: prepare=%.3f ms execute=%.3f ms cache=%.3f ms",
      prepare_time.count(),
      execute_time.count(),
      cache_time.count());

  cur_token = decoder_runner_->logits_to_token(
      output_tensors_[0],
      (num_prompt_tokens + metadata_.ar_len - 1) % metadata_.ar_len);
  return cur_token;
}

} // namespace example
