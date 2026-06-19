/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/kv_manager.h>
#include <executorch/runtime/platform/assert.h>
#include <algorithm>
#include <cstring>
#include <limits>
namespace example {

namespace {
inline bool is_causal_chain_attention_map(
    const std::vector<int32_t>& attention_map,
    int32_t ar_len) {
  if (ar_len <= 0) {
    return false;
  }
  if (attention_map.size() < static_cast<size_t>(ar_len)) {
    return false;
  }
  if (attention_map[0] >= 0) {
    return false;
  }
  for (int32_t i = 1; i < ar_len; ++i) {
    if (attention_map[i] != i - 1) {
      return false;
    }
  }
  return true;
}

inline std::vector<int32_t> collect_selected_indices(
    const std::vector<bool>& selected,
    int32_t n_update) {
  std::vector<int32_t> indices;
  if (selected.empty() || n_update <= 0) {
    return indices;
  }

  indices.reserve(n_update);
  for (int32_t i = 0; i < static_cast<int32_t>(selected.size()); ++i) {
    if (!selected[i]) {
      continue;
    }
    indices.push_back(i);
    if (static_cast<int32_t>(indices.size()) == n_update) {
      break;
    }
  }
  return indices;
}

template <typename T>
void fill_cache_input(T* cache, size_t bytes, int32_t zero_point) {
  std::fill_n(cache, bytes / sizeof(T), static_cast<T>(zero_point));
}
} // namespace

template <typename T>
KVManager<T>::KVManager(Metadata metadata) : metadata_(metadata) {
  k_cache_.resize(metadata_.num_layers);
  v_cache_.resize(metadata_.num_layers);
  recompute_total_cache_size();
}

template <typename T>
void KVManager<T>::recompute_total_cache_size(
    const std::vector<int32_t>* layer_cache_len,
    const std::vector<bool>* layer_is_swa) {
  if (!metadata_.cache_specs.empty()) {
    ET_CHECK_MSG(
        metadata_.cache_specs.size() == static_cast<size_t>(metadata_.num_layers),
        "cache_specs size (%zu) must match num_layers (%ld)",
        metadata_.cache_specs.size(),
        metadata_.num_layers);
    total_cache_size_ = 0;
    for (const auto& spec : metadata_.cache_specs) {
      if (spec.update_mode == Metadata::kFullOverwrite) {
        ET_CHECK_MSG(
            spec.k_input_bytes == spec.k_output_bytes &&
                spec.v_input_bytes == spec.v_output_bytes,
            "Full-overwrite cache requires matching input/output byte sizes");
        total_cache_size_ +=
            align_shared_buffer_allocation_size(spec.k_input_bytes) +
            align_shared_buffer_allocation_size(spec.v_input_bytes);
        continue;
      }
      total_cache_size_ +=
          align_shared_buffer_allocation_size(spec.k_input_bytes) +
          align_shared_buffer_allocation_size(spec.k_output_bytes) +
          align_shared_buffer_allocation_size(spec.v_input_bytes) +
          align_shared_buffer_allocation_size(spec.v_output_bytes);
    }
    return;
  }

  const bool use_swa_layout = layer_cache_len != nullptr &&
      layer_is_swa != nullptr && !layer_cache_len->empty() &&
      !layer_is_swa->empty();
  if (use_swa_layout) {
    ET_CHECK_MSG(
        layer_cache_len->size() == static_cast<size_t>(metadata_.num_layers) &&
            layer_is_swa->size() == static_cast<size_t>(metadata_.num_layers),
        "layer_cache_len/layer_is_swa sizes must match num_layers");
  }

  size_t cache_in_bytes = 0;
  for (int64_t layer = 0; layer < metadata_.num_layers; ++layer) {
    int32_t cache_len_in = metadata_.max_cache_len;
    if (use_swa_layout && (*layer_is_swa)[layer]) {
      cache_len_in = (*layer_cache_len)[layer];
    }
    ET_CHECK_MSG(cache_len_in >= 0, "cache_len_in must be >= 0, got %d", cache_len_in);
    const size_t layer_cache_in_bytes = static_cast<size_t>(metadata_.num_heads) *
        static_cast<size_t>(metadata_.head_dim) *
        static_cast<size_t>(cache_len_in) * sizeof(T);
    cache_in_bytes += align_shared_buffer_allocation_size(layer_cache_in_bytes);
  }

  const size_t layer_cache_out_bytes = static_cast<size_t>(metadata_.num_heads) *
      static_cast<size_t>(metadata_.head_dim) *
      static_cast<size_t>(metadata_.max_ar_len) * sizeof(T);
  const size_t cache_out_bytes = static_cast<size_t>(metadata_.num_layers) *
      align_shared_buffer_allocation_size(layer_cache_out_bytes);
  total_cache_size_ = 2 * (cache_in_bytes + cache_out_bytes);
}

template <typename T>
void KVManager<T>::configure_cache_size_swa(
    const std::vector<int32_t>& layer_cache_len,
    const std::vector<bool>& layer_is_swa) {
  if (!metadata_.cache_specs.empty()) {
    recompute_total_cache_size();
    return;
  }
  recompute_total_cache_size(&layer_cache_len, &layer_is_swa);
}

template <typename T>
void KVManager<T>::init_attention_mask(
    uint16_t* attention_mask,
    const std::vector<int32_t>& attention_map,
    int32_t ar_len,
    int32_t n_past) {
  // Delegate to the sliding-window implementation with a sufficiently large
  // window so the "mask by sliding_window" code path is a no-op.
  init_attention_mask(
      attention_mask,
      attention_map,
      ar_len,
      n_past,
      std::numeric_limits<int32_t>::max(),
      {});
}

template <typename T>
void KVManager<T>::init_attention_mask(
    uint16_t* attention_mask,
    const std::vector<int32_t>& attention_map,
    int32_t ar_len,
    int32_t n_past,
    int32_t sliding_window,
    const std::vector<int32_t>& position_offset) {
  ET_CHECK_MSG(
      attention_map.size() >= static_cast<size_t>(ar_len),
      "attention_map size (%zu) must be >= ar_len (%d)",
      attention_map.size(),
      ar_len);
  ET_CHECK_MSG(
      position_offset.empty() ||
          position_offset.size() >= static_cast<size_t>(ar_len),
      "position_offset size (%zu) must be empty or >= ar_len (%d)",
      position_offset.size(),
      ar_len);
  uint16_t neg_val = 0;
  uint16_t pos_val = 65535;
  // Clear the attention mask
  std::fill_n(attention_mask, ar_len * metadata_.context_len, neg_val);

  // SMART_MASK requires special handling of attention mask
  uint16_t* past_ptr = attention_mask;
  uint16_t* new_ptr = attention_mask + (metadata_.context_len - ar_len);
  // All inputs will necessarily attend to n_past and itself
  for (int i = 0; i < ar_len; i++) {
    // Iterate across ar_len
    if (attention_map[i] < 0) {
      // If negative, attend to only past tokens
      std::fill_n(past_ptr, n_past, pos_val);
    } else {
      // If positive, copy attention map from (relative to 0th input) parent
      // Parent token index
      const int32_t pidx = attention_map[i];
      uint16_t* parent_ptr = attention_mask + pidx * metadata_.context_len;
      std::memcpy(
          past_ptr, parent_ptr, metadata_.context_len * sizeof(uint16_t));
    }
    // Attend to itself
    new_ptr[i] = pos_val;

    // mask by limitation of sliding_window
    int32_t available_context_len = position_offset.empty()
        ? sliding_window - (i + 1) - n_past
        : sliding_window - (position_offset[i] + 1) - n_past;
    // if available_context_len is less than 0, it means we need to mask some
    // tokens in the past to avoid exceeding the sliding window
    if (available_context_len < 0) {
      std::fill_n(past_ptr, -available_context_len, neg_val);
    }

    past_ptr += metadata_.context_len;
    new_ptr += metadata_.context_len;
  }
}

template <typename T>
void KVManager<T>::update_attention_mask(
    uint16_t* attention_mask,
    int32_t ar_len,
    int32_t n_past,
    int32_t n_update) {
  // Delegate to the sliding-window implementation with a sufficiently large
  // window so the "mask by sliding_window" code path is a no-op.
  update_attention_mask(
      attention_mask,
      ar_len,
      n_past,
      n_update,
      std::numeric_limits<int32_t>::max(),
      {});
}

template <typename T>
void KVManager<T>::update_attention_mask(
    uint16_t* attention_mask,
    int32_t ar_len,
    int32_t n_past,
    int32_t n_update,
    int32_t sliding_window,
    const std::vector<int32_t>& position_offset) {
  uint16_t pos_val = 65535;
  uint16_t neg_val = 0;
  ET_CHECK_MSG(
      position_offset.empty() ||
          position_offset.size() >= static_cast<size_t>(ar_len),
      "position_offset size (%zu) must be empty or >= ar_len (%d)",
      position_offset.size(),
      ar_len);
  uint16_t* cur_ptr = attention_mask;
  cur_ptr += n_past;

  for (int i = 0; i < ar_len; i++) {
    std::fill_n(cur_ptr, n_update, pos_val);
    const int32_t available_cache_len_raw = position_offset.empty()
        ? sliding_window - (i + 1)
        : sliding_window - (position_offset[i] + 1);
    const int32_t available_cache_len = std::max(0, available_cache_len_raw);

    // Incremental sliding-window update: only newly masked tokens need to be
    // updated. The previous call already masked the prefix corresponding to
    // n_past.
    const int32_t masked_prev =
        (n_past > available_cache_len) ? (n_past - available_cache_len) : 0;
    const int32_t masked_now =
        (n_past + n_update > available_cache_len)
        ? (n_past + n_update - available_cache_len)
        : 0;
    const int32_t delta = masked_now - masked_prev;
    if (delta > 0) {
      std::fill_n((cur_ptr - n_past) + masked_prev, delta, neg_val);
    }
    cur_ptr += metadata_.context_len;
  }
}

template <typename T>
void KVManager<T>::init_cache(IMemAlloc* buffer_manager, int32_t ar_len) {
  cur_ar_len_ = ar_len;
  if (!metadata_.cache_specs.empty()) {
    ET_CHECK_MSG(
        metadata_.cache_specs.size() == static_cast<size_t>(metadata_.num_layers),
        "cache_specs size (%zu) must match num_layers (%ld)",
        metadata_.cache_specs.size(),
        metadata_.num_layers);
    has_swa_layers_swa_ = false;
    swa_cache_len_swa_ = 0;
    swa_write_pos_swa_ = 0;
    swa_valid_len_swa_ = 0;
    layer_is_swa_.assign(metadata_.num_layers, false);
    layer_cache_len_swa_.assign(metadata_.num_layers, 0);

    for (int layer = 0; layer < metadata_.num_layers; ++layer) {
      const auto& spec = metadata_.cache_specs[layer];
      const bool in_place_overwrite =
          spec.update_mode == Metadata::kFullOverwrite;
      if (spec.update_mode == Metadata::kCircular) {
        const size_t cache_len_denom = static_cast<size_t>(metadata_.num_heads) *
            static_cast<size_t>(metadata_.head_dim) * sizeof(T);
        ET_CHECK_MSG(
            cache_len_denom > 0 && spec.k_input_bytes % cache_len_denom == 0,
            "Circular key cache bytes (%zu) are not divisible by heads*head_dim*sizeof(T) (%zu)",
            spec.k_input_bytes,
            cache_len_denom);
        ET_CHECK_MSG(
            spec.v_input_bytes % cache_len_denom == 0,
            "Circular value cache bytes (%zu) are not divisible by heads*head_dim*sizeof(T) (%zu)",
            spec.v_input_bytes,
            cache_len_denom);
        const int32_t k_cache_len =
            static_cast<int32_t>(spec.k_input_bytes / cache_len_denom);
        const int32_t v_cache_len =
            static_cast<int32_t>(spec.v_input_bytes / cache_len_denom);
        ET_CHECK_MSG(
            k_cache_len == v_cache_len,
            "Circular key/value cache length mismatch: %d vs %d",
            k_cache_len,
            v_cache_len);
        layer_is_swa_[layer] = true;
        layer_cache_len_swa_[layer] = k_cache_len;
        has_swa_layers_swa_ = true;
        if (swa_cache_len_swa_ == 0) {
          swa_cache_len_swa_ = k_cache_len;
        } else {
          ET_CHECK_MSG(
              swa_cache_len_swa_ == k_cache_len,
              "All circular SWA cache specs must share the same cache length");
        }
      }
      if (in_place_overwrite) {
        ET_CHECK_MSG(
            spec.k_input_bytes == spec.k_output_bytes &&
                spec.v_input_bytes == spec.v_output_bytes,
            "Full-overwrite cache requires matching input/output byte sizes");
      }
      T* single_layer_k_cache_in =
          reinterpret_cast<T*>(buffer_manager->allocate(spec.k_input_bytes));
      T* single_layer_k_cache_out = in_place_overwrite
          ? single_layer_k_cache_in
          : reinterpret_cast<T*>(buffer_manager->allocate(spec.k_output_bytes));
      T* single_layer_v_cache_in =
          reinterpret_cast<T*>(buffer_manager->allocate(spec.v_input_bytes));
      T* single_layer_v_cache_out = in_place_overwrite
          ? single_layer_v_cache_in
          : reinterpret_cast<T*>(buffer_manager->allocate(spec.v_output_bytes));

      fill_cache_input(
          single_layer_k_cache_in, spec.k_input_bytes, spec.k_zero_point);
      if (!in_place_overwrite) {
        std::memset(single_layer_k_cache_out, 0, spec.k_output_bytes);
      }
      fill_cache_input(
          single_layer_v_cache_in, spec.v_input_bytes, spec.v_zero_point);
      if (!in_place_overwrite) {
        std::memset(single_layer_v_cache_out, 0, spec.v_output_bytes);
      }

      k_cache_[layer].buffer = single_layer_k_cache_in;
      k_cache_[layer].output_buffer = single_layer_k_cache_out;
      v_cache_[layer].buffer = single_layer_v_cache_in;
      v_cache_[layer].output_buffer = single_layer_v_cache_out;
    }
    return;
  }

  const size_t max_in_cache_block_in_bytes =
      metadata_.max_cache_len * sizeof(T);
  const size_t max_out_cache_block_in_bytes = metadata_.max_ar_len * sizeof(T);

  const size_t cache_in_bytes =
      metadata_.num_heads * metadata_.head_dim * max_in_cache_block_in_bytes;
  const size_t cache_out_bytes =
      metadata_.num_heads * metadata_.head_dim * max_out_cache_block_in_bytes;
  for (int layer = 0; layer < metadata_.num_layers; ++layer) {
    // Allocate buffer for key cache and value cache
    T* single_layer_k_cache_in =
        reinterpret_cast<T*>(buffer_manager->allocate(cache_in_bytes));
    T* single_layer_k_cache_out =
        reinterpret_cast<T*>(buffer_manager->allocate(cache_out_bytes));
    T* single_layer_v_cache_in =
        reinterpret_cast<T*>(buffer_manager->allocate(cache_in_bytes));
    T* single_layer_v_cache_out =
        reinterpret_cast<T*>(buffer_manager->allocate(cache_out_bytes));

    k_cache_[layer].buffer = single_layer_k_cache_in;
    k_cache_[layer].output_buffer = single_layer_k_cache_out;
    v_cache_[layer].buffer = single_layer_v_cache_in;
    v_cache_[layer].output_buffer = single_layer_v_cache_out;
  }
}

template <typename T>
void KVManager<T>::reset_state(int32_t ar_len, bool clear_buffers) {
  cur_ar_len_ = ar_len;
  swa_write_pos_swa_ = 0;
  swa_valid_len_swa_ = 0;

  if (!clear_buffers) {
    return;
  }

  if (!metadata_.cache_specs.empty()) {
    for (int layer = 0; layer < metadata_.num_layers; ++layer) {
      const auto& spec = metadata_.cache_specs[layer];
      fill_cache_input(
          k_cache_[layer].buffer, spec.k_input_bytes, spec.k_zero_point);
      fill_cache_input(
          v_cache_[layer].buffer, spec.v_input_bytes, spec.v_zero_point);
      if (k_cache_[layer].output_buffer != k_cache_[layer].buffer) {
        std::memset(k_cache_[layer].output_buffer, 0, spec.k_output_bytes);
      }
      if (v_cache_[layer].output_buffer != v_cache_[layer].buffer) {
        std::memset(v_cache_[layer].output_buffer, 0, spec.v_output_bytes);
      }
    }
    return;
  }

  const size_t out_elems = static_cast<size_t>(metadata_.num_heads) *
      static_cast<size_t>(metadata_.head_dim) *
      static_cast<size_t>(metadata_.max_ar_len);
  const size_t out_bytes = out_elems * sizeof(T);

  for (int layer = 0; layer < metadata_.num_layers; ++layer) {
    int32_t cache_len_in = metadata_.max_cache_len;
    if (has_swa_layers_swa_ && !layer_is_swa_.empty() &&
        static_cast<size_t>(layer) < layer_is_swa_.size() && layer_is_swa_[layer] &&
        static_cast<size_t>(layer) < layer_cache_len_swa_.size()) {
      cache_len_in = layer_cache_len_swa_[layer];
    }
    const size_t in_elems = static_cast<size_t>(metadata_.num_heads) *
        static_cast<size_t>(metadata_.head_dim) *
        static_cast<size_t>(cache_len_in);
    const size_t in_bytes = in_elems * sizeof(T);

    std::memset(k_cache_[layer].buffer, 0, in_bytes);
    std::memset(v_cache_[layer].buffer, 0, in_bytes);
    std::memset(k_cache_[layer].output_buffer, 0, out_bytes);
    std::memset(v_cache_[layer].output_buffer, 0, out_bytes);
  }
}

template <typename T>
void KVManager<T>::rearrange_cache(int32_t ar_len_dst) {
  // Don't need to rearrange if cur_ar_len_ is equal to target ar_len
  if (cur_ar_len_ == ar_len_dst)
    return;
  for (int layer = 0; layer < metadata_.num_layers; ++layer) {
    if (!metadata_.cache_specs.empty() &&
        metadata_.cache_specs[layer].update_mode == Metadata::kFullOverwrite) {
      continue;
    }
    rearrange_key(k_cache_[layer], ar_len_dst);
    rearrange_value(v_cache_[layer], ar_len_dst);
  }
  // rearrange done.
  cur_ar_len_ = ar_len_dst;
}

template <typename T>
void KVManager<T>::rearrange_key(KVCache<T>& k_cache, int32_t ar_len_dst) {
  const int32_t src_cache_num = (cur_ar_len_ == metadata_.context_len)
      ? metadata_.context_len
      : metadata_.context_len - cur_ar_len_;
  const int32_t dst_cache_num = metadata_.context_len - ar_len_dst;
  T* k_cache_in_read_ptr = k_cache.buffer;
  T* k_cache_in_write_ptr = k_cache.buffer;

  if (src_cache_num > dst_cache_num) {
    // copy from first dimension
    for (int i = 0; i < metadata_.head_dim * metadata_.num_heads; i++) {
      std::memmove(
          k_cache_in_write_ptr, k_cache_in_read_ptr, dst_cache_num * sizeof(T));
      k_cache_in_read_ptr += src_cache_num;
      k_cache_in_write_ptr += dst_cache_num;
    }
  } else {
    k_cache_in_read_ptr +=
        (metadata_.head_dim * metadata_.num_heads - 1) * src_cache_num;
    k_cache_in_write_ptr +=
        (metadata_.head_dim * metadata_.num_heads - 1) * dst_cache_num;
    // copy from last dimension
    for (int i = 0; i < metadata_.head_dim * metadata_.num_heads; i++) {
      std::memmove(
          k_cache_in_write_ptr, k_cache_in_read_ptr, src_cache_num * sizeof(T));
      k_cache_in_read_ptr -= src_cache_num;
      k_cache_in_write_ptr -= dst_cache_num;
    }
  }
}

template <typename T>
void KVManager<T>::rearrange_value(KVCache<T>& v_cache, int32_t ar_len_dst) {
  const int32_t src_cache_num = (cur_ar_len_ == metadata_.context_len)
      ? metadata_.context_len
      : metadata_.context_len - cur_ar_len_;
  const int32_t dst_cache_num = metadata_.context_len - ar_len_dst;
  T* v_cache_in_read_ptr = v_cache.buffer;
  T* v_cache_in_write_ptr = v_cache.buffer;
  if (src_cache_num > dst_cache_num) {
    // copy from first dimension
    for (int i = 0; i < metadata_.num_heads; i++) {
      std::memmove(
          v_cache_in_write_ptr,
          v_cache_in_read_ptr,
          dst_cache_num * metadata_.head_dim * sizeof(T));
      v_cache_in_read_ptr += src_cache_num * metadata_.head_dim;
      v_cache_in_write_ptr += dst_cache_num * metadata_.head_dim;
    }
  } else {
    v_cache_in_read_ptr +=
        metadata_.head_dim * (metadata_.num_heads - 1) * src_cache_num;
    v_cache_in_write_ptr +=
        metadata_.head_dim * (metadata_.num_heads - 1) * dst_cache_num;
    // copy from last dimension
    for (int i = 0; i < metadata_.num_heads; i++) {
      std::memmove(
          v_cache_in_write_ptr,
          v_cache_in_read_ptr,
          src_cache_num * metadata_.head_dim * sizeof(T));
      v_cache_in_read_ptr -= src_cache_num * metadata_.head_dim;
      v_cache_in_write_ptr -= dst_cache_num * metadata_.head_dim;
    }
  }
}

template <typename T>
void KVManager<T>::update_cache(
    int32_t ar_len,
    int32_t n_past,
    int32_t n_update,
    const std::vector<bool>& selected) {
  ET_CHECK_MSG(
      cur_ar_len_ == ar_len,
      "Current AR length (%d) is not matched with target AR length (%d). Please rearrange cache first.",
      cur_ar_len_,
      ar_len);
  const std::vector<int32_t> selected_indices =
      collect_selected_indices(selected, n_update);
  ET_CHECK_MSG(
      selected.empty() ||
          static_cast<int32_t>(selected_indices.size()) == n_update,
      "selected contains %zu true values, expected at least n_update (%d)",
      selected_indices.size(),
      n_update);
  for (int layer = 0; layer < metadata_.num_layers; ++layer) {
    if (!metadata_.cache_specs.empty() &&
        metadata_.cache_specs[layer].update_mode == Metadata::kFullOverwrite) {
      update_full_overwrite_cache(k_cache_[layer], v_cache_[layer], layer);
      continue;
    }
    update_key(k_cache_[layer], n_past, n_update, selected_indices);
    update_value(v_cache_[layer], n_past, n_update, selected_indices);
  }
}

template <typename T>
void KVManager<T>::update_key(
    KVCache<T>& k_cache,
    int32_t n_past,
    int32_t n_update,
    const std::vector<int32_t>& selected_indices) {
  T* write_ptr = k_cache.buffer;
  T* read_ptr = k_cache.output_buffer;
  const int32_t copy_size = n_update * sizeof(T);
  const int32_t iter_size = (cur_ar_len_ == metadata_.context_len)
      ? metadata_.context_len
      : metadata_.context_len - cur_ar_len_;
  const int32_t out_size = cur_ar_len_;
  const int32_t past_size = n_past;
  const int32_t n_iter = metadata_.head_dim * metadata_.num_heads;

  write_ptr += past_size;
  if (selected_indices.empty()) {
    for (int i = 0; i < n_iter; ++i) {
      std::memcpy(write_ptr, read_ptr, copy_size);
      write_ptr += iter_size;
      read_ptr += out_size;
    }
  } else {
    for (int i = 0; i < n_iter; ++i) {
      auto wp = write_ptr, rp = read_ptr;
      for (int32_t ind : selected_indices) {
        *wp++ = rp[ind];
      }
      write_ptr += iter_size;
      read_ptr += out_size;
    }
  }
}

template <typename T>
void KVManager<T>::update_value(
    KVCache<T>& v_cache,
    int32_t n_past,
    int32_t n_update,
    const std::vector<int32_t>& selected_indices) {
  T* write_ptr = v_cache.buffer;
  T* read_ptr = v_cache.output_buffer;
  const int32_t copy_size = n_update * metadata_.head_dim * sizeof(T);
  const int32_t past_size = n_past * metadata_.head_dim;
  const int32_t n_iter = metadata_.num_heads;
  const int32_t iter_size = (cur_ar_len_ == metadata_.context_len)
      ? metadata_.context_len * metadata_.head_dim
      : (metadata_.context_len - cur_ar_len_) * metadata_.head_dim;
  const int32_t out_size = cur_ar_len_ * metadata_.head_dim;

  write_ptr += past_size;

  if (selected_indices.empty()) {
    for (int i = 0; i < n_iter; i++) {
      std::memcpy(write_ptr, read_ptr, copy_size);
      write_ptr += iter_size;
      read_ptr += out_size;
    }
  } else {
    for (int i = 0; i < n_iter; ++i) {
      auto wp = write_ptr, rp = read_ptr;
      for (int32_t ind : selected_indices) {
        std::memcpy(
            wp,
            rp + static_cast<size_t>(ind) * metadata_.head_dim,
            metadata_.head_dim * sizeof(T));
        wp += metadata_.head_dim;
      }
      write_ptr += iter_size;
      read_ptr += out_size;
    }
  }
}

template <typename T>
void KVManager<T>::update_full_overwrite_cache(
    const KVCache<T>& k_cache,
    const KVCache<T>& v_cache,
    int32_t layer) {
  ET_CHECK_MSG(
      !metadata_.cache_specs.empty() &&
          static_cast<size_t>(layer) < metadata_.cache_specs.size(),
      "Full-overwrite cache update requires per-layer cache specs");
  const auto& spec = metadata_.cache_specs[layer];
  ET_CHECK_MSG(
      spec.k_input_bytes == spec.k_output_bytes &&
          spec.v_input_bytes == spec.v_output_bytes,
      "Full-overwrite cache requires matching input/output byte sizes");
  if (k_cache.buffer != k_cache.output_buffer) {
    std::memcpy(k_cache.buffer, k_cache.output_buffer, spec.k_input_bytes);
  }
  if (v_cache.buffer != v_cache.output_buffer) {
    std::memcpy(v_cache.buffer, v_cache.output_buffer, spec.v_input_bytes);
  }
}

template <typename T>
void KVManager<T>::init_cache_swa(
    IMemAlloc* buffer_manager,
    int32_t ar_len,
    const std::vector<int32_t>& layer_cache_len,
    const std::vector<bool>& layer_is_swa) {
  if (!metadata_.cache_specs.empty()) {
    init_cache(buffer_manager, ar_len);
    return;
  }
  if (layer_cache_len.empty() || layer_is_swa.empty()) {
    init_cache(buffer_manager, ar_len);
    return;
  }
  ET_CHECK_MSG(
      layer_cache_len.size() == static_cast<size_t>(metadata_.num_layers) &&
          layer_is_swa.size() == static_cast<size_t>(metadata_.num_layers),
      "layer_cache_len/layer_is_swa sizes must match num_layers");

  layer_cache_len_swa_ = layer_cache_len;
  layer_is_swa_ = layer_is_swa;
  recompute_total_cache_size(&layer_cache_len_swa_, &layer_is_swa_);

  has_swa_layers_swa_ = false;
  swa_cache_len_swa_ = 0;
  for (int64_t layer = 0; layer < metadata_.num_layers; ++layer) {
    if (!layer_is_swa_[layer]) {
      continue;
    }
    has_swa_layers_swa_ = true;
    if (swa_cache_len_swa_ == 0) {
      swa_cache_len_swa_ = layer_cache_len_swa_[layer];
    } else {
      ET_CHECK_MSG(
          swa_cache_len_swa_ == layer_cache_len_swa_[layer],
          "All SWA layers must share the same cache length");
    }
  }

  swa_write_pos_swa_ = 0;
  swa_valid_len_swa_ = 0;

  cur_ar_len_ = ar_len;
  const size_t max_out_cache_block_in_bytes = metadata_.max_ar_len * sizeof(T);
  const size_t cache_out_bytes =
      metadata_.num_heads * metadata_.head_dim * max_out_cache_block_in_bytes;

  for (int layer = 0; layer < metadata_.num_layers; ++layer) {
    const int32_t cache_len_in =
        layer_is_swa_[layer] ? layer_cache_len_swa_[layer]
                                 : metadata_.max_cache_len;
    const size_t cache_in_bytes = metadata_.num_heads * metadata_.head_dim *
        static_cast<size_t>(cache_len_in) * sizeof(T);

    // Allocate buffer for key cache and value cache
    T* single_layer_k_cache_in =
        reinterpret_cast<T*>(buffer_manager->allocate(cache_in_bytes));
    T* single_layer_k_cache_out =
        reinterpret_cast<T*>(buffer_manager->allocate(cache_out_bytes));
    T* single_layer_v_cache_in =
        reinterpret_cast<T*>(buffer_manager->allocate(cache_in_bytes));
    T* single_layer_v_cache_out =
        reinterpret_cast<T*>(buffer_manager->allocate(cache_out_bytes));

    k_cache_[layer].buffer = single_layer_k_cache_in;
    k_cache_[layer].output_buffer = single_layer_k_cache_out;
    v_cache_[layer].buffer = single_layer_v_cache_in;
    v_cache_[layer].output_buffer = single_layer_v_cache_out;
  }
}

template <typename T>
void KVManager<T>::rearrange_cache_swa(int32_t ar_len_dst) {
  if (!metadata_.cache_specs.empty()) {
    if (cur_ar_len_ == ar_len_dst) {
      return;
    }
    for (int layer = 0; layer < metadata_.num_layers; ++layer) {
      const auto update_mode = metadata_.cache_specs[layer].update_mode;
      if (update_mode == Metadata::kFullOverwrite ||
          update_mode == Metadata::kCircular) {
        continue;
      }
      rearrange_key(k_cache_[layer], ar_len_dst);
      rearrange_value(v_cache_[layer], ar_len_dst);
    }
    cur_ar_len_ = ar_len_dst;
    return;
  }
  if (!has_swa_layers_swa_ || layer_is_swa_.empty()) {
    rearrange_cache(ar_len_dst);
    return;
  }
  if (cur_ar_len_ == ar_len_dst) {
    return;
  }
  for (int layer = 0; layer < metadata_.num_layers; ++layer) {
    if (layer_is_swa_[layer]) {
      continue;
    }
    rearrange_key(k_cache_[layer], ar_len_dst);
    rearrange_value(v_cache_[layer], ar_len_dst);
  }
  cur_ar_len_ = ar_len_dst;
}

template <typename T>
void KVManager<T>::update_cache_swa(
    int32_t ar_len,
    int32_t n_past,
    int32_t n_update,
    const std::vector<bool>& selected) {
  if (!metadata_.cache_specs.empty()) {
    ET_CHECK_MSG(
        cur_ar_len_ == ar_len,
        "Current AR length (%d) is not matched with target AR length (%d). Please rearrange cache first.",
        cur_ar_len_,
        ar_len);
    const std::vector<int32_t> selected_indices =
        collect_selected_indices(selected, n_update);
    ET_CHECK_MSG(
        selected.empty() ||
            static_cast<int32_t>(selected_indices.size()) == n_update,
        "selected contains %zu true values, expected at least n_update (%d)",
        selected_indices.size(),
        n_update);

    bool updated_circular_cache = false;
    for (int layer = 0; layer < metadata_.num_layers; ++layer) {
      const auto& spec = metadata_.cache_specs[layer];
      if (spec.update_mode == Metadata::kFullOverwrite) {
        update_full_overwrite_cache(k_cache_[layer], v_cache_[layer], layer);
        continue;
      }
      if (spec.update_mode == Metadata::kCircular) {
        const int32_t cache_len =
            !layer_cache_len_swa_.empty() ? layer_cache_len_swa_[layer] : 0;
        update_key_swa(k_cache_[layer], cache_len, n_update, selected_indices);
        update_value_swa(v_cache_[layer], cache_len, n_update, selected_indices);
        updated_circular_cache = true;
        continue;
      }
      update_key(k_cache_[layer], n_past, n_update, selected_indices);
      update_value(v_cache_[layer], n_past, n_update, selected_indices);
    }

    if (updated_circular_cache && swa_cache_len_swa_ > 0 && n_update > 0) {
      swa_write_pos_swa_ =
          (swa_write_pos_swa_ + n_update) % swa_cache_len_swa_;
      swa_valid_len_swa_ =
          std::min(swa_cache_len_swa_, swa_valid_len_swa_ + n_update);
    }
    return;
  }
  if (!has_swa_layers_swa_ || layer_is_swa_.empty()) {
    update_cache(ar_len, n_past, n_update, selected);
    return;
  }
  ET_CHECK_MSG(
      cur_ar_len_ == ar_len,
      "Current AR length (%d) is not matched with target AR length (%d). Please rearrange cache first.",
      cur_ar_len_,
      ar_len);
  const std::vector<int32_t> selected_indices =
      collect_selected_indices(selected, n_update);
  ET_CHECK_MSG(
      selected.empty() ||
          static_cast<int32_t>(selected_indices.size()) == n_update,
      "selected contains %zu true values, expected at least n_update (%d)",
      selected_indices.size(),
      n_update);

  for (int layer = 0; layer < metadata_.num_layers; ++layer) {
    if (!layer_is_swa_[layer]) {
      update_key(k_cache_[layer], n_past, n_update, selected_indices);
      update_value(v_cache_[layer], n_past, n_update, selected_indices);
      continue;
    }
    const int32_t cache_len = layer_cache_len_swa_[layer];
    update_key_swa(k_cache_[layer], cache_len, n_update, selected_indices);
    update_value_swa(v_cache_[layer], cache_len, n_update, selected_indices);
  }

  if (swa_cache_len_swa_ > 0 && n_update > 0) {
    swa_write_pos_swa_ = (swa_write_pos_swa_ + n_update) % swa_cache_len_swa_;
    swa_valid_len_swa_ =
        std::min(swa_cache_len_swa_, swa_valid_len_swa_ + n_update);
  }
}

template <typename T>
void KVManager<T>::maybe_sync_swa_state_from_n_past_swa(
    int32_t n_past,
    int32_t cache_len) {
  if (cache_len <= 0) {
    return;
  }
  if (swa_valid_len_swa_ != 0) {
    return;
  }

  const int32_t inferred_valid_len = std::min(n_past, cache_len);
  const int32_t inferred_write_pos =
      (inferred_valid_len == cache_len) ? (n_past % cache_len)
                                        : inferred_valid_len;
  swa_valid_len_swa_ = inferred_valid_len;
  swa_write_pos_swa_ = inferred_write_pos;
}

template <typename T>
void KVManager<T>::fill_attention_mask_swa(
    uint16_t* attention_mask,
    const std::vector<int32_t>& attention_map,
    int32_t ar_len,
    int32_t context_len,
    int32_t sliding_window,
    const std::vector<int32_t>& position_offset) {
  ET_CHECK_MSG(
      attention_map.size() >= static_cast<size_t>(ar_len),
      "attention_map size (%zu) must be >= ar_len (%d)",
      attention_map.size(),
      ar_len);

  constexpr uint16_t neg_val = 0;
  constexpr uint16_t pos_val = 65535;
  std::fill_n(attention_mask, static_cast<size_t>(ar_len) * context_len, neg_val);

  ET_CHECK_MSG(
      position_offset.empty() ||
          position_offset.size() >= static_cast<size_t>(ar_len),
      "position_offset size (%zu) must be empty or >= ar_len (%d)",
      position_offset.size(),
      ar_len);

  const int32_t cache_len = context_len - ar_len;
  const int32_t effective_cache_len = std::max(cache_len, 0);
  int32_t valid_len = 0;
  int32_t write_pos = 0;
  if (effective_cache_len > 0) {
    valid_len = std::min(swa_valid_len_swa_, effective_cache_len);
    write_pos = swa_write_pos_swa_ % effective_cache_len;
  }

  for (int i = 0; i < ar_len; ++i) {
    uint16_t* row_ptr = attention_mask + i * context_len;
    if (attention_map[i] < 0) {
      if (effective_cache_len > 0) {
        if (valid_len == effective_cache_len) {
          std::fill_n(row_ptr, effective_cache_len, pos_val);
        } else {
          std::fill_n(row_ptr, valid_len, pos_val);
        }
      }
    } else {
      const int32_t pidx = attention_map[i];
      ET_CHECK_MSG(
          pidx >= 0 && pidx < ar_len,
          "Invalid attention_map parent index %d for row %d",
          pidx,
          i);
      uint16_t* parent_ptr = attention_mask + pidx * context_len;
      std::memcpy(row_ptr, parent_ptr, context_len * sizeof(uint16_t));
    }

    if (cache_len >= 0 && (cache_len + i) < context_len) {
      row_ptr[cache_len + i] = pos_val;
    }

    if (effective_cache_len > 0 && valid_len > 0) {
      const int32_t rel_i =
          position_offset.empty() ? i : position_offset[i];
      const int32_t allowed_cache_len_raw =
          sliding_window - (rel_i + 1);
      const int32_t allowed_cache_len =
          std::min(valid_len, std::max(0, std::min(allowed_cache_len_raw, effective_cache_len)));
      const int32_t oldest_to_mask = valid_len - allowed_cache_len;
      if (oldest_to_mask > 0) {
        const int32_t oldest_start =
            (valid_len == effective_cache_len) ? write_pos : 0;
        for (int j = 0; j < oldest_to_mask; ++j) {
          row_ptr[(oldest_start + j) % effective_cache_len] = neg_val;
        }
      }
    }
  }
}

template <typename T>
void KVManager<T>::init_attention_mask_swa(
    uint16_t* attention_mask,
    const std::vector<int32_t>& attention_map,
    int32_t ar_len,
    int32_t n_past,
    int32_t context_len,
    int32_t sliding_window,
    const std::vector<int32_t>& position_offset) {
  const int32_t cache_len = context_len - ar_len;
  maybe_sync_swa_state_from_n_past_swa(n_past, cache_len);
  fill_attention_mask_swa(
      attention_mask,
      attention_map,
      ar_len,
      context_len,
      sliding_window,
      position_offset);
}

template <typename T>
void KVManager<T>::update_attention_mask_swa(
    uint16_t* attention_mask,
    const std::vector<int32_t>& attention_map,
    int32_t ar_len,
    int32_t n_past,
    int32_t n_update,
    int32_t context_len,
    int32_t sliding_window,
    const std::vector<int32_t>& position_offset) {
  const int32_t cache_len = context_len - ar_len;
  maybe_sync_swa_state_from_n_past_swa(n_past + n_update, cache_len);
  if (position_offset.empty() && is_causal_chain_attention_map(attention_map, ar_len)) {
    constexpr uint16_t neg_val = 0;
    constexpr uint16_t pos_val = 65535;

    const int32_t effective_cache_len = std::max(cache_len, 0);
    int32_t valid_len = 0;
    int32_t write_pos = 0;
    if (effective_cache_len > 0) {
      valid_len = std::min(swa_valid_len_swa_, effective_cache_len);
      write_pos = swa_write_pos_swa_ % effective_cache_len;
    }

    auto fill_ring_segment = [&](uint16_t* row_ptr,
                                 int32_t start,
                                 int32_t count,
                                 uint16_t value) {
      if (count <= 0 || effective_cache_len <= 0) {
        return;
      }
      start %= effective_cache_len;
      if (start < 0) {
        start += effective_cache_len;
      }
      const int32_t first =
          std::min(count, effective_cache_len - start);
      std::fill_n(row_ptr + start, first, value);
      if (first < count) {
        std::fill_n(row_ptr, count - first, value);
      }
    };

    // Fast path for long prefill/decode runs where SWA cache is already full
    // and remains full: only the "oldest-to-mask" segment rotates each step.
    const bool cache_full_now =
        (effective_cache_len > 0) && (valid_len == effective_cache_len);
    const bool cache_full_before = cache_full_now && (n_past >= effective_cache_len);
    if (cache_full_before && n_update > 0) {
      const int32_t shift = n_update % effective_cache_len;
      const int32_t prev_write_pos =
          (write_pos - shift + effective_cache_len) % effective_cache_len;
      for (int32_t i = 0; i < ar_len; ++i) {
        uint16_t* row_ptr =
            attention_mask +
            static_cast<size_t>(i) * static_cast<size_t>(context_len);
        const int32_t allowed_cache_len_raw = sliding_window - (i + 1);
        const int32_t allowed_cache_len =
            std::max(0, std::min(allowed_cache_len_raw, effective_cache_len));
        const int32_t oldest_to_mask = effective_cache_len - allowed_cache_len;
        if (oldest_to_mask <= 0) {
          continue;
        }
        fill_ring_segment(row_ptr, prev_write_pos, oldest_to_mask, pos_val);
        fill_ring_segment(row_ptr, write_pos, oldest_to_mask, neg_val);
      }
      return;
    }

    for (int32_t i = 0; i < ar_len; ++i) {
      uint16_t* row_ptr =
          attention_mask + static_cast<size_t>(i) * static_cast<size_t>(context_len);
      if (effective_cache_len <= 0) {
        continue;
      }
      // Only update the cache portion of the window attention mask. The
      // "new token" triangular portion (last ar_len columns) is static for the
      // common causal-chain attention_map used by prompt prefill and KV decode.
      std::fill_n(row_ptr, effective_cache_len, neg_val);
      if (valid_len <= 0) {
        continue;
      }

      const int32_t allowed_cache_len_raw = sliding_window - (i + 1);
      const int32_t allowed_cache_len =
          std::min(valid_len,
                   std::max(0,
                            std::min(allowed_cache_len_raw, effective_cache_len)));
      if (allowed_cache_len <= 0) {
        continue;
      }

      int32_t start_idx = 0;
      if (valid_len == effective_cache_len) {
        // Cache is full: newest tokens start after the masked "oldest" segment.
        start_idx = (write_pos + (effective_cache_len - allowed_cache_len)) %
            effective_cache_len;
      } else {
        // Cache is not full: tokens are laid out linearly [0..valid_len).
        start_idx = valid_len - allowed_cache_len;
      }

      const int32_t first =
          std::min(allowed_cache_len, effective_cache_len - start_idx);
      std::fill_n(row_ptr + start_idx, first, pos_val);
      if (first < allowed_cache_len) {
        std::fill_n(row_ptr, allowed_cache_len - first, pos_val);
      }
    }
    return;
  }

  fill_attention_mask_swa(
      attention_mask,
      attention_map,
      ar_len,
      context_len,
      sliding_window,
      position_offset);
}

template <typename T>
void KVManager<T>::update_key_swa(
    KVCache<T>& k_cache,
    int32_t cache_len,
    int32_t n_update,
    const std::vector<int32_t>& selected_indices) {
  if (cache_len <= 0 || n_update <= 0) {
    return;
  }
  const int32_t write_pos = swa_write_pos_swa_ % cache_len;
  T* write_ptr = k_cache.buffer;
  T* read_ptr = k_cache.output_buffer;
  const int32_t out_size = cur_ar_len_;
  const int32_t n_iter = metadata_.head_dim * metadata_.num_heads;

  if (selected_indices.empty()) {
    const size_t copy_size = static_cast<size_t>(n_update) * sizeof(T);
    for (int i = 0; i < n_iter; ++i) {
      if (write_pos + n_update <= cache_len) {
        std::memcpy(write_ptr + write_pos, read_ptr, copy_size);
      } else {
        const int32_t first = cache_len - write_pos;
        const int32_t second = n_update - first;
        std::memcpy(write_ptr + write_pos, read_ptr, first * sizeof(T));
        std::memcpy(write_ptr, read_ptr + first, second * sizeof(T));
      }
      write_ptr += cache_len;
      read_ptr += out_size;
    }
    return;
  }

  for (int i = 0; i < n_iter; ++i) {
    int32_t p = write_pos;
    for (int32_t idx : selected_indices) {
      write_ptr[p] = read_ptr[idx];
      p = (p + 1) % cache_len;
    }
    write_ptr += cache_len;
    read_ptr += out_size;
  }
}

template <typename T>
void KVManager<T>::update_value_swa(
    KVCache<T>& v_cache,
    int32_t cache_len,
    int32_t n_update,
    const std::vector<int32_t>& selected_indices) {
  if (cache_len <= 0 || n_update <= 0) {
    return;
  }
  const int32_t write_pos = swa_write_pos_swa_ % cache_len;
  T* write_ptr = v_cache.buffer;
  T* read_ptr = v_cache.output_buffer;
  const int32_t out_size = cur_ar_len_;
  const int32_t head_dim = static_cast<int32_t>(metadata_.head_dim);
  const int32_t num_heads = static_cast<int32_t>(metadata_.num_heads);

  for (int h = 0; h < num_heads; ++h) {
    T* dst = write_ptr + static_cast<size_t>(h) * cache_len * head_dim;
    T* src = read_ptr + static_cast<size_t>(h) * out_size * head_dim;

    if (selected_indices.empty()) {
      const int32_t first = std::min(n_update, cache_len - write_pos);
      const int32_t second = n_update - first;
      std::memcpy(
          dst + static_cast<size_t>(write_pos) * head_dim,
          src,
          static_cast<size_t>(first) * head_dim * sizeof(T));
      if (second > 0) {
        std::memcpy(
            dst,
            src + static_cast<size_t>(first) * head_dim,
            static_cast<size_t>(second) * head_dim * sizeof(T));
      }
      continue;
    }

    int32_t p = write_pos;
    for (int32_t idx : selected_indices) {
      std::memcpy(
          dst + static_cast<size_t>(p) * head_dim,
          src + static_cast<size_t>(idx) * head_dim,
          static_cast<size_t>(head_dim) * sizeof(T));
      p = (p + 1) % cache_len;
    }
  }
}

// Explicit instantiations
template class KVManager<uint16_t>;
template class KVManager<uint8_t>;

} // namespace example
