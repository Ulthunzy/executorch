/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/kv_manager.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/platform/assert.h>
#include <algorithm>
#include <cstring>

using executorch::runtime::MethodMeta;
using executorch::runtime::Result;
using executorch::runtime::TensorInfo;

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

void fill_cache_input(
    executorch::aten::ScalarType scalar_type,
    std::byte* cache,
    size_t bytes,
    int32_t zero_point) {
  switch (scalar_type) {
    case executorch::aten::ScalarType::Byte:
      std::fill_n(
          reinterpret_cast<uint8_t*>(cache),
          bytes / sizeof(uint8_t),
          static_cast<uint8_t>(zero_point));
      break;
    case executorch::aten::ScalarType::UInt16:
      std::fill_n(
          reinterpret_cast<uint16_t*>(cache),
          bytes / sizeof(uint16_t),
          static_cast<uint16_t>(zero_point));
      break;
    case executorch::aten::ScalarType::Half:
      std::fill_n(
          reinterpret_cast<executorch::aten::Half*>(cache),
          bytes / sizeof(executorch::aten::Half),
          executorch::aten::Half(static_cast<float>(zero_point)));
      break;
    case executorch::aten::ScalarType::Float:
      std::fill_n(
          reinterpret_cast<float*>(cache),
          bytes / sizeof(float),
          static_cast<float>(zero_point));
      break;
    case executorch::aten::ScalarType::Double:
      std::fill_n(
          reinterpret_cast<double*>(cache),
          bytes / sizeof(double),
          static_cast<double>(zero_point));
      break;
    case executorch::aten::ScalarType::Int:
      std::fill_n(
          reinterpret_cast<int32_t*>(cache),
          bytes / sizeof(int32_t),
          zero_point);
      break;
    case executorch::aten::ScalarType::Long:
      std::fill_n(
          reinterpret_cast<int64_t*>(cache),
          bytes / sizeof(int64_t),
          static_cast<int64_t>(zero_point));
      break;
    default:
      ET_CHECK_MSG(
          false,
          "Unsupported scalar type %s",
          executorch::runtime::toString(scalar_type));
  }
}

void fill_mask(
    executorch::aten::ScalarType scalar_type,
    std::byte* buf,
    size_t size,
    bool use_pos_value) {
  if (use_pos_value) {
    switch (scalar_type) {
      case executorch::aten::ScalarType::UInt16:
        std::fill_n(reinterpret_cast<uint16_t*>(buf), size, 65535u);
        break;
      case executorch::aten::ScalarType::Byte:
        std::fill_n(reinterpret_cast<uint8_t*>(buf), size, 255u);
        break;
      case executorch::aten::ScalarType::Float:
        std::fill_n(reinterpret_cast<float*>(buf), size, 0.0);
        break;
      default:
        ET_CHECK_MSG(
            false,
            "Unsupported scalar type %s",
            executorch::runtime::toString(scalar_type));
        break;
    }
  } else {
    switch (scalar_type) {
      case executorch::aten::ScalarType::UInt16:
        std::fill_n(reinterpret_cast<uint16_t*>(buf), size, 0u);
        break;
      case executorch::aten::ScalarType::Byte:
        std::fill_n(reinterpret_cast<uint8_t*>(buf), size, 0u);
        break;
      // -65535 acts as the additive "very negative" attention-mask value;
      // chosen as a large finite negative so masked positions effectively
      // zero out after softmax without relying on -inf.
      case executorch::aten::ScalarType::Float:
        std::fill_n(reinterpret_cast<float*>(buf), size, -65535.0);
        break;
      default:
        ET_CHECK_MSG(
            false,
            "Unsupported scalar type %s",
            executorch::runtime::toString(scalar_type));
        break;
    }
  }
}

template <typename T>
void mask_oldest_visible_impl(
    T* buf,
    size_t size,
    size_t count,
    T visible_value,
    T mask_value) {
  // Lookahead rows can be sparse, and an AR chunk can be longer than the
  // window, so the entries to drop are not necessarily a contiguous prefix.
  for (size_t i = 0; i < size && count > 0; i++) {
    if (buf[i] == visible_value) {
      buf[i] = mask_value;
      --count;
    }
  }
}

void mask_oldest_visible(
    executorch::aten::ScalarType scalar_type,
    std::byte* buf,
    size_t size,
    size_t count) {
  switch (scalar_type) {
    case executorch::aten::ScalarType::UInt16:
      mask_oldest_visible_impl(
          reinterpret_cast<uint16_t*>(buf),
          size,
          count,
          static_cast<uint16_t>(65535),
          static_cast<uint16_t>(0));
      break;
    case executorch::aten::ScalarType::Byte:
      mask_oldest_visible_impl(
          reinterpret_cast<uint8_t*>(buf),
          size,
          count,
          static_cast<uint8_t>(255),
          static_cast<uint8_t>(0));
      break;
    case executorch::aten::ScalarType::Float:
      mask_oldest_visible_impl(
          reinterpret_cast<float*>(buf), size, count, 0.0f, -65535.0f);
      break;
    default:
      ET_CHECK_MSG(
          false,
          "Unsupported scalar type %s",
          executorch::runtime::toString(scalar_type));
      break;
  }
}
} // namespace

KVManager::Metadata::CacheUpdateMode KVManager::infer_cache_update_mode(
    const TensorInfo& k_input,
    const TensorInfo& v_input,
    const TensorInfo& k_output,
    const TensorInfo& v_output,
    bool is_swa) {
  // Qwen3.5's convolution state is 3D; attention K/V caches are 4D.
  if (k_input.sizes().size() == 3) {
    ET_CHECK_MSG(
        !is_swa && k_output.sizes().size() == 3 &&
            v_input.sizes().size() == 4 && v_output.sizes().size() == 4 &&
            k_input.scalar_type() == k_output.scalar_type() &&
            v_input.scalar_type() == v_output.scalar_type() &&
            k_input.nbytes() == k_output.nbytes() &&
            v_input.nbytes() == v_output.nbytes(),
        "Recurrent cache requires matching input/output state sizes");
    return Metadata::kFullOverwrite;
  }
  ET_CHECK_MSG(
      k_input.sizes().size() == 4 && k_output.sizes().size() == 4 &&
          v_input.sizes().size() == 4 && v_output.sizes().size() == 4,
      "Attention cache requires 4D K/V tensors");
  return is_swa ? Metadata::kCircular : Metadata::kAppend;
}

KVManager::KVManager(Metadata metadata, std::unique_ptr<MethodMeta> method_meta)
    : metadata_(metadata) {
  Result<TensorInfo> attention_mask = method_meta->input_tensor_meta(1);
  attention_mask_dtype_ = attention_mask->scalar_type();

  // inputs: [tokens, atten_mask, (window_atten_mask), (input_pos), k_caches...,
  // v_caches...].
  // outputs: [logits, k_cache_0..n-1, v_cache_0..n-1]
  // k_cache shape: [bsz, n_kv_head, head_dim, ar_len]
  // v_cache v_cache shape: [bsz, n_kv_head, ar_len, head_dim]
  k_cache_.resize(metadata_.num_layers);
  v_cache_.resize(metadata_.num_layers);
  k_cache_in_nbytes_.resize(metadata_.num_layers);
  k_cache_out_nbytes_.resize(metadata_.num_layers);
  v_cache_in_nbytes_.resize(metadata_.num_layers);
  v_cache_out_nbytes_.resize(metadata_.num_layers);
  k_cache_dtypes_.resize(metadata_.num_layers);
  v_cache_dtypes_.resize(metadata_.num_layers);
  layer_is_swa_.assign(metadata_.num_layers, false);
  layer_cache_len_swa_.assign(metadata_.num_layers, 0);

  ET_CHECK_MSG(
      metadata_.cache_specs.empty() ||
          metadata_.cache_specs.size() ==
              static_cast<size_t>(metadata_.num_layers),
      "cache_specs size (%zu) must match num_layers (%ld)",
      metadata_.cache_specs.size(),
      metadata_.num_layers);

  for (int layer = 0; layer < metadata_.num_layers; ++layer) {
    // k output: index 1 + layer
    Result<TensorInfo> k_out = method_meta->output_tensor_meta(1 + layer);
    // v output: index 1 + num_layers + layer
    Result<TensorInfo> v_out =
        method_meta->output_tensor_meta(1 + metadata_.num_layers + layer);
    ET_CHECK_MSG(
        k_out.ok() && v_out.ok(),
        "Failed to read KV output metadata for layer %d",
        layer);

    const Metadata::CacheSpec* spec = metadata_.cache_specs.empty()
        ? nullptr
        : &metadata_.cache_specs[layer];
    const bool full_overwrite =
        spec != nullptr && spec->update_mode == Metadata::kFullOverwrite;
    if (!full_overwrite &&
        kv_cache_dtype_ == executorch::aten::ScalarType::Undefined) {
      kv_cache_dtype_ = k_out->scalar_type();
    }
    k_cache_dtypes_[layer] = k_out->scalar_type();
    v_cache_dtypes_[layer] = v_out->scalar_type();
    ET_CHECK_MSG(
        full_overwrite ||
            (k_out->scalar_type() == kv_cache_dtype_ &&
             v_out->scalar_type() == kv_cache_dtype_),
        "Append/circular KV caches must share their scalar type (layer %d)",
        layer);
    int64_t k_head_dim = 0;
    int64_t v_head_dim = 0;
    if (!full_overwrite) {
      ET_CHECK_MSG(
          k_out->sizes().size() == 4 && v_out->sizes().size() == 4,
          "Append/circular KV cache must use 4D K/V tensors at layer %d",
          layer);
      // k: [1, n_kv, head_dim, ar_len], v: [1, n_kv, ar_len, head_dim]
      k_head_dim = k_out->sizes()[2];
      v_head_dim = v_out->sizes()[3];
      ET_CHECK_MSG(
          k_head_dim == v_head_dim,
          "K/V head_dim mismatch at layer %d (k %d, v %d)",
          layer,
          static_cast<int>(k_head_dim),
          static_cast<int>(v_head_dim));
    }
    k_head_dim_.push_back(k_head_dim);
    v_head_dim_.push_back(v_head_dim);

    const size_t dtype_size = getDtypeSize(k_out->scalar_type());
    if (spec != nullptr) {
      k_cache_in_nbytes_[layer] = spec->k_input_bytes;
      k_cache_out_nbytes_[layer] = spec->k_output_bytes;
      v_cache_in_nbytes_[layer] = spec->v_input_bytes;
      v_cache_out_nbytes_[layer] = spec->v_output_bytes;
      if (full_overwrite) {
        ET_CHECK_MSG(
            spec->k_input_bytes == spec->k_output_bytes &&
                spec->v_input_bytes == spec->v_output_bytes,
            "Full-overwrite cache requires matching input/output byte sizes");
      } else if (spec->update_mode == Metadata::kCircular) {
        const size_t k_stride = static_cast<size_t>(metadata_.num_heads) *
            static_cast<size_t>(k_head_dim) * dtype_size;
        const size_t v_stride = static_cast<size_t>(metadata_.num_heads) *
            static_cast<size_t>(v_head_dim) * dtype_size;
        ET_CHECK_MSG(
            k_stride > 0 && v_stride > 0 &&
                spec->k_input_bytes % k_stride == 0 &&
                spec->v_input_bytes % v_stride == 0,
            "Circular KV cache byte size is inconsistent at layer %d",
            layer);
        const int32_t k_cache_len =
            static_cast<int32_t>(spec->k_input_bytes / k_stride);
        const int32_t v_cache_len =
            static_cast<int32_t>(spec->v_input_bytes / v_stride);
        ET_CHECK_MSG(
            k_cache_len == v_cache_len,
            "Circular K/V cache length mismatch at layer %d",
            layer);
        layer_is_swa_[layer] = true;
        layer_cache_len_swa_[layer] = k_cache_len;
      }
    } else {
      k_cache_out_nbytes_[layer] = metadata_.num_heads * k_head_dim *
          metadata_.max_ar_len * dtype_size;
      v_cache_out_nbytes_[layer] = metadata_.num_heads * v_head_dim *
          metadata_.max_ar_len * dtype_size;
      k_cache_in_nbytes_[layer] = metadata_.num_heads * k_head_dim *
          metadata_.max_cache_len * dtype_size;
      v_cache_in_nbytes_[layer] = metadata_.num_heads * v_head_dim *
          metadata_.max_cache_len * dtype_size;
    }
  }

  if (kv_cache_dtype_ == executorch::aten::ScalarType::Undefined &&
      !k_cache_dtypes_.empty()) {
    kv_cache_dtype_ = k_cache_dtypes_.front();
  }
  ET_CHECK_MSG(
      kv_cache_dtype_ != executorch::aten::ScalarType::Undefined,
      "kv_cache_dtype was not detected from method outputs");

  for (int layer = 0; layer < metadata_.num_layers; ++layer) {
    if (!layer_is_swa_[layer]) {
      continue;
    }
    has_swa_layers_swa_ = true;
    if (swa_cache_len_swa_ == 0) {
      swa_cache_len_swa_ = layer_cache_len_swa_[layer];
    } else {
      ET_CHECK_MSG(
          swa_cache_len_swa_ == layer_cache_len_swa_[layer],
          "All circular SWA caches must share the same cache length");
    }
  }
  recompute_total_cache_size();
};

void KVManager::recompute_total_cache_size() {
  total_cache_size_ = 0;
  for (int layer = 0; layer < metadata_.num_layers; ++layer) {
    const bool full_overwrite = !metadata_.cache_specs.empty() &&
        metadata_.cache_specs[layer].update_mode == Metadata::kFullOverwrite;
    total_cache_size_ +=
        align_shared_buffer_allocation_size(k_cache_in_nbytes_[layer]) +
        align_shared_buffer_allocation_size(v_cache_in_nbytes_[layer]);
    if (!full_overwrite) {
      total_cache_size_ +=
          align_shared_buffer_allocation_size(k_cache_out_nbytes_[layer]) +
          align_shared_buffer_allocation_size(v_cache_out_nbytes_[layer]);
    }
  }
}

void KVManager::configure_cache_size_swa(
    const std::vector<int32_t>& layer_cache_len,
    const std::vector<bool>& layer_is_swa) {
  ET_CHECK_MSG(
      layer_cache_len.size() == static_cast<size_t>(metadata_.num_layers) &&
          layer_is_swa.size() == static_cast<size_t>(metadata_.num_layers),
      "layer_cache_len/layer_is_swa sizes must match num_layers");

  layer_cache_len_swa_ = layer_cache_len;
  layer_is_swa_ = layer_is_swa;
  has_swa_layers_swa_ = false;
  swa_cache_len_swa_ = 0;
  const size_t dtype_size = getDtypeSize(kv_cache_dtype_);
  for (int layer = 0; layer < metadata_.num_layers; ++layer) {
    if (!layer_is_swa_[layer]) {
      continue;
    }
    ET_CHECK_MSG(
        layer_cache_len_swa_[layer] >= 0,
        "SWA cache length must be non-negative at layer %d",
        layer);
    has_swa_layers_swa_ = true;
    if (swa_cache_len_swa_ == 0) {
      swa_cache_len_swa_ = layer_cache_len_swa_[layer];
    } else {
      ET_CHECK_MSG(
          swa_cache_len_swa_ == layer_cache_len_swa_[layer],
          "All SWA layers must share the same cache length");
    }
    if (metadata_.cache_specs.empty()) {
      k_cache_in_nbytes_[layer] = static_cast<size_t>(metadata_.num_heads) *
          static_cast<size_t>(k_head_dim_[layer]) *
          static_cast<size_t>(layer_cache_len_swa_[layer]) * dtype_size;
      v_cache_in_nbytes_[layer] = static_cast<size_t>(metadata_.num_heads) *
          static_cast<size_t>(v_head_dim_[layer]) *
          static_cast<size_t>(layer_cache_len_swa_[layer]) * dtype_size;
    }
  }
  recompute_total_cache_size();
}

void KVManager::init_attention_mask(
    std::byte* attention_mask,
    const std::vector<int32_t>& attention_map,
    int32_t ar_len,
    int32_t n_past) {
  ET_CHECK_MSG(
      attention_map.size() <= ar_len,
      "The size of attention_map (%zu) doesn't match with ar_len (%d)",
      attention_map.size(),
      ar_len);
  // Clear the attention mask
  fill_mask(
      attention_mask_dtype_,
      attention_mask,
      ar_len * metadata_.context_len,
      /*use_pos_value=*/false);

  // SMART_MASK requires special handling of attention mask
  std::byte* past_ptr = attention_mask;
  std::byte* new_ptr = attention_mask +
      (metadata_.context_len - ar_len) * getDtypeSize(attention_mask_dtype_);
  // All inputs will necessarily attend to n_past and itself
  for (int i = 0; i < ar_len; i++) {
    // Iterate across ar_len
    if (attention_map[i] < 0) {
      // If negative, attend to only past tokens
      fill_mask(
          attention_mask_dtype_,
          past_ptr,
          n_past,
          /*use_pos_value=*/true);
    } else {
      // If positive, copy attention map from (relative to 0th input) parent
      // Parent token index
      const int32_t pidx = attention_map[i];
      std::byte* parent_ptr = attention_mask +
          pidx * metadata_.context_len * getDtypeSize(attention_mask_dtype_);
      std::memcpy(
          past_ptr,
          parent_ptr,
          metadata_.context_len * getDtypeSize(attention_mask_dtype_));
    }
    // Attend to itself
    fill_mask(
        attention_mask_dtype_,
        new_ptr + i * getDtypeSize(attention_mask_dtype_),
        1,
        /*use_pos_value=*/true);
    past_ptr += metadata_.context_len * getDtypeSize(attention_mask_dtype_);
    new_ptr += metadata_.context_len * getDtypeSize(attention_mask_dtype_);
  }
}

void KVManager::init_attention_mask(
    std::byte* attention_mask,
    const std::vector<int32_t>& attention_map,
    int32_t ar_len,
    int32_t n_past,
    int32_t sliding_window,
    const std::vector<int32_t>& /*position_offset*/) {
  ET_CHECK_MSG(
      attention_map.size() >= static_cast<size_t>(ar_len),
      "attention_map size (%zu) must be >= ar_len (%d)",
      attention_map.size(),
      ar_len);
  // Clear the attention mask
  fill_mask(
      attention_mask_dtype_,
      attention_mask,
      ar_len * metadata_.context_len,
      /*use_pos_value=*/false);

  // SMART_MASK requires special handling of attention mask
  std::byte* past_ptr = attention_mask;
  std::byte* new_ptr = attention_mask +
      (metadata_.context_len - ar_len) * getDtypeSize(attention_mask_dtype_);
  const size_t window_size = static_cast<size_t>(sliding_window);
  std::vector<size_t> visible_counts(ar_len);
  // All inputs will necessarily attend to n_past and itself
  for (int i = 0; i < ar_len; i++) {
    size_t visible_count;
    // Iterate across ar_len
    if (attention_map[i] < 0) {
      // If negative, attend to only past tokens
      fill_mask(
          attention_mask_dtype_,
          past_ptr,
          n_past,
          /*use_pos_value=*/true);
      visible_count = n_past;
    } else {
      // If positive, copy attention map from (relative to 0th input) parent
      // Parent token index
      const int32_t pidx = attention_map[i];
      std::byte* parent_ptr = attention_mask +
          pidx * metadata_.context_len * getDtypeSize(attention_mask_dtype_);
      std::memcpy(
          past_ptr,
          parent_ptr,
          metadata_.context_len * getDtypeSize(attention_mask_dtype_));
      visible_count = visible_counts[pidx];
    }
    // Attend to itself
    fill_mask(
        attention_mask_dtype_,
        new_ptr + i * getDtypeSize(attention_mask_dtype_),
        1,
        /*use_pos_value=*/true);
    ++visible_count;
    if (visible_count > window_size) {
      mask_oldest_visible(
          attention_mask_dtype_,
          past_ptr,
          metadata_.context_len,
          visible_count - window_size);
      visible_count = window_size;
    }
    visible_counts[i] = visible_count;

    past_ptr += metadata_.context_len * getDtypeSize(attention_mask_dtype_);
    new_ptr += metadata_.context_len * getDtypeSize(attention_mask_dtype_);
  }
}

void KVManager::update_attention_mask(
    std::byte* attention_mask,
    int32_t ar_len,
    int32_t n_past,
    int32_t n_update) {
  std::byte* cur_ptr =
      attention_mask + n_past * getDtypeSize(attention_mask_dtype_);

  for (int i = 0; i < ar_len; i++) {
    fill_mask(attention_mask_dtype_, cur_ptr, n_update, /*use_pos_value=*/true);
    cur_ptr += metadata_.context_len * getDtypeSize(attention_mask_dtype_);
  }
}

void KVManager::update_attention_mask(
    std::byte* attention_mask,
    int32_t ar_len,
    int32_t n_past,
    int32_t n_update,
    int32_t sliding_window,
    const std::vector<int32_t>& position_offset) {
  ET_CHECK_MSG(
      position_offset.empty() ||
          position_offset.size() >= static_cast<size_t>(ar_len),
      "position_offset size (%zu) must be empty or >= ar_len (%d)",
      position_offset.size(),
      ar_len);
  std::byte* cur_ptr =
      attention_mask + n_past * getDtypeSize(attention_mask_dtype_);
  const size_t window_size = static_cast<size_t>(sliding_window);

  for (int i = 0; i < ar_len; i++) {
    fill_mask(attention_mask_dtype_, cur_ptr, n_update, /*use_pos_value=*/true);
    const int32_t position = position_offset.empty() ? i : position_offset[i];
    const size_t visible_count =
        std::min(static_cast<size_t>(n_past + position + 1), window_size) +
        n_update;
    if (visible_count > window_size) {
      mask_oldest_visible(
          attention_mask_dtype_,
          cur_ptr - n_past * getDtypeSize(attention_mask_dtype_),
          metadata_.context_len,
          visible_count - window_size);
    }
    cur_ptr += metadata_.context_len * getDtypeSize(attention_mask_dtype_);
  }
}

bool KVManager::has_inplace_cache_updates() const {
  for (size_t layer = 0; layer < k_cache_.size(); ++layer) {
    if (k_cache_[layer].buffer == k_cache_[layer].output_buffer ||
        v_cache_[layer].buffer == v_cache_[layer].output_buffer) {
      return true;
    }
  }
  return false;
}

void KVManager::init_cache(IMemAlloc* buffer_manager, int32_t ar_len) {
  cur_ar_len_ = ar_len;
  swa_write_pos_swa_ = 0;
  swa_valid_len_swa_ = 0;
  for (int layer = 0; layer < metadata_.num_layers; ++layer) {
    const bool full_overwrite = !metadata_.cache_specs.empty() &&
        metadata_.cache_specs[layer].update_mode == Metadata::kFullOverwrite;
    k_cache_[layer].buffer = buffer_manager->allocate(k_cache_in_nbytes_[layer]);
    k_cache_[layer].output_buffer = full_overwrite
        ? k_cache_[layer].buffer
        : buffer_manager->allocate(k_cache_out_nbytes_[layer]);
    v_cache_[layer].buffer = buffer_manager->allocate(v_cache_in_nbytes_[layer]);
    v_cache_[layer].output_buffer = full_overwrite
        ? v_cache_[layer].buffer
        : buffer_manager->allocate(v_cache_out_nbytes_[layer]);

    const int32_t k_zero_point = metadata_.cache_specs.empty()
        ? 0
        : metadata_.cache_specs[layer].k_zero_point;
    const int32_t v_zero_point = metadata_.cache_specs.empty()
        ? 0
        : metadata_.cache_specs[layer].v_zero_point;
    fill_cache_input(
        k_cache_dtypes_[layer],
        k_cache_[layer].buffer,
        k_cache_in_nbytes_[layer],
        k_zero_point);
    fill_cache_input(
        v_cache_dtypes_[layer],
        v_cache_[layer].buffer,
        v_cache_in_nbytes_[layer],
        v_zero_point);
    if (!full_overwrite) {
      std::memset(
          k_cache_[layer].output_buffer, 0, k_cache_out_nbytes_[layer]);
      std::memset(
          v_cache_[layer].output_buffer, 0, v_cache_out_nbytes_[layer]);
    }
  }
}

void KVManager::reset_state(int32_t ar_len, bool clear_buffers) {
  cur_ar_len_ = ar_len;
  swa_write_pos_swa_ = 0;
  swa_valid_len_swa_ = 0;

  if (!clear_buffers) {
    return;
  }

  for (int layer = 0; layer < metadata_.num_layers; ++layer) {
    const int32_t k_zero_point = metadata_.cache_specs.empty()
        ? 0
        : metadata_.cache_specs[layer].k_zero_point;
    const int32_t v_zero_point = metadata_.cache_specs.empty()
        ? 0
        : metadata_.cache_specs[layer].v_zero_point;
    fill_cache_input(
        k_cache_dtypes_[layer],
        k_cache_[layer].buffer,
        k_cache_in_nbytes_[layer],
        k_zero_point);
    fill_cache_input(
        v_cache_dtypes_[layer],
        v_cache_[layer].buffer,
        v_cache_in_nbytes_[layer],
        v_zero_point);
    if (k_cache_[layer].output_buffer != k_cache_[layer].buffer) {
      std::memset(
          k_cache_[layer].output_buffer, 0, k_cache_out_nbytes_[layer]);
    }
    if (v_cache_[layer].output_buffer != v_cache_[layer].buffer) {
      std::memset(
          v_cache_[layer].output_buffer, 0, v_cache_out_nbytes_[layer]);
    }
  }
}

void KVManager::rearrange_cache(int32_t ar_len_dst) {
  // Don't need to rearrange if cur_ar_len_ is equal to target ar_len
  if (cur_ar_len_ == ar_len_dst)
    return;
  for (int layer = 0; layer < metadata_.num_layers; ++layer) {
    const bool state_or_circular = !metadata_.cache_specs.empty() &&
        metadata_.cache_specs[layer].update_mode != Metadata::kAppend;
    if (state_or_circular ||
        (!layer_is_swa_.empty() && layer_is_swa_[layer])) {
      continue;
    }
    rearrange_key(k_cache_[layer], ar_len_dst, k_head_dim_[layer]);
    rearrange_value(v_cache_[layer], ar_len_dst, v_head_dim_[layer]);
  }
  // rearrange done.
  cur_ar_len_ = ar_len_dst;
}

void KVManager::rearrange_key(
    KVCache& k_cache,
    int32_t ar_len_dst,
    int64_t head_dim) {
  const int32_t src_cache_num = (cur_ar_len_ == metadata_.context_len)
      ? metadata_.context_len
      : metadata_.context_len - cur_ar_len_;
  const int32_t dst_cache_num = metadata_.context_len - ar_len_dst;
  std::byte* k_cache_in_read_ptr = k_cache.buffer;
  std::byte* k_cache_in_write_ptr = k_cache.buffer;
  size_t src_cache_nbytes = src_cache_num * getDtypeSize(kv_cache_dtype_);
  size_t dst_cache_nbytes = dst_cache_num * getDtypeSize(kv_cache_dtype_);
  if (src_cache_num > dst_cache_num) {
    // copy from first dimension
    for (int i = 0; i < head_dim * metadata_.num_heads; i++) {
      std::memmove(k_cache_in_write_ptr, k_cache_in_read_ptr, dst_cache_nbytes);
      k_cache_in_read_ptr += src_cache_nbytes;
      k_cache_in_write_ptr += dst_cache_nbytes;
    }
  } else {
    k_cache_in_read_ptr +=
        (head_dim * metadata_.num_heads - 1) * src_cache_nbytes;
    k_cache_in_write_ptr +=
        (head_dim * metadata_.num_heads - 1) * dst_cache_nbytes;
    // copy from last dimension
    for (int i = 0; i < head_dim * metadata_.num_heads; i++) {
      std::memmove(k_cache_in_write_ptr, k_cache_in_read_ptr, src_cache_nbytes);
      k_cache_in_read_ptr -= src_cache_nbytes;
      k_cache_in_write_ptr -= dst_cache_nbytes;
    }
  }
}

void KVManager::rearrange_value(
    KVCache& v_cache,
    int32_t ar_len_dst,
    int64_t head_dim) {
  const int32_t src_cache_num = (cur_ar_len_ == metadata_.context_len)
      ? metadata_.context_len
      : metadata_.context_len - cur_ar_len_;
  const int32_t dst_cache_num = metadata_.context_len - ar_len_dst;
  std::byte* v_cache_in_read_ptr = v_cache.buffer;
  std::byte* v_cache_in_write_ptr = v_cache.buffer;
  size_t src_cache_nbytes = src_cache_num * getDtypeSize(kv_cache_dtype_);
  size_t dst_cache_nbytes = dst_cache_num * getDtypeSize(kv_cache_dtype_);
  if (src_cache_num > dst_cache_num) {
    // copy from first dimension
    for (int i = 0; i < metadata_.num_heads; i++) {
      std::memmove(
          v_cache_in_write_ptr,
          v_cache_in_read_ptr,
          dst_cache_nbytes * head_dim);
      v_cache_in_read_ptr += src_cache_nbytes * head_dim;
      v_cache_in_write_ptr += dst_cache_nbytes * head_dim;
    }
  } else {
    v_cache_in_read_ptr +=
        head_dim * (metadata_.num_heads - 1) * src_cache_nbytes;
    v_cache_in_write_ptr +=
        head_dim * (metadata_.num_heads - 1) * dst_cache_nbytes;
    // copy from last dimension
    for (int i = 0; i < metadata_.num_heads; i++) {
      std::memmove(
          v_cache_in_write_ptr,
          v_cache_in_read_ptr,
          src_cache_nbytes * head_dim);
      v_cache_in_read_ptr -= src_cache_nbytes * head_dim;
      v_cache_in_write_ptr -= dst_cache_nbytes * head_dim;
    }
  }
}

void KVManager::update_cache(
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
  int32_t cache_ar_len = cur_ar_len_;
  if (cur_ar_len_ != metadata_.context_len &&
      n_past + n_update > metadata_.context_len - cur_ar_len_) {
    for (int layer = 0; layer < metadata_.num_layers; ++layer) {
      if ((!metadata_.cache_specs.empty() &&
           metadata_.cache_specs[layer].update_mode != Metadata::kAppend) ||
          (!layer_is_swa_.empty() && layer_is_swa_[layer])) {
        continue;
      }
      ET_CHECK_MSG(
          n_past + n_update <= metadata_.max_cache_len,
          "Cache update exceeds the allocated capacity (%d)",
          metadata_.max_cache_len);
      cache_ar_len = metadata_.context_len - metadata_.max_cache_len;
      break;
    }
  }
  bool updated_circular_cache = false;
  for (int layer = 0; layer < metadata_.num_layers; ++layer) {
    if (!metadata_.cache_specs.empty() &&
        metadata_.cache_specs[layer].update_mode == Metadata::kFullOverwrite) {
      update_full_overwrite_cache(k_cache_[layer], v_cache_[layer], layer);
      continue;
    }
    if (!layer_is_swa_.empty() && layer_is_swa_[layer]) {
      update_key_swa(
          k_cache_[layer],
          layer_cache_len_swa_[layer],
          n_update,
          selected_indices,
          k_head_dim_[layer]);
      update_value_swa(
          v_cache_[layer],
          layer_cache_len_swa_[layer],
          n_update,
          selected_indices,
          v_head_dim_[layer]);
      updated_circular_cache = true;
      continue;
    }
    // The final prefill chunk can fill the larger decode cache. Expand each
    // input layout before appending, while retaining the prefill output stride.
    if (cache_ar_len != cur_ar_len_) {
      rearrange_key(k_cache_[layer], cache_ar_len, k_head_dim_[layer]);
      rearrange_value(v_cache_[layer], cache_ar_len, v_head_dim_[layer]);
    }
    update_key(
        k_cache_[layer],
        cache_ar_len,
        n_past,
        n_update,
        selected_indices,
        k_head_dim_[layer]);
    update_value(
        v_cache_[layer],
        cache_ar_len,
        n_past,
        n_update,
        selected_indices,
        v_head_dim_[layer]);
  }
  cur_ar_len_ = cache_ar_len;
  if (updated_circular_cache && swa_cache_len_swa_ > 0 && n_update > 0) {
    swa_write_pos_swa_ =
        (swa_write_pos_swa_ + n_update) % swa_cache_len_swa_;
    swa_valid_len_swa_ =
        std::min(swa_cache_len_swa_, swa_valid_len_swa_ + n_update);
  }
}

void KVManager::update_key(
    KVCache& k_cache,
    int32_t cache_ar_len,
    int32_t n_past,
    int32_t n_update,
    const std::vector<int32_t>& selected_indices,
    int64_t head_dim) {
  std::byte* write_ptr = k_cache.buffer;
  std::byte* read_ptr = k_cache.output_buffer;
  const size_t element_size = getDtypeSize(kv_cache_dtype_);
  const size_t copy_size = static_cast<size_t>(n_update) * element_size;
  const int32_t iter_size = (cache_ar_len == metadata_.context_len)
      ? metadata_.context_len * element_size
      : (metadata_.context_len - cache_ar_len) * element_size;
  const int32_t out_size = cur_ar_len_ * element_size;
  const int32_t past_size = n_past * element_size;
  const int32_t n_iter = head_dim * metadata_.num_heads;

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
        std::memmove(
            wp,
            rp + static_cast<size_t>(ind) * element_size,
            element_size);
        wp += element_size;
      }
      write_ptr += iter_size;
      read_ptr += out_size;
    }
  }
}

void KVManager::update_value(
    KVCache& v_cache,
    int32_t cache_ar_len,
    int32_t n_past,
    int32_t n_update,
    const std::vector<int32_t>& selected_indices,
    int64_t head_dim) {
  std::byte* write_ptr = v_cache.buffer;
  std::byte* read_ptr = v_cache.output_buffer;
  const size_t element_size = getDtypeSize(kv_cache_dtype_);
  const size_t copy_size =
      static_cast<size_t>(n_update) * head_dim * element_size;
  const size_t past_size = static_cast<size_t>(n_past) * head_dim * element_size;
  const int32_t n_iter = metadata_.num_heads;
  const int32_t iter_size = (cache_ar_len == metadata_.context_len)
      ? metadata_.context_len * head_dim * getDtypeSize(kv_cache_dtype_)
      : (metadata_.context_len - cache_ar_len) * head_dim *
          getDtypeSize(kv_cache_dtype_);
  const int32_t out_size =
      cur_ar_len_ * head_dim * getDtypeSize(kv_cache_dtype_);

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
            rp + static_cast<size_t>(ind) * head_dim * element_size,
            head_dim * element_size);
        wp += head_dim * element_size;
      }
      write_ptr += iter_size;
      read_ptr += out_size;
    }
  }
}

void KVManager::update_full_overwrite_cache(
    const KVCache& k_cache,
    const KVCache& v_cache,
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

void KVManager::init_cache_swa(
    IMemAlloc* buffer_manager,
    int32_t ar_len,
    const std::vector<int32_t>& layer_cache_len,
    const std::vector<bool>& layer_is_swa) {
  if (layer_cache_len.empty() || layer_is_swa.empty()) {
    init_cache(buffer_manager, ar_len);
    return;
  }
  configure_cache_size_swa(layer_cache_len, layer_is_swa);
  init_cache(buffer_manager, ar_len);
}

void KVManager::rearrange_cache_swa(int32_t ar_len_dst) {
  rearrange_cache(ar_len_dst);
}

void KVManager::update_cache_swa(
    int32_t ar_len,
    int32_t n_past,
    int32_t n_update,
    const std::vector<bool>& selected) {
  update_cache(ar_len, n_past, n_update, selected);
}

void KVManager::maybe_sync_swa_state_from_n_past_swa(
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

void KVManager::fill_attention_mask_swa(
    std::byte* attention_mask,
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

  const size_t element_size = getDtypeSize(attention_mask_dtype_);
  fill_mask(
      attention_mask_dtype_,
      attention_mask,
      static_cast<size_t>(ar_len) * context_len,
      /*use_pos_value=*/false);

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
    std::byte* row_ptr = attention_mask +
        static_cast<size_t>(i) * context_len * element_size;
    if (attention_map[i] < 0) {
      if (valid_len > 0) {
        fill_mask(
            attention_mask_dtype_,
            row_ptr,
            valid_len,
            /*use_pos_value=*/true);
      }
    } else {
      const int32_t pidx = attention_map[i];
      ET_CHECK_MSG(
          pidx >= 0 && pidx < ar_len,
          "Invalid attention_map parent index %d for row %d",
          pidx,
          i);
      std::byte* parent_ptr = attention_mask +
          static_cast<size_t>(pidx) * context_len * element_size;
      std::memcpy(row_ptr, parent_ptr, context_len * element_size);
    }

    if (cache_len >= 0 && (cache_len + i) < context_len) {
      fill_mask(
          attention_mask_dtype_,
          row_ptr + static_cast<size_t>(cache_len + i) * element_size,
          1,
          /*use_pos_value=*/true);
    }

    const int32_t rel_i = position_offset.empty() ? i : position_offset[i];
    if (position_offset.empty()) {
      const int32_t expired_tokens = std::max(0, i + 1 - sliding_window);
      if (expired_tokens > 0) {
        fill_mask(
            attention_mask_dtype_,
            row_ptr + static_cast<size_t>(effective_cache_len) * element_size,
            expired_tokens,
            /*use_pos_value=*/false);
      }
    } else {
      for (int32_t j = 0; j < ar_len; ++j) {
        if (rel_i - position_offset[j] >= sliding_window) {
          fill_mask(
              attention_mask_dtype_,
              row_ptr +
                  static_cast<size_t>(effective_cache_len + j) * element_size,
              1,
              /*use_pos_value=*/false);
        }
      }
    }

    if (effective_cache_len > 0 && valid_len > 0) {
      const int32_t allowed_cache_len_raw =
          sliding_window - (rel_i + 1);
      const int32_t allowed_cache_len = std::min(
          valid_len,
          std::max(
              0, std::min(allowed_cache_len_raw, effective_cache_len)));
      const int32_t oldest_to_mask = valid_len - allowed_cache_len;
      if (oldest_to_mask > 0) {
        const int32_t oldest_start =
            (valid_len == effective_cache_len) ? write_pos : 0;
        for (int j = 0; j < oldest_to_mask; ++j) {
          fill_mask(
              attention_mask_dtype_,
              row_ptr + static_cast<size_t>(
                            (oldest_start + j) % effective_cache_len) *
                  element_size,
              1,
              /*use_pos_value=*/false);
        }
      }
    }
  }
}

void KVManager::init_attention_mask_swa(
    std::byte* attention_mask,
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

void KVManager::update_attention_mask_swa(
    std::byte* attention_mask,
    const std::vector<int32_t>& attention_map,
    int32_t ar_len,
    int32_t n_past,
    int32_t n_update,
    int32_t context_len,
    int32_t sliding_window,
    const std::vector<int32_t>& position_offset) {
  const int32_t cache_len = context_len - ar_len;
  maybe_sync_swa_state_from_n_past_swa(n_past + n_update, cache_len);
  if (position_offset.empty() &&
      is_causal_chain_attention_map(attention_map, ar_len)) {
    const size_t element_size = getDtypeSize(attention_mask_dtype_);
    const int32_t effective_cache_len = std::max(cache_len, 0);
    int32_t valid_len = 0;
    int32_t write_pos = 0;
    if (effective_cache_len > 0) {
      valid_len = std::min(swa_valid_len_swa_, effective_cache_len);
      write_pos = swa_write_pos_swa_ % effective_cache_len;
    }

    auto fill_ring_segment = [&](std::byte* row_ptr,
                                 int32_t start,
                                 int32_t count,
                                 bool visible) {
      if (count <= 0 || effective_cache_len <= 0) {
        return;
      }
      start %= effective_cache_len;
      if (start < 0) {
        start += effective_cache_len;
      }
      const int32_t first =
          std::min(count, effective_cache_len - start);
      fill_mask(
          attention_mask_dtype_,
          row_ptr + static_cast<size_t>(start) * element_size,
          first,
          visible);
      if (first < count) {
        fill_mask(
            attention_mask_dtype_, row_ptr, count - first, visible);
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
        std::byte* row_ptr = attention_mask + static_cast<size_t>(i) *
            static_cast<size_t>(context_len) * element_size;
        const int32_t allowed_cache_len_raw = sliding_window - (i + 1);
        const int32_t allowed_cache_len =
            std::max(0, std::min(allowed_cache_len_raw, effective_cache_len));
        const int32_t oldest_to_mask = effective_cache_len - allowed_cache_len;
        if (oldest_to_mask <= 0) {
          continue;
        }
        fill_ring_segment(
            row_ptr, prev_write_pos, oldest_to_mask, /*visible=*/true);
        fill_ring_segment(
            row_ptr, write_pos, oldest_to_mask, /*visible=*/false);
      }
      return;
    }

    for (int32_t i = 0; i < ar_len; ++i) {
      std::byte* row_ptr = attention_mask + static_cast<size_t>(i) *
          static_cast<size_t>(context_len) * element_size;
      if (effective_cache_len <= 0) {
        continue;
      }
      // Only update the cache portion of the window attention mask. The
      // "new token" triangular portion (last ar_len columns) is static for the
      // common causal-chain attention_map used by prompt prefill and KV decode.
      fill_mask(
          attention_mask_dtype_,
          row_ptr,
          effective_cache_len,
          /*use_pos_value=*/false);
      if (valid_len <= 0) {
        continue;
      }

      const int32_t allowed_cache_len_raw = sliding_window - (i + 1);
      const int32_t allowed_cache_len = std::min(
          valid_len,
          std::max(
              0, std::min(allowed_cache_len_raw, effective_cache_len)));
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
      fill_mask(
          attention_mask_dtype_,
          row_ptr + static_cast<size_t>(start_idx) * element_size,
          first,
          /*use_pos_value=*/true);
      if (first < allowed_cache_len) {
        fill_mask(
            attention_mask_dtype_,
            row_ptr,
            allowed_cache_len - first,
            /*use_pos_value=*/true);
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

void KVManager::update_key_swa(
    KVCache& k_cache,
    int32_t cache_len,
    int32_t n_update,
    const std::vector<int32_t>& selected_indices,
    int64_t head_dim) {
  if (cache_len <= 0 || n_update <= 0) {
    return;
  }
  const int32_t write_pos = swa_write_pos_swa_ % cache_len;
  std::byte* write_ptr = k_cache.buffer;
  std::byte* read_ptr = k_cache.output_buffer;
  const size_t element_size = getDtypeSize(kv_cache_dtype_);
  const int32_t out_size = cur_ar_len_;
  const int32_t n_iter = head_dim * metadata_.num_heads;

  if (selected_indices.empty()) {
    const int32_t copy_count = std::min(n_update, cache_len);
    const int32_t source_offset = n_update - copy_count;
    const int32_t copy_write_pos = (write_pos + source_offset) % cache_len;
    for (int i = 0; i < n_iter; ++i) {
      const int32_t first = std::min(copy_count, cache_len - copy_write_pos);
      const int32_t second = copy_count - first;
      std::memcpy(
          write_ptr + static_cast<size_t>(copy_write_pos) * element_size,
          read_ptr + static_cast<size_t>(source_offset) * element_size,
          static_cast<size_t>(first) * element_size);
      if (second > 0) {
        std::memcpy(
            write_ptr,
            read_ptr +
                static_cast<size_t>(source_offset + first) * element_size,
            static_cast<size_t>(second) * element_size);
      }
      write_ptr += static_cast<size_t>(cache_len) * element_size;
      read_ptr += static_cast<size_t>(out_size) * element_size;
    }
    return;
  }

  for (int i = 0; i < n_iter; ++i) {
    int32_t p = write_pos;
    for (int32_t idx : selected_indices) {
      std::memcpy(
          write_ptr + static_cast<size_t>(p) * element_size,
          read_ptr + static_cast<size_t>(idx) * element_size,
          element_size);
      p = (p + 1) % cache_len;
    }
    write_ptr += static_cast<size_t>(cache_len) * element_size;
    read_ptr += static_cast<size_t>(out_size) * element_size;
  }
}

void KVManager::update_value_swa(
    KVCache& v_cache,
    int32_t cache_len,
    int32_t n_update,
    const std::vector<int32_t>& selected_indices,
    int64_t head_dim) {
  if (cache_len <= 0 || n_update <= 0) {
    return;
  }
  const int32_t write_pos = swa_write_pos_swa_ % cache_len;
  std::byte* write_ptr = v_cache.buffer;
  std::byte* read_ptr = v_cache.output_buffer;
  const size_t element_size = getDtypeSize(kv_cache_dtype_);
  const int32_t out_size = cur_ar_len_;
  const int32_t num_heads = static_cast<int32_t>(metadata_.num_heads);

  for (int h = 0; h < num_heads; ++h) {
    std::byte* dst = write_ptr +
        static_cast<size_t>(h) * cache_len * head_dim * element_size;
    std::byte* src =
        read_ptr + static_cast<size_t>(h) * out_size * head_dim * element_size;

    if (selected_indices.empty()) {
      const int32_t copy_count = std::min(n_update, cache_len);
      const int32_t source_offset = n_update - copy_count;
      const int32_t copy_write_pos = (write_pos + source_offset) % cache_len;
      const int32_t first = std::min(copy_count, cache_len - copy_write_pos);
      const int32_t second = copy_count - first;
      std::memcpy(
          dst + static_cast<size_t>(copy_write_pos) * head_dim * element_size,
          src + static_cast<size_t>(source_offset) * head_dim * element_size,
          static_cast<size_t>(first) * head_dim * element_size);
      if (second > 0) {
        std::memcpy(
            dst,
            src +
                static_cast<size_t>(source_offset + first) * head_dim *
                    element_size,
            static_cast<size_t>(second) * head_dim * element_size);
      }
      continue;
    }

    int32_t p = write_pos;
    for (int32_t idx : selected_indices) {
      std::memcpy(
          dst + static_cast<size_t>(p) * head_dim * element_size,
          src + static_cast<size_t>(idx) * head_dim * element_size,
          static_cast<size_t>(head_dim) * element_size);
      p = (p + 1) % cache_len;
    }
  }
}

} // namespace example
