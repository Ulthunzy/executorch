/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/imem_alloc.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace example {

// Structure to hold key-value cache buffers
template <typename T>
struct KVCache {
  T* buffer;
  T* output_buffer;
};

/**
 * @class KVManager
 * @brief Class for kv cache update, rearrangement, and buffer allocatation.
 */
template <typename T>
class KVManager {
 public:
  struct Metadata {
    enum CacheUpdateMode {
      kAppend,
      kCircular,
      kFullOverwrite,
    };

    struct CacheSpec {
      size_t k_input_bytes;
      size_t k_output_bytes;
      size_t v_input_bytes;
      size_t v_output_bytes;
      CacheUpdateMode update_mode;
      int32_t k_zero_point = 0;
      int32_t v_zero_point = 0;
    };

    int32_t context_len;
    int64_t head_dim;
    int32_t max_ar_len;
    int32_t max_cache_len;
    int64_t num_heads;
    int64_t num_layers;
    std::vector<CacheSpec> cache_specs;
  };
  KVManager(Metadata metadata);

  /**
   * @brief Allocate buffer for KV cache and set the cur_ar_len_.
   * @param buffer_manager Pointer to IMemAlloc instance; by default, it uses a
   * shared buffer with RPC memory.
   * @param ar_len Length of input tokens.
   */
  void init_cache(IMemAlloc* buffer_manager, int32_t ar_len);

  /**
   * @brief Reset internal KV-cache state.
   *
   * This resets the current AR length and (optionally) clears cache buffers.
   */
  void reset_state(int32_t ar_len, bool clear_buffers = false);

  /**
   * @brief Switch key and value cache from AR-cur to AR-dst.
   * @param ar_len_dst Target length of input tokens.
   */
  void rearrange_cache(int32_t ar_len_dst);

  /**
   * @brief Initialize attention mask based on kv manager mode, and attention
   * map.
   * For example,
   * ar_len = 4, CL = 6, n_past = 0,
   * attention map: {-1, 0, 1, 2} and SMART_MASK.
   * Attention_mask will be:
   * [     0     0 65535     0     0     0 ]
   * [     0     0 65535 65535     0     0 ]
   * [     0     0 65535 65535 65535     0 ]
   * [     0     0 65535 65535 65535 65535 ]
   * @param attention_mask Pointer to the attention mask array to be
   * initialized.
   * @param attention_map Vector containing the attention map values. The shape
   * of attention map should be [ar_len].
   * @param ar_len Length of input tokens.
   * @param n_past Number of past elements in the cache.
   */
  void init_attention_mask(
      uint16_t* attention_mask,
      const std::vector<int32_t>& attention_map,
      int32_t ar_len,
      int32_t n_past);

  /**
   * @brief Initialize attention mask based on kv manager mode, and attention
   * map.
   * For example,
   * ar_len = 4, CL = 6, n_past = 0,
   * attention map: {-1, 0, 1, 2} and SMART_MASK.
   * Attention_mask will be:
   * [     0     0 65535     0     0     0 ]
   * [     0     0 65535 65535     0     0 ]
   * [     0     0 65535 65535 65535     0 ]
   * [     0     0 65535 65535 65535 65535 ]
   * @param attention_mask Pointer to the attention mask array to be
   * initialized.
   * @param attention_map Vector containing the attention map values. The shape
   * of attention map should be [ar_len].
   * @param ar_len Length of input tokens.
   * @param n_past Number of past elements in the cache.
   * @param sliding_window Length of sliding window for sliding window attention
   * mask
   * @param position_offset (optional) attention mask position offset of
   */
  void init_attention_mask(
      uint16_t* attention_mask,
      const std::vector<int32_t>& attention_map,
      int32_t ar_len,
      int32_t n_past,
      int32_t sliding_window,
      const std::vector<int32_t>& position_offset = {});

  /**
   * @brief Update attention mask based on kv manager mode, and n_update.
   * @param attention_mask Pointer to the attention mask array to be
   * initialized.
   * @param ar_len Length of input tokens.
   * @param n_past Number of past elements in the cache.
   * @param n_update Number of elements to be updated.
   */
  void update_attention_mask(
      uint16_t* attention_mask,
      int32_t ar_len,
      int32_t n_past,
      int32_t n_update);

  /**
   * @brief Update attention mask based on kv manager mode, and n_update.
   * @param attention_mask Pointer to the attention mask array to be
   * initialized.
   * @param ar_len Length of input tokens.
   * @param n_past Number of past elements in the cache.
   * @param n_update Number of elements to be updated.
   * @param sliding_window Length of sliding window for sliding window attention
   * mask
   * @param position_offset (optional) attention mask position offset of
   * lookahead decoder
   */
  void update_attention_mask(
      uint16_t* attention_mask,
      int32_t ar_len,
      int32_t n_past,
      int32_t n_update,
      int32_t sliding_window,
      const std::vector<int32_t>& position_offset = {});

  /**
   * @brief Based on cur_ar_len_ to update cache
   * @param ar_len Length of input tokens.
   * @param n_past Number of past elements in the cache.
   * @param n_update Number of elements to be updated.
   * @param selected Indicate which position to be updated
   */
  void update_cache(
      int32_t ar_len,
      int32_t n_past,
      int32_t n_update,
      const std::vector<bool>& selected);

  /**
   * @brief Allocate KV cache buffers with per-layer cache length for true SWA.
   *
   * For non-SWA layers, this uses metadata_.max_cache_len to preserve the
   * existing pack/rearrange behavior across different AR lengths.
   *
   * @param layer_cache_len Length of cache (last dim) for each layer's KV input.
   * @param layer_is_swa Indicates whether each layer uses SWA cache update.
   */
  void init_cache_swa(
      IMemAlloc* buffer_manager,
      int32_t ar_len,
      const std::vector<int32_t>& layer_cache_len,
      const std::vector<bool>& layer_is_swa);

  /**
   * @brief Update the KV-cache size estimate to reflect true-SWA layer layouts.
   *
   * This should be called before reserving a shared KV buffer so the estimate
   * matches the actual per-layer allocation sizes used by init_cache_swa().
   */
  void configure_cache_size_swa(
      const std::vector<int32_t>& layer_cache_len,
      const std::vector<bool>& layer_is_swa);

  /**
   * @brief Rearrange KV cache between AR lengths, skipping SWA layers.
   */
  void rearrange_cache_swa(int32_t ar_len_dst);

  /**
   * @brief Update KV cache, using circular update for SWA layers.
   */
  void update_cache_swa(
      int32_t ar_len,
      int32_t n_past,
      int32_t n_update,
      const std::vector<bool>& selected);

  /**
   * @brief Initialize SWA window attention mask with a shorter context length.
   *
   * Note: This is intended for the window attention mask input, whose context
   * length can differ from metadata_.context_len when true SWA is enabled.
   */
  void init_attention_mask_swa(
      uint16_t* attention_mask,
      const std::vector<int32_t>& attention_map,
      int32_t ar_len,
      int32_t n_past,
      int32_t context_len,
      int32_t sliding_window,
      const std::vector<int32_t>& position_offset = {});

  /**
   * @brief Update (recompute) SWA window attention mask after cache update.
   */
  void update_attention_mask_swa(
      uint16_t* attention_mask,
      const std::vector<int32_t>& attention_map,
      int32_t ar_len,
      int32_t n_past,
      int32_t n_update,
      int32_t context_len,
      int32_t sliding_window,
      const std::vector<int32_t>& position_offset = {});

  inline bool is_true_swa_enabled_swa() const {
    return has_swa_layers_swa_;
  }

  inline int32_t swa_cache_len_swa() const {
    return swa_cache_len_swa_;
  }

  const std::vector<KVCache<T>>& get_k_cache_() const {
    return k_cache_;
  }
  const std::vector<KVCache<T>>& get_v_cache_() const {
    return v_cache_;
  }

  inline const size_t total_cache_size_in_bytes() const {
    return total_cache_size_;
  }

  int64_t get_head_dim() const {
    return metadata_.head_dim;
  }

 private:
  // Helper functions to rearrange and update key and value caches
  void rearrange_key(KVCache<T>& k_cache, int32_t ar_len_dst);
  void rearrange_value(KVCache<T>& v_cache, int32_t ar_len_dst);
  void update_key(
      KVCache<T>& k_cache,
      int32_t n_past,
      int32_t n_update,
      const std::vector<int32_t>& selected_indices);
  void update_value(
      KVCache<T>& v_cache,
      int32_t n_past,
      int32_t n_update,
      const std::vector<int32_t>& selected_indices);

  void update_full_overwrite_cache(
      const KVCache<T>& k_cache,
      const KVCache<T>& v_cache,
      int32_t layer);

  void update_key_swa(
      KVCache<T>& k_cache,
      int32_t cache_len,
      int32_t n_update,
      const std::vector<int32_t>& selected_indices);
  void update_value_swa(
      KVCache<T>& v_cache,
      int32_t cache_len,
      int32_t n_update,
      const std::vector<int32_t>& selected_indices);

  void fill_attention_mask_swa(
      uint16_t* attention_mask,
      const std::vector<int32_t>& attention_map,
      int32_t ar_len,
      int32_t context_len,
      int32_t sliding_window,
      const std::vector<int32_t>& position_offset);

  void maybe_sync_swa_state_from_n_past_swa(int32_t n_past, int32_t cache_len);

  void compact_prefill_to_swa(int32_t n_past, int32_t cache_len);

  void recompute_total_cache_size(
      const std::vector<int32_t>* layer_cache_len = nullptr,
      const std::vector<bool>* layer_is_swa = nullptr);

  // metadata
  Metadata metadata_;
  size_t total_cache_size_;
  int32_t cur_ar_len_;

  // True SWA (sliding-window attention) runtime state.
  bool has_swa_layers_swa_{false};
  int32_t swa_cache_len_swa_{0};
  int32_t swa_write_pos_swa_{0};
  int32_t swa_valid_len_swa_{0};
  std::vector<bool> layer_is_swa_;
  std::vector<int32_t> layer_cache_len_swa_;

  // Store start pointer of k and v cache for input and output
  // input: layer -> head * head_dim * cache_len_in(layer)
  // output: layer -> head * head_dim * max_ar_len
  std::vector<KVCache<T>> k_cache_;
  std::vector<KVCache<T>> v_cache_;
};
} // namespace example
