/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace example {

class RepetitionPenalty {
 public:
  static constexpr size_t kDefaultWindowSize = 256;

  explicit RepetitionPenalty(
      float penalty = 1.0f,
      size_t window_size = kDefaultWindowSize)
      : penalty_(penalty), recent_tokens_(window_size) {
    if (window_size == 0) {
      throw std::invalid_argument("repeat_last_n must be positive");
    }
  }

  bool enabled() const {
    return penalty_ != 1.0f;
  }

  void begin_turn(const std::vector<uint64_t>& prompt, bool reset) {
    if (reset) {
      token_counts_.clear();
      next_token_ = 0;
      num_tokens_ = 0;
    }
    if (enabled()) {
      for (uint64_t token : prompt) {
        record_token(token);
      }
    }
  }

  // Only commit accepted tokens, never speculative lookahead candidates.
  void record_token(uint64_t token) {
    if (!enabled()) {
      return;
    }
    if (num_tokens_ == recent_tokens_.size()) {
      auto oldest = token_counts_.find(recent_tokens_[next_token_]);
      if (--oldest->second == 0) {
        token_counts_.erase(oldest);
      }
    } else {
      ++num_tokens_;
    }
    recent_tokens_[next_token_] = token;
    next_token_ = (next_token_ + 1) % recent_tokens_.size();
    ++token_counts_[token];
  }

  void apply(
      std::vector<float>& logits,
      float scale = 1.0f,
      int64_t zero_point = 0) const {
    if (!enabled()) {
      return;
    }
    // The sign-dependent penalty must operate on real logits, not unsigned
    // quantized values. For floating-point logits use scale=1, zero_point=0.
    for (float& logit : logits) {
      logit = (logit - zero_point) * scale;
    }
    for (const auto& entry : token_counts_) {
      uint64_t token = entry.first;
      if (token < logits.size()) {
        float& logit = logits[token];
        logit = logit < 0 ? logit * penalty_ : logit / penalty_;
      }
    }
  }

 private:
  float penalty_;
  std::vector<uint64_t> recent_tokens_;
  size_t next_token_ = 0;
  size_t num_tokens_ = 0;
  std::unordered_map<uint64_t, size_t> token_counts_;
};

} // namespace example
