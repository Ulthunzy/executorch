/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <unordered_set>
#include <vector>

namespace example {

class RepetitionPenalty {
 public:
  explicit RepetitionPenalty(float penalty = 1.0f) : penalty_(penalty) {}

  bool enabled() const {
    return penalty_ != 1.0f;
  }

  void begin_turn(const std::vector<uint64_t>& prompt, bool reset) {
    if (reset) {
      seen_.clear();
    }
    if (enabled()) {
      seen_.insert(prompt.begin(), prompt.end());
    }
  }

  // Only commit accepted tokens, never speculative lookahead candidates.
  void record_token(uint64_t token) {
    if (enabled()) {
      seen_.insert(token);
    }
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
    for (uint64_t token : seen_) {
      if (token < logits.size()) {
        float& logit = logits[token];
        logit = logit < 0 ? logit * penalty_ : logit / penalty_;
      }
    }
  }

 private:
  float penalty_;
  std::unordered_set<uint64_t> seen_;
};

} // namespace example
