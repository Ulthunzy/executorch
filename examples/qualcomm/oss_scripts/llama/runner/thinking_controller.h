/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace example {

// Controls the leading reasoning block, not literal tags in the final answer.
class ThinkingController {
 public:
  enum class Reason { None, Eos, Budget, ContextLimit };

  void begin_turn(
      bool repair_eos,
      bool enable_budget,
      int32_t budget,
      uint64_t open,
      uint64_t close,
      uint64_t message_start,
      uint64_t message_end,
      const std::vector<uint64_t>& assistant_header,
      const std::vector<uint64_t>& prompt) {
    *this = ThinkingController();
    repair_eos_ = repair_eos;
    enable_budget_ = enable_budget;
    budget_ = budget;
    open_ = open;
    close_ = close;
    const auto last = std::find(prompt.rbegin(), prompt.rend(), message_start);
    if (last == prompt.rend() || assistant_header.empty()) {
      return;
    }
    const auto start = last.base() - 1;
    if (static_cast<size_t>(prompt.end() - start) < assistant_header.size() ||
        !std::equal(assistant_header.begin(), assistant_header.end(), start)) {
      return;
    }
    for (auto it = start + assistant_header.size(); it != prompt.end(); ++it) {
      if (*it == message_end) {
        depth_ = 0;
        finished_ = true;
        break;
      }
      observe(*it);
    }
  }

  uint64_t select(uint64_t sampled, bool is_eos, int64_t remaining_slots) {
    reason_ = Reason::None;
    if (!enabled() || finished_) {
      return sampled;
    }
    if (depth_ > 0 && sampled != close_) {
      if (closing_reason_ != Reason::None) {
        reason_ = closing_reason_;
      } else if (repair_eos_ && is_eos) {
        reason_ = Reason::Eos;
      } else if (enable_budget_ && generated_think_tokens_ >= budget_) {
        reason_ = Reason::Budget;
      } else if (enable_budget_ && remaining_slots <= depth_ + 1) {
        // Leave room for closing tags and at least one answer/stop token.
        reason_ = Reason::ContextLimit;
      }
      if (reason_ != Reason::None) {
        sampled = close_;
        closing_reason_ = reason_;
      }
    }
    if (depth_ > 0 && sampled != close_) {
      ++generated_think_tokens_;
    }
    observe(sampled);
    return sampled;
  }

  bool enabled() const {
    return repair_eos_ || enable_budget_;
  }
  Reason reason() const {
    return reason_;
  }
  int64_t generated_think_tokens() const {
    return generated_think_tokens_;
  }

 private:
  void observe(uint64_t token) {
    if (finished_) {
      return;
    }
    if (token == open_) {
      ++depth_;
    } else if (token == close_ && depth_ > 0) {
      if (--depth_ == 0) {
        finished_ = true;
      }
    }
  }

  bool repair_eos_{false};
  bool enable_budget_{false};
  bool finished_{false};
  int32_t budget_{512};
  int64_t depth_{0};
  int64_t generated_think_tokens_{0};
  uint64_t open_{0};
  uint64_t close_{0};
  Reason reason_{Reason::None};
  Reason closing_reason_{Reason::None};
};

} // namespace example
