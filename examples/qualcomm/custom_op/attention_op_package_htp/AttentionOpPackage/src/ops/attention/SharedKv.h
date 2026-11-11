// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
namespace attention_shared {
struct alignas(64) Control {
  uint32_t ready[2], consumed[2][2];
};
// Each reader publishes its own counter after HMX stops accessing the slot.
inline bool available(
    Control& state,
    attention_hmx::Queue& queue,
    unsigned slot,
    unsigned previous) {
  if (!previous)
    return true;
  for (unsigned worker = 0; worker < 2; ++worker) {
    while (__atomic_load_n(&state.consumed[worker][slot], __ATOMIC_ACQUIRE) !=
           previous) {
      if (__atomic_load_n(&queue.error, __ATOMIC_ACQUIRE))
        return false;
      attention_hmx::pause();
    }
  }
  return true;
}
} // namespace attention_shared
