/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <type_traits>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace example {

// Positive per-tensor scales preserve argmax, including the first-index tie
// rule.
template <typename T>
int32_t quantized_argmax(const T* logits, int32_t size) {
#if defined(__aarch64__)
  if constexpr (std::is_same_v<T, uint16_t>) {
    const int32_t vector_end = size - size % 8;
    uint16x8_t maxima = vdupq_n_u16(0);
    for (int32_t i = 0; i < vector_end; i += 8) {
      maxima = vmaxq_u16(maxima, vld1q_u16(logits + i));
    }
    uint16_t maximum = vmaxvq_u16(maxima);
    for (int32_t i = vector_end; i < size; ++i) {
      maximum = std::max(maximum, logits[i]);
    }
    const uint16x8_t target = vdupq_n_u16(maximum);
    for (int32_t i = 0; i < vector_end; i += 8) {
      const uint16x8_t matches = vceqq_u16(vld1q_u16(logits + i), target);
      if (vmaxvq_u16(matches) != 0) {
        for (int32_t j = 0; j < 8; ++j) {
          if (logits[i + j] == maximum) {
            return i + j;
          }
        }
      }
    }
    return std::find(logits + vector_end, logits + size, maximum) - logits;
  }
#endif
  return std::max_element(logits, logits + size) - logits;
}

} // namespace example
