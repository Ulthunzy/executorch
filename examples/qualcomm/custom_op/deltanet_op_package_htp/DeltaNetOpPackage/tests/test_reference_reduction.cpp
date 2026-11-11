// Copyright (c) Qualcomm Innovation Center, Inc.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef DELTANET_U16_KARATSUBA_REDUCTION
#define DELTANET_U16_KARATSUBA_REDUCTION 1
#endif
#include <cstdio>
#include <cstring>
#include <random>
#include "../src/kernels/DeltaNetU16.h"

int main() {
  alignas(128) uint16_t state[128 * 128];
  alignas(128) int32_t key[128], query[128];
  alignas(128) float actual_k[128], actual_q[128];
  std::mt19937 random(87508850);
  size_t checked = 0;
  for (const int state_zero : {0, 1, 256, 32768, 40024, 65535})
    for (const int key_zero : {0, 32768, 40024, 65535})
      for (int trial = 0; trial < 25; ++trial) {
        for (int k = 0; k < 128; ++k) {
          key[k] = int(random() & 65535) - key_zero;
          query[k] = int(random() & 65535) - (65535 - key_zero);
        }
        for (auto& value : state)
          value = trial == 0 ? 0 : trial == 1 ? 65535 : uint16_t(random());
        deltanetQhpiHvxReduceStateU16<true, 2>(
            state,
            actual_k,
            actual_q,
            state_zero,
            key_zero,
            65535 - key_zero,
            key,
            query,
            128,
            128,
            128);
        for (int projection = 0; projection < 2; ++projection)
          for (int v = 0; v < 128; ++v) {
            const auto* coefficients = projection ? query : key;
            int64_t high = 0, cross = 0, low = 0;
            for (int k = 0; k < 128; ++k) {
              const int raw = state[k * 128 + v];
              const int sh = (raw >> 4) - (state_zero >> 4);
              const int sl = (raw & 15) - (state_zero & 15);
              const int kh = coefficients[k] >> 4;
              const int kl = coefficients[k] & 15;
              high += int64_t(sh) * kh;
              cross += int64_t(sh) * kl + int64_t(sl) * kh;
              low += int64_t(sl) * kl;
            }
            // The original radix-16 FP32 combine is part of the numerical
            // contract; an exact int64 dot cannot detect reassociation.
            volatile float first = float(high) * 256.f + float(cross) * 16.f;
            const float expected = first + float(low);
            const float actual = projection ? actual_q[v] : actual_k[v];
            if (std::memcmp(&actual, &expected, sizeof(float))) {
              std::printf(
                  "FAIL state_zero=%d key_zero=%d trial=%d projection=%d lane=%d actual=%.9g reference=%.9g\n",
                  state_zero,
                  key_zero,
                  trial,
                  projection,
                  v,
                  actual,
                  expected);
              return 1;
            }
            ++checked;
          }
      }
  std::printf(
      "reference FP32 reduction bits matched: %zu projection lanes\n", checked);
}
