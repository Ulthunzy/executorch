// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <cstdint>
#include "hmx_hexagon_protos.h"
namespace attention_u16 {
// Two U16 conversions expose the low/high parts of the scaled I32 result.
inline void multiply(
    const uint8_t* a,
    const uint8_t* b,
    uint16_t* low,
    uint16_t* high,
    const uint32_t* bias,
    unsigned k) {
  Q6_bias_mxmem2_A((void*)bias);
  Q6_mxclracc();
  Q6_activation_ub_mxmem_RR_deep(uintptr_t(a), ((k / 32 - 1) << 11) | 0x77c);
  Q6_weight_b_mxmem_RR(uintptr_t(b), k * 32 - 1);
  Q6_mxmem_AR_after_retain_uh_2x1(low, 0x77c);
  Q6_bias_mxmem2_A((void*)(bias + 64));
  Q6_mxmem_AR_after_uh_2x1(high, 0x77c);
}
} // namespace attention_u16
