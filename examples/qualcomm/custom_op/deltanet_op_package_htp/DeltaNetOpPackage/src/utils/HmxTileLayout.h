// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include "HvxUtils.h"

namespace hmx_tile {
template <bool Column>
inline void pack(
    const uint16_t* source,
    uint16_t* target,
    unsigned rows,
    unsigned columns) {
  for (unsigned r = 0; r < rows; r += 2)
    for (unsigned c = 0; c < columns; c += 64) {
      const auto pair = Q6_W_vshuff_VVR(
          vmemu(source + (r + 1) * columns + c),
          vmemu(source + r * columns + c),
          -2);
      const unsigned at = Column
          ? (c / 32) * rows * 32 + (r / 2) * 64
          : (r / 32) * columns * 32 + (c / 32) * 1024 + (r % 32 / 2) * 64;
      vmemu(target + at) = Q6_V_lo_W(pair);
      vmemu(target + at + (Column ? rows * 32 : 1024)) = Q6_V_hi_W(pair);
    }
}

inline void unpack(
    const uint16_t* source,
    uint16_t* target,
    unsigned rows,
    unsigned columns) {
  for (unsigned r = 0; r < rows; r += 32)
    for (unsigned i = 0; i < 32; i += 2)
      for (unsigned c = 0; c < columns; c += 64) {
        const unsigned at =
            (r / 32) * columns * 32 + (c / 32) * 1024 + (i / 2) * 64;
        const auto pair =
            Q6_W_vdeal_VVR(vmemu(source + at + 1024), vmemu(source + at), -2);
        vmemu(target + (r + i) * columns + c) = Q6_V_lo_W(pair);
        vmemu(target + (r + i + 1) * columns + c) = Q6_V_hi_W(pair);
      }
}
} // namespace hmx_tile
