// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <algorithm>
#include <cstdint>

namespace attention {

struct PackedKv8 {
  uint8_t* keys;
  uint8_t* values;
  uint16_t* key_sums;
  unsigned rows;

  // Preserve reserved offsets shared with existing FP16 cache allocations.
  PackedKv8(uint8_t* storage, unsigned heads, unsigned rows)
      : keys(storage),
        values(storage + heads * rows * 256 * 2),
        key_sums(reinterpret_cast<uint16_t*>(storage + heads * rows * 256)),
        rows(rows) {}

  uint8_t* key_panel(unsigned head, unsigned first) const {
    return keys + (head * rows + first) * 256;
  }

  uint8_t* value_column(
      unsigned head,
      unsigned first,
      unsigned column,
      unsigned padded) const {
    const unsigned width = std::min(256u, padded - first / 256 * 256);
    return values + (head * (rows / 256) + first / 256) * 65536 +
        column / 32 * width * 32 + first % 256 * 32;
  }
};

} // namespace attention
