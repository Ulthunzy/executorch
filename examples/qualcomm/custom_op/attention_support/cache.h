// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace attention_cache {

constexpr uint32_t packed_storage_flag = 1u << 31;
constexpr unsigned max_packed_rows = 16640;

inline unsigned packed_rows(unsigned capacity) {
  return std::min(max_packed_rows, (capacity + 511u) & ~255u);
}

inline size_t packed_offset(size_t logical_bytes) {
  return (logical_bytes + 4095u) & ~size_t(4095u);
}

inline size_t
packed_bytes(unsigned heads, unsigned dimension, unsigned capacity) {
  return size_t(heads) * dimension * packed_rows(capacity) * 4;
}

} // namespace attention_cache
