// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>

namespace attention {

constexpr uint32_t exact_append_flag = 1u << 30;
constexpr uint32_t logical_size_mask = exact_append_flag - 1;
constexpr unsigned max_workers = 8;
constexpr unsigned fast_kv_rows = 16640;
constexpr unsigned online_threshold = 8192;

struct Geometry {
  unsigned batch, heads, queries, dimension, kv_heads;

  bool valid() const {
    return batch && heads && queries && dimension && dimension <= 4096 &&
        kv_heads && heads % kv_heads == 0 &&
        batch <= unsigned(INT32_MAX) / heads &&
        batch * heads <= unsigned(INT32_MAX) / queries &&
        batch * heads * queries <= unsigned(INT32_MAX) / dimension;
  }

  bool fast() const {
    return valid() && batch == 1 && kv_heads <= max_workers &&
        heads / kv_heads == 8 && dimension == 256 && queries <= 256;
  }
};

struct Quantization {
  float scale;
  int zero;

  bool valid() const {
    return std::isfinite(scale) && scale > 0 && zero >= 0 && zero <= 65535;
  }

  float dequantize(uint16_t code) const {
    return (int(code) - zero) * scale;
  }

  uint16_t quantize(float value) const {
    const float code = std::floor(value / scale + zero + 0.5f);
    return code <= 0 ? 0 : code >= 65535 ? 65535 : uint16_t(code);
  }
};

struct Configuration {
  Geometry geometry{};
  unsigned capacity = 0, valid = 0, first = 0, history = 0;

  bool configure(
      const int32_t* key,
      const int32_t* value,
      const int32_t* lengths,
      const float* grids) {
    if (!geometry.valid() || key[0] < 0 || value[0] < 0 || key[0] == value[0] ||
        uint32_t(key[1]) >> 31 || uint32_t(value[1]) >> 31 ||
        key[1] != value[1] || key[2] != value[2] || lengths[0] < 1 ||
        unsigned(lengths[0]) > geometry.queries || lengths[1] < 0 ||
        lengths[2] < 0) {
      return false;
    }
    const uint64_t row_bytes = uint64_t(geometry.batch) * geometry.kv_heads *
        geometry.dimension * sizeof(uint16_t);
    const unsigned bytes = uint32_t(key[1]) & logical_size_mask;
    if (!bytes || row_bytes > bytes || bytes % row_bytes) {
      return false;
    }
    capacity = bytes / row_bytes;
    valid = lengths[0];
    first = lengths[1];
    history = lengths[2];
    if (first > capacity || history > capacity - first) {
      return false;
    }
    for (unsigned i = 0; i < 13; ++i) {
      if (!std::isfinite(grids[i])) {
        return false;
      }
    }
    if (grids[2] <= 0) {
      return false;
    }
    for (unsigned i : {0u, 3u, 5u, 7u, 9u, 11u}) {
      if (grids[i] <= 0 || grids[i + 1] < 0 || grids[i + 1] > 65535 ||
          std::floor(grids[i + 1]) != grids[i + 1]) {
        return false;
      }
    }
    return true;
  }

  bool fast() const {
    return geometry.fast() &&
        uint64_t(capacity) + geometry.queries <= fast_kv_rows;
  }
};

} // namespace attention
