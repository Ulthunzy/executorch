// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <cstdint>

namespace attention {

struct alignas(2048) GroupedScratchU16 {
  uint8_t query[2 * 32 * 256 * 2];
  uint8_t probability[2 * 32 * 1024 * 2];
  uint16_t product[2 * 32 * 256];
  int32_t accumulator[2 * 32 * 256];
  uint8_t kv[131072];
  uint32_t bias[128];
  int32_t query_sum[2 * 32];
  uint32_t probability_sum[2 * 32];
  uint32_t maximum[2 * 32];
  uint32_t normalizer[2 * 32];
  uint32_t rescale[2 * 32];
};
static_assert(alignof(GroupedScratchU16) == 2048);
static_assert(sizeof(GroupedScratchU16) == 386 * 1024);
static_assert(sizeof(GroupedScratchU16) % 2048 == 0);
static_assert(offsetof(GroupedScratchU16, query) % 2048 == 0);
static_assert(offsetof(GroupedScratchU16, probability) % 2048 == 0);
static_assert(offsetof(GroupedScratchU16, product) % 2048 == 0);
static_assert(offsetof(GroupedScratchU16, kv) % 2048 == 0);
} // namespace attention
