// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <cstdint>

namespace attention {

struct alignas(2048) GroupedScratch {
  uint8_t query[2 * 32 * 256 * 2];
  uint8_t probability[2 * 32 * 512 * 2];
  uint16_t product[2 * 32 * 256];
  uint16_t score[2 * 32 * 512];
  float accumulator[2 * 32 * 256];
  uint8_t kv[512 * 128 * 2];
  uint32_t bias[64];
  uint32_t maximum[2 * 32];
  float normalizer[2 * 32];
  float rescale[2 * 32];
};

static_assert(alignof(GroupedScratch) == 2048);
static_assert(sizeof(GroupedScratch) == 386 * 1024);
static_assert(sizeof(GroupedScratch) % 2048 == 0);
static_assert(offsetof(GroupedScratch, query) % 2048 == 0);
static_assert(offsetof(GroupedScratch, probability) % 2048 == 0);
static_assert(offsetof(GroupedScratch, product) % 2048 == 0);
static_assert(offsetof(GroupedScratch, kv) % 2048 == 0);
static_assert(offsetof(GroupedScratch, bias) % 128 == 0);

} // namespace attention
