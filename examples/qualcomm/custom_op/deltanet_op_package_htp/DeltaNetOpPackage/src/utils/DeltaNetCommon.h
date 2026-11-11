// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "HTP/core/qhpi.h"

#include "HvxUtils.h"
#include "NumericUtils.h"
#include "QhpiUtils.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>

#ifndef DELTANET_PROFILE_PHASES
#define DELTANET_PROFILE_PHASES 0
#endif
#ifndef DELTANET_PROFILE_OUTPUT
#define DELTANET_PROFILE_OUTPUT 0
#endif
#ifndef DELTANET_U16_COPY_WITH_REDUCTION
#define DELTANET_U16_COPY_WITH_REDUCTION 0
#endif
#ifndef DELTANET_U16_PREPACK_COEFFICIENTS
#define DELTANET_U16_PREPACK_COEFFICIENTS 0
#endif
#ifndef DELTANET_U16_SPLIT_VECTORS
#define DELTANET_U16_SPLIT_VECTORS 1
#endif
#if (DELTANET_PROFILE_PHASES || DELTANET_PROFILE_OUTPUT) && defined(__hexagon__)
#include "HAP_farf.h"
#include "HAP_perf.h"
#endif

struct DeltaNetPhaseProfile {
#if (DELTANET_PROFILE_PHASES || DELTANET_PROFILE_OUTPUT) && defined(__hexagon__)
  uint64_t ticks[5]{};
  uint64_t started = 0;
  size_t tasks = 0;
  size_t vector_reductions = 0;
  size_t integer_updates = 0;

  void start() {
    ++tasks;
    asm volatile("" ::: "memory");
    started = HAP_perf_get_qtimer_count();
    asm volatile("" ::: "memory");
  }

  void record(unsigned phase) {
    asm volatile("" ::: "memory");
    const uint64_t now = HAP_perf_get_qtimer_count();
    asm volatile("" ::: "memory");
    ticks[phase] += now - started;
    started = now;
  }

  void paths(bool vector_reduction, bool integer_update) {
    vector_reductions += vector_reduction;
    integer_updates += integer_update;
  }

  void report(
      size_t slice,
      size_t slices,
      size_t tokens,
      size_t heads,
      size_t key_dim,
      size_t value_dim,
      bool vtcm) const {
#if DELTANET_PROFILE_PHASES
    FARF(
        ALWAYS,
        "GDN_PHASE slice=%u slices=%u tokens=%u heads=%u k=%u v=%u vtcm=%u tasks=%u qtimer_hz=19200000 setup_ticks=%llu reduce_ticks=%llu delta_ticks=%llu update_ticks=%llu copy_ticks=%llu vector_reductions=%u integer_updates=%u",
        unsigned(slice),
        unsigned(slices),
        unsigned(tokens),
        unsigned(heads),
        unsigned(key_dim),
        unsigned(value_dim),
        unsigned(vtcm),
        unsigned(tasks),
        (unsigned long long)ticks[0],
        (unsigned long long)ticks[1],
        (unsigned long long)ticks[2],
        (unsigned long long)ticks[3],
        (unsigned long long)ticks[4],
        unsigned(vector_reductions),
        unsigned(integer_updates));
#endif
  }
  void save(uint16_t* output) const {
    if (output)
      for (unsigned i = 0; i < 5; ++i)
        output[i] = std::min<uint64_t>(65535, ticks[i]);
  }
#else
  void start() {}
  void record(unsigned) {}
  void paths(bool, bool) {}
  void report(size_t, size_t, size_t, size_t, size_t, size_t, bool) const {}
  void save(uint16_t* output) const {
    if (output)
      std::fill_n(output, 5, uint16_t(0));
  }
#endif
};

static constexpr size_t kMaxDeltaNetDim = 256;
static constexpr size_t kValueTileMinDim = 256;
static constexpr size_t kQUint16HvxValuesPerVec = 128 / sizeof(uint16_t);
// 归约时 state、Q、K 各右移 4 位，乘积的尺度在后面乘 2^8 还原。
// 这是归约计算的精度取舍；state 的存储格式仍然是 QUInt16。
static constexpr int kQUint16HvxAccumInputShift = 4;
// Keep all calibrated bits in the integer reduction. Dropping
// the low four bits degrades low-amplitude readouts after RMSNorm.
#include "DeltaNetConfig.h"
static constexpr bool kQUint16KaratsubaReduction =
    DELTANET_U16_KARATSUBA_REDUCTION != 0;
#ifndef DELTANET_U16_KARATSUBA_VECTORS
#define DELTANET_U16_KARATSUBA_VECTORS 2
#endif
static constexpr bool kQUint16SplitReduction =
    DELTANET_U16_SPLIT_REDUCTION != 0 || kQUint16KaratsubaReduction;
static constexpr bool kQUint16FullPrecisionReduction =
    DELTANET_U16_FULL_PRECISION_REDUCTION != 0 || kQUint16SplitReduction;
using QUint16StateAccum =
    std::conditional_t<kQUint16FullPrecisionReduction, float, int32_t>;
// 更新不缩位 state/key；定点系数由 20 位小数降低 7 位，HVX 最终使用 13 位小数。
static constexpr int kQUint16HvxStateUpdateInputShift = 0;
static constexpr int kQUint16HvxStateUpdateCoeffShift = 7;
static constexpr int kQUint16StateUpdateShift = 20;
static constexpr bool kEnableQUint16HvxStateUpdate =
    DELTANET_U16_INTEGER_STATE_UPDATE != 0;
static constexpr bool kEnableQUint16HvxStateUpdateF32 = true;

static inline void deltanetQhpiCopyStateU16(
    uint16_t* destination,
    const uint16_t* source,
    size_t elements) {
#if defined(__hexagon__) && (QCOM_HTP_V81 || QCOM_HTP_V85)
  // Keep VTCM state copies vectorized on V85 as well as V81; the scalar
  // memcpy fallback adds substantial overhead to single-token recurrence.
  const uintptr_t addresses = reinterpret_cast<uintptr_t>(destination) |
      reinterpret_cast<uintptr_t>(source);
  if (addresses % 128 == 0 && elements % kQUint16HvxValuesPerVec == 0) {
    for (size_t offset = 0; offset < elements;
         offset += kQUint16HvxValuesPerVec) {
      *reinterpret_cast<HVX_Vector*>(destination + offset) =
          *reinterpret_cast<const HVX_Vector*>(source + offset);
    }
    return;
  }
#endif
  std::memcpy(destination, source, elements * sizeof(uint16_t));
}

// 决定一个 head 的 V 维是否再拆给多个 worker。
// 当前仅在 B*H=1、V>=256 且 worker 有余量时分块；常见 H=16,V=128
// 只按 head 并行，分块数受最小 SIMD tile 大小和 worker 数限制。
static inline size_t deltanetQhpiValueTileCount(
    size_t rows,
    size_t slices,
    size_t value_dim,
    size_t min_tile_elements) {
  // SM8850 实验仅在单 head、最大 V 的情况下观察到 V 分块收益。
  if (rows != 1 || rows >= slices || value_dim < kValueTileMinDim ||
      value_dim <= min_tile_elements) {
    return 1;
  }
  const size_t wanted_tiles = op_utils::divUp(slices, rows);
  const size_t max_vector_tiles =
      std::max<size_t>(value_dim / min_tile_elements, 1);
  return std::max<size_t>(std::min(wanted_tiles, max_vector_tiles), 1);
}
