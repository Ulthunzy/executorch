// Copyright (c) Qualcomm Innovation Center, Inc.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "HTP/core/qhpi.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

// QHPI 仿射量化及 B/T/H 张量索引工具；不依赖 recurrence 或 state 布局。
namespace op_utils {

static inline bool hasShape(
    const QHPI_Shape& shape,
    uint32_t batch,
    uint32_t heads,
    uint32_t rows,
    uint32_t columns) {
  return shape.rank == 4 && shape.dims[0] == batch && shape.dims[1] == heads &&
      shape.dims[2] == rows && shape.dims[3] == columns;
}

static inline size_t numElements(const QHPI_Shape& shape) {
  size_t result = 1;
  for (unsigned i = 0; i < shape.rank; ++i)
    result *= shape.dims[i];
  return result;
}

// 将 U16 量化 code 转成浮点实值：real=(code-zero_offset)*stepsize。
static inline float dequantU16(uint16_t value, QHPI_Quant_Parameters qparams) {
  return (static_cast<float>(value) - static_cast<float>(qparams.zero_offset)) *
      qparams.stepsize;
}

// 将浮点实值映射为 round(zero_offset+real/stepsize)，饱和到 [0,65535] 后写入
// U16。
static inline uint16_t quantU16(float value, QHPI_Quant_Parameters qparams) {
  const float scaled = std::round(
      static_cast<float>(qparams.zero_offset) + value / qparams.stepsize);
  const float clamped = std::min(std::max(scaled, 0.0f), 65535.0f);
  return static_cast<uint16_t>(clamped);
}

// 检查零点是否已提供，以及 stepsize 是否为有限正数。
// 只排除未初始化标记 INT32_MAX，不检查零点是否在 U16 范围内。
static inline bool isValidQuant(QHPI_Quant_Parameters qparams) {
  return qparams.zero_offset != INT32_MAX && std::isfinite(qparams.stepsize) &&
      qparams.stepsize > 0.0f;
}

// 返回中心化整数 code-zero_offset；没有乘 stepsize，尚不是反量化实值。
static inline int32_t centeredU16(
    uint16_t value,
    QHPI_Quant_Parameters qparams) {
  return static_cast<int32_t>(value) - qparams.zero_offset;
}

// 检查零点是否位于 U16 code 范围 [0,65535]。
// 这不保证未缩位的 code-z 可以放进 int16；调用方另行检查数值范围。
static inline bool hasU16ZeroOffset(QHPI_Quant_Parameters qparams) {
  return qparams.zero_offset >= 0 && qparams.zero_offset <= 65535;
}

// 计算 (batch,token,head) 在受支持三维/四维张量布局中的线性下标。
// 依次匹配 [B,T,H]、[B,H,T] 或指定位置带一个单例维度的四维布局。
// 采用首个匹配分支；调用前需用 hasTimeHeadLayout 检查兼容性。
static inline size_t timeHeadIndex(
    const QHPI_Shape& shape,
    size_t batch,
    size_t token,
    size_t head) {
  if (shape.rank == 3) {
    if (shape.dims[1] > token && shape.dims[2] > head) {
      return (batch * shape.dims[1] + token) * shape.dims[2] + head;
    }
    if (shape.dims[1] > head && shape.dims[2] > token) {
      return (batch * shape.dims[1] + head) * shape.dims[2] + token;
    }
  }
  if (shape.rank == 4) {
    if (shape.dims[1] > token && shape.dims[2] == 1 && shape.dims[3] > head) {
      return ((batch * shape.dims[1] + token) * shape.dims[2]) * shape.dims[3] +
          head;
    }
    if (shape.dims[1] == 1 && shape.dims[2] > token && shape.dims[3] > head) {
      return ((batch * shape.dims[1]) * shape.dims[2] + token) * shape.dims[3] +
          head;
    }
    if (shape.dims[1] > token && shape.dims[2] > head && shape.dims[3] == 1) {
      return ((batch * shape.dims[1] + token) * shape.dims[2] + head) *
          shape.dims[3];
    }
    if (shape.dims[1] > head && shape.dims[2] > token && shape.dims[3] == 1) {
      return ((batch * shape.dims[1] + head) * shape.dims[2] + token) *
          shape.dims[3];
    }
  }
  return 0;
}

// 检查张量是否符合 timeHeadIndex 支持的 B/T/H 布局，允许 T/H 维度大于使用范围。
// 仅支持函数列出的排列及单例维度，不进行任意维度重排。
static inline bool hasTimeHeadLayout(
    const QHPI_Shape& shape,
    size_t batch,
    size_t seq_len,
    size_t heads) {
  if (shape.dims[0] != batch) {
    return false;
  }
  if (shape.rank == 3) {
    return (shape.dims[1] >= seq_len && shape.dims[2] >= heads) ||
        (shape.dims[1] >= heads && shape.dims[2] >= seq_len);
  }
  if (shape.rank == 4) {
    return (shape.dims[1] >= seq_len && shape.dims[2] == 1 &&
            shape.dims[3] >= heads) ||
        (shape.dims[1] == 1 && shape.dims[2] >= seq_len &&
         shape.dims[3] >= heads) ||
        (shape.dims[1] >= seq_len && shape.dims[2] >= heads &&
         shape.dims[3] == 1) ||
        (shape.dims[1] >= heads && shape.dims[2] >= seq_len &&
         shape.dims[3] == 1);
  }
  return false;
}

} // namespace op_utils
