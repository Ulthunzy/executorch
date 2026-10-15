// Copyright (c) Qualcomm Innovation Center, Inc.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

// 通用整数范围、定点舍入和 U16 code 转换；不依赖 QHPI 或具体算子。
namespace op_utils {

// 整数向上取整除法（d > 0）。
static inline size_t divUp(size_t n, size_t d) {
  return (n + d - 1) / d;
}

// 把已经计算好的整数 code 饱和到 U16 范围；不做尺度换算或舍入。
static inline uint16_t clampCodeToU16(int64_t code) {
  if (code <= 0) {
    return 0;
  }
  if (code >= 65535) {
    return 65535;
  }
  return static_cast<uint16_t>(code);
}

// 输入已是 double 精度的 code 域数值（通常为 z+real/s），不再除 stepsize。
// 非有限值或非正值返回 0；其他值舍入到最近整数并饱和到 65535。
static inline uint16_t quantU16FromScaledDouble(double scaled) {
  if (!std::isfinite(scaled) || scaled <= 0.0) {
    return 0;
  }
  if (scaled >= 65535.0) {
    return 65535;
  }
  return static_cast<uint16_t>(scaled + 0.5);
}

// quantU16FromScaledDouble 的 float 输入版本；输入已在 code 域，非有限值返回 0。
static inline uint16_t quantU16FromScaledFloat(float scaled) {
  if (!std::isfinite(scaled) || scaled <= 0.0f) {
    return 0;
  }
  if (scaled >= 65535.0f) {
    return 65535;
  }
  return static_cast<uint16_t>(scaled + 0.5f);
}

// 浮点数 -> 带 shift 个小数位的定点整数：round(value*2^shift)。
// 半整数向远离零舍入，例如 shift=13 时 0.75 -> 6144。
// 有限值超出 int64 范围时饱和，非有限值返回 0。
static inline int64_t roundFixed(double value, int shift) {
  const double scaled = std::ldexp(value, shift);
  if (!std::isfinite(scaled)) {
    return 0;
  }
  // 先检查转换边界，避免超范围的 double 直接转 int64；double 中的上界近似为 2^63。
  constexpr double kI64Max =
      static_cast<double>(std::numeric_limits<int64_t>::max());
  constexpr double kI64Min =
      static_cast<double>(std::numeric_limits<int64_t>::min());
  if (scaled >= kI64Max) {
    return std::numeric_limits<int64_t>::max();
  }
  if (scaled <= kI64Min) {
    return std::numeric_limits<int64_t>::min();
  }
  return static_cast<int64_t>(scaled >= 0.0 ? scaled + 0.5 : scaled - 0.5);
}

// 将定点整数舍入后右移 shift 位（shift > 0），降低小数位数。
// 正负半整数都向远离零舍入；调用方须保证取绝对值及加舍入量不溢出。
static inline int64_t roundShiftI64(int64_t value, int shift) {
  const int64_t rounding = INT64_C(1) << (shift - 1);
  if (value >= 0) {
    return (value + rounding) >> shift;
  }
  return -(((-value) + rounding) >> shift);
}

// 检查有符号整数能否用 int16 表示；不执行类型转换。
static inline bool fitsI16(int32_t value) {
  return value >= std::numeric_limits<int16_t>::min() &&
      value <= std::numeric_limits<int16_t>::max();
}

// 检查有符号整数能否用 int16 表示；不执行类型转换。
static inline bool fitsI16(int64_t value) {
  return value >= std::numeric_limits<int16_t>::min() &&
      value <= std::numeric_limits<int16_t>::max();
}

// 检查 int64 数值能否用 int32 表示；不执行类型转换。
static inline bool fitsI32(int64_t value) {
  return value >= std::numeric_limits<int32_t>::min() &&
      value <= std::numeric_limits<int32_t>::max();
}

} // namespace op_utils
