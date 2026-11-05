// Copyright (c) Qualcomm Innovation Center, Inc.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "HTP/core/float16.h"
#include "HTP/core/intrinsics.h"

#include <cstdint>

#ifndef __HVX_ARCH__
#define __HVX_ARCH__ 79
#endif

// 通用 HVX 算术、广播和格式转换；仅用于 Hexagon DSP 或 libnative 仿真。
namespace op_utils {

// 32 个 FP32 lane 逐元素乘法；v79 起使用原生 FP32 指令。
// 旧架构经 QF32 中间格式再转回 FP32，QF32 与 U16 仿射量化无关。
static inline HVX_Vector hvxMulF32(HVX_Vector a, HVX_Vector b) {
#if __HVX_ARCH__ < 79
  return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b));
#else
  return Q6_Vsf_vmpy_VsfVsf(a, b);
#endif
}

// 32 个 FP32 lane 逐元素加法；旧架构通过 QF32 中间格式适配。
static inline HVX_Vector hvxAddF32(HVX_Vector a, HVX_Vector b) {
#if __HVX_ARCH__ < 79
  return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b));
#else
  return Q6_Vsf_vadd_VsfVsf(a, b);
#endif
}

// 按各 lane 的正负分别加 +0.5/-0.5，再将 FP32 转成 int32。
// 不进行数值范围检查或饱和处理，调用方须保证可表示。
static inline HVX_Vector hvxRoundF32ToI32(HVX_Vector value) {
  const HVX_Vector v_half = Q6_V_vsplat_R(0x3F000000);
  const HVX_Vector v_neg_half = Q6_V_vsplat_R(0xBF000000);
  const HVX_VectorPred q_neg =
      Q6_Q_vcmp_gt_VsfVsf(Q6_V_vzero(), value);
  return Q6_Vw_equals_Vsf(
      hvxAddF32(value, Q6_V_vmux_QVV(q_neg, v_neg_half, v_half)));
}

// 将一个 float 广播到 HVX 的 32 个 FP32 lane。
static inline HVX_Vector hvxSplatF32(float value) {
  return q6op_V_vsplat_float32(value);
}

// 把 lo/hi 两个 FP32 向量转成 QF32 向量对，保留低/高半组的顺序。
// 可供后续 FP16 打包使用：两个向量的 64 个值压成一个 FP16 向量。
static inline HVX_VectorPair hvxF32PairToQF32(
    HVX_Vector lo,
    HVX_Vector hi) {
  return Q6_W_vcombine_VV(Q6_Vqf32_equals_Vsf(hi), Q6_Vqf32_equals_Vsf(lo));
}

// 将 64 个 U16 展开为两个连续排列的 int32 向量，并减去缩位后的零点。
// 计算 (raw>>input_shift)-(zero_offset>>input_shift)，一般不等于 (raw-z)>>input_shift。
// input_shift=0 时保留完整中心化值；调用方负责零点和移位范围。
static inline HVX_VectorPair hvxCenteredShiftedU16ToI32FromVector(
    HVX_Vector input,
    int32_t zero_offset,
    int input_shift) {
  const HVX_Vector v_shifted_raw =
      input_shift == 0 ? input : Q6_Vuh_vlsr_VuhR(input, input_shift);
  HVX_VectorPair v_words = Q6_Wuw_vzxt_Vuh(v_shifted_raw);
  v_words = Q6_W_vshuff_VVR(Q6_V_hi_W(v_words), Q6_V_lo_W(v_words), -4);
  const HVX_Vector v_zero = Q6_V_vsplat_R(zero_offset >> input_shift);
  const HVX_Vector v_lo = Q6_Vw_vsub_VwVw(Q6_V_lo_W(v_words), v_zero);
  const HVX_Vector v_hi = Q6_Vw_vsub_VwVw(Q6_V_hi_W(v_words), v_zero);
  return Q6_W_vcombine_VV(v_hi, v_lo);
}

// 从可不对齐的地址加载 64 个 U16，复用 FromVector 的缩位、中心化与重排。
static inline HVX_VectorPair hvxCenteredShiftedU16ToI32(
    const uint16_t* input,
    int32_t zero_offset,
    int input_shift) {
  return hvxCenteredShiftedU16ToI32FromVector(
      vmemu(input), zero_offset, input_shift);
}

// 将 Float16 的原始位模式广播到 HVX 的 64 个 FP16 lane。
static inline HVX_Vector hvxSplatF16(Float16 value) {
  return Q6_Vh_vsplat_R(value.raw());
}

// 64 个 FP16 lane 逐元素乘法，经 QF16 运算后立即转回 FP16，保留半精度舍入。
static inline HVX_Vector hvxMulF16(
    HVX_Vector a,
    HVX_Vector b) {
  return Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(a, b));
}

// 64 个 FP16 lane 逐元素加法，结果立即转回 FP16。
static inline HVX_Vector hvxAddF16(
    HVX_Vector a,
    HVX_Vector b) {
  return Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(a, b));
}

// 64 个 FP16 lane 逐元素减法，结果立即转回 FP16。
static inline HVX_Vector hvxSubF16(
    HVX_Vector a,
    HVX_Vector b) {
  return Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_VhfVhf(a, b));
}

// 将 float 显式舍入为 Float16。
static inline Float16 roundF16(float value) {
  return Float16(value);
}

// 标量半精度乘法：输入转为 float 相乘，再将结果舍入为 Float16。
static inline Float16 mulF16(Float16 a, Float16 b) {
  return roundF16(static_cast<float>(a) * static_cast<float>(b));
}

// 标量半精度加法：输入转为 float 相加，再将结果舍入为 Float16。
static inline Float16 addF16(Float16 a, Float16 b) {
  return roundF16(static_cast<float>(a) + static_cast<float>(b));
}

// 标量半精度减法：输入转为 float 相减，再将结果舍入为 Float16。
static inline Float16 subF16(Float16 a, Float16 b) {
  return roundF16(static_cast<float>(a) - static_cast<float>(b));
}

} // namespace op_utils
