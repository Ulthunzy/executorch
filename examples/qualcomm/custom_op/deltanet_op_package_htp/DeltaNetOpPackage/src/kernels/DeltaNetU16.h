// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "../utils/DeltaNetCommon.h"

static inline HVX_VectorPair deltanetQhpiHvxMulCenteredRawU16I32(
    HVX_Vector input,
    int32_t zero_offset,
    int32_t multiplier) {
  const HVX_Vector v_zero = Q6_Vh_vsplat_R(
      static_cast<int>((zero_offset >> kQUint16HvxAccumInputShift) & 0xFFFF));
  const HVX_Vector v_shifted_raw =
      Q6_Vuh_vlsr_VuhR(input, kQUint16HvxAccumInputShift);
  const HVX_Vector v_shifted = Q6_Vh_vsub_VhVh(v_shifted_raw, v_zero);
  const HVX_VectorPair v_product =
      Q6_Ww_vmpy_VhRh(v_shifted, Q6_R_combine_RlRl(multiplier, multiplier));
  return Q6_W_vshuff_VVR(Q6_V_hi_W(v_product), Q6_V_lo_W(v_product), -4);
}

// 从可不对齐的地址加载一个 128 字节 state 向量，复用 RawU16I32 的缩位乘法。
static inline HVX_VectorPair deltanetQhpiHvxMulCenteredU16I32(
    const uint16_t* input,
    int32_t zero_offset,
    int32_t multiplier) {
  return deltanetQhpiHvxMulCenteredRawU16I32(
      vmemu(input), zero_offset, multiplier);
}

// 计算两个 U16 向量的中心化点积，返回尚未乘 stepsize 的 int64 累加值。
// 完整 64 元素块各右移 4 位后用 HVX 相乘，再将乘积写入临时数组做标量归约，
// 最后乘 2^8 恢复尺度（不恢复丢失的低位）；不足一个块的尾部用未缩位标量计算。
static inline int64_t deltanetQhpiHvxDotCenteredU16(
    const uint16_t* lhs,
    QHPI_Quant_Parameters lhs_q,
    const uint16_t* rhs,
    QHPI_Quant_Parameters rhs_q,
    size_t dim) {
  constexpr size_t kHalfValuesPerVector = 128 / sizeof(uint16_t);
  constexpr size_t kWordValuesPerVector = 128 / sizeof(int32_t);
  constexpr int kInputShift = kQUint16HvxAccumInputShift;
  alignas(128) int32_t products[kHalfValuesPerVector];
  const HVX_Vector lhs_zero =
      Q6_Vh_vsplat_R((lhs_q.zero_offset >> kInputShift) & 0xFFFF);
  const HVX_Vector rhs_zero =
      Q6_Vh_vsplat_R((rhs_q.zero_offset >> kInputShift) & 0xFFFF);
  const size_t vector_dim = dim - dim % kHalfValuesPerVector;
  int64_t shifted_acc = 0;
  for (size_t offset = 0; offset < vector_dim; offset += kHalfValuesPerVector) {
    const HVX_Vector lhs_shifted =
        Q6_Vuh_vlsr_VuhR(vmemu(lhs + offset), kInputShift);
    const HVX_Vector rhs_shifted =
        Q6_Vuh_vlsr_VuhR(vmemu(rhs + offset), kInputShift);
    const HVX_Vector lhs_centered = Q6_Vh_vsub_VhVh(lhs_shifted, lhs_zero);
    const HVX_Vector rhs_centered = Q6_Vh_vsub_VhVh(rhs_shifted, rhs_zero);
    const HVX_VectorPair product = Q6_Ww_vmpy_VhVh(lhs_centered, rhs_centered);
    vmemu(products) = Q6_V_lo_W(product);
    vmemu(products + kWordValuesPerVector) = Q6_V_hi_W(product);
    for (size_t i = 0; i < kHalfValuesPerVector; ++i) {
      shifted_acc += products[i];
    }
  }

  int64_t scalar_acc = 0;
  for (size_t i = vector_dim; i < dim; ++i) {
    scalar_acc += static_cast<int64_t>(op_utils::centeredU16(lhs[i], lhs_q)) *
        op_utils::centeredU16(rhs[i], rhs_q);
  }
  return (shifted_acc << (2 * kInputShift)) + scalar_acc;
}

// Stream 路径一次读取一列 state，同时计算 K^T*S 和 Q^T*S 两个点积。
// SIMD lane 沿 K 展开，向量乘积仍写入临时数组，由标量 int64 循环做水平归约。
// 完整块缩位 4 位并恢复乘积尺度，尾部未缩位；返回值还需调用方乘量化 stepsize。
static inline void deltanetQhpiHvxDotCenteredU16Pair(
    const uint16_t* state,
    QHPI_Quant_Parameters state_q,
    const uint16_t* key,
    QHPI_Quant_Parameters key_q,
    const uint16_t* query,
    QHPI_Quant_Parameters query_q,
    size_t dim,
    int64_t* kv_acc,
    int64_t* output_acc) {
  constexpr size_t kHalfValuesPerVector = 128 / sizeof(uint16_t);
  constexpr size_t kWordValuesPerVector = 128 / sizeof(int32_t);
  constexpr int kInputShift = kQUint16HvxAccumInputShift;
  alignas(128) int32_t kv_products[kHalfValuesPerVector];
  alignas(128) int32_t output_products[kHalfValuesPerVector];
  const HVX_Vector state_zero =
      Q6_Vh_vsplat_R((state_q.zero_offset >> kInputShift) & 0xFFFF);
  const HVX_Vector key_zero =
      Q6_Vh_vsplat_R((key_q.zero_offset >> kInputShift) & 0xFFFF);
  const HVX_Vector query_zero =
      Q6_Vh_vsplat_R((query_q.zero_offset >> kInputShift) & 0xFFFF);
  const size_t vector_dim = dim - dim % kHalfValuesPerVector;
  int64_t kv_shifted_acc = 0;
  int64_t output_shifted_acc = 0;
  for (size_t offset = 0; offset < vector_dim; offset += kHalfValuesPerVector) {
    const HVX_Vector state_centered = Q6_Vh_vsub_VhVh(
        Q6_Vuh_vlsr_VuhR(vmemu(state + offset), kInputShift), state_zero);
    const HVX_Vector key_centered = Q6_Vh_vsub_VhVh(
        Q6_Vuh_vlsr_VuhR(vmemu(key + offset), kInputShift), key_zero);
    const HVX_Vector query_centered = Q6_Vh_vsub_VhVh(
        Q6_Vuh_vlsr_VuhR(vmemu(query + offset), kInputShift), query_zero);
    const HVX_VectorPair kv_product =
        Q6_Ww_vmpy_VhVh(state_centered, key_centered);
    const HVX_VectorPair output_product =
        Q6_Ww_vmpy_VhVh(state_centered, query_centered);
    vmemu(kv_products) = Q6_V_lo_W(kv_product);
    vmemu(kv_products + kWordValuesPerVector) = Q6_V_hi_W(kv_product);
    vmemu(output_products) = Q6_V_lo_W(output_product);
    vmemu(output_products + kWordValuesPerVector) = Q6_V_hi_W(output_product);
    for (size_t i = 0; i < kHalfValuesPerVector; ++i) {
      kv_shifted_acc += kv_products[i];
      output_shifted_acc += output_products[i];
    }
  }

  int64_t kv_scalar_acc = 0;
  int64_t output_scalar_acc = 0;
  for (size_t i = vector_dim; i < dim; ++i) {
    const int32_t state_center = op_utils::centeredU16(state[i], state_q);
    kv_scalar_acc += static_cast<int64_t>(state_center) *
        op_utils::centeredU16(key[i], key_q);
    output_scalar_acc += static_cast<int64_t>(state_center) *
        op_utils::centeredU16(query[i], query_q);
  }
  *kv_acc = (kv_shifted_acc << (2 * kInputShift)) + kv_scalar_acc;
  *output_acc = (output_shifted_acc << (2 * kInputShift)) + output_scalar_acc;
}

// 沿 K 更新 Stream 的一个连续 state 列，允许输入/输出指向同一列缓冲区。
// code=zOut+(rawS-zPrev)*prev_scale+(rawK-zKey)*key_scale，
// 其中 prev_scale=decay*sPrev/sOut，key_scale=delta*sKey/sOut。
// 64 元素块用 FP32 向量运算并量化回 U16，尾部用 double 计算。
static inline void deltanetQhpiHvxUpdateStateStreamU16F32(
    const uint16_t* prev_state,
    uint16_t* state,
    const uint16_t* key,
    QHPI_Quant_Parameters prev_q,
    QHPI_Quant_Parameters key_q,
    QHPI_Quant_Parameters state_q,
    float prev_scale,
    float key_scale,
    size_t dim) {
  constexpr size_t kHalfValuesPerVector = 128 / sizeof(uint16_t);
  const size_t vector_dim = dim - dim % kHalfValuesPerVector;
  const HVX_Vector v_state_zero =
      op_utils::hvxSplatF32(static_cast<float>(state_q.zero_offset) + 0.5f);
  const HVX_Vector v_prev_scale = op_utils::hvxSplatF32(prev_scale);
  const HVX_Vector v_key_scale = op_utils::hvxSplatF32(key_scale);
  for (size_t offset = 0; offset < vector_dim; offset += kHalfValuesPerVector) {
    const HVX_VectorPair v_prev = op_utils::hvxCenteredShiftedU16ToI32(
        prev_state + offset, prev_q.zero_offset, 0);
    const HVX_VectorPair v_key = op_utils::hvxCenteredShiftedU16ToI32(
        key + offset, key_q.zero_offset, 0);
    const HVX_Vector v_prev_lo = Q6_Vsf_equals_Vw(Q6_V_lo_W(v_prev));
    const HVX_Vector v_prev_hi = Q6_Vsf_equals_Vw(Q6_V_hi_W(v_prev));
    const HVX_Vector v_key_lo = Q6_Vsf_equals_Vw(Q6_V_lo_W(v_key));
    const HVX_Vector v_key_hi = Q6_Vsf_equals_Vw(Q6_V_hi_W(v_key));
    const HVX_Vector v_code_lo = op_utils::hvxAddF32(
        v_state_zero,
        op_utils::hvxAddF32(
            op_utils::hvxMulF32(v_prev_lo, v_prev_scale),
            op_utils::hvxMulF32(v_key_lo, v_key_scale)));
    const HVX_Vector v_code_hi = op_utils::hvxAddF32(
        v_state_zero,
        op_utils::hvxAddF32(
            op_utils::hvxMulF32(v_prev_hi, v_prev_scale),
            op_utils::hvxMulF32(v_key_hi, v_key_scale)));
    const HVX_Vector v_code_i_lo = Q6_Vw_equals_Vsf(v_code_lo);
    const HVX_Vector v_code_i_hi = Q6_Vw_equals_Vsf(v_code_hi);
    const HVX_Vector v_state_half =
        Q6_Vuh_vasr_VwVwR_sat(v_code_i_hi, v_code_i_lo, 0);
    vmemu(state + offset) = Q6_Vh_vdeal_Vh(v_state_half);
  }

  for (size_t i = vector_dim; i < dim; ++i) {
    const double prev =
        static_cast<double>(op_utils::centeredU16(prev_state[i], prev_q));
    const double code = static_cast<double>(state_q.zero_offset) +
        prev * prev_scale +
        static_cast<double>(op_utils::centeredU16(key[i], key_q)) * key_scale;
    state[i] = op_utils::quantU16FromScaledDouble(code);
  }
}

// 用 k=0 的 state 行初始化两个 int32 归约数组，避免先清零再累加。
// state 和传入的 Q/K 系数均按归约约定缩位；尽管名字含 Hvx，当前 vec_count=0，
// 初始化实际走标量循环，后续 k>=1 的 Accum 函数才使用 HVX。
static inline void deltanetQhpiHvxAssignAccumU16I32(
    const uint16_t* prev_state,
    int32_t* kv_acc,
    int32_t* output_pre_acc,
    int32_t prev_zero_offset,
    int32_t key_value,
    int32_t query_value,
    size_t value_dim) {
  const size_t halfs_per_vec = kQUint16HvxValuesPerVec;
  const size_t words_per_vec = 128 / sizeof(int32_t);
  // SM8850 实验中一次性的 k=0 标量初始化更快；k>=1 仍走 HVX。
  const size_t vec_count = 0;

  for (size_t i = 0; i < vec_count; ++i) {
    const size_t offset = i * halfs_per_vec;
    const HVX_VectorPair v_kv = deltanetQhpiHvxMulCenteredU16I32(
        prev_state + offset, prev_zero_offset, key_value);
    const HVX_VectorPair v_out = deltanetQhpiHvxMulCenteredU16I32(
        prev_state + offset, prev_zero_offset, query_value);
    vmemu(kv_acc + offset) = Q6_V_lo_W(v_kv);
    vmemu(kv_acc + offset + words_per_vec) = Q6_V_hi_W(v_kv);
    vmemu(output_pre_acc + offset) = Q6_V_lo_W(v_out);
    vmemu(output_pre_acc + offset + words_per_vec) = Q6_V_hi_W(v_out);
  }

  for (size_t v = vec_count * halfs_per_vec; v < value_dim; ++v) {
    const int32_t prev =
        (static_cast<int32_t>(prev_state[v]) >> kQUint16HvxAccumInputShift) -
        (prev_zero_offset >> kQUint16HvxAccumInputShift);
    kv_acc[v] = prev * key_value;
    output_pre_acc[v] = prev * query_value;
  }
}

// 把当前 K 行的 state 同时累加到 K^T*S、Q^T*S 两个 int32 scratch 数组。
// HVX 每次处理 64 个 V，尾部标量处理；每行都读写 scratch，
// 寄存器 tile 归约 ReduceStateU16I32 则将累加器保留到整个 K 循环结束。
static inline void deltanetQhpiHvxAccumU16I32(
    const uint16_t* prev_state,
    int32_t* kv_acc,
    int32_t* output_pre_acc,
    int32_t prev_zero_offset,
    int32_t key_value,
    int32_t query_value,
    size_t value_dim) {
  const size_t halfs_per_vec = kQUint16HvxValuesPerVec;
  const size_t words_per_vec = 128 / sizeof(int32_t);
  const size_t vec_count = value_dim / halfs_per_vec;

  for (size_t i = 0; i < vec_count; ++i) {
    const size_t offset = i * halfs_per_vec;
    const HVX_Vector v_prev_raw = vmemu(prev_state + offset);
    const HVX_VectorPair v_kv = deltanetQhpiHvxMulCenteredRawU16I32(
        v_prev_raw, prev_zero_offset, key_value);
    const HVX_VectorPair v_out = deltanetQhpiHvxMulCenteredRawU16I32(
        v_prev_raw, prev_zero_offset, query_value);
    vmemu(kv_acc + offset) =
        Q6_Vw_vadd_VwVw(vmemu(kv_acc + offset), Q6_V_lo_W(v_kv));
    vmemu(kv_acc + offset + words_per_vec) = Q6_Vw_vadd_VwVw(
        vmemu(kv_acc + offset + words_per_vec), Q6_V_hi_W(v_kv));
    vmemu(output_pre_acc + offset) =
        Q6_Vw_vadd_VwVw(vmemu(output_pre_acc + offset), Q6_V_lo_W(v_out));
    vmemu(output_pre_acc + offset + words_per_vec) = Q6_Vw_vadd_VwVw(
        vmemu(output_pre_acc + offset + words_per_vec), Q6_V_hi_W(v_out));
  }

  for (size_t v = vec_count * halfs_per_vec; v < value_dim; ++v) {
    const int32_t prev =
        (static_cast<int32_t>(prev_state[v]) >> kQUint16HvxAccumInputShift) -
        (prev_zero_offset >> kQUint16HvxAccumInputShift);
    kv_acc[v] += prev * key_value;
    output_pre_acc[v] += prev * query_value;
  }
}

// 每个 token/head 准备一次：生成未缩位的中心化 Q/K、检查 key 更新范围并计算
// Q·K。 dot=sum(rawQ*rawK)-zQ*sum(rawK)-zK*sum(rawQ)+K*zQ*zK。 U16
// 乘积拆成低/高 16 位分别做向量归约，再用 int64 重建，避免 dot 溢出；
// 调用方确保 key_dim 是 64 的倍数，本函数不处理尾部，返回值尚未乘 sQ*sK。
static inline int64_t deltanetQhpiHvxPrepareQkU16(
    const uint16_t* query,
    const uint16_t* key,
    int32_t* query_center,
    int32_t* key_center,
    int32_t query_zero,
    int32_t key_zero,
    size_t key_dim,
    bool& key_update_fits) {
  constexpr size_t kWordValuesPerVector = 128 / sizeof(int32_t);
  HVX_Vector prod_low = Q6_V_vzero();
  HVX_Vector prod_high = Q6_V_vzero();
  HVX_Vector query_sum = Q6_V_vzero();
  HVX_Vector key_sum = Q6_V_vzero();
  HVX_Vector key_min = Q6_Vh_vsplat_R(65535);
  HVX_Vector key_max = Q6_V_vzero();
  const HVX_Vector mask = Q6_V_vsplat_R(65535);
  for (size_t k = 0; k < key_dim; k += kQUint16HvxValuesPerVec) {
    const HVX_Vector q = vmemu(query + k);
    const HVX_Vector v_key = vmemu(key + k);
    const HVX_VectorPair product = Q6_Wuw_vmpy_VuhVuh(q, v_key);
    // 先拆分无符号乘积再归约，最后重建 64 位点积，保留完整整数乘积精度。
    prod_low = Q6_Vw_vadd_VwVw(
        prod_low,
        Q6_Vw_vadd_VwVw(
            Q6_V_vand_VV(Q6_V_lo_W(product), mask),
            Q6_V_vand_VV(Q6_V_hi_W(product), mask)));
    prod_high = Q6_Vw_vadd_VwVw(
        prod_high,
        Q6_Vw_vadd_VwVw(
            Q6_Vuw_vlsr_VuwR(Q6_V_lo_W(product), 16),
            Q6_Vuw_vlsr_VuwR(Q6_V_hi_W(product), 16)));
    const HVX_VectorPair q_raw = Q6_Wuw_vzxt_Vuh(q);
    const HVX_VectorPair k_raw = Q6_Wuw_vzxt_Vuh(v_key);
    query_sum = Q6_Vw_vadd_VwVw(
        query_sum, Q6_Vw_vadd_VwVw(Q6_V_lo_W(q_raw), Q6_V_hi_W(q_raw)));
    key_sum = Q6_Vw_vadd_VwVw(
        key_sum, Q6_Vw_vadd_VwVw(Q6_V_lo_W(k_raw), Q6_V_hi_W(k_raw)));
    key_min = Q6_Vuh_vmin_VuhVuh(key_min, v_key);
    key_max = Q6_Vuh_vmax_VuhVuh(key_max, v_key);
    const HVX_VectorPair q_center =
        op_utils::hvxCenteredShiftedU16ToI32FromVector(q, query_zero, 0);
    const HVX_VectorPair k_center =
        op_utils::hvxCenteredShiftedU16ToI32FromVector(v_key, key_zero, 0);
    vmemu(query_center + k) = Q6_V_lo_W(q_center);
    vmemu(query_center + k + kWordValuesPerVector) = Q6_V_hi_W(q_center);
    vmemu(key_center + k) = Q6_V_lo_W(k_center);
    vmemu(key_center + k + kWordValuesPerVector) = Q6_V_hi_W(k_center);
  }
  for (int shift = 64; shift >= 4; shift >>= 1) {
    prod_low = Q6_Vw_vadd_VwVw(prod_low, Q6_V_vror_VR(prod_low, shift));
    prod_high = Q6_Vw_vadd_VwVw(prod_high, Q6_V_vror_VR(prod_high, shift));
    query_sum = Q6_Vw_vadd_VwVw(query_sum, Q6_V_vror_VR(query_sum, shift));
    key_sum = Q6_Vw_vadd_VwVw(key_sum, Q6_V_vror_VR(key_sum, shift));
    key_min = Q6_Vuh_vmin_VuhVuh(key_min, Q6_V_vror_VR(key_min, shift));
    key_max = Q6_Vuh_vmax_VuhVuh(key_max, Q6_V_vror_VR(key_max, shift));
  }
  key_min = Q6_Vuh_vmin_VuhVuh(key_min, Q6_V_vror_VR(key_min, 2));
  key_max = Q6_Vuh_vmax_VuhVuh(key_max, Q6_V_vror_VR(key_max, 2));
  const int32_t min_key = (Q6_R_vextract_VR(key_min, 0) & 65535) - key_zero;
  const int32_t max_key = (Q6_R_vextract_VR(key_max, 0) & 65535) - key_zero;
  key_update_fits =
      op_utils::fitsI16(min_key >> kQUint16HvxStateUpdateInputShift) &&
      op_utils::fitsI16(max_key >> kQUint16HvxStateUpdateInputShift);
  return static_cast<int64_t>(Q6_R_vextract_VR(prod_low, 0)) +
      (static_cast<int64_t>(Q6_R_vextract_VR(prod_high, 0)) << 16) -
      static_cast<int64_t>(query_zero) * Q6_R_vextract_VR(key_sum, 0) -
      static_cast<int64_t>(key_zero) * Q6_R_vextract_VR(query_sum, 0) +
      static_cast<int64_t>(key_dim) * query_zero * key_zero;
}

// 第一遍读取 state：同时归约 K^T*S 和 Q^T*S，每个 lane 对应一个 V 通道。
// Vectors=1/2 每轮处理 64/128 个 V；Aligned 控制 128 字节对齐访存。
// 默认使用 radix-16。radix-256 仅供显式实验：整数位完整，但 FP32 重组舍入不同。
// 累加器跨 K 留在寄存器，归约完成后仅写一次 FP32 scratch。
template <bool Aligned, size_t Vectors = 1, bool CopyState = false>
static inline void deltanetQhpiHvxReduceStateU16(
    const uint16_t* prev_state,
    QUint16StateAccum* kv_acc,
    QUint16StateAccum* output_pre_acc,
    int32_t prev_zero_offset,
    int32_t key_zero_offset,
    int32_t query_zero_offset,
    const int32_t* key_center,
    const int32_t* query_center,
    size_t key_dim,
    size_t state_stride,
    size_t tile_size,
    uint16_t* copied_state = nullptr) {
  constexpr size_t kHalfValuesPerVector = 128 / sizeof(uint16_t);
  constexpr size_t kWordValuesPerVector = 128 / sizeof(int32_t);
  if constexpr (
      DELTANET_U16_RADIX256_REDUCTION && kQUint16FullPrecisionReduction) {
    if (key_dim <= 128 && key_zero_offset >= 0 && key_zero_offset <= 65535 &&
        query_zero_offset >= 0 && query_zero_offset <= 65535) {
      // S=256*Sh+Sl, Q/K=(raw-32768)+(32768-zero). Accumulate the shared
      // state sum with int16 adds, then correct the Q/K bias outside K.
      // Integer terms fit int32, but the FP32 combine differs from radix-16.
      const int key_bias = 32768 - key_zero_offset;
      const int query_bias = 32768 - query_zero_offset;
      const auto bias_product = [](HVX_Vector sum, int bias) {
        const HVX_VectorPair product =
            Q6_Ww_vmpy_VhRh(sum, Q6_R_combine_RlRl(bias, bias));
        return bias == 32768 ? Q6_Ww_vsub_WwWw(Q6_W_vzero(), product) : product;
      };
      const HVX_Vector zero_hi = Q6_Vh_vsplat_R(prev_zero_offset >> 8);
      const HVX_Vector zero_lo = Q6_Vh_vsplat_R(prev_zero_offset & 255);
      const HVX_Vector mask = Q6_Vh_vsplat_R(255);
      const HVX_Vector scale = op_utils::hvxSplatF32(256.0f);
      for (size_t offset = 0; offset < tile_size;
           offset += kHalfValuesPerVector * Vectors) {
        HVX_VectorPair kv_hi[Vectors], kv_lo[Vectors];
        HVX_VectorPair out_hi[Vectors], out_lo[Vectors];
        HVX_Vector sum_hi[Vectors], sum_lo[Vectors];
        for (size_t v = 0; v < Vectors; ++v) {
          kv_hi[v] = kv_lo[v] = Q6_W_vzero();
          out_hi[v] = out_lo[v] = Q6_W_vzero();
          sum_hi[v] = sum_lo[v] = Q6_V_vzero();
        }
        for (size_t k = 0; k < key_dim; ++k) {
          const int key_signed = key_center[k] - key_bias;
          const int query_signed = query_center[k] - query_bias;
          const int key = Q6_R_combine_RlRl(key_signed, key_signed);
          const int query = Q6_R_combine_RlRl(query_signed, query_signed);
          for (size_t v = 0; v < Vectors; ++v) {
            const size_t at =
                k * state_stride + offset + v * kHalfValuesPerVector;
            const HVX_Vector raw = Aligned
                ? *reinterpret_cast<const HVX_Vector*>(prev_state + at)
                : vmemu(prev_state + at);
            const HVX_Vector hi =
                Q6_Vh_vsub_VhVh(Q6_Vuh_vlsr_VuhR(raw, 8), zero_hi);
            const HVX_Vector lo =
                Q6_Vh_vsub_VhVh(Q6_V_vand_VV(raw, mask), zero_lo);
            sum_hi[v] = Q6_Vh_vadd_VhVh(sum_hi[v], hi);
            sum_lo[v] = Q6_Vh_vadd_VhVh(sum_lo[v], lo);
            kv_hi[v] = Q6_Ww_vmpyacc_WwVhRh(kv_hi[v], hi, key);
            kv_lo[v] = Q6_Ww_vmpyacc_WwVhRh(kv_lo[v], lo, key);
            out_hi[v] = Q6_Ww_vmpyacc_WwVhRh(out_hi[v], hi, query);
            out_lo[v] = Q6_Ww_vmpyacc_WwVhRh(out_lo[v], lo, query);
          }
        }
        for (size_t v = 0; v < Vectors; ++v) {
          const size_t at = offset + v * kHalfValuesPerVector;
          kv_hi[v] =
              Q6_Ww_vadd_WwWw(kv_hi[v], bias_product(sum_hi[v], key_bias));
          kv_lo[v] =
              Q6_Ww_vadd_WwWw(kv_lo[v], bias_product(sum_lo[v], key_bias));
          out_hi[v] =
              Q6_Ww_vadd_WwWw(out_hi[v], bias_product(sum_hi[v], query_bias));
          out_lo[v] =
              Q6_Ww_vadd_WwWw(out_lo[v], bias_product(sum_lo[v], query_bias));
          const HVX_VectorPair kh =
              Q6_W_vshuff_VVR(Q6_V_hi_W(kv_hi[v]), Q6_V_lo_W(kv_hi[v]), -4);
          const HVX_VectorPair kl =
              Q6_W_vshuff_VVR(Q6_V_hi_W(kv_lo[v]), Q6_V_lo_W(kv_lo[v]), -4);
          const HVX_VectorPair qh =
              Q6_W_vshuff_VVR(Q6_V_hi_W(out_hi[v]), Q6_V_lo_W(out_hi[v]), -4);
          const HVX_VectorPair ql =
              Q6_W_vshuff_VVR(Q6_V_hi_W(out_lo[v]), Q6_V_lo_W(out_lo[v]), -4);
          vmemu(kv_acc + at) = op_utils::hvxAddF32(
              op_utils::hvxMulF32(Q6_Vsf_equals_Vw(Q6_V_lo_W(kh)), scale),
              Q6_Vsf_equals_Vw(Q6_V_lo_W(kl)));
          vmemu(kv_acc + at + kWordValuesPerVector) = op_utils::hvxAddF32(
              op_utils::hvxMulF32(Q6_Vsf_equals_Vw(Q6_V_hi_W(kh)), scale),
              Q6_Vsf_equals_Vw(Q6_V_hi_W(kl)));
          vmemu(output_pre_acc + at) = op_utils::hvxAddF32(
              op_utils::hvxMulF32(Q6_Vsf_equals_Vw(Q6_V_lo_W(qh)), scale),
              Q6_Vsf_equals_Vw(Q6_V_lo_W(ql)));
          vmemu(output_pre_acc + at + kWordValuesPerVector) =
              op_utils::hvxAddF32(
                  op_utils::hvxMulF32(Q6_Vsf_equals_Vw(Q6_V_hi_W(qh)), scale),
                  Q6_Vsf_equals_Vw(Q6_V_hi_W(ql)));
        }
      }
      return;
    }
  }
  if constexpr (kQUint16SplitReduction) {
    // S=16*S_hi+S_lo and K=16*K_hi+K_lo. Accumulate the three
    // integer terms separately, then combine 256*A+16*B+C in FP32.
    // For K<=128, each int32 accumulator fits even at U16 endpoints:
    // |A|<=128*4095*4096 < INT32_MAX. No state/coeff low bits are lost.
    if (key_dim <= 128) {
      constexpr size_t kSplitVectors = kQUint16KaratsubaReduction
          ? std::min<size_t>(Vectors, DELTANET_U16_KARATSUBA_VECTORS)
          : std::min<size_t>(Vectors, DELTANET_U16_SPLIT_VECTORS);
      constexpr int kSplitShift = 4;
      constexpr int kSplitMask = (1 << kSplitShift) - 1;
#if DELTANET_U16_PREPACK_COEFFICIENTS
      int packed[128][4];
      for (size_t k = 0; k < key_dim; ++k) {
        const int kh = key_center[k] >> kSplitShift;
        const int kl = key_center[k] & kSplitMask;
        const int qh = query_center[k] >> kSplitShift;
        const int ql = query_center[k] & kSplitMask;
        packed[k][0] = Q6_R_combine_RlRl(kh, kh);
        packed[k][1] = Q6_R_combine_RlRl(kl, kl);
        packed[k][2] = Q6_R_combine_RlRl(qh, qh);
        packed[k][3] = Q6_R_combine_RlRl(ql, ql);
      }
#endif
      const HVX_Vector v_zero_hi =
          Q6_Vh_vsplat_R(prev_zero_offset >> kSplitShift);
      const HVX_Vector v_zero_lo =
          Q6_Vh_vsplat_R(prev_zero_offset & kSplitMask);
      const HVX_Vector v_mask_lo = Q6_Vh_vsplat_R(kSplitMask);
      const HVX_Vector v_scale_hi =
          op_utils::hvxSplatF32(1 << (2 * kSplitShift));
      const HVX_Vector v_scale_cross = op_utils::hvxSplatF32(1 << kSplitShift);
      for (size_t offset = 0; offset < tile_size;
           offset += kHalfValuesPerVector * kSplitVectors) {
        HVX_VectorPair kv_hi[kSplitVectors], kv_cross[kSplitVectors],
            kv_lo[kSplitVectors];
        HVX_VectorPair out_hi[kSplitVectors], out_cross[kSplitVectors],
            out_lo[kSplitVectors];
        for (size_t v = 0; v < kSplitVectors; ++v) {
          kv_hi[v] = kv_cross[v] = kv_lo[v] = Q6_W_vzero();
          out_hi[v] = out_cross[v] = out_lo[v] = Q6_W_vzero();
        }
        for (size_t k = 0; k < key_dim; ++k) {
#if DELTANET_U16_PREPACK_COEFFICIENTS
          const int kh = packed[k][0], kl = packed[k][1];
          const int qh = packed[k][2], ql = packed[k][3];
#else
          const int kh = Q6_R_combine_RlRl(
              key_center[k] >> kSplitShift, key_center[k] >> kSplitShift);
          const int kl = Q6_R_combine_RlRl(
              key_center[k] & kSplitMask, key_center[k] & kSplitMask);
          const int qh = Q6_R_combine_RlRl(
              query_center[k] >> kSplitShift, query_center[k] >> kSplitShift);
          const int ql = Q6_R_combine_RlRl(
              query_center[k] & kSplitMask, query_center[k] & kSplitMask);
#endif
          for (size_t v = 0; v < kSplitVectors; ++v) {
            const size_t at = offset + v * kHalfValuesPerVector;
            const HVX_Vector raw = Aligned
                ? *reinterpret_cast<const HVX_Vector*>(
                      prev_state + k * state_stride + at)
                : vmemu(prev_state + k * state_stride + at);
            if constexpr (CopyState)
              vmemu(copied_state + k * state_stride + at) = raw;
            const HVX_Vector sh =
                Q6_Vh_vsub_VhVh(Q6_Vuh_vlsr_VuhR(raw, kSplitShift), v_zero_hi);
            const HVX_Vector sl =
                Q6_Vh_vsub_VhVh(Q6_V_vand_VV(raw, v_mask_lo), v_zero_lo);
            kv_hi[v] = Q6_Ww_vmpyacc_WwVhRh(kv_hi[v], sh, kh);
            kv_lo[v] = Q6_Ww_vmpyacc_WwVhRh(kv_lo[v], sl, kl);
            out_hi[v] = Q6_Ww_vmpyacc_WwVhRh(out_hi[v], sh, qh);
            out_lo[v] = Q6_Ww_vmpyacc_WwVhRh(out_lo[v], sl, ql);
            if constexpr (kQUint16KaratsubaReduction) {
              // MAC/sub are modulo 2^32. B may wrap at U16 endpoints,
              // but B-A-C recovers the bounded cross term exactly.
              // Keep the original radix so the FP32 combine is unchanged.
              const HVX_Vector sum = Q6_Vh_vadd_VhVh(sh, sl);
              const int km =
                  (key_center[k] >> kSplitShift) + (key_center[k] & kSplitMask);
              const int qm = (query_center[k] >> kSplitShift) +
                  (query_center[k] & kSplitMask);
              kv_cross[v] = Q6_Ww_vmpyacc_WwVhRh(
                  kv_cross[v], sum, Q6_R_combine_RlRl(km, km));
              out_cross[v] = Q6_Ww_vmpyacc_WwVhRh(
                  out_cross[v], sum, Q6_R_combine_RlRl(qm, qm));
            } else {
              kv_cross[v] = Q6_Ww_vmpyacc_WwVhRh(kv_cross[v], sh, kl);
              kv_cross[v] = Q6_Ww_vmpyacc_WwVhRh(kv_cross[v], sl, kh);
              out_cross[v] = Q6_Ww_vmpyacc_WwVhRh(out_cross[v], sh, ql);
              out_cross[v] = Q6_Ww_vmpyacc_WwVhRh(out_cross[v], sl, qh);
            }
          }
        }
        for (size_t v = 0; v < kSplitVectors; ++v) {
          const size_t at = offset + v * kHalfValuesPerVector;
          if constexpr (kQUint16KaratsubaReduction) {
            kv_cross[v] = Q6_Ww_vsub_WwWw(
                Q6_Ww_vsub_WwWw(kv_cross[v], kv_hi[v]), kv_lo[v]);
            out_cross[v] = Q6_Ww_vsub_WwWw(
                Q6_Ww_vsub_WwWw(out_cross[v], out_hi[v]), out_lo[v]);
          }
          const HVX_VectorPair ka =
              Q6_W_vshuff_VVR(Q6_V_hi_W(kv_hi[v]), Q6_V_lo_W(kv_hi[v]), -4);
          const HVX_VectorPair kb = Q6_W_vshuff_VVR(
              Q6_V_hi_W(kv_cross[v]), Q6_V_lo_W(kv_cross[v]), -4);
          const HVX_VectorPair kc =
              Q6_W_vshuff_VVR(Q6_V_hi_W(kv_lo[v]), Q6_V_lo_W(kv_lo[v]), -4);
          const HVX_VectorPair qa =
              Q6_W_vshuff_VVR(Q6_V_hi_W(out_hi[v]), Q6_V_lo_W(out_hi[v]), -4);
          const HVX_VectorPair qb = Q6_W_vshuff_VVR(
              Q6_V_hi_W(out_cross[v]), Q6_V_lo_W(out_cross[v]), -4);
          const HVX_VectorPair qc =
              Q6_W_vshuff_VVR(Q6_V_hi_W(out_lo[v]), Q6_V_lo_W(out_lo[v]), -4);
          vmemu(kv_acc + at) = op_utils::hvxAddF32(
              op_utils::hvxAddF32(
                  op_utils::hvxMulF32(
                      Q6_Vsf_equals_Vw(Q6_V_lo_W(ka)), v_scale_hi),
                  op_utils::hvxMulF32(
                      Q6_Vsf_equals_Vw(Q6_V_lo_W(kb)), v_scale_cross)),
              Q6_Vsf_equals_Vw(Q6_V_lo_W(kc)));
          vmemu(kv_acc + at + kWordValuesPerVector) = op_utils::hvxAddF32(
              op_utils::hvxAddF32(
                  op_utils::hvxMulF32(
                      Q6_Vsf_equals_Vw(Q6_V_hi_W(ka)), v_scale_hi),
                  op_utils::hvxMulF32(
                      Q6_Vsf_equals_Vw(Q6_V_hi_W(kb)), v_scale_cross)),
              Q6_Vsf_equals_Vw(Q6_V_hi_W(kc)));
          vmemu(output_pre_acc + at) = op_utils::hvxAddF32(
              op_utils::hvxAddF32(
                  op_utils::hvxMulF32(
                      Q6_Vsf_equals_Vw(Q6_V_lo_W(qa)), v_scale_hi),
                  op_utils::hvxMulF32(
                      Q6_Vsf_equals_Vw(Q6_V_lo_W(qb)), v_scale_cross)),
              Q6_Vsf_equals_Vw(Q6_V_lo_W(qc)));
          vmemu(output_pre_acc + at + kWordValuesPerVector) =
              op_utils::hvxAddF32(
                  op_utils::hvxAddF32(
                      op_utils::hvxMulF32(
                          Q6_Vsf_equals_Vw(Q6_V_hi_W(qa)), v_scale_hi),
                      op_utils::hvxMulF32(
                          Q6_Vsf_equals_Vw(Q6_V_hi_W(qb)), v_scale_cross)),
                  Q6_Vsf_equals_Vw(Q6_V_hi_W(qc)));
        }
      }
      return;
    }
  }
  if constexpr (kQUint16FullPrecisionReduction) {
    for (size_t offset = 0; offset < tile_size;
         offset += kHalfValuesPerVector * Vectors) {
      HVX_Vector kv_lo[Vectors], kv_hi[Vectors];
      HVX_Vector out_lo[Vectors], out_hi[Vectors];
      for (size_t v = 0; v < Vectors; ++v) {
        kv_lo[v] = kv_hi[v] = Q6_V_vzero();
        out_lo[v] = out_hi[v] = Q6_V_vzero();
      }
      for (size_t k = 0; k < key_dim; ++k) {
        const HVX_Vector v_key =
            op_utils::hvxSplatF32(static_cast<float>(key_center[k]));
        const HVX_Vector v_query =
            op_utils::hvxSplatF32(static_cast<float>(query_center[k]));
        for (size_t v = 0; v < Vectors; ++v) {
          const size_t at = offset + v * kHalfValuesPerVector;
          const HVX_Vector raw =
              (Aligned ? *reinterpret_cast<const HVX_Vector*>(
                             prev_state + k * state_stride + at)
                       : vmemu(prev_state + k * state_stride + at));
          // Center before converting; there is no low-bit truncation or
          // int16 overflow, even when raw-z is outside signed int16.
          const HVX_VectorPair centered =
              op_utils::hvxCenteredShiftedU16ToI32FromVector(
                  raw, prev_zero_offset, 0);
          const HVX_Vector lo = Q6_Vsf_equals_Vw(Q6_V_lo_W(centered));
          const HVX_Vector hi = Q6_Vsf_equals_Vw(Q6_V_hi_W(centered));
          kv_lo[v] =
              op_utils::hvxAddF32(kv_lo[v], op_utils::hvxMulF32(lo, v_key));
          kv_hi[v] =
              op_utils::hvxAddF32(kv_hi[v], op_utils::hvxMulF32(hi, v_key));
          out_lo[v] =
              op_utils::hvxAddF32(out_lo[v], op_utils::hvxMulF32(lo, v_query));
          out_hi[v] =
              op_utils::hvxAddF32(out_hi[v], op_utils::hvxMulF32(hi, v_query));
        }
      }
      for (size_t v = 0; v < Vectors; ++v) {
        const size_t at = offset + v * kHalfValuesPerVector;
        vmemu(kv_acc + at) = kv_lo[v];
        vmemu(kv_acc + at + kWordValuesPerVector) = kv_hi[v];
        vmemu(output_pre_acc + at) = out_lo[v];
        vmemu(output_pre_acc + at + kWordValuesPerVector) = out_hi[v];
      }
    }
  } else {
    const HVX_Vector v_zero = Q6_Vh_vsplat_R(
        (prev_zero_offset >> kQUint16HvxAccumInputShift) & 0xFFFF);

    for (size_t offset = 0; offset < tile_size;
         offset += kHalfValuesPerVector * Vectors) {
      // 一个 vector block 的 K/Q 累加器跨整个 K 循环保留在 HVX 寄存器中。
      // 两个 block 共 8 个向量累加器，避免逐 K 读写 kv_acc/output_pre_acc。
      HVX_VectorPair kv_accum[Vectors], out_accum[Vectors];
      for (size_t v = 0; v < Vectors; ++v) {
        kv_accum[v] = Q6_W_vzero();
        out_accum[v] = Q6_W_vzero();
      }
      for (size_t k = 0; k < key_dim; ++k) {
        const int32_t key = key_center[k] >> kQUint16HvxAccumInputShift;
        const int32_t query = query_center[k] >> kQUint16HvxAccumInputShift;
        for (size_t v = 0; v < Vectors; ++v) {
          const size_t at = offset + v * kHalfValuesPerVector;
          const HVX_Vector v_raw =
              (Aligned ? *reinterpret_cast<const HVX_Vector*>(
                             prev_state + k * state_stride + at)
                       : vmemu(prev_state + k * state_stride + at));
          const HVX_Vector v_prev = Q6_Vh_vsub_VhVh(
              Q6_Vuh_vlsr_VuhR(v_raw, kQUint16HvxAccumInputShift), v_zero);
          kv_accum[v] = Q6_Ww_vmpyacc_WwVhRh(
              kv_accum[v], v_prev, Q6_R_combine_RlRl(key, key));
          out_accum[v] = Q6_Ww_vmpyacc_WwVhRh(
              out_accum[v], v_prev, Q6_R_combine_RlRl(query, query));
        }
      }
      // 全部 K 完成后，将乘法产生的偶/奇 lane 恢复为连续 V 顺序，只写一次。
      for (size_t v = 0; v < Vectors; ++v) {
        const size_t at = offset + v * kHalfValuesPerVector;
        const HVX_VectorPair kv =
            Q6_W_vshuff_VVR(Q6_V_hi_W(kv_accum[v]), Q6_V_lo_W(kv_accum[v]), -4);
        const HVX_VectorPair out = Q6_W_vshuff_VVR(
            Q6_V_hi_W(out_accum[v]), Q6_V_lo_W(out_accum[v]), -4);
        vmemu(kv_acc + at) = Q6_V_lo_W(kv);
        vmemu(kv_acc + at + kWordValuesPerVector) = Q6_V_hi_W(kv);
        vmemu(output_pre_acc + at) = Q6_V_lo_W(out);
        vmemu(output_pre_acc + at + kWordValuesPerVector) = Q6_V_hi_W(out);
      }
    }
  }
}

// 逐 state 行的定点更新保底路径，负 decay 系数也由此处理。
// 计算 C*(rawS-zPrev)+keyCenter*D，再右移 final_shift、加输出零点并饱和为 U16。
// 当前 state 输入不缩位；调用位于 K 循环内，因此每行都重新加载 delta，尾部用
// int64。
static inline void deltanetQhpiHvxUpdateStateU16Approx(
    const uint16_t* prev_state,
    uint16_t* state,
    const int32_t* delta_update_fixed32,
    int32_t prev_zero_offset,
    int32_t state_zero_offset,
    int32_t key_value,
    int32_t prev_update_value,
    int final_shift,
    size_t value_dim) {
  const size_t halfs_per_vec = 128 / sizeof(uint16_t);
  const size_t words_per_vec = 128 / sizeof(int32_t);
  const size_t vec_count = value_dim / halfs_per_vec;
  const int v_key = Q6_R_combine_RlRl(key_value, key_value);
  const int v_prev_update =
      Q6_R_combine_RlRl(prev_update_value, prev_update_value);
  const HVX_Vector v_state_zero =
      Q6_V_vsplat_R(state_zero_offset << final_shift);

  for (size_t i = 0; i < vec_count; ++i) {
    const size_t offset = i * halfs_per_vec;
    const HVX_VectorPair v_prev = op_utils::hvxCenteredShiftedU16ToI32(
        prev_state + offset,
        prev_zero_offset,
        kQUint16HvxStateUpdateInputShift);
    const HVX_Vector v_delta_lo = vmemu(delta_update_fixed32 + offset);
    const HVX_Vector v_delta_hi =
        vmemu(delta_update_fixed32 + offset + words_per_vec);
    HVX_Vector v_state_lo = Q6_Vw_vmpyi_VwRh(v_delta_lo, v_key);
    HVX_Vector v_state_hi = Q6_Vw_vmpyi_VwRh(v_delta_hi, v_key);
    v_state_lo =
        Q6_Vw_vmpyiacc_VwVwRh(v_state_lo, Q6_V_lo_W(v_prev), v_prev_update);
    v_state_hi =
        Q6_Vw_vmpyiacc_VwVwRh(v_state_hi, Q6_V_hi_W(v_prev), v_prev_update);
    v_state_lo = Q6_Vw_vadd_VwVw(v_state_lo, v_state_zero);
    v_state_hi = Q6_Vw_vadd_VwVw(v_state_hi, v_state_zero);
    const HVX_Vector v_state_half =
        Q6_Vuh_vasr_VwVwR_rnd_sat(v_state_hi, v_state_lo, final_shift);
    vmemu(state + offset) = Q6_Vh_vdeal_Vh(v_state_half);
  }

  for (size_t v = vec_count * halfs_per_vec; v < value_dim; ++v) {
    const int32_t prev = (static_cast<int32_t>(prev_state[v]) >>
                          kQUint16HvxStateUpdateInputShift) -
        (prev_zero_offset >> kQUint16HvxStateUpdateInputShift);
    const int64_t state_center = op_utils::roundShiftI64(
        static_cast<int64_t>(prev_update_value) * prev +
            static_cast<int64_t>(key_value) * delta_update_fixed32[v],
        final_shift);
    state[v] = op_utils::clampCodeToU16(
        static_cast<int64_t>(state_zero_offset) + state_center);
  }
}

// 第二遍读写 state，非负 C 的整数快速路径：F=final_shift，D[v] 为定点 delta。
// 当前 input_shift=0：code=round((C*rawState+keyCenter*D+offset)/2^F)，
// offset=zOut*2^F-zPrev*C，把逐元素减零点改为一次常量准备。
// 直接用 raw U16 乘 C，避免 U16->I32 展开；delta 预排偶/奇 lane 并跨 K 复用。
// Vectors=1/2 处理 64/128 个 V，Aligned 选择对齐访存；系数范围和 tile
// 对齐由调用方保证。
template <bool Aligned, size_t Vectors = 1>
static inline void deltanetQhpiHvxUpdateStateTileU16Approx(
    const uint16_t* prev_state,
    uint16_t* state,
    const int32_t* delta_update_fixed32,
    int32_t prev_zero_offset,
    int32_t state_zero_offset,
    const int32_t* key_center,
    int32_t prev_update_value,
    int final_shift,
    size_t key_dim,
    size_t state_stride,
    size_t tile_size) {
  constexpr size_t kHalfValuesPerVector = 128 / sizeof(uint16_t);
  constexpr size_t kWordValuesPerVector = 128 / sizeof(int32_t);
  const int v_prev_update =
      Q6_R_combine_RlRl(prev_update_value, prev_update_value);
  const int64_t state_offset =
      (static_cast<int64_t>(state_zero_offset) << final_shift) +
      -static_cast<int64_t>(
          prev_zero_offset >> kQUint16HvxStateUpdateInputShift) *
          prev_update_value;
  const HVX_Vector v_state_offset =
      Q6_V_vsplat_R(static_cast<int32_t>(state_offset));

  for (size_t offset = 0; offset < tile_size;
       offset += kHalfValuesPerVector * Vectors) {
    HVX_VectorPair v_delta[Vectors];
    // delta 只加载一次，并预排成偶/奇 V 通道以匹配 unsigned halfword 乘积。
    // 因而 K 循环末尾 pack 可直接写回连续 state，不再逐行做 vdeal。
    for (size_t v = 0; v < Vectors; ++v) {
      const size_t at = offset + v * kHalfValuesPerVector;
      v_delta[v] = Q6_W_vdeal_VVR(
          vmemu(delta_update_fixed32 + at + kWordValuesPerVector),
          vmemu(delta_update_fixed32 + at),
          -4);
    }
    for (size_t k = 0; k < key_dim; ++k) {
      const int key = key_center[k] >> kQUint16HvxStateUpdateInputShift;
      const int v_key = Q6_R_combine_RlRl(key, key);
      for (size_t v = 0; v < Vectors; ++v) {
        const size_t index =
            k * state_stride + offset + v * kHalfValuesPerVector;
        const HVX_Vector raw = Aligned
            ? *reinterpret_cast<const HVX_Vector*>(prev_state + index)
            : vmemu(prev_state + index);
        const HVX_Vector shifted = kQUint16HvxStateUpdateInputShift == 0
            ? raw
            : Q6_Vuh_vlsr_VuhR(raw, kQUint16HvxStateUpdateInputShift);
        const HVX_VectorPair v_prev_term =
            Q6_Wuw_vmpy_VuhRuh(shifted, v_prev_update);
        HVX_Vector v_state_lo = Q6_Vw_vmpyiacc_VwVwRh(
            Q6_V_lo_W(v_prev_term), Q6_V_lo_W(v_delta[v]), v_key);
        HVX_Vector v_state_hi = Q6_Vw_vmpyiacc_VwVwRh(
            Q6_V_hi_W(v_prev_term), Q6_V_hi_W(v_delta[v]), v_key);
        v_state_lo = Q6_Vw_vadd_VwVw(v_state_lo, v_state_offset);
        v_state_hi = Q6_Vw_vadd_VwVw(v_state_hi, v_state_offset);
        const HVX_Vector v_state_half =
            Q6_Vuh_vasr_VwVwR_rnd_sat(v_state_hi, v_state_lo, final_shift);
        if constexpr (Aligned) {
          *reinterpret_cast<HVX_Vector*>(state + index) = v_state_half;
        } else {
          vmemu(state + index) = v_state_half;
        }
      }
    }
  }
}

// 逐 K 行的 FP32 更新保底路径：在 code 域计算再舍入、饱和回 U16。
// prev_scale=decay*sPrev/sOut；这里的 key_scale 已包含本行
// keyCenter*sKey/sOut， 与 TileU16F32 的 key_scale 参数含义不同；delta
// 是浮点实值，state 存储仍为 U16。
static inline void deltanetQhpiHvxUpdateStateU16F32(
    const uint16_t* prev_state,
    uint16_t* state,
    const float* delta_value,
    int32_t prev_zero_offset,
    int32_t state_zero_offset,
    float prev_scale,
    float key_scale,
    size_t value_dim) {
  const size_t halfs_per_vec = 128 / sizeof(uint16_t);
  const size_t words_per_vec = 128 / sizeof(int32_t);
  const size_t vec_count = value_dim / halfs_per_vec;
  const HVX_Vector v_prev_scale = op_utils::hvxSplatF32(prev_scale);
  const HVX_Vector v_key_scale = op_utils::hvxSplatF32(key_scale);
  const HVX_Vector v_state_zero =
      op_utils::hvxSplatF32(static_cast<float>(state_zero_offset) + 0.5f);

  for (size_t i = 0; i < vec_count; ++i) {
    const size_t offset = i * halfs_per_vec;
    const HVX_VectorPair v_prev = op_utils::hvxCenteredShiftedU16ToI32(
        prev_state + offset, prev_zero_offset, 0);
    const HVX_Vector v_prev_lo = Q6_Vsf_equals_Vw(Q6_V_lo_W(v_prev));
    const HVX_Vector v_prev_hi = Q6_Vsf_equals_Vw(Q6_V_hi_W(v_prev));
    const HVX_Vector v_delta_lo = vmemu(delta_value + offset);
    const HVX_Vector v_delta_hi = vmemu(delta_value + offset + words_per_vec);
    const HVX_Vector v_code_lo = op_utils::hvxAddF32(
        v_state_zero,
        op_utils::hvxAddF32(
            op_utils::hvxMulF32(v_prev_lo, v_prev_scale),
            op_utils::hvxMulF32(v_delta_lo, v_key_scale)));
    const HVX_Vector v_code_hi = op_utils::hvxAddF32(
        v_state_zero,
        op_utils::hvxAddF32(
            op_utils::hvxMulF32(v_prev_hi, v_prev_scale),
            op_utils::hvxMulF32(v_delta_hi, v_key_scale)));
    const HVX_Vector v_code_i_lo = Q6_Vw_equals_Vsf(v_code_lo);
    const HVX_Vector v_code_i_hi = Q6_Vw_equals_Vsf(v_code_hi);
    const HVX_Vector v_state_half =
        Q6_Vuh_vasr_VwVwR_sat(v_code_i_hi, v_code_i_lo, 0);
    vmemu(state + offset) = Q6_Vh_vdeal_Vh(v_state_half);
  }

  for (size_t v = vec_count * halfs_per_vec; v < value_dim; ++v) {
    const float prev = static_cast<float>(
        static_cast<int32_t>(prev_state[v]) - prev_zero_offset);
    const float code = static_cast<float>(state_zero_offset) +
        prev * prev_scale + delta_value[v] * key_scale;
    state[v] = op_utils::quantU16FromScaledDouble(static_cast<double>(code));
  }
}

// 系数/key 不满足整数快速路径时，用 FP32 在 code 域更新，再舍入到 U16。
// prev_scale=decay*sPrev/sOut，key_scale=sKey/sOut，每行另乘 keyCenter。
// delta 向量跨 K 复用，Vectors=1/2 的 block 共用每行 key 系数；
// Aligned 选择对齐访存，tile 大小由调用方对齐，state 的存储格式始终是 U16。
template <bool Aligned, size_t Vectors = 1>
static inline void deltanetQhpiHvxUpdateStateTileU16F32(
    const uint16_t* prev_state,
    uint16_t* state,
    const float* delta_value,
    int32_t prev_zero_offset,
    int32_t state_zero_offset,
    const int32_t* key_center,
    float prev_scale,
    double key_scale,
    size_t key_dim,
    size_t state_stride,
    size_t tile_size) {
  constexpr size_t kWordValuesPerVector = 128 / sizeof(int32_t);
  const HVX_Vector v_prev_scale = op_utils::hvxSplatF32(prev_scale);
  const HVX_Vector v_state_zero =
      op_utils::hvxSplatF32(static_cast<float>(state_zero_offset) + 0.5f);
  for (size_t offset = 0; offset < tile_size;
       offset += kQUint16HvxValuesPerVec * Vectors) {
    HVX_Vector v_delta_lo[Vectors], v_delta_hi[Vectors];
    for (size_t v = 0; v < Vectors; ++v) {
      const size_t at = offset + v * kQUint16HvxValuesPerVec;
      v_delta_lo[v] = vmemu(delta_value + at);
      v_delta_hi[v] = vmemu(delta_value + at + kWordValuesPerVector);
    }
    for (size_t k = 0; k < key_dim; ++k) {
      const HVX_Vector v_key_scale = op_utils::hvxSplatF32(
          static_cast<float>(static_cast<double>(key_center[k]) * key_scale));
      for (size_t v = 0; v < Vectors; ++v) {
        const size_t index =
            k * state_stride + offset + v * kQUint16HvxValuesPerVec;
        const HVX_Vector raw = Aligned
            ? *reinterpret_cast<const HVX_Vector*>(prev_state + index)
            : vmemu(prev_state + index);
        const HVX_VectorPair v_prev =
            op_utils::hvxCenteredShiftedU16ToI32FromVector(
                raw, prev_zero_offset, 0);
        const HVX_Vector v_code_lo = op_utils::hvxAddF32(
            v_state_zero,
            op_utils::hvxAddF32(
                op_utils::hvxMulF32(
                    Q6_Vsf_equals_Vw(Q6_V_lo_W(v_prev)), v_prev_scale),
                op_utils::hvxMulF32(v_delta_lo[v], v_key_scale)));
        const HVX_Vector v_code_hi = op_utils::hvxAddF32(
            v_state_zero,
            op_utils::hvxAddF32(
                op_utils::hvxMulF32(
                    Q6_Vsf_equals_Vw(Q6_V_hi_W(v_prev)), v_prev_scale),
                op_utils::hvxMulF32(v_delta_hi[v], v_key_scale)));
        const HVX_Vector packed = Q6_Vh_vdeal_Vh(Q6_Vuh_vasr_VwVwR_sat(
            Q6_Vw_equals_Vsf(v_code_hi), Q6_Vw_equals_Vsf(v_code_lo), 0));
        if constexpr (Aligned) {
          *reinterpret_cast<HVX_Vector*>(state + index) = packed;
        } else {
          vmemu(state + index) = packed;
        }
      }
    }
  }
}

// 归约完成后计算 delta=beta*(value-decay*K^T*S)，y=decay*Q^T*S+(Q·K)*delta。
// 直接用旧 state 的归约结果求输出，避免更新后再读第三遍 state。
// delta 可选写为 FP32 实值或定点 int32（对应指针可为 nullptr），y 量化成 U16。
// 只处理完整的 64 个 V 块，返回已处理的 V 数，标量尾部由调用方继续处理。
static inline size_t deltanetQhpiHvxDeltaOutputU16F32(
    uint16_t* output,
    const uint16_t* value,
    const QUint16StateAccum* kv_acc,
    const QUint16StateAccum* output_pre_acc,
    float* delta_value,
    int32_t* delta_update_fixed32,
    int32_t value_zero_offset,
    int32_t output_zero_offset,
    float value_beta_scale,
    float kv_beta_scale,
    float output_acc_scale,
    float output_delta_scale,
    float delta_update_fixed32_scale,
    size_t value_dim) {
  const size_t halfs_per_vec = 128 / sizeof(uint16_t);
  const size_t words_per_vec = 128 / sizeof(int32_t);
  const size_t vec_count = value_dim / halfs_per_vec;
  const HVX_Vector v_value_beta = op_utils::hvxSplatF32(value_beta_scale);
  const HVX_Vector v_neg_kv_beta = op_utils::hvxSplatF32(-kv_beta_scale);
  const HVX_Vector v_output_acc = op_utils::hvxSplatF32(output_acc_scale);
  const HVX_Vector v_output_delta = op_utils::hvxSplatF32(output_delta_scale);
  const HVX_Vector v_delta_update =
      op_utils::hvxSplatF32(delta_update_fixed32_scale);
  const HVX_Vector v_output_zero =
      op_utils::hvxSplatF32(static_cast<float>(output_zero_offset) + 0.5f);

  for (size_t i = 0; i < vec_count; ++i) {
    const size_t offset = i * halfs_per_vec;
    const HVX_VectorPair v_value = op_utils::hvxCenteredShiftedU16ToI32(
        value + offset, value_zero_offset, 0);
    const HVX_Vector v_value_lo = Q6_Vsf_equals_Vw(Q6_V_lo_W(v_value));
    const HVX_Vector v_value_hi = Q6_Vsf_equals_Vw(Q6_V_hi_W(v_value));
    HVX_Vector v_kv_lo = vmemu(kv_acc + offset);
    HVX_Vector v_kv_hi = vmemu(kv_acc + offset + words_per_vec);
    HVX_Vector v_pre_lo = vmemu(output_pre_acc + offset);
    HVX_Vector v_pre_hi = vmemu(output_pre_acc + offset + words_per_vec);
    if constexpr (!kQUint16FullPrecisionReduction) {
      v_kv_lo = Q6_Vsf_equals_Vw(v_kv_lo);
      v_kv_hi = Q6_Vsf_equals_Vw(v_kv_hi);
      v_pre_lo = Q6_Vsf_equals_Vw(v_pre_lo);
      v_pre_hi = Q6_Vsf_equals_Vw(v_pre_hi);
    }

    const HVX_Vector v_delta_lo = op_utils::hvxAddF32(
        op_utils::hvxMulF32(v_value_lo, v_value_beta),
        op_utils::hvxMulF32(v_kv_lo, v_neg_kv_beta));
    const HVX_Vector v_delta_hi = op_utils::hvxAddF32(
        op_utils::hvxMulF32(v_value_hi, v_value_beta),
        op_utils::hvxMulF32(v_kv_hi, v_neg_kv_beta));
    const HVX_Vector v_code_lo = op_utils::hvxAddF32(
        v_output_zero,
        op_utils::hvxAddF32(
            op_utils::hvxMulF32(v_pre_lo, v_output_acc),
            op_utils::hvxMulF32(v_delta_lo, v_output_delta)));
    const HVX_Vector v_code_hi = op_utils::hvxAddF32(
        v_output_zero,
        op_utils::hvxAddF32(
            op_utils::hvxMulF32(v_pre_hi, v_output_acc),
            op_utils::hvxMulF32(v_delta_hi, v_output_delta)));

    if (delta_value != nullptr) {
      vmemu(delta_value + offset) = v_delta_lo;
      vmemu(delta_value + offset + words_per_vec) = v_delta_hi;
    }
    if (delta_update_fixed32 != nullptr) {
      vmemu(delta_update_fixed32 + offset) = op_utils::hvxRoundF32ToI32(
          op_utils::hvxMulF32(v_delta_lo, v_delta_update));
      vmemu(delta_update_fixed32 + offset + words_per_vec) =
          op_utils::hvxRoundF32ToI32(
              op_utils::hvxMulF32(v_delta_hi, v_delta_update));
    }
    const HVX_Vector v_code_i_lo = Q6_Vw_equals_Vsf(v_code_lo);
    const HVX_Vector v_code_i_hi = Q6_Vw_equals_Vsf(v_code_hi);
    const HVX_Vector v_output_half =
        Q6_Vuh_vasr_VwVwR_sat(v_code_i_hi, v_code_i_lo, 0);
    vmemu(output + offset) = Q6_Vh_vdeal_Vh(v_output_half);
  }

  return vec_count * halfs_per_vec;
}

// 主 recurrence：一个 (batch,head,V tile) 完整执行 seq_len 个 token。
// Q/K/value/output 布局 [B,T,H,D]，state 布局 [B,H,K,V]，V 连续。
// P0 准备 Q/K 和系数 -> P1 state 归约 -> P2 delta/y -> P3 state 更新。
// 整数归约尚未含 decay/stepsize，尺度在 P2 合入；更新选择定点快速路径或 FP32
// 保底。 同一 head 的 token 顺序执行，不同 head/V tile 的 state 独立；
// StateResident=false 使用 DDR，true 使用 signature 指定的 VTCM
// 并在首步拷入初始 state。
template <bool StateResident, bool DecodeGate = false, bool FixedDecode = false>
static inline void deltanetCoreQhpiRunQUint16Tile(
    uint16_t* output_data,
    uint16_t* state_data,
    const uint16_t* query_data,
    const uint16_t* key_data,
    const uint16_t* value_data,
    const uint16_t* decay_data,
    const uint16_t* beta_data,
    const uint16_t* recurrent_state_data,
    QHPI_Quant_Parameters query_q,
    QHPI_Quant_Parameters key_q,
    QHPI_Quant_Parameters value_q,
    QHPI_Quant_Parameters decay_q,
    QHPI_Quant_Parameters beta_q,
    QHPI_Quant_Parameters recurrent_state_q,
    QHPI_Quant_Parameters output_q,
    QHPI_Quant_Parameters state_q,
    const QHPI_Shape& decay_shape,
    const QHPI_Shape& beta_shape,
    size_t runtime_seq_len,
    size_t runtime_heads,
    size_t runtime_key_dim,
    size_t runtime_value_dim,
    size_t runtime_row,
    size_t runtime_v_begin,
    size_t runtime_v_end,
    int64_t* kv_acc,
    int64_t* output_pre_acc,
    QUint16StateAccum* kv_projection,
    QUint16StateAccum* output_pre_projection,
    int64_t* delta_update_fixed,
    int32_t* delta_update_fixed32,
    float* delta_value_scratch,
    int32_t* key_center,
    int32_t* query_center,
    DeltaNetPhaseProfile& profile,
    const uint16_t* decay_lookup = nullptr) {
  const size_t seq_len = FixedDecode ? 1 : runtime_seq_len;
  const size_t heads = FixedDecode ? 1 : runtime_heads;
  const size_t key_dim = FixedDecode ? 128 : runtime_key_dim;
  const size_t value_dim = FixedDecode ? 128 : runtime_value_dim;
  const size_t row = FixedDecode ? 0 : runtime_row;
  const size_t v_begin = FixedDecode ? 0 : runtime_v_begin;
  const size_t v_end = FixedDecode ? 128 : runtime_v_end;
  const size_t tile_size = v_end - v_begin;
  if (tile_size == 0) {
    return;
  }

  const size_t b = row / heads;
  const size_t h = row % heads;
  const size_t state_base = row * key_dim * value_dim;

  profile.start();
  // VTCM 入口把初始 state 的原始 U16 code 拷入常驻输出区一次。
  // 此处没有重量化；首 token 仍按 recurrent_state_q 解释这些 code。
  // H=16,K=V=128 时整个 state 占 512 KiB，每 head 32 KiB。
  const bool copy_with_reduction = DELTANET_U16_COPY_WITH_REDUCTION &&
      StateResident && seq_len == 1 && key_dim == 128 && value_dim == 128 &&
      tile_size == 128 && kQUint16SplitReduction &&
      op_utils::hasU16ZeroOffset(recurrent_state_q) &&
      op_utils::hasU16ZeroOffset(key_q) && op_utils::hasU16ZeroOffset(query_q);
  if (StateResident && !copy_with_reduction) {
    if (tile_size == value_dim) {
      deltanetQhpiCopyStateU16(
          state_data + state_base,
          recurrent_state_data + state_base,
          key_dim * value_dim);
    } else {
      for (size_t k = 0; k < key_dim; ++k) {
        const size_t row_base = state_base + k * value_dim + v_begin;
        std::memcpy(
            state_data + row_base,
            recurrent_state_data + row_base,
            tile_size * sizeof(uint16_t));
      }
    }
  }

  profile.record(4);
  const bool bounded_qk =
      op_utils::hasU16ZeroOffset(key_q) && op_utils::hasU16ZeroOffset(query_q);
  const double delta_update_scale = key_q.stepsize / state_q.stepsize;
  const double key_state_scale =
      static_cast<double>(key_q.stepsize) / state_q.stepsize;

  for (size_t t = 0; t < seq_len; ++t) {
    const size_t qkv_base = ((b * seq_len + t) * heads + h);
    const uint16_t* query_row = query_data + qkv_base * key_dim;
    const uint16_t* key_row = key_data + qkv_base * key_dim;
    const uint16_t* value_tile = value_data + qkv_base * value_dim + v_begin;

    // P0：本 token 的 Q/K、Q·K、decay/beta 和定点更新系数。
    // Q/K 的 L2 norm、beta=sigmoid(b)、decay=exp(g) 已在本算子之前算好。
    int64_t key_query_acc = 0;
    bool key_update_fits = true;
    bool qk_accum_fits = true;
    if (bounded_qk && key_dim % kQUint16HvxValuesPerVec == 0) {
      key_query_acc = deltanetQhpiHvxPrepareQkU16(
          query_row,
          key_row,
          query_center,
          key_center,
          query_q.zero_offset,
          key_q.zero_offset,
          key_dim,
          key_update_fits);
    } else {
      for (size_t k = 0; k < key_dim; ++k) {
        key_center[k] = op_utils::centeredU16(key_row[k], key_q);
        query_center[k] = op_utils::centeredU16(query_row[k], query_q);
        key_query_acc += static_cast<int64_t>(key_center[k]) * query_center[k];
        key_update_fits &= op_utils::fitsI16(
            key_center[k] >> kQUint16HvxStateUpdateInputShift);
        if (!bounded_qk) {
          qk_accum_fits &=
              op_utils::fitsI16(key_center[k] >> kQUint16HvxAccumInputShift) &&
              op_utils::fitsI16(query_center[k] >> kQUint16HvxAccumInputShift);
        }
      }
    }
    const double key_query_dot =
        static_cast<double>(key_query_acc) * key_q.stepsize * query_q.stepsize;

    uint16_t decay_code =
        decay_data[op_utils::timeHeadIndex(decay_shape, b, t, h)];
    if constexpr (DecodeGate)
      decay_code = decay_lookup[h * 65536 + decay_code];
    const double decay = op_utils::dequantU16(decay_code, decay_q);
    const double beta_value = op_utils::dequantU16(
        beta_data[op_utils::timeHeadIndex(beta_shape, b, t, h)], beta_q);
    // DDR 首步读输入 state；VTCM 首步读其拷贝。二者均使用输入 qparams。
    // 第一次更新已写成 state_q，此后所有 token 均原地使用输出 qparams。
    const QHPI_Quant_Parameters prev_q = t == 0 ? recurrent_state_q : state_q;
    bool hvx_accum = op_utils::hasU16ZeroOffset(prev_q) &&
        (kQUint16FullPrecisionReduction
             ? tile_size % kQUint16HvxValuesPerVec == 0
             : qk_accum_fits);

    // C20=round(decay*sPrev/sOut*2^20)，再舍入右移 7 得到 C13。
    // delta 系数 D13=round(delta*sKey/sOut*2^13)；更新最终右移 13 位。
    const int64_t prev_update_fixed = op_utils::roundFixed(
        decay * prev_q.stepsize / state_q.stepsize, kQUint16StateUpdateShift);
    int state_update_final_shift = 0;
    int64_t prev_update_shifted = 0;
    bool hvx_update = false;
    constexpr bool hvx_update_f32 = kEnableQUint16HvxStateUpdateF32;
    if constexpr (kEnableQUint16HvxStateUpdate) {
      state_update_final_shift = kQUint16StateUpdateShift -
          kQUint16HvxStateUpdateInputShift - kQUint16HvxStateUpdateCoeffShift;
      prev_update_shifted = op_utils::roundShiftI64(
          prev_update_fixed, kQUint16HvxStateUpdateCoeffShift);
      hvx_update = state_update_final_shift > 0 &&
          op_utils::hasU16ZeroOffset(prev_q) &&
          op_utils::fitsI16(prev_update_shifted) && key_update_fits &&
          op_utils::fitsI32(
                       static_cast<int64_t>(state_q.zero_offset)
                       << state_update_final_shift);
    }
    profile.record(0);
    // P1：一遍 state 同时得到 kv_acc=K^T*S、output_pre_acc=Q^T*S。
    // 64 通道整块走寄存器归约；不满足条件则走逐 K scratch/int64 路径。
    if (hvx_accum && tile_size % kQUint16HvxValuesPerVec == 0) {
      const uint16_t* prev_state =
          (t == 0 && (!StateResident || copy_with_reduction)
               ? recurrent_state_data
               : state_data) +
          state_base + v_begin;
      if (copy_with_reduction) {
        deltanetQhpiHvxReduceStateU16<false, 1, true>(
            prev_state,
            kv_projection,
            output_pre_projection,
            prev_q.zero_offset,
            key_q.zero_offset,
            query_q.zero_offset,
            key_center,
            query_center,
            key_dim,
            value_dim,
            tile_size,
            state_data + state_base + v_begin);
      } else if (
          reinterpret_cast<uintptr_t>(prev_state) % 128 == 0 &&
          value_dim % kQUint16HvxValuesPerVec == 0) {
        if (tile_size % (2 * kQUint16HvxValuesPerVec) == 0) {
          deltanetQhpiHvxReduceStateU16<true, 2>(
              prev_state,
              kv_projection,
              output_pre_projection,
              prev_q.zero_offset,
              key_q.zero_offset,
              query_q.zero_offset,
              key_center,
              query_center,
              key_dim,
              value_dim,
              tile_size);
        } else {
          deltanetQhpiHvxReduceStateU16<true>(
              prev_state,
              kv_projection,
              output_pre_projection,
              prev_q.zero_offset,
              key_q.zero_offset,
              query_q.zero_offset,
              key_center,
              query_center,
              key_dim,
              value_dim,
              tile_size);
        }
      } else {
        deltanetQhpiHvxReduceStateU16<false>(
            prev_state,
            kv_projection,
            output_pre_projection,
            prev_q.zero_offset,
            key_q.zero_offset,
            query_q.zero_offset,
            key_center,
            query_center,
            key_dim,
            value_dim,
            tile_size);
      }
    } else {
      {
        const size_t row_base = state_base + v_begin;
        const uint16_t* prev_state = t == 0 && !StateResident
            ? recurrent_state_data + row_base
            : state_data + row_base;
        const int32_t key_value = key_center[0];
        const int32_t query_value = query_center[0];
        if (hvx_accum) {
          deltanetQhpiHvxAssignAccumU16I32(
              prev_state,
              reinterpret_cast<int32_t*>(kv_projection),
              reinterpret_cast<int32_t*>(output_pre_projection),
              prev_q.zero_offset,
              key_value >> kQUint16HvxAccumInputShift,
              query_value >> kQUint16HvxAccumInputShift,
              tile_size);
        } else {
          for (size_t v = 0; v < tile_size; ++v) {
            const int32_t prev = op_utils::centeredU16(prev_state[v], prev_q);
            kv_acc[v] = static_cast<int64_t>(prev) * key_value;
            output_pre_acc[v] = static_cast<int64_t>(prev) * query_value;
          }
        }
      }

      for (size_t k = 1; k < key_dim; ++k) {
        const size_t row_base = state_base + k * value_dim + v_begin;
        const uint16_t* prev_state = t == 0 && !StateResident
            ? recurrent_state_data + row_base
            : state_data + row_base;
        const int32_t key_value = key_center[k];
        const int32_t query_value = query_center[k];
        if (hvx_accum) {
          deltanetQhpiHvxAccumU16I32(
              prev_state,
              reinterpret_cast<int32_t*>(kv_projection),
              reinterpret_cast<int32_t*>(output_pre_projection),
              prev_q.zero_offset,
              key_value >> kQUint16HvxAccumInputShift,
              query_value >> kQUint16HvxAccumInputShift,
              tile_size);
        } else {
          for (size_t v = 0; v < tile_size; ++v) {
            const int32_t prev = op_utils::centeredU16(prev_state[v], prev_q);
            kv_acc[v] += static_cast<int64_t>(prev) * key_value;
            output_pre_acc[v] += static_cast<int64_t>(prev) * query_value;
          }
        }
      }
    }

    profile.record(1);
    // P2：计算 delta 和本 token 的输出；尚未覆盖旧 state。
    // 完整精度归约无需补偿，旧版缩位路径乘 2^(4+4)=256。
    const double hvx_accum_scale = hvx_accum && !kQUint16FullPrecisionReduction
        ? std::ldexp(1.0, 2 * kQUint16HvxAccumInputShift)
        : 1.0;
    const double kv_scale =
        decay * prev_q.stepsize * key_q.stepsize * hvx_accum_scale;
    const double output_pre_scale =
        decay * prev_q.stepsize * query_q.stepsize * hvx_accum_scale;
    const double output_acc_scale = output_pre_scale / output_q.stepsize;
    const double output_delta_scale = key_query_dot / output_q.stepsize;
    uint16_t* output_tile = output_data + qkv_base * value_dim + v_begin;
    size_t scalar_v_begin = 0;
    if (hvx_accum && hvx_update_f32) {
      scalar_v_begin = deltanetQhpiHvxDeltaOutputU16F32(
          output_tile,
          value_tile,
          kv_projection,
          output_pre_projection,
          hvx_update && tile_size % kQUint16HvxValuesPerVec == 0
              ? nullptr
              : delta_value_scratch,
          hvx_update ? delta_update_fixed32 : nullptr,
          value_q.zero_offset,
          output_q.zero_offset,
          static_cast<float>(value_q.stepsize * beta_value),
          static_cast<float>(kv_scale * beta_value),
          static_cast<float>(output_acc_scale),
          static_cast<float>(output_delta_scale),
          static_cast<float>(
              delta_update_scale *
              std::ldexp(
                  1.0,
                  kQUint16StateUpdateShift - kQUint16HvxStateUpdateCoeffShift)),
          tile_size);
    }
    for (size_t v = scalar_v_begin; v < tile_size; ++v) {
      const double value_real =
          static_cast<double>(op_utils::centeredU16(value_tile[v], value_q)) *
          value_q.stepsize;
      const double kv_acc_value = hvx_accum
          ? static_cast<double>(kv_projection[v])
          : static_cast<double>(kv_acc[v]);
      const double delta_value =
          (value_real - kv_acc_value * kv_scale) * beta_value;
      delta_value_scratch[v] = static_cast<float>(delta_value);
      if constexpr (!kEnableQUint16HvxStateUpdateF32) {
        delta_update_fixed[v] = op_utils::roundFixed(
            delta_value * delta_update_scale, kQUint16StateUpdateShift);
      }
      if constexpr (kEnableQUint16HvxStateUpdate) {
        delta_update_fixed[v] = op_utils::roundFixed(
            delta_value * delta_update_scale, kQUint16StateUpdateShift);
        const int64_t delta_update_shifted = op_utils::roundShiftI64(
            delta_update_fixed[v], kQUint16HvxStateUpdateCoeffShift);
        if (op_utils::fitsI32(delta_update_shifted)) {
          delta_update_fixed32[v] = static_cast<int32_t>(delta_update_shifted);
        } else {
          delta_update_fixed32[v] = 0;
          hvx_update = false;
        }
      }
      const double output_pre_acc_value = hvx_accum
          ? static_cast<double>(output_pre_projection[v])
          : static_cast<double>(output_pre_acc[v]);
      output_tile[v] = op_utils::quantU16FromScaledDouble(
          static_cast<double>(output_q.zero_offset) +
          output_pre_acc_value * output_acc_scale +
          delta_value * output_delta_scale);
    }

    profile.record(2);
    profile.paths(
        hvx_accum && tile_size % kQUint16HvxValuesPerVec == 0, hvx_update);
    // P3：第二遍 state，S_new=decay*S_old+key*delta^T，写成 state_q 的 code。
    // 非负整数系数用 unsigned halfword 快速路径；整数条件失败走 F32 tile。
    // 负整数系数/非整块使用下面的逐行路径，不能误用 unsigned 乘法。
    if (hvx_update && prev_update_shifted >= 0 &&
        tile_size % kQUint16HvxValuesPerVec == 0) {
      const uint16_t* prev_state =
          (t == 0 && !StateResident ? recurrent_state_data : state_data) +
          state_base + v_begin;
      if (reinterpret_cast<uintptr_t>(prev_state) % 128 == 0 &&
          value_dim % kQUint16HvxValuesPerVec == 0 &&
          reinterpret_cast<uintptr_t>(state_data + state_base + v_begin) %
                  128 ==
              0) {
        if (tile_size % (2 * kQUint16HvxValuesPerVec) == 0) {
          deltanetQhpiHvxUpdateStateTileU16Approx<true, 2>(
              prev_state,
              state_data + state_base + v_begin,
              delta_update_fixed32,
              prev_q.zero_offset,
              state_q.zero_offset,
              key_center,
              static_cast<int32_t>(prev_update_shifted),
              state_update_final_shift,
              key_dim,
              value_dim,
              tile_size);
        } else {
          deltanetQhpiHvxUpdateStateTileU16Approx<true>(
              prev_state,
              state_data + state_base + v_begin,
              delta_update_fixed32,
              prev_q.zero_offset,
              state_q.zero_offset,
              key_center,
              static_cast<int32_t>(prev_update_shifted),
              state_update_final_shift,
              key_dim,
              value_dim,
              tile_size);
        }
      } else {
        deltanetQhpiHvxUpdateStateTileU16Approx<false>(
            prev_state,
            state_data + state_base + v_begin,
            delta_update_fixed32,
            prev_q.zero_offset,
            state_q.zero_offset,
            key_center,
            static_cast<int32_t>(prev_update_shifted),
            state_update_final_shift,
            key_dim,
            value_dim,
            tile_size);
      }
    } else if (
        !hvx_update && hvx_update_f32 &&
        tile_size % kQUint16HvxValuesPerVec == 0) {
      const uint16_t* prev_state =
          (t == 0 && !StateResident ? recurrent_state_data : state_data) +
          state_base + v_begin;
      const float prev_scale =
          static_cast<float>(decay * prev_q.stepsize / state_q.stepsize);
      if (reinterpret_cast<uintptr_t>(prev_state) % 128 == 0 &&
          value_dim % kQUint16HvxValuesPerVec == 0 &&
          reinterpret_cast<uintptr_t>(state_data + state_base + v_begin) %
                  128 ==
              0) {
        if (tile_size % (2 * kQUint16HvxValuesPerVec) == 0) {
          deltanetQhpiHvxUpdateStateTileU16F32<true, 2>(
              prev_state,
              state_data + state_base + v_begin,
              delta_value_scratch,
              prev_q.zero_offset,
              state_q.zero_offset,
              key_center,
              prev_scale,
              key_state_scale,
              key_dim,
              value_dim,
              tile_size);
        } else {
          deltanetQhpiHvxUpdateStateTileU16F32<true>(
              prev_state,
              state_data + state_base + v_begin,
              delta_value_scratch,
              prev_q.zero_offset,
              state_q.zero_offset,
              key_center,
              prev_scale,
              key_state_scale,
              key_dim,
              value_dim,
              tile_size);
        }
      } else {
        deltanetQhpiHvxUpdateStateTileU16F32<false>(
            prev_state,
            state_data + state_base + v_begin,
            delta_value_scratch,
            prev_q.zero_offset,
            state_q.zero_offset,
            key_center,
            prev_scale,
            key_state_scale,
            key_dim,
            value_dim,
            tile_size);
      }
    } else {
      for (size_t k = 0; k < key_dim; ++k) {
        const size_t row_base = state_base + k * value_dim + v_begin;
        const uint16_t* prev_state = t == 0 && !StateResident
            ? recurrent_state_data + row_base
            : state_data + row_base;
        const int32_t key_value = key_center[k];
        uint16_t* state_row = state_data + row_base;
        if (hvx_update) {
          deltanetQhpiHvxUpdateStateU16Approx(
              prev_state,
              state_row,
              delta_update_fixed32,
              prev_q.zero_offset,
              state_q.zero_offset,
              key_value >> kQUint16HvxStateUpdateInputShift,
              static_cast<int32_t>(prev_update_shifted),
              state_update_final_shift,
              tile_size);
        } else if (hvx_update_f32) {
          deltanetQhpiHvxUpdateStateU16F32(
              prev_state,
              state_row,
              delta_value_scratch,
              prev_q.zero_offset,
              state_q.zero_offset,
              static_cast<float>(decay * prev_q.stepsize / state_q.stepsize),
              static_cast<float>(
                  static_cast<double>(key_value) * key_q.stepsize /
                  state_q.stepsize),
              tile_size);
        } else {
          const int64_t key_update = key_value;
          const int64_t state_zero = state_q.zero_offset;
          const int64_t prev_zero = prev_q.zero_offset;
          constexpr int64_t kStateUpdateRounding = INT64_C(1)
              << (kQUint16StateUpdateShift - 1);
          for (size_t v = 0; v < tile_size; ++v) {
            const int64_t prev =
                static_cast<int64_t>(prev_state[v]) - prev_zero;
            const int64_t update =
                prev_update_fixed * prev + key_update * delta_update_fixed[v];
            const int64_t state_center = update >= 0
                ? (update + kStateUpdateRounding) >> kQUint16StateUpdateShift
                : -(((-update) + kStateUpdateRounding) >>
                    kQUint16StateUpdateShift);
            const int64_t code = state_zero + state_center;
            if (code <= 0) {
              state_row[v] = 0;
            } else if (code >= 65535) {
              state_row[v] = 65535;
            } else {
              state_row[v] = static_cast<uint16_t>(code);
            }
          }
        }
      }
    }
    profile.record(3);
  }
}

// QUInt16 的 worker 调度层，分配本 worker 独占的 int64/int32/float scratch。
// 常规单 V tile 时连续分配 head，减少 worker 间交错访存；额外 V
// 分块时按任务交错分配。 DDR/VTCM 通过 StateResident 复用同一计算流程；worker
// 并行与内部 HVX SIMD 分属两层。
template <bool StateResident, bool DecodeGate = false>
static void deltanetCoreQhpiRunQUint16(
    uint16_t* output_data,
    uint16_t* state_data,
    const uint16_t* query_data,
    const uint16_t* key_data,
    const uint16_t* value_data,
    const uint16_t* decay_data,
    const uint16_t* beta_data,
    const uint16_t* recurrent_state_data,
    QHPI_Quant_Parameters query_q,
    QHPI_Quant_Parameters key_q,
    QHPI_Quant_Parameters value_q,
    QHPI_Quant_Parameters decay_q,
    QHPI_Quant_Parameters beta_q,
    QHPI_Quant_Parameters recurrent_state_q,
    QHPI_Quant_Parameters output_q,
    QHPI_Quant_Parameters state_q,
    const QHPI_Shape& decay_shape,
    const QHPI_Shape& beta_shape,
    size_t batch,
    size_t seq_len,
    size_t heads,
    size_t key_dim,
    size_t value_dim,
    size_t slice,
    size_t num_slices,
    size_t value_tile_count,
    const uint16_t* decay_lookup = nullptr,
    uint16_t* profile_output = nullptr) {
  int64_t kv_acc[kMaxDeltaNetDim] __attribute__((aligned(128)));
  int64_t output_pre_acc[kMaxDeltaNetDim] __attribute__((aligned(128)));
  QUint16StateAccum kv_projection[kMaxDeltaNetDim]
      __attribute__((aligned(128)));
  QUint16StateAccum output_pre_projection[kMaxDeltaNetDim]
      __attribute__((aligned(128)));
  int64_t delta_update_fixed[kMaxDeltaNetDim] __attribute__((aligned(128)));
  int32_t delta_update_fixed32[kMaxDeltaNetDim] __attribute__((aligned(128)));
  float delta_value_scratch[kMaxDeltaNetDim] __attribute__((aligned(128)));
  int32_t key_center[kMaxDeltaNetDim] __attribute__((aligned(128)));
  int32_t query_center[kMaxDeltaNetDim] __attribute__((aligned(128)));

  DeltaNetPhaseProfile profile;
  const size_t total_rows = batch * heads;
  if (value_tile_count <= 1) {
    const size_t row_begin = op_utils::divUp(total_rows * slice, num_slices);
    const size_t row_end =
        op_utils::divUp(total_rows * (slice + 1), num_slices);
    for (size_t row = row_begin; row < row_end; ++row) {
      deltanetCoreQhpiRunQUint16Tile<StateResident, DecodeGate>(
          output_data,
          state_data,
          query_data,
          key_data,
          value_data,
          decay_data,
          beta_data,
          recurrent_state_data,
          query_q,
          key_q,
          value_q,
          decay_q,
          beta_q,
          recurrent_state_q,
          output_q,
          state_q,
          decay_shape,
          beta_shape,
          seq_len,
          heads,
          key_dim,
          value_dim,
          row,
          0,
          value_dim,
          kv_acc,
          output_pre_acc,
          kv_projection,
          output_pre_projection,
          delta_update_fixed,
          delta_update_fixed32,
          delta_value_scratch,
          key_center,
          query_center,
          profile,
          decay_lookup);
    }
    profile.report(
        slice, num_slices, seq_len, heads, key_dim, value_dim, StateResident);
    profile.save(profile_output);
    return;
  }

  const size_t total_tasks = total_rows * value_tile_count;
  for (size_t task = slice; task < total_tasks; task += num_slices) {
    const size_t row = task / value_tile_count;
    const size_t tile = task % value_tile_count;
    const size_t v_begin = value_dim * tile / value_tile_count;
    const size_t v_end = value_dim * (tile + 1) / value_tile_count;
    deltanetCoreQhpiRunQUint16Tile<StateResident, DecodeGate>(
        output_data,
        state_data,
        query_data,
        key_data,
        value_data,
        decay_data,
        beta_data,
        recurrent_state_data,
        query_q,
        key_q,
        value_q,
        decay_q,
        beta_q,
        recurrent_state_q,
        output_q,
        state_q,
        decay_shape,
        beta_shape,
        seq_len,
        heads,
        key_dim,
        value_dim,
        row,
        v_begin,
        v_end,
        kv_acc,
        output_pre_acc,
        kv_projection,
        output_pre_projection,
        delta_update_fixed,
        delta_update_fixed32,
        delta_value_scratch,
        key_center,
        query_center,
        profile,
        decay_lookup);
  }
  profile.report(
      slice, num_slices, seq_len, heads, key_dim, value_dim, StateResident);
}

// 按 token -> V 列处理的实验路径：将一个 V 通道的全部 K 个 state 读入小缓冲区，
// 完成两个归约、delta/y 和该列 state 更新，再写回并处理下一列。
// state 物理布局改为 [B,H,V,K]，沿 K 连续；与主路径 [B,H,K,V] 不兼容。
// 点积沿 K 做 SIMD 后仍含标量水平归约；K>=64 用 FP32
// 向量更新，否则用定点标量更新。 列复用不代表更快，本路径独立于新版 DDR/VTCM
// 主内核。
static inline void deltanetCoreQhpiRunQUint16StreamTile(
    uint16_t* output_data,
    uint16_t* state_data,
    const uint16_t* query_data,
    const uint16_t* key_data,
    const uint16_t* value_data,
    const uint16_t* decay_data,
    const uint16_t* beta_data,
    const uint16_t* recurrent_state_data,
    QHPI_Quant_Parameters query_q,
    QHPI_Quant_Parameters key_q,
    QHPI_Quant_Parameters value_q,
    QHPI_Quant_Parameters decay_q,
    QHPI_Quant_Parameters beta_q,
    QHPI_Quant_Parameters recurrent_state_q,
    QHPI_Quant_Parameters output_q,
    QHPI_Quant_Parameters state_q,
    const QHPI_Shape& decay_shape,
    const QHPI_Shape& beta_shape,
    size_t seq_len,
    size_t heads,
    size_t key_dim,
    size_t value_dim,
    size_t row,
    size_t v_begin,
    size_t v_end,
    uint16_t* state_column) {
  const size_t b = row / heads;
  const size_t h = row % heads;
  // Stream state is physically [B, H, V, K].  The logical state column
  // S[:, v] is one contiguous row in this layout.
  const size_t state_base = row * value_dim * key_dim;

  for (size_t t = 0; t < seq_len; ++t) {
    const size_t qkv_base = ((b * seq_len + t) * heads + h);
    const uint16_t* query_row = query_data + qkv_base * key_dim;
    const uint16_t* key_row = key_data + qkv_base * key_dim;
    const uint16_t* value_row = value_data + qkv_base * value_dim;
    int32_t key_center[kMaxDeltaNetDim];
    int32_t query_center[kMaxDeltaNetDim];
    int64_t key_query_acc = 0;
    for (size_t k = 0; k < key_dim; ++k) {
      key_center[k] = op_utils::centeredU16(key_row[k], key_q);
      query_center[k] = op_utils::centeredU16(query_row[k], query_q);
      key_query_acc += static_cast<int64_t>(key_center[k]) * query_center[k];
    }
    const float key_query_dot =
        static_cast<float>(key_query_acc) * key_q.stepsize * query_q.stepsize;
    const float decay = op_utils::dequantU16(
        decay_data[op_utils::timeHeadIndex(decay_shape, b, t, h)], decay_q);
    const float beta_value = op_utils::dequantU16(
        beta_data[op_utils::timeHeadIndex(beta_shape, b, t, h)], beta_q);
    const QHPI_Quant_Parameters prev_q = t == 0 ? recurrent_state_q : state_q;
    const float kv_scale = decay * prev_q.stepsize * key_q.stepsize;
    const float output_scale =
        decay * prev_q.stepsize * query_q.stepsize / output_q.stepsize;
    const float delta_output_scale = key_query_dot / output_q.stepsize;
    const float delta_update_scale = key_q.stepsize / state_q.stepsize;
    const int64_t prev_update_fixed = op_utils::roundFixed(
        decay * prev_q.stepsize / state_q.stepsize, kQUint16StateUpdateShift);

    for (size_t v = v_begin; v < v_end; ++v) {
      const size_t state_row = state_base + v * key_dim;
      const uint16_t* previous_row =
          t == 0 ? recurrent_state_data + state_row : state_data + state_row;
      std::memcpy(state_column, previous_row, key_dim * sizeof(uint16_t));
      int64_t kv_acc = 0;
      int64_t output_pre_acc = 0;
      deltanetQhpiHvxDotCenteredU16Pair(
          state_column,
          prev_q,
          key_row,
          key_q,
          query_row,
          query_q,
          key_dim,
          &kv_acc,
          &output_pre_acc);
      const float value_real =
          static_cast<float>(op_utils::centeredU16(value_row[v], value_q)) *
          value_q.stepsize;
      const float delta_value =
          (value_real - static_cast<float>(kv_acc) * kv_scale) * beta_value;
      output_data[qkv_base * value_dim + v] = op_utils::quantU16FromScaledFloat(
          static_cast<float>(output_q.zero_offset) +
          static_cast<float>(output_pre_acc) * output_scale +
          delta_value * delta_output_scale);

      if (key_dim >= 128 / sizeof(uint16_t)) {
        deltanetQhpiHvxUpdateStateStreamU16F32(
            state_column,
            state_column,
            key_row,
            prev_q,
            key_q,
            state_q,
            decay * prev_q.stepsize / state_q.stepsize,
            delta_value * key_q.stepsize / state_q.stepsize,
            key_dim);
      } else {
        const int64_t delta_update_fixed = op_utils::roundFixed(
            delta_value * delta_update_scale, kQUint16StateUpdateShift);
        for (size_t k = 0; k < key_dim; ++k) {
          const int64_t previous =
              static_cast<int64_t>(state_column[k]) - prev_q.zero_offset;
          const int64_t update = prev_update_fixed * previous +
              static_cast<int64_t>(key_center[k]) * delta_update_fixed;
          const int64_t state_center = update >= 0
              ? (update + (INT64_C(1) << (kQUint16StateUpdateShift - 1))) >>
                  kQUint16StateUpdateShift
              : -(((-update) +
                   (INT64_C(1) << (kQUint16StateUpdateShift - 1))) >>
                  kQUint16StateUpdateShift);
          state_column[k] = op_utils::clampCodeToU16(
              static_cast<int64_t>(state_q.zero_offset) + state_center);
        }
      }
      std::memcpy(
          state_data + state_row, state_column, key_dim * sizeof(uint16_t));
    }
  }
}

// Stream 路径的 worker 调度层，每个 worker 只分配一个对齐的 U16 state
// 列缓冲区。 按 slice 交错分配 (batch,head)，每个 head 的全部 token 和 V
// 列由同一个 worker 处理。
static void deltanetCoreQhpiRunQUint16Stream(
    uint16_t* output_data,
    uint16_t* state_data,
    const uint16_t* query_data,
    const uint16_t* key_data,
    const uint16_t* value_data,
    const uint16_t* decay_data,
    const uint16_t* beta_data,
    const uint16_t* recurrent_state_data,
    QHPI_Quant_Parameters query_q,
    QHPI_Quant_Parameters key_q,
    QHPI_Quant_Parameters value_q,
    QHPI_Quant_Parameters decay_q,
    QHPI_Quant_Parameters beta_q,
    QHPI_Quant_Parameters recurrent_state_q,
    QHPI_Quant_Parameters output_q,
    QHPI_Quant_Parameters state_q,
    const QHPI_Shape& decay_shape,
    const QHPI_Shape& beta_shape,
    size_t batch,
    size_t seq_len,
    size_t heads,
    size_t key_dim,
    size_t value_dim,
    size_t slice,
    size_t num_slices) {
  alignas(128) uint16_t state_column[kMaxDeltaNetDim];
  const size_t total_rows = batch * heads;
  for (size_t row = slice; row < total_rows; row += num_slices) {
    deltanetCoreQhpiRunQUint16StreamTile(
        output_data,
        state_data,
        query_data,
        key_data,
        value_data,
        decay_data,
        beta_data,
        recurrent_state_data,
        query_q,
        key_q,
        value_q,
        decay_q,
        beta_q,
        recurrent_state_q,
        output_q,
        state_q,
        decay_shape,
        beta_shape,
        seq_len,
        heads,
        key_dim,
        value_dim,
        row,
        0,
        value_dim,
        state_column);
  }
}

// QHPI 的普通 FP16 kernel 入口：检查输入/输出数及主要维度，提取数据指针，
// 取得 worker slice 并决定 V 分块数，转交 RunFp16；内部投影/累加使用 FP32。
