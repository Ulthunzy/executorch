// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "../utils/DeltaNetCommon.h"

static inline void deltanetQhpiHvxAccumF16(
    const Float16* prev_state,
    float* kv_mem,
    float* output_pre,
    float key_scale,
    float query_scale,
    size_t value_dim) {
  const size_t halfs_per_vec = 128 / sizeof(Float16);
  const size_t floats_per_vec = 128 / sizeof(float);
  const size_t vec_count = value_dim / halfs_per_vec;
  const HVX_Vector v_one = Q6_Vh_vsplat_R(0x3C00);
  const HVX_Vector v_key = op_utils::hvxSplatF32(key_scale);
  const HVX_Vector v_query = op_utils::hvxSplatF32(query_scale);

  for (size_t i = 0; i < vec_count; ++i) {
    const size_t offset = i * halfs_per_vec;
    const HVX_Vector v_half = vmemu(prev_state + offset);
    const HVX_VectorPair v_prev_q = Q6_Wqf32_vmpy_VhfVhf(v_half, v_one);
    const HVX_Vector v_prev_lo = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(v_prev_q));
    const HVX_Vector v_prev_hi = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(v_prev_q));

    const HVX_Vector v_kv_lo = op_utils::hvxAddF32(
        vmemu(kv_mem + offset), op_utils::hvxMulF32(v_prev_lo, v_key));
    const HVX_Vector v_kv_hi = op_utils::hvxAddF32(
        vmemu(kv_mem + offset + floats_per_vec),
        op_utils::hvxMulF32(v_prev_hi, v_key));
    const HVX_Vector v_out_lo = op_utils::hvxAddF32(
        vmemu(output_pre + offset), op_utils::hvxMulF32(v_prev_lo, v_query));
    const HVX_Vector v_out_hi = op_utils::hvxAddF32(
        vmemu(output_pre + offset + floats_per_vec),
        op_utils::hvxMulF32(v_prev_hi, v_query));

    vmemu(kv_mem + offset) = v_kv_lo;
    vmemu(kv_mem + offset + floats_per_vec) = v_kv_hi;
    vmemu(output_pre + offset) = v_out_lo;
    vmemu(output_pre + offset + floats_per_vec) = v_out_hi;
  }

  for (size_t v = vec_count * halfs_per_vec; v < value_dim; ++v) {
    const float prev = static_cast<float>(prev_state[v]);
    kv_mem[v] += prev * key_scale;
    output_pre[v] += prev * query_scale;
  }
}

// 用 k=0 的 FP16 state 行直接初始化 FP32 kv_mem/output_pre，避免清零和读取旧
// scratch。 传入系数已包含 decay，后续 K 行由 AccumF16 累加。
static inline void deltanetQhpiHvxAssignAccumF16(
    const Float16* prev_state,
    float* kv_mem,
    float* output_pre,
    float key_scale,
    float query_scale,
    size_t value_dim) {
  const size_t halfs_per_vec = 128 / sizeof(Float16);
  const size_t floats_per_vec = 128 / sizeof(float);
  const size_t vec_count = value_dim / halfs_per_vec;
  const HVX_Vector v_one = Q6_Vh_vsplat_R(0x3C00);
  const HVX_Vector v_key = op_utils::hvxSplatF32(key_scale);
  const HVX_Vector v_query = op_utils::hvxSplatF32(query_scale);

  for (size_t i = 0; i < vec_count; ++i) {
    const size_t offset = i * halfs_per_vec;
    const HVX_Vector v_half = vmemu(prev_state + offset);
    const HVX_VectorPair v_prev_q = Q6_Wqf32_vmpy_VhfVhf(v_half, v_one);
    const HVX_Vector v_prev_lo = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(v_prev_q));
    const HVX_Vector v_prev_hi = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(v_prev_q));

    vmemu(kv_mem + offset) = op_utils::hvxMulF32(v_prev_lo, v_key);
    vmemu(kv_mem + offset + floats_per_vec) =
        op_utils::hvxMulF32(v_prev_hi, v_key);
    vmemu(output_pre + offset) = op_utils::hvxMulF32(v_prev_lo, v_query);
    vmemu(output_pre + offset + floats_per_vec) =
        op_utils::hvxMulF32(v_prev_hi, v_query);
  }

  for (size_t v = vec_count * halfs_per_vec; v < value_dim; ++v) {
    const float prev = static_cast<float>(prev_state[v]);
    kv_mem[v] = prev * key_scale;
    output_pre[v] = prev * query_scale;
  }
}

// FP16 state、系数和 scratch 的纯半精度投影累加，每次处理 64 个 V。
// 乘法和加法分别舍入到 FP16，避免与普通 AccumF16 的 FP32 累加混淆。
static inline void deltanetQhpiHvxAccumF16AccumF16(
    const Float16* prev_state,
    Float16* kv_mem,
    Float16* output_pre,
    Float16 key_scale,
    Float16 query_scale,
    size_t value_dim) {
  const size_t halfs_per_vec = 128 / sizeof(Float16);
  const size_t vec_count = value_dim / halfs_per_vec;
  const HVX_Vector v_key = op_utils::hvxSplatF16(key_scale);
  const HVX_Vector v_query = op_utils::hvxSplatF16(query_scale);

  for (size_t i = 0; i < vec_count; ++i) {
    const size_t offset = i * halfs_per_vec;
    const HVX_Vector v_prev = vmemu(prev_state + offset);
    vmemu(kv_mem + offset) = op_utils::hvxAddF16(
        vmemu(kv_mem + offset), op_utils::hvxMulF16(v_prev, v_key));
    vmemu(output_pre + offset) = op_utils::hvxAddF16(
        vmemu(output_pre + offset), op_utils::hvxMulF16(v_prev, v_query));
  }

  for (size_t v = vec_count * halfs_per_vec; v < value_dim; ++v) {
    kv_mem[v] =
        op_utils::addF16(kv_mem[v], op_utils::mulF16(prev_state[v], key_scale));
    output_pre[v] = op_utils::addF16(
        output_pre[v], op_utils::mulF16(prev_state[v], query_scale));
  }
}

// 用 k=0 的 state 行初始化 FP16 投影 scratch，乘法结果即舍入到半精度。
// 后续 K 行由 AccumF16AccumF16 继续累加。
static inline void deltanetQhpiHvxAssignAccumF16AccumF16(
    const Float16* prev_state,
    Float16* kv_mem,
    Float16* output_pre,
    Float16 key_scale,
    Float16 query_scale,
    size_t value_dim) {
  const size_t halfs_per_vec = 128 / sizeof(Float16);
  const size_t vec_count = value_dim / halfs_per_vec;
  const HVX_Vector v_key = op_utils::hvxSplatF16(key_scale);
  const HVX_Vector v_query = op_utils::hvxSplatF16(query_scale);

  for (size_t i = 0; i < vec_count; ++i) {
    const size_t offset = i * halfs_per_vec;
    const HVX_Vector v_prev = vmemu(prev_state + offset);
    vmemu(kv_mem + offset) = op_utils::hvxMulF16(v_prev, v_key);
    vmemu(output_pre + offset) = op_utils::hvxMulF16(v_prev, v_query);
  }

  for (size_t v = vec_count * halfs_per_vec; v < value_dim; ++v) {
    kv_mem[v] = op_utils::mulF16(prev_state[v], key_scale);
    output_pre[v] = op_utils::mulF16(prev_state[v], query_scale);
  }
}

// 读取一行 FP32 state，同时累加 decay*K^T*S 和 decay*Q^T*S 的行贡献。
// state、系数和 scratch 均为 FP32，HVX 每次处理 32 个 V，尾部标量计算。
static inline void deltanetQhpiHvxAccumF32(
    const float* prev_state,
    float* kv_mem,
    float* output_pre,
    float key_scale,
    float query_scale,
    size_t value_dim) {
  const size_t epv = 128 / sizeof(float);
  const size_t vec_count = value_dim / epv;
  const HVX_Vector v_key = op_utils::hvxSplatF32(key_scale);
  const HVX_Vector v_query = op_utils::hvxSplatF32(query_scale);

  for (size_t i = 0; i < vec_count; ++i) {
    const size_t offset = i * epv;
    const HVX_Vector v_prev = vmemu(prev_state + offset);
    vmemu(kv_mem + offset) = op_utils::hvxAddF32(
        vmemu(kv_mem + offset), op_utils::hvxMulF32(v_prev, v_key));
    vmemu(output_pre + offset) = op_utils::hvxAddF32(
        vmemu(output_pre + offset), op_utils::hvxMulF32(v_prev, v_query));
  }

  for (size_t v = vec_count * epv; v < value_dim; ++v) {
    const float prev = prev_state[v];
    kv_mem[v] += prev * key_scale;
    output_pre[v] += prev * query_scale;
  }
}

// 以 k=0 的 FP32 state 行初始化两个投影 scratch，不读取旧累加值。
// 系数由调用方预先乘 decay，后续 K 行用 AccumF32 累加。
static inline void deltanetQhpiHvxAssignAccumF32(
    const float* prev_state,
    float* kv_mem,
    float* output_pre,
    float key_scale,
    float query_scale,
    size_t value_dim) {
  const size_t epv = 128 / sizeof(float);
  const size_t vec_count = value_dim / epv;
  const HVX_Vector v_key = op_utils::hvxSplatF32(key_scale);
  const HVX_Vector v_query = op_utils::hvxSplatF32(query_scale);

  for (size_t i = 0; i < vec_count; ++i) {
    const size_t offset = i * epv;
    const HVX_Vector v_prev = vmemu(prev_state + offset);
    vmemu(kv_mem + offset) = op_utils::hvxMulF32(v_prev, v_key);
    vmemu(output_pre + offset) = op_utils::hvxMulF32(v_prev, v_query);
  }

  for (size_t v = vec_count * epv; v < value_dim; ++v) {
    const float prev = prev_state[v];
    kv_mem[v] = prev * key_scale;
    output_pre[v] = prev * query_scale;
  }
}

// 更新一个 K 行：S_new[k,v]=decay*S_old[k,v]+key[k]*delta[v]。
// FP32 读取、计算和写回，HVX 每次处理 32 个 V，并处理标量尾部。
static inline void deltanetQhpiHvxUpdateStateF32(
    const float* prev_state,
    float* state,
    const float* delta,
    float decay,
    float key_value,
    size_t value_dim) {
  const size_t epv = 128 / sizeof(float);
  const size_t vec_count = value_dim / epv;
  const HVX_Vector v_decay = op_utils::hvxSplatF32(decay);
  const HVX_Vector v_key = op_utils::hvxSplatF32(key_value);

  for (size_t i = 0; i < vec_count; ++i) {
    const size_t offset = i * epv;
    const HVX_Vector v_updated = op_utils::hvxAddF32(
        op_utils::hvxMulF32(vmemu(prev_state + offset), v_decay),
        op_utils::hvxMulF32(vmemu(delta + offset), v_key));
    vmemu(state + offset) = v_updated;
  }

  for (size_t v = vec_count * epv; v < value_dim; ++v) {
    state[v] = prev_state[v] * decay + delta[v] * key_value;
  }
}

// 在 FP32 路径一遍生成 delta 并写 y；kv_mem/output_pre 已包含 decay。
// delta=beta*(value-kv_mem)，y=output_pre+(Q·K)*delta。
// delta 保存供 state 更新使用，因此输出阶段无需读取更新后的 state。
static inline void deltanetQhpiHvxDeltaAndWriteOutputF32(
    float* output,
    const float* value,
    const float* kv_mem,
    const float* output_pre,
    float* delta,
    float beta_value,
    float key_query_dot,
    size_t value_dim) {
  const size_t epv = 128 / sizeof(float);
  const size_t vec_count = value_dim / epv;
  const HVX_Vector v_neg_one = Q6_V_vsplat_R(0xBF800000);
  const HVX_Vector v_beta = op_utils::hvxSplatF32(beta_value);
  const HVX_Vector v_dot = op_utils::hvxSplatF32(key_query_dot);

  for (size_t i = 0; i < vec_count; ++i) {
    const size_t offset = i * epv;
    const HVX_Vector v_delta = op_utils::hvxMulF32(
        op_utils::hvxAddF32(
            vmemu(value + offset),
            op_utils::hvxMulF32(vmemu(kv_mem + offset), v_neg_one)),
        v_beta);
    const HVX_Vector v_out = op_utils::hvxAddF32(
        vmemu(output_pre + offset), op_utils::hvxMulF32(v_delta, v_dot));
    vmemu(delta + offset) = v_delta;
    vmemu(output + offset) = v_out;
  }

  for (size_t v = vec_count * epv; v < value_dim; ++v) {
    const float delta_value = (value[v] - kv_mem[v]) * beta_value;
    delta[v] = delta_value;
    output[v] = output_pre[v] + delta_value * key_query_dot;
  }
}

// 纯 FP16 路径逐行更新 state：decay*S_old+key*delta。
// 两次乘法和随后加法各自舍入为 FP16，每次处理 64 个 V，并处理标量尾部。
static inline void deltanetQhpiHvxUpdateStateF16AccumF16(
    const Float16* prev_state,
    Float16* state,
    const Float16* delta,
    Float16 decay,
    Float16 key_value,
    size_t value_dim) {
  const size_t halfs_per_vec = 128 / sizeof(Float16);
  const size_t vec_count = value_dim / halfs_per_vec;
  const HVX_Vector v_decay = op_utils::hvxSplatF16(decay);
  const HVX_Vector v_key = op_utils::hvxSplatF16(key_value);

  for (size_t i = 0; i < vec_count; ++i) {
    const size_t offset = i * halfs_per_vec;
    const HVX_Vector v_prev = vmemu(prev_state + offset);
    const HVX_Vector v_delta = vmemu(delta + offset);
    const HVX_Vector v_updated = op_utils::hvxAddF16(
        op_utils::hvxMulF16(v_prev, v_decay),
        op_utils::hvxMulF16(v_delta, v_key));
    vmemu(state + offset) = v_updated;
  }

  for (size_t v = vec_count * halfs_per_vec; v < value_dim; ++v) {
    state[v] = op_utils::addF16(
        op_utils::mulF16(prev_state[v], decay),
        op_utils::mulF16(delta[v], key_value));
  }
}

// 以 FP16 投影结果生成 FP16 delta 和输出；kv_mem/output_pre 已包含 decay。
// delta=beta*(value-kv_mem)，y=output_pre+(Q·K)*delta，各次运算分别舍入为半精度。
static inline void deltanetQhpiHvxDeltaAndWriteOutputF16AccumF16(
    Float16* output,
    const Float16* value,
    const Float16* kv_mem,
    const Float16* output_pre,
    Float16* delta,
    Float16 beta_value,
    Float16 key_query_dot,
    size_t value_dim) {
  const size_t halfs_per_vec = 128 / sizeof(Float16);
  const size_t vec_count = value_dim / halfs_per_vec;
  const HVX_Vector v_beta = op_utils::hvxSplatF16(beta_value);
  const HVX_Vector v_dot = op_utils::hvxSplatF16(key_query_dot);

  for (size_t i = 0; i < vec_count; ++i) {
    const size_t offset = i * halfs_per_vec;
    const HVX_Vector v_delta = op_utils::hvxMulF16(
        op_utils::hvxSubF16(vmemu(value + offset), vmemu(kv_mem + offset)),
        v_beta);
    const HVX_Vector v_out = op_utils::hvxAddF16(
        vmemu(output_pre + offset), op_utils::hvxMulF16(v_delta, v_dot));
    vmemu(delta + offset) = v_delta;
    vmemu(output + offset) = v_out;
  }

  for (size_t v = vec_count * halfs_per_vec; v < value_dim; ++v) {
    const Float16 delta_value =
        op_utils::mulF16(op_utils::subF16(value[v], kv_mem[v]), beta_value);
    delta[v] = delta_value;
    output[v] = op_utils::addF16(
        output_pre[v], op_utils::mulF16(delta_value, key_query_dot));
  }
}

// 普通 FP16 路径的逐行更新：state 从 FP16 展开，使用 FP32 系数/delta 计算。
// S_new=decay*S_old+key*delta，写回时才舍入成 FP16；每次处理 64 个 V。
// 与 UpdateStateF16AccumF16 的逐次半精度舍入不同。
static inline void deltanetQhpiHvxUpdateStateF16(
    const Float16* prev_state,
    Float16* state,
    const float* delta,
    float decay,
    float key_value,
    size_t value_dim) {
  const size_t halfs_per_vec = 128 / sizeof(Float16);
  const size_t floats_per_vec = 128 / sizeof(float);
  const size_t vec_count = value_dim / halfs_per_vec;
  const HVX_Vector v_one = Q6_Vh_vsplat_R(0x3C00);
  const HVX_Vector v_decay = op_utils::hvxSplatF32(decay);
  const HVX_Vector v_key = op_utils::hvxSplatF32(key_value);

  for (size_t i = 0; i < vec_count; ++i) {
    const size_t offset = i * halfs_per_vec;
    const HVX_Vector v_half = vmemu(prev_state + offset);
    const HVX_VectorPair v_prev_q = Q6_Wqf32_vmpy_VhfVhf(v_half, v_one);
    const HVX_Vector v_prev_lo = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(v_prev_q));
    const HVX_Vector v_prev_hi = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(v_prev_q));
    const HVX_Vector v_delta_lo = vmemu(delta + offset);
    const HVX_Vector v_delta_hi = vmemu(delta + offset + floats_per_vec);
    const HVX_Vector v_updated_lo = op_utils::hvxAddF32(
        op_utils::hvxMulF32(v_prev_lo, v_decay),
        op_utils::hvxMulF32(v_delta_lo, v_key));
    const HVX_Vector v_updated_hi = op_utils::hvxAddF32(
        op_utils::hvxMulF32(v_prev_hi, v_decay),
        op_utils::hvxMulF32(v_delta_hi, v_key));
    vmemu(state + offset) = Q6_Vhf_equals_Wqf32(
        op_utils::hvxF32PairToQF32(v_updated_lo, v_updated_hi));
  }

  for (size_t v = vec_count * halfs_per_vec; v < value_dim; ++v) {
    state[v] = Float16(
        static_cast<float>(prev_state[v]) * decay + delta[v] * key_value);
  }
}

// 从 FP16 value 和 FP32 投影 scratch 计算 delta/y，投影已包含 decay。
// delta 保持 FP32 供 state 更新，输出 y 则打包为 FP16；每次处理 64 个 V。
static inline void deltanetQhpiHvxDeltaAndWriteOutputF16(
    Float16* output,
    const Float16* value,
    const float* kv_mem,
    const float* output_pre,
    float* delta,
    float beta_value,
    float key_query_dot,
    size_t value_dim) {
  const size_t halfs_per_vec = 128 / sizeof(Float16);
  const size_t floats_per_vec = 128 / sizeof(float);
  const size_t vec_count = value_dim / halfs_per_vec;
  const HVX_Vector v_one = Q6_Vh_vsplat_R(0x3C00);
  const HVX_Vector v_neg_one = Q6_V_vsplat_R(0xBF800000);
  const HVX_Vector v_beta = op_utils::hvxSplatF32(beta_value);
  const HVX_Vector v_dot = op_utils::hvxSplatF32(key_query_dot);

  for (size_t i = 0; i < vec_count; ++i) {
    const size_t offset = i * halfs_per_vec;
    const HVX_Vector v_half = vmemu(value + offset);
    const HVX_VectorPair v_value_q = Q6_Wqf32_vmpy_VhfVhf(v_half, v_one);
    const HVX_Vector v_value_lo = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(v_value_q));
    const HVX_Vector v_value_hi = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(v_value_q));
    const HVX_Vector v_delta_lo = op_utils::hvxMulF32(
        op_utils::hvxAddF32(
            v_value_lo, op_utils::hvxMulF32(vmemu(kv_mem + offset), v_neg_one)),
        v_beta);
    const HVX_Vector v_delta_hi = op_utils::hvxMulF32(
        op_utils::hvxAddF32(
            v_value_hi,
            op_utils::hvxMulF32(
                vmemu(kv_mem + offset + floats_per_vec), v_neg_one)),
        v_beta);
    const HVX_Vector v_out_lo = op_utils::hvxAddF32(
        vmemu(output_pre + offset), op_utils::hvxMulF32(v_delta_lo, v_dot));
    const HVX_Vector v_out_hi = op_utils::hvxAddF32(
        vmemu(output_pre + offset + floats_per_vec),
        op_utils::hvxMulF32(v_delta_hi, v_dot));
    vmemu(delta + offset) = v_delta_lo;
    vmemu(delta + offset + floats_per_vec) = v_delta_hi;
    vmemu(output + offset) =
        Q6_Vhf_equals_Wqf32(op_utils::hvxF32PairToQF32(v_out_lo, v_out_hi));
  }

  for (size_t v = vec_count * halfs_per_vec; v < value_dim; ++v) {
    const float delta_value =
        (static_cast<float>(value[v]) - kv_mem[v]) * beta_value;
    delta[v] = delta_value;
    output[v] = Float16(output_pre[v] + delta_value * key_query_dot);
  }
}

template <bool Aligned, size_t Vectors = 1, typename Coefficient = Float16>
static inline void deltanetQhpiHvxReduceStateF16(
    const Float16* previous,
    float* kv,
    float* out,
    const Coefficient* key,
    const Coefficient* query,
    float decay,
    size_t key_dim,
    size_t stride,
    size_t tile_size) {
  constexpr size_t halfs = 128 / sizeof(Float16);
  constexpr size_t words = 128 / sizeof(float);
  const HVX_Vector one = Q6_Vh_vsplat_R(0x3C00);
  for (size_t offset = 0; offset < tile_size; offset += halfs * Vectors) {
    HVX_Vector kv_lo[Vectors], kv_hi[Vectors], out_lo[Vectors], out_hi[Vectors];
    for (size_t k = 0; k < key_dim; ++k) {
      const HVX_Vector key_scale =
          op_utils::hvxSplatF32(decay * static_cast<float>(key[k]));
      const HVX_Vector query_scale =
          op_utils::hvxSplatF32(decay * static_cast<float>(query[k]));
      for (size_t v = 0; v < Vectors; ++v) {
        const size_t at = k * stride + offset + v * halfs;
        const HVX_Vector raw = Aligned
            ? *reinterpret_cast<const HVX_Vector*>(previous + at)
            : vmemu(previous + at);
        HVX_Vector lo, hi;
        if constexpr (std::is_same_v<Coefficient, float>) {
          const HVX_VectorPair expanded = Q6_Wsf_vcvt_Vhf(raw);
          lo = Q6_V_lo_W(expanded);
          hi = Q6_V_hi_W(expanded);
        } else {
          const HVX_VectorPair expanded = Q6_Wqf32_vmpy_VhfVhf(raw, one);
          lo = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(expanded));
          hi = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(expanded));
        }
        const HVX_Vector kl = op_utils::hvxMulF32(lo, key_scale);
        const HVX_Vector kh = op_utils::hvxMulF32(hi, key_scale);
        const HVX_Vector ql = op_utils::hvxMulF32(lo, query_scale);
        const HVX_Vector qh = op_utils::hvxMulF32(hi, query_scale);
        if (k == 0) {
          kv_lo[v] = kl;
          kv_hi[v] = kh;
          out_lo[v] = ql;
          out_hi[v] = qh;
        } else {
          kv_lo[v] = op_utils::hvxAddF32(kv_lo[v], kl);
          kv_hi[v] = op_utils::hvxAddF32(kv_hi[v], kh);
          out_lo[v] = op_utils::hvxAddF32(out_lo[v], ql);
          out_hi[v] = op_utils::hvxAddF32(out_hi[v], qh);
        }
      }
    }
    for (size_t v = 0; v < Vectors; ++v) {
      const size_t at = offset + v * halfs;
      vmemu(kv + at) = kv_lo[v];
      vmemu(kv + at + words) = kv_hi[v];
      vmemu(out + at) = out_lo[v];
      vmemu(out + at + words) = out_hi[v];
    }
  }
}

template <bool Aligned, size_t Vectors = 1, typename Coefficient = Float16>
static inline void deltanetQhpiHvxUpdateStateTileF16(
    const Float16* previous,
    Float16* state,
    const float* delta,
    const Coefficient* key,
    float decay,
    size_t key_dim,
    size_t stride,
    size_t tile_size) {
  constexpr size_t halfs = 128 / sizeof(Float16);
  constexpr size_t words = 128 / sizeof(float);
  const HVX_Vector one = Q6_Vh_vsplat_R(0x3C00);
  const HVX_Vector decay_scale = op_utils::hvxSplatF32(decay);
  for (size_t offset = 0; offset < tile_size; offset += halfs * Vectors) {
    HVX_Vector dl[Vectors], dh[Vectors];
    for (size_t v = 0; v < Vectors; ++v) {
      const size_t at = offset + v * halfs;
      dl[v] = vmemu(delta + at);
      dh[v] = vmemu(delta + at + words);
    }
    for (size_t k = 0; k < key_dim; ++k) {
      const HVX_Vector key_scale =
          op_utils::hvxSplatF32(static_cast<float>(key[k]));
      for (size_t v = 0; v < Vectors; ++v) {
        const size_t at = k * stride + offset + v * halfs;
        const HVX_Vector raw = Aligned
            ? *reinterpret_cast<const HVX_Vector*>(previous + at)
            : vmemu(previous + at);
        HVX_Vector lo, hi;
        if constexpr (std::is_same_v<Coefficient, float>) {
          const HVX_VectorPair expanded = Q6_Wsf_vcvt_Vhf(raw);
          lo = Q6_V_lo_W(expanded);
          hi = Q6_V_hi_W(expanded);
        } else {
          const HVX_VectorPair expanded = Q6_Wqf32_vmpy_VhfVhf(raw, one);
          lo = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(expanded));
          hi = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(expanded));
        }
        const HVX_Vector updated_lo = op_utils::hvxAddF32(
            op_utils::hvxMulF32(lo, decay_scale),
            op_utils::hvxMulF32(dl[v], key_scale));
        const HVX_Vector updated_hi = op_utils::hvxAddF32(
            op_utils::hvxMulF32(hi, decay_scale),
            op_utils::hvxMulF32(dh[v], key_scale));
        // The mixed core uses IEEE nearest-even Half storage, without a
        // QF32 round trip that changes ties. Preserve the old FP16 operator.
        const HVX_Vector packed = std::is_same_v<Coefficient, float>
            ? Q6_Vhf_vcvt_VsfVsf(updated_lo, updated_hi)
            : Q6_Vhf_equals_Wqf32(
                  op_utils::hvxF32PairToQF32(updated_lo, updated_hi));
        if constexpr (Aligned)
          *reinterpret_cast<HVX_Vector*>(state + at) = packed;
        else
          vmemu(state + at) = packed;
      }
    }
  }
}

// 普通 FP16 recurrence：一个 (batch,head,V tile) 顺序执行全部 token。
// state 为 [B,H,K,V]，Q/K/value/y 为 [B,T,H,D]；投影、delta、Q·K 用 FP32。
// 每步先归约（系数已含 decay）、求 delta/y，再更新 FP16 state；
// t=0 读取初始 state，后续 token 在输出 state 上原位递推。
template <bool ValidLength = false>
static inline void deltanetCoreQhpiRunFp16Tile(
    Float16* output_data,
    Float16* state_data,
    const Float16* query_data,
    const Float16* key_data,
    const Float16* value_data,
    const Float16* decay_data,
    const Float16* beta_data,
    const Float16* recurrent_state_data,
    size_t seq_len,
    size_t heads,
    size_t key_dim,
    size_t value_dim,
    size_t row,
    size_t v_begin,
    size_t v_end,
    float* kv_mem,
    float* delta,
    float* output_pre,
    size_t valid_tokens = 0) {
  const size_t tile_size = v_end - v_begin;
  if (tile_size == 0) {
    return;
  }

  const size_t b = row / heads;
  const size_t h = row % heads;
  const size_t state_base = row * key_dim * value_dim;

  for (size_t t = 0; t < (ValidLength ? valid_tokens : seq_len); ++t) {
    const size_t qkv_base = ((b * seq_len + t) * heads + h);
    const Float16* query_row = query_data + qkv_base * key_dim;
    const Float16* key_row = key_data + qkv_base * key_dim;
    const Float16* value_tile = value_data + qkv_base * value_dim + v_begin;

    float key_query_dot = 0.0f;
    for (size_t k = 0; k < key_dim; ++k) {
      key_query_dot +=
          static_cast<float>(key_row[k]) * static_cast<float>(query_row[k]);
    }

    const float decay =
        static_cast<float>(decay_data[(b * seq_len + t) * heads + h]);
    const float beta_value =
        static_cast<float>(beta_data[(b * seq_len + t) * heads + h]);

#if DELTANET_FP16_REGISTER_REUSE
    const Float16* previous =
        (t == 0 ? recurrent_state_data : state_data) + state_base + v_begin;
    const bool aligned =
        reinterpret_cast<uintptr_t>(previous) % 128 == 0 && value_dim % 64 == 0;
    if (tile_size % 64 == 0) {
      if (aligned && tile_size % 128 == 0) {
        deltanetQhpiHvxReduceStateF16<true, 2>(
            previous,
            kv_mem,
            output_pre,
            key_row,
            query_row,
            decay,
            key_dim,
            value_dim,
            tile_size);
      } else {
        deltanetQhpiHvxReduceStateF16<false>(
            previous,
            kv_mem,
            output_pre,
            key_row,
            query_row,
            decay,
            key_dim,
            value_dim,
            tile_size);
      }
    } else
#endif
    {
      {
        const size_t row_base = state_base + v_begin;
        const Float16* prev_state =
            t == 0 ? recurrent_state_data + row_base : state_data + row_base;
        deltanetQhpiHvxAssignAccumF16(
            prev_state,
            kv_mem,
            output_pre,
            decay * static_cast<float>(key_row[0]),
            decay * static_cast<float>(query_row[0]),
            tile_size);
      }
      for (size_t k = 1; k < key_dim; ++k) {
        const size_t row_base = state_base + k * value_dim + v_begin;
        const Float16* prev_state =
            t == 0 ? recurrent_state_data + row_base : state_data + row_base;
        deltanetQhpiHvxAccumF16(
            prev_state,
            kv_mem,
            output_pre,
            decay * static_cast<float>(key_row[k]),
            decay * static_cast<float>(query_row[k]),
            tile_size);
      }
    }

    deltanetQhpiHvxDeltaAndWriteOutputF16(
        output_data + qkv_base * value_dim + v_begin,
        value_tile,
        kv_mem,
        output_pre,
        delta,
        beta_value,
        key_query_dot,
        tile_size);

#if DELTANET_FP16_REGISTER_REUSE
    if (tile_size % 64 == 0) {
      if (aligned &&
          reinterpret_cast<uintptr_t>(state_data + state_base + v_begin) %
                  128 ==
              0 &&
          tile_size % 128 == 0) {
        deltanetQhpiHvxUpdateStateTileF16<true, 2>(
            previous,
            state_data + state_base + v_begin,
            delta,
            key_row,
            decay,
            key_dim,
            value_dim,
            tile_size);
      } else {
        deltanetQhpiHvxUpdateStateTileF16<false>(
            previous,
            state_data + state_base + v_begin,
            delta,
            key_row,
            decay,
            key_dim,
            value_dim,
            tile_size);
      }
    } else
#endif
    {
      for (size_t k = 0; k < key_dim; ++k) {
        const size_t row_base = state_base + k * value_dim + v_begin;
        const Float16* prev_state =
            t == 0 ? recurrent_state_data + row_base : state_data + row_base;
        deltanetQhpiHvxUpdateStateF16(
            prev_state,
            state_data + row_base,
            delta,
            decay,
            static_cast<float>(key_row[k]),
            tile_size);
      }
    }
  }
}

// 普通 FP16 路径的 worker 调度层，为每个 worker 分配三个对齐的 FP32 scratch。
// 按 slice 交错分配 (batch,head)，必要时再拆独立的 V tile；
// 每个任务内部 token 顺序执行，worker 之间处理互不重叠的 state 区域。
template <bool ValidLength = false>
static void deltanetCoreQhpiRunFp16(
    Float16* output_data,
    Float16* state_data,
    const Float16* query_data,
    const Float16* key_data,
    const Float16* value_data,
    const Float16* decay_data,
    const Float16* beta_data,
    const Float16* recurrent_state_data,
    size_t batch,
    size_t seq_len,
    size_t heads,
    size_t key_dim,
    size_t value_dim,
    size_t slice,
    size_t num_slices,
    size_t value_tile_count,
    size_t valid_tokens = 0) {
  float kv_mem[kMaxDeltaNetDim] __attribute__((aligned(128)));
  float delta[kMaxDeltaNetDim] __attribute__((aligned(128)));
  float output_pre[kMaxDeltaNetDim] __attribute__((aligned(128)));

  const size_t total_rows = batch * heads;
  if (value_tile_count <= 1) {
    for (size_t row = slice; row < total_rows; row += num_slices) {
      deltanetCoreQhpiRunFp16Tile<ValidLength>(
          output_data,
          state_data,
          query_data,
          key_data,
          value_data,
          decay_data,
          beta_data,
          recurrent_state_data,
          seq_len,
          heads,
          key_dim,
          value_dim,
          row,
          0,
          value_dim,
          kv_mem,
          delta,
          output_pre,
          valid_tokens);
    }
    return;
  }

  const size_t total_tasks = total_rows * value_tile_count;
  for (size_t task = slice; task < total_tasks; task += num_slices) {
    const size_t row = task / value_tile_count;
    const size_t tile = task % value_tile_count;
    const size_t v_begin = value_dim * tile / value_tile_count;
    const size_t v_end = value_dim * (tile + 1) / value_tile_count;
    deltanetCoreQhpiRunFp16Tile<ValidLength>(
        output_data,
        state_data,
        query_data,
        key_data,
        value_data,
        decay_data,
        beta_data,
        recurrent_state_data,
        seq_len,
        heads,
        key_dim,
        value_dim,
        row,
        v_begin,
        v_end,
        kv_mem,
        delta,
        output_pre,
        valid_tokens);
  }
}

// 纯 FP16 recurrence：一个 (batch,head,V tile) 顺序处理全部 token。
// Q·K、投影、delta 和 state 更新均逐步舍入到 FP16，scratch 也为 FP16。
// state 布局与普通路径相同；首步读取初始 state，后续原位递推。
static inline void deltanetCoreQhpiRunFp16AccumF16Tile(
    Float16* output_data,
    Float16* state_data,
    const Float16* query_data,
    const Float16* key_data,
    const Float16* value_data,
    const Float16* decay_data,
    const Float16* beta_data,
    const Float16* recurrent_state_data,
    size_t seq_len,
    size_t heads,
    size_t key_dim,
    size_t value_dim,
    size_t row,
    size_t v_begin,
    size_t v_end,
    Float16* kv_mem,
    Float16* delta,
    Float16* output_pre) {
  const size_t tile_size = v_end - v_begin;
  if (tile_size == 0) {
    return;
  }

  const size_t b = row / heads;
  const size_t h = row % heads;
  const size_t state_base = row * key_dim * value_dim;

  for (size_t t = 0; t < seq_len; ++t) {
    const size_t qkv_base = ((b * seq_len + t) * heads + h);
    const Float16* query_row = query_data + qkv_base * key_dim;
    const Float16* key_row = key_data + qkv_base * key_dim;
    const Float16* value_tile = value_data + qkv_base * value_dim + v_begin;

    Float16 key_query_dot = Float16(0.0f);
    for (size_t k = 0; k < key_dim; ++k) {
      key_query_dot = op_utils::addF16(
          key_query_dot, op_utils::mulF16(key_row[k], query_row[k]));
    }

    const Float16 decay = decay_data[(b * seq_len + t) * heads + h];
    const Float16 beta_value = beta_data[(b * seq_len + t) * heads + h];

    {
      const size_t row_base = state_base + v_begin;
      const Float16* prev_state =
          t == 0 ? recurrent_state_data + row_base : state_data + row_base;
      deltanetQhpiHvxAssignAccumF16AccumF16(
          prev_state,
          kv_mem,
          output_pre,
          op_utils::mulF16(decay, key_row[0]),
          op_utils::mulF16(decay, query_row[0]),
          tile_size);
    }
    for (size_t k = 1; k < key_dim; ++k) {
      const size_t row_base = state_base + k * value_dim + v_begin;
      const Float16* prev_state =
          t == 0 ? recurrent_state_data + row_base : state_data + row_base;
      deltanetQhpiHvxAccumF16AccumF16(
          prev_state,
          kv_mem,
          output_pre,
          op_utils::mulF16(decay, key_row[k]),
          op_utils::mulF16(decay, query_row[k]),
          tile_size);
    }

    deltanetQhpiHvxDeltaAndWriteOutputF16AccumF16(
        output_data + qkv_base * value_dim + v_begin,
        value_tile,
        kv_mem,
        output_pre,
        delta,
        beta_value,
        key_query_dot,
        tile_size);

    for (size_t k = 0; k < key_dim; ++k) {
      const size_t row_base = state_base + k * value_dim + v_begin;
      const Float16* prev_state =
          t == 0 ? recurrent_state_data + row_base : state_data + row_base;
      deltanetQhpiHvxUpdateStateF16AccumF16(
          prev_state,
          state_data + row_base,
          delta,
          decay,
          key_row[k],
          tile_size);
    }
  }
}

// 纯 FP16 路径的 worker 调度层，使用三个对齐的 Float16 scratch。
// 按 slice 交错分配 head/V tile，并调用 RunFp16AccumF16Tile，避免共享 scratch。
static void deltanetCoreQhpiRunFp16AccumF16(
    Float16* output_data,
    Float16* state_data,
    const Float16* query_data,
    const Float16* key_data,
    const Float16* value_data,
    const Float16* decay_data,
    const Float16* beta_data,
    const Float16* recurrent_state_data,
    size_t batch,
    size_t seq_len,
    size_t heads,
    size_t key_dim,
    size_t value_dim,
    size_t slice,
    size_t num_slices,
    size_t value_tile_count) {
  Float16 kv_mem[kMaxDeltaNetDim] __attribute__((aligned(128)));
  Float16 delta[kMaxDeltaNetDim] __attribute__((aligned(128)));
  Float16 output_pre[kMaxDeltaNetDim] __attribute__((aligned(128)));

  const size_t total_rows = batch * heads;
  if (value_tile_count <= 1) {
    for (size_t row = slice; row < total_rows; row += num_slices) {
      deltanetCoreQhpiRunFp16AccumF16Tile(
          output_data,
          state_data,
          query_data,
          key_data,
          value_data,
          decay_data,
          beta_data,
          recurrent_state_data,
          seq_len,
          heads,
          key_dim,
          value_dim,
          row,
          0,
          value_dim,
          kv_mem,
          delta,
          output_pre);
    }
    return;
  }

  const size_t total_tasks = total_rows * value_tile_count;
  for (size_t task = slice; task < total_tasks; task += num_slices) {
    const size_t row = task / value_tile_count;
    const size_t tile = task % value_tile_count;
    const size_t v_begin = value_dim * tile / value_tile_count;
    const size_t v_end = value_dim * (tile + 1) / value_tile_count;
    deltanetCoreQhpiRunFp16AccumF16Tile(
        output_data,
        state_data,
        query_data,
        key_data,
        value_data,
        decay_data,
        beta_data,
        recurrent_state_data,
        seq_len,
        heads,
        key_dim,
        value_dim,
        row,
        v_begin,
        v_end,
        kv_mem,
        delta,
        output_pre);
  }
}

// FP32 recurrence：一个 (batch,head,V tile) 按 token
// 顺序执行归约、delta/y、state 更新。 输入、scratch 和输出均为
// FP32，投影系数已含 decay，state 为 [B,H,K,V]。 首步使用
// recurrent_state，后续在 state 输出缓冲区上原位递推。
static inline void deltanetCoreQhpiRunFp32Tile(
    float* output_data,
    float* state_data,
    const float* query_data,
    const float* key_data,
    const float* value_data,
    const float* decay_data,
    const float* beta_data,
    const float* recurrent_state_data,
    size_t seq_len,
    size_t heads,
    size_t key_dim,
    size_t value_dim,
    size_t row,
    size_t v_begin,
    size_t v_end,
    float* kv_mem,
    float* delta,
    float* output_pre) {
  const size_t tile_size = v_end - v_begin;
  if (tile_size == 0) {
    return;
  }

  const size_t b = row / heads;
  const size_t h = row % heads;
  const size_t state_base = row * key_dim * value_dim;

  for (size_t t = 0; t < seq_len; ++t) {
    const size_t qkv_base = ((b * seq_len + t) * heads + h);
    const float* query_row = query_data + qkv_base * key_dim;
    const float* key_row = key_data + qkv_base * key_dim;
    const float* value_tile = value_data + qkv_base * value_dim + v_begin;

    float key_query_dot = 0.0f;
    for (size_t k = 0; k < key_dim; ++k) {
      key_query_dot += key_row[k] * query_row[k];
    }

    const float decay = decay_data[(b * seq_len + t) * heads + h];
    const float beta_value = beta_data[(b * seq_len + t) * heads + h];

    {
      const size_t row_base = state_base + v_begin;
      const float* prev_state =
          t == 0 ? recurrent_state_data + row_base : state_data + row_base;
      deltanetQhpiHvxAssignAccumF32(
          prev_state,
          kv_mem,
          output_pre,
          decay * key_row[0],
          decay * query_row[0],
          tile_size);
    }
    for (size_t k = 1; k < key_dim; ++k) {
      const size_t row_base = state_base + k * value_dim + v_begin;
      const float* prev_state =
          t == 0 ? recurrent_state_data + row_base : state_data + row_base;
      deltanetQhpiHvxAccumF32(
          prev_state,
          kv_mem,
          output_pre,
          decay * key_row[k],
          decay * query_row[k],
          tile_size);
    }

    deltanetQhpiHvxDeltaAndWriteOutputF32(
        output_data + qkv_base * value_dim + v_begin,
        value_tile,
        kv_mem,
        output_pre,
        delta,
        beta_value,
        key_query_dot,
        tile_size);

    for (size_t k = 0; k < key_dim; ++k) {
      const size_t row_base = state_base + k * value_dim + v_begin;
      const float* prev_state =
          t == 0 ? recurrent_state_data + row_base : state_data + row_base;
      deltanetQhpiHvxUpdateStateF32(
          prev_state,
          state_data + row_base,
          delta,
          decay,
          key_row[k],
          tile_size);
    }
  }
}

// FP32 路径的 worker 调度层，分配三个对齐的 FP32 scratch。
// 按 slice 交错分配 head，或在允许时分配 head/V tile，调用对应 Tile 完成全部
// token。
static void deltanetCoreQhpiRunFp32(
    float* output_data,
    float* state_data,
    const float* query_data,
    const float* key_data,
    const float* value_data,
    const float* decay_data,
    const float* beta_data,
    const float* recurrent_state_data,
    size_t batch,
    size_t seq_len,
    size_t heads,
    size_t key_dim,
    size_t value_dim,
    size_t slice,
    size_t num_slices,
    size_t value_tile_count) {
  float kv_mem[kMaxDeltaNetDim] __attribute__((aligned(128)));
  float delta[kMaxDeltaNetDim] __attribute__((aligned(128)));
  float output_pre[kMaxDeltaNetDim] __attribute__((aligned(128)));

  const size_t total_rows = batch * heads;
  if (value_tile_count <= 1) {
    for (size_t row = slice; row < total_rows; row += num_slices) {
      deltanetCoreQhpiRunFp32Tile(
          output_data,
          state_data,
          query_data,
          key_data,
          value_data,
          decay_data,
          beta_data,
          recurrent_state_data,
          seq_len,
          heads,
          key_dim,
          value_dim,
          row,
          0,
          value_dim,
          kv_mem,
          delta,
          output_pre);
    }
    return;
  }

  const size_t total_tasks = total_rows * value_tile_count;
  for (size_t task = slice; task < total_tasks; task += num_slices) {
    const size_t row = task / value_tile_count;
    const size_t tile = task % value_tile_count;
    const size_t v_begin = value_dim * tile / value_tile_count;
    const size_t v_end = value_dim * (tile + 1) / value_tile_count;
    deltanetCoreQhpiRunFp32Tile(
        output_data,
        state_data,
        query_data,
        key_data,
        value_data,
        decay_data,
        beta_data,
        recurrent_state_data,
        seq_len,
        heads,
        key_dim,
        value_dim,
        row,
        v_begin,
        v_end,
        kv_mem,
        delta,
        output_pre);
  }
}
