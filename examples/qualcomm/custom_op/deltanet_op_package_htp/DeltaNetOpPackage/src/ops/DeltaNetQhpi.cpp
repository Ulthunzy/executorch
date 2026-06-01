//==============================================================================
// QHPI self-sliced DeltaNetCore kernel.
//==============================================================================

#if defined(__hexagon__) || !defined(ANDROID)

#include "HTP/core/float16.h"
#include "HTP/core/intrinsics.h"
#include "HTP/core/qhpi.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#ifndef __HVX_ARCH__
#define __HVX_ARCH__ 79
#endif

#ifndef THIS_PKG_NAME_STR
#define STRINGIZE_DETAIL(X) #X
#define STRINGIZE(X) STRINGIZE_DETAIL(X)
#define THIS_PKG_NAME_STR STRINGIZE(THIS_PKG_NAME)
#endif

static constexpr size_t kMaxDeltaNetDim = 256;
static constexpr size_t kValueTileMinDim = 256;
static constexpr size_t kQUint16HvxValuesPerVec = 128 / sizeof(uint16_t);
static constexpr int kQUint16HvxAccumInputShift = 4;
static constexpr int kQUint16HvxStateUpdateInputShift = 0;
static constexpr int kQUint16HvxStateUpdateCoeffShift = 7;
static constexpr int kQUint16StateUpdateShift = 20;
static constexpr bool kEnableQUint16HvxStateUpdate = true;
static constexpr bool kEnableQUint16HvxStateUpdateF32 = true;

static inline size_t deltanetQhpiDivUp(size_t n, size_t d) {
  return (n + d - 1) / d;
}

static inline float deltanetQhpiDequantU16(
    uint16_t value,
    QHPI_Quant_Parameters qparams) {
  return (static_cast<float>(value) -
          static_cast<float>(qparams.zero_offset)) *
      qparams.stepsize;
}

static inline uint16_t deltanetQhpiQuantU16(
    float value,
    QHPI_Quant_Parameters qparams) {
  const float scaled =
      std::round(static_cast<float>(qparams.zero_offset) +
                 value / qparams.stepsize);
  const float clamped = std::min(std::max(scaled, 0.0f), 65535.0f);
  return static_cast<uint16_t>(clamped);
}

static inline bool deltanetQhpiIsValidQuant(QHPI_Quant_Parameters qparams) {
  return qparams.zero_offset != INT32_MAX && std::isfinite(qparams.stepsize) &&
      qparams.stepsize > 0.0f;
}

static inline int32_t deltanetQhpiCenteredU16(
    uint16_t value,
    QHPI_Quant_Parameters qparams) {
  return static_cast<int32_t>(value) - qparams.zero_offset;
}

static inline uint16_t deltanetQhpiClampCodeToU16(int64_t code) {
  if (code <= 0) {
    return 0;
  }
  if (code >= 65535) {
    return 65535;
  }
  return static_cast<uint16_t>(code);
}

static inline uint16_t deltanetQhpiQuantU16FromScaledDouble(double scaled) {
  if (!std::isfinite(scaled) || scaled <= 0.0) {
    return 0;
  }
  if (scaled >= 65535.0) {
    return 65535;
  }
  return static_cast<uint16_t>(scaled + 0.5);
}

static inline int64_t deltanetQhpiRoundFixed(double value, int shift) {
  const double scaled = std::ldexp(value, shift);
  if (!std::isfinite(scaled)) {
    return 0;
  }
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

static inline int64_t deltanetQhpiRoundShiftI64(int64_t value, int shift) {
  const int64_t rounding = INT64_C(1) << (shift - 1);
  if (value >= 0) {
    return (value + rounding) >> shift;
  }
  return -(((-value) + rounding) >> shift);
}

static inline bool deltanetQhpiFitsI16(int32_t value) {
  return value >= std::numeric_limits<int16_t>::min() &&
      value <= std::numeric_limits<int16_t>::max();
}

static inline bool deltanetQhpiFitsI16(int64_t value) {
  return value >= std::numeric_limits<int16_t>::min() &&
      value <= std::numeric_limits<int16_t>::max();
}

static inline bool deltanetQhpiFitsI32(int64_t value) {
  return value >= std::numeric_limits<int32_t>::min() &&
      value <= std::numeric_limits<int32_t>::max();
}

static inline bool deltanetQhpiCanCenterU16AsI16(
    QHPI_Quant_Parameters qparams) {
  return qparams.zero_offset >= 0 && qparams.zero_offset <= 65535;
}

static inline size_t deltanetQhpiTimeHeadIndex(
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
    if (shape.dims[1] > token && shape.dims[2] == 1 &&
        shape.dims[3] > head) {
      return ((batch * shape.dims[1] + token) * shape.dims[2]) *
          shape.dims[3] +
          head;
    }
    if (shape.dims[1] == 1 && shape.dims[2] > token &&
        shape.dims[3] > head) {
      return ((batch * shape.dims[1]) * shape.dims[2] + token) *
          shape.dims[3] +
          head;
    }
    if (shape.dims[1] > token && shape.dims[2] > head &&
        shape.dims[3] == 1) {
      return ((batch * shape.dims[1] + token) * shape.dims[2] + head) *
          shape.dims[3];
    }
    if (shape.dims[1] > head && shape.dims[2] > token &&
        shape.dims[3] == 1) {
      return ((batch * shape.dims[1] + head) * shape.dims[2] + token) *
          shape.dims[3];
    }
  }
  return 0;
}

static inline bool deltanetQhpiHasTimeHeadLayout(
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

static inline size_t deltanetQhpiValueTileCount(
    size_t rows,
    size_t slices,
    size_t value_dim,
    size_t min_tile_elements) {
  // SM8850 tests show value tiling helps only for the single-row max-V case.
  if (rows != 1 || rows >= slices || value_dim < kValueTileMinDim ||
      value_dim <= min_tile_elements) {
    return 1;
  }
  const size_t wanted_tiles = deltanetQhpiDivUp(slices, rows);
  const size_t max_vector_tiles =
      std::max<size_t>(value_dim / min_tile_elements, 1);
  return std::max<size_t>(std::min(wanted_tiles, max_vector_tiles), 1);
}

static float deltanetCoreQhpiCost(
    const uint32_t num_inputs,
    const QHPI_Tensor* const* inputs) {
  if (num_inputs < 3) {
    return 1.0f;
  }
  const QHPI_Shape query_shape = qhpi_tensor_shape(inputs[0]);
  const QHPI_Shape value_shape = qhpi_tensor_shape(inputs[2]);
  if (query_shape.rank != 4 || value_shape.rank != 4) {
    return 1.0f;
  }
  const float batch = static_cast<float>(query_shape.dims[0]);
  const float seq_len = static_cast<float>(query_shape.dims[1]);
  const float heads = static_cast<float>(query_shape.dims[2]);
  const float key_dim = static_cast<float>(query_shape.dims[3]);
  const float value_dim = static_cast<float>(value_shape.dims[3]);
  return std::max(1024.0f, batch * seq_len * heads * key_dim * value_dim);
}

static inline HVX_Vector deltanetQhpiHvxMulF32(HVX_Vector a, HVX_Vector b) {
#if __HVX_ARCH__ < 79
  return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b));
#else
  return Q6_Vsf_vmpy_VsfVsf(a, b);
#endif
}

static inline HVX_Vector deltanetQhpiHvxAddF32(HVX_Vector a, HVX_Vector b) {
#if __HVX_ARCH__ < 79
  return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b));
#else
  return Q6_Vsf_vadd_VsfVsf(a, b);
#endif
}

static inline HVX_Vector deltanetQhpiHvxRoundF32ToI32(HVX_Vector value) {
  const HVX_Vector v_half = Q6_V_vsplat_R(0x3F000000);
  const HVX_Vector v_neg_half = Q6_V_vsplat_R(0xBF000000);
  const HVX_VectorPred q_neg =
      Q6_Q_vcmp_gt_VsfVsf(Q6_V_vzero(), value);
  return Q6_Vw_equals_Vsf(
      deltanetQhpiHvxAddF32(value, Q6_V_vmux_QVV(q_neg, v_neg_half, v_half)));
}

static inline HVX_Vector deltanetQhpiHvxSplatF32(float value) {
  return q6op_V_vsplat_float32(value);
}

static inline HVX_VectorPair deltanetQhpiHvxF32PairToQF32(
    HVX_Vector lo,
    HVX_Vector hi) {
  return Q6_W_vcombine_VV(Q6_Vqf32_equals_Vsf(hi), Q6_Vqf32_equals_Vsf(lo));
}

static inline HVX_VectorPair deltanetQhpiHvxCenteredShiftedU16ToI32FromVector(
    HVX_Vector input,
    int32_t zero_offset,
    int input_shift) {
  const HVX_Vector v_shifted_raw = Q6_Vuh_vlsr_VuhR(input, input_shift);
  HVX_VectorPair v_words = Q6_Wuw_vzxt_Vuh(v_shifted_raw);
  v_words = Q6_W_vshuff_VVR(Q6_V_hi_W(v_words), Q6_V_lo_W(v_words), -4);
  const HVX_Vector v_zero = Q6_V_vsplat_R(zero_offset >> input_shift);
  const HVX_Vector v_lo = Q6_Vw_vsub_VwVw(Q6_V_lo_W(v_words), v_zero);
  const HVX_Vector v_hi = Q6_Vw_vsub_VwVw(Q6_V_hi_W(v_words), v_zero);
  return Q6_W_vcombine_VV(v_hi, v_lo);
}

static inline HVX_VectorPair deltanetQhpiHvxMulCenteredRawU16I32(
    HVX_Vector input,
    int32_t zero_offset,
    int32_t multiplier) {
  const HVX_Vector v_zero =
      Q6_Vh_vsplat_R(static_cast<int>(
          (zero_offset >> kQUint16HvxAccumInputShift) & 0xFFFF));
  const HVX_Vector v_shifted_raw =
      Q6_Vuh_vlsr_VuhR(input, kQUint16HvxAccumInputShift);
  const HVX_Vector v_shifted = Q6_Vh_vsub_VhVh(v_shifted_raw, v_zero);
  const HVX_VectorPair v_product = Q6_Ww_vmpy_VhRh(
      v_shifted, Q6_R_combine_RlRl(multiplier, multiplier));
  return Q6_W_vshuff_VVR(Q6_V_hi_W(v_product), Q6_V_lo_W(v_product), -4);
}

static inline HVX_VectorPair deltanetQhpiHvxMulCenteredU16I32(
    const uint16_t* input,
    int32_t zero_offset,
    int32_t multiplier) {
  return deltanetQhpiHvxMulCenteredRawU16I32(
      vmemu(input), zero_offset, multiplier);
}

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
  // On SM8850 the one-time k=0 scratch init is faster scalar; k>=1 stays HVX.
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
        (static_cast<int32_t>(prev_state[v]) >>
         kQUint16HvxAccumInputShift) -
        (prev_zero_offset >> kQUint16HvxAccumInputShift);
    kv_acc[v] = prev * key_value;
    output_pre_acc[v] = prev * query_value;
  }
}

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
        (static_cast<int32_t>(prev_state[v]) >>
         kQUint16HvxAccumInputShift) -
        (prev_zero_offset >> kQUint16HvxAccumInputShift);
    kv_acc[v] += prev * key_value;
    output_pre_acc[v] += prev * query_value;
  }
}

static inline HVX_VectorPair deltanetQhpiHvxCenteredShiftedU16ToI32(
    const uint16_t* input,
    int32_t zero_offset,
    int input_shift) {
  return deltanetQhpiHvxCenteredShiftedU16ToI32FromVector(
      vmemu(input), zero_offset, input_shift);
}

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
    const HVX_VectorPair v_prev = deltanetQhpiHvxCenteredShiftedU16ToI32(
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
    const int32_t prev =
        (static_cast<int32_t>(prev_state[v]) >>
         kQUint16HvxStateUpdateInputShift) -
        (prev_zero_offset >> kQUint16HvxStateUpdateInputShift);
    const int64_t state_center = deltanetQhpiRoundShiftI64(
        static_cast<int64_t>(prev_update_value) * prev +
            static_cast<int64_t>(key_value) * delta_update_fixed32[v],
        final_shift);
    state[v] = deltanetQhpiClampCodeToU16(
        static_cast<int64_t>(state_zero_offset) + state_center);
  }
}

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
  const HVX_Vector v_prev_scale = deltanetQhpiHvxSplatF32(prev_scale);
  const HVX_Vector v_key_scale = deltanetQhpiHvxSplatF32(key_scale);
  const HVX_Vector v_state_zero =
      deltanetQhpiHvxSplatF32(static_cast<float>(state_zero_offset) + 0.5f);

  for (size_t i = 0; i < vec_count; ++i) {
    const size_t offset = i * halfs_per_vec;
    const HVX_VectorPair v_prev = deltanetQhpiHvxCenteredShiftedU16ToI32(
        prev_state + offset, prev_zero_offset, 0);
    const HVX_Vector v_prev_lo = Q6_Vsf_equals_Vw(Q6_V_lo_W(v_prev));
    const HVX_Vector v_prev_hi = Q6_Vsf_equals_Vw(Q6_V_hi_W(v_prev));
    const HVX_Vector v_delta_lo = vmemu(delta_value + offset);
    const HVX_Vector v_delta_hi = vmemu(delta_value + offset + words_per_vec);
    const HVX_Vector v_code_lo = deltanetQhpiHvxAddF32(
        v_state_zero,
        deltanetQhpiHvxAddF32(
            deltanetQhpiHvxMulF32(v_prev_lo, v_prev_scale),
            deltanetQhpiHvxMulF32(v_delta_lo, v_key_scale)));
    const HVX_Vector v_code_hi = deltanetQhpiHvxAddF32(
        v_state_zero,
        deltanetQhpiHvxAddF32(
            deltanetQhpiHvxMulF32(v_prev_hi, v_prev_scale),
            deltanetQhpiHvxMulF32(v_delta_hi, v_key_scale)));
    const HVX_Vector v_code_i_lo = Q6_Vw_equals_Vsf(v_code_lo);
    const HVX_Vector v_code_i_hi = Q6_Vw_equals_Vsf(v_code_hi);
    const HVX_Vector v_state_half =
        Q6_Vuh_vasr_VwVwR_sat(v_code_i_hi, v_code_i_lo, 0);
    vmemu(state + offset) = Q6_Vh_vdeal_Vh(v_state_half);
  }

  for (size_t v = vec_count * halfs_per_vec; v < value_dim; ++v) {
    const float prev =
        static_cast<float>(static_cast<int32_t>(prev_state[v]) -
                           prev_zero_offset);
    const float code =
        static_cast<float>(state_zero_offset) + prev * prev_scale +
        delta_value[v] * key_scale;
    state[v] = deltanetQhpiQuantU16FromScaledDouble(
        static_cast<double>(code));
  }
}

static inline size_t deltanetQhpiHvxDeltaOutputU16F32(
    uint16_t* output,
    const uint16_t* value,
    const int32_t* kv_acc,
    const int32_t* output_pre_acc,
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
  const HVX_Vector v_value_beta =
      deltanetQhpiHvxSplatF32(value_beta_scale);
  const HVX_Vector v_neg_kv_beta =
      deltanetQhpiHvxSplatF32(-kv_beta_scale);
  const HVX_Vector v_output_acc =
      deltanetQhpiHvxSplatF32(output_acc_scale);
  const HVX_Vector v_output_delta =
      deltanetQhpiHvxSplatF32(output_delta_scale);
  const HVX_Vector v_delta_update =
      deltanetQhpiHvxSplatF32(delta_update_fixed32_scale);
  const HVX_Vector v_output_zero =
      deltanetQhpiHvxSplatF32(static_cast<float>(output_zero_offset) + 0.5f);

  for (size_t i = 0; i < vec_count; ++i) {
    const size_t offset = i * halfs_per_vec;
    const HVX_VectorPair v_value =
        deltanetQhpiHvxCenteredShiftedU16ToI32(
            value + offset, value_zero_offset, 0);
    const HVX_Vector v_value_lo = Q6_Vsf_equals_Vw(Q6_V_lo_W(v_value));
    const HVX_Vector v_value_hi = Q6_Vsf_equals_Vw(Q6_V_hi_W(v_value));
    const HVX_Vector v_kv_lo = Q6_Vsf_equals_Vw(vmemu(kv_acc + offset));
    const HVX_Vector v_kv_hi =
        Q6_Vsf_equals_Vw(vmemu(kv_acc + offset + words_per_vec));
    const HVX_Vector v_pre_lo =
        Q6_Vsf_equals_Vw(vmemu(output_pre_acc + offset));
    const HVX_Vector v_pre_hi =
        Q6_Vsf_equals_Vw(vmemu(output_pre_acc + offset + words_per_vec));

    const HVX_Vector v_delta_lo = deltanetQhpiHvxAddF32(
        deltanetQhpiHvxMulF32(v_value_lo, v_value_beta),
        deltanetQhpiHvxMulF32(v_kv_lo, v_neg_kv_beta));
    const HVX_Vector v_delta_hi = deltanetQhpiHvxAddF32(
        deltanetQhpiHvxMulF32(v_value_hi, v_value_beta),
        deltanetQhpiHvxMulF32(v_kv_hi, v_neg_kv_beta));
    const HVX_Vector v_code_lo = deltanetQhpiHvxAddF32(
        v_output_zero,
        deltanetQhpiHvxAddF32(
            deltanetQhpiHvxMulF32(v_pre_lo, v_output_acc),
            deltanetQhpiHvxMulF32(v_delta_lo, v_output_delta)));
    const HVX_Vector v_code_hi = deltanetQhpiHvxAddF32(
        v_output_zero,
        deltanetQhpiHvxAddF32(
            deltanetQhpiHvxMulF32(v_pre_hi, v_output_acc),
            deltanetQhpiHvxMulF32(v_delta_hi, v_output_delta)));

    if (delta_value != nullptr) {
      vmemu(delta_value + offset) = v_delta_lo;
      vmemu(delta_value + offset + words_per_vec) = v_delta_hi;
    }
    if (delta_update_fixed32 != nullptr) {
      vmemu(delta_update_fixed32 + offset) = deltanetQhpiHvxRoundF32ToI32(
          deltanetQhpiHvxMulF32(v_delta_lo, v_delta_update));
      vmemu(delta_update_fixed32 + offset + words_per_vec) =
          deltanetQhpiHvxRoundF32ToI32(
              deltanetQhpiHvxMulF32(v_delta_hi, v_delta_update));
    }
    const HVX_Vector v_code_i_lo = Q6_Vw_equals_Vsf(v_code_lo);
    const HVX_Vector v_code_i_hi = Q6_Vw_equals_Vsf(v_code_hi);
    const HVX_Vector v_output_half =
        Q6_Vuh_vasr_VwVwR_sat(v_code_i_hi, v_code_i_lo, 0);
    vmemu(output + offset) = Q6_Vh_vdeal_Vh(v_output_half);
  }

  return vec_count * halfs_per_vec;
}

static inline HVX_Vector deltanetQhpiHvxSplatF16(Float16 value) {
  return Q6_Vh_vsplat_R(value.raw());
}

static inline HVX_Vector deltanetQhpiHvxMulF16(
    HVX_Vector a,
    HVX_Vector b) {
  return Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(a, b));
}

static inline HVX_Vector deltanetQhpiHvxAddF16(
    HVX_Vector a,
    HVX_Vector b) {
  return Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(a, b));
}

static inline HVX_Vector deltanetQhpiHvxSubF16(
    HVX_Vector a,
    HVX_Vector b) {
  return Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_VhfVhf(a, b));
}

static inline Float16 deltanetQhpiRoundF16(float value) {
  return Float16(value);
}

static inline Float16 deltanetQhpiMulF16(Float16 a, Float16 b) {
  return deltanetQhpiRoundF16(static_cast<float>(a) * static_cast<float>(b));
}

static inline Float16 deltanetQhpiAddF16(Float16 a, Float16 b) {
  return deltanetQhpiRoundF16(static_cast<float>(a) + static_cast<float>(b));
}

static inline Float16 deltanetQhpiSubF16(Float16 a, Float16 b) {
  return deltanetQhpiRoundF16(static_cast<float>(a) - static_cast<float>(b));
}

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
  const HVX_Vector v_key = deltanetQhpiHvxSplatF32(key_scale);
  const HVX_Vector v_query = deltanetQhpiHvxSplatF32(query_scale);

  for (size_t i = 0; i < vec_count; ++i) {
    const size_t offset = i * halfs_per_vec;
    const HVX_Vector v_half = vmemu(prev_state + offset);
    const HVX_VectorPair v_prev_q = Q6_Wqf32_vmpy_VhfVhf(v_half, v_one);
    const HVX_Vector v_prev_lo = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(v_prev_q));
    const HVX_Vector v_prev_hi = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(v_prev_q));

    const HVX_Vector v_kv_lo = deltanetQhpiHvxAddF32(
        vmemu(kv_mem + offset), deltanetQhpiHvxMulF32(v_prev_lo, v_key));
    const HVX_Vector v_kv_hi = deltanetQhpiHvxAddF32(
        vmemu(kv_mem + offset + floats_per_vec),
        deltanetQhpiHvxMulF32(v_prev_hi, v_key));
    const HVX_Vector v_out_lo = deltanetQhpiHvxAddF32(
        vmemu(output_pre + offset),
        deltanetQhpiHvxMulF32(v_prev_lo, v_query));
    const HVX_Vector v_out_hi = deltanetQhpiHvxAddF32(
        vmemu(output_pre + offset + floats_per_vec),
        deltanetQhpiHvxMulF32(v_prev_hi, v_query));

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
  const HVX_Vector v_key = deltanetQhpiHvxSplatF32(key_scale);
  const HVX_Vector v_query = deltanetQhpiHvxSplatF32(query_scale);

  for (size_t i = 0; i < vec_count; ++i) {
    const size_t offset = i * halfs_per_vec;
    const HVX_Vector v_half = vmemu(prev_state + offset);
    const HVX_VectorPair v_prev_q = Q6_Wqf32_vmpy_VhfVhf(v_half, v_one);
    const HVX_Vector v_prev_lo = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(v_prev_q));
    const HVX_Vector v_prev_hi = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(v_prev_q));

    vmemu(kv_mem + offset) = deltanetQhpiHvxMulF32(v_prev_lo, v_key);
    vmemu(kv_mem + offset + floats_per_vec) =
        deltanetQhpiHvxMulF32(v_prev_hi, v_key);
    vmemu(output_pre + offset) =
        deltanetQhpiHvxMulF32(v_prev_lo, v_query);
    vmemu(output_pre + offset + floats_per_vec) =
        deltanetQhpiHvxMulF32(v_prev_hi, v_query);
  }

  for (size_t v = vec_count * halfs_per_vec; v < value_dim; ++v) {
    const float prev = static_cast<float>(prev_state[v]);
    kv_mem[v] = prev * key_scale;
    output_pre[v] = prev * query_scale;
  }
}

static inline void deltanetQhpiHvxAccumF16AccumF16(
    const Float16* prev_state,
    Float16* kv_mem,
    Float16* output_pre,
    Float16 key_scale,
    Float16 query_scale,
    size_t value_dim) {
  const size_t halfs_per_vec = 128 / sizeof(Float16);
  const size_t vec_count = value_dim / halfs_per_vec;
  const HVX_Vector v_key = deltanetQhpiHvxSplatF16(key_scale);
  const HVX_Vector v_query = deltanetQhpiHvxSplatF16(query_scale);

  for (size_t i = 0; i < vec_count; ++i) {
    const size_t offset = i * halfs_per_vec;
    const HVX_Vector v_prev = vmemu(prev_state + offset);
    vmemu(kv_mem + offset) = deltanetQhpiHvxAddF16(
        vmemu(kv_mem + offset), deltanetQhpiHvxMulF16(v_prev, v_key));
    vmemu(output_pre + offset) = deltanetQhpiHvxAddF16(
        vmemu(output_pre + offset),
        deltanetQhpiHvxMulF16(v_prev, v_query));
  }

  for (size_t v = vec_count * halfs_per_vec; v < value_dim; ++v) {
    kv_mem[v] = deltanetQhpiAddF16(
        kv_mem[v], deltanetQhpiMulF16(prev_state[v], key_scale));
    output_pre[v] = deltanetQhpiAddF16(
        output_pre[v], deltanetQhpiMulF16(prev_state[v], query_scale));
  }
}

static inline void deltanetQhpiHvxAssignAccumF16AccumF16(
    const Float16* prev_state,
    Float16* kv_mem,
    Float16* output_pre,
    Float16 key_scale,
    Float16 query_scale,
    size_t value_dim) {
  const size_t halfs_per_vec = 128 / sizeof(Float16);
  const size_t vec_count = value_dim / halfs_per_vec;
  const HVX_Vector v_key = deltanetQhpiHvxSplatF16(key_scale);
  const HVX_Vector v_query = deltanetQhpiHvxSplatF16(query_scale);

  for (size_t i = 0; i < vec_count; ++i) {
    const size_t offset = i * halfs_per_vec;
    const HVX_Vector v_prev = vmemu(prev_state + offset);
    vmemu(kv_mem + offset) = deltanetQhpiHvxMulF16(v_prev, v_key);
    vmemu(output_pre + offset) = deltanetQhpiHvxMulF16(v_prev, v_query);
  }

  for (size_t v = vec_count * halfs_per_vec; v < value_dim; ++v) {
    kv_mem[v] = deltanetQhpiMulF16(prev_state[v], key_scale);
    output_pre[v] = deltanetQhpiMulF16(prev_state[v], query_scale);
  }
}

static inline void deltanetQhpiHvxAccumF32(
    const float* prev_state,
    float* kv_mem,
    float* output_pre,
    float key_scale,
    float query_scale,
    size_t value_dim) {
  const size_t epv = 128 / sizeof(float);
  const size_t vec_count = value_dim / epv;
  const HVX_Vector v_key = deltanetQhpiHvxSplatF32(key_scale);
  const HVX_Vector v_query = deltanetQhpiHvxSplatF32(query_scale);

  for (size_t i = 0; i < vec_count; ++i) {
    const size_t offset = i * epv;
    const HVX_Vector v_prev = vmemu(prev_state + offset);
    vmemu(kv_mem + offset) = deltanetQhpiHvxAddF32(
        vmemu(kv_mem + offset), deltanetQhpiHvxMulF32(v_prev, v_key));
    vmemu(output_pre + offset) = deltanetQhpiHvxAddF32(
        vmemu(output_pre + offset),
        deltanetQhpiHvxMulF32(v_prev, v_query));
  }

  for (size_t v = vec_count * epv; v < value_dim; ++v) {
    const float prev = prev_state[v];
    kv_mem[v] += prev * key_scale;
    output_pre[v] += prev * query_scale;
  }
}

static inline void deltanetQhpiHvxAssignAccumF32(
    const float* prev_state,
    float* kv_mem,
    float* output_pre,
    float key_scale,
    float query_scale,
    size_t value_dim) {
  const size_t epv = 128 / sizeof(float);
  const size_t vec_count = value_dim / epv;
  const HVX_Vector v_key = deltanetQhpiHvxSplatF32(key_scale);
  const HVX_Vector v_query = deltanetQhpiHvxSplatF32(query_scale);

  for (size_t i = 0; i < vec_count; ++i) {
    const size_t offset = i * epv;
    const HVX_Vector v_prev = vmemu(prev_state + offset);
    vmemu(kv_mem + offset) = deltanetQhpiHvxMulF32(v_prev, v_key);
    vmemu(output_pre + offset) = deltanetQhpiHvxMulF32(v_prev, v_query);
  }

  for (size_t v = vec_count * epv; v < value_dim; ++v) {
    const float prev = prev_state[v];
    kv_mem[v] = prev * key_scale;
    output_pre[v] = prev * query_scale;
  }
}

static inline void deltanetQhpiHvxUpdateStateF32(
    const float* prev_state,
    float* state,
    const float* delta,
    float decay,
    float key_value,
    size_t value_dim) {
  const size_t epv = 128 / sizeof(float);
  const size_t vec_count = value_dim / epv;
  const HVX_Vector v_decay = deltanetQhpiHvxSplatF32(decay);
  const HVX_Vector v_key = deltanetQhpiHvxSplatF32(key_value);

  for (size_t i = 0; i < vec_count; ++i) {
    const size_t offset = i * epv;
    const HVX_Vector v_updated = deltanetQhpiHvxAddF32(
        deltanetQhpiHvxMulF32(vmemu(prev_state + offset), v_decay),
        deltanetQhpiHvxMulF32(vmemu(delta + offset), v_key));
    vmemu(state + offset) = v_updated;
  }

  for (size_t v = vec_count * epv; v < value_dim; ++v) {
    state[v] = prev_state[v] * decay + delta[v] * key_value;
  }
}

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
  const HVX_Vector v_beta = deltanetQhpiHvxSplatF32(beta_value);
  const HVX_Vector v_dot = deltanetQhpiHvxSplatF32(key_query_dot);

  for (size_t i = 0; i < vec_count; ++i) {
    const size_t offset = i * epv;
    const HVX_Vector v_delta = deltanetQhpiHvxMulF32(
        deltanetQhpiHvxAddF32(
            vmemu(value + offset),
            deltanetQhpiHvxMulF32(vmemu(kv_mem + offset), v_neg_one)),
        v_beta);
    const HVX_Vector v_out = deltanetQhpiHvxAddF32(
        vmemu(output_pre + offset),
        deltanetQhpiHvxMulF32(v_delta, v_dot));
    vmemu(delta + offset) = v_delta;
    vmemu(output + offset) = v_out;
  }

  for (size_t v = vec_count * epv; v < value_dim; ++v) {
    const float delta_value = (value[v] - kv_mem[v]) * beta_value;
    delta[v] = delta_value;
    output[v] = output_pre[v] + delta_value * key_query_dot;
  }
}

static inline void deltanetQhpiHvxUpdateStateF16AccumF16(
    const Float16* prev_state,
    Float16* state,
    const Float16* delta,
    Float16 decay,
    Float16 key_value,
    size_t value_dim) {
  const size_t halfs_per_vec = 128 / sizeof(Float16);
  const size_t vec_count = value_dim / halfs_per_vec;
  const HVX_Vector v_decay = deltanetQhpiHvxSplatF16(decay);
  const HVX_Vector v_key = deltanetQhpiHvxSplatF16(key_value);

  for (size_t i = 0; i < vec_count; ++i) {
    const size_t offset = i * halfs_per_vec;
    const HVX_Vector v_prev = vmemu(prev_state + offset);
    const HVX_Vector v_delta = vmemu(delta + offset);
    const HVX_Vector v_updated = deltanetQhpiHvxAddF16(
        deltanetQhpiHvxMulF16(v_prev, v_decay),
        deltanetQhpiHvxMulF16(v_delta, v_key));
    vmemu(state + offset) = v_updated;
  }

  for (size_t v = vec_count * halfs_per_vec; v < value_dim; ++v) {
    state[v] = deltanetQhpiAddF16(
        deltanetQhpiMulF16(prev_state[v], decay),
        deltanetQhpiMulF16(delta[v], key_value));
  }
}

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
  const HVX_Vector v_beta = deltanetQhpiHvxSplatF16(beta_value);
  const HVX_Vector v_dot = deltanetQhpiHvxSplatF16(key_query_dot);

  for (size_t i = 0; i < vec_count; ++i) {
    const size_t offset = i * halfs_per_vec;
    const HVX_Vector v_delta = deltanetQhpiHvxMulF16(
        deltanetQhpiHvxSubF16(vmemu(value + offset), vmemu(kv_mem + offset)),
        v_beta);
    const HVX_Vector v_out = deltanetQhpiHvxAddF16(
        vmemu(output_pre + offset), deltanetQhpiHvxMulF16(v_delta, v_dot));
    vmemu(delta + offset) = v_delta;
    vmemu(output + offset) = v_out;
  }

  for (size_t v = vec_count * halfs_per_vec; v < value_dim; ++v) {
    const Float16 delta_value = deltanetQhpiMulF16(
        deltanetQhpiSubF16(value[v], kv_mem[v]), beta_value);
    delta[v] = delta_value;
    output[v] = deltanetQhpiAddF16(
        output_pre[v], deltanetQhpiMulF16(delta_value, key_query_dot));
  }
}

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
  const HVX_Vector v_decay = deltanetQhpiHvxSplatF32(decay);
  const HVX_Vector v_key = deltanetQhpiHvxSplatF32(key_value);

  for (size_t i = 0; i < vec_count; ++i) {
    const size_t offset = i * halfs_per_vec;
    const HVX_Vector v_half = vmemu(prev_state + offset);
    const HVX_VectorPair v_prev_q = Q6_Wqf32_vmpy_VhfVhf(v_half, v_one);
    const HVX_Vector v_prev_lo = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(v_prev_q));
    const HVX_Vector v_prev_hi = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(v_prev_q));
    const HVX_Vector v_delta_lo = vmemu(delta + offset);
    const HVX_Vector v_delta_hi = vmemu(delta + offset + floats_per_vec);
    const HVX_Vector v_updated_lo = deltanetQhpiHvxAddF32(
        deltanetQhpiHvxMulF32(v_prev_lo, v_decay),
        deltanetQhpiHvxMulF32(v_delta_lo, v_key));
    const HVX_Vector v_updated_hi = deltanetQhpiHvxAddF32(
        deltanetQhpiHvxMulF32(v_prev_hi, v_decay),
        deltanetQhpiHvxMulF32(v_delta_hi, v_key));
    vmemu(state + offset) = Q6_Vhf_equals_Wqf32(
        deltanetQhpiHvxF32PairToQF32(v_updated_lo, v_updated_hi));
  }

  for (size_t v = vec_count * halfs_per_vec; v < value_dim; ++v) {
    state[v] = Float16(
        static_cast<float>(prev_state[v]) * decay + delta[v] * key_value);
  }
}

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
  const HVX_Vector v_beta = deltanetQhpiHvxSplatF32(beta_value);
  const HVX_Vector v_dot = deltanetQhpiHvxSplatF32(key_query_dot);

  for (size_t i = 0; i < vec_count; ++i) {
    const size_t offset = i * halfs_per_vec;
    const HVX_Vector v_half = vmemu(value + offset);
    const HVX_VectorPair v_value_q = Q6_Wqf32_vmpy_VhfVhf(v_half, v_one);
    const HVX_Vector v_value_lo = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(v_value_q));
    const HVX_Vector v_value_hi = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(v_value_q));
    const HVX_Vector v_delta_lo = deltanetQhpiHvxMulF32(
        deltanetQhpiHvxAddF32(
            v_value_lo,
            deltanetQhpiHvxMulF32(vmemu(kv_mem + offset), v_neg_one)),
        v_beta);
    const HVX_Vector v_delta_hi = deltanetQhpiHvxMulF32(
        deltanetQhpiHvxAddF32(
            v_value_hi,
            deltanetQhpiHvxMulF32(
                vmemu(kv_mem + offset + floats_per_vec), v_neg_one)),
        v_beta);
    const HVX_Vector v_out_lo = deltanetQhpiHvxAddF32(
        vmemu(output_pre + offset),
        deltanetQhpiHvxMulF32(v_delta_lo, v_dot));
    const HVX_Vector v_out_hi = deltanetQhpiHvxAddF32(
        vmemu(output_pre + offset + floats_per_vec),
        deltanetQhpiHvxMulF32(v_delta_hi, v_dot));
    vmemu(delta + offset) = v_delta_lo;
    vmemu(delta + offset + floats_per_vec) = v_delta_hi;
    vmemu(output + offset) = Q6_Vhf_equals_Wqf32(
        deltanetQhpiHvxF32PairToQF32(v_out_lo, v_out_hi));
  }

  for (size_t v = vec_count * halfs_per_vec; v < value_dim; ++v) {
    const float delta_value =
        (static_cast<float>(value[v]) - kv_mem[v]) * beta_value;
    delta[v] = delta_value;
    output[v] = Float16(output_pre[v] + delta_value * key_query_dot);
  }
}

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

    deltanetQhpiHvxDeltaAndWriteOutputF16(
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
    size_t value_tile_count) {
  float kv_mem[kMaxDeltaNetDim] __attribute__((aligned(128)));
  float delta[kMaxDeltaNetDim] __attribute__((aligned(128)));
  float output_pre[kMaxDeltaNetDim] __attribute__((aligned(128)));

  const size_t total_rows = batch * heads;
  if (value_tile_count <= 1) {
    for (size_t row = slice; row < total_rows; row += num_slices) {
      deltanetCoreQhpiRunFp16Tile(
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
    deltanetCoreQhpiRunFp16Tile(
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
      key_query_dot = deltanetQhpiAddF16(
          key_query_dot, deltanetQhpiMulF16(key_row[k], query_row[k]));
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
          deltanetQhpiMulF16(decay, key_row[0]),
          deltanetQhpiMulF16(decay, query_row[0]),
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
          deltanetQhpiMulF16(decay, key_row[k]),
          deltanetQhpiMulF16(decay, query_row[k]),
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
    size_t seq_len,
    size_t heads,
    size_t key_dim,
    size_t value_dim,
    size_t row,
    size_t v_begin,
    size_t v_end,
    int64_t* kv_acc,
    int64_t* output_pre_acc,
    int32_t* kv_acc32,
    int32_t* output_pre_acc32,
    int64_t* delta_update_fixed,
    int32_t* delta_update_fixed32,
    float* delta_value_scratch,
    int32_t* key_center,
    int32_t* query_center) {
  const size_t tile_size = v_end - v_begin;
  if (tile_size == 0) {
    return;
  }

  const size_t b = row / heads;
  const size_t h = row % heads;
  const size_t state_base = row * key_dim * value_dim;

  for (size_t t = 0; t < seq_len; ++t) {
    const size_t qkv_base = ((b * seq_len + t) * heads + h);
    const uint16_t* query_row = query_data + qkv_base * key_dim;
    const uint16_t* key_row = key_data + qkv_base * key_dim;
    const uint16_t* value_tile = value_data + qkv_base * value_dim + v_begin;

    int64_t key_query_acc = 0;
    for (size_t k = 0; k < key_dim; ++k) {
      key_center[k] = deltanetQhpiCenteredU16(key_row[k], key_q);
      query_center[k] = deltanetQhpiCenteredU16(query_row[k], query_q);
      key_query_acc +=
          static_cast<int64_t>(key_center[k]) * query_center[k];
    }
    const double key_query_dot =
        static_cast<double>(key_query_acc) * key_q.stepsize *
        query_q.stepsize;

    const double decay = deltanetQhpiDequantU16(
        decay_data[deltanetQhpiTimeHeadIndex(decay_shape, b, t, h)],
        decay_q);
    const double beta_value = deltanetQhpiDequantU16(
        beta_data[deltanetQhpiTimeHeadIndex(beta_shape, b, t, h)],
        beta_q);
    const QHPI_Quant_Parameters prev_q = t == 0 ? recurrent_state_q : state_q;
    bool hvx_accum = deltanetQhpiCanCenterU16AsI16(prev_q);
    for (size_t k = 0; k < key_dim; ++k) {
      hvx_accum &=
          deltanetQhpiFitsI16(key_center[k] >>
                              kQUint16HvxAccumInputShift) &&
          deltanetQhpiFitsI16(query_center[k] >>
                              kQUint16HvxAccumInputShift);
    }

    const double delta_update_scale = key_q.stepsize / state_q.stepsize;
    const int64_t prev_update_fixed = deltanetQhpiRoundFixed(
        decay * prev_q.stepsize / state_q.stepsize,
        kQUint16StateUpdateShift);
    int state_update_final_shift = 0;
    int64_t prev_update_shifted = 0;
    bool hvx_update = false;
    constexpr bool hvx_update_f32 = kEnableQUint16HvxStateUpdateF32;
    if constexpr (kEnableQUint16HvxStateUpdate) {
      state_update_final_shift = kQUint16StateUpdateShift -
          kQUint16HvxStateUpdateInputShift -
          kQUint16HvxStateUpdateCoeffShift;
      prev_update_shifted = deltanetQhpiRoundShiftI64(
          prev_update_fixed, kQUint16HvxStateUpdateCoeffShift);
      hvx_update = state_update_final_shift > 0 &&
          deltanetQhpiCanCenterU16AsI16(prev_q) &&
          deltanetQhpiFitsI16(prev_update_shifted) &&
          deltanetQhpiFitsI32(
              static_cast<int64_t>(state_q.zero_offset)
              << state_update_final_shift);
      for (size_t k = 0; k < key_dim; ++k) {
        hvx_update &=
            deltanetQhpiFitsI16(key_center[k] >>
                                kQUint16HvxStateUpdateInputShift);
      }
    }
    {
      const size_t row_base = state_base + v_begin;
      const uint16_t* prev_state =
          t == 0 ? recurrent_state_data + row_base : state_data + row_base;
      const int32_t key_value = key_center[0];
      const int32_t query_value = query_center[0];
      if (hvx_accum) {
        deltanetQhpiHvxAssignAccumU16I32(
            prev_state,
            kv_acc32,
            output_pre_acc32,
            prev_q.zero_offset,
            key_value >> kQUint16HvxAccumInputShift,
            query_value >> kQUint16HvxAccumInputShift,
            tile_size);
      } else {
        for (size_t v = 0; v < tile_size; ++v) {
          const int32_t prev = deltanetQhpiCenteredU16(prev_state[v], prev_q);
          kv_acc[v] = static_cast<int64_t>(prev) * key_value;
          output_pre_acc[v] = static_cast<int64_t>(prev) * query_value;
        }
      }
    }

    for (size_t k = 1; k < key_dim; ++k) {
      const size_t row_base = state_base + k * value_dim + v_begin;
      const uint16_t* prev_state =
          t == 0 ? recurrent_state_data + row_base : state_data + row_base;
      const int32_t key_value = key_center[k];
      const int32_t query_value = query_center[k];
      if (hvx_accum) {
        deltanetQhpiHvxAccumU16I32(
            prev_state,
            kv_acc32,
            output_pre_acc32,
            prev_q.zero_offset,
            key_value >> kQUint16HvxAccumInputShift,
            query_value >> kQUint16HvxAccumInputShift,
            tile_size);
      } else {
        for (size_t v = 0; v < tile_size; ++v) {
          const int32_t prev = deltanetQhpiCenteredU16(prev_state[v], prev_q);
          kv_acc[v] += static_cast<int64_t>(prev) * key_value;
          output_pre_acc[v] += static_cast<int64_t>(prev) * query_value;
        }
      }
    }

    const double hvx_accum_scale =
        hvx_accum ? std::ldexp(1.0, 2 * kQUint16HvxAccumInputShift) : 1.0;
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
          kv_acc32,
          output_pre_acc32,
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
          static_cast<float>(delta_update_scale *
                             std::ldexp(
                                 1.0,
                                 kQUint16StateUpdateShift -
                                     kQUint16HvxStateUpdateCoeffShift)),
          tile_size);
    }
    for (size_t v = scalar_v_begin; v < tile_size; ++v) {
      const double value_real =
          static_cast<double>(deltanetQhpiCenteredU16(value_tile[v], value_q)) *
          value_q.stepsize;
      const double kv_acc_value =
          hvx_accum ? static_cast<double>(kv_acc32[v])
                    : static_cast<double>(kv_acc[v]);
      const double delta_value =
          (value_real - kv_acc_value * kv_scale) * beta_value;
      delta_value_scratch[v] = static_cast<float>(delta_value);
      if constexpr (!kEnableQUint16HvxStateUpdateF32) {
        delta_update_fixed[v] = deltanetQhpiRoundFixed(
            delta_value * delta_update_scale, kQUint16StateUpdateShift);
      }
      if constexpr (kEnableQUint16HvxStateUpdate) {
        delta_update_fixed[v] = deltanetQhpiRoundFixed(
            delta_value * delta_update_scale, kQUint16StateUpdateShift);
        const int64_t delta_update_shifted = deltanetQhpiRoundShiftI64(
            delta_update_fixed[v], kQUint16HvxStateUpdateCoeffShift);
        if (deltanetQhpiFitsI32(delta_update_shifted)) {
          delta_update_fixed32[v] = static_cast<int32_t>(delta_update_shifted);
        } else {
          delta_update_fixed32[v] = 0;
          hvx_update = false;
        }
      }
      const double output_pre_acc_value =
          hvx_accum ? static_cast<double>(output_pre_acc32[v])
                    : static_cast<double>(output_pre_acc[v]);
      output_tile[v] = deltanetQhpiQuantU16FromScaledDouble(
          static_cast<double>(output_q.zero_offset) +
          output_pre_acc_value * output_acc_scale + delta_value *
              output_delta_scale);
    }

    for (size_t k = 0; k < key_dim; ++k) {
      const size_t row_base = state_base + k * value_dim + v_begin;
      const uint16_t* prev_state =
          t == 0 ? recurrent_state_data + row_base : state_data + row_base;
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
            static_cast<float>(
                decay * prev_q.stepsize / state_q.stepsize),
            static_cast<float>(
                static_cast<double>(key_value) * key_q.stepsize /
                state_q.stepsize),
            tile_size);
      } else {
        const int64_t key_update = key_value;
        const int64_t state_zero = state_q.zero_offset;
        const int64_t prev_zero = prev_q.zero_offset;
        constexpr int64_t kStateUpdateRounding =
            INT64_C(1) << (kQUint16StateUpdateShift - 1);
        for (size_t v = 0; v < tile_size; ++v) {
          const int64_t prev = static_cast<int64_t>(prev_state[v]) - prev_zero;
          const int64_t update =
              prev_update_fixed * prev + key_update * delta_update_fixed[v];
          const int64_t state_center =
              update >= 0
                  ? (update + kStateUpdateRounding) >>
                        kQUint16StateUpdateShift
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
}

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
    size_t value_tile_count) {
  int64_t kv_acc[kMaxDeltaNetDim] __attribute__((aligned(128)));
  int64_t output_pre_acc[kMaxDeltaNetDim] __attribute__((aligned(128)));
  int32_t kv_acc32[kMaxDeltaNetDim] __attribute__((aligned(128)));
  int32_t output_pre_acc32[kMaxDeltaNetDim] __attribute__((aligned(128)));
  int64_t delta_update_fixed[kMaxDeltaNetDim] __attribute__((aligned(128)));
  int32_t delta_update_fixed32[kMaxDeltaNetDim] __attribute__((aligned(128)));
  float delta_value_scratch[kMaxDeltaNetDim] __attribute__((aligned(128)));
  int32_t key_center[kMaxDeltaNetDim] __attribute__((aligned(128)));
  int32_t query_center[kMaxDeltaNetDim] __attribute__((aligned(128)));

  const size_t total_rows = batch * heads;
  if (value_tile_count <= 1) {
    for (size_t row = slice; row < total_rows; row += num_slices) {
      deltanetCoreQhpiRunQUint16Tile(
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
          kv_acc32,
          output_pre_acc32,
          delta_update_fixed,
          delta_update_fixed32,
          delta_value_scratch,
          key_center,
          query_center);
    }
    return;
  }

  const size_t total_tasks = total_rows * value_tile_count;
  for (size_t task = slice; task < total_tasks; task += num_slices) {
    const size_t row = task / value_tile_count;
    const size_t tile = task % value_tile_count;
    const size_t v_begin = value_dim * tile / value_tile_count;
    const size_t v_end = value_dim * (tile + 1) / value_tile_count;
    deltanetCoreQhpiRunQUint16Tile(
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
        kv_acc32,
        output_pre_acc32,
        delta_update_fixed,
        delta_update_fixed32,
        delta_value_scratch,
        key_center,
        query_center);
  }
}

static uint32_t deltanetCoreQhpiFp16(
    QHPI_RuntimeHandle* handle,
    uint32_t num_outputs,
    QHPI_Tensor** outputs,
    uint32_t num_inputs,
    const QHPI_Tensor* const* inputs) {
  if (num_inputs != 6 || num_outputs != 2) {
    return QHPI_ErrorFatal;
  }

  const QHPI_Shape query_shape = qhpi_tensor_shape(inputs[0]);
  const QHPI_Shape value_shape = qhpi_tensor_shape(inputs[2]);
  if (query_shape.rank != 4 || value_shape.rank != 4) {
    return QHPI_ErrorFatal;
  }

  const size_t batch = query_shape.dims[0];
  const size_t seq_len = query_shape.dims[1];
  const size_t heads = query_shape.dims[2];
  const size_t key_dim = query_shape.dims[3];
  const size_t value_dim = value_shape.dims[3];
  if (batch == 0 || seq_len == 0 || heads == 0 || key_dim == 0 ||
      value_dim == 0 || key_dim > kMaxDeltaNetDim ||
      value_dim > kMaxDeltaNetDim) {
    return QHPI_Unsupported;
  }

  const uint32_t num_slices = std::max<uint32_t>(qhpi_num_slices(handle), 1);
  const uint32_t slice = qhpi_slice_number(handle);
  const size_t value_tile_count = deltanetQhpiValueTileCount(
      batch * heads, num_slices, value_dim, 128 / sizeof(Float16));
  deltanetCoreQhpiRunFp16(
      static_cast<Float16*>(qhpi_tensor_raw_data(outputs[0])),
      static_cast<Float16*>(qhpi_tensor_raw_data(outputs[1])),
      static_cast<const Float16*>(qhpi_tensor_raw_data(inputs[0])),
      static_cast<const Float16*>(qhpi_tensor_raw_data(inputs[1])),
      static_cast<const Float16*>(qhpi_tensor_raw_data(inputs[2])),
      static_cast<const Float16*>(qhpi_tensor_raw_data(inputs[3])),
      static_cast<const Float16*>(qhpi_tensor_raw_data(inputs[4])),
      static_cast<const Float16*>(qhpi_tensor_raw_data(inputs[5])),
      batch,
      seq_len,
      heads,
      key_dim,
      value_dim,
      slice,
      num_slices,
      value_tile_count);
  return QHPI_Success;
}

static uint32_t deltanetCoreQhpiFp32(
    QHPI_RuntimeHandle* handle,
    uint32_t num_outputs,
    QHPI_Tensor** outputs,
    uint32_t num_inputs,
    const QHPI_Tensor* const* inputs) {
  if (num_inputs != 6 || num_outputs != 2) {
    return QHPI_ErrorFatal;
  }

  const QHPI_Shape query_shape = qhpi_tensor_shape(inputs[0]);
  const QHPI_Shape value_shape = qhpi_tensor_shape(inputs[2]);
  if (query_shape.rank != 4 || value_shape.rank != 4) {
    return QHPI_ErrorFatal;
  }

  const size_t batch = query_shape.dims[0];
  const size_t seq_len = query_shape.dims[1];
  const size_t heads = query_shape.dims[2];
  const size_t key_dim = query_shape.dims[3];
  const size_t value_dim = value_shape.dims[3];
  if (batch == 0 || seq_len == 0 || heads == 0 || key_dim == 0 ||
      value_dim == 0 || key_dim > kMaxDeltaNetDim ||
      value_dim > kMaxDeltaNetDim) {
    return QHPI_Unsupported;
  }

  const uint32_t num_slices = std::max<uint32_t>(qhpi_num_slices(handle), 1);
  const uint32_t slice = qhpi_slice_number(handle);
  const size_t value_tile_count = deltanetQhpiValueTileCount(
      batch * heads, num_slices, value_dim, 128 / sizeof(float));
  deltanetCoreQhpiRunFp32(
      static_cast<float*>(qhpi_tensor_raw_data(outputs[0])),
      static_cast<float*>(qhpi_tensor_raw_data(outputs[1])),
      static_cast<const float*>(qhpi_tensor_raw_data(inputs[0])),
      static_cast<const float*>(qhpi_tensor_raw_data(inputs[1])),
      static_cast<const float*>(qhpi_tensor_raw_data(inputs[2])),
      static_cast<const float*>(qhpi_tensor_raw_data(inputs[3])),
      static_cast<const float*>(qhpi_tensor_raw_data(inputs[4])),
      static_cast<const float*>(qhpi_tensor_raw_data(inputs[5])),
      batch,
      seq_len,
      heads,
      key_dim,
      value_dim,
      slice,
      num_slices,
      value_tile_count);
  return QHPI_Success;
}

static uint32_t deltanetCoreQhpiQUint16(
    QHPI_RuntimeHandle* handle,
    uint32_t num_outputs,
    QHPI_Tensor** outputs,
    uint32_t num_inputs,
    const QHPI_Tensor* const* inputs) {
  if (num_inputs != 6 || num_outputs != 2) {
    return QHPI_ErrorFatal;
  }

  const QHPI_Shape query_shape = qhpi_tensor_shape(inputs[0]);
  const QHPI_Shape key_shape = qhpi_tensor_shape(inputs[1]);
  const QHPI_Shape value_shape = qhpi_tensor_shape(inputs[2]);
  const QHPI_Shape decay_shape = qhpi_tensor_shape(inputs[3]);
  const QHPI_Shape beta_shape = qhpi_tensor_shape(inputs[4]);
  const QHPI_Shape recurrent_shape = qhpi_tensor_shape(inputs[5]);
  if (query_shape.rank != 4 || key_shape.rank != 4 ||
      value_shape.rank != 4 || recurrent_shape.rank != 4 ||
      (decay_shape.rank != 3 && decay_shape.rank != 4) ||
      (beta_shape.rank != 3 && beta_shape.rank != 4)) {
    return QHPI_ErrorFatal;
  }

  const size_t batch = query_shape.dims[0];
  const size_t seq_len = query_shape.dims[1];
  const size_t heads = query_shape.dims[2];
  const size_t key_dim = query_shape.dims[3];
  const size_t value_dim = value_shape.dims[3];
  if (batch == 0 || seq_len == 0 || heads == 0 || key_dim == 0 ||
      value_dim == 0 || key_dim > kMaxDeltaNetDim ||
      value_dim > kMaxDeltaNetDim) {
    return QHPI_Unsupported;
  }
  if (key_shape.dims[0] != batch || key_shape.dims[1] != seq_len ||
      key_shape.dims[2] != heads || key_shape.dims[3] != key_dim ||
      value_shape.dims[0] != batch || value_shape.dims[1] != seq_len ||
      value_shape.dims[2] != heads || recurrent_shape.dims[0] != batch ||
      recurrent_shape.dims[1] != heads ||
      recurrent_shape.dims[2] != key_dim ||
      recurrent_shape.dims[3] != value_dim ||
      !deltanetQhpiHasTimeHeadLayout(decay_shape, batch, seq_len, heads) ||
      !deltanetQhpiHasTimeHeadLayout(beta_shape, batch, seq_len, heads)) {
    return QHPI_ErrorFatal;
  }

  const QHPI_Quant_Parameters query_q = qhpi_tensor_quant_parameters(inputs[0]);
  const QHPI_Quant_Parameters key_q = qhpi_tensor_quant_parameters(inputs[1]);
  const QHPI_Quant_Parameters value_q = qhpi_tensor_quant_parameters(inputs[2]);
  const QHPI_Quant_Parameters decay_q = qhpi_tensor_quant_parameters(inputs[3]);
  const QHPI_Quant_Parameters beta_q = qhpi_tensor_quant_parameters(inputs[4]);
  const QHPI_Quant_Parameters recurrent_state_q =
      qhpi_tensor_quant_parameters(inputs[5]);
  const QHPI_Quant_Parameters output_q =
      qhpi_tensor_quant_parameters(outputs[0]);
  const QHPI_Quant_Parameters state_q =
      qhpi_tensor_quant_parameters(outputs[1]);
  if (!deltanetQhpiIsValidQuant(query_q) ||
      !deltanetQhpiIsValidQuant(key_q) ||
      !deltanetQhpiIsValidQuant(value_q) ||
      !deltanetQhpiIsValidQuant(decay_q) ||
      !deltanetQhpiIsValidQuant(beta_q) ||
      !deltanetQhpiIsValidQuant(recurrent_state_q) ||
      !deltanetQhpiIsValidQuant(output_q) ||
      !deltanetQhpiIsValidQuant(state_q)) {
    return QHPI_ErrorFatal;
  }

  const uint32_t num_slices = std::max<uint32_t>(qhpi_num_slices(handle), 1);
  const uint32_t slice = qhpi_slice_number(handle);
  const size_t value_tile_count = deltanetQhpiValueTileCount(
      batch * heads, num_slices, value_dim, 128 / sizeof(uint16_t));
  deltanetCoreQhpiRunQUint16(
      static_cast<uint16_t*>(qhpi_tensor_raw_data(outputs[0])),
      static_cast<uint16_t*>(qhpi_tensor_raw_data(outputs[1])),
      static_cast<const uint16_t*>(qhpi_tensor_raw_data(inputs[0])),
      static_cast<const uint16_t*>(qhpi_tensor_raw_data(inputs[1])),
      static_cast<const uint16_t*>(qhpi_tensor_raw_data(inputs[2])),
      static_cast<const uint16_t*>(qhpi_tensor_raw_data(inputs[3])),
      static_cast<const uint16_t*>(qhpi_tensor_raw_data(inputs[4])),
      static_cast<const uint16_t*>(qhpi_tensor_raw_data(inputs[5])),
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
      batch,
      seq_len,
      heads,
      key_dim,
      value_dim,
      slice,
      num_slices,
      value_tile_count);
  return QHPI_Success;
}

static uint32_t deltanetCoreQhpiFp16AccumF16(
    QHPI_RuntimeHandle* handle,
    uint32_t num_outputs,
    QHPI_Tensor** outputs,
    uint32_t num_inputs,
    const QHPI_Tensor* const* inputs) {
  if (num_inputs != 6 || num_outputs != 2) {
    return QHPI_ErrorFatal;
  }

  const QHPI_Shape query_shape = qhpi_tensor_shape(inputs[0]);
  const QHPI_Shape value_shape = qhpi_tensor_shape(inputs[2]);
  if (query_shape.rank != 4 || value_shape.rank != 4) {
    return QHPI_ErrorFatal;
  }

  const size_t batch = query_shape.dims[0];
  const size_t seq_len = query_shape.dims[1];
  const size_t heads = query_shape.dims[2];
  const size_t key_dim = query_shape.dims[3];
  const size_t value_dim = value_shape.dims[3];
  if (batch == 0 || seq_len == 0 || heads == 0 || key_dim == 0 ||
      value_dim == 0 || key_dim > kMaxDeltaNetDim ||
      value_dim > kMaxDeltaNetDim) {
    return QHPI_Unsupported;
  }

  const uint32_t num_slices = std::max<uint32_t>(qhpi_num_slices(handle), 1);
  const uint32_t slice = qhpi_slice_number(handle);
  const size_t value_tile_count = deltanetQhpiValueTileCount(
      batch * heads, num_slices, value_dim, 128 / sizeof(Float16));
  deltanetCoreQhpiRunFp16AccumF16(
      static_cast<Float16*>(qhpi_tensor_raw_data(outputs[0])),
      static_cast<Float16*>(qhpi_tensor_raw_data(outputs[1])),
      static_cast<const Float16*>(qhpi_tensor_raw_data(inputs[0])),
      static_cast<const Float16*>(qhpi_tensor_raw_data(inputs[1])),
      static_cast<const Float16*>(qhpi_tensor_raw_data(inputs[2])),
      static_cast<const Float16*>(qhpi_tensor_raw_data(inputs[3])),
      static_cast<const Float16*>(qhpi_tensor_raw_data(inputs[4])),
      static_cast<const Float16*>(qhpi_tensor_raw_data(inputs[5])),
      batch,
      seq_len,
      heads,
      key_dim,
      value_dim,
      slice,
      num_slices,
      value_tile_count);
  return QHPI_Success;
}

static QHPI_Tensor_Signature_v1 deltanet_core_fp16_inputs[] = {
    {QHPI_Float16, QHPI_Layout_Any, QHPI_Storage_Direct, QHPI_MemLoc_DDR_Only},
    {QHPI_Float16, QHPI_Layout_Any, QHPI_Storage_Direct, QHPI_MemLoc_DDR_Only},
    {QHPI_Float16, QHPI_Layout_Any, QHPI_Storage_Direct, QHPI_MemLoc_DDR_Only},
    {QHPI_Float16, QHPI_Layout_Any, QHPI_Storage_Direct, QHPI_MemLoc_DDR_Only},
    {QHPI_Float16, QHPI_Layout_Any, QHPI_Storage_Direct, QHPI_MemLoc_DDR_Only},
    {QHPI_Float16, QHPI_Layout_Any, QHPI_Storage_Direct, QHPI_MemLoc_DDR_Only},
};

static QHPI_Tensor_Signature_v1 deltanet_core_fp16_outputs[] = {
    {QHPI_Float16, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_DDR_Only},
    {QHPI_Float16, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_DDR_Only},
};

static QHPI_Tensor_Signature_v1 deltanet_core_fp32_inputs[] = {
    {QHPI_Float32, QHPI_Layout_Any, QHPI_Storage_Direct, QHPI_MemLoc_DDR_Only},
    {QHPI_Float32, QHPI_Layout_Any, QHPI_Storage_Direct, QHPI_MemLoc_DDR_Only},
    {QHPI_Float32, QHPI_Layout_Any, QHPI_Storage_Direct, QHPI_MemLoc_DDR_Only},
    {QHPI_Float32, QHPI_Layout_Any, QHPI_Storage_Direct, QHPI_MemLoc_DDR_Only},
    {QHPI_Float32, QHPI_Layout_Any, QHPI_Storage_Direct, QHPI_MemLoc_DDR_Only},
    {QHPI_Float32, QHPI_Layout_Any, QHPI_Storage_Direct, QHPI_MemLoc_DDR_Only},
};

static QHPI_Tensor_Signature_v1 deltanet_core_fp32_outputs[] = {
    {QHPI_Float32, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_DDR_Only},
    {QHPI_Float32, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_DDR_Only},
};

static QHPI_Tensor_Signature_v1 deltanet_core_quint16_inputs[] = {
    {QHPI_QUInt16, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_DDR_Only},
    {QHPI_QUInt16, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_DDR_Only},
    {QHPI_QUInt16, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_DDR_Only},
    {QHPI_QUInt16, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_DDR_Only},
    {QHPI_QUInt16, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_DDR_Only},
    {QHPI_QUInt16, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_DDR_Only},
};

static QHPI_Tensor_Signature_v1 deltanet_core_quint16_outputs[] = {
    {QHPI_QUInt16, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_DDR_Only},
    {QHPI_QUInt16, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_DDR_Only},
};

static QHPI_Kernel_v1 deltanet_core_kernels[] = {
    {.function_name = THIS_PKG_NAME_STR "::DeltaNetCore.fp16.hvx",
     .function = deltanetCoreQhpiFp16,
     .resources = QHPI_RESOURCE_HVX,
     .source_destructive = false,
     .multithreaded = true,
     .variable_inputs = false,
     .variable_outputs = false,
     .min_inputs = 6,
     .input_signature = deltanet_core_fp16_inputs,
     .min_outputs = 2,
     .output_signature = deltanet_core_fp16_outputs,
     .cost_function = deltanetCoreQhpiCost,
     .sync_block_size = 0,
     .precomputed_data_size = 0,
     .do_precomputation_function = nullptr,
     .function_with_precomputed_data = nullptr,
     .predicate = nullptr},
    {.function_name = THIS_PKG_NAME_STR "::DeltaNetCore.fp32.hvx",
     .function = deltanetCoreQhpiFp32,
     .resources = QHPI_RESOURCE_HVX,
     .source_destructive = false,
     .multithreaded = true,
     .variable_inputs = false,
     .variable_outputs = false,
     .min_inputs = 6,
     .input_signature = deltanet_core_fp32_inputs,
     .min_outputs = 2,
     .output_signature = deltanet_core_fp32_outputs,
     .cost_function = deltanetCoreQhpiCost,
     .sync_block_size = 0,
     .precomputed_data_size = 0,
     .do_precomputation_function = nullptr,
     .function_with_precomputed_data = nullptr,
     .predicate = nullptr},
    {.function_name = THIS_PKG_NAME_STR "::DeltaNetCore.quint16.fixed",
     .function = deltanetCoreQhpiQUint16,
     .resources = QHPI_RESOURCE_HVX,
     .source_destructive = false,
     .multithreaded = true,
     .variable_inputs = false,
     .variable_outputs = false,
     .min_inputs = 6,
     .input_signature = deltanet_core_quint16_inputs,
     .min_outputs = 2,
     .output_signature = deltanet_core_quint16_outputs,
     .cost_function = deltanetCoreQhpiCost,
     .sync_block_size = 0,
     .precomputed_data_size = 0,
     .do_precomputation_function = nullptr,
     .function_with_precomputed_data = nullptr,
     .predicate = nullptr},
};

static QHPI_Kernel_v1 deltanet_core_fp16_accum_kernels[] = {
    {.function_name = THIS_PKG_NAME_STR "::DeltaNetCoreFp16Accum.fp16.hvx",
     .function = deltanetCoreQhpiFp16AccumF16,
     .resources = QHPI_RESOURCE_HVX,
     .source_destructive = false,
     .multithreaded = true,
     .variable_inputs = false,
     .variable_outputs = false,
     .min_inputs = 6,
     .input_signature = deltanet_core_fp16_inputs,
     .min_outputs = 2,
     .output_signature = deltanet_core_fp16_outputs,
     .cost_function = deltanetCoreQhpiCost,
     .sync_block_size = 0,
     .precomputed_data_size = 0,
     .do_precomputation_function = nullptr,
     .function_with_precomputed_data = nullptr,
     .predicate = nullptr},
};

static QHPI_OpInfo_v1 deltanet_qhpi_ops[] = {
    {.name = THIS_PKG_NAME_STR "::DeltaNetCore",
     .num_kernels = sizeof(deltanet_core_kernels) /
         sizeof(deltanet_core_kernels[0]),
     .kernels = deltanet_core_kernels,
     .early_rewrite = nullptr,
     .shape_required = nullptr,
     .shape_legalized = nullptr,
     .build_tile = nullptr,
     .late_rewrite = nullptr},
    {.name = THIS_PKG_NAME_STR "::DeltaNetCoreFp16Accum",
     .num_kernels = sizeof(deltanet_core_fp16_accum_kernels) /
         sizeof(deltanet_core_fp16_accum_kernels[0]),
     .kernels = deltanet_core_fp16_accum_kernels,
     .early_rewrite = nullptr,
     .shape_required = nullptr,
     .shape_legalized = nullptr,
     .build_tile = nullptr,
     .late_rewrite = nullptr},
};

void register_deltanet_qhpi_ops() {
  qhpi_register_ops_v1(
      sizeof(deltanet_qhpi_ops) / sizeof(deltanet_qhpi_ops[0]),
      deltanet_qhpi_ops,
      THIS_PKG_NAME_STR);
}

#else

void register_deltanet_qhpi_ops() {}

#endif
