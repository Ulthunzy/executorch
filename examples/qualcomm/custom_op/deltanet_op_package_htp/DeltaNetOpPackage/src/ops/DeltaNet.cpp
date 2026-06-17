//==============================================================================
// Auto Generated Code for DeltaNetOpPackage
//==============================================================================

#include "HTP/core/constraints.h"
#include "HTP/core/float16.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"
#include "QnnOpPackage.h"
#ifdef __hexagon__
#include "HTP/core/intrinsics.h"
#endif

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <limits>

BEGIN_PKG_OP_DEFINITION(PKG_DeltaNet);

template <typename TensorType>
GraphStatus deltanetImpl(
    TensorType& core_attn_out,
    TensorType& new_recurrent_state,
    const TensorType& query,
    const TensorType& key,
    const TensorType& value,
    const TensorType& g,
    const TensorType& beta,
    const TensorType& recurrent_state,
    const Tensor& scaling,
    const Tensor& epsilon);

template <typename TensorType>
GraphStatus deltanetPackedImpl(
    TensorType& packed_out,
    const TensorType& query,
    const TensorType& key,
    const TensorType& value,
    const TensorType& g,
    const TensorType& beta,
    const TensorType& recurrent_state);

template <typename TensorType>
GraphStatus deltanetCoreImpl(
    TensorType& core_attn_out,
    TensorType& new_recurrent_state,
    const TensorType& query,
    const TensorType& key,
    const TensorType& value,
    const TensorType& g,
    const TensorType& beta,
    const TensorType& recurrent_state);

template <typename TensorType>
GraphStatus deltanetIdentityImpl(
    TensorType& output,
    const TensorType& input);

template <typename TensorType>
GraphStatus deltanetDecayStateImpl(
    TensorType& output,
    const TensorType& recurrent_state,
    const TensorType& g);

template <typename TensorType>
GraphStatus deltanetStatePassthroughImpl(
    TensorType& output,
    const TensorType& recurrent_state,
    const TensorType& g);

static float deltanetCostFunc(const Op* op);

static constexpr size_t kMaxDeltaNetDim = 256;
[[maybe_unused]] static constexpr bool kUseDeltaNetAr1SpecializedPath = false;
[[maybe_unused]] static constexpr bool kUseDeltaNetBlock4Hvx = false;

DEF_PACKAGE_OP((deltanetImpl<Tensor>), "DeltaNet")
DEF_PACKAGE_OP((deltanetPackedImpl<Tensor>), "DeltaNetPacked")
DEF_PACKAGE_OP((deltanetIdentityImpl<Tensor>), "DeltaNetIdentity")
DEF_PACKAGE_OP((deltanetDecayStateImpl<Tensor>), "DeltaNetDecayState")
DEF_PACKAGE_OP(
    (deltanetStatePassthroughImpl<Tensor>),
    "DeltaNetStatePassthrough")

DEF_PACKAGE_PARAM_ORDER(
    "DeltaNet",
    "scaling",
    true,
    nullptr,
    "epsilon",
    true,
    nullptr)

static bool checkRank(const Tensor& tensor, size_t rank) {
  return tensor.rank() == rank;
}

static bool checkDim(const Tensor& tensor, size_t dim, size_t expected) {
  return tensor.dim(dim) == expected;
}

static float readScalarFloat32(const Tensor& tensor) {
  return static_cast<float>(tensor(0, 0, 0, 0));
}

static float readTimeHeadValue(
    const Tensor& tensor,
    size_t batch,
    size_t token,
    size_t head) {
  if (tensor.rank() == 3) {
    if (tensor.dim(1) > token && tensor.dim(2) > head) {
      return static_cast<float>(tensor(batch, token, head));
    }
    if (tensor.dim(1) > head && tensor.dim(2) > token) {
      return static_cast<float>(tensor(batch, head, token));
    }
  }
  if (tensor.rank() == 4) {
    if (tensor.dim(1) > token && tensor.dim(2) == 1 &&
        tensor.dim(3) > head) {
      return static_cast<float>(tensor(batch, token, 0, head));
    }
    if (tensor.dim(1) == 1 && tensor.dim(2) > token &&
        tensor.dim(3) > head) {
      return static_cast<float>(tensor(batch, 0, token, head));
    }
    if (tensor.dim(1) > token && tensor.dim(2) > head &&
        tensor.dim(3) == 1) {
      return static_cast<float>(tensor(batch, token, head, 0));
    }
    if (tensor.dim(1) > head && tensor.dim(2) > token &&
        tensor.dim(3) == 1) {
      return static_cast<float>(tensor(batch, head, token, 0));
    }
  }
  return 0.0f;
}

static size_t packedOutputIndex(
    size_t b,
    size_t t,
    size_t h,
    size_t v,
    size_t seq_len,
    size_t heads,
    size_t value_dim) {
  return (((b * seq_len + t) * heads + h) * value_dim + v);
}

static size_t packedStateIndex(
    size_t output_numel,
    size_t b,
    size_t h,
    size_t k,
    size_t v,
    size_t heads,
    size_t key_dim,
    size_t value_dim) {
  return output_numel + (((b * heads + h) * key_dim + k) * value_dim + v);
}

static bool isFloat32Dtype(const Tensor& tensor) {
  return tensor.get_dtype_intfc().dtype == DType::Float32;
}

static bool isFloat16Dtype(const Tensor& tensor) {
  return tensor.get_dtype_intfc().dtype == DType::Float16;
}

static bool isQUInt16Dtype(const Tensor& tensor) {
  return tensor.get_dtype_intfc().dtype == DType::QUInt16;
}

static bool isQUInt8Dtype(const Tensor& tensor) {
  return tensor.get_dtype_intfc().dtype == DType::QUInt8;
}

template <typename Scalar>
static float readRawScalar(const Scalar* data, size_t index) {
  return static_cast<float>(data[index]);
}

template <typename Scalar>
static void writeRawScalar(Scalar* data, size_t index, float value) {
  data[index] = static_cast<Scalar>(value);
}

template <typename Scalar>
static float readQuantRawScalar(
    const Scalar* data,
    const DTypeScaleOff& intfc,
    size_t index) {
  return (static_cast<float>(data[index]) -
          static_cast<float>(intfc.offset)) *
      intfc.scale;
}

template <typename Scalar>
static void writeQuantRawScalar(
    Scalar* data,
    const DTypeScaleOff& intfc,
    size_t index,
    float value) {
  const float scaled =
      std::round(static_cast<float>(intfc.offset) + value / intfc.scale);
  const float clamped = std::min(
      std::max(scaled, static_cast<float>(std::numeric_limits<Scalar>::min())),
      static_cast<float>(std::numeric_limits<Scalar>::max()));
  data[index] = static_cast<Scalar>(clamped);
}

#ifdef __hexagon__
static inline HVX_Vector deltanetHvxMulF32(HVX_Vector a, HVX_Vector b) {
#if __HVX_ARCH__ < 79
  return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b));
#else
  return Q6_Vsf_vmpy_VsfVsf(a, b);
#endif
}

static inline HVX_Vector deltanetHvxAddF32(HVX_Vector a, HVX_Vector b) {
#if __HVX_ARCH__ < 79
  return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b));
#else
  return Q6_Vsf_vadd_VsfVsf(a, b);
#endif
}

static inline HVX_Vector deltanetHvxSplatF32(float value) {
  return q6op_V_vsplat_float32(value);
}

static inline HVX_VectorPair deltanetHvxF32PairToQF32(
    HVX_Vector lo,
    HVX_Vector hi) {
  return Q6_W_vcombine_VV(Q6_Vqf32_equals_Vsf(hi), Q6_Vqf32_equals_Vsf(lo));
}

static inline void deltanetHvxAccumF32(
    const float* prev_state,
    float* kv_mem,
    float* output_pre,
    float key_scale,
    float query_scale,
    size_t value_dim) {
  const size_t epv = 128 / sizeof(float);
  const size_t vec_count = value_dim / epv;
  const HVX_Vector v_key = deltanetHvxSplatF32(key_scale);
  const HVX_Vector v_query = deltanetHvxSplatF32(query_scale);

  for (size_t i = 0; i < vec_count; ++i) {
    const size_t offset = i * epv;
    const HVX_Vector v_prev = vmemu(prev_state + offset);
    const HVX_Vector v_kv =
        deltanetHvxAddF32(vmemu(kv_mem + offset), deltanetHvxMulF32(v_prev, v_key));
    const HVX_Vector v_out = deltanetHvxAddF32(
        vmemu(output_pre + offset), deltanetHvxMulF32(v_prev, v_query));
    vmemu(kv_mem + offset) = v_kv;
    vmemu(output_pre + offset) = v_out;
  }

  for (size_t v = vec_count * epv; v < value_dim; ++v) {
    kv_mem[v] += prev_state[v] * key_scale;
    output_pre[v] += prev_state[v] * query_scale;
  }
}

static inline void deltanetHvxAccum4F32(
    const float* prev0,
    const float* prev1,
    const float* prev2,
    const float* prev3,
    float* kv_mem,
    float* output_pre,
    const float* key_scales,
    const float* query_scales,
    size_t value_dim) {
  const size_t epv = 128 / sizeof(float);
  const size_t vec_count = value_dim / epv;
  const HVX_Vector v_key0 = deltanetHvxSplatF32(key_scales[0]);
  const HVX_Vector v_key1 = deltanetHvxSplatF32(key_scales[1]);
  const HVX_Vector v_key2 = deltanetHvxSplatF32(key_scales[2]);
  const HVX_Vector v_key3 = deltanetHvxSplatF32(key_scales[3]);
  const HVX_Vector v_query0 = deltanetHvxSplatF32(query_scales[0]);
  const HVX_Vector v_query1 = deltanetHvxSplatF32(query_scales[1]);
  const HVX_Vector v_query2 = deltanetHvxSplatF32(query_scales[2]);
  const HVX_Vector v_query3 = deltanetHvxSplatF32(query_scales[3]);

  for (size_t i = 0; i < vec_count; ++i) {
    const size_t offset = i * epv;
    const HVX_Vector v_prev0 = vmemu(prev0 + offset);
    const HVX_Vector v_prev1 = vmemu(prev1 + offset);
    const HVX_Vector v_prev2 = vmemu(prev2 + offset);
    const HVX_Vector v_prev3 = vmemu(prev3 + offset);
    HVX_Vector v_kv = vmemu(kv_mem + offset);
    HVX_Vector v_out = vmemu(output_pre + offset);

    v_kv = deltanetHvxAddF32(v_kv, deltanetHvxMulF32(v_prev0, v_key0));
    v_kv = deltanetHvxAddF32(v_kv, deltanetHvxMulF32(v_prev1, v_key1));
    v_kv = deltanetHvxAddF32(v_kv, deltanetHvxMulF32(v_prev2, v_key2));
    v_kv = deltanetHvxAddF32(v_kv, deltanetHvxMulF32(v_prev3, v_key3));
    v_out = deltanetHvxAddF32(v_out, deltanetHvxMulF32(v_prev0, v_query0));
    v_out = deltanetHvxAddF32(v_out, deltanetHvxMulF32(v_prev1, v_query1));
    v_out = deltanetHvxAddF32(v_out, deltanetHvxMulF32(v_prev2, v_query2));
    v_out = deltanetHvxAddF32(v_out, deltanetHvxMulF32(v_prev3, v_query3));

    vmemu(kv_mem + offset) = v_kv;
    vmemu(output_pre + offset) = v_out;
  }

  for (size_t v = vec_count * epv; v < value_dim; ++v) {
    kv_mem[v] += prev0[v] * key_scales[0] + prev1[v] * key_scales[1] +
        prev2[v] * key_scales[2] + prev3[v] * key_scales[3];
    output_pre[v] += prev0[v] * query_scales[0] +
        prev1[v] * query_scales[1] + prev2[v] * query_scales[2] +
        prev3[v] * query_scales[3];
  }
}

static inline void deltanetHvxUpdateStateF32(
    const float* prev_state,
    float* state,
    const float* delta,
    float decay,
    float key_value,
    size_t value_dim) {
  const size_t epv = 128 / sizeof(float);
  const size_t vec_count = value_dim / epv;
  const HVX_Vector v_decay = deltanetHvxSplatF32(decay);
  const HVX_Vector v_key = deltanetHvxSplatF32(key_value);

  for (size_t i = 0; i < vec_count; ++i) {
    const size_t offset = i * epv;
    const HVX_Vector v_prev = vmemu(prev_state + offset);
    const HVX_Vector v_delta = vmemu(delta + offset);
    const HVX_Vector v_updated = deltanetHvxAddF32(
        deltanetHvxMulF32(v_prev, v_decay),
        deltanetHvxMulF32(v_delta, v_key));
    vmemu(state + offset) = v_updated;
  }

  for (size_t v = vec_count * epv; v < value_dim; ++v) {
    state[v] = prev_state[v] * decay + delta[v] * key_value;
  }
}

static inline void deltanetHvxUpdateState4F32(
    const float* prev0,
    const float* prev1,
    const float* prev2,
    const float* prev3,
    float* state0,
    float* state1,
    float* state2,
    float* state3,
    const float* delta,
    float decay,
    const float* key_values,
    size_t value_dim) {
  const size_t epv = 128 / sizeof(float);
  const size_t vec_count = value_dim / epv;
  const HVX_Vector v_decay = deltanetHvxSplatF32(decay);
  const HVX_Vector v_key0 = deltanetHvxSplatF32(key_values[0]);
  const HVX_Vector v_key1 = deltanetHvxSplatF32(key_values[1]);
  const HVX_Vector v_key2 = deltanetHvxSplatF32(key_values[2]);
  const HVX_Vector v_key3 = deltanetHvxSplatF32(key_values[3]);

  for (size_t i = 0; i < vec_count; ++i) {
    const size_t offset = i * epv;
    const HVX_Vector v_delta = vmemu(delta + offset);
    vmemu(state0 + offset) = deltanetHvxAddF32(
        deltanetHvxMulF32(vmemu(prev0 + offset), v_decay),
        deltanetHvxMulF32(v_delta, v_key0));
    vmemu(state1 + offset) = deltanetHvxAddF32(
        deltanetHvxMulF32(vmemu(prev1 + offset), v_decay),
        deltanetHvxMulF32(v_delta, v_key1));
    vmemu(state2 + offset) = deltanetHvxAddF32(
        deltanetHvxMulF32(vmemu(prev2 + offset), v_decay),
        deltanetHvxMulF32(v_delta, v_key2));
    vmemu(state3 + offset) = deltanetHvxAddF32(
        deltanetHvxMulF32(vmemu(prev3 + offset), v_decay),
        deltanetHvxMulF32(v_delta, v_key3));
  }

  for (size_t v = vec_count * epv; v < value_dim; ++v) {
    state0[v] = prev0[v] * decay + delta[v] * key_values[0];
    state1[v] = prev1[v] * decay + delta[v] * key_values[1];
    state2[v] = prev2[v] * decay + delta[v] * key_values[2];
    state3[v] = prev3[v] * decay + delta[v] * key_values[3];
  }
}

static inline void deltanetHvxWriteOutputF32(
    float* output,
    const float* output_pre,
    const float* delta,
    float key_query_dot,
    size_t value_dim) {
  const size_t epv = 128 / sizeof(float);
  const size_t vec_count = value_dim / epv;
  const HVX_Vector v_dot = deltanetHvxSplatF32(key_query_dot);

  for (size_t i = 0; i < vec_count; ++i) {
    const size_t offset = i * epv;
    const HVX_Vector v_out = deltanetHvxAddF32(
        vmemu(output_pre + offset),
        deltanetHvxMulF32(vmemu(delta + offset), v_dot));
    vmemu(output + offset) = v_out;
  }

  for (size_t v = vec_count * epv; v < value_dim; ++v) {
    output[v] = output_pre[v] + delta[v] * key_query_dot;
  }
}

static inline void deltanetHvxDeltaAndWriteOutputF32(
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
  const HVX_Vector v_beta = deltanetHvxSplatF32(beta_value);
  const HVX_Vector v_dot = deltanetHvxSplatF32(key_query_dot);

  for (size_t i = 0; i < vec_count; ++i) {
    const size_t offset = i * epv;
    const HVX_Vector v_delta = deltanetHvxMulF32(
        deltanetHvxAddF32(
            vmemu(value + offset),
            deltanetHvxMulF32(vmemu(kv_mem + offset), v_neg_one)),
        v_beta);
    const HVX_Vector v_out = deltanetHvxAddF32(
        vmemu(output_pre + offset), deltanetHvxMulF32(v_delta, v_dot));
    vmemu(delta + offset) = v_delta;
    vmemu(output + offset) = v_out;
  }

  for (size_t v = vec_count * epv; v < value_dim; ++v) {
    const float delta_value = (value[v] - kv_mem[v]) * beta_value;
    delta[v] = delta_value;
    output[v] = output_pre[v] + delta_value * key_query_dot;
  }
}

static inline void deltanetHvxAccumF16(
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
  const HVX_Vector v_key = deltanetHvxSplatF32(key_scale);
  const HVX_Vector v_query = deltanetHvxSplatF32(query_scale);

  for (size_t i = 0; i < vec_count; ++i) {
    const size_t offset = i * halfs_per_vec;
    const HVX_Vector v_half = vmemu(prev_state + offset);
    const HVX_VectorPair v_prev_q = Q6_Wqf32_vmpy_VhfVhf(v_half, v_one);
    const HVX_Vector v_prev_lo = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(v_prev_q));
    const HVX_Vector v_prev_hi = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(v_prev_q));

    const HVX_Vector v_kv_lo = deltanetHvxAddF32(
        vmemu(kv_mem + offset), deltanetHvxMulF32(v_prev_lo, v_key));
    const HVX_Vector v_kv_hi = deltanetHvxAddF32(
        vmemu(kv_mem + offset + floats_per_vec),
        deltanetHvxMulF32(v_prev_hi, v_key));
    const HVX_Vector v_out_lo = deltanetHvxAddF32(
        vmemu(output_pre + offset), deltanetHvxMulF32(v_prev_lo, v_query));
    const HVX_Vector v_out_hi = deltanetHvxAddF32(
        vmemu(output_pre + offset + floats_per_vec),
        deltanetHvxMulF32(v_prev_hi, v_query));

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

static inline void deltanetHvxAccum4F16(
    const Float16* prev0,
    const Float16* prev1,
    const Float16* prev2,
    const Float16* prev3,
    float* kv_mem,
    float* output_pre,
    const float* key_scales,
    const float* query_scales,
    size_t value_dim) {
  const size_t halfs_per_vec = 128 / sizeof(Float16);
  const size_t floats_per_vec = 128 / sizeof(float);
  const size_t vec_count = value_dim / halfs_per_vec;
  const HVX_Vector v_one = Q6_Vh_vsplat_R(0x3C00);
  const HVX_Vector v_key0 = deltanetHvxSplatF32(key_scales[0]);
  const HVX_Vector v_key1 = deltanetHvxSplatF32(key_scales[1]);
  const HVX_Vector v_key2 = deltanetHvxSplatF32(key_scales[2]);
  const HVX_Vector v_key3 = deltanetHvxSplatF32(key_scales[3]);
  const HVX_Vector v_query0 = deltanetHvxSplatF32(query_scales[0]);
  const HVX_Vector v_query1 = deltanetHvxSplatF32(query_scales[1]);
  const HVX_Vector v_query2 = deltanetHvxSplatF32(query_scales[2]);
  const HVX_Vector v_query3 = deltanetHvxSplatF32(query_scales[3]);

  for (size_t i = 0; i < vec_count; ++i) {
    const size_t offset = i * halfs_per_vec;
    const HVX_VectorPair v_prev0_q =
        Q6_Wqf32_vmpy_VhfVhf(vmemu(prev0 + offset), v_one);
    const HVX_VectorPair v_prev1_q =
        Q6_Wqf32_vmpy_VhfVhf(vmemu(prev1 + offset), v_one);
    const HVX_VectorPair v_prev2_q =
        Q6_Wqf32_vmpy_VhfVhf(vmemu(prev2 + offset), v_one);
    const HVX_VectorPair v_prev3_q =
        Q6_Wqf32_vmpy_VhfVhf(vmemu(prev3 + offset), v_one);

    const HVX_Vector v_prev0_lo =
        Q6_Vsf_equals_Vqf32(Q6_V_lo_W(v_prev0_q));
    const HVX_Vector v_prev0_hi =
        Q6_Vsf_equals_Vqf32(Q6_V_hi_W(v_prev0_q));
    const HVX_Vector v_prev1_lo =
        Q6_Vsf_equals_Vqf32(Q6_V_lo_W(v_prev1_q));
    const HVX_Vector v_prev1_hi =
        Q6_Vsf_equals_Vqf32(Q6_V_hi_W(v_prev1_q));
    const HVX_Vector v_prev2_lo =
        Q6_Vsf_equals_Vqf32(Q6_V_lo_W(v_prev2_q));
    const HVX_Vector v_prev2_hi =
        Q6_Vsf_equals_Vqf32(Q6_V_hi_W(v_prev2_q));
    const HVX_Vector v_prev3_lo =
        Q6_Vsf_equals_Vqf32(Q6_V_lo_W(v_prev3_q));
    const HVX_Vector v_prev3_hi =
        Q6_Vsf_equals_Vqf32(Q6_V_hi_W(v_prev3_q));

    HVX_Vector v_kv_lo = vmemu(kv_mem + offset);
    HVX_Vector v_kv_hi = vmemu(kv_mem + offset + floats_per_vec);
    HVX_Vector v_out_lo = vmemu(output_pre + offset);
    HVX_Vector v_out_hi = vmemu(output_pre + offset + floats_per_vec);

    v_kv_lo = deltanetHvxAddF32(
        v_kv_lo, deltanetHvxMulF32(v_prev0_lo, v_key0));
    v_kv_hi = deltanetHvxAddF32(
        v_kv_hi, deltanetHvxMulF32(v_prev0_hi, v_key0));
    v_kv_lo = deltanetHvxAddF32(
        v_kv_lo, deltanetHvxMulF32(v_prev1_lo, v_key1));
    v_kv_hi = deltanetHvxAddF32(
        v_kv_hi, deltanetHvxMulF32(v_prev1_hi, v_key1));
    v_kv_lo = deltanetHvxAddF32(
        v_kv_lo, deltanetHvxMulF32(v_prev2_lo, v_key2));
    v_kv_hi = deltanetHvxAddF32(
        v_kv_hi, deltanetHvxMulF32(v_prev2_hi, v_key2));
    v_kv_lo = deltanetHvxAddF32(
        v_kv_lo, deltanetHvxMulF32(v_prev3_lo, v_key3));
    v_kv_hi = deltanetHvxAddF32(
        v_kv_hi, deltanetHvxMulF32(v_prev3_hi, v_key3));

    v_out_lo = deltanetHvxAddF32(
        v_out_lo, deltanetHvxMulF32(v_prev0_lo, v_query0));
    v_out_hi = deltanetHvxAddF32(
        v_out_hi, deltanetHvxMulF32(v_prev0_hi, v_query0));
    v_out_lo = deltanetHvxAddF32(
        v_out_lo, deltanetHvxMulF32(v_prev1_lo, v_query1));
    v_out_hi = deltanetHvxAddF32(
        v_out_hi, deltanetHvxMulF32(v_prev1_hi, v_query1));
    v_out_lo = deltanetHvxAddF32(
        v_out_lo, deltanetHvxMulF32(v_prev2_lo, v_query2));
    v_out_hi = deltanetHvxAddF32(
        v_out_hi, deltanetHvxMulF32(v_prev2_hi, v_query2));
    v_out_lo = deltanetHvxAddF32(
        v_out_lo, deltanetHvxMulF32(v_prev3_lo, v_query3));
    v_out_hi = deltanetHvxAddF32(
        v_out_hi, deltanetHvxMulF32(v_prev3_hi, v_query3));

    vmemu(kv_mem + offset) = v_kv_lo;
    vmemu(kv_mem + offset + floats_per_vec) = v_kv_hi;
    vmemu(output_pre + offset) = v_out_lo;
    vmemu(output_pre + offset + floats_per_vec) = v_out_hi;
  }

  for (size_t v = vec_count * halfs_per_vec; v < value_dim; ++v) {
    const float prev_value0 = static_cast<float>(prev0[v]);
    const float prev_value1 = static_cast<float>(prev1[v]);
    const float prev_value2 = static_cast<float>(prev2[v]);
    const float prev_value3 = static_cast<float>(prev3[v]);
    kv_mem[v] += prev_value0 * key_scales[0] +
        prev_value1 * key_scales[1] + prev_value2 * key_scales[2] +
        prev_value3 * key_scales[3];
    output_pre[v] += prev_value0 * query_scales[0] +
        prev_value1 * query_scales[1] + prev_value2 * query_scales[2] +
        prev_value3 * query_scales[3];
  }
}

static inline void deltanetHvxDeltaFromValueF16(
    const Float16* value,
    const float* kv_mem,
    float* delta,
    float beta_value,
    size_t value_dim) {
  const size_t halfs_per_vec = 128 / sizeof(Float16);
  const size_t floats_per_vec = 128 / sizeof(float);
  const size_t vec_count = value_dim / halfs_per_vec;
  const HVX_Vector v_one = Q6_Vh_vsplat_R(0x3C00);
  const HVX_Vector v_neg_one = Q6_V_vsplat_R(0xBF800000);
  const HVX_Vector v_beta = deltanetHvxSplatF32(beta_value);

  for (size_t i = 0; i < vec_count; ++i) {
    const size_t offset = i * halfs_per_vec;
    const HVX_Vector v_half = vmemu(value + offset);
    const HVX_VectorPair v_value_q = Q6_Wqf32_vmpy_VhfVhf(v_half, v_one);
    const HVX_Vector v_value_lo = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(v_value_q));
    const HVX_Vector v_value_hi = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(v_value_q));
    const HVX_Vector v_neg_kv_lo =
        deltanetHvxMulF32(vmemu(kv_mem + offset), v_neg_one);
    const HVX_Vector v_neg_kv_hi = deltanetHvxMulF32(
        vmemu(kv_mem + offset + floats_per_vec), v_neg_one);
    const HVX_Vector v_delta_lo = deltanetHvxMulF32(
        deltanetHvxAddF32(v_value_lo, v_neg_kv_lo), v_beta);
    const HVX_Vector v_delta_hi = deltanetHvxMulF32(
        deltanetHvxAddF32(v_value_hi, v_neg_kv_hi), v_beta);
    vmemu(delta + offset) = v_delta_lo;
    vmemu(delta + offset + floats_per_vec) = v_delta_hi;
  }

  for (size_t v = vec_count * halfs_per_vec; v < value_dim; ++v) {
    delta[v] = (static_cast<float>(value[v]) - kv_mem[v]) * beta_value;
  }
}

static inline void deltanetHvxUpdateStateF16(
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
  const HVX_Vector v_decay = deltanetHvxSplatF32(decay);
  const HVX_Vector v_key = deltanetHvxSplatF32(key_value);

  for (size_t i = 0; i < vec_count; ++i) {
    const size_t offset = i * halfs_per_vec;
    const HVX_Vector v_half = vmemu(prev_state + offset);
    const HVX_VectorPair v_prev_q = Q6_Wqf32_vmpy_VhfVhf(v_half, v_one);
    const HVX_Vector v_prev_lo = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(v_prev_q));
    const HVX_Vector v_prev_hi = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(v_prev_q));
    const HVX_Vector v_delta_lo = vmemu(delta + offset);
    const HVX_Vector v_delta_hi = vmemu(delta + offset + floats_per_vec);
    const HVX_Vector v_updated_lo = deltanetHvxAddF32(
        deltanetHvxMulF32(v_prev_lo, v_decay),
        deltanetHvxMulF32(v_delta_lo, v_key));
    const HVX_Vector v_updated_hi = deltanetHvxAddF32(
        deltanetHvxMulF32(v_prev_hi, v_decay),
        deltanetHvxMulF32(v_delta_hi, v_key));
    vmemu(state + offset) =
        Q6_Vhf_equals_Wqf32(deltanetHvxF32PairToQF32(v_updated_lo, v_updated_hi));
  }

  for (size_t v = vec_count * halfs_per_vec; v < value_dim; ++v) {
    state[v] = Float16(static_cast<float>(prev_state[v]) * decay +
                       delta[v] * key_value);
  }
}

static inline void deltanetHvxUpdateState4F16(
    const Float16* prev0,
    const Float16* prev1,
    const Float16* prev2,
    const Float16* prev3,
    Float16* state0,
    Float16* state1,
    Float16* state2,
    Float16* state3,
    const float* delta,
    float decay,
    const float* key_values,
    size_t value_dim) {
  const size_t halfs_per_vec = 128 / sizeof(Float16);
  const size_t floats_per_vec = 128 / sizeof(float);
  const size_t vec_count = value_dim / halfs_per_vec;
  const HVX_Vector v_one = Q6_Vh_vsplat_R(0x3C00);
  const HVX_Vector v_decay = deltanetHvxSplatF32(decay);
  const HVX_Vector v_key0 = deltanetHvxSplatF32(key_values[0]);
  const HVX_Vector v_key1 = deltanetHvxSplatF32(key_values[1]);
  const HVX_Vector v_key2 = deltanetHvxSplatF32(key_values[2]);
  const HVX_Vector v_key3 = deltanetHvxSplatF32(key_values[3]);

  for (size_t i = 0; i < vec_count; ++i) {
    const size_t offset = i * halfs_per_vec;
    const HVX_Vector v_delta_lo = vmemu(delta + offset);
    const HVX_Vector v_delta_hi = vmemu(delta + offset + floats_per_vec);

    const HVX_VectorPair v_prev0_q =
        Q6_Wqf32_vmpy_VhfVhf(vmemu(prev0 + offset), v_one);
    const HVX_VectorPair v_prev1_q =
        Q6_Wqf32_vmpy_VhfVhf(vmemu(prev1 + offset), v_one);
    const HVX_VectorPair v_prev2_q =
        Q6_Wqf32_vmpy_VhfVhf(vmemu(prev2 + offset), v_one);
    const HVX_VectorPair v_prev3_q =
        Q6_Wqf32_vmpy_VhfVhf(vmemu(prev3 + offset), v_one);

    const HVX_Vector v_updated0_lo = deltanetHvxAddF32(
        deltanetHvxMulF32(Q6_Vsf_equals_Vqf32(Q6_V_lo_W(v_prev0_q)), v_decay),
        deltanetHvxMulF32(v_delta_lo, v_key0));
    const HVX_Vector v_updated0_hi = deltanetHvxAddF32(
        deltanetHvxMulF32(Q6_Vsf_equals_Vqf32(Q6_V_hi_W(v_prev0_q)), v_decay),
        deltanetHvxMulF32(v_delta_hi, v_key0));
    const HVX_Vector v_updated1_lo = deltanetHvxAddF32(
        deltanetHvxMulF32(Q6_Vsf_equals_Vqf32(Q6_V_lo_W(v_prev1_q)), v_decay),
        deltanetHvxMulF32(v_delta_lo, v_key1));
    const HVX_Vector v_updated1_hi = deltanetHvxAddF32(
        deltanetHvxMulF32(Q6_Vsf_equals_Vqf32(Q6_V_hi_W(v_prev1_q)), v_decay),
        deltanetHvxMulF32(v_delta_hi, v_key1));
    const HVX_Vector v_updated2_lo = deltanetHvxAddF32(
        deltanetHvxMulF32(Q6_Vsf_equals_Vqf32(Q6_V_lo_W(v_prev2_q)), v_decay),
        deltanetHvxMulF32(v_delta_lo, v_key2));
    const HVX_Vector v_updated2_hi = deltanetHvxAddF32(
        deltanetHvxMulF32(Q6_Vsf_equals_Vqf32(Q6_V_hi_W(v_prev2_q)), v_decay),
        deltanetHvxMulF32(v_delta_hi, v_key2));
    const HVX_Vector v_updated3_lo = deltanetHvxAddF32(
        deltanetHvxMulF32(Q6_Vsf_equals_Vqf32(Q6_V_lo_W(v_prev3_q)), v_decay),
        deltanetHvxMulF32(v_delta_lo, v_key3));
    const HVX_Vector v_updated3_hi = deltanetHvxAddF32(
        deltanetHvxMulF32(Q6_Vsf_equals_Vqf32(Q6_V_hi_W(v_prev3_q)), v_decay),
        deltanetHvxMulF32(v_delta_hi, v_key3));

    vmemu(state0 + offset) = Q6_Vhf_equals_Wqf32(
        deltanetHvxF32PairToQF32(v_updated0_lo, v_updated0_hi));
    vmemu(state1 + offset) = Q6_Vhf_equals_Wqf32(
        deltanetHvxF32PairToQF32(v_updated1_lo, v_updated1_hi));
    vmemu(state2 + offset) = Q6_Vhf_equals_Wqf32(
        deltanetHvxF32PairToQF32(v_updated2_lo, v_updated2_hi));
    vmemu(state3 + offset) = Q6_Vhf_equals_Wqf32(
        deltanetHvxF32PairToQF32(v_updated3_lo, v_updated3_hi));
  }

  for (size_t v = vec_count * halfs_per_vec; v < value_dim; ++v) {
    state0[v] =
        Float16(static_cast<float>(prev0[v]) * decay + delta[v] * key_values[0]);
    state1[v] =
        Float16(static_cast<float>(prev1[v]) * decay + delta[v] * key_values[1]);
    state2[v] =
        Float16(static_cast<float>(prev2[v]) * decay + delta[v] * key_values[2]);
    state3[v] =
        Float16(static_cast<float>(prev3[v]) * decay + delta[v] * key_values[3]);
  }
}

static inline void deltanetHvxWriteOutputF16(
    Float16* output,
    const float* output_pre,
    const float* delta,
    float key_query_dot,
    size_t value_dim) {
  const size_t halfs_per_vec = 128 / sizeof(Float16);
  const size_t floats_per_vec = 128 / sizeof(float);
  const size_t vec_count = value_dim / halfs_per_vec;
  const HVX_Vector v_dot = deltanetHvxSplatF32(key_query_dot);

  for (size_t i = 0; i < vec_count; ++i) {
    const size_t offset = i * halfs_per_vec;
    const HVX_Vector v_out_lo = deltanetHvxAddF32(
        vmemu(output_pre + offset),
        deltanetHvxMulF32(vmemu(delta + offset), v_dot));
    const HVX_Vector v_out_hi = deltanetHvxAddF32(
        vmemu(output_pre + offset + floats_per_vec),
        deltanetHvxMulF32(vmemu(delta + offset + floats_per_vec), v_dot));
    vmemu(output + offset) =
        Q6_Vhf_equals_Wqf32(deltanetHvxF32PairToQF32(v_out_lo, v_out_hi));
  }

  for (size_t v = vec_count * halfs_per_vec; v < value_dim; ++v) {
    output[v] = Float16(output_pre[v] + delta[v] * key_query_dot);
  }
}

static inline void deltanetHvxDeltaAndWriteOutputF16(
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
  const HVX_Vector v_beta = deltanetHvxSplatF32(beta_value);
  const HVX_Vector v_dot = deltanetHvxSplatF32(key_query_dot);

  for (size_t i = 0; i < vec_count; ++i) {
    const size_t offset = i * halfs_per_vec;
    const HVX_Vector v_half = vmemu(value + offset);
    const HVX_VectorPair v_value_q = Q6_Wqf32_vmpy_VhfVhf(v_half, v_one);
    const HVX_Vector v_value_lo = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(v_value_q));
    const HVX_Vector v_value_hi = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(v_value_q));
    const HVX_Vector v_delta_lo = deltanetHvxMulF32(
        deltanetHvxAddF32(
            v_value_lo,
            deltanetHvxMulF32(vmemu(kv_mem + offset), v_neg_one)),
        v_beta);
    const HVX_Vector v_delta_hi = deltanetHvxMulF32(
        deltanetHvxAddF32(
            v_value_hi,
            deltanetHvxMulF32(
                vmemu(kv_mem + offset + floats_per_vec), v_neg_one)),
        v_beta);
    const HVX_Vector v_out_lo = deltanetHvxAddF32(
        vmemu(output_pre + offset), deltanetHvxMulF32(v_delta_lo, v_dot));
    const HVX_Vector v_out_hi = deltanetHvxAddF32(
        vmemu(output_pre + offset + floats_per_vec),
        deltanetHvxMulF32(v_delta_hi, v_dot));
    vmemu(delta + offset) = v_delta_lo;
    vmemu(delta + offset + floats_per_vec) = v_delta_hi;
    vmemu(output + offset) =
        Q6_Vhf_equals_Wqf32(deltanetHvxF32PairToQF32(v_out_lo, v_out_hi));
  }

  for (size_t v = vec_count * halfs_per_vec; v < value_dim; ++v) {
    const float delta_value =
        (static_cast<float>(value[v]) - kv_mem[v]) * beta_value;
    delta[v] = delta_value;
    output[v] = Float16(output_pre[v] + delta_value * key_query_dot);
  }
}

static GraphStatus deltanetImplRawFloat16Hvx(
    Tensor& core_attn_out,
    Tensor& new_recurrent_state,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& g,
    const Tensor& beta,
    const Tensor& recurrent_state,
    float scaling_value,
    float epsilon_value) {
  const size_t batch = query.dim(0);
  const size_t seq_len = query.dim(1);
  const size_t heads = query.dim(2);
  const size_t key_dim = query.dim(3);
  const size_t value_dim = value.dim(3);

  if (key_dim > kMaxDeltaNetDim || value_dim > kMaxDeltaNetDim) {
    return GraphStatus::ErrorUnsupported;
  }

  const Float16* query_data =
      static_cast<const Float16*>(query.raw_data_const());
  const Float16* key_data = static_cast<const Float16*>(key.raw_data_const());
  const Float16* value_data =
      static_cast<const Float16*>(value.raw_data_const());
  const Float16* recurrent_state_data =
      static_cast<const Float16*>(recurrent_state.raw_data_const());
  Float16* output_data = static_cast<Float16*>(core_attn_out.raw_data());
  Float16* state_data = static_cast<Float16*>(new_recurrent_state.raw_data());

  float local_query[kMaxDeltaNetDim] __attribute__((aligned(128)));
  float local_key[kMaxDeltaNetDim] __attribute__((aligned(128)));
  float kv_mem[kMaxDeltaNetDim] __attribute__((aligned(128)));
  float delta[kMaxDeltaNetDim] __attribute__((aligned(128)));
  float output_pre[kMaxDeltaNetDim] __attribute__((aligned(128)));

  if (kUseDeltaNetAr1SpecializedPath && seq_len == 1) {
    for (size_t b = 0; b < batch; ++b) {
      for (size_t h = 0; h < heads; ++h) {
        const size_t qkv_base = b * heads + h;
        const Float16* query_row = query_data + qkv_base * key_dim;
        const Float16* key_row = key_data + qkv_base * key_dim;
        const Float16* value_row = value_data + qkv_base * value_dim;
        const size_t state_base = (b * heads + h) * key_dim * value_dim;

        float query_square_sum = 0.0f;
        float key_square_sum = 0.0f;
        for (size_t k = 0; k < key_dim; ++k) {
          const float query_value = static_cast<float>(query_row[k]);
          const float key_value = static_cast<float>(key_row[k]);
          query_square_sum += query_value * query_value;
          key_square_sum += key_value * key_value;
          local_query[k] = query_value;
          local_key[k] = key_value;
        }

        const float query_norm =
            scaling_value / std::sqrt(query_square_sum + epsilon_value);
        const float key_norm = 1.0f / std::sqrt(key_square_sum + epsilon_value);
        float key_query_dot = 0.0f;
        for (size_t k = 0; k < key_dim; ++k) {
          local_query[k] *= query_norm;
          local_key[k] *= key_norm;
          key_query_dot += local_key[k] * local_query[k];
        }

        std::fill(kv_mem, kv_mem + value_dim, 0.0f);
        std::fill(output_pre, output_pre + value_dim, 0.0f);
        const float decay = std::exp(readTimeHeadValue(g, b, 0, h));
        const float beta_value = readTimeHeadValue(beta, b, 0, h);

        size_t k = 0;
        for (; kUseDeltaNetBlock4Hvx && k + 4 <= key_dim; k += 4) {
          const size_t row_base0 = state_base + (k + 0) * value_dim;
          const size_t row_base1 = state_base + (k + 1) * value_dim;
          const size_t row_base2 = state_base + (k + 2) * value_dim;
          const size_t row_base3 = state_base + (k + 3) * value_dim;
          const float key_scales[4] = {
              decay * local_key[k + 0],
              decay * local_key[k + 1],
              decay * local_key[k + 2],
              decay * local_key[k + 3],
          };
          const float query_scales[4] = {
              decay * local_query[k + 0],
              decay * local_query[k + 1],
              decay * local_query[k + 2],
              decay * local_query[k + 3],
          };
          deltanetHvxAccum4F16(
              recurrent_state_data + row_base0,
              recurrent_state_data + row_base1,
              recurrent_state_data + row_base2,
              recurrent_state_data + row_base3,
              kv_mem,
              output_pre,
              key_scales,
              query_scales,
              value_dim);
        }
        for (; k < key_dim; ++k) {
          const size_t row_base = state_base + k * value_dim;
          deltanetHvxAccumF16(
              recurrent_state_data + row_base,
              kv_mem,
              output_pre,
              decay * local_key[k],
              decay * local_query[k],
              value_dim);
        }

        deltanetHvxDeltaAndWriteOutputF16(
            output_data + qkv_base * value_dim,
            value_row,
            kv_mem,
            output_pre,
            delta,
            beta_value,
            key_query_dot,
            value_dim);

        k = 0;
        for (; kUseDeltaNetBlock4Hvx && k + 4 <= key_dim; k += 4) {
          const size_t row_base0 = state_base + (k + 0) * value_dim;
          const size_t row_base1 = state_base + (k + 1) * value_dim;
          const size_t row_base2 = state_base + (k + 2) * value_dim;
          const size_t row_base3 = state_base + (k + 3) * value_dim;
          const float key_values[4] = {
              local_key[k + 0],
              local_key[k + 1],
              local_key[k + 2],
              local_key[k + 3],
          };
          deltanetHvxUpdateState4F16(
              recurrent_state_data + row_base0,
              recurrent_state_data + row_base1,
              recurrent_state_data + row_base2,
              recurrent_state_data + row_base3,
              state_data + row_base0,
              state_data + row_base1,
              state_data + row_base2,
              state_data + row_base3,
              delta,
              decay,
              key_values,
              value_dim);
        }
        for (; k < key_dim; ++k) {
          const size_t row_base = state_base + k * value_dim;
          deltanetHvxUpdateStateF16(
              recurrent_state_data + row_base,
              state_data + row_base,
              delta,
              decay,
              local_key[k],
              value_dim);
        }

      }
    }

    return GraphStatus::Success;
  }

  for (size_t b = 0; b < batch; ++b) {
    for (size_t t = 0; t < seq_len; ++t) {
      for (size_t h = 0; h < heads; ++h) {
        const size_t qkv_base = ((b * seq_len + t) * heads + h);
        const Float16* query_row = query_data + qkv_base * key_dim;
        const Float16* key_row = key_data + qkv_base * key_dim;
        const Float16* value_row = value_data + qkv_base * value_dim;
        const size_t state_base = (b * heads + h) * key_dim * value_dim;

        float query_square_sum = 0.0f;
        float key_square_sum = 0.0f;
        for (size_t k = 0; k < key_dim; ++k) {
          const float query_value = static_cast<float>(query_row[k]);
          const float key_value = static_cast<float>(key_row[k]);
          query_square_sum += query_value * query_value;
          key_square_sum += key_value * key_value;
          local_query[k] = query_value;
          local_key[k] = key_value;
        }

        const float query_norm =
            scaling_value / std::sqrt(query_square_sum + epsilon_value);
        const float key_norm = 1.0f / std::sqrt(key_square_sum + epsilon_value);
        float key_query_dot = 0.0f;
        for (size_t k = 0; k < key_dim; ++k) {
          local_query[k] *= query_norm;
          local_key[k] *= key_norm;
          key_query_dot += local_key[k] * local_query[k];
        }

        std::fill(kv_mem, kv_mem + value_dim, 0.0f);
        std::fill(output_pre, output_pre + value_dim, 0.0f);
        const float decay = std::exp(readTimeHeadValue(g, b, t, h));
        const float beta_value = readTimeHeadValue(beta, b, t, h);

        size_t k = 0;
        for (; kUseDeltaNetBlock4Hvx && k + 4 <= key_dim; k += 4) {
          const size_t row_base0 = state_base + (k + 0) * value_dim;
          const size_t row_base1 = state_base + (k + 1) * value_dim;
          const size_t row_base2 = state_base + (k + 2) * value_dim;
          const size_t row_base3 = state_base + (k + 3) * value_dim;
          const Float16* prev0 = t == 0 ? recurrent_state_data + row_base0
                                        : state_data + row_base0;
          const Float16* prev1 = t == 0 ? recurrent_state_data + row_base1
                                        : state_data + row_base1;
          const Float16* prev2 = t == 0 ? recurrent_state_data + row_base2
                                        : state_data + row_base2;
          const Float16* prev3 = t == 0 ? recurrent_state_data + row_base3
                                        : state_data + row_base3;
          const float key_scales[4] = {
              decay * local_key[k + 0],
              decay * local_key[k + 1],
              decay * local_key[k + 2],
              decay * local_key[k + 3],
          };
          const float query_scales[4] = {
              decay * local_query[k + 0],
              decay * local_query[k + 1],
              decay * local_query[k + 2],
              decay * local_query[k + 3],
          };
          deltanetHvxAccum4F16(
              prev0,
              prev1,
              prev2,
              prev3,
              kv_mem,
              output_pre,
              key_scales,
              query_scales,
              value_dim);
        }
        for (; k < key_dim; ++k) {
          const size_t row_base = state_base + k * value_dim;
          const Float16* prev_state =
              t == 0 ? recurrent_state_data + row_base : state_data + row_base;
          deltanetHvxAccumF16(
              prev_state,
              kv_mem,
              output_pre,
              decay * local_key[k],
              decay * local_query[k],
              value_dim);
        }

        deltanetHvxDeltaAndWriteOutputF16(
            output_data + qkv_base * value_dim,
            value_row,
            kv_mem,
            output_pre,
            delta,
            beta_value,
            key_query_dot,
            value_dim);

        k = 0;
        for (; kUseDeltaNetBlock4Hvx && k + 4 <= key_dim; k += 4) {
          const size_t row_base0 = state_base + (k + 0) * value_dim;
          const size_t row_base1 = state_base + (k + 1) * value_dim;
          const size_t row_base2 = state_base + (k + 2) * value_dim;
          const size_t row_base3 = state_base + (k + 3) * value_dim;
          const Float16* prev0 = t == 0 ? recurrent_state_data + row_base0
                                        : state_data + row_base0;
          const Float16* prev1 = t == 0 ? recurrent_state_data + row_base1
                                        : state_data + row_base1;
          const Float16* prev2 = t == 0 ? recurrent_state_data + row_base2
                                        : state_data + row_base2;
          const Float16* prev3 = t == 0 ? recurrent_state_data + row_base3
                                        : state_data + row_base3;
          const float key_values[4] = {
              local_key[k + 0],
              local_key[k + 1],
              local_key[k + 2],
              local_key[k + 3],
          };
          deltanetHvxUpdateState4F16(
              prev0,
              prev1,
              prev2,
              prev3,
              state_data + row_base0,
              state_data + row_base1,
              state_data + row_base2,
              state_data + row_base3,
              delta,
              decay,
              key_values,
              value_dim);
        }
        for (; k < key_dim; ++k) {
          const size_t row_base = state_base + k * value_dim;
          const Float16* prev_state =
              t == 0 ? recurrent_state_data + row_base : state_data + row_base;
          Float16* state_row = state_data + row_base;
          deltanetHvxUpdateStateF16(
              prev_state, state_row, delta, decay, local_key[k], value_dim);
        }

      }
    }
  }

  return GraphStatus::Success;
}

static GraphStatus deltanetImplRawFloat32Hvx(
    Tensor& core_attn_out,
    Tensor& new_recurrent_state,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& g,
    const Tensor& beta,
    const Tensor& recurrent_state,
    float scaling_value,
    float epsilon_value) {
  const size_t batch = query.dim(0);
  const size_t seq_len = query.dim(1);
  const size_t heads = query.dim(2);
  const size_t key_dim = query.dim(3);
  const size_t value_dim = value.dim(3);

  if (key_dim > kMaxDeltaNetDim || value_dim > kMaxDeltaNetDim) {
    return GraphStatus::ErrorUnsupported;
  }

  const float* query_data = static_cast<const float*>(query.raw_data_const());
  const float* key_data = static_cast<const float*>(key.raw_data_const());
  const float* value_data = static_cast<const float*>(value.raw_data_const());
  const float* recurrent_state_data =
      static_cast<const float*>(recurrent_state.raw_data_const());
  float* output_data = static_cast<float*>(core_attn_out.raw_data());
  float* state_data = static_cast<float*>(new_recurrent_state.raw_data());

  float local_query[kMaxDeltaNetDim] __attribute__((aligned(128)));
  float local_key[kMaxDeltaNetDim] __attribute__((aligned(128)));
  float kv_mem[kMaxDeltaNetDim] __attribute__((aligned(128)));
  float delta[kMaxDeltaNetDim] __attribute__((aligned(128)));
  float output_pre[kMaxDeltaNetDim] __attribute__((aligned(128)));

  if (kUseDeltaNetAr1SpecializedPath && seq_len == 1) {
    for (size_t b = 0; b < batch; ++b) {
      for (size_t h = 0; h < heads; ++h) {
        const size_t qkv_base = b * heads + h;
        const float* query_row = query_data + qkv_base * key_dim;
        const float* key_row = key_data + qkv_base * key_dim;
        const float* value_row = value_data + qkv_base * value_dim;
        const size_t state_base = (b * heads + h) * key_dim * value_dim;

        float query_square_sum = 0.0f;
        float key_square_sum = 0.0f;
        for (size_t k = 0; k < key_dim; ++k) {
          const float query_value = query_row[k];
          const float key_value = key_row[k];
          query_square_sum += query_value * query_value;
          key_square_sum += key_value * key_value;
          local_query[k] = query_value;
          local_key[k] = key_value;
        }

        const float query_norm =
            scaling_value / std::sqrt(query_square_sum + epsilon_value);
        const float key_norm = 1.0f / std::sqrt(key_square_sum + epsilon_value);
        float key_query_dot = 0.0f;
        for (size_t k = 0; k < key_dim; ++k) {
          local_query[k] *= query_norm;
          local_key[k] *= key_norm;
          key_query_dot += local_key[k] * local_query[k];
        }

        std::fill(kv_mem, kv_mem + value_dim, 0.0f);
        std::fill(output_pre, output_pre + value_dim, 0.0f);
        const float decay = std::exp(readTimeHeadValue(g, b, 0, h));
        const float beta_value = readTimeHeadValue(beta, b, 0, h);

        size_t k = 0;
        for (; kUseDeltaNetBlock4Hvx && k + 4 <= key_dim; k += 4) {
          const size_t row_base0 = state_base + (k + 0) * value_dim;
          const size_t row_base1 = state_base + (k + 1) * value_dim;
          const size_t row_base2 = state_base + (k + 2) * value_dim;
          const size_t row_base3 = state_base + (k + 3) * value_dim;
          const float key_scales[4] = {
              decay * local_key[k + 0],
              decay * local_key[k + 1],
              decay * local_key[k + 2],
              decay * local_key[k + 3],
          };
          const float query_scales[4] = {
              decay * local_query[k + 0],
              decay * local_query[k + 1],
              decay * local_query[k + 2],
              decay * local_query[k + 3],
          };
          deltanetHvxAccum4F32(
              recurrent_state_data + row_base0,
              recurrent_state_data + row_base1,
              recurrent_state_data + row_base2,
              recurrent_state_data + row_base3,
              kv_mem,
              output_pre,
              key_scales,
              query_scales,
              value_dim);
        }
        for (; k < key_dim; ++k) {
          const size_t row_base = state_base + k * value_dim;
          deltanetHvxAccumF32(
              recurrent_state_data + row_base,
              kv_mem,
              output_pre,
              decay * local_key[k],
              decay * local_query[k],
              value_dim);
        }

        deltanetHvxDeltaAndWriteOutputF32(
            output_data + qkv_base * value_dim,
            value_row,
            kv_mem,
            output_pre,
            delta,
            beta_value,
            key_query_dot,
            value_dim);

        k = 0;
        for (; kUseDeltaNetBlock4Hvx && k + 4 <= key_dim; k += 4) {
          const size_t row_base0 = state_base + (k + 0) * value_dim;
          const size_t row_base1 = state_base + (k + 1) * value_dim;
          const size_t row_base2 = state_base + (k + 2) * value_dim;
          const size_t row_base3 = state_base + (k + 3) * value_dim;
          const float key_values[4] = {
              local_key[k + 0],
              local_key[k + 1],
              local_key[k + 2],
              local_key[k + 3],
          };
          deltanetHvxUpdateState4F32(
              recurrent_state_data + row_base0,
              recurrent_state_data + row_base1,
              recurrent_state_data + row_base2,
              recurrent_state_data + row_base3,
              state_data + row_base0,
              state_data + row_base1,
              state_data + row_base2,
              state_data + row_base3,
              delta,
              decay,
              key_values,
              value_dim);
        }
        for (; k < key_dim; ++k) {
          const size_t row_base = state_base + k * value_dim;
          deltanetHvxUpdateStateF32(
              recurrent_state_data + row_base,
              state_data + row_base,
              delta,
              decay,
              local_key[k],
              value_dim);
        }

      }
    }

    return GraphStatus::Success;
  }

  for (size_t b = 0; b < batch; ++b) {
    for (size_t t = 0; t < seq_len; ++t) {
      for (size_t h = 0; h < heads; ++h) {
        const size_t qkv_base = ((b * seq_len + t) * heads + h);
        const float* query_row = query_data + qkv_base * key_dim;
        const float* key_row = key_data + qkv_base * key_dim;
        const float* value_row = value_data + qkv_base * value_dim;
        const size_t state_base = (b * heads + h) * key_dim * value_dim;

        float query_square_sum = 0.0f;
        float key_square_sum = 0.0f;
        for (size_t k = 0; k < key_dim; ++k) {
          const float query_value = query_row[k];
          const float key_value = key_row[k];
          query_square_sum += query_value * query_value;
          key_square_sum += key_value * key_value;
          local_query[k] = query_value;
          local_key[k] = key_value;
        }

        const float query_norm =
            scaling_value / std::sqrt(query_square_sum + epsilon_value);
        const float key_norm = 1.0f / std::sqrt(key_square_sum + epsilon_value);
        float key_query_dot = 0.0f;
        for (size_t k = 0; k < key_dim; ++k) {
          local_query[k] *= query_norm;
          local_key[k] *= key_norm;
          key_query_dot += local_key[k] * local_query[k];
        }

        std::fill(kv_mem, kv_mem + value_dim, 0.0f);
        std::fill(output_pre, output_pre + value_dim, 0.0f);
        const float decay = std::exp(readTimeHeadValue(g, b, t, h));
        const float beta_value = readTimeHeadValue(beta, b, t, h);

        size_t k = 0;
        for (; kUseDeltaNetBlock4Hvx && k + 4 <= key_dim; k += 4) {
          const size_t row_base0 = state_base + (k + 0) * value_dim;
          const size_t row_base1 = state_base + (k + 1) * value_dim;
          const size_t row_base2 = state_base + (k + 2) * value_dim;
          const size_t row_base3 = state_base + (k + 3) * value_dim;
          const float* prev0 = t == 0 ? recurrent_state_data + row_base0
                                      : state_data + row_base0;
          const float* prev1 = t == 0 ? recurrent_state_data + row_base1
                                      : state_data + row_base1;
          const float* prev2 = t == 0 ? recurrent_state_data + row_base2
                                      : state_data + row_base2;
          const float* prev3 = t == 0 ? recurrent_state_data + row_base3
                                      : state_data + row_base3;
          const float key_scales[4] = {
              decay * local_key[k + 0],
              decay * local_key[k + 1],
              decay * local_key[k + 2],
              decay * local_key[k + 3],
          };
          const float query_scales[4] = {
              decay * local_query[k + 0],
              decay * local_query[k + 1],
              decay * local_query[k + 2],
              decay * local_query[k + 3],
          };
          deltanetHvxAccum4F32(
              prev0,
              prev1,
              prev2,
              prev3,
              kv_mem,
              output_pre,
              key_scales,
              query_scales,
              value_dim);
        }
        for (; k < key_dim; ++k) {
          const size_t row_base = state_base + k * value_dim;
          const float* prev_state =
              t == 0 ? recurrent_state_data + row_base : state_data + row_base;
          deltanetHvxAccumF32(
              prev_state,
              kv_mem,
              output_pre,
              decay * local_key[k],
              decay * local_query[k],
              value_dim);
        }

        deltanetHvxDeltaAndWriteOutputF32(
            output_data + qkv_base * value_dim,
            value_row,
            kv_mem,
            output_pre,
            delta,
            beta_value,
            key_query_dot,
            value_dim);

        k = 0;
        for (; kUseDeltaNetBlock4Hvx && k + 4 <= key_dim; k += 4) {
          const size_t row_base0 = state_base + (k + 0) * value_dim;
          const size_t row_base1 = state_base + (k + 1) * value_dim;
          const size_t row_base2 = state_base + (k + 2) * value_dim;
          const size_t row_base3 = state_base + (k + 3) * value_dim;
          const float* prev0 = t == 0 ? recurrent_state_data + row_base0
                                      : state_data + row_base0;
          const float* prev1 = t == 0 ? recurrent_state_data + row_base1
                                      : state_data + row_base1;
          const float* prev2 = t == 0 ? recurrent_state_data + row_base2
                                      : state_data + row_base2;
          const float* prev3 = t == 0 ? recurrent_state_data + row_base3
                                      : state_data + row_base3;
          const float key_values[4] = {
              local_key[k + 0],
              local_key[k + 1],
              local_key[k + 2],
              local_key[k + 3],
          };
          deltanetHvxUpdateState4F32(
              prev0,
              prev1,
              prev2,
              prev3,
              state_data + row_base0,
              state_data + row_base1,
              state_data + row_base2,
              state_data + row_base3,
              delta,
              decay,
              key_values,
              value_dim);
        }
        for (; k < key_dim; ++k) {
          const size_t row_base = state_base + k * value_dim;
          const float* prev_state =
              t == 0 ? recurrent_state_data + row_base : state_data + row_base;
          deltanetHvxUpdateStateF32(
              prev_state,
              state_data + row_base,
              delta,
              decay,
              local_key[k],
              value_dim);
        }

      }
    }
  }

  return GraphStatus::Success;
}

static GraphStatus deltanetPackedImplRawFloat16Hvx(
    Tensor& packed_out,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& g,
    const Tensor& beta,
    const Tensor& recurrent_state) {
  const size_t batch = query.dim(0);
  const size_t seq_len = query.dim(1);
  const size_t heads = query.dim(2);
  const size_t key_dim = query.dim(3);
  const size_t value_dim = value.dim(3);

  if (key_dim > kMaxDeltaNetDim || value_dim > kMaxDeltaNetDim) {
    return GraphStatus::ErrorUnsupported;
  }

  const size_t output_numel = batch * seq_len * heads * value_dim;

  const Float16* query_data =
      static_cast<const Float16*>(query.raw_data_const());
  const Float16* key_data = static_cast<const Float16*>(key.raw_data_const());
  const Float16* value_data =
      static_cast<const Float16*>(value.raw_data_const());
  const Float16* recurrent_state_data =
      static_cast<const Float16*>(recurrent_state.raw_data_const());
  Float16* packed_data = static_cast<Float16*>(packed_out.raw_data());

  float local_query[kMaxDeltaNetDim] __attribute__((aligned(128)));
  float local_key[kMaxDeltaNetDim] __attribute__((aligned(128)));
  float kv_mem[kMaxDeltaNetDim] __attribute__((aligned(128)));
  float delta[kMaxDeltaNetDim] __attribute__((aligned(128)));
  float output_pre[kMaxDeltaNetDim] __attribute__((aligned(128)));

  for (size_t b = 0; b < batch; ++b) {
    for (size_t t = 0; t < seq_len; ++t) {
      for (size_t h = 0; h < heads; ++h) {
        const size_t qkv_base = ((b * seq_len + t) * heads + h);
        const Float16* query_row = query_data + qkv_base * key_dim;
        const Float16* key_row = key_data + qkv_base * key_dim;
        const Float16* value_row = value_data + qkv_base * value_dim;

        float key_query_dot = 0.0f;
        for (size_t k = 0; k < key_dim; ++k) {
          local_query[k] = static_cast<float>(query_row[k]);
          local_key[k] = static_cast<float>(key_row[k]);
          key_query_dot += local_key[k] * local_query[k];
        }

        std::fill(kv_mem, kv_mem + value_dim, 0.0f);
        std::fill(output_pre, output_pre + value_dim, 0.0f);
        const float decay = std::exp(readTimeHeadValue(g, b, t, h));
        const float beta_value = readTimeHeadValue(beta, b, t, h);

        for (size_t k = 0; k < key_dim; ++k) {
          const size_t state_index = packedStateIndex(
              output_numel, b, h, k, 0, heads, key_dim, value_dim);
          const size_t input_state_index =
              ((b * heads + h) * key_dim + k) * value_dim;
          const Float16* prev_state =
              t == 0 ? recurrent_state_data + input_state_index
                     : packed_data + state_index;
          deltanetHvxAccumF16(
              prev_state,
              kv_mem,
              output_pre,
              decay * local_key[k],
              decay * local_query[k],
              value_dim);
        }

        const size_t output_base =
            packedOutputIndex(b, t, h, 0, seq_len, heads, value_dim);
        deltanetHvxDeltaAndWriteOutputF16(
            packed_data + output_base,
            value_row,
            kv_mem,
            output_pre,
            delta,
            beta_value,
            key_query_dot,
            value_dim);

        for (size_t k = 0; k < key_dim; ++k) {
          const size_t state_index = packedStateIndex(
              output_numel, b, h, k, 0, heads, key_dim, value_dim);
          const size_t input_state_index =
              ((b * heads + h) * key_dim + k) * value_dim;
          const Float16* prev_state =
              t == 0 ? recurrent_state_data + input_state_index
                     : packed_data + state_index;
          deltanetHvxUpdateStateF16(
              prev_state,
              packed_data + state_index,
              delta,
              decay,
              local_key[k],
              value_dim);
        }

      }
    }
  }

  return GraphStatus::Success;
}

static GraphStatus deltanetPackedImplRawFloat32Hvx(
    Tensor& packed_out,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& g,
    const Tensor& beta,
    const Tensor& recurrent_state) {
  const size_t batch = query.dim(0);
  const size_t seq_len = query.dim(1);
  const size_t heads = query.dim(2);
  const size_t key_dim = query.dim(3);
  const size_t value_dim = value.dim(3);

  if (key_dim > kMaxDeltaNetDim || value_dim > kMaxDeltaNetDim) {
    return GraphStatus::ErrorUnsupported;
  }

  const size_t output_numel = batch * seq_len * heads * value_dim;

  const float* query_data = static_cast<const float*>(query.raw_data_const());
  const float* key_data = static_cast<const float*>(key.raw_data_const());
  const float* value_data = static_cast<const float*>(value.raw_data_const());
  const float* recurrent_state_data =
      static_cast<const float*>(recurrent_state.raw_data_const());
  float* packed_data = static_cast<float*>(packed_out.raw_data());

  float local_query[kMaxDeltaNetDim] __attribute__((aligned(128)));
  float local_key[kMaxDeltaNetDim] __attribute__((aligned(128)));
  float kv_mem[kMaxDeltaNetDim] __attribute__((aligned(128)));
  float delta[kMaxDeltaNetDim] __attribute__((aligned(128)));
  float output_pre[kMaxDeltaNetDim] __attribute__((aligned(128)));

  for (size_t b = 0; b < batch; ++b) {
    for (size_t t = 0; t < seq_len; ++t) {
      for (size_t h = 0; h < heads; ++h) {
        const size_t qkv_base = ((b * seq_len + t) * heads + h);
        const float* query_row = query_data + qkv_base * key_dim;
        const float* key_row = key_data + qkv_base * key_dim;
        const float* value_row = value_data + qkv_base * value_dim;

        float key_query_dot = 0.0f;
        for (size_t k = 0; k < key_dim; ++k) {
          local_query[k] = query_row[k];
          local_key[k] = key_row[k];
          key_query_dot += local_key[k] * local_query[k];
        }

        std::fill(kv_mem, kv_mem + value_dim, 0.0f);
        std::fill(output_pre, output_pre + value_dim, 0.0f);
        const float decay = std::exp(readTimeHeadValue(g, b, t, h));
        const float beta_value = readTimeHeadValue(beta, b, t, h);

        for (size_t k = 0; k < key_dim; ++k) {
          const size_t state_index = packedStateIndex(
              output_numel, b, h, k, 0, heads, key_dim, value_dim);
          const size_t input_state_index =
              ((b * heads + h) * key_dim + k) * value_dim;
          const float* prev_state =
              t == 0 ? recurrent_state_data + input_state_index
                     : packed_data + state_index;
          deltanetHvxAccumF32(
              prev_state,
              kv_mem,
              output_pre,
              decay * local_key[k],
              decay * local_query[k],
              value_dim);
        }

        const size_t output_base =
            packedOutputIndex(b, t, h, 0, seq_len, heads, value_dim);
        deltanetHvxDeltaAndWriteOutputF32(
            packed_data + output_base,
            value_row,
            kv_mem,
            output_pre,
            delta,
            beta_value,
            key_query_dot,
            value_dim);

        for (size_t k = 0; k < key_dim; ++k) {
          const size_t state_index = packedStateIndex(
              output_numel, b, h, k, 0, heads, key_dim, value_dim);
          const size_t input_state_index =
              ((b * heads + h) * key_dim + k) * value_dim;
          const float* prev_state =
              t == 0 ? recurrent_state_data + input_state_index
                     : packed_data + state_index;
          deltanetHvxUpdateStateF32(
              prev_state,
              packed_data + state_index,
              delta,
              decay,
              local_key[k],
              value_dim);
        }

      }
    }
  }

  return GraphStatus::Success;
}

static void deltanetCoreRawFloat16Hvx(
    Float16* output_data,
    Float16* state_data,
    const Float16* query_data,
    const Float16* key_data,
    const Float16* value_data,
    const Float16* g_data,
    const Float16* beta_data,
    const Float16* recurrent_state_data,
    size_t batch,
    size_t seq_len,
    size_t heads,
    size_t key_dim,
    size_t value_dim,
    size_t row_start,
    size_t row_stride) {
  float kv_mem[kMaxDeltaNetDim] __attribute__((aligned(128)));
  float delta[kMaxDeltaNetDim] __attribute__((aligned(128)));
  float output_pre[kMaxDeltaNetDim] __attribute__((aligned(128)));

  const size_t total_rows = batch * heads;
  for (size_t row = row_start; row < total_rows; row += row_stride) {
    const size_t b = row / heads;
    const size_t h = row % heads;
    const size_t state_base = row * key_dim * value_dim;

    for (size_t t = 0; t < seq_len; ++t) {
      const size_t qkv_base = ((b * seq_len + t) * heads + h);
      const Float16* query_row = query_data + qkv_base * key_dim;
      const Float16* key_row = key_data + qkv_base * key_dim;
      const Float16* value_row = value_data + qkv_base * value_dim;

      float key_query_dot = 0.0f;
      for (size_t k = 0; k < key_dim; ++k) {
        key_query_dot +=
            static_cast<float>(key_row[k]) * static_cast<float>(query_row[k]);
      }

      std::fill(kv_mem, kv_mem + value_dim, 0.0f);
      std::fill(output_pre, output_pre + value_dim, 0.0f);
      const float decay =
          static_cast<float>(g_data[(b * seq_len + t) * heads + h]);
      const float beta_value =
          static_cast<float>(beta_data[(b * seq_len + t) * heads + h]);

      for (size_t k = 0; k < key_dim; ++k) {
        const size_t row_base = state_base + k * value_dim;
        const Float16* prev_state =
            t == 0 ? recurrent_state_data + row_base : state_data + row_base;
        const float key_value = static_cast<float>(key_row[k]);
        const float query_value = static_cast<float>(query_row[k]);
        deltanetHvxAccumF16(
            prev_state,
            kv_mem,
            output_pre,
            decay * key_value,
            decay * query_value,
            value_dim);
      }

      deltanetHvxDeltaAndWriteOutputF16(
          output_data + qkv_base * value_dim,
          value_row,
          kv_mem,
          output_pre,
          delta,
          beta_value,
          key_query_dot,
          value_dim);

      for (size_t k = 0; k < key_dim; ++k) {
        const size_t row_base = state_base + k * value_dim;
        const Float16* prev_state =
            t == 0 ? recurrent_state_data + row_base : state_data + row_base;
        deltanetHvxUpdateStateF16(
            prev_state,
            state_data + row_base,
            delta,
            decay,
            static_cast<float>(key_row[k]),
            value_dim);
      }

    }
  }
}

static GraphStatus deltanetCoreImplRawFloat16Hvx(
    Tensor& core_attn_out,
    Tensor& new_recurrent_state,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& g,
    const Tensor& beta,
    const Tensor& recurrent_state) {
  const size_t batch = query.dim(0);
  const size_t seq_len = query.dim(1);
  const size_t heads = query.dim(2);
  const size_t key_dim = query.dim(3);
  const size_t value_dim = value.dim(3);

  if (key_dim > kMaxDeltaNetDim || value_dim > kMaxDeltaNetDim) {
    return GraphStatus::ErrorUnsupported;
  }

  deltanetCoreRawFloat16Hvx(
      static_cast<Float16*>(core_attn_out.raw_data()),
      static_cast<Float16*>(new_recurrent_state.raw_data()),
      static_cast<const Float16*>(query.raw_data_const()),
      static_cast<const Float16*>(key.raw_data_const()),
      static_cast<const Float16*>(value.raw_data_const()),
      static_cast<const Float16*>(g.raw_data_const()),
      static_cast<const Float16*>(beta.raw_data_const()),
      static_cast<const Float16*>(recurrent_state.raw_data_const()),
      batch,
      seq_len,
      heads,
      key_dim,
      value_dim,
      0,
      1);
  return GraphStatus::Success;
}

static GraphStatus deltanetCoreImplRawFloat32Hvx(
    Tensor& core_attn_out,
    Tensor& new_recurrent_state,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& g,
    const Tensor& beta,
    const Tensor& recurrent_state) {
  const size_t batch = query.dim(0);
  const size_t seq_len = query.dim(1);
  const size_t heads = query.dim(2);
  const size_t key_dim = query.dim(3);
  const size_t value_dim = value.dim(3);

  if (key_dim > kMaxDeltaNetDim || value_dim > kMaxDeltaNetDim) {
    return GraphStatus::ErrorUnsupported;
  }

  const float* query_data = static_cast<const float*>(query.raw_data_const());
  const float* key_data = static_cast<const float*>(key.raw_data_const());
  const float* value_data = static_cast<const float*>(value.raw_data_const());
  const float* g_data = static_cast<const float*>(g.raw_data_const());
  const float* beta_data = static_cast<const float*>(beta.raw_data_const());
  const float* recurrent_state_data =
      static_cast<const float*>(recurrent_state.raw_data_const());
  float* output_data = static_cast<float*>(core_attn_out.raw_data());
  float* state_data = static_cast<float*>(new_recurrent_state.raw_data());

  float kv_mem[kMaxDeltaNetDim] __attribute__((aligned(128)));
  float delta[kMaxDeltaNetDim] __attribute__((aligned(128)));
  float output_pre[kMaxDeltaNetDim] __attribute__((aligned(128)));

  for (size_t b = 0; b < batch; ++b) {
    for (size_t t = 0; t < seq_len; ++t) {
      for (size_t h = 0; h < heads; ++h) {
        const size_t qkv_base = ((b * seq_len + t) * heads + h);
        const float* query_row = query_data + qkv_base * key_dim;
        const float* key_row = key_data + qkv_base * key_dim;
        const float* value_row = value_data + qkv_base * value_dim;
        const size_t state_base = (b * heads + h) * key_dim * value_dim;

        float key_query_dot = 0.0f;
        for (size_t k = 0; k < key_dim; ++k) {
          key_query_dot += key_row[k] * query_row[k];
        }

        std::fill(kv_mem, kv_mem + value_dim, 0.0f);
        std::fill(output_pre, output_pre + value_dim, 0.0f);
        const float decay = g_data[(b * seq_len + t) * heads + h];
        const float beta_value = beta_data[(b * seq_len + t) * heads + h];

        for (size_t k = 0; k < key_dim; ++k) {
          const size_t row_base = state_base + k * value_dim;
          const float* prev_state =
              t == 0 ? recurrent_state_data + row_base : state_data + row_base;
          deltanetHvxAccumF32(
              prev_state,
              kv_mem,
              output_pre,
              decay * key_row[k],
              decay * query_row[k],
              value_dim);
        }

        deltanetHvxDeltaAndWriteOutputF32(
            output_data + qkv_base * value_dim,
            value_row,
            kv_mem,
            output_pre,
            delta,
            beta_value,
            key_query_dot,
            value_dim);

        for (size_t k = 0; k < key_dim; ++k) {
          const size_t row_base = state_base + k * value_dim;
          const float* prev_state =
              t == 0 ? recurrent_state_data + row_base : state_data + row_base;
          deltanetHvxUpdateStateF32(
              prev_state,
              state_data + row_base,
              delta,
              decay,
              key_row[k],
              value_dim);
        }

      }
    }
  }

  return GraphStatus::Success;
}

#endif

template <typename Scalar>
static GraphStatus deltanetImplRawSameDtype(
    Tensor& core_attn_out,
    Tensor& new_recurrent_state,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& g,
    const Tensor& beta,
    const Tensor& recurrent_state,
    float scaling_value,
    float epsilon_value) {
  const size_t batch = query.dim(0);
  const size_t seq_len = query.dim(1);
  const size_t heads = query.dim(2);
  const size_t key_dim = query.dim(3);
  const size_t value_dim = value.dim(3);

  if (key_dim > kMaxDeltaNetDim || value_dim > kMaxDeltaNetDim) {
    return GraphStatus::ErrorUnsupported;
  }

  const Scalar* query_data =
      static_cast<const Scalar*>(query.raw_data_const());
  const Scalar* key_data = static_cast<const Scalar*>(key.raw_data_const());
  const Scalar* value_data =
      static_cast<const Scalar*>(value.raw_data_const());
  const Scalar* recurrent_state_data =
      static_cast<const Scalar*>(recurrent_state.raw_data_const());
  Scalar* output_data = static_cast<Scalar*>(core_attn_out.raw_data());
  Scalar* state_data = static_cast<Scalar*>(new_recurrent_state.raw_data());

  float local_query[kMaxDeltaNetDim] __attribute__((aligned(128)));
  float local_key[kMaxDeltaNetDim] __attribute__((aligned(128)));
  float kv_mem[kMaxDeltaNetDim] __attribute__((aligned(128)));
  float delta[kMaxDeltaNetDim] __attribute__((aligned(128)));
  float output_acc[kMaxDeltaNetDim] __attribute__((aligned(128)));

  for (size_t b = 0; b < batch; ++b) {
    for (size_t t = 0; t < seq_len; ++t) {
      for (size_t h = 0; h < heads; ++h) {
        const size_t qkv_base = ((b * seq_len + t) * heads + h);
        const Scalar* query_row = query_data + qkv_base * key_dim;
        const Scalar* key_row = key_data + qkv_base * key_dim;
        const Scalar* value_row = value_data + qkv_base * value_dim;
        const size_t state_base = (b * heads + h) * key_dim * value_dim;

        float query_square_sum = 0.0f;
        float key_square_sum = 0.0f;
        for (size_t k = 0; k < key_dim; ++k) {
          const float query_value = readRawScalar(query_row, k);
          const float key_value = readRawScalar(key_row, k);
          query_square_sum += query_value * query_value;
          key_square_sum += key_value * key_value;
          local_query[k] = query_value;
          local_key[k] = key_value;
        }

        const float query_norm =
            scaling_value / std::sqrt(query_square_sum + epsilon_value);
        const float key_norm = 1.0f / std::sqrt(key_square_sum + epsilon_value);
        for (size_t k = 0; k < key_dim; ++k) {
          local_query[k] *= query_norm;
          local_key[k] *= key_norm;
        }

        std::fill(kv_mem, kv_mem + value_dim, 0.0f);
        const float decay = std::exp(readTimeHeadValue(g, b, t, h));
        const float beta_value = readTimeHeadValue(beta, b, t, h);

        for (size_t k = 0; k < key_dim; ++k) {
          const size_t row_base = state_base + k * value_dim;
          for (size_t v = 0; v < value_dim; ++v) {
            const size_t state_index = row_base + v;
            const float prev_state =
                t == 0 ? readRawScalar(recurrent_state_data, state_index)
                       : readRawScalar(state_data, state_index);
            const float decayed_state = prev_state * decay;
            kv_mem[v] += decayed_state * local_key[k];
          }
        }

        for (size_t v = 0; v < value_dim; ++v) {
          delta[v] = (readRawScalar(value_row, v) - kv_mem[v]) * beta_value;
          output_acc[v] = 0.0f;
        }

        for (size_t k = 0; k < key_dim; ++k) {
          const size_t row_base = state_base + k * value_dim;
          for (size_t v = 0; v < value_dim; ++v) {
            const size_t state_index = row_base + v;
            const float prev_state =
                t == 0 ? readRawScalar(recurrent_state_data, state_index)
                       : readRawScalar(state_data, state_index);
            const float updated_state =
                prev_state * decay + local_key[k] * delta[v];
            writeRawScalar(state_data, state_index, updated_state);
            output_acc[v] += updated_state * local_query[k];
          }
        }

        const size_t output_base = qkv_base * value_dim;
        for (size_t v = 0; v < value_dim; ++v) {
          writeRawScalar(output_data, output_base + v, output_acc[v]);
        }
      }
    }
  }

  return GraphStatus::Success;
}

template <typename Scalar>
static GraphStatus deltanetImplRawQuantSameDtype(
    Tensor& core_attn_out,
    Tensor& new_recurrent_state,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& g,
    const Tensor& beta,
    const Tensor& recurrent_state,
    float scaling_value,
    float epsilon_value) {
  const size_t batch = query.dim(0);
  const size_t seq_len = query.dim(1);
  const size_t heads = query.dim(2);
  const size_t key_dim = query.dim(3);
  const size_t value_dim = value.dim(3);

  if (key_dim > kMaxDeltaNetDim || value_dim > kMaxDeltaNetDim) {
    return GraphStatus::ErrorUnsupported;
  }

  const DTypeScaleOff query_intfc = query.get_dtype_intfc();
  const DTypeScaleOff key_intfc = key.get_dtype_intfc();
  const DTypeScaleOff value_intfc = value.get_dtype_intfc();
  const DTypeScaleOff recurrent_state_intfc =
      recurrent_state.get_dtype_intfc();
  const DTypeScaleOff output_intfc = core_attn_out.get_dtype_intfc();
  const DTypeScaleOff state_intfc = new_recurrent_state.get_dtype_intfc();

  const Scalar* query_data =
      static_cast<const Scalar*>(query.raw_data_const());
  const Scalar* key_data = static_cast<const Scalar*>(key.raw_data_const());
  const Scalar* value_data =
      static_cast<const Scalar*>(value.raw_data_const());
  const Scalar* recurrent_state_data =
      static_cast<const Scalar*>(recurrent_state.raw_data_const());
  Scalar* output_data = static_cast<Scalar*>(core_attn_out.raw_data());
  Scalar* state_data = static_cast<Scalar*>(new_recurrent_state.raw_data());

  float local_query[kMaxDeltaNetDim] __attribute__((aligned(128)));
  float local_key[kMaxDeltaNetDim] __attribute__((aligned(128)));
  float kv_mem[kMaxDeltaNetDim] __attribute__((aligned(128)));
  float delta[kMaxDeltaNetDim] __attribute__((aligned(128)));
  float output_acc[kMaxDeltaNetDim] __attribute__((aligned(128)));

  for (size_t b = 0; b < batch; ++b) {
    for (size_t t = 0; t < seq_len; ++t) {
      for (size_t h = 0; h < heads; ++h) {
        const size_t qkv_base = ((b * seq_len + t) * heads + h);
        const Scalar* query_row = query_data + qkv_base * key_dim;
        const Scalar* key_row = key_data + qkv_base * key_dim;
        const Scalar* value_row = value_data + qkv_base * value_dim;
        const size_t state_base = (b * heads + h) * key_dim * value_dim;

        float query_square_sum = 0.0f;
        float key_square_sum = 0.0f;
        for (size_t k = 0; k < key_dim; ++k) {
          const float query_value =
              readQuantRawScalar(query_row, query_intfc, k);
          const float key_value = readQuantRawScalar(key_row, key_intfc, k);
          query_square_sum += query_value * query_value;
          key_square_sum += key_value * key_value;
          local_query[k] = query_value;
          local_key[k] = key_value;
        }

        const float query_norm =
            scaling_value / std::sqrt(query_square_sum + epsilon_value);
        const float key_norm = 1.0f / std::sqrt(key_square_sum + epsilon_value);
        for (size_t k = 0; k < key_dim; ++k) {
          local_query[k] *= query_norm;
          local_key[k] *= key_norm;
        }

        std::fill(kv_mem, kv_mem + value_dim, 0.0f);
        const float decay = std::exp(readTimeHeadValue(g, b, t, h));
        const float beta_value = readTimeHeadValue(beta, b, t, h);

        for (size_t k = 0; k < key_dim; ++k) {
          const size_t row_base = state_base + k * value_dim;
          for (size_t v = 0; v < value_dim; ++v) {
            const size_t state_index = row_base + v;
            const float prev_state =
                t == 0
                ? readQuantRawScalar(
                      recurrent_state_data, recurrent_state_intfc, state_index)
                : readQuantRawScalar(state_data, state_intfc, state_index);
            const float decayed_state = prev_state * decay;
            kv_mem[v] += decayed_state * local_key[k];
          }
        }

        for (size_t v = 0; v < value_dim; ++v) {
          delta[v] =
              (readQuantRawScalar(value_row, value_intfc, v) - kv_mem[v]) *
              beta_value;
          output_acc[v] = 0.0f;
        }

        for (size_t k = 0; k < key_dim; ++k) {
          const size_t row_base = state_base + k * value_dim;
          for (size_t v = 0; v < value_dim; ++v) {
            const size_t state_index = row_base + v;
            const float prev_state =
                t == 0
                ? readQuantRawScalar(
                      recurrent_state_data, recurrent_state_intfc, state_index)
                : readQuantRawScalar(state_data, state_intfc, state_index);
            const float updated_state =
                prev_state * decay + local_key[k] * delta[v];
            writeQuantRawScalar(
                state_data, state_intfc, state_index, updated_state);
            output_acc[v] += updated_state * local_query[k];
          }
        }

        const size_t output_base = qkv_base * value_dim;
        for (size_t v = 0; v < value_dim; ++v) {
          writeQuantRawScalar(
              output_data, output_intfc, output_base + v, output_acc[v]);
        }
      }
    }
  }

  return GraphStatus::Success;
}

template <typename Scalar>
static GraphStatus deltanetPackedImplRawSameDtype(
    Tensor& packed_out,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& g,
    const Tensor& beta,
    const Tensor& recurrent_state) {
  const size_t batch = query.dim(0);
  const size_t seq_len = query.dim(1);
  const size_t heads = query.dim(2);
  const size_t key_dim = query.dim(3);
  const size_t value_dim = value.dim(3);

  if (key_dim > kMaxDeltaNetDim || value_dim > kMaxDeltaNetDim) {
    return GraphStatus::ErrorUnsupported;
  }

  const size_t output_numel = batch * seq_len * heads * value_dim;

  const Scalar* query_data =
      static_cast<const Scalar*>(query.raw_data_const());
  const Scalar* key_data = static_cast<const Scalar*>(key.raw_data_const());
  const Scalar* value_data =
      static_cast<const Scalar*>(value.raw_data_const());
  const Scalar* recurrent_state_data =
      static_cast<const Scalar*>(recurrent_state.raw_data_const());
  Scalar* packed_data = static_cast<Scalar*>(packed_out.raw_data());

  float kv_mem[kMaxDeltaNetDim] __attribute__((aligned(128)));
  float delta[kMaxDeltaNetDim] __attribute__((aligned(128)));
  float output_acc[kMaxDeltaNetDim] __attribute__((aligned(128)));

  for (size_t b = 0; b < batch; ++b) {
    for (size_t t = 0; t < seq_len; ++t) {
      for (size_t h = 0; h < heads; ++h) {
        const size_t qkv_base = ((b * seq_len + t) * heads + h);
        const Scalar* query_row = query_data + qkv_base * key_dim;
        const Scalar* key_row = key_data + qkv_base * key_dim;
        const Scalar* value_row = value_data + qkv_base * value_dim;

        std::fill(kv_mem, kv_mem + value_dim, 0.0f);
        const float decay = std::exp(readTimeHeadValue(g, b, t, h));
        const float beta_value = readTimeHeadValue(beta, b, t, h);

        for (size_t k = 0; k < key_dim; ++k) {
          const float key_value = readRawScalar(key_row, k);
          for (size_t v = 0; v < value_dim; ++v) {
            const size_t state_index = packedStateIndex(
                output_numel, b, h, k, v, heads, key_dim, value_dim);
            const size_t input_state_index =
                ((b * heads + h) * key_dim + k) * value_dim + v;
            const float prev_state =
                t == 0 ? readRawScalar(recurrent_state_data, input_state_index)
                       : readRawScalar(packed_data, state_index);
            const float decayed_state = prev_state * decay;
            kv_mem[v] += decayed_state * key_value;
          }
        }

        for (size_t v = 0; v < value_dim; ++v) {
          delta[v] = (readRawScalar(value_row, v) - kv_mem[v]) * beta_value;
          output_acc[v] = 0.0f;
        }

        for (size_t k = 0; k < key_dim; ++k) {
          const float key_value = readRawScalar(key_row, k);
          const float query_value = readRawScalar(query_row, k);
          for (size_t v = 0; v < value_dim; ++v) {
            const size_t state_index = packedStateIndex(
                output_numel, b, h, k, v, heads, key_dim, value_dim);
            const size_t input_state_index =
                ((b * heads + h) * key_dim + k) * value_dim + v;
            const float prev_state =
                t == 0 ? readRawScalar(recurrent_state_data, input_state_index)
                       : readRawScalar(packed_data, state_index);
            const float updated_state =
                prev_state * decay + key_value * delta[v];
            writeRawScalar(packed_data, state_index, updated_state);
            output_acc[v] += updated_state * query_value;
          }
        }

        const size_t output_base =
            packedOutputIndex(b, t, h, 0, seq_len, heads, value_dim);
        for (size_t v = 0; v < value_dim; ++v) {
          writeRawScalar(packed_data, output_base + v, output_acc[v]);
        }
      }
    }
  }

  return GraphStatus::Success;
}

template <typename TensorType>
GraphStatus deltanetImpl(
    TensorType& core_attn_out,
    TensorType& new_recurrent_state,
    const TensorType& query,
    const TensorType& key,
    const TensorType& value,
    const TensorType& g,
    const TensorType& beta,
    const TensorType& recurrent_state,
    const Tensor& scaling,
    const Tensor& epsilon) {
  if (!checkRank(query, 4) || !checkRank(key, 4) || !checkRank(value, 4) ||
      !checkRank(recurrent_state, 4)) {
    return GraphStatus::ErrorDimensions;
  }

  const size_t batch = query.dim(0);
  const size_t seq_len = query.dim(1);
  const size_t heads = query.dim(2);
  const size_t key_dim = query.dim(3);
  const size_t value_dim = value.dim(3);

  if (key_dim > kMaxDeltaNetDim || value_dim > kMaxDeltaNetDim) {
    return GraphStatus::ErrorUnsupported;
  }

  if (!checkDim(key, 0, batch) || !checkDim(key, 1, seq_len) ||
      !checkDim(key, 2, heads) || !checkDim(key, 3, key_dim) ||
      !checkDim(value, 0, batch) || !checkDim(value, 1, seq_len) ||
      !checkDim(value, 2, heads) || !checkDim(g, 0, batch) ||
      !checkDim(beta, 0, batch) ||
      !checkDim(recurrent_state, 0, batch) ||
      !checkDim(recurrent_state, 1, heads) ||
      !checkDim(recurrent_state, 2, key_dim) ||
      !checkDim(recurrent_state, 3, value_dim)) {
    return GraphStatus::ErrorDimensions;
  }

  const size_t output_dims[4] = {batch, seq_len, heads, value_dim};
  const size_t state_dims[4] = {batch, heads, key_dim, value_dim};
  core_attn_out.set_dims(output_dims);
  new_recurrent_state.set_dims(state_dims);

  const float scaling_value = readScalarFloat32(scaling);
  const float epsilon_value = readScalarFloat32(epsilon);
  if (isFloat32Dtype(query) && isFloat32Dtype(key) &&
      isFloat32Dtype(value) && isFloat32Dtype(recurrent_state) &&
      isFloat32Dtype(core_attn_out) && isFloat32Dtype(new_recurrent_state)) {
#ifdef __hexagon__
    const GraphStatus hvx_status = deltanetImplRawFloat32Hvx(
        core_attn_out,
        new_recurrent_state,
        query,
        key,
        value,
        g,
        beta,
        recurrent_state,
        scaling_value,
        epsilon_value);
    if (hvx_status == GraphStatus::Success) {
      return hvx_status;
    }
#endif
    const GraphStatus status = deltanetImplRawSameDtype<float>(
        core_attn_out,
        new_recurrent_state,
        query,
        key,
        value,
        g,
        beta,
        recurrent_state,
        scaling_value,
        epsilon_value);
    if (status == GraphStatus::Success) {
      return status;
    }
  }
  if (isFloat16Dtype(query) && isFloat16Dtype(key) &&
      isFloat16Dtype(value) && isFloat16Dtype(recurrent_state) &&
      isFloat16Dtype(core_attn_out) && isFloat16Dtype(new_recurrent_state)) {
#ifdef __hexagon__
    const GraphStatus hvx_status = deltanetImplRawFloat16Hvx(
        core_attn_out,
        new_recurrent_state,
        query,
        key,
        value,
        g,
        beta,
        recurrent_state,
        scaling_value,
        epsilon_value);
    if (hvx_status == GraphStatus::Success) {
      return hvx_status;
    }
#endif
    const GraphStatus status = deltanetImplRawSameDtype<Float16>(
        core_attn_out,
        new_recurrent_state,
        query,
        key,
        value,
        g,
        beta,
        recurrent_state,
        scaling_value,
        epsilon_value);
    if (status == GraphStatus::Success) {
      return status;
    }
  }
  if (isQUInt16Dtype(query) && isQUInt16Dtype(key) &&
      isQUInt16Dtype(value) && isQUInt16Dtype(recurrent_state) &&
      isQUInt16Dtype(core_attn_out) && isQUInt16Dtype(new_recurrent_state)) {
    const GraphStatus status = deltanetImplRawQuantSameDtype<uint16_t>(
        core_attn_out,
        new_recurrent_state,
        query,
        key,
        value,
        g,
        beta,
        recurrent_state,
        scaling_value,
        epsilon_value);
    if (status == GraphStatus::Success) {
      return status;
    }
  }
  if (isQUInt8Dtype(query) && isQUInt8Dtype(key) && isQUInt8Dtype(value) &&
      isQUInt8Dtype(recurrent_state) && isQUInt8Dtype(core_attn_out) &&
      isQUInt8Dtype(new_recurrent_state)) {
    const GraphStatus status = deltanetImplRawQuantSameDtype<uint8_t>(
        core_attn_out,
        new_recurrent_state,
        query,
        key,
        value,
        g,
        beta,
        recurrent_state,
        scaling_value,
        epsilon_value);
    if (status == GraphStatus::Success) {
      return status;
    }
  }

  for (size_t b = 0; b < batch; ++b) {
    for (size_t t = 0; t < seq_len; ++t) {
      for (size_t h = 0; h < heads; ++h) {
        float query_square_sum = 0.0f;
        float key_square_sum = 0.0f;
        for (size_t k = 0; k < key_dim; ++k) {
          const float query_value = static_cast<float>(query(b, t, h, k));
          const float key_value = static_cast<float>(key(b, t, h, k));
          query_square_sum += query_value * query_value;
          key_square_sum += key_value * key_value;
        }

        const float query_norm =
            scaling_value / std::sqrt(query_square_sum + epsilon_value);
        const float key_norm = 1.0f / std::sqrt(key_square_sum + epsilon_value);
        const float decay = std::exp(readTimeHeadValue(g, b, t, h));
        const float beta_value = readTimeHeadValue(beta, b, t, h);

        for (size_t v = 0; v < value_dim; ++v) {
          float kv_mem = 0.0f;
          for (size_t k = 0; k < key_dim; ++k) {
            const float prev_state =
                t == 0 ? static_cast<float>(recurrent_state(b, h, k, v))
                       : static_cast<float>(new_recurrent_state(b, h, k, v));
            const float decayed_state = prev_state * decay;
            const float normalized_key =
                static_cast<float>(key(b, t, h, k)) * key_norm;
            new_recurrent_state(b, h, k, v) = decayed_state;
            kv_mem += decayed_state * normalized_key;
          }

          const float delta =
              (static_cast<float>(value(b, t, h, v)) - kv_mem) * beta_value;
          float output_value = 0.0f;
          for (size_t k = 0; k < key_dim; ++k) {
            const float normalized_key =
                static_cast<float>(key(b, t, h, k)) * key_norm;
            const float normalized_query =
                static_cast<float>(query(b, t, h, k)) * query_norm;
            const float updated_state =
                static_cast<float>(new_recurrent_state(b, h, k, v)) +
                normalized_key * delta;
            new_recurrent_state(b, h, k, v) = updated_state;
            output_value += updated_state * normalized_query;
          }
          core_attn_out(b, t, h, v) = output_value;
        }
      }
    }
  }

  return GraphStatus::Success;
}

template <typename TensorType>
GraphStatus deltanetPackedImpl(
    TensorType& packed_out,
    const TensorType& query,
    const TensorType& key,
    const TensorType& value,
    const TensorType& g,
    const TensorType& beta,
    const TensorType& recurrent_state) {
  if (!checkRank(query, 4) || !checkRank(key, 4) || !checkRank(value, 4) ||
      !checkRank(recurrent_state, 4)) {
    return GraphStatus::ErrorDimensions;
  }

  const size_t batch = query.dim(0);
  const size_t seq_len = query.dim(1);
  const size_t heads = query.dim(2);
  const size_t key_dim = query.dim(3);
  const size_t value_dim = value.dim(3);

  if (key_dim > kMaxDeltaNetDim || value_dim > kMaxDeltaNetDim) {
    return GraphStatus::ErrorUnsupported;
  }

  if (!checkDim(key, 0, batch) || !checkDim(key, 1, seq_len) ||
      !checkDim(key, 2, heads) || !checkDim(key, 3, key_dim) ||
      !checkDim(value, 0, batch) || !checkDim(value, 1, seq_len) ||
      !checkDim(value, 2, heads) || !checkDim(g, 0, batch) ||
      !checkDim(beta, 0, batch) ||
      !checkDim(recurrent_state, 0, batch) ||
      !checkDim(recurrent_state, 1, heads) ||
      !checkDim(recurrent_state, 2, key_dim) ||
      !checkDim(recurrent_state, 3, value_dim)) {
    return GraphStatus::ErrorDimensions;
  }

  const size_t output_numel = batch * seq_len * heads * value_dim;
  const size_t state_numel = batch * heads * key_dim * value_dim;
  const size_t packed_dims[1] = {output_numel + state_numel};
  packed_out.set_dims(packed_dims);

  if (isFloat32Dtype(query) && isFloat32Dtype(key) &&
      isFloat32Dtype(value) && isFloat32Dtype(recurrent_state) &&
      isFloat32Dtype(packed_out)) {
#ifdef __hexagon__
    const GraphStatus hvx_status = deltanetPackedImplRawFloat32Hvx(
        packed_out, query, key, value, g, beta, recurrent_state);
    if (hvx_status == GraphStatus::Success) {
      return hvx_status;
    }
#endif
    const GraphStatus status = deltanetPackedImplRawSameDtype<float>(
        packed_out, query, key, value, g, beta, recurrent_state);
    if (status == GraphStatus::Success) {
      return status;
    }
  }
  if (isFloat16Dtype(query) && isFloat16Dtype(key) &&
      isFloat16Dtype(value) && isFloat16Dtype(recurrent_state) &&
      isFloat16Dtype(packed_out)) {
#ifdef __hexagon__
    const GraphStatus hvx_status = deltanetPackedImplRawFloat16Hvx(
        packed_out, query, key, value, g, beta, recurrent_state);
    if (hvx_status == GraphStatus::Success) {
      return hvx_status;
    }
#endif
    const GraphStatus status = deltanetPackedImplRawSameDtype<Float16>(
        packed_out, query, key, value, g, beta, recurrent_state);
    if (status == GraphStatus::Success) {
      return status;
    }
  }

  for (size_t b = 0; b < batch; ++b) {
    for (size_t t = 0; t < seq_len; ++t) {
      for (size_t h = 0; h < heads; ++h) {
        const float decay = std::exp(readTimeHeadValue(g, b, t, h));
        const float beta_value = readTimeHeadValue(beta, b, t, h);

        for (size_t v = 0; v < value_dim; ++v) {
          float kv_mem = 0.0f;
          for (size_t k = 0; k < key_dim; ++k) {
            const size_t state_index = packedStateIndex(
                output_numel, b, h, k, v, heads, key_dim, value_dim);
            const float prev_state =
                t == 0 ? static_cast<float>(recurrent_state(b, h, k, v))
                       : static_cast<float>(packed_out(state_index));
            const float decayed_state = prev_state * decay;
            const float key_value = static_cast<float>(key(b, t, h, k));
            packed_out(state_index) = decayed_state;
            kv_mem += decayed_state * key_value;
          }

          const float delta =
              (static_cast<float>(value(b, t, h, v)) - kv_mem) * beta_value;
          float output_value = 0.0f;
          for (size_t k = 0; k < key_dim; ++k) {
            const size_t state_index = packedStateIndex(
                output_numel, b, h, k, v, heads, key_dim, value_dim);
            const float key_value = static_cast<float>(key(b, t, h, k));
            const float query_value = static_cast<float>(query(b, t, h, k));
            const float updated_state =
                static_cast<float>(packed_out(state_index)) + key_value * delta;
            packed_out(state_index) = updated_state;
            output_value += updated_state * query_value;
          }

          packed_out(packedOutputIndex(b, t, h, v, seq_len, heads, value_dim)) =
              output_value;
        }
      }
    }
  }

  return GraphStatus::Success;
}

template <typename TensorType>
GraphStatus deltanetCoreImpl(
    TensorType& core_attn_out,
    TensorType& new_recurrent_state,
    const TensorType& query,
    const TensorType& key,
    const TensorType& value,
    const TensorType& g,
    const TensorType& beta,
    const TensorType& recurrent_state) {
  if (!checkRank(query, 4) || !checkRank(key, 4) || !checkRank(value, 4) ||
      !checkRank(recurrent_state, 4)) {
    return GraphStatus::ErrorDimensions;
  }

  const size_t batch = query.dim(0);
  const size_t seq_len = query.dim(1);
  const size_t heads = query.dim(2);
  const size_t key_dim = query.dim(3);
  const size_t value_dim = value.dim(3);

  if (!checkDim(key, 0, batch) || !checkDim(key, 1, seq_len) ||
      !checkDim(key, 2, heads) || !checkDim(key, 3, key_dim) ||
      !checkDim(value, 0, batch) || !checkDim(value, 1, seq_len) ||
      !checkDim(value, 2, heads) || !checkDim(g, 0, batch) ||
      !checkDim(beta, 0, batch) ||
      !checkDim(recurrent_state, 0, batch) ||
      !checkDim(recurrent_state, 1, heads) ||
      !checkDim(recurrent_state, 2, key_dim) ||
      !checkDim(recurrent_state, 3, value_dim)) {
    return GraphStatus::ErrorDimensions;
  }

  const size_t output_dims[4] = {batch, seq_len, heads, value_dim};
  const size_t state_dims[4] = {batch, heads, key_dim, value_dim};
  core_attn_out.set_dims(output_dims);
  new_recurrent_state.set_dims(state_dims);

  if (isFloat32Dtype(query) && isFloat32Dtype(key) &&
      isFloat32Dtype(value) && isFloat32Dtype(recurrent_state) &&
      isFloat32Dtype(core_attn_out) && isFloat32Dtype(new_recurrent_state)) {
#ifdef __hexagon__
    const GraphStatus hvx_status = deltanetCoreImplRawFloat32Hvx(
        core_attn_out,
        new_recurrent_state,
        query,
        key,
        value,
        g,
        beta,
        recurrent_state);
    if (hvx_status == GraphStatus::Success) {
      return hvx_status;
    }
#endif
  }
  if (isFloat16Dtype(query) && isFloat16Dtype(key) &&
      isFloat16Dtype(value) && isFloat16Dtype(recurrent_state) &&
      isFloat16Dtype(core_attn_out) && isFloat16Dtype(new_recurrent_state)) {
#ifdef __hexagon__
    const GraphStatus hvx_status = deltanetCoreImplRawFloat16Hvx(
        core_attn_out,
        new_recurrent_state,
        query,
        key,
        value,
        g,
        beta,
        recurrent_state);
    if (hvx_status == GraphStatus::Success) {
      return hvx_status;
    }
#endif
  }

  float kv_mem[kMaxDeltaNetDim] __attribute__((aligned(128)));
  float delta[kMaxDeltaNetDim] __attribute__((aligned(128)));
  float output_acc[kMaxDeltaNetDim] __attribute__((aligned(128)));

  for (size_t b = 0; b < batch; ++b) {
    for (size_t t = 0; t < seq_len; ++t) {
      for (size_t h = 0; h < heads; ++h) {
        std::fill(kv_mem, kv_mem + value_dim, 0.0f);
        const float decay = readTimeHeadValue(g, b, t, h);
        const float beta_value = readTimeHeadValue(beta, b, t, h);

        for (size_t k = 0; k < key_dim; ++k) {
          const float key_value = static_cast<float>(key(b, t, h, k));
          for (size_t v = 0; v < value_dim; ++v) {
            const float prev_state =
                t == 0 ? static_cast<float>(recurrent_state(b, h, k, v))
                       : static_cast<float>(new_recurrent_state(b, h, k, v));
            kv_mem[v] += prev_state * decay * key_value;
          }
        }

        for (size_t v = 0; v < value_dim; ++v) {
          delta[v] =
              (static_cast<float>(value(b, t, h, v)) - kv_mem[v]) * beta_value;
          output_acc[v] = 0.0f;
        }

        for (size_t k = 0; k < key_dim; ++k) {
          const float key_value = static_cast<float>(key(b, t, h, k));
          const float query_value = static_cast<float>(query(b, t, h, k));
          for (size_t v = 0; v < value_dim; ++v) {
            const float prev_state =
                t == 0 ? static_cast<float>(recurrent_state(b, h, k, v))
                       : static_cast<float>(new_recurrent_state(b, h, k, v));
            const float updated_state =
                prev_state * decay + key_value * delta[v];
            new_recurrent_state(b, h, k, v) = updated_state;
            output_acc[v] += updated_state * query_value;
          }
        }

        for (size_t v = 0; v < value_dim; ++v) {
          core_attn_out(b, t, h, v) = output_acc[v];
        }
      }
    }
  }

  return GraphStatus::Success;
}

template <typename TensorType>
GraphStatus deltanetIdentityImpl(
    TensorType& output,
    const TensorType& input) {
  if (!checkRank(input, 4)) {
    return GraphStatus::ErrorDimensions;
  }

  const size_t dim0 = input.dim(0);
  const size_t dim1 = input.dim(1);
  const size_t dim2 = input.dim(2);
  const size_t dim3 = input.dim(3);
  const size_t output_dims[4] = {dim0, dim1, dim2, dim3};
  output.set_dims(output_dims);

  for (size_t i = 0; i < dim0; ++i) {
    for (size_t j = 0; j < dim1; ++j) {
      for (size_t k = 0; k < dim2; ++k) {
        for (size_t l = 0; l < dim3; ++l) {
          output(i, j, k, l) = static_cast<float>(input(i, j, k, l));
        }
      }
    }
  }

  return GraphStatus::Success;
}

template <typename TensorType>
GraphStatus deltanetDecayStateImpl(
    TensorType& output,
    const TensorType& recurrent_state,
    const TensorType& g) {
  if (!checkRank(recurrent_state, 4)) {
    return GraphStatus::ErrorDimensions;
  }

  const size_t batch = recurrent_state.dim(0);
  const size_t heads = recurrent_state.dim(1);
  const size_t key_dim = recurrent_state.dim(2);
  const size_t value_dim = recurrent_state.dim(3);
  if (!checkDim(g, 0, batch)) {
    return GraphStatus::ErrorDimensions;
  }

  const size_t output_dims[4] = {batch, heads, key_dim, value_dim};
  output.set_dims(output_dims);
  for (size_t b = 0; b < batch; ++b) {
    for (size_t h = 0; h < heads; ++h) {
      const float decay = std::exp(readTimeHeadValue(g, b, 0, h));
      for (size_t k = 0; k < key_dim; ++k) {
        for (size_t v = 0; v < value_dim; ++v) {
          output(b, h, k, v) =
              static_cast<float>(recurrent_state(b, h, k, v)) * decay;
        }
      }
    }
  }

  return GraphStatus::Success;
}

template <typename TensorType>
GraphStatus deltanetStatePassthroughImpl(
    TensorType& output,
    const TensorType& recurrent_state,
    const TensorType& g) {
  (void)g;
  if (!checkRank(recurrent_state, 4)) {
    return GraphStatus::ErrorDimensions;
  }

  const size_t dim0 = recurrent_state.dim(0);
  const size_t dim1 = recurrent_state.dim(1);
  const size_t dim2 = recurrent_state.dim(2);
  const size_t dim3 = recurrent_state.dim(3);
  const size_t output_dims[4] = {dim0, dim1, dim2, dim3};
  output.set_dims(output_dims);
  for (size_t i = 0; i < dim0; ++i) {
    for (size_t j = 0; j < dim1; ++j) {
      for (size_t k = 0; k < dim2; ++k) {
        for (size_t l = 0; l < dim3; ++l) {
          output(i, j, k, l) =
              static_cast<float>(recurrent_state(i, j, k, l));
        }
      }
    }
  }

  return GraphStatus::Success;
}

__attribute__((unused)) static float deltanetCostFunc(const Op* op) {
  (void)op;
  return 0.0f;
}

END_PKG_OP_DEFINITION(PKG_DeltaNet);
