// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#if defined(__hexagon__) || !defined(ANDROID)
#include "../kernels/DeltaNetFloat.h"
#include "../kernels/DeltaNetMixed.h"
#include "../kernels/DeltaNetU16.h"
#include "OpRegistration.h"

#ifndef THIS_PKG_NAME_STR
#define STRINGIZE_DETAIL(X) #X
#define STRINGIZE(X) STRINGIZE_DETAIL(X)
#define THIS_PKG_NAME_STR STRINGIZE(THIS_PKG_NAME)
#endif

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

// 适配带 RuntimeHandle 的 QHPI cost 回调签名，复用上面的形状代价估计。
static float deltanetCoreQhpiCost(
    QHPI_RuntimeHandle*,
    const uint32_t num_inputs,
    const QHPI_Tensor* const* inputs) {
  return deltanetCoreQhpiCost(num_inputs, inputs);
}

static uint32_t deltanetCoreQhpiFp16(
    QHPI_RuntimeHandle* handle,
    uint32_t num_outputs,
    QHPI_Tensor** outputs,
    uint32_t num_inputs,
    const QHPI_Tensor* const* inputs) {
  if (num_inputs != 6 || num_outputs != 2) {
    return QHPI_ERROR_FATAL;
  }

  const QHPI_Shape query_shape = qhpi_tensor_shape(inputs[0]);
  const QHPI_Shape value_shape = qhpi_tensor_shape(inputs[2]);
  if (query_shape.rank != 4 || value_shape.rank != 4) {
    return QHPI_ERROR_FATAL;
  }

  const size_t batch = query_shape.dims[0];
  const size_t seq_len = query_shape.dims[1];
  const size_t heads = query_shape.dims[2];
  const size_t key_dim = query_shape.dims[3];
  const size_t value_dim = value_shape.dims[3];
  if (batch == 0 || seq_len == 0 || heads == 0 || key_dim == 0 ||
      value_dim == 0 || key_dim > kMaxDeltaNetDim ||
      value_dim > kMaxDeltaNetDim) {
    return QHPI_UNSUPPORTED;
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
  return QHPI_SUCCESS;
}

static uint32_t deltanetCoreValidQhpiFp16(
    QHPI_RuntimeHandle* handle,
    uint32_t num_outputs,
    QHPI_Tensor** outputs,
    uint32_t num_inputs,
    const QHPI_Tensor* const* inputs) {
  if (num_inputs != 7 || num_outputs != 2)
    return QHPI_ERROR_FATAL;
  const auto query_shape = qhpi_tensor_shape(inputs[0]);
  const auto key_shape = qhpi_tensor_shape(inputs[1]);
  const auto value_shape = qhpi_tensor_shape(inputs[2]);
  const auto state_shape = qhpi_tensor_shape(inputs[5]);
  const auto valid_shape = qhpi_tensor_shape(inputs[6]);
  const auto output_shape = qhpi_tensor_shape(outputs[0]);
  const auto final_shape = qhpi_tensor_shape(outputs[1]);
  if (query_shape.rank != 4 || key_shape.rank != 4 || value_shape.rank != 4 ||
      state_shape.rank != 4 || output_shape.rank != 4 ||
      final_shape.rank != 4 || valid_shape.rank > 4)
    return QHPI_ERROR_FATAL;
  size_t length_elements = 1;
  for (uint32_t i = 0; i < valid_shape.rank; ++i)
    length_elements *= valid_shape.dims[i];
  if (length_elements != 1)
    return QHPI_ERROR_FATAL;
  const int32_t valid =
      *static_cast<const int32_t*>(qhpi_tensor_raw_data(inputs[6]));
  const size_t batch = query_shape.dims[0], physical = query_shape.dims[1],
               heads = query_shape.dims[2], key_dim = query_shape.dims[3],
               value_dim = value_shape.dims[3];
  if (!batch || !physical || !heads || !key_dim || !value_dim ||
      key_dim > kMaxDeltaNetDim || value_dim > kMaxDeltaNetDim || valid < 0 ||
      size_t(valid) > physical)
    return QHPI_UNSUPPORTED;
  for (uint32_t i = 0; i < 4; ++i) {
    if (key_shape.dims[i] != query_shape.dims[i] ||
        (i < 3 && value_shape.dims[i] != query_shape.dims[i]) ||
        output_shape.dims[i] != value_shape.dims[i] ||
        final_shape.dims[i] != state_shape.dims[i])
      return QHPI_ERROR_FATAL;
  }
  if (state_shape.dims[0] != batch || state_shape.dims[1] != heads ||
      state_shape.dims[2] != key_dim || state_shape.dims[3] != value_dim)
    return QHPI_ERROR_FATAL;
  for (int i : {3, 4}) {
    const auto shape = qhpi_tensor_shape(inputs[i]);
    if (!((shape.rank == 3 && shape.dims[0] == batch &&
           shape.dims[1] == physical && shape.dims[2] == heads) ||
          (shape.rank == 4 && shape.dims[0] == batch &&
           ((shape.dims[1] == physical &&
             ((shape.dims[2] == heads && shape.dims[3] == 1) ||
              (shape.dims[2] == 1 && shape.dims[3] == heads))) ||
            (shape.dims[1] == 1 && shape.dims[2] == physical &&
             shape.dims[3] == heads)))))
      return QHPI_ERROR_FATAL;
  }
  const uint32_t slices = std::max<uint32_t>(qhpi_num_slices(handle), 1);
  const uint32_t slice = qhpi_slice_number(handle);
  auto* output = static_cast<Float16*>(qhpi_tensor_raw_data(outputs[0]));
  auto* state = static_cast<Float16*>(qhpi_tensor_raw_data(outputs[1]));
  const auto* initial =
      static_cast<const Float16*>(qhpi_tensor_raw_data(inputs[5]));
  for (size_t row = slice; row < batch * heads; row += slices) {
    const size_t b = row / heads, h = row % heads;
    for (size_t t = size_t(valid); t < physical; ++t)
      memset(
          output + ((b * physical + t) * heads + h) * value_dim,
          0,
          value_dim * sizeof(Float16));
    if (!valid && state != initial)
      memcpy(
          state + row * key_dim * value_dim,
          initial + row * key_dim * value_dim,
          key_dim * value_dim * sizeof(Float16));
  }
  if (!valid)
    return QHPI_SUCCESS;
  const size_t tiles = deltanetQhpiValueTileCount(
      batch * heads, slices, value_dim, 128 / sizeof(Float16));
  deltanetCoreQhpiRunFp16<true>(
      output,
      state,
      static_cast<const Float16*>(qhpi_tensor_raw_data(inputs[0])),
      static_cast<const Float16*>(qhpi_tensor_raw_data(inputs[1])),
      static_cast<const Float16*>(qhpi_tensor_raw_data(inputs[2])),
      static_cast<const Float16*>(qhpi_tensor_raw_data(inputs[3])),
      static_cast<const Float16*>(qhpi_tensor_raw_data(inputs[4])),
      initial,
      batch,
      physical,
      heads,
      key_dim,
      value_dim,
      slice,
      slices,
      tiles,
      size_t(valid));
  return QHPI_SUCCESS;
}

static uint32_t deltanetCoreQhpiFp16State(
    QHPI_RuntimeHandle* handle,
    uint32_t num_outputs,
    QHPI_Tensor** outputs,
    uint32_t num_inputs,
    const QHPI_Tensor* const* inputs) {
  if (num_inputs != 6 || num_outputs != 2)
    return QHPI_ERROR_FATAL;
  const auto q = qhpi_tensor_shape(inputs[0]);
  const auto k = qhpi_tensor_shape(inputs[1]);
  const auto v = qhpi_tensor_shape(inputs[2]);
  const auto d = qhpi_tensor_shape(inputs[3]);
  const auto b = qhpi_tensor_shape(inputs[4]);
  const auto s = qhpi_tensor_shape(inputs[5]);
  const auto y = qhpi_tensor_shape(outputs[0]);
  const auto ns = qhpi_tensor_shape(outputs[1]);
  if (q.rank != 4 || k.rank != 4 || v.rank != 4 || s.rank != 4 || y.rank != 4 ||
      ns.rank != 4)
    return QHPI_ERROR_FATAL;
  const size_t batch = q.dims[0], tokens = q.dims[1], heads = q.dims[2];
  const size_t kd = q.dims[3], vd = v.dims[3];
  if (!batch || !tokens || !heads || !kd || !vd || kd > kMaxDeltaNetDim ||
      vd > kMaxDeltaNetDim)
    return QHPI_UNSUPPORTED;
  for (size_t i = 0; i < 4; ++i) {
    if (k.dims[i] != q.dims[i] || ns.dims[i] != s.dims[i] ||
        y.dims[i] != v.dims[i])
      return QHPI_ERROR_FATAL;
    if (i < 3 && v.dims[i] != q.dims[i])
      return QHPI_ERROR_FATAL;
  }
  if (s.dims[0] != batch || s.dims[1] != heads || s.dims[2] != kd ||
      s.dims[3] != vd ||
      !op_utils::hasTimeHeadLayout(d, batch, tokens, heads) ||
      !op_utils::hasTimeHeadLayout(b, batch, tokens, heads))
    return QHPI_ERROR_FATAL;
  QHPI_Quant_Parameters quant[5];
  for (size_t i = 0; i < 5; ++i) {
    quant[i] = qhpi_tensor_quant_parameters(inputs[i]);
    if (!op_utils::isValidQuant(quant[i]) ||
        !op_utils::hasU16ZeroOffset(quant[i]))
      return QHPI_ERROR_FATAL;
  }
  const auto oq = qhpi_tensor_quant_parameters(outputs[0]);
  if (!op_utils::isValidQuant(oq) || !op_utils::hasU16ZeroOffset(oq))
    return QHPI_ERROR_FATAL;
  const size_t slices = std::max<uint32_t>(qhpi_num_slices(handle), 1);
  deltanetCoreQhpiRunFp16State(
      static_cast<uint16_t*>(qhpi_tensor_raw_data(outputs[0])),
      static_cast<Float16*>(qhpi_tensor_raw_data(outputs[1])),
      static_cast<const uint16_t*>(qhpi_tensor_raw_data(inputs[0])),
      static_cast<const uint16_t*>(qhpi_tensor_raw_data(inputs[1])),
      static_cast<const uint16_t*>(qhpi_tensor_raw_data(inputs[2])),
      static_cast<const uint16_t*>(qhpi_tensor_raw_data(inputs[3])),
      static_cast<const uint16_t*>(qhpi_tensor_raw_data(inputs[4])),
      static_cast<const Float16*>(qhpi_tensor_raw_data(inputs[5])),
      quant,
      oq,
      d,
      b,
      batch,
      tokens,
      heads,
      kd,
      vd,
      qhpi_slice_number(handle),
      slices,
      deltanetQhpiValueTileCount(batch * heads, slices, vd, 64));
  return QHPI_SUCCESS;
}

static QHPI_Tensor_Signature_v1 deltanet_core_fp16_state_inputs[] = {
    {QHPI_QUINT16,
     QHPI_LAYOUT_FLAT_4,
     QHPI_STORAGE_DIRECT,
     QHPI_MEM_LOC_DDR_ONLY},
    {QHPI_QUINT16,
     QHPI_LAYOUT_FLAT_4,
     QHPI_STORAGE_DIRECT,
     QHPI_MEM_LOC_DDR_ONLY},
    {QHPI_QUINT16,
     QHPI_LAYOUT_FLAT_4,
     QHPI_STORAGE_DIRECT,
     QHPI_MEM_LOC_DDR_ONLY},
    {QHPI_QUINT16,
     QHPI_LAYOUT_FLAT_4,
     QHPI_STORAGE_DIRECT,
     QHPI_MEM_LOC_DDR_ONLY},
    {QHPI_QUINT16,
     QHPI_LAYOUT_FLAT_4,
     QHPI_STORAGE_DIRECT,
     QHPI_MEM_LOC_DDR_ONLY},
    {QHPI_FLOAT16,
     QHPI_LAYOUT_FLAT_4,
     QHPI_STORAGE_DIRECT,
     QHPI_MEM_LOC_DDR_ONLY},
};
static QHPI_Tensor_Signature_v1 deltanet_core_fp16_state_outputs[] = {
    {QHPI_QUINT16,
     QHPI_LAYOUT_FLAT_4,
     QHPI_STORAGE_DIRECT,
     QHPI_MEM_LOC_DDR_ONLY},
    {QHPI_FLOAT16,
     QHPI_LAYOUT_FLAT_4,
     QHPI_STORAGE_DIRECT,
     QHPI_MEM_LOC_DDR_ONLY},
};
static QHPI_Kernel_v1 deltanet_core_fp16_state_kernels[] = {
    {.function_name = THIS_PKG_NAME_STR "::DeltaNetCoreFp16State.mixed.hvx",
     .function = deltanetCoreQhpiFp16State,
     .resources = QHPI_RESOURCE_HVX,
     .source_destructive = false,
     .multithreaded = true,
     .variable_inputs = false,
     .variable_outputs = false,
     .min_inputs = 6,
     .input_signature = deltanet_core_fp16_state_inputs,
     .min_outputs = 2,
     .output_signature = deltanet_core_fp16_state_outputs,
     .cost_function = deltanetCoreQhpiCost,
     .sync_block_size = 0,
     .precomputed_data_size = 0,
     .do_precomputation_function = nullptr,
     .function_with_precomputed_data = nullptr,
     .predicate = nullptr},
};

// QHPI 的 FP32 kernel 入口：检查输入/输出数及主要维度，提取 FP32 指针，
// 取得 worker slice 和 V 分块数，转交 RunFp32 执行递推。
static uint32_t deltanetCoreQhpiFp32(
    QHPI_RuntimeHandle* handle,
    uint32_t num_outputs,
    QHPI_Tensor** outputs,
    uint32_t num_inputs,
    const QHPI_Tensor* const* inputs) {
  if (num_inputs != 6 || num_outputs != 2) {
    return QHPI_ERROR_FATAL;
  }

  const QHPI_Shape query_shape = qhpi_tensor_shape(inputs[0]);
  const QHPI_Shape value_shape = qhpi_tensor_shape(inputs[2]);
  if (query_shape.rank != 4 || value_shape.rank != 4) {
    return QHPI_ERROR_FATAL;
  }

  const size_t batch = query_shape.dims[0];
  const size_t seq_len = query_shape.dims[1];
  const size_t heads = query_shape.dims[2];
  const size_t key_dim = query_shape.dims[3];
  const size_t value_dim = value_shape.dims[3];
  if (batch == 0 || seq_len == 0 || heads == 0 || key_dim == 0 ||
      value_dim == 0 || key_dim > kMaxDeltaNetDim ||
      value_dim > kMaxDeltaNetDim) {
    return QHPI_UNSUPPORTED;
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
  return QHPI_SUCCESS;
}

// QHPI 的 QUInt16 kernel 入口：校验六输入/两输出的形状、gate 布局和量化参数，
// 提取各张量指针、stepsize/zero_offset，并按 worker slice 调度 RunQUint16。
// StateResident=false/true 分别对应 DDR/VTCM 注册项，输入顺序为
// Q,K,V,decay,beta,S_initial。
template <bool StateResident, bool DecodeGate = false>
static uint32_t deltanetCoreQhpiQUint16(
    QHPI_RuntimeHandle* handle,
    uint32_t num_outputs,
    QHPI_Tensor** outputs,
    uint32_t num_inputs,
    const QHPI_Tensor* const* inputs) {
  if (num_inputs != (DecodeGate ? 8u : 6u) || num_outputs != 2) {
    return QHPI_ERROR_FATAL;
  }

  const QHPI_Shape query_shape = qhpi_tensor_shape(inputs[0]);
  const QHPI_Shape key_shape = qhpi_tensor_shape(inputs[1]);
  const QHPI_Shape value_shape = qhpi_tensor_shape(inputs[2]);
  const QHPI_Shape decay_shape = qhpi_tensor_shape(inputs[3]);
  const QHPI_Shape beta_shape = qhpi_tensor_shape(inputs[4]);
  const QHPI_Shape recurrent_shape = qhpi_tensor_shape(inputs[5]);
  if (query_shape.rank != 4 || key_shape.rank != 4 || value_shape.rank != 4 ||
      recurrent_shape.rank != 4 ||
      (decay_shape.rank != 3 && decay_shape.rank != 4) ||
      (beta_shape.rank != 3 && beta_shape.rank != 4)) {
    return QHPI_ERROR_FATAL;
  }

  const size_t batch = query_shape.dims[0];
  const size_t seq_len = query_shape.dims[1];
  const size_t heads = query_shape.dims[2];
  const size_t key_dim = query_shape.dims[3];
  const size_t value_dim = value_shape.dims[3];
  if (batch == 0 || seq_len == 0 || heads == 0 || key_dim == 0 ||
      value_dim == 0 || key_dim > kMaxDeltaNetDim ||
      value_dim > kMaxDeltaNetDim) {
    return QHPI_UNSUPPORTED;
  }
  if (key_shape.dims[0] != batch || key_shape.dims[1] != seq_len ||
      key_shape.dims[2] != heads || key_shape.dims[3] != key_dim ||
      value_shape.dims[0] != batch || value_shape.dims[1] != seq_len ||
      value_shape.dims[2] != heads || recurrent_shape.dims[0] != batch ||
      recurrent_shape.dims[1] != heads || recurrent_shape.dims[2] != key_dim ||
      recurrent_shape.dims[3] != value_dim ||
      !op_utils::hasTimeHeadLayout(decay_shape, batch, seq_len, heads) ||
      !op_utils::hasTimeHeadLayout(beta_shape, batch, seq_len, heads)) {
    return QHPI_ERROR_FATAL;
  }

  const QHPI_Quant_Parameters query_q = qhpi_tensor_quant_parameters(inputs[0]);
  const QHPI_Quant_Parameters key_q = qhpi_tensor_quant_parameters(inputs[1]);
  const QHPI_Quant_Parameters value_q = qhpi_tensor_quant_parameters(inputs[2]);
  const uint16_t* decay_lookup = nullptr;
  QHPI_Quant_Parameters decay_q = qhpi_tensor_quant_parameters(inputs[3]);
  if constexpr (DecodeGate) {
    const auto table = qhpi_tensor_shape(inputs[6]);
    const auto grid = qhpi_tensor_shape(inputs[7]);
    if (seq_len != 1 || table.rank != 4 || table.dims[0] != 1 ||
        table.dims[1] != heads || table.dims[2] != 256 ||
        table.dims[3] != 128 || grid.rank != 4 || grid.dims[3] != 4)
      return QHPI_UNSUPPORTED;
    const auto* parameters =
        static_cast<const float*>(qhpi_tensor_raw_data(inputs[7]));
    const auto activation_q = qhpi_tensor_quant_parameters(inputs[3]);
    if (activation_q.stepsize != parameters[0] ||
        activation_q.zero_offset != int32_t(parameters[1]))
      return QHPI_UNSUPPORTED;
    decay_q = {int32_t(parameters[3]), parameters[2]};
    decay_lookup =
        static_cast<const uint16_t*>(qhpi_tensor_raw_data(inputs[6]));
  }
  const QHPI_Quant_Parameters beta_q = qhpi_tensor_quant_parameters(inputs[4]);
  const QHPI_Quant_Parameters recurrent_state_q =
      qhpi_tensor_quant_parameters(inputs[5]);
  const QHPI_Quant_Parameters output_q =
      qhpi_tensor_quant_parameters(outputs[0]);
  const QHPI_Quant_Parameters state_q =
      qhpi_tensor_quant_parameters(outputs[1]);
  if (!op_utils::isValidQuant(query_q) || !op_utils::isValidQuant(key_q) ||
      !op_utils::isValidQuant(value_q) || !op_utils::isValidQuant(decay_q) ||
      !op_utils::isValidQuant(beta_q) ||
      !op_utils::isValidQuant(recurrent_state_q) ||
      !op_utils::isValidQuant(output_q) || !op_utils::isValidQuant(state_q)) {
    return QHPI_ERROR_FATAL;
  }

  const uint32_t num_slices = std::max<uint32_t>(qhpi_num_slices(handle), 1);
  const uint32_t slice = qhpi_slice_number(handle);
  const size_t value_tile_count = deltanetQhpiValueTileCount(
      batch * heads, num_slices, value_dim, 128 / sizeof(uint16_t));
  deltanetCoreQhpiRunQUint16<StateResident, DecodeGate>(
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
      value_tile_count,
      decay_lookup);
  return QHPI_SUCCESS;
}

// QHPI 的 Stream kernel 入口：校验 [B,H,V,K] state、gate 布局和量化参数。
// 提取 U16 数据及 worker slice，调度 RunQUint16Stream；调用方须显式使用列连续
// state。
static uint32_t deltanetCoreQhpiQUint16Stream(
    QHPI_RuntimeHandle* handle,
    uint32_t num_outputs,
    QHPI_Tensor** outputs,
    uint32_t num_inputs,
    const QHPI_Tensor* const* inputs) {
  if (num_inputs != 6 || num_outputs != 2) {
    return QHPI_ERROR_FATAL;
  }
  const QHPI_Shape query_shape = qhpi_tensor_shape(inputs[0]);
  const QHPI_Shape key_shape = qhpi_tensor_shape(inputs[1]);
  const QHPI_Shape value_shape = qhpi_tensor_shape(inputs[2]);
  const QHPI_Shape decay_shape = qhpi_tensor_shape(inputs[3]);
  const QHPI_Shape beta_shape = qhpi_tensor_shape(inputs[4]);
  const QHPI_Shape recurrent_shape = qhpi_tensor_shape(inputs[5]);
  if (query_shape.rank != 4 || key_shape.rank != 4 || value_shape.rank != 4 ||
      recurrent_shape.rank != 4 ||
      (decay_shape.rank != 3 && decay_shape.rank != 4) ||
      (beta_shape.rank != 3 && beta_shape.rank != 4)) {
    return QHPI_ERROR_FATAL;
  }
  const size_t batch = query_shape.dims[0];
  const size_t seq_len = query_shape.dims[1];
  const size_t heads = query_shape.dims[2];
  const size_t key_dim = query_shape.dims[3];
  const size_t value_dim = value_shape.dims[3];
  if (batch == 0 || seq_len == 0 || heads == 0 || key_dim == 0 ||
      value_dim == 0 || key_dim > kMaxDeltaNetDim ||
      value_dim > kMaxDeltaNetDim) {
    return QHPI_UNSUPPORTED;
  }
  if (key_shape.dims[0] != batch || key_shape.dims[1] != seq_len ||
      key_shape.dims[2] != heads || key_shape.dims[3] != key_dim ||
      value_shape.dims[0] != batch || value_shape.dims[1] != seq_len ||
      value_shape.dims[2] != heads || recurrent_shape.dims[0] != batch ||
      recurrent_shape.dims[1] != heads ||
      recurrent_shape.dims[2] != value_dim ||
      recurrent_shape.dims[3] != key_dim ||
      !op_utils::hasTimeHeadLayout(decay_shape, batch, seq_len, heads) ||
      !op_utils::hasTimeHeadLayout(beta_shape, batch, seq_len, heads)) {
    return QHPI_ERROR_FATAL;
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
  if (!op_utils::isValidQuant(query_q) || !op_utils::isValidQuant(key_q) ||
      !op_utils::isValidQuant(value_q) || !op_utils::isValidQuant(decay_q) ||
      !op_utils::isValidQuant(beta_q) ||
      !op_utils::isValidQuant(recurrent_state_q) ||
      !op_utils::isValidQuant(output_q) || !op_utils::isValidQuant(state_q)) {
    return QHPI_ERROR_FATAL;
  }
  const uint32_t num_slices = std::max<uint32_t>(qhpi_num_slices(handle), 1);
  deltanetCoreQhpiRunQUint16Stream(
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
      qhpi_slice_number(handle),
      num_slices);
  return QHPI_SUCCESS;
}

// QHPI 的纯 FP16 累加入口：检查输入/输出数及主要维度，设置 head/V 分块调度。
// 转交 RunFp16AccumF16，避免与普通 FP16 入口所使用的 FP32 scratch 混淆。
static uint32_t deltanetCoreQhpiFp16AccumF16(
    QHPI_RuntimeHandle* handle,
    uint32_t num_outputs,
    QHPI_Tensor** outputs,
    uint32_t num_inputs,
    const QHPI_Tensor* const* inputs) {
  if (num_inputs != 6 || num_outputs != 2) {
    return QHPI_ERROR_FATAL;
  }

  const QHPI_Shape query_shape = qhpi_tensor_shape(inputs[0]);
  const QHPI_Shape value_shape = qhpi_tensor_shape(inputs[2]);
  if (query_shape.rank != 4 || value_shape.rank != 4) {
    return QHPI_ERROR_FATAL;
  }

  const size_t batch = query_shape.dims[0];
  const size_t seq_len = query_shape.dims[1];
  const size_t heads = query_shape.dims[2];
  const size_t key_dim = query_shape.dims[3];
  const size_t value_dim = value_shape.dims[3];
  if (batch == 0 || seq_len == 0 || heads == 0 || key_dim == 0 ||
      value_dim == 0 || key_dim > kMaxDeltaNetDim ||
      value_dim > kMaxDeltaNetDim) {
    return QHPI_UNSUPPORTED;
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
  return QHPI_SUCCESS;
}

// 以下输入 signature 的顺序均为 Q,K,V,decay,beta,S_initial；Direct
// 表示直接访问缓冲区。 普通 FP16 和 Fp16AccumF16 共用 FP16 存储
// signature，计算精度由各自 kernel 决定。
static QHPI_Tensor_Signature_v1 deltanet_core_fp16_inputs[] = {
    {QHPI_FLOAT16,
     QHPI_LAYOUT_FLAT_4,
     QHPI_STORAGE_DIRECT,
     QHPI_MEM_LOC_DDR_ONLY},
    {QHPI_FLOAT16,
     QHPI_LAYOUT_FLAT_4,
     QHPI_STORAGE_DIRECT,
     QHPI_MEM_LOC_DDR_ONLY},
    {QHPI_FLOAT16,
     QHPI_LAYOUT_FLAT_4,
     QHPI_STORAGE_DIRECT,
     QHPI_MEM_LOC_DDR_ONLY},
    {QHPI_FLOAT16,
     QHPI_LAYOUT_FLAT_4,
     QHPI_STORAGE_DIRECT,
     QHPI_MEM_LOC_DDR_ONLY},
    {QHPI_FLOAT16,
     QHPI_LAYOUT_FLAT_4,
     QHPI_STORAGE_DIRECT,
     QHPI_MEM_LOC_DDR_ONLY},
    {QHPI_FLOAT16,
     QHPI_LAYOUT_FLAT_4,
     QHPI_STORAGE_DIRECT,
     QHPI_MEM_LOC_DDR_ONLY},
};

// 两输出依次为 y [B,T,H,V]、最终 state [B,H,K,V]，均为 DDR 中的 FP16。
static QHPI_Tensor_Signature_v1 deltanet_core_fp16_outputs[] = {
    {QHPI_FLOAT16,
     QHPI_LAYOUT_FLAT_4,
     QHPI_STORAGE_DIRECT,
     QHPI_MEM_LOC_DDR_ONLY},
    {QHPI_FLOAT16,
     QHPI_LAYOUT_FLAT_4,
     QHPI_STORAGE_DIRECT,
     QHPI_MEM_LOC_DDR_ONLY},
};

static QHPI_Tensor_Signature_v1 deltanet_core_valid_fp16_inputs[] = {
    {QHPI_FLOAT16,
     QHPI_LAYOUT_FLAT_4,
     QHPI_STORAGE_DIRECT,
     QHPI_MEM_LOC_DDR_ONLY},
    {QHPI_FLOAT16,
     QHPI_LAYOUT_FLAT_4,
     QHPI_STORAGE_DIRECT,
     QHPI_MEM_LOC_DDR_ONLY},
    {QHPI_FLOAT16,
     QHPI_LAYOUT_FLAT_4,
     QHPI_STORAGE_DIRECT,
     QHPI_MEM_LOC_DDR_ONLY},
    {QHPI_FLOAT16,
     QHPI_LAYOUT_FLAT_4,
     QHPI_STORAGE_DIRECT,
     QHPI_MEM_LOC_DDR_ONLY},
    {QHPI_FLOAT16,
     QHPI_LAYOUT_FLAT_4,
     QHPI_STORAGE_DIRECT,
     QHPI_MEM_LOC_DDR_ONLY},
    {QHPI_FLOAT16,
     QHPI_LAYOUT_FLAT_4,
     QHPI_STORAGE_DIRECT,
     QHPI_MEM_LOC_DDR_ONLY},
    {QHPI_INT32,
     QHPI_LAYOUT_FLAT_4,
     QHPI_STORAGE_DIRECT,
     QHPI_MEM_LOC_DDR_ONLY},
};

static QHPI_Tensor_Signature_v1 deltanet_core_valid_fp16_vtcm_outputs[] = {
    {QHPI_FLOAT16,
     QHPI_LAYOUT_FLAT_4,
     QHPI_STORAGE_DIRECT,
     QHPI_MEM_LOC_DDR_ONLY},
    {QHPI_FLOAT16,
     QHPI_LAYOUT_FLAT_4,
     QHPI_STORAGE_DIRECT,
     QHPI_MEM_LOC_TCM_ONLY},
};

// FP32 六输入的 dtype/布局/存储位置约束：直接读取 DDR 缓冲区。
static QHPI_Tensor_Signature_v1 deltanet_core_fp32_inputs[] = {
    {QHPI_FLOAT32,
     QHPI_LAYOUT_FLAT_4,
     QHPI_STORAGE_DIRECT,
     QHPI_MEM_LOC_DDR_ONLY},
    {QHPI_FLOAT32,
     QHPI_LAYOUT_FLAT_4,
     QHPI_STORAGE_DIRECT,
     QHPI_MEM_LOC_DDR_ONLY},
    {QHPI_FLOAT32,
     QHPI_LAYOUT_FLAT_4,
     QHPI_STORAGE_DIRECT,
     QHPI_MEM_LOC_DDR_ONLY},
    {QHPI_FLOAT32,
     QHPI_LAYOUT_FLAT_4,
     QHPI_STORAGE_DIRECT,
     QHPI_MEM_LOC_DDR_ONLY},
    {QHPI_FLOAT32,
     QHPI_LAYOUT_FLAT_4,
     QHPI_STORAGE_DIRECT,
     QHPI_MEM_LOC_DDR_ONLY},
    {QHPI_FLOAT32,
     QHPI_LAYOUT_FLAT_4,
     QHPI_STORAGE_DIRECT,
     QHPI_MEM_LOC_DDR_ONLY},
};

// FP32 输出 y 和最终 state 均使用 Flat4 布局并存放在 DDR。
static QHPI_Tensor_Signature_v1 deltanet_core_fp32_outputs[] = {
    {QHPI_FLOAT32,
     QHPI_LAYOUT_FLAT_4,
     QHPI_STORAGE_DIRECT,
     QHPI_MEM_LOC_DDR_ONLY},
    {QHPI_FLOAT32,
     QHPI_LAYOUT_FLAT_4,
     QHPI_STORAGE_DIRECT,
     QHPI_MEM_LOC_DDR_ONLY},
};

// 量化 core 的六输入均为 DDR 中的 QUInt16，各自零点和尺度从 tensor 元数据读取。
static QHPI_Tensor_Signature_v1 deltanet_core_quint16_inputs[] = {
    {QHPI_QUINT16,
     QHPI_LAYOUT_FLAT_4,
     QHPI_STORAGE_DIRECT,
     QHPI_MEM_LOC_DDR_ONLY},
    {QHPI_QUINT16,
     QHPI_LAYOUT_FLAT_4,
     QHPI_STORAGE_DIRECT,
     QHPI_MEM_LOC_DDR_ONLY},
    {QHPI_QUINT16,
     QHPI_LAYOUT_FLAT_4,
     QHPI_STORAGE_DIRECT,
     QHPI_MEM_LOC_DDR_ONLY},
    {QHPI_QUINT16,
     QHPI_LAYOUT_FLAT_4,
     QHPI_STORAGE_DIRECT,
     QHPI_MEM_LOC_DDR_ONLY},
    {QHPI_QUINT16,
     QHPI_LAYOUT_FLAT_4,
     QHPI_STORAGE_DIRECT,
     QHPI_MEM_LOC_DDR_ONLY},
    {QHPI_QUINT16,
     QHPI_LAYOUT_FLAT_4,
     QHPI_STORAGE_DIRECT,
     QHPI_MEM_LOC_DDR_ONLY},
};

// DDR 量化输出 signature：y 和最终 state 均为 QUInt16。
// Stream 也复用此 dtype/存储约束，但它的 state 维度顺序为 [B,H,V,K]。
static QHPI_Tensor_Signature_v1 deltanet_core_quint16_outputs[] = {
    {QHPI_QUINT16,
     QHPI_LAYOUT_FLAT_4,
     QHPI_STORAGE_DIRECT,
     QHPI_MEM_LOC_DDR_ONLY},
    {QHPI_QUINT16,
     QHPI_LAYOUT_FLAT_4,
     QHPI_STORAGE_DIRECT,
     QHPI_MEM_LOC_DDR_ONLY},
};

// 只有 final_state 放入 VTCM；y 和所有输入仍在 DDR。
// QNN 分配完整常驻 state 并管理图输出传输，本 kernel 未实现手动 DMA。
static QHPI_Tensor_Signature_v1 deltanet_core_quint16_vtcm_outputs[] = {
    {QHPI_QUINT16,
     QHPI_LAYOUT_FLAT_4,
     QHPI_STORAGE_DIRECT,
     QHPI_MEM_LOC_DDR_ONLY},
    {QHPI_QUINT16,
     QHPI_LAYOUT_FLAT_4,
     QHPI_STORAGE_DIRECT,
     QHPI_MEM_LOC_TCM_ONLY},
};

// 同一 DeltaNetCore 按 signature 选择普通 FP16、FP32 或 QUInt16 DDR 实现。
// resources=HVX 请求 SIMD 资源；multithreaded=true 启用 QHPI worker slice
// 调度， 各 worker 自行划分 head/V 任务，不产生 SIMT lane 线程。
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
    // 新 DDR：<false> -> RunQUint16<false> -> RunQUint16Tile<false>。
    // 名称 fixed 下也包含 HVX F32 更新保底分支。
    {.function_name = THIS_PKG_NAME_STR "::DeltaNetCore.quint16.fixed",
     .function = deltanetCoreQhpiQUint16<false>,
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

// 新 VTCM：<true> 使用相同优化内核，区别是常驻 state 和首步拷贝。
static QHPI_Kernel_v1 deltanet_core_vtcm_kernels[] = {
    {.function_name = THIS_PKG_NAME_STR "::DeltaNetCoreVtcm.quint16.hvx",
     .function = deltanetCoreQhpiQUint16<true>,
     .resources = QHPI_RESOURCE_HVX,
     .source_destructive = false,
     .multithreaded = true,
     .variable_inputs = false,
     .variable_outputs = false,
     .min_inputs = 6,
     .input_signature = deltanet_core_quint16_inputs,
     .min_outputs = 2,
     .output_signature = deltanet_core_quint16_vtcm_outputs,
     .cost_function = deltanetCoreQhpiCost,
     .sync_block_size = 0,
     .precomputed_data_size = 0,
     .do_precomputation_function = nullptr,
     .function_with_precomputed_data = nullptr,
     .predicate = nullptr},
};

// 注册列连续 state 的实验实现，仅接受
// QUInt16；数学递推相同，访存/归约方式不同。
static QHPI_Kernel_v1 deltanet_core_stream_kernels[] = {
    {.function_name = THIS_PKG_NAME_STR "::DeltaNetCoreStream.quint16.hvx",
     .function = deltanetCoreQhpiQUint16Stream,
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

// 单独注册逐步 FP16 累加的实现，以便与普通 FP16 存储、FP32 累加路径对照。
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

static QHPI_Kernel_v1 deltanet_core_valid_kernels[] = {
    {.function_name = THIS_PKG_NAME_STR "::DeltaNetCoreValid.fp16.hvx",
     .function = deltanetCoreValidQhpiFp16,
     .resources = QHPI_RESOURCE_HVX,
     .source_destructive = false,
     .multithreaded = true,
     .variable_inputs = false,
     .variable_outputs = false,
     .min_inputs = 7,
     .input_signature = deltanet_core_valid_fp16_inputs,
     .min_outputs = 2,
     .output_signature = deltanet_core_fp16_outputs,
     .cost_function = deltanetCoreQhpiCost,
     .sync_block_size = 0,
     .precomputed_data_size = 0,
     .do_precomputation_function = nullptr,
     .function_with_precomputed_data = nullptr,
     .predicate = nullptr},
};

static QHPI_Kernel_v1 deltanet_core_valid_vtcm_kernels[] = {
    {.function_name = THIS_PKG_NAME_STR "::DeltaNetCoreValidVtcm.fp16.hvx",
     .function = deltanetCoreValidQhpiFp16,
     .resources = QHPI_RESOURCE_HVX,
     .source_destructive = false,
     .multithreaded = true,
     .variable_inputs = false,
     .variable_outputs = false,
     .min_inputs = 7,
     .input_signature = deltanet_core_valid_fp16_inputs,
     .min_outputs = 2,
     .output_signature = deltanet_core_valid_fp16_vtcm_outputs,
     .cost_function = deltanetCoreQhpiCost,
     .sync_block_size = 0,
     .precomputed_data_size = 0,
     .do_precomputation_function = nullptr,
     .function_with_precomputed_data = nullptr,
     .predicate = nullptr},
};

// 对外算子名称与 kernel 表的绑定。rewrite/build_tile 等回调为空，
// 任务切分由上述 self-sliced 入口通过 QHPI worker slice 自行完成。
static QHPI_Tensor_Signature_v1 gated_inputs[] = {
    {QHPI_QUINT16,
     QHPI_LAYOUT_FLAT_4,
     QHPI_STORAGE_DIRECT,
     QHPI_MEM_LOC_DDR_OR_TCM},
    {QHPI_QUINT16,
     QHPI_LAYOUT_FLAT_4,
     QHPI_STORAGE_DIRECT,
     QHPI_MEM_LOC_DDR_OR_TCM},
    {QHPI_QUINT16,
     QHPI_LAYOUT_FLAT_4,
     QHPI_STORAGE_DIRECT,
     QHPI_MEM_LOC_DDR_OR_TCM},
    {QHPI_QUINT16,
     QHPI_LAYOUT_FLAT_4,
     QHPI_STORAGE_DIRECT,
     QHPI_MEM_LOC_DDR_OR_TCM},
    {QHPI_QUINT16,
     QHPI_LAYOUT_FLAT_4,
     QHPI_STORAGE_DIRECT,
     QHPI_MEM_LOC_DDR_OR_TCM},
    {QHPI_QUINT16,
     QHPI_LAYOUT_FLAT_4,
     QHPI_STORAGE_DIRECT,
     QHPI_MEM_LOC_DDR_OR_TCM},
    {QHPI_INT32,
     QHPI_LAYOUT_FLAT_4,
     QHPI_STORAGE_DIRECT,
     QHPI_MEM_LOC_DDR_ONLY},
    {QHPI_FLOAT32,
     QHPI_LAYOUT_FLAT_4,
     QHPI_STORAGE_DIRECT,
     QHPI_MEM_LOC_DDR_OR_TCM}};
static QHPI_Kernel_v1 gated_kernels[] = {
    {.function_name = THIS_PKG_NAME_STR "::DeltaNetCoreGatedVtcm.quint16.hvx",
     .function = deltanetCoreQhpiQUint16<true, true>,
     .resources = QHPI_RESOURCE_HVX,
     .source_destructive = false,
     .multithreaded = true,
     .min_inputs = 8,
     .input_signature = gated_inputs,
     .min_outputs = 2,
     .output_signature = deltanet_core_quint16_vtcm_outputs,
     .cost_function = deltanetCoreQhpiCost}};
static QHPI_OpInfo_v1 deltanet_qhpi_ops[] = {
    {.name = THIS_PKG_NAME_STR "::DeltaNetCoreGatedVtcm",
     .num_kernels = 1,
     .kernels = gated_kernels},
    {.name = THIS_PKG_NAME_STR "::DeltaNetCoreValidVtcm",
     .num_kernels = sizeof(deltanet_core_valid_vtcm_kernels) /
         sizeof(deltanet_core_valid_vtcm_kernels[0]),
     .kernels = deltanet_core_valid_vtcm_kernels,
     .early_rewrite = nullptr,
     .shape_required = nullptr,
     .shape_legalized = nullptr,
     .build_tile = nullptr,
     .late_rewrite = nullptr},
    {.name = THIS_PKG_NAME_STR "::DeltaNetCoreValid",
     .num_kernels = sizeof(deltanet_core_valid_kernels) /
         sizeof(deltanet_core_valid_kernels[0]),
     .kernels = deltanet_core_valid_kernels,
     .early_rewrite = nullptr,
     .shape_required = nullptr,
     .shape_legalized = nullptr,
     .build_tile = nullptr,
     .late_rewrite = nullptr},
    {.name = THIS_PKG_NAME_STR "::DeltaNetCoreFp16State",
     .num_kernels = sizeof(deltanet_core_fp16_state_kernels) /
         sizeof(deltanet_core_fp16_state_kernels[0]),
     .kernels = deltanet_core_fp16_state_kernels,
     .early_rewrite = nullptr,
     .shape_required = nullptr,
     .shape_legalized = nullptr,
     .build_tile = nullptr,
     .late_rewrite = nullptr},
    {.name = THIS_PKG_NAME_STR "::DeltaNetCoreStream",
     .num_kernels = sizeof(deltanet_core_stream_kernels) /
         sizeof(deltanet_core_stream_kernels[0]),
     .kernels = deltanet_core_stream_kernels,
     .early_rewrite = nullptr,
     .shape_required = nullptr,
     .shape_legalized = nullptr,
     .build_tile = nullptr,
     .late_rewrite = nullptr},
    {.name = THIS_PKG_NAME_STR "::DeltaNetCoreVtcm",
     .num_kernels = sizeof(deltanet_core_vtcm_kernels) /
         sizeof(deltanet_core_vtcm_kernels[0]),
     .kernels = deltanet_core_vtcm_kernels,
     .early_rewrite = nullptr,
     .shape_required = nullptr,
     .shape_legalized = nullptr,
     .build_tile = nullptr,
     .late_rewrite = nullptr},
    {.name = THIS_PKG_NAME_STR "::DeltaNetCore",
     .num_kernels =
         sizeof(deltanet_core_kernels) / sizeof(deltanet_core_kernels[0]),
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

// 向 QHPI 注册上述算子名称及各自的 kernel/signature/cost 元数据。
// 由 OpPackage 接口初始化时调用，供 QNN 选择 DSP/x86 实现；本函数不执行
// recurrence。
void register_deltanet_qhpi_ops() {
  qhpi_register_ops_v1(
      sizeof(deltanet_qhpi_ops) / sizeof(deltanet_qhpi_ops[0]),
      deltanet_qhpi_ops,
      THIS_PKG_NAME_STR);
}

#else

// Android 桥接库的空入口：保留统一链接符号，实际 QHPI 计算内核在 DSP/x86
// 编译分支中。 这里不执行计算，也不是 CPU 推理回退实现。
void register_deltanet_qhpi_ops() {}

#endif
