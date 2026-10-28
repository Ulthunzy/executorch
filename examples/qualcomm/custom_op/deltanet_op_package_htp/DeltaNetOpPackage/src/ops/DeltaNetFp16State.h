// Typed mixed core: five affine U16 inputs, FP16 state, U16 readout.
// Included after the shared FP16/FP32 HVX helpers in DeltaNetQhpi.cpp.
#pragma once

static void deltanetCoreQhpiRunFp16State(
    uint16_t* output, Float16* state, const uint16_t* query,
    const uint16_t* key, const uint16_t* value, const uint16_t* decay,
    const uint16_t* beta, const Float16* initial,
    const QHPI_Quant_Parameters* quant, QHPI_Quant_Parameters output_quant,
    const QHPI_Shape& decay_shape, const QHPI_Shape& beta_shape,
    size_t batch, size_t tokens, size_t heads, size_t key_dim,
    size_t value_dim, size_t slice, size_t slices, size_t tiles) {
  alignas(128) float key_float[kMaxDeltaNetDim], query_float[kMaxDeltaNetDim];
  alignas(128) int32_t key_center[kMaxDeltaNetDim], query_center[kMaxDeltaNetDim];
  alignas(128) float kv[kMaxDeltaNetDim], pre[kMaxDeltaNetDim], delta[kMaxDeltaNetDim];
  DeltaNetPhaseProfile profile;
  const size_t tasks = batch * heads * tiles;
  const HVX_Vector key_scale = op_utils::hvxSplatF32(quant[1].stepsize);
  const HVX_Vector query_scale = op_utils::hvxSplatF32(quant[0].stepsize);
  const HVX_Vector value_scale = op_utils::hvxSplatF32(quant[2].stepsize);
  const HVX_Vector output_scale = op_utils::hvxSplatF32(1.f / output_quant.stepsize);
  const HVX_Vector output_zero = op_utils::hvxSplatF32(float(output_quant.zero_offset) + .5f);
  for (size_t task = slice; task < tasks; task += slices) {
    const size_t row = task / tiles, tile = task % tiles;
    const size_t batch_id = row / heads, head = row % heads;
    const size_t begin = value_dim * tile / tiles;
    const size_t count = value_dim * (tile + 1) / tiles - begin;
    if (!count) continue;
    const size_t state_base = row * key_dim * value_dim + begin;
    profile.start();
    profile.record(4);
    for (size_t t = 0; t < tokens; ++t) {
      const size_t row_index = (batch_id * tokens + t) * heads + head;
      const auto* q = query + row_index * key_dim;
      const auto* k = key + row_index * key_dim;
      const auto* v = value + row_index * value_dim + begin;
      int64_t dot = 0;
      if (key_dim % 64 == 0) {
        bool unused;
        dot = deltanetQhpiHvxPrepareQkU16(q, k, query_center, key_center,
            quant[0].zero_offset, quant[1].zero_offset, key_dim, unused);
      } else {
        for (size_t i = 0; i < key_dim; ++i) {
          key_center[i] = int(k[i]) - quant[1].zero_offset;
          query_center[i] = int(q[i]) - quant[0].zero_offset;
          dot += int64_t(key_center[i]) * query_center[i];
        }
      }
      size_t i = 0;
      for (; i + 32 <= key_dim; i += 32) {
        vmemu(key_float + i) = op_utils::hvxMulF32(
            Q6_Vsf_equals_Vw(vmemu(key_center + i)), key_scale);
        vmemu(query_float + i) = op_utils::hvxMulF32(
            Q6_Vsf_equals_Vw(vmemu(query_center + i)), query_scale);
      }
      for (; i < key_dim; ++i) {
        key_float[i] = float(key_center[i]) * quant[1].stepsize;
        query_float[i] = float(query_center[i]) * quant[0].stepsize;
      }
      const float d = op_utils::dequantU16(
          decay[op_utils::timeHeadIndex(decay_shape, batch_id, t, head)], quant[3]);
      const float b = op_utils::dequantU16(
          beta[op_utils::timeHeadIndex(beta_shape, batch_id, t, head)], quant[4]);
      const float qk = float(double(dot) * quant[0].stepsize * quant[1].stepsize);
      const auto* previous = (t ? state : initial) + state_base;
      const bool aligned = reinterpret_cast<uintptr_t>(previous) % 128 == 0 &&
          value_dim % 64 == 0;
      profile.record(0);
      if (count % 64 == 0) {
#if __HVX_ARCH__ == 81
        if (aligned && key_dim == 128 && value_dim == 128 && count == 128)
          deltanetQhpiHvxReduceStateF16<true, 2>(
              previous, kv, pre, key_float, query_float, d, 128, 128, 128);
        else
#endif
        if (aligned && count % 128 == 0)
          deltanetQhpiHvxReduceStateF16<true, 2>(previous, kv, pre,
              key_float, query_float, d, key_dim, value_dim, count);
        else
          deltanetQhpiHvxReduceStateF16<false>(previous, kv, pre,
              key_float, query_float, d, key_dim, value_dim, count);
      } else {
        for (size_t col = 0; col < count; ++col) {
          kv[col] = float(previous[col]) * (d * key_float[0]);
          pre[col] = float(previous[col]) * (d * query_float[0]);
          for (size_t j = 1; j < key_dim; ++j) {
            kv[col] += float(previous[j * value_dim + col]) * (d * key_float[j]);
            pre[col] += float(previous[j * value_dim + col]) * (d * query_float[j]);
          }
        }
      }
      profile.record(1);
      auto* out = output + row_index * value_dim + begin;
      const HVX_Vector beta_scale = op_utils::hvxSplatF32(b);
      const HVX_Vector dot_scale = op_utils::hvxSplatF32(qk);
      i = 0;
      for (; count % 64 == 0 && i + 64 <= count; i += 64) {
        const auto contiguous = op_utils::hvxCenteredShiftedU16ToI32(v + i, quant[2].zero_offset, 0);
        // FP16 expansion stores even/odd lanes in the two FP32 scratch vectors.
        const auto centered = Q6_W_vdeal_VVR(
            Q6_V_hi_W(contiguous), Q6_V_lo_W(contiguous), -4);
        const auto lo = op_utils::hvxMulF32(Q6_Vsf_equals_Vw(Q6_V_lo_W(centered)), value_scale);
        const auto hi = op_utils::hvxMulF32(Q6_Vsf_equals_Vw(Q6_V_hi_W(centered)), value_scale);
        const auto dl = op_utils::hvxMulF32(op_utils::hvxAddF32(lo,
            op_utils::hvxMulF32(vmemu(kv + i), op_utils::hvxSplatF32(-1.f))), beta_scale);
        const auto dh = op_utils::hvxMulF32(op_utils::hvxAddF32(hi,
            op_utils::hvxMulF32(vmemu(kv + i + 32), op_utils::hvxSplatF32(-1.f))), beta_scale);
        vmemu(delta + i) = dl;
        vmemu(delta + i + 32) = dh;
        const auto yl = op_utils::hvxAddF32(vmemu(pre + i), op_utils::hvxMulF32(dl, dot_scale));
        const auto yh = op_utils::hvxAddF32(vmemu(pre + i + 32), op_utils::hvxMulF32(dh, dot_scale));
        const auto cl = op_utils::hvxAddF32(output_zero, op_utils::hvxMulF32(yl, output_scale));
        const auto ch = op_utils::hvxAddF32(output_zero, op_utils::hvxMulF32(yh, output_scale));
        vmemu(out + i) = Q6_Vuh_vasr_VwVwR_sat(
            Q6_Vw_equals_Vsf(ch), Q6_Vw_equals_Vsf(cl), 0);
      }
      for (; i < count; ++i) {
        delta[i] = (op_utils::dequantU16(v[i], quant[2]) - kv[i]) * b;
        out[i] = op_utils::quantU16FromScaledFloat(
            float(output_quant.zero_offset) + (pre[i] + delta[i] * qk) / output_quant.stepsize);
      }
      profile.record(2);
      profile.paths(count % 64 == 0, false);
      auto* next = state + state_base;
      if (count % 64 == 0) {
        if (aligned && reinterpret_cast<uintptr_t>(next) % 128 == 0 && count % 128 == 0)
          deltanetQhpiHvxUpdateStateTileF16<true, 2>(previous, next, delta,
              key_float, d, key_dim, value_dim, count);
        else
          deltanetQhpiHvxUpdateStateTileF16<false>(previous, next, delta,
              key_float, d, key_dim, value_dim, count);
      } else {
        for (size_t j = 0; j < key_dim; ++j)
          for (size_t col = 0; col < count; ++col)
            next[j * value_dim + col] = Float16(
                float(previous[j * value_dim + col]) * d + delta[col] * key_float[j]);
      }
      profile.record(3);
    }
  }
  profile.report(slice, slices, tokens, heads, key_dim, value_dim, false);
}

static uint32_t deltanetCoreQhpiFp16State(
    QHPI_RuntimeHandle* handle, uint32_t num_outputs, QHPI_Tensor** outputs,
    uint32_t num_inputs, const QHPI_Tensor* const* inputs) {
  if (num_inputs != 6 || num_outputs != 2) return QHPI_ERROR_FATAL;
  const auto q = qhpi_tensor_shape(inputs[0]);
  const auto k = qhpi_tensor_shape(inputs[1]);
  const auto v = qhpi_tensor_shape(inputs[2]);
  const auto d = qhpi_tensor_shape(inputs[3]);
  const auto b = qhpi_tensor_shape(inputs[4]);
  const auto s = qhpi_tensor_shape(inputs[5]);
  const auto y = qhpi_tensor_shape(outputs[0]);
  const auto ns = qhpi_tensor_shape(outputs[1]);
  if (q.rank != 4 || k.rank != 4 || v.rank != 4 || s.rank != 4 ||
      y.rank != 4 || ns.rank != 4) return QHPI_ERROR_FATAL;
  const size_t batch = q.dims[0], tokens = q.dims[1], heads = q.dims[2];
  const size_t kd = q.dims[3], vd = v.dims[3];
  if (!batch || !tokens || !heads || !kd || !vd ||
      kd > kMaxDeltaNetDim || vd > kMaxDeltaNetDim) return QHPI_UNSUPPORTED;
  for (size_t i = 0; i < 4; ++i) {
    if (k.dims[i] != q.dims[i] || ns.dims[i] != s.dims[i] ||
        y.dims[i] != v.dims[i]) return QHPI_ERROR_FATAL;
    if (i < 3 && v.dims[i] != q.dims[i]) return QHPI_ERROR_FATAL;
  }
  if (s.dims[0] != batch || s.dims[1] != heads || s.dims[2] != kd || s.dims[3] != vd ||
      !op_utils::hasTimeHeadLayout(d, batch, tokens, heads) ||
      !op_utils::hasTimeHeadLayout(b, batch, tokens, heads)) return QHPI_ERROR_FATAL;
  QHPI_Quant_Parameters quant[5];
  for (size_t i = 0; i < 5; ++i) {
    quant[i] = qhpi_tensor_quant_parameters(inputs[i]);
    if (!op_utils::isValidQuant(quant[i]) || !op_utils::hasU16ZeroOffset(quant[i]))
      return QHPI_ERROR_FATAL;
  }
  const auto oq = qhpi_tensor_quant_parameters(outputs[0]);
  if (!op_utils::isValidQuant(oq) || !op_utils::hasU16ZeroOffset(oq)) return QHPI_ERROR_FATAL;
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
      quant, oq, d, b, batch, tokens, heads, kd, vd, qhpi_slice_number(handle), slices,
      deltanetQhpiValueTileCount(batch * heads, slices, vd, 64));
  return QHPI_SUCCESS;
}

static QHPI_Tensor_Signature_v1 deltanet_core_fp16_state_inputs[] = {
    {QHPI_QUINT16, QHPI_LAYOUT_FLAT_4, QHPI_STORAGE_DIRECT, QHPI_MEM_LOC_DDR_ONLY},
    {QHPI_QUINT16, QHPI_LAYOUT_FLAT_4, QHPI_STORAGE_DIRECT, QHPI_MEM_LOC_DDR_ONLY},
    {QHPI_QUINT16, QHPI_LAYOUT_FLAT_4, QHPI_STORAGE_DIRECT, QHPI_MEM_LOC_DDR_ONLY},
    {QHPI_QUINT16, QHPI_LAYOUT_FLAT_4, QHPI_STORAGE_DIRECT, QHPI_MEM_LOC_DDR_ONLY},
    {QHPI_QUINT16, QHPI_LAYOUT_FLAT_4, QHPI_STORAGE_DIRECT, QHPI_MEM_LOC_DDR_ONLY},
    {QHPI_FLOAT16, QHPI_LAYOUT_FLAT_4, QHPI_STORAGE_DIRECT, QHPI_MEM_LOC_DDR_ONLY},
};
static QHPI_Tensor_Signature_v1 deltanet_core_fp16_state_outputs[] = {
    {QHPI_QUINT16, QHPI_LAYOUT_FLAT_4, QHPI_STORAGE_DIRECT, QHPI_MEM_LOC_DDR_ONLY},
    {QHPI_FLOAT16, QHPI_LAYOUT_FLAT_4, QHPI_STORAGE_DIRECT, QHPI_MEM_LOC_DDR_ONLY},
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
