// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

struct DecodeShared {
  uint32_t ready[8];
  uint32_t maximum[8][4];
  uint32_t normalizer[8][4];
};
static_assert(sizeof(DecodeShared) <= 448);

static uint32_t integer_decode(
    QHPI_RuntimeHandle* handle,
    QHPI_Tensor** out,
    const QHPI_Tensor* const* in,
    unsigned kv_rows,
    CacheMappings::Slot* supplied_cache) {
  const unsigned slice = qhpi_slice_number(handle);
  const unsigned slices = qhpi_num_slices(handle);
  const unsigned kvheads = qhpi_tensor_shape(in[3]).dims[1];
  const auto* lengths =
      static_cast<const int32_t*>(qhpi_tensor_raw_data(in[6]));
  const unsigned keys = unsigned(lengths[2]) + 1;
  const unsigned padded = (keys + 63) & ~63u;
  const unsigned available = std::min(slices / kvheads, (padded + 127) / 128);
  // Power-of-two partitions keep the DMA banks aligned.
  const unsigned parts = available >= 4 ? 4 : available >= 2 ? 2 : 1;
  const unsigned active = kvheads * parts;
  auto* work = static_cast<uint8_t*>(qhpi_tensor_raw_data(in[0]));
  auto& queue = *reinterpret_cast<attention_hmx::Queue*>(work);
  if (slice >= active) {
    __atomic_store_n(&queue.jobs[slice].finished, 1u, __ATOMIC_RELEASE);
    return QHPI_SUCCESS;
  }
  const unsigned kv = slice / parts, part = slice % parts;
  const unsigned tiles = (padded + 127) / 128;
  const unsigned first = (tiles * part / parts) * 128;
  const unsigned last = std::min(padded, (tiles * (part + 1) / parts) * 128);
  const unsigned score_stride = stride;
  const unsigned slice_bytes = kvheads * per_worker / active;
  auto* base = work + worker_start + slice * slice_bytes;
  auto* accumulator = static_cast<int32_t*>(qhpi_tensor_raw_data(in[8])) +
      4 * kvheads * stride / 2 + slice * 1024;
  // Keep scalar multiply operands in cacheable memory rather than VTCM.
  alignas(128) uint32_t query_low[256], query_high[256];
  auto* scores = static_cast<uint16_t*>(qhpi_tensor_raw_data(in[8])) +
      kv * 4 * score_stride + first;
  auto& shared = *reinterpret_cast<DecodeShared*>(work + 1600);
  auto fail = [&]() {
    __atomic_store_n(&queue.error, 1u, __ATOMIC_RELEASE);
    __atomic_store_n(&shared.ready[slice], 2u, __ATOMIC_RELEASE);
    __atomic_store_n(&queue.jobs[slice].finished, 1u, __ATOMIC_RELEASE);
    return uint32_t(QHPI_ERROR_FATAL);
  };
  const unsigned kv_bytes = kvheads * kv_rows * 256 * 2;
  auto* cached = supplied_cache
      ? supplied_cache
      : packed_cache(
            static_cast<const int32_t*>(qhpi_tensor_raw_data(in[2]))[0],
            kv_bytes);
  const PackedKv8 packed(cached->packed, kvheads, kv_rows);
  const auto* key_sums = packed.key_sums;
  auto* descriptors =
      reinterpret_cast<DmaDescriptor*>(cached->dma_words[slice]);
  const auto* q = static_cast<const uint16_t*>(qhpi_tensor_raw_data(in[1]));
  auto* y = static_cast<uint16_t*>(qhpi_tensor_raw_data(out[0]));
  const auto qq = qhpi_tensor_quant_parameters(in[1]);
  const auto oq = qhpi_tensor_quant_parameters(out[0]);
  const auto* meta = static_cast<const float*>(qhpi_tensor_raw_data(in[7]));
  const auto raw_scale = attention_fixed::Scale::from_double(
      double(qq.stepsize) * meta[9] / meta[0]);
  const auto logit_scale =
      attention_fixed::Scale::from_double(double(meta[0]) * meta[2] / meta[5]);
  const auto exp_parameters = attention_exp::Parameters::make(meta[5]);
  const auto exp_table = vmemu(attention_exp::powers);
  const auto exp_slopes = vmemu(attention_exp::slopes);
  constexpr int byte_bias = 128 * (256 + 1);
  const int key_zero = int(meta[10]), value_zero = int(meta[12]);
  const int raw_zero = int(meta[1]), logits_zero = int(meta[6]);
  int query_correction[4];
  for (unsigned h = 0; h < 4; ++h) {
    auto sum = Q6_V_vzero();
    for (unsigned d = 0; d < 256; d += 128) {
      auto a = vmemu(q + (kv * 4 + h) * 256 + d);
      auto b = vmemu(q + (kv * 4 + h) * 256 + d + 64);
      auto lo = Q6_Vb_vpacke_VhVh(b, a);
      auto hi = Q6_Vb_vpacko_VhVh(b, a);
      vmemu(query_low + h * 64 + d / 4) =
          Q6_V_vxor_VV(lo, Q6_V_vsplat_R(0x80808080));
      vmemu(query_high + h * 64 + d / 4) =
          Q6_V_vxor_VV(hi, Q6_V_vsplat_R(0x80808080));
      auto ua = Q6_Wuw_vunpack_Vuh(a), ub = Q6_Wuw_vunpack_Vuh(b);
      sum = Q6_Vw_vadd_VwVw(
          sum,
          Q6_Vw_vadd_VwVw(
              Q6_Vw_vadd_VwVw(Q6_V_lo_W(ua), Q6_V_hi_W(ua)),
              Q6_Vw_vadd_VwVw(Q6_V_lo_W(ub), Q6_V_hi_W(ub))));
    }
    for (unsigned shift = 64; shift >= 4; shift /= 2)
      sum = Q6_Vw_vadd_VwVw(sum, Q6_V_vror_VR(sum, shift));
    query_correction[h] =
        -key_zero * (Q6_R_vextract_VR(sum, 0) - 256 * qq.zero_offset);
    shared.maximum[slice][h] = 0;
  }
  auto load_keys = [&](unsigned at, unsigned bank) {
    dma_start(
        base + bank * 32768,
        packed.key_panel(kv, at),
        std::min(128u, last - at) * 256,
        descriptors + bank * 8);
  };
  load_keys(first, 0);
  HVX_Vector maxima[4] = {
      Q6_V_vzero(), Q6_V_vzero(), Q6_V_vzero(), Q6_V_vzero()};
  for (unsigned at = first; at < last; at += 32) {
    const unsigned bank = ((at - first) / 128) % 2;
    if (at % 128 == 0) {
      if (!dma_wait())
        return fail();
      if (at + 128 < last)
        load_keys(at + 128, bank ^ 1);
    }
    const auto* panel = reinterpret_cast<const HVX_Vector*>(
        base + bank * 32768 + (at % 128) * 256);
    HVX_Vector low[4] = {
        Q6_V_vzero(), Q6_V_vzero(), Q6_V_vzero(), Q6_V_vzero()};
    HVX_Vector high[4] = {
        Q6_V_vzero(), Q6_V_vzero(), Q6_V_vzero(), Q6_V_vzero()};
    for (unsigned d = 0; d < 64; ++d) {
      const auto weights = Q6_V_vxor_VV(panel[d], Q6_V_vsplat_R(0x80808080));
      for (unsigned h = 0; h < 4; ++h) {
        low[h] = Q6_Vw_vrmpyacc_VwVubRb(low[h], weights, query_low[h * 64 + d]);
        high[h] =
            Q6_Vw_vrmpyacc_VwVubRb(high[h], weights, query_high[h * 64 + d]);
      }
    }
    const auto sums =
        Q6_V_lo_W(Q6_Wuw_vunpack_Vuh(vmemu(key_sums + kv * kv_rows + at)));
    const int coefficient = byte_bias - qq.zero_offset;
    const auto correction = coefficient > 32767
        ? Q6_Vw_vmpyie_VwVuh(sums, Q6_V_vsplat_R(coefficient))
        : Q6_Vw_vmpyi_VwRh(sums, Q6_R_combine_RlRl(coefficient, coefficient));
    for (unsigned h = 0; h < 4; ++h) {
      auto dot = Q6_Vw_vadd_VwVw(
          Q6_Vw_vadd_VwVw(low[h], Q6_Vw_vasl_VwR(high[h], 8)),
          Q6_Vw_vadd_VwVw(correction, Q6_V_vsplat_R(query_correction[h])));
      auto code = attention_fixed::affine_u16(dot, raw_scale, raw_zero);
      code = attention_fixed::affine_u16(
          Q6_Vw_vsub_VwVw(code, Q6_V_vsplat_R(raw_zero)),
          logit_scale,
          logits_zero);
      if (at + 32 > keys)
        code = Q6_V_vmux_QVV(
            Q6_Q_vsetq_R((keys > at ? keys - at : 0) * 4), code, Q6_V_vzero());
      maxima[h] = Q6_Vw_vmax_VwVw(maxima[h], code);
      const auto mask =
          at % 64 ? Q6_Q_not_Q(Q6_Q_vsetq_R(64)) : Q6_Q_vsetq_R(64);
      Q6_vmem_QRIV(
          mask,
          reinterpret_cast<HVX_Vector*>(
              scores + h * score_stride + (at - first) / 64 * 64),
          Q6_Vuh_vpack_VwVw_sat(code, code));
    }
  }
  for (unsigned h = 0; h < 4; ++h) {
    auto mx = maxima[h];
    for (unsigned shift = 64; shift >= 4; shift /= 2)
      mx = Q6_Vw_vmax_VwVw(mx, Q6_V_vror_VR(mx, shift));
    shared.maximum[slice][h] = Q6_R_vextract_VR(mx, 0);
  }
  __atomic_store_n(&shared.ready[slice], 1u, __ATOMIC_RELEASE);
  for (unsigned s = kv * parts; s < (kv + 1) * parts; ++s)
    while (__atomic_load_n(&shared.ready[s], __ATOMIC_ACQUIRE) < 1)
      attention_hmx::pause();
  for (unsigned h = 0; h < 4; ++h) {
    uint32_t mx = 0;
    for (unsigned s = kv * parts; s < (kv + 1) * parts; ++s)
      mx = std::max(mx, shared.maximum[s][h]);
    auto sum = Q6_V_vzero();
    for (unsigned at = first; at < last; at += 64) {
      auto pair =
          Q6_Wuw_vunpack_Vuh(vmemu(scores + h * score_stride + at - first));
      auto exponential = [&](HVX_Vector code, unsigned offset) {
        auto e = attention_exp::codes(
            Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(mx), code),
            exp_parameters.coefficient,
            exp_table,
            exp_slopes,
            exp_parameters.bits);
        return at + offset + 32 <= keys
            ? e
            : Q6_V_vmux_QVV(
                  Q6_Q_vsetq_R(
                      (keys > at + offset ? keys - at - offset : 0) * 4),
                  e,
                  Q6_V_vzero());
      };
      const auto lo = exponential(Q6_V_lo_W(pair), 0);
      const auto hi = exponential(Q6_V_hi_W(pair), 32);
      sum = Q6_Vw_vadd_VwVw(sum, Q6_Vw_vadd_VwVw(lo, hi));
      vmemu(scores + h * score_stride + at - first) =
          Q6_Vuh_vpack_VwVw_sat(hi, lo);
    }
    for (unsigned shift = 64; shift >= 4; shift /= 2)
      sum = Q6_Vw_vadd_VwVw(sum, Q6_V_vror_VR(sum, shift));
    shared.normalizer[slice][h] = Q6_R_vextract_VR(sum, 0);
  }
  for (unsigned i = 0; i < 4 * 256 / 32; ++i)
    reinterpret_cast<HVX_Vector*>(accumulator)[i] = Q6_V_vzero();
  auto load_values = [&](unsigned at, unsigned bank) {
    const unsigned count = std::min(128u, last - at);
    auto* list = descriptors + bank * 8;
    for (unsigned c = 0; c < 8; ++c) {
      list[c].next = c < 7 ? list + c + 1 : nullptr;
      list[c].control = count * 32 | (1u << 28) | (1u << 29);
      list[c].source = packed.value_column(kv, at, c * 32, padded);
      list[c].destination = base + bank * 32768 + c * count * 32;
    }
    dma_start(list);
  };
  load_values(first, 0);
  for (unsigned at = first; at < last; at += 32) {
    const unsigned bank = ((at - first) / 128) % 2;
    if (at % 128 == 0) {
      if (!dma_wait())
        return fail();
      if (at + 128 < last)
        load_values(at + 128, bank ^ 1);
    }
    alignas(128) uint32_t low_codes[4][32], high_codes[4][32];
    uint32_t weight_sum[4];
    for (unsigned h = 0; h < 4; ++h) {
      const auto code = vmemu(scores + h * score_stride + at - first);
      auto lo = Q6_Vb_vpacke_VhVh(code, code),
           hi = Q6_Vb_vpacko_VhVh(code, code);
      vmemu(low_codes[h]) = Q6_V_vxor_VV(lo, Q6_V_vsplat_R(0x80808080));
      vmemu(high_codes[h]) = Q6_V_vxor_VV(hi, Q6_V_vsplat_R(0x80808080));
      auto sum = Q6_V_lo_W(Q6_Wuw_vunpack_Vuh(code));
      for (unsigned shift = 64; shift >= 4; shift /= 2)
        sum = Q6_Vw_vadd_VwVw(sum, Q6_V_vror_VR(sum, shift));
      weight_sum[h] = Q6_R_vextract_VR(sum, 0);
    }
    for (unsigned c = 0; c < 8; ++c) {
      const auto* panel = reinterpret_cast<const HVX_Vector*>(
          base + bank * 32768 + c * std::min(128u, last - at / 128 * 128) * 32 +
          (at % 128) / 32 * 1024);
      HVX_Vector lo[4] = {
          Q6_V_vzero(), Q6_V_vzero(), Q6_V_vzero(), Q6_V_vzero()};
      HVX_Vector hi[4] = {
          Q6_V_vzero(), Q6_V_vzero(), Q6_V_vzero(), Q6_V_vzero()};
      auto value_sum = Q6_V_vzero();
      for (unsigned d = 0; d < 8; ++d) {
        const auto weights = Q6_V_vxor_VV(panel[d], Q6_V_vsplat_R(0x80808080));
        value_sum = Q6_Vuw_vrmpyacc_VuwVubRub(value_sum, weights, 0x01010101);
        for (unsigned h = 0; h < 4; ++h) {
          lo[h] = Q6_Vw_vrmpyacc_VwVubRb(lo[h], weights, low_codes[h][d]);
          hi[h] = Q6_Vw_vrmpyacc_VwVubRb(hi[h], weights, high_codes[h][d]);
        }
      }
      const auto correction =
          Q6_Vw_vmpyie_VwVuh(value_sum, Q6_V_vsplat_R(byte_bias));
      for (unsigned h = 0; h < 4; ++h) {
        auto weighted = Q6_Vw_vadd_VwVw(
            Q6_Vw_vadd_VwVw(lo[h], Q6_Vw_vasl_VwR(hi[h], 8)), correction);
        weighted = Q6_Vw_vadd_VwVw(
            weighted, Q6_V_vsplat_R(-int(value_zero * weight_sum[h])));
        weighted =
            Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(weighted, Q6_V_vsplat_R(128)), 8);
        auto* acc =
            reinterpret_cast<HVX_Vector*>(accumulator + h * 256 + c * 32);
        *acc = Q6_Vw_vadd_VwVw(*acc, weighted);
      }
    }
  }
  __atomic_store_n(&shared.ready[slice], 2u, __ATOMIC_RELEASE);
  if (!part) {
    for (unsigned s = kv * parts; s < (kv + 1) * parts; ++s)
      while (__atomic_load_n(&shared.ready[s], __ATOMIC_ACQUIRE) < 2)
        attention_hmx::pause();
    for (unsigned h = 0; h < 4; ++h) {
      uint32_t denominator = 0;
      for (unsigned s = kv * parts; s < (kv + 1) * parts; ++s)
        denominator += shared.normalizer[s][h];
      const auto scale = attention_fixed::Scale::from_double(
          double(meta[11]) * 256 / (double(denominator) * oq.stepsize));
      for (unsigned c = 0; c < 256; c += 64) {
        auto lo = Q6_V_vzero(), hi = Q6_V_vzero();
        for (unsigned s = kv * parts; s < (kv + 1) * parts; ++s) {
          const auto* acc =
              static_cast<const int32_t*>(qhpi_tensor_raw_data(in[8])) +
              4 * kvheads * stride / 2 + s * 1024;
          lo = Q6_Vw_vadd_VwVw(lo, vmemu(acc + h * 256 + c));
          hi = Q6_Vw_vadd_VwVw(hi, vmemu(acc + h * 256 + c + 32));
        }
        lo = attention_fixed::affine_u16(lo, scale, oq.zero_offset);
        hi = attention_fixed::affine_u16(hi, scale, oq.zero_offset);
        vmemu(y + (kv * 4 + h) * 256 + c) = Q6_Vuh_vpack_VwVw_sat(hi, lo);
      }
    }
  }
  __atomic_store_n(&queue.jobs[slice].finished, 1u, __ATOMIC_RELEASE);
  return QHPI_SUCCESS;
}
