// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

constexpr unsigned short_key_limit = 8;
static void short_attention(
    const uint16_t* q,
    uint16_t* y,
    const uint8_t* kpack,
    const uint8_t* vpack,
    unsigned keys,
    unsigned slice,
    uint8_t* work,
    QHPI_Quant_Parameters qq,
    QHPI_Quant_Parameters yq,
    const float* meta,
    const uint16_t* nk = nullptr,
    const uint16_t* nv = nullptr,
    QHPI_Quant_Parameters kq = {},
    QHPI_Quant_Parameters vq = {}) {
  auto* qa = reinterpret_cast<uint16_t*>(work + worker_start + slice * 32768);
  auto* current_k = qa + 512;
  auto* current_v = qa + 1024;
  if (nk) {
    to_half(current_k, nk, 512, kq);
    to_half(current_v, nv, 512, vq);
  }
  const float qk_scale = meta[0], factor = meta[2], scaled_scale = meta[3],
              logit_scale = meta[5], prob_scale = meta[7];
  const int qk_zero = int(meta[1]), scaled_zero = int(meta[4]),
            logit_zero = int(meta[6]), prob_zero = int(meta[8]);
  auto rounded = [](HVX_Vector v) {
    return Q6_V_lo_W(Q6_Wsf_vcvt_Vhf(Q6_Vhf_vcvt_VsfVsf(v, v)));
  };
  alignas(128) float dots[32]{};
  for (unsigned h = 0; h < 2; ++h) {
    const unsigned head = slice + h * workers, kv = head / 8;
    to_half(qa + h * 256, q + head * 256, 256, qq);
    for (unsigned key = 0; key < keys; ++key) {
      auto sum = Q6_V_vzero();
      const auto* k = nk && key + 1 == keys
          ? current_k + kv * 256
          : reinterpret_cast<const uint16_t*>(kpack) + kv * kv_rows * 256 +
              key * 256;
      for (unsigned d = 0; d < 256; d += 64) {
        const auto product = Q6_Wqf32_vmpy_VhfVhf(
            *reinterpret_cast<const HVX_Vector*>(qa + h * 256 + d),
            *reinterpret_cast<const HVX_Vector*>(k + d));
        sum = hvxAddF32(
            sum,
            hvxAddF32(
                Q6_Vsf_equals_Vqf32(Q6_V_lo_W(product)),
                Q6_Vsf_equals_Vqf32(Q6_V_hi_W(product))));
      }
      dots[2 * key + h] = reduce(sum, false);
    }
  }
  auto z = rounded(*reinterpret_cast<const HVX_Vector*>(dots));
  z = qdq(z, qk_scale, qk_zero);
  z = qdq(hvxMulF32(z, hvxSplatF32(factor)), scaled_scale, scaled_zero);
  auto codes = Q6_Vw_equals_Vsf(hvxAddF32(
      hvxMulF32(z, hvxSplatF32(1.f / logit_scale)),
      hvxSplatF32(logit_zero + .5f)));
  codes = Q6_Vw_vmax_VwVw(
      Q6_V_vzero(), Q6_Vw_vmin_VwVw(codes, Q6_V_vsplat_R(65535)));
  codes = Q6_V_vmux_QVV(Q6_Q_vsetq_R(keys * 8), codes, Q6_V_vzero());
  auto maximum = codes;
  for (unsigned shift = 64; shift >= 8; shift /= 2)
    maximum = Q6_Vw_vmax_VwVw(maximum, Q6_V_vror_VR(maximum, shift));
  auto e = expv(hvxMulF32(
      Q6_Vsf_equals_Vw(Q6_Vw_vsub_VwVw(codes, maximum)),
      hvxSplatF32(logit_scale)));
  e = Q6_V_vmux_QVV(Q6_Q_vsetq_R(keys * 8), e, Q6_V_vzero());
  auto sums = e;
  for (unsigned shift = 64; shift >= 8; shift /= 2)
    sums = hvxAddF32(sums, Q6_V_vror_VR(sums, shift));
  union {
    uint32_t u;
    float f;
  } sum_a{uint32_t(Q6_R_vextract_VR(sums, 0))},
      sum_b{uint32_t(Q6_R_vextract_VR(sums, 4))};
  const auto inverse = Q6_V_lo_W(Q6_W_vshuff_VVR(
      hvxSplatF32(1.f / (sum_b.f * prob_scale)),
      hvxSplatF32(1.f / (sum_a.f * prob_scale)),
      -4));
  codes = Q6_Vw_equals_Vsf(
      hvxAddF32(hvxMulF32(e, inverse), hvxSplatF32(prob_zero + .5f)));
  codes = Q6_Vw_vmax_VwVw(
      Q6_V_vzero(), Q6_Vw_vmin_VwVw(codes, Q6_V_vsplat_R(65535)));
  alignas(128) uint16_t probability[64];
  const auto probability_f32 = hvxMulF32(
      Q6_Vsf_equals_Vw(Q6_Vw_vsub_VwVw(codes, Q6_V_vsplat_R(prob_zero))),
      hvxSplatF32(prob_scale));
  *reinterpret_cast<HVX_Vector*>(probability) =
      Q6_Vhf_vcvt_VsfVsf(probability_f32, probability_f32);
  for (unsigned h = 0; h < 2; ++h) {
    const unsigned head = slice + h * workers, kv = head / 8;
    for (unsigned d = 0; d < 256; d += 64) {
      auto even = Q6_V_vzero(), odd = Q6_V_vzero();
      for (unsigned key = 0; key < keys; ++key) {
        const auto* v = nk && key + 1 == keys
            ? current_v + kv * 256 + d
            : reinterpret_cast<const uint16_t*>(vpack) + kv * kv_rows * 256 +
                key * 256 + d;
        const auto product = Q6_Wqf32_vmpy_VhfVhf(
            *reinterpret_cast<const HVX_Vector*>(v),
            Q6_Vh_vsplat_R(probability[4 * key + 2 * h]));
        even = hvxAddF32(even, Q6_Vsf_equals_Vqf32(Q6_V_lo_W(product)));
        odd = hvxAddF32(odd, Q6_Vsf_equals_Vqf32(Q6_V_hi_W(product)));
      }
      const auto contiguous = Q6_W_vshuff_VVR(rounded(odd), rounded(even), -4);
      auto a = hvxAddF32(
          hvxMulF32(Q6_V_lo_W(contiguous), hvxSplatF32(1.f / yq.stepsize)),
          hvxSplatF32(yq.zero_offset + .5f));
      auto b = hvxAddF32(
          hvxMulF32(Q6_V_hi_W(contiguous), hvxSplatF32(1.f / yq.stepsize)),
          hvxSplatF32(yq.zero_offset + .5f));
      *reinterpret_cast<HVX_Vector*>(y + head * 256 + d) =
          Q6_Vuh_vpack_VwVw_sat(Q6_Vw_equals_Vsf(b), Q6_Vw_equals_Vsf(a));
    }
  }
  if (nk) {
    const unsigned kv = slice / 4, col = (slice % 4) * 64;
    auto* kd = reinterpret_cast<uint16_t*>(const_cast<uint8_t*>(kpack)) +
        kv * kv_rows * 256 + (keys - 1) * 256 + col;
    auto* vd = reinterpret_cast<uint16_t*>(const_cast<uint8_t*>(vpack)) +
        kv * kv_rows * 256 + (keys - 1) * 256 + col;
    *reinterpret_cast<HVX_Vector*>(kd) =
        *reinterpret_cast<const HVX_Vector*>(current_k + kv * 256 + col);
    *reinterpret_cast<HVX_Vector*>(vd) =
        *reinterpret_cast<const HVX_Vector*>(current_v + kv * 256 + col);
  }
}
