// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include "HTP/core/intrinsics.h"
#include "hexagon_protos.h"
#include "hvx_hexagon_protos.h"
namespace attention_fixed {
struct Scale {
  int coefficient;
  int shift;
  static Scale from_double(double ratio) {
    int exponent;
    const double fraction = std::frexp(ratio, &exponent);
    int coefficient = int(std::llround(fraction * 32768));
    if (coefficient == 32768) {
      coefficient = 16384;
      ++exponent;
    }
    return {coefficient, -exponent};
  }
  static Scale ratio(uint64_t numerator, uint64_t denominator) {
    if (!numerator)
      return {0, 0};
    if (numerator > (1ULL << 48) || denominator > (1ULL << 48))
      return from_double(double(numerator) / double(denominator));
    int shift = 0;
    while (numerator < ((denominator + 1) / 2)) {
      numerator *= 2;
      ++shift;
    }
    while (numerator >= denominator) {
      denominator *= 2;
      --shift;
    }
    uint64_t coefficient = (numerator * 32768 + denominator / 2) / denominator;
    if (coefficient == 32768) {
      coefficient = 16384;
      --shift;
    }
    return {int(coefficient), shift};
  }
  HVX_Vector apply(HVX_Vector value) const {
    auto scaled =
        Q6_Vw_vmpyo_VwVh_s1_rnd_sat(value, Q6_Vh_vsplat_R(coefficient));
    if (shift >= 32)
      return Q6_V_vzero();
    if (shift > 0 && shift <= 17) {
      scaled = Q6_Vw_vadd_VwVw(scaled, Q6_V_vsplat_R(1u << (shift - 1)));
      scaled = Q6_Vw_vasr_VwR(scaled, shift);
    } else if (shift > 0) {
      auto rounding =
          Q6_V_vand_VV(Q6_Vw_vasr_VwR(scaled, shift - 1), Q6_V_vsplat_R(1));
      scaled = Q6_Vw_vadd_VwVw(Q6_Vw_vasr_VwR(scaled, shift), rounding);
    } else if (shift < 0) {
      unsigned amount = -shift;
      if (amount >= 31) {
        auto saturated = Q6_V_vmux_QVV(
            Q6_Q_vcmp_gt_VwVw(Q6_V_vzero(), scaled),
            Q6_V_vsplat_R(INT32_MIN),
            Q6_V_vsplat_R(INT32_MAX));
        return Q6_V_vmux_QVV(
            Q6_Q_vcmp_eq_VwVw(scaled, Q6_V_vzero()), Q6_V_vzero(), saturated);
      }
      auto hi = Q6_V_vsplat_R(INT32_MAX >> amount),
           lo = Q6_V_vsplat_R(INT32_MIN >> amount);
      scaled = Q6_Vw_vmax_VwVw(lo, Q6_Vw_vmin_VwVw(scaled, hi));
      scaled = Q6_Vw_vasl_VwR(scaled, amount);
    }
    return scaled;
  }
};
inline HVX_Vector unpack_result(HVX_Vector low, HVX_Vector high, bool second) {
  auto words = second
      ? Q6_V_vor_VV(
            Q6_Vuw_vlsr_VuwR(low, 16),
            Q6_V_vand_VV(high, Q6_V_vsplat_R(0xffff0000)))
      : Q6_V_vor_VV(
            Q6_V_vand_VV(low, Q6_V_vsplat_R(65535)), Q6_Vw_vasl_VwR(high, 16));
  return Q6_Vw_vasr_VwR(words, 1);
}
inline HVX_Vector affine_u16(HVX_Vector value, const Scale& scale, int zero) {
  auto code = scale.apply(value);
  code = Q6_Vw_vmax_VwVw(
      Q6_V_vsplat_R(-zero), Q6_Vw_vmin_VwVw(code, Q6_V_vsplat_R(65535 - zero)));
  return Q6_Vw_vadd_VwVw(code, Q6_V_vsplat_R(zero));
}
inline HVX_Vector multiply_q15(HVX_Vector value, unsigned coefficient) {
  if (coefficient == 32768)
    return value;
  if (!coefficient)
    return Q6_V_vzero();
  return Q6_Vw_vmpyo_VwVh_s1_rnd_sat(value, Q6_Vh_vsplat_R(coefficient));
}
inline uint32_t multiply_q15(uint32_t value, unsigned coefficient) {
  return uint32_t((uint64_t(value) * coefficient + 16384) >> 15);
}
inline void pack_query(const uint16_t* source, uint8_t* destination) {
  for (unsigned row = 0; row < 32; row += 2)
    for (unsigned column = 0; column < 256; column += 64) {
      auto a =
          *reinterpret_cast<const HVX_Vector*>(source + row * 256 + column);
      auto b = *reinterpret_cast<const HVX_Vector*>(
          source + (row + 1) * 256 + column);
      auto pair = Q6_W_vshuff_VVR(b, a, -2);
      *reinterpret_cast<HVX_Vector*>(
          destination + (column / 32) * 2048 + (row / 2) * 128) =
          Q6_V_lo_W(pair);
      *reinterpret_cast<HVX_Vector*>(
          destination + (column / 32 + 1) * 2048 + (row / 2) * 128) =
          Q6_V_hi_W(pair);
    }
}
inline HVX_Vector load_codes64(const uint8_t* source, unsigned offset) {
  auto bytes = vmemu(source + (offset & ~127u));
  if ((offset & 127u) > 64)
    bytes = Q6_V_valign_VVR(
        vmemu(source + (offset & ~127u) + 128), bytes, offset & 127u);
  else if ((offset & 127u) != 0 && (offset & 127u) != 64)
    bytes = Q6_V_vror_VR(bytes, offset & 127u);
  auto pair = Q6_Wuh_vunpack_Vub(bytes);
  return (offset & 127u) == 64 ? Q6_V_hi_W(pair) : Q6_V_lo_W(pair);
}
} // namespace attention_fixed
