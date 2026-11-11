// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <cmath>
#include <cstdint>
#include "hexagon_protos.h"
#include "hvx_hexagon_protos.h"
namespace attention_exp {
// Interleaved zeros let a byte-indexed lookup return one U16 entry per I32
// lane.
alignas(128) static constexpr uint16_t powers[64] = {
    65535, 0, 64131, 0, 62757, 0, 61412, 0, 60096, 0, 58808, 0, 57548, 0,
    56315, 0, 55108, 0, 53927, 0, 52772, 0, 51641, 0, 50534, 0, 49452, 0,
    48392, 0, 47355, 0, 46340, 0, 45347, 0, 44376, 0, 43425, 0, 42494, 0,
    41584, 0, 40693, 0, 39821, 0, 38967, 0, 38132, 0, 37315, 0, 36516, 0,
    35733, 0, 34968, 0, 34218, 0, 33485, 0};
alignas(128) static constexpr uint16_t slopes[64] = {
    1404, 0, 1374, 0, 1345, 0, 1316, 0, 1288, 0, 1260, 0, 1233, 0, 1207, 0,
    1181, 0, 1155, 0, 1131, 0, 1107, 0, 1082, 0, 1060, 0, 1037, 0, 1015, 0,
    993,  0, 971,  0, 951,  0, 931,  0, 910,  0, 891,  0, 872,  0, 854,  0,
    835,  0, 817,  0, 799,  0, 783,  0, 765,  0, 750,  0, 733,  0, 717,  0};
struct Parameters {
  unsigned coefficient, bits;
  static Parameters make(float step) {
    double scale = double(step) * 1.4426950408889634;
    if (!(scale >= 1024. / (1ULL << 30)) || scale * 32 > 32767)
      return {0, 0};
    unsigned bits = 20;
    while (bits < 30 && scale * double(1ULL << (bits + 1)) <= 32767)
      ++bits;
    while (bits > 5 && scale * double(1ULL << bits) > 32767)
      --bits;
    return {unsigned(std::llround(scale * double(1ULL << bits))), bits};
  }
};
inline HVX_Vector codes(
    HVX_Vector delta,
    unsigned coefficient,
    HVX_Vector table,
    HVX_Vector slope_table,
    unsigned bits = 20) {
  auto fixed =
      Q6_Vw_vmpyi_VwRh(delta, Q6_R_combine_RlRl(coefficient, coefficient));
  auto index =
      Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(fixed, bits - 5), Q6_V_vsplat_R(31));
  auto a = Q6_Wh_vlut16_VbVhR(index, table, 0),
       b = Q6_Wh_vlut16_VbVhR(index, table, 1);
  auto base = Q6_V_vand_VV(
      Q6_V_vor_VV(Q6_V_lo_W(a), Q6_V_lo_W(b)), Q6_V_vsplat_R(65535));
  auto sa = Q6_Wh_vlut16_VbVhR(index, slope_table, 0),
       sb = Q6_Wh_vlut16_VbVhR(index, slope_table, 1);
  auto slope = Q6_V_vand_VV(
      Q6_V_vor_VV(Q6_V_lo_W(sa), Q6_V_lo_W(sb)), Q6_V_vsplat_R(65535));
  auto aligned = bits >= 20 ? Q6_Vuw_vlsr_VuwR(fixed, bits - 20)
                            : Q6_Vw_vasl_VwR(fixed, 20 - bits);
  auto fraction = Q6_V_vand_VV(aligned, Q6_V_vsplat_R(32767));
  auto adjustment = Q6_Vuw_vlsr_VuwR(
      Q6_Vw_vadd_VwVw(
          Q6_Vw_vmpyie_VwVuh(fraction, slope), Q6_V_vsplat_R(16384)),
      15);
  auto mantissa = Q6_Vw_vsub_VwVw(base, adjustment);
  auto exponent =
      Q6_Vw_vmin_VwVw(Q6_Vuw_vlsr_VuwR(fixed, bits), Q6_V_vsplat_R(17));
  auto rounding =
      Q6_Vuw_vlsr_VuwR(Q6_Vw_vasl_VwVw(Q6_V_vsplat_R(1), exponent), 1);
  return Q6_Vw_vasr_VwVw(Q6_Vw_vadd_VwVw(mantissa, rounding), exponent);
}
} // namespace attention_exp
