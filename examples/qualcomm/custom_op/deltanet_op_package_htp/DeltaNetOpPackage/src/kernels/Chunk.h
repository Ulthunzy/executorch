// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <algorithm>
#include <array>
#include "../utils/DeltaNetConfig.h"
#include "../utils/HmxTileLayout.h"
#if defined(__hexagon__) && QCOM_HTP_V81
#include "HmxDispatch.h"
#endif

namespace gdn_chunk {
constexpr unsigned workers = 6, chunk = 128, width = 128;
[[maybe_unused]] constexpr unsigned t_offset = 0,
                                    z_offset = t_offset + chunk * chunk * 2,
                                    a_offset = z_offset + chunk * width * 2,
                                    k_offset = a_offset,
                                    c_offset = k_offset + width * chunk * 2,
                                    y_offset = c_offset + chunk * width * 2,
                                    s_offset = y_offset + chunk * width * 2,
                                    row_offset = s_offset + width * width * 2,
                                    bias_offset =
                                        row_offset + chunk * width * 2;
constexpr unsigned correction_bytes = (bias_offset + 256 + 2047) & ~2047u;
constexpr unsigned matrix_bytes = 32768;
constexpr unsigned pre_bytes = 9 * matrix_bytes + 2048;
constexpr unsigned worker_bytes = correction_bytes + pre_bytes;
constexpr unsigned qa = 0, ka = matrix_bytes, ba = 2 * matrix_bytes,
                   kt = 3 * matrix_bytes, st = 4 * matrix_bytes,
                   gram = 5 * matrix_bytes, qk = 6 * matrix_bytes,
                   row0 = 7 * matrix_bytes, row1 = 8 * matrix_bytes,
                   pre_bias = 9 * matrix_bytes;
constexpr unsigned workspace_bytes = 2048 + workers * worker_bytes;
static_assert(bias_offset + 256 <= correction_bytes);
static_assert(workspace_bytes == 3172352);

constexpr float floor = 1.1253517471925912e-7f;
constexpr unsigned tile_offset(unsigned r, unsigned c) {
  return ((r / 32) * 4 + c / 32) * 1024;
}
inline unsigned packed_index(unsigned r, unsigned c) {
  return tile_offset(r, c) + (r % 32 / 2) * 64 + (c % 32) * 2 + r % 2;
}

alignas(128) inline constexpr auto pair_lanes = [] {
  std::array<uint8_t, 128> value{};
  for (unsigned i = 0; i < 128; ++i)
    value[i] = i & 124;
  return value;
}();
inline unsigned tile_index(unsigned row, unsigned column, unsigned rows) {
  return (column / 32) * (rows / 32) + row / 32;
}

inline void weight_columns(
    const uint16_t* source,
    const uint16_t* weights,
    uint16_t* target,
    unsigned rows,
    unsigned columns) {
  const auto lanes = vmemu(pair_lanes.data());
  for (unsigned r = 0; r < rows; r += 2) {
    const auto packed_weights = vmemu(weights + (r / 64) * 64);
    const auto weight = Q6_V_vdelta_VV(
        packed_weights, Q6_V_vxor_VV(lanes, Q6_Vb_vsplat_R((r % 64) * 2)));
    for (unsigned c = 0; c < columns; c += 32) {
      const unsigned at = (c / 32) * rows * 32 + (r / 2) * 64;
      vmemu(target + at) = Q6_Vhf_vmpy_VhfVhf(vmemu(source + at), weight);
    }
  }
}

template <bool Decay>
inline void finish(
    const uint16_t* product,
    const uint16_t* previous,
    uint16_t decay,
    uint16_t* target,
    unsigned rows,
    unsigned columns) {
  for (unsigned r = 0; r < rows; r += 32)
    for (unsigned i = 0; i < 32; i += 2)
      for (unsigned c = 0; c < columns; c += 64) {
        const unsigned at =
            (r / 32) * columns * 32 + (c / 32) * 1024 + (i / 2) * 64;
        const auto pair =
            Q6_W_vdeal_VVR(vmemu(product + at + 1024), vmemu(product + at), -2);
        const unsigned output = (r + i) * columns + c;
        auto a = vmemu(previous + output),
             b = vmemu(previous + output + columns);
        if constexpr (Decay) {
          a = Q6_Vhf_vmpy_VhfVhf(a, Q6_Vh_vsplat_R(decay));
          b = Q6_Vhf_vmpy_VhfVhf(b, Q6_Vh_vsplat_R(decay));
        }
        vmemu(target + output) = Q6_Vhf_vadd_VhfVhf(a, Q6_V_lo_W(pair));
        vmemu(target + output + columns) =
            Q6_Vhf_vadd_VhfVhf(b, Q6_V_hi_W(pair));
      }
}

alignas(128) inline constexpr auto columns16 = [] {
  std::array<uint16_t, 128> values{};
  for (unsigned i = 0; i < values.size(); ++i)
    values[i] = i;
  return values;
}();

inline void prepare(
    const uint16_t* decay,
    const uint16_t* gram,
    const uint16_t* qk,
    uint16_t* lower,
    uint16_t* attention,
    uint16_t* weight,
    uint16_t* prefix,
    unsigned heads,
    unsigned slice,
    unsigned slices) {
  for (unsigned h = slice; h < heads; h += slices) {
    const auto* dh = decay + h * chunk;
    const auto* kh = gram + h * chunk * chunk;
    const auto* qh = qk + h * chunk * chunk;
    auto* lh = lower + h * chunk * chunk;
    auto* ah = attention + h * chunk * chunk;
    float cumulative = 1.f;
    alignas(128) float prefix_rows[chunk], decay_rows[chunk];
    for (unsigned j = 0; j < chunk; j += 64) {
      const auto expanded = Q6_Wsf_vcvt_Vhf(vmemu(dh + j));
      const auto linear =
          Q6_W_vshuff_VVR(Q6_V_hi_W(expanded), Q6_V_lo_W(expanded), -4);
      vmemu(decay_rows + j) = Q6_V_lo_W(linear);
      vmemu(decay_rows + j + 32) = Q6_V_hi_W(linear);
    }
    HVX_Vector w[2] = {Q6_V_vzero(), Q6_V_vzero()};
    for (unsigned r = 0; r < chunk; r += 2) {
      HVX_Vector rows[2][2];
      for (unsigned row = 0; row < 2; ++row) {
        const unsigned i = r + row;
        const float d = decay_rows[i];
        cumulative = std::max(cumulative * d, floor);
        prefix_rows[i] = cumulative;
        HVX_Vector e[2];
        for (unsigned j = 0; j < 2; ++j) {
          if (j * 64 > i) {
            e[j] = Q6_V_vzero();
            continue;
          }
          const auto columns = vmemu(columns16.data() + j * 64);
          auto next = Q6_Vhf_vmax_VhfVhf(
              Q6_Vhf_vmpy_VhfVhf(w[j], Q6_Vh_vsplat_R(dh[i])),
              Q6_Vh_vsplat_R(2));
          next = Q6_V_vmux_QVV(
              Q6_Q_vcmp_eq_VhVh(columns, Q6_Vh_vsplat_R(i)),
              Q6_Vh_vsplat_R(0x3c00),
              next);
          w[j] = Q6_V_vmux_QVV(
              Q6_Q_vcmp_gt_VuhVuh(Q6_Vh_vsplat_R(i + 1), columns),
              next,
              Q6_V_vzero());
          e[j] = w[j];
        }
        for (unsigned j = 0; j < 2; ++j) {
          const unsigned c = j * 64;
          auto l = Q6_V_vzero();
          rows[row][j] = Q6_V_vzero();
          if (c <= i) {
            rows[row][j] = Q6_Vhf_vmpy_VhfVhf(vmemu(qh + i * chunk + c), e[j]);
            l = Q6_V_vmux_QVV(
                Q6_Q_vcmp_gt_VuhVuh(
                    Q6_Vh_vsplat_R(i), vmemu(columns16.data() + c)),
                Q6_Vhf_vmpy_VhfVhf(
                    vmemu(kh + i * chunk + c),
                    Q6_V_vxor_VV(e[j], Q6_Vh_vsplat_R(0x8000))),
                Q6_V_vzero());
          }
          vmemu(lh + i * chunk + c) = l;
          if (i == chunk - 1)
            vmemu(weight + h * chunk + c) = e[j];
        }
      }
      for (unsigned j = 0; j < 2; ++j) {
        const auto packed = Q6_W_vshuff_VVR(rows[1][j], rows[0][j], -2);
        const unsigned at = packed_index(r, j * 64);
        vmemu(ah + at) = Q6_V_lo_W(packed);
        vmemu(ah + at + 1024) = Q6_V_hi_W(packed);
      }
    }
    for (unsigned j = 0; j < chunk; j += 64) {
      const auto pair = Q6_W_vdeal_VVR(
          vmemu(prefix_rows + j + 32), vmemu(prefix_rows + j), -4);
      vmemu(prefix + h * chunk + j) =
          Q6_Vhf_vcvt_VsfVsf(Q6_V_lo_W(pair), Q6_V_hi_W(pair));
    }
  }
}
__attribute__((noinline)) inline void
solve_pair(uint16_t* left, uint16_t* right, uint16_t* rows) {
  auto* paired_rows = reinterpret_cast<HVX_Vector*>(rows);
  auto* left_vectors = reinterpret_cast<HVX_Vector*>(left);
  auto* right_vectors = reinterpret_cast<HVX_Vector*>(right);
  for (unsigned r = 0; r < 32; r += 2) {
    const auto raw =
        Q6_W_vdeal_VVR(right_vectors[r / 2], left_vectors[r / 2], -2);
    paired_rows[r] = Q6_V_lo_W(raw);
    paired_rows[r + 1] = Q6_V_hi_W(raw);
  }
  alignas(128) static constexpr auto half_lanes = [] {
    std::array<uint8_t, 128> a{};
    for (unsigned i = 0; i < a.size(); ++i)
      a[i] = i & 62;
    return a;
  }();
  alignas(128) static constexpr auto half_columns = [] {
    std::array<uint16_t, 64> a{};
    for (unsigned i = 0; i < a.size(); ++i)
      a[i] = i % 32;
    return a;
  }();
  const auto lanes = vmemu(half_lanes.data());
  const auto column = vmemu(half_columns.data());
  for (unsigned r = 0; r < 32; r += 8) {
    auto sum0 = paired_rows[r + 0];
    auto sum1 = paired_rows[r + 1];
    auto sum2 = paired_rows[r + 2];
    auto sum3 = paired_rows[r + 3];
    auto sum4 = paired_rows[r + 4];
    auto sum5 = paired_rows[r + 5];
    auto sum6 = paired_rows[r + 6];
    auto sum7 = paired_rows[r + 7];
    for (unsigned j = 1; j < r; ++j) {
      const auto prior = paired_rows[j];
      const auto control = Q6_V_vxor_VV(lanes, Q6_Vb_vsplat_R(j * 2));
      sum0 = Q6_Vhf_vadd_VhfVhf(
          sum0, Q6_Vhf_vmpy_VhfVhf(Q6_V_vdelta_VV(sum0, control), prior));
      sum1 = Q6_Vhf_vadd_VhfVhf(
          sum1, Q6_Vhf_vmpy_VhfVhf(Q6_V_vdelta_VV(sum1, control), prior));
      sum2 = Q6_Vhf_vadd_VhfVhf(
          sum2, Q6_Vhf_vmpy_VhfVhf(Q6_V_vdelta_VV(sum2, control), prior));
      sum3 = Q6_Vhf_vadd_VhfVhf(
          sum3, Q6_Vhf_vmpy_VhfVhf(Q6_V_vdelta_VV(sum3, control), prior));
      sum4 = Q6_Vhf_vadd_VhfVhf(
          sum4, Q6_Vhf_vmpy_VhfVhf(Q6_V_vdelta_VV(sum4, control), prior));
      sum5 = Q6_Vhf_vadd_VhfVhf(
          sum5, Q6_Vhf_vmpy_VhfVhf(Q6_V_vdelta_VV(sum5, control), prior));
      sum6 = Q6_Vhf_vadd_VhfVhf(
          sum6, Q6_Vhf_vmpy_VhfVhf(Q6_V_vdelta_VV(sum6, control), prior));
      sum7 = Q6_Vhf_vadd_VhfVhf(
          sum7, Q6_Vhf_vmpy_VhfVhf(Q6_V_vdelta_VV(sum7, control), prior));
    }
    if (r + 0 > 0) {
      const auto control = Q6_V_vxor_VV(lanes, Q6_Vb_vsplat_R((r + 0) * 2));
      sum1 = Q6_Vhf_vadd_VhfVhf(
          sum1, Q6_Vhf_vmpy_VhfVhf(Q6_V_vdelta_VV(sum1, control), sum0));
    }
    if (r + 0 > 0) {
      const auto control = Q6_V_vxor_VV(lanes, Q6_Vb_vsplat_R((r + 0) * 2));
      sum2 = Q6_Vhf_vadd_VhfVhf(
          sum2, Q6_Vhf_vmpy_VhfVhf(Q6_V_vdelta_VV(sum2, control), sum0));
    }
    if (r + 1 > 0) {
      const auto control = Q6_V_vxor_VV(lanes, Q6_Vb_vsplat_R((r + 1) * 2));
      sum2 = Q6_Vhf_vadd_VhfVhf(
          sum2, Q6_Vhf_vmpy_VhfVhf(Q6_V_vdelta_VV(sum2, control), sum1));
    }
    if (r + 0 > 0) {
      const auto control = Q6_V_vxor_VV(lanes, Q6_Vb_vsplat_R((r + 0) * 2));
      sum3 = Q6_Vhf_vadd_VhfVhf(
          sum3, Q6_Vhf_vmpy_VhfVhf(Q6_V_vdelta_VV(sum3, control), sum0));
    }
    if (r + 1 > 0) {
      const auto control = Q6_V_vxor_VV(lanes, Q6_Vb_vsplat_R((r + 1) * 2));
      sum3 = Q6_Vhf_vadd_VhfVhf(
          sum3, Q6_Vhf_vmpy_VhfVhf(Q6_V_vdelta_VV(sum3, control), sum1));
    }
    if (r + 2 > 0) {
      const auto control = Q6_V_vxor_VV(lanes, Q6_Vb_vsplat_R((r + 2) * 2));
      sum3 = Q6_Vhf_vadd_VhfVhf(
          sum3, Q6_Vhf_vmpy_VhfVhf(Q6_V_vdelta_VV(sum3, control), sum2));
    }
    if (r + 0 > 0) {
      const auto control = Q6_V_vxor_VV(lanes, Q6_Vb_vsplat_R((r + 0) * 2));
      sum4 = Q6_Vhf_vadd_VhfVhf(
          sum4, Q6_Vhf_vmpy_VhfVhf(Q6_V_vdelta_VV(sum4, control), sum0));
    }
    if (r + 1 > 0) {
      const auto control = Q6_V_vxor_VV(lanes, Q6_Vb_vsplat_R((r + 1) * 2));
      sum4 = Q6_Vhf_vadd_VhfVhf(
          sum4, Q6_Vhf_vmpy_VhfVhf(Q6_V_vdelta_VV(sum4, control), sum1));
    }
    if (r + 2 > 0) {
      const auto control = Q6_V_vxor_VV(lanes, Q6_Vb_vsplat_R((r + 2) * 2));
      sum4 = Q6_Vhf_vadd_VhfVhf(
          sum4, Q6_Vhf_vmpy_VhfVhf(Q6_V_vdelta_VV(sum4, control), sum2));
    }
    if (r + 3 > 0) {
      const auto control = Q6_V_vxor_VV(lanes, Q6_Vb_vsplat_R((r + 3) * 2));
      sum4 = Q6_Vhf_vadd_VhfVhf(
          sum4, Q6_Vhf_vmpy_VhfVhf(Q6_V_vdelta_VV(sum4, control), sum3));
    }
    if (r + 0 > 0) {
      const auto control = Q6_V_vxor_VV(lanes, Q6_Vb_vsplat_R((r + 0) * 2));
      sum5 = Q6_Vhf_vadd_VhfVhf(
          sum5, Q6_Vhf_vmpy_VhfVhf(Q6_V_vdelta_VV(sum5, control), sum0));
    }
    if (r + 1 > 0) {
      const auto control = Q6_V_vxor_VV(lanes, Q6_Vb_vsplat_R((r + 1) * 2));
      sum5 = Q6_Vhf_vadd_VhfVhf(
          sum5, Q6_Vhf_vmpy_VhfVhf(Q6_V_vdelta_VV(sum5, control), sum1));
    }
    if (r + 2 > 0) {
      const auto control = Q6_V_vxor_VV(lanes, Q6_Vb_vsplat_R((r + 2) * 2));
      sum5 = Q6_Vhf_vadd_VhfVhf(
          sum5, Q6_Vhf_vmpy_VhfVhf(Q6_V_vdelta_VV(sum5, control), sum2));
    }
    if (r + 3 > 0) {
      const auto control = Q6_V_vxor_VV(lanes, Q6_Vb_vsplat_R((r + 3) * 2));
      sum5 = Q6_Vhf_vadd_VhfVhf(
          sum5, Q6_Vhf_vmpy_VhfVhf(Q6_V_vdelta_VV(sum5, control), sum3));
    }
    if (r + 4 > 0) {
      const auto control = Q6_V_vxor_VV(lanes, Q6_Vb_vsplat_R((r + 4) * 2));
      sum5 = Q6_Vhf_vadd_VhfVhf(
          sum5, Q6_Vhf_vmpy_VhfVhf(Q6_V_vdelta_VV(sum5, control), sum4));
    }
    if (r + 0 > 0) {
      const auto control = Q6_V_vxor_VV(lanes, Q6_Vb_vsplat_R((r + 0) * 2));
      sum6 = Q6_Vhf_vadd_VhfVhf(
          sum6, Q6_Vhf_vmpy_VhfVhf(Q6_V_vdelta_VV(sum6, control), sum0));
    }
    if (r + 1 > 0) {
      const auto control = Q6_V_vxor_VV(lanes, Q6_Vb_vsplat_R((r + 1) * 2));
      sum6 = Q6_Vhf_vadd_VhfVhf(
          sum6, Q6_Vhf_vmpy_VhfVhf(Q6_V_vdelta_VV(sum6, control), sum1));
    }
    if (r + 2 > 0) {
      const auto control = Q6_V_vxor_VV(lanes, Q6_Vb_vsplat_R((r + 2) * 2));
      sum6 = Q6_Vhf_vadd_VhfVhf(
          sum6, Q6_Vhf_vmpy_VhfVhf(Q6_V_vdelta_VV(sum6, control), sum2));
    }
    if (r + 3 > 0) {
      const auto control = Q6_V_vxor_VV(lanes, Q6_Vb_vsplat_R((r + 3) * 2));
      sum6 = Q6_Vhf_vadd_VhfVhf(
          sum6, Q6_Vhf_vmpy_VhfVhf(Q6_V_vdelta_VV(sum6, control), sum3));
    }
    if (r + 4 > 0) {
      const auto control = Q6_V_vxor_VV(lanes, Q6_Vb_vsplat_R((r + 4) * 2));
      sum6 = Q6_Vhf_vadd_VhfVhf(
          sum6, Q6_Vhf_vmpy_VhfVhf(Q6_V_vdelta_VV(sum6, control), sum4));
    }
    if (r + 5 > 0) {
      const auto control = Q6_V_vxor_VV(lanes, Q6_Vb_vsplat_R((r + 5) * 2));
      sum6 = Q6_Vhf_vadd_VhfVhf(
          sum6, Q6_Vhf_vmpy_VhfVhf(Q6_V_vdelta_VV(sum6, control), sum5));
    }
    if (r + 0 > 0) {
      const auto control = Q6_V_vxor_VV(lanes, Q6_Vb_vsplat_R((r + 0) * 2));
      sum7 = Q6_Vhf_vadd_VhfVhf(
          sum7, Q6_Vhf_vmpy_VhfVhf(Q6_V_vdelta_VV(sum7, control), sum0));
    }
    if (r + 1 > 0) {
      const auto control = Q6_V_vxor_VV(lanes, Q6_Vb_vsplat_R((r + 1) * 2));
      sum7 = Q6_Vhf_vadd_VhfVhf(
          sum7, Q6_Vhf_vmpy_VhfVhf(Q6_V_vdelta_VV(sum7, control), sum1));
    }
    if (r + 2 > 0) {
      const auto control = Q6_V_vxor_VV(lanes, Q6_Vb_vsplat_R((r + 2) * 2));
      sum7 = Q6_Vhf_vadd_VhfVhf(
          sum7, Q6_Vhf_vmpy_VhfVhf(Q6_V_vdelta_VV(sum7, control), sum2));
    }
    if (r + 3 > 0) {
      const auto control = Q6_V_vxor_VV(lanes, Q6_Vb_vsplat_R((r + 3) * 2));
      sum7 = Q6_Vhf_vadd_VhfVhf(
          sum7, Q6_Vhf_vmpy_VhfVhf(Q6_V_vdelta_VV(sum7, control), sum3));
    }
    if (r + 4 > 0) {
      const auto control = Q6_V_vxor_VV(lanes, Q6_Vb_vsplat_R((r + 4) * 2));
      sum7 = Q6_Vhf_vadd_VhfVhf(
          sum7, Q6_Vhf_vmpy_VhfVhf(Q6_V_vdelta_VV(sum7, control), sum4));
    }
    if (r + 5 > 0) {
      const auto control = Q6_V_vxor_VV(lanes, Q6_Vb_vsplat_R((r + 5) * 2));
      sum7 = Q6_Vhf_vadd_VhfVhf(
          sum7, Q6_Vhf_vmpy_VhfVhf(Q6_V_vdelta_VV(sum7, control), sum5));
    }
    if (r + 6 > 0) {
      const auto control = Q6_V_vxor_VV(lanes, Q6_Vb_vsplat_R((r + 6) * 2));
      sum7 = Q6_Vhf_vadd_VhfVhf(
          sum7, Q6_Vhf_vmpy_VhfVhf(Q6_V_vdelta_VV(sum7, control), sum6));
    }
    paired_rows[r + 0] = sum0;
    paired_rows[r + 1] = sum1;
    paired_rows[r + 2] = sum2;
    paired_rows[r + 3] = sum3;
    paired_rows[r + 4] = sum4;
    paired_rows[r + 5] = sum5;
    paired_rows[r + 6] = sum6;
    paired_rows[r + 7] = sum7;
    const auto diagonal0 = Q6_V_vmux_QVV(
        Q6_Q_vcmp_eq_VhVh(column, Q6_Vh_vsplat_R(r + 0)),
        Q6_Vh_vsplat_R(0x3c00),
        sum0);
    const auto diagonal1 = Q6_V_vmux_QVV(
        Q6_Q_vcmp_eq_VhVh(column, Q6_Vh_vsplat_R(r + 1)),
        Q6_Vh_vsplat_R(0x3c00),
        sum1);
    const auto packed0 = Q6_W_vshuff_VVR(diagonal1, diagonal0, -2);
    left_vectors[(r + 0) / 2] = Q6_V_lo_W(packed0);
    right_vectors[(r + 0) / 2] = Q6_V_hi_W(packed0);
    const auto diagonal2 = Q6_V_vmux_QVV(
        Q6_Q_vcmp_eq_VhVh(column, Q6_Vh_vsplat_R(r + 2)),
        Q6_Vh_vsplat_R(0x3c00),
        sum2);
    const auto diagonal3 = Q6_V_vmux_QVV(
        Q6_Q_vcmp_eq_VhVh(column, Q6_Vh_vsplat_R(r + 3)),
        Q6_Vh_vsplat_R(0x3c00),
        sum3);
    const auto packed2 = Q6_W_vshuff_VVR(diagonal3, diagonal2, -2);
    left_vectors[(r + 2) / 2] = Q6_V_lo_W(packed2);
    right_vectors[(r + 2) / 2] = Q6_V_hi_W(packed2);
    const auto diagonal4 = Q6_V_vmux_QVV(
        Q6_Q_vcmp_eq_VhVh(column, Q6_Vh_vsplat_R(r + 4)),
        Q6_Vh_vsplat_R(0x3c00),
        sum4);
    const auto diagonal5 = Q6_V_vmux_QVV(
        Q6_Q_vcmp_eq_VhVh(column, Q6_Vh_vsplat_R(r + 5)),
        Q6_Vh_vsplat_R(0x3c00),
        sum5);
    const auto packed4 = Q6_W_vshuff_VVR(diagonal5, diagonal4, -2);
    left_vectors[(r + 4) / 2] = Q6_V_lo_W(packed4);
    right_vectors[(r + 4) / 2] = Q6_V_hi_W(packed4);
    const auto diagonal6 = Q6_V_vmux_QVV(
        Q6_Q_vcmp_eq_VhVh(column, Q6_Vh_vsplat_R(r + 6)),
        Q6_Vh_vsplat_R(0x3c00),
        sum6);
    const auto diagonal7 = Q6_V_vmux_QVV(
        Q6_Q_vcmp_eq_VhVh(column, Q6_Vh_vsplat_R(r + 7)),
        Q6_Vh_vsplat_R(0x3c00),
        sum7);
    const auto packed6 = Q6_W_vshuff_VVR(diagonal7, diagonal6, -2);
    left_vectors[(r + 6) / 2] = Q6_V_lo_W(packed6);
    right_vectors[(r + 6) / 2] = Q6_V_hi_W(packed6);
  }
}

#if defined(__hexagon__) && QCOM_HTP_V81
static_assert(sizeof(attention_hmx::Queue) <= 2048);
template <bool Columns, bool Causal = false>
static inline void multiply(
    uint8_t* a,
    uint8_t* b,
    uint8_t* c,
    unsigned m,
    unsigned k,
    unsigned n) {
  for (unsigned r = 0; r < m; r += 32) {
    const unsigned prefix = Causal ? (r / 64 + 1) * 64 : k;
    const unsigned ac = ((prefix / 32) << 11) | (31 << 2);
    const unsigned bc = ((prefix / 32) << 11) - 1;
    for (unsigned col = 0; col < n; col += 32) {
      auto* ap = a + r * k * 2;
      auto* bp = b + col * k * 2;
      const unsigned tile =
          Columns ? tile_index(r, col, m) : (r / 32) * (n / 32) + col / 32;
      auto* cp = c + tile * 2048;
      asm volatile(
          "{ activation.hf = mxmem(%0,%1):deep; weight.hf = mxmem(%2,%3) }\n"
          "{ cvt.hf = acc(%4) }\n"
          "{ mxmem(%5,%6) = cvt }\n" ::"r"(ap),
          "r"(ac),
          "r"(bp),
          "r"(bc),
          "r"(2),
          "r"(cp),
          "r"(0)
          : "memory");
    }
  }
}

static inline void
block_multiply(uint8_t* a, uint8_t* b, uint8_t* c, unsigned output_row) {
  const unsigned ac = (2 << 11) | (31 << 2), bc = (2 << 11) - 1;
  for (unsigned r = 0; r < 64; r += 32)
    for (unsigned col = 0; col < width; col += 32) {
      auto* ap = a + r * chunk * 2;
      auto* bp = b + col * chunk * 2;
      auto* cp = c + tile_index(r + output_row, col, chunk) * 2048;
      asm volatile(
          "{ activation.hf = mxmem(%0,%1):deep; weight.hf = mxmem(%2,%3) }\n"
          "{ cvt.hf = acc(%4) }\n"
          "{ mxmem(%5,%6) = cvt }\n" ::"r"(ap),
          "r"(ac),
          "r"(bp),
          "r"(bc),
          "r"(2),
          "r"(cp),
          "r"(0)
          : "memory");
    }
}
static inline void drain(const uint8_t* source, unsigned bytes) {
  const volatile HVX_Vector value = vmemu(source + bytes - 128);
  (void)value;
}

static inline void merge_triangle(uint8_t* base, unsigned begin, unsigned end) {
  const unsigned ac = (1 << 11) | (31 << 2), bc = (1 << 11) - 1;
  auto* packed = base + t_offset;
  auto* temp = base + row_offset;
  for (unsigned start = begin; start < end; start += 64) {
    auto* left = packed + tile_offset(start, start) * 2;
    auto* right = packed + tile_offset(start + 32, start + 32) * 2;
    auto* cross = packed + tile_offset(start + 32, start) * 2;
    auto gemm = [&](uint8_t* a, uint8_t* b, uint8_t* c) {
      asm volatile(
          "{ activation.hf = mxmem(%0,%1):deep; weight.hf = mxmem(%2,%3) }\n"
          "{ cvt.hf = acc(%4) }\n"
          "{ mxmem(%5,%6) = cvt }\n" ::"r"(a),
          "r"(ac),
          "r"(b),
          "r"(bc),
          "r"(2),
          "r"(c),
          "r"(0)
          : "memory");
      drain(c, 2048);
    };
    gemm(right, cross, temp);
    gemm(temp, left, cross);
  }
}

inline bool serve(attention_hmx::Queue& queue) {
  bool scales_loaded = false;
  return attention_hmx::serve(queue, [&](const attention_hmx::Job& job) {
    if (!scales_loaded) {
      asm volatile("bias = mxmem2(%0)" ::"r"(job.bias) : "memory");
      scales_loaded = true;
    }
    auto* base = job.a;

    if (job.panel == 0) {
      multiply<false>(base + ba, base + kt, base + gram, chunk, chunk, width);
      multiply<false>(base + qa, base + kt, base + qk, chunk, chunk, width);
      multiply<false>(base + ka, base + st, base + ba, chunk, chunk, width);
      multiply<false>(base + qa, base + st, base + kt, chunk, chunk, width);
      drain(base + kt, matrix_bytes);
    } else if (job.panel == 1) {
      merge_triangle(base, 0, 64);
      block_multiply(base + t_offset, base + z_offset, base + c_offset, 0);
      drain(base + c_offset + (width / 32 - 1) * chunk * 64, 64 * 64);
      block_multiply(
          base + t_offset + 64 * chunk * 2,
          base + c_offset,
          base + row_offset,
          64);
      drain(base + row_offset, chunk * width * 2);

    } else if (job.panel == 2) {
      merge_triangle(base, 64, 128);
      block_multiply(
          base + t_offset + 64 * chunk * 2 + 64 * 64,
          base + z_offset + 64 * 64,
          base + c_offset,
          64);
      drain(base + c_offset, chunk * width * 2);

    } else {
      multiply<false, true>(
          job.b, base + c_offset, base + y_offset, chunk, chunk, width);
      multiply<false>(
          base + k_offset,
          base + z_offset,
          base + s_offset,
          width,
          chunk,
          width);
      drain(base + s_offset, width * width * 2);
    }

    return attention_hmx::Status::Done;
  });
}
inline bool run(
    const uint16_t* const* global,
    uint16_t* global_y,
    uint16_t* global_updated,
    uint8_t* scratch,
    unsigned heads,
    unsigned slice,
    unsigned slices,
    attention_hmx::Queue& queue) {
  __atomic_store_n(&queue.workers, slices, __ATOMIC_RELEASE);
  auto* base = scratch + 2048 + slice * worker_bytes;
  auto* pre = base + correction_bytes;
  auto ptr = [&](unsigned offset) {
    return reinterpret_cast<uint16_t*>(pre + offset);
  };
  auto* bias = base + bias_offset;
  vmemu(bias) = Q6_V_vsplat_R(0x3c00);
  vmemu(bias + 128) = Q6_V_vzero();
  uint8_t* packed_attention = nullptr;
  auto enqueue = [&](unsigned phase) {
    if (__atomic_load_n(&queue.error, __ATOMIC_ACQUIRE))
      return false;
    auto& job = queue.jobs[slice];
    job.a = base;
    job.b = packed_attention;
    job.c = nullptr;
    job.bias = bias;
    job.m = 32;
    job.k = chunk;
    job.n = width;
    job.batches = 1;
    job.a_stride = job.c_stride = job.bias_stride = 0;
    job.high = nullptr;
    job.b_stride = chunk;
    job.panel = phase;
    __atomic_store_n(&job.pending, 1u, __ATOMIC_RELEASE);
    return true;
  };
  auto wait = [&]() {
    while (__atomic_load_n(&queue.jobs[slice].pending, __ATOMIC_ACQUIRE))
      attention_hmx::pause();
    return __atomic_load_n(&queue.error, __ATOMIC_ACQUIRE) == 0;
  };
  bool okay = true;
  for (unsigned head = slice; head < heads; head += slices) {
    const unsigned offset = head * chunk * width;
    alignas(128) uint16_t beta_rows[chunk];
    vmemu(beta_rows) = vmemu(global[4] + head * chunk);
    vmemu(beta_rows + 64) = vmemu(global[4] + head * chunk + 64);
    hmx_tile::pack<false>(global[0] + offset, ptr(qa), chunk, width);
    hmx_tile::pack<false>(global[1] + offset, ptr(ka), chunk, width);
    hmx_tile::pack<true>(global[6] + offset, ptr(kt), chunk, width);
    hmx_tile::pack<true>(global[5] + offset, ptr(st), chunk, width);
    for (unsigned r = 0; r < chunk; r += 2) {
      const auto beta = Q6_V_vsplat_R(
          unsigned(beta_rows[r]) | (unsigned(beta_rows[r + 1]) << 16));
      for (unsigned c = 0; c < width; c += 32) {
        const unsigned at =
            (r / 32) * width * 32 + (c / 32) * 1024 + (r % 32 / 2) * 64;
        vmemu(ptr(ba) + at) = Q6_Vhf_vmpy_VhfVhf(vmemu(ptr(ka) + at), beta);
      }
    }
    vmemu(pre + pre_bias) = Q6_V_vsplat_R(0x3c00);
    vmemu(pre + pre_bias + 128) = Q6_V_vzero();

    if (!attention_hmx::submit(
            queue,
            slice,
            pre,
            nullptr,
            nullptr,
            pre + pre_bias,
            chunk,
            chunk,
            width)) {
      okay = false;
      break;
    }

    hmx_tile::unpack(ptr(gram), ptr(row0), chunk, width);
    hmx_tile::unpack(ptr(qk), ptr(row1), chunk, width);
    alignas(128) uint16_t decay[chunk], weight[chunk], prefix[chunk];
    for (unsigned r = 0; r < chunk; r += 64)
      vmemu(decay + r) = Q6_Vhf_vmax_VhfVhf(
          vmemu(global[3] + head * chunk + r), Q6_Vh_vsplat_R(0x400));

    prepare(
        decay,
        ptr(row0),
        ptr(row1),
        ptr(gram),
        ptr(qk),
        weight,
        prefix,
        1,
        0,
        1);

    auto* source_tiles = reinterpret_cast<uint16_t*>(base + z_offset);
    for (unsigned r = 0; r < chunk; r += 2) {
      const auto decay_prefix =
          Q6_V_vsplat_R(unsigned(prefix[r]) | (unsigned(prefix[r + 1]) << 16));
      const auto beta = Q6_V_vsplat_R(
          unsigned(beta_rows[r]) | (unsigned(beta_rows[r + 1]) << 16));
      for (unsigned c = 0; c < width; c += 64) {
        const unsigned at =
            (r / 32) * width * 32 + (c / 32) * 1024 + (r % 32 / 2) * 64;
        const auto values = Q6_W_vshuff_VVR(
            vmemu(global[2] + offset + (r + 1) * width + c),
            vmemu(global[2] + offset + r * width + c),
            -2);
        const auto predict0 =
            Q6_Vhf_vmpy_VhfVhf(decay_prefix, vmemu(ptr(ba) + at));
        const auto predict1 =
            Q6_Vhf_vmpy_VhfVhf(decay_prefix, vmemu(ptr(ba) + at + 1024));
        const auto z0 = Q6_Vhf_vmpy_VhfVhf(
            beta, Q6_Vhf_vsub_VhfVhf(Q6_V_lo_W(values), predict0));
        const auto z1 = Q6_Vhf_vmpy_VhfVhf(
            beta, Q6_Vhf_vsub_VhfVhf(Q6_V_hi_W(values), predict1));
        const unsigned target = (c / 32) * chunk * 32 + (r / 2) * 64;
        vmemu(source_tiles + target) = z0;
        vmemu(source_tiles + target + chunk * 32) = z1;
        const auto read0 =
            Q6_Vhf_vmpy_VhfVhf(decay_prefix, vmemu(ptr(kt) + at));
        const auto read1 =
            Q6_Vhf_vmpy_VhfVhf(decay_prefix, vmemu(ptr(kt) + at + 1024));
        const auto readout = Q6_W_vdeal_VVR(read1, read0, -2);
        vmemu(ptr(ka) + r * width + c) = Q6_V_lo_W(readout);
        vmemu(ptr(ka) + (r + 1) * width + c) = Q6_V_hi_W(readout);
      }
    }

    const uint16_t* input[8] = {
        ptr(gram),
        source_tiles,
        ptr(qk),
        global[5] + offset,
        global[6] + offset,
        weight,
        prefix + chunk - 1,
        ptr(ka)};
    auto* y = global_y + offset;
    auto* updated = global_updated + offset;

    hmx_tile::pack<false>(
        input[0], reinterpret_cast<uint16_t*>(base + t_offset), chunk, chunk);

    auto solve = [&](unsigned begin, unsigned end) {
      for (unsigned row = begin; row < end; row += 64)
        solve_pair(
            reinterpret_cast<uint16_t*>(base + t_offset) +
                tile_offset(row, row),
            reinterpret_cast<uint16_t*>(base + t_offset) +
                tile_offset(row + 32, row + 32),
            reinterpret_cast<uint16_t*>(base + s_offset));
    };
    solve(0, 64);

    packed_attention =
        reinterpret_cast<uint8_t*>(const_cast<uint16_t*>(input[2]));
    if (!enqueue(1)) {
      okay = false;
      break;
    }
    solve(64, 128);
    hmx_tile::pack<false>(
        input[4], reinterpret_cast<uint16_t*>(base + k_offset), width, chunk);

    if (!wait()) {
      okay = false;
      break;
    }

    // Z1 += L10*C0 directly in B-operand tile order; C0 stays packed.
    for (unsigned col = 0; col < width; col += 32)
      for (unsigned r = 64; r < chunk; r += 2) {
        const unsigned at = (col / 32) * chunk * 32 + (r / 2) * 64;
        auto* z = reinterpret_cast<uint16_t*>(base + z_offset) + at;
        const auto* product =
            reinterpret_cast<const uint16_t*>(base + row_offset) + at;
        vmemu(z) = Q6_Vhf_vadd_VhfVhf(vmemu(z), vmemu(product));
      }

    if (!enqueue(2)) {
      okay = false;
      break;
    }

    if (!wait()) {
      okay = false;
      break;
    }

    auto* correction = reinterpret_cast<uint16_t*>(base + c_offset);
    // Preserve FP16 rounding of weight*correction; the Z operand is now dead.
    weight_columns(
        correction,
        input[5],
        reinterpret_cast<uint16_t*>(base + z_offset),
        chunk,
        width);

    if (!enqueue(3)) {
      okay = false;
      break;
    }

    if (!wait()) {
      okay = false;
      break;
    }

    finish<false>(
        reinterpret_cast<const uint16_t*>(base + y_offset),
        input[7],
        0,
        y,
        chunk,
        width);
    finish<true>(
        reinterpret_cast<const uint16_t*>(base + s_offset),
        input[3],
        input[6][0],
        updated,
        width,
        width);
  }
  __atomic_store_n(&queue.jobs[slice].finished, 1u, __ATOMIC_RELEASE);
  return okay;
}
#endif
} // namespace gdn_chunk
