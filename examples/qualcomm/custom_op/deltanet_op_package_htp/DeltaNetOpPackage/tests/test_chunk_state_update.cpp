#include <algorithm>
#include <cstdio>
#include <random>
// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "HvxUtils.h"

int main() {
  alignas(128) uint16_t a[64], b[64], actual[128];
  std::mt19937 random(47);
  for (unsigned trial = 0; trial < 1024; ++trial) {
    for (unsigned i = 0; i < 64; ++i) {
      a[i] = random();
      b[i] = random();
    }
    const auto pair = Q6_W_vshuff_VVR(vmemu(b), vmemu(a), -2);
    const auto inverse = Q6_W_vdeal_VVR(Q6_V_hi_W(pair), Q6_V_lo_W(pair), -2);
    vmemu(actual) = Q6_V_lo_W(inverse);
    vmemu(actual + 64) = Q6_V_hi_W(inverse);
    if (!std::equal(a, a + 64, actual) || !std::equal(b, b + 64, actual + 64))
      return std::puts("FAIL packed row-pair inverse"), 1;
  }
  alignas(128) Float16 state[64], product[64], result[64];
  unsigned checks = 0;
  for (unsigned trial = 0; trial < 1024; ++trial) {
    const Float16 decay(float(random() % 65536) / 65536);
    for (unsigned i = 0; i < 64; ++i) {
      state[i] = Float16((int(random() % 65536) - 32768) / 8192.f);
      product[i] = Float16((int(random() % 65536) - 32768) / 65536.f);
    }
    const auto scaled =
        Q6_Vhf_vmpy_VhfVhf(vmemu(state), Q6_Vh_vsplat_R(decay.raw()));
    vmemu(result) = Q6_Vhf_vadd_VhfVhf(scaled, vmemu(product));
    for (unsigned i = 0; i < 64; ++i) {
      const auto old = Float16(float(state[i]) * float(decay));
      const auto expected = Float16(float(old) + float(product[i]));
      if (expected.raw() != result[i].raw())
        return std::printf(
                   "FAIL half feedback trial=%u lane=%u got=%u expected=%u\n",
                   trial,
                   i,
                   result[i].raw(),
                   expected.raw()),
               1;
      ++checks;
    }
  }
  std::printf(
      "PASS 131072 packed half lanes and %u decay/add feedback lanes\n",
      checks);
  alignas(128) uint8_t lanes[128];
  alignas(128) uint32_t values[32], broadcast[32];
  for (unsigned i = 0; i < 128; ++i)
    lanes[i] = i & 124;
  for (unsigned i = 0; i < 32; ++i)
    values[i] = random();
  const HVX_Vector vector = vmemu(values), base = vmemu(lanes);
  for (unsigned j = 0; j < 32; ++j) {
    const auto control = Q6_V_vxor_VV(base, Q6_Vb_vsplat_R(j * 4));
    vmemu(broadcast) = Q6_V_vdelta_VV(vector, control);
    for (unsigned i = 0; i < 32; ++i)
      if (broadcast[i] != values[j])
        return std::printf(
                   "FAIL broadcast word=%u lane=%u got=%08x expected=%08x\n",
                   j,
                   i,
                   broadcast[i],
                   values[j]),
               1;
  }
  std::puts("PASS 1024 register broadcast lanes");
}
