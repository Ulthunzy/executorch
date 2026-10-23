#pragma once
#include <cstdint>
#include "hmx_hexagon_protos.h"

// Each native 32x32 tile uses 2048 activation bytes and 512 packed N4 bytes.
// The upper byte of each activation pair carries the unsigned A8 code.
static inline int activation_offset(int m, int k) {
  return (k / 32) * 2048 + (m / 2) * 128 + (k % 32) * 4 + (m % 2) * 2 + 1;
}

static inline int weight_offset(int n, int k) {
  return (k / 32) * 512 + (k % 32 / 8) * 128 + n * 4 + k % 4;
}

static inline int output_offset(int m, int n) {
  return (m / 2) * 64 + n * 2 + m % 2;
}

static inline void integer_hmx(
    const uint8_t* a,
    const uint8_t* w,
    const uint32_t* bias,
    uint16_t* output,
    int k) {
  // bias 同时携带定点缩放及 A8 零点修正；输出仍为 HMX 原始 tile，HVX
  // 再恢复实数尺度。
  Q6_bias_mxmem2_A((void*)bias);
  Q6_mxclracc();
  Q6_activation_ub_mxmem_RR_deep(uintptr_t(a), ((k / 32 - 1) << 11) | 0x77c);
  Q6_weight_n_mxmem_RR(uintptr_t(w), k * 16 - 1);
  Q6_mxmem_AR_after_uh_2x1(output, 0x77c);
}
