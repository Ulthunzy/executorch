#include "sampling.h"
#include <arm_neon.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

// temperature=0 的 greedy 采样；NEON 扫描同时拒绝
// NaN/Inf，并保持首个最大值的索引。
size_t checked_argmax(const float* values, size_t count) {
  if (!count)
    throw std::runtime_error("empty logits");
  float32x4_t best = vdupq_n_f32(-std::numeric_limits<float>::infinity());
  uint32x4_t indices = vdupq_n_u32(0), invalid = vdupq_n_u32(0);
  const uint32x4_t exponent = vdupq_n_u32(0x7f800000);
  const uint32_t lanes[4] = {0, 1, 2, 3};
  uint32x4_t current = vld1q_u32(lanes);
  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    const auto value = vld1q_f32(values + i);
    invalid = vorrq_u32(
        invalid,
        vceqq_u32(vandq_u32(vreinterpretq_u32_f32(value), exponent), exponent));
    const auto greater = vcgtq_f32(value, best);
    best = vbslq_f32(greater, value, best);
    indices = vbslq_u32(greater, current, indices);
    current = vaddq_u32(current, vdupq_n_u32(4));
  }
  if (vmaxvq_u32(invalid))
    throw std::runtime_error("nonfinite logits");
  float maximum = -std::numeric_limits<float>::infinity();
  size_t index = 0;
  float maxima[4];
  uint32_t positions[4];
  vst1q_f32(maxima, best);
  vst1q_u32(positions, indices);
  // Preserve std::max_element's first-index tie rule across SIMD lanes.
  for (size_t lane = 0; lane < 4; lane++)
    if (maxima[lane] > maximum ||
        (maxima[lane] == maximum && positions[lane] < index)) {
      maximum = maxima[lane];
      index = positions[lane];
    }
  for (; i < count; i++) {
    uint32_t bits;
    __builtin_memcpy(&bits, values + i, 4);
    if ((bits & 0x7f800000) == 0x7f800000)
      throw std::runtime_error("nonfinite logits");
    if (values[i] > maximum) {
      maximum = values[i];
      index = i;
    }
  }
  return index;
}
