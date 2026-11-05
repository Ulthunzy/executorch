// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <algorithm>
#include <limits>
#include "AttentionConfig.h"

namespace attention {

struct Inputs {
  const uint16_t *query, *key, *new_key, *value, *new_value;
  const float* grids;
  Quantization query_quant, new_key_quant, new_value_quant, output_quant;
};

// Portable FP32 fallback. Work is one output vector and a bounded score tile;
// neither allocation nor iteration depends on unused cache capacity.
inline void reference_row(
    const Configuration& c,
    const Inputs& in,
    unsigned batch,
    unsigned head,
    unsigned row,
    float* work,
    uint16_t* output) {
  const auto& g = c.geometry;
  const size_t query_index =
      ((size_t(batch) * g.heads + head) * g.queries + row) * g.dimension;
  if (row >= c.valid) {
    std::fill(
        output + query_index,
        output + query_index + g.dimension,
        in.output_quant.zero);
    return;
  }
  const unsigned kv_head = head / (g.heads / g.kv_heads);
  const size_t kv = size_t(batch) * g.kv_heads + kv_head;
  const auto* meta = in.grids;
  const Quantization kq{meta[9], int(meta[10])}, vq{meta[11], int(meta[12])};
  const Quantization dot_quant{meta[0], int(meta[1])};
  const Quantization scaled_quant{meta[3], int(meta[4])};
  const Quantization logit_quant{meta[5], int(meta[6])};
  float* scores = work + g.dimension;
  std::fill(work, work + g.dimension, 0.0f);
  float maximum = -std::numeric_limits<float>::infinity(), normalizer = 0;
  const unsigned keys = c.history + row + 1;
  for (unsigned start = 0; start < keys; start += 256) {
    const unsigned width = std::min(256u, keys - start);
    float next_maximum = maximum;
    for (unsigned col = 0; col < width; ++col) {
      const unsigned logical = start + col;
      float dot = 0;
      for (unsigned d = 0; d < g.dimension; ++d) {
        const float k = logical < c.history
            ? kq.dequantize(
                  in.key
                      [(kv * g.dimension + d) * c.capacity + c.first + logical])
            : in.new_key_quant.dequantize(
                  in.new_key
                      [(kv * g.dimension + d) * g.queries + logical -
                       c.history]);
        dot += in.query_quant.dequantize(in.query[query_index + d]) * k;
      }
      dot = dot_quant.dequantize(dot_quant.quantize(dot));
      dot = scaled_quant.dequantize(scaled_quant.quantize(dot * meta[2]));
      scores[col] = logit_quant.dequantize(logit_quant.quantize(dot));
      next_maximum = std::max(next_maximum, scores[col]);
    }
    const float alpha = std::exp(maximum - next_maximum);
    normalizer *= alpha;
    for (unsigned d = 0; d < g.dimension; ++d) {
      work[d] *= alpha;
    }
    for (unsigned col = 0; col < width; ++col) {
      const unsigned logical = start + col;
      const float probability = std::exp(scores[col] - next_maximum);
      normalizer += probability;
      for (unsigned d = 0; d < g.dimension; ++d) {
        const float v = logical < c.history
            ? vq.dequantize(
                  in.value
                      [(kv * c.capacity + c.first + logical) * g.dimension + d])
            : in.new_value_quant.dequantize(
                  in.new_value
                      [(kv * g.queries + logical - c.history) * g.dimension +
                       d]);
        work[d] += probability * v;
      }
    }
    maximum = next_maximum;
  }
  for (unsigned d = 0; d < g.dimension; ++d) {
    output[query_index + d] = in.output_quant.quantize(work[d] / normalizer);
  }
}

} // namespace attention
