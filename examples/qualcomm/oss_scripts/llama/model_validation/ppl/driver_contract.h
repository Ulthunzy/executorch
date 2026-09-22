/* Copyright (c) Meta Platforms, Inc. and affiliates.
 * Licensed under the BSD-style license in the root LICENSE file.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <stdexcept>
#include <vector>

namespace ppl {

inline size_t prefill_count(size_t context, size_t ar, size_t inputs) {
  if (context < 2 || !ar || ar > context || !inputs || inputs >= context) {
    throw std::runtime_error("Invalid compiled context/AR or input count");
  }
  if (ar == context || inputs <= context - ar) {
    return inputs;
  }
  const size_t count = ((context - ar) / ar) * ar;
  if (!count) {
    throw std::runtime_error("No AR-aligned prefill fits the compiled context");
  }
  return count;
}

inline double float_nll(const std::vector<float>& logits, size_t target) {
  if (logits.empty() || target >= logits.size() ||
      !std::all_of(logits.begin(), logits.end(), [](float x) { return std::isfinite(x); })) {
    throw std::runtime_error("Invalid floating logits or target ID");
  }
  const double maximum = *std::max_element(logits.begin(), logits.end());
  double sum = 0;
  for (float value : logits) {
    sum += std::exp(static_cast<double>(value) - maximum);
  }
  return std::log(sum) + maximum - logits[target];
}

inline double quantized_nll(
    const std::vector<uint16_t>& logits,
    size_t target,
    const std::vector<double>& exponential,
    double scale) {
  if (logits.empty() || target >= logits.size() || exponential.size() != 65536 ||
      !std::isfinite(scale) || scale <= 0) {
    throw std::runtime_error("Invalid quantized logits, scale or target ID");
  }
  const auto maximum = *std::max_element(logits.begin(), logits.end());
  double sum = 0;
  for (uint16_t value : logits) {
    sum += exponential[maximum - value];
  }
  return std::log(sum) + (maximum - logits[target]) * scale;
}

template <typename T>
inline size_t top1_token(const std::vector<T>& logits) {
  if (logits.empty()) {
    throw std::runtime_error("Cannot select top-1 from empty logits");
  }
  return static_cast<size_t>(std::distance(
      logits.begin(), std::max_element(logits.begin(), logits.end())));
}

} // namespace ppl
