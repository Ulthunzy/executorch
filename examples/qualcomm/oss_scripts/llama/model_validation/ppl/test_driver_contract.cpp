/* Copyright (c) Meta Platforms, Inc. and affiliates.
 * Licensed under the BSD-style license in the root LICENSE file.
 */
#include "driver_contract.h"

#include <cassert>
#include <limits>

int main() {
  assert(ppl::prefill_count(32, 8, 31) == 24);
  assert(ppl::prefill_count(35, 8, 34) == 24);
  assert(ppl::prefill_count(32, 8, 7) == 7);
  assert(ppl::prefill_count(32, 32, 31) == 31);
  assert(ppl::prefill_count(32, 1, 31) == 31);
  bool rejected = false;
  try {
    ppl::prefill_count(32, 20, 31);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  assert(rejected);
  assert(std::abs(ppl::float_nll({0.f, 0.f, 0.f}, 1) - std::log(3.)) < 1e-12);
  const double scale = 0.25;
  std::vector<double> exponential(65536);
  for (size_t i = 0; i < exponential.size(); ++i) {
    exponential[i] = std::exp(-static_cast<double>(i) * scale);
  }
  const auto quantized = ppl::quantized_nll({100, 104, 98}, 2, exponential, scale);
  const auto floating = ppl::float_nll({25.f, 26.f, 24.5f}, 2);
  assert(std::abs(quantized - floating) < 1e-12);
  rejected = false;
  try {
    ppl::float_nll({0.f, std::numeric_limits<float>::infinity()}, 0);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  assert(rejected);
}
