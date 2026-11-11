// Copyright (c) Qualcomm Innovation Center, Inc.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#define DELTANET_U16_KARATSUBA_REDUCTION 1
#define DELTANET_U16_RADIX256_REDUCTION 1
#include <cstdio>
#include <limits>
#include <random>
#include "../src/kernels/DeltaNetU16.h"

int main() {
  alignas(128) uint16_t state[256 * 256 + 64];
  alignas(128) int32_t key[256], query[256];
  alignas(128) QUint16StateAccum kv[256], out[256];
  std::mt19937 rng(87508850);
  double worst = 0, worst_fallback = 0;
  for (const size_t kdim :
       {size_t(1), size_t(16), size_t(64), size_t(128), size_t(256)}) {
    for (const size_t vdim :
         {size_t(64), size_t(128), size_t(192), size_t(256)}) {
      for (const int zero : {0, 1, 255, 256, 32767, 32768, 65535}) {
        for (const int coefficient_zero : {-1, 0, 32768, 65535, 65536}) {
          for (const size_t misalignment : {size_t(0), size_t(1)}) {
            uint16_t* state_data = state + misalignment;
            for (int trial = 0; trial != 6; ++trial) {
              for (size_t k = 0; k != kdim; ++k) {
                key[k] = (trial == 0       ? 0
                              : trial == 1 ? 65535
                                           : int(rng() & 65535)) -
                    coefficient_zero;
                query[k] = (trial == 1       ? 0
                                : trial == 0 ? 65535
                                             : int(rng() & 65535)) -
                    (65535 - coefficient_zero);
              }
              for (size_t i = 0; i != kdim * vdim; ++i) {
                state_data[i] = trial == 0 ? 0
                    : trial == 1           ? 65535
                                           : uint16_t(rng());
              }
              if (misalignment) {
                deltanetQhpiHvxReduceStateU16<false>(
                    state_data,
                    kv,
                    out,
                    zero,
                    coefficient_zero,
                    65535 - coefficient_zero,
                    key,
                    query,
                    kdim,
                    vdim,
                    vdim);
              } else if (vdim % 128 == 0) {
                deltanetQhpiHvxReduceStateU16<true, 2>(
                    state_data,
                    kv,
                    out,
                    zero,
                    coefficient_zero,
                    65535 - coefficient_zero,
                    key,
                    query,
                    kdim,
                    vdim,
                    vdim);
              } else {
                deltanetQhpiHvxReduceStateU16<true>(
                    state_data,
                    kv,
                    out,
                    zero,
                    coefficient_zero,
                    65535 - coefficient_zero,
                    key,
                    query,
                    kdim,
                    vdim,
                    vdim);
              }
              for (size_t v = 0; v != vdim; ++v) {
                int64_t reference_k = 0, reference_q = 0;
                double magnitude_k = 0, magnitude_q = 0;
                for (size_t k = 0; k != kdim; ++k) {
                  const int64_t s = int64_t(state_data[k * vdim + v]) - zero;
                  reference_k += s * key[k];
                  reference_q += s * query[k];
                  magnitude_k += std::abs(double(s * key[k]));
                  magnitude_q += std::abs(double(s * query[k]));
                }
                const double error_k = std::abs(double(kv[v]) - reference_k);
                const double error_q = std::abs(double(out[v]) - reference_q);
                const bool fast = kdim <= 128 && coefficient_zero >= 0 &&
                    coefficient_zero <= 65535;
                double& maximum_error = fast ? worst : worst_fallback;
                maximum_error =
                    std::max(maximum_error, std::max(error_k, error_q));
                const double tolerance = kdim <= 128
                    ? 4e-7
                    : (2 * kdim + 1) * std::numeric_limits<float>::epsilon();
                const double bound_k = std::max(1.0, magnitude_k * tolerance);
                const double bound_q = std::max(1.0, magnitude_q * tolerance);
                if (error_k > bound_k || error_q > bound_q) {
                  std::printf(
                      "FAIL K=%zu V=%zu zero=%d trial=%d lane=%zu errors=%g/%g bounds=%g/%g\n",
                      kdim,
                      vdim,
                      zero,
                      trial,
                      v,
                      error_k,
                      error_q,
                      bound_k,
                      bound_q);
                  return 1;
                }
              }
            }
          }
        }
      }
    }
  }
  std::printf(
      "radix256 endpoint/random/fallback reductions passed; "
      "worst integer-domain abs error=%g (old fallback=%g)\n",
      worst,
      worst_fallback);
}
