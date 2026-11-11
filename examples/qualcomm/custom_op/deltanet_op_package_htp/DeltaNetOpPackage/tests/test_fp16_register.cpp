// Copyright (c) Qualcomm Innovation Center, Inc.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cassert>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <random>
#include "../src/kernels/DeltaNetFloat.h"

int main(int argc, char** argv) {
  if (argc != 2)
    return 1;
  std::mt19937 random(88508750);
  std::uniform_real_distribution<float> values(-.2f, .2f);
#if DELTANET_FP16_REGISTER_REUSE
  alignas(128) static Float16 input[256 * 256 + 64];
  alignas(128) static Float16 expected[256 * 256 + 64], actual[256 * 256 + 64];
  alignas(128) Float16 key[256], query[256];
  alignas(128) float expected_kv[256], expected_out[256];
  alignas(128) float actual_kv[256], actual_out[256], delta[256];
  for (size_t kdim : {1u, 16u, 128u, 256u})
    for (size_t vdim : {64u, 128u, 192u, 256u})
      for (size_t misalignment : {0u, 1u})
        for (float decay : {0.f, .98f, 1.f, -.5f}) {
          auto* state = input + misalignment;
          for (size_t i = 0; i < kdim * vdim; ++i)
            state[i] = Float16(i % 7 == 0 ? -0.f : values(random));
          for (size_t k = 0; k < kdim; ++k) {
            key[k] = Float16(values(random));
            query[k] = Float16(values(random));
          }
          for (size_t v = 0; v < vdim; ++v)
            delta[v] = values(random);
          deltanetQhpiHvxAssignAccumF16(
              state,
              expected_kv,
              expected_out,
              decay * float(key[0]),
              decay * float(query[0]),
              vdim);
          for (size_t k = 1; k < kdim; ++k)
            deltanetQhpiHvxAccumF16(
                state + k * vdim,
                expected_kv,
                expected_out,
                decay * float(key[k]),
                decay * float(query[k]),
                vdim);
          if (!misalignment && vdim % 128 == 0)
            deltanetQhpiHvxReduceStateF16<true, 2>(
                state,
                actual_kv,
                actual_out,
                key,
                query,
                decay,
                kdim,
                vdim,
                vdim);
          else
            deltanetQhpiHvxReduceStateF16<false>(
                state,
                actual_kv,
                actual_out,
                key,
                query,
                decay,
                kdim,
                vdim,
                vdim);
          assert(!std::memcmp(expected_kv, actual_kv, vdim * sizeof(float)));
          assert(!std::memcmp(expected_out, actual_out, vdim * sizeof(float)));
          for (size_t k = 0; k < kdim; ++k)
            deltanetQhpiHvxUpdateStateF16(
                state + k * vdim,
                expected + k * vdim,
                delta,
                decay,
                float(key[k]),
                vdim);
          if (!misalignment && vdim % 128 == 0)
            deltanetQhpiHvxUpdateStateTileF16<true, 2>(
                state, actual, delta, key, decay, kdim, vdim, vdim);
          else
            deltanetQhpiHvxUpdateStateTileF16<false>(
                state, actual, delta, key, decay, kdim, vdim, vdim);
          assert(!std::memcmp(expected, actual, kdim * vdim * sizeof(Float16)));
        }
  std::fprintf(
      stderr,
      "FP16 projections match original FP32 scratch bit-for-bit; state updates match FP16 bytes\n");
#endif
  random.seed(88508750);
  constexpr size_t rows = 6, timesteps = 3, guard = 64;
  alignas(128) static Float16 initial[rows * 256 * 256 + 2 * guard];
  alignas(128) static Float16 state[rows * 256 * 256 + 2 * guard];
  alignas(128) static Float16 output[rows * timesteps * 256 + 2 * guard];
  alignas(128) Float16 q[rows * timesteps * 256], k[rows * timesteps * 256];
  alignas(128) Float16 value[rows * timesteps * 256];
  Float16 decay[rows * timesteps], beta[rows * timesteps];
  const auto sentinel = Float16(-37.125f);
  FILE* file = std::fopen(argv[1], "wb");
  assert(file);
  unsigned cases = 0;
  for (size_t kdim : {1u, 16u, 128u, 256u})
    for (size_t vdim : {1u, 63u, 64u, 65u, 127u, 128u, 129u, 192u, 193u, 256u})
      for (size_t misalignment : {0u, 1u})
        for (size_t tiles : {1u, 2u, 3u}) {
          const size_t state_size = rows * kdim * vdim;
          const size_t output_size = rows * timesteps * vdim;
          for (size_t i = 0; i < rows * timesteps * kdim; ++i) {
            q[i] = Float16(values(random));
            k[i] = Float16(values(random));
          }
          for (size_t i = 0; i < output_size; ++i)
            value[i] = Float16(values(random));
          for (size_t i = 0; i < rows * timesteps; ++i) {
            decay[i] = Float16(i % 3 == 0 ? 0.f : i % 3 == 1 ? .98f : 1.f);
            beta[i] = Float16(i % 3 == 0 ? 0.f : i % 3 == 1 ? .5f : 1.f);
          }
          std::fill(std::begin(state), std::end(state), sentinel);
          std::fill(std::begin(output), std::end(output), sentinel);
          auto* state_data = state + guard + misalignment;
          auto* output_data = output + guard + misalignment;
          auto* initial_data = initial + guard + (1 - misalignment);
          for (size_t i = 0; i < state_size; ++i)
            initial_data[i] = Float16(values(random));
          for (size_t slice = 0; slice < 3; ++slice)
            deltanetCoreQhpiRunFp16(
                output_data,
                state_data,
                q,
                k,
                value,
                decay,
                beta,
                initial_data,
                2,
                timesteps,
                3,
                kdim,
                vdim,
                slice,
                3,
                tiles);
          for (size_t i = 0; i < guard + misalignment; ++i) {
            assert(!std::memcmp(state + i, &sentinel, sizeof(Float16)));
            assert(!std::memcmp(output + i, &sentinel, sizeof(Float16)));
          }
          for (size_t i = 0; i < guard - misalignment; ++i) {
            assert(!std::memcmp(
                state_data + state_size + i, &sentinel, sizeof(Float16)));
            assert(!std::memcmp(
                output_data + output_size + i, &sentinel, sizeof(Float16)));
          }
          if (std::fwrite(output_data, sizeof(Float16), output_size, file) !=
                  output_size ||
              std::fwrite(state_data, sizeof(Float16), state_size, file) !=
                  state_size)
            return 1;
          ++cases;
        }
  if (std::fclose(file))
    return 1;
  std::fprintf(
      stderr,
      "FP16 recurrence %u cases: nonzero initial state, three tokens, aligned/unaligned, tails and V partitions\n",
      cases);
}
