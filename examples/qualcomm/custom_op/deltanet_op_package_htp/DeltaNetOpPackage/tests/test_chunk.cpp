// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstdio>
#include <random>
#include <vector>
#include "../src/kernels/Chunk.h"

int main() {
  std::mt19937 random(47);
  unsigned checks = 0;
  for (unsigned trial = 0; trial < 64; ++trial) {
    std::vector<uint16_t> source(128 * 128), tiled(source.size()),
        restored(source.size()), weighted(source.size()), weights(128);
    for (unsigned i = 0; i < source.size(); ++i)
      source[i] = uint16_t(random());
    hmx_tile::pack<true>(source.data(), tiled.data(), 128, 128);
    for (unsigned r = 0; r < 128; ++r)
      for (unsigned c = 0; c < 128; ++c) {
        const unsigned tile = gdn_chunk::tile_index(r, c, 128);
        const unsigned at =
            tile * 1024 + (r % 32 / 2) * 64 + (c % 32) * 2 + r % 2;
        if (tiled[at] != source[r * 128 + c])
          return std::puts("FAIL HMX correction tile order"), 1;
        ++checks;
      }
    for (unsigned r = 0; r < 128; ++r)
      for (unsigned c = 0; c < 128; ++c)
        restored[r * 128 + c] = tiled
            [gdn_chunk::tile_index(r, c, 128) * 1024 + (r % 32 / 2) * 64 +
             (c % 32) * 2 + r % 2];
    if (source != restored)
      return std::puts("FAIL correction column unpack"), 2;
    for (unsigned i = 0; i < source.size(); ++i)
      source[i] = Float16((int(random() % 65536) - 32768) / 8192.f).raw();
    for (unsigned i = 0; i < 128; ++i)
      weights[i] = Float16((random() % 65536) / 65536.f).raw();
    hmx_tile::pack<true>(source.data(), tiled.data(), 128, 128);
    gdn_chunk::weight_columns(
        tiled.data(), weights.data(), weighted.data(), 128, 128);
    for (unsigned r = 0; r < 128; ++r)
      for (unsigned c = 0; c < 128; ++c)
        restored[r * 128 + c] = weighted
            [gdn_chunk::tile_index(r, c, 128) * 1024 + (r % 32 / 2) * 64 +
             (c % 32) * 2 + r % 2];
    for (unsigned r = 0; r < 128; ++r)
      for (unsigned c = 0; c < 128; ++c) {
        const Float16 expected(
            float(Float16::from_raw(source[r * 128 + c])) *
            float(Float16::from_raw(weights[r])));
        if (restored[r * 128 + c] != expected.raw())
          return std::puts("FAIL weighted correction rounding"), 3;
        ++checks;
      }
    for (unsigned rows : {64u, 128u}) {
      std::vector<uint16_t> prior(rows * 128), product(rows * 128),
          packed(rows * 128), output(rows * 128);
      for (unsigned i = 0; i < prior.size(); ++i) {
        prior[i] = Float16((int(random() % 65536) - 32768) / 32768.f).raw();
        product[i] = Float16((int(random() % 65536) - 32768) / 65536.f).raw();
      }
      hmx_tile::pack<false>(product.data(), packed.data(), rows, 128);
      const Float16 decay(0.9375f);
      gdn_chunk::finish<true>(
          packed.data(), prior.data(), decay.raw(), output.data(), rows, 128);
      for (unsigned i = 0; i < prior.size(); ++i) {
        const Float16 scaled(float(Float16::from_raw(prior[i])) * float(decay));
        const Float16 expected(
            float(scaled) + float(Float16::from_raw(product[i])));
        if (output[i] != expected.raw())
          return std::puts("FAIL state epilogue"), 4;
        ++checks;
      }
      gdn_chunk::finish<false>(
          packed.data(), prior.data(), 0, output.data(), rows, 128);
      for (unsigned i = 0; i < prior.size(); ++i) {
        const Float16 expected(
            float(Float16::from_raw(prior[i])) +
            float(Float16::from_raw(product[i])));
        if (output[i] != expected.raw())
          return std::puts("FAIL output epilogue"), 5;
        ++checks;
      }
    }
  }
  std::printf("PASS %u correction layout/weight/epilogue lanes\n", checks);
}
