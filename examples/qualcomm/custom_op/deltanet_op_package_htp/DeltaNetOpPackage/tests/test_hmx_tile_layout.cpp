// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cstdio>
#include <random>
#include <vector>
#include "../src/utils/HmxTileLayout.h"

int main() {
  std::mt19937 random(47);
  unsigned checks = 0;
  for (unsigned rows : {64u, 128u})
    for (unsigned columns : {64u, 128u}) {
      std::vector<uint16_t> input(rows * columns), packed(rows * columns),
          restored(rows * columns);
      for (unsigned trial = 0; trial < 32; ++trial) {
        for (unsigned i = 0; i < input.size(); ++i)
          input[i] = trial == 0 ? i : uint16_t(random());
        hmx_tile::pack<false>(input.data(), packed.data(), rows, columns);
        for (unsigned r = 0; r < rows; ++r)
          for (unsigned c = 0; c < columns; ++c) {
            const unsigned at = ((r / 32) * (columns / 32) + c / 32) * 1024 +
                ((r % 32) / 2) * 64 + 2 * (c % 32) + r % 2;
            if (packed[at] != input[r * columns + c])
              return 1;
            ++checks;
          }
        hmx_tile::unpack(packed.data(), restored.data(), rows, columns);
        if (input != restored)
          return 2;
        hmx_tile::pack<true>(input.data(), packed.data(), rows, columns);
        for (unsigned r = 0; r < rows; ++r)
          for (unsigned c = 0; c < columns; ++c) {
            const unsigned at = ((c / 32) * (rows / 32) + r / 32) * 1024 +
                ((r % 32) / 2) * 64 + 2 * (c % 32) + r % 2;
            if (packed[at] != input[r * columns + c])
              return 3;
            ++checks;
          }
      }
    }
  std::printf(
      "PASS %u raw FP16 layout lanes and row-layout roundtrips\n", checks);
  return 0;
}
