/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/repetition_penalty.h>

#include <cassert>
#include <cmath>
#include <limits>

int main() {
  example::RepetitionPenalty disabled;
  disabled.begin_turn({0, 1}, true);
  std::vector<float> raw{12, 8, 10};
  disabled.apply(raw, 0.5f, 10);
  assert((raw == std::vector<float>{12, 8, 10}));

  example::RepetitionPenalty penalty(2.0f);
  penalty.begin_turn({0, 0, 1, 2, std::numeric_limits<uint64_t>::max()}, true);
  raw = {12, 8, 10, 16};
  penalty.apply(raw, 0.5f, 10);
  assert((raw == std::vector<float>{0.5f, -2.0f, 0.0f, 3.0f}));
  // Float logits use the identical sign-aware formula; repeated IDs are
  // penalized once, not once per occurrence.
  raw = {1, -1, 0, 3};
  penalty.apply(raw);
  assert((raw == std::vector<float>{0.5f, -2.0f, 0.0f, 3.0f}));

  // Merely scoring a speculative candidate cannot commit it to history.
  penalty.begin_turn({0}, true);
  raw = {1, 2};
  penalty.apply(raw);
  assert((raw == std::vector<float>{0.5f, 2.0f}));
  raw = {1, 2};
  penalty.apply(raw);
  assert((raw == std::vector<float>{0.5f, 2.0f}));
  penalty.record_token(1);
  raw = {1, 2};
  penalty.apply(raw);
  assert((raw == std::vector<float>{0.5f, 1.0f}));

  penalty.begin_turn({2}, false);
  raw = {1, 2, 3};
  penalty.apply(raw);
  assert((raw == std::vector<float>{0.5f, 1.0f, 1.5f}));
  penalty.begin_turn({2}, true);
  raw = {1, 2, 3};
  penalty.apply(raw);
  assert((raw == std::vector<float>{1.0f, 2.0f, 1.5f}));
}
