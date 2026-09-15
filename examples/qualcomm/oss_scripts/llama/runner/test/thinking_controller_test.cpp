/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/thinking_controller.h>
#include <cassert>
#include <iostream>

using example::ThinkingController;
using Reason = ThinkingController::Reason;
constexpr uint64_t open = 10, close = 11, eos = 12, start = 13, assistant = 14;

static ThinkingController
make(bool repair, bool budget, const std::vector<uint64_t>& prompt = {}) {
  ThinkingController result;
  result.begin_turn(
      repair, budget, 2, open, close, start, eos, {start, assistant}, prompt);
  return result;
}

int main() {
  for (bool repair : {false, true}) {
    for (bool budget : {false, true}) {
      auto c = make(repair, budget);
      assert(c.select(open, false, 100) == open);
      assert(c.select(1, false, 99) == 1);
      assert(c.select(eos, true, 98) == (repair ? close : eos));
      if (repair) {
        assert(c.reason() == Reason::Eos);
        assert(c.select(eos, true, 97) == eos);
      }
      c = make(repair, budget, {start, assistant, open});
      assert(c.select(1, false, 100) == 1);
      assert(c.select(2, false, 99) == 2);
      assert(c.select(3, false, 98) == (budget ? close : 3));
      if (budget) {
        assert(c.reason() == Reason::Budget);
        assert(c.select(4, false, 97) == 4);
      }
    }
  }

  auto c = make(true, false, {start, assistant, open});
  assert(c.select(eos, true, 100) == close); // first prefill sample
  assert(c.select(1, false, 99) == 1);
  assert(c.select(eos, true, 98) == eos);

  c = make(true, true);
  assert(c.select(eos, true, 100) == eos); // no think was opened
  c = make(true, true, {start, assistant, open, close});
  assert(c.select(eos, true, 100) == eos); // explicitly non-thinking prompt
  c = make(true, true, {start, 99, open, start, assistant});
  assert(c.select(eos, true, 100) == eos); // user text is not reasoning
  c = make(true, true, {start, assistant, open, eos, start, 99, open});
  assert(c.select(eos, true, 100) == eos); // historical assistant excluded

  c = make(true, true, {start, assistant, open, 1, 2, 3, 4});
  assert(c.select(1, false, 100) == 1); // count generated tokens, not prefix
  assert(c.select(close, false, 99) == close);
  assert(c.reason() == Reason::None);
  assert(c.select(open, false, 98) == open); // literal tags in final prose
  assert(c.select(eos, true, 97) == eos);

  c = make(true, false, {start, assistant, open, open});
  assert(c.select(eos, true, 100) == close);
  assert(c.select(1, false, 99) == close); // drain nested closings into KV
  assert(c.select(2, false, 98) == 2);
  assert(c.select(eos, true, 97) == eos);

  c = make(false, true, {start, assistant, open});
  assert(c.select(1, false, 2) == close); // do not exceed total length
  assert(c.reason() == Reason::ContextLimit);
  assert(c.select(eos, true, 1) == eos);
  c = make(true, false, {start, assistant, open});
  assert(c.select(1, false, 2) == 1); // EOS-only must not impose a budget

  c = make(true, true, {start, assistant, open});
  assert(c.select(eos, true, 100) == close);
  c.begin_turn(
      false, false, 512, open, close, start, eos, {start, assistant}, {});
  assert(c.select(open, false, 100) == open);
  assert(c.select(eos, true, 99) == eos); // next turn can disable both
  std::cout << "Thinking controller: four combinations and boundaries passed\n";
}
