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
#include <numeric>
#include <random>
#include <unordered_set>

static void test_configurable_window(size_t window) {
  example::RepetitionPenalty penalty(2.f, window);
  example::RepetitionPenalty disabled(1.f, window);
  std::vector<uint64_t> history;
  std::vector<float> raw;
  std::mt19937 rng(42);
  for (size_t step = 0; step < 1024; ++step) {
    if (step % 521 == 0) {
      penalty.begin_turn({}, true);
      history.clear();
    } else if (step % 71 == 0) {
      penalty.begin_turn({}, false);
    }
    const uint64_t token = step < window + 1 ? step : rng() % (window + 64);
    if (step % 2) {
      penalty.begin_turn({token}, false);
    } else {
      penalty.record_token(token);
    }
    disabled.record_token(token);
    history.push_back(token);
    const auto start =
        history.size() > window ? history.end() - window : history.begin();
    const std::unordered_set<uint64_t> expected(start, history.end());
    raw.assign(window + 64, 2.f);
    disabled.apply(raw);
    assert(raw == std::vector<float>(window + 64, 2.f));
    penalty.apply(raw);
    for (size_t i = 0; i < raw.size(); ++i) {
      assert(raw[i] == (expected.count(i) ? 1.f : 2.f));
    }
  }
}

int main() {
  for (size_t window : {1, 64, 128, 256, 512}) {
    test_configurable_window(window);
  }
  bool rejected = false;
  try {
    example::RepetitionPenalty invalid(1.f, 0);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);
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

  constexpr size_t window = 256;
  std::vector<uint64_t> prompt(window);
  std::iota(prompt.begin(), prompt.end(), 0);
  penalty.begin_turn(prompt, true);
  raw.assign(window + 1, 2.f);
  penalty.apply(raw);
  for (size_t i = 0; i < window; ++i) {
    assert(raw[i] == 1.f);
  }
  assert(raw[window] == 2.f);
  penalty.record_token(window);
  raw.assign(window + 1, 2.f);
  penalty.apply(raw);
  assert(raw[0] == 2.f);
  for (size_t i = 1; i <= window; ++i) {
    assert(raw[i] == 1.f);
  }

  // The window counts occurrences, not distinct IDs; evict only the last copy.
  prompt.assign(window, 1);
  prompt.front() = prompt.back() = 0;
  penalty.begin_turn(prompt, true);
  penalty.record_token(2);
  raw = {2, 2, 2};
  penalty.apply(raw);
  assert((raw == std::vector<float>{1, 1, 1}));
  penalty.begin_turn(std::vector<uint64_t>(window - 2, 2), false);
  raw = {2, 2, 2};
  penalty.apply(raw);
  assert((raw == std::vector<float>{1, 2, 1}));
  penalty.record_token(2);
  raw = {2, 2, 2};
  penalty.apply(raw);
  assert((raw == std::vector<float>{2, 2, 1}));

  prompt.resize(400);
  std::iota(prompt.begin(), prompt.end(), 0);
  penalty.begin_turn(prompt, true);
  raw.assign(400, 2.f);
  penalty.apply(raw);
  for (size_t i = 0; i < 400; ++i) {
    assert(raw[i] == (i < 400 - window ? 2.f : 1.f));
  }
  penalty.begin_turn({}, false);
  penalty.record_token(400);
  raw.assign(401, 2.f);
  penalty.apply(raw);
  assert(
      raw[400 - window] == 2.f && raw[401 - window] == 1.f && raw[400] == 1.f);
  penalty.begin_turn({}, true);
  raw.assign(401, 2.f);
  penalty.apply(raw);
  assert(raw == std::vector<float>(401, 2.f));

  std::mt19937 rng(42);
  std::vector<uint64_t> history;
  for (size_t step = 0; step < 4096; ++step) {
    if (step % 521 == 0) {
      penalty.begin_turn({}, true);
      history.clear();
    }
    uint64_t token = rng() % (window + 64);
    penalty.record_token(token);
    history.push_back(token);
    auto start =
        history.size() > window ? history.end() - window : history.begin();
    std::unordered_set<uint64_t> expected(start, history.end());
    raw.assign(window + 64, 0);
    for (size_t i = 0; i < raw.size(); ++i) {
      raw[i] = i % 2 ? 8.f : 12.f;
    }
    penalty.apply(raw, 0.5f, 10);
    for (size_t i = 0; i < raw.size(); ++i) {
      float value = i % 2 ? -1.f : 1.f;
      if (expected.count(i)) {
        value = value < 0 ? value * 2.f : value / 2.f;
      }
      assert(raw[i] == value);
    }
  }
}
