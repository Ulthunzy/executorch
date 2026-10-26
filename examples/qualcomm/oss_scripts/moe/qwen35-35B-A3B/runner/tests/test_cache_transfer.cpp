#include "../cache_transfer.h"
#include "../../native/prefill/expert_stream.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

static size_t
element(CacheLayout layout, size_t window, size_t channel, size_t token) {
  return layout == CacheLayout::Key
      ? channel * window + token
      : (channel / 256 * window + token) * 256 + channel % 256;
}

int main() {
  for (auto layout : {CacheLayout::Key, CacheLayout::Value}) {
    for (size_t valid : {1, 2, 63, 64, 65, 127, 128}) {
      constexpr size_t past = 400, prefix = 137;
      std::vector<uint16_t> history(512 * past), block(512 * 128),
          full(512 * (past + 128)), compact(512 * (prefix + valid)),
          encoding(65536);
      for (size_t i = 0; i < encoding.size(); ++i)
        encoding[i] = std::min(size_t(65535), i + 3);
      QuantizedCacheView state{history.data(), history.size(), 0.5f, -1000};
      clear_cache(state);
      for (size_t c = 0; c < 512; ++c) {
        for (size_t t = 0; t < prefix; ++t)
          history[element(layout, past, c, past - prefix + t)] = 2000 + c + t;
        for (size_t t = 0; t < 128; ++t)
          block[element(layout, 128, c, t)] = 4000 + c + t;
      }
      append_prefill_kv(
          state, block.data(), encoding.data(), full.data(), layout, valid);
      copy_cache_range(
          {full.data(), full.size(), 0.5f, -1000},
          {compact.data(), compact.size(), 0.5f, -1000},
          layout,
          past + 128 - prefix - valid,
          0,
          prefix + valid);
      for (size_t c = 0; c < 512; ++c) {
        for (size_t t = 0; t < prefix + valid; ++t) {
          const uint16_t expected =
              t < prefix ? 2000 + c + t : 4003 + c + t - prefix;
          assert(compact[element(layout, prefix + valid, c, t)] == expected);
          assert(
              history[element(layout, past, c, past - prefix - valid + t)] ==
              expected);
        }
        assert(history[element(layout, past, c, 0)] == 1000);
      }
      assert(
          expert_stream::token_count(
              expert_stream::mapping_control(3, 128, valid)) == 128);
      assert(
          expert_stream::valid_token_count(
              expert_stream::mapping_control(3, 128, valid)) == valid);
    }
  }
  for (int physical : {128, 256, 512, 1024})
    for (int valid = 1; valid <= physical; ++valid) {
      const int control = expert_stream::mapping_control(3, physical, valid);
      assert(expert_stream::token_count(control) == physical);
      assert(expert_stream::valid_token_count(control) == valid);
      assert((control & 255) == 3);
    }
  uint16_t source[] = {0, 9, 10, 11, 65535}, destination[5]{};
  copy_cache_range(
      {source, 5, 0.5f, -10},
      {destination, 5, 0.25f, -100},
      CacheLayout::State,
      0,
      0,
      5);
  const uint16_t expected[] = {80, 98, 100, 102, 65535};
  assert(std::equal(std::begin(expected), std::end(expected), destination));
  bool rejected = false;
  try {
    copy_cache_range(
        {source, 5, 1.f, 0},
        {destination, 5, 1.f, 0},
        CacheLayout::State,
        4,
        0,
        2);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  assert(rejected);
  std::cout
      << "padded KV, compact prefix, calibrated grids and descriptor tests passed\n";
}
