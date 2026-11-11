// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <random>
#include <thread>
#include "../src/ops/attention/ExpLut.h"
#include "../src/ops/attention/Fixed32.h"
#include "../src/ops/attention/HmxQueue.h"
#include "../src/ops/attention/SharedKv.h"

namespace fixed {
alignas(2048) uint16_t query[32 * 256], qpack[32 * 256], codes[256];
alignas(128) int32_t input[32], output[32];
alignas(128) uint8_t bytes[512];
int run() {
  std::mt19937 rng(47);
  unsigned checked = 0;
  for (auto& x : query)
    x = rng();
  attention_fixed::pack_query(query, reinterpret_cast<uint8_t*>(qpack));
  for (unsigned r = 0; r < 32; ++r)
    for (unsigned c = 0; c < 256; ++c) {
      unsigned at = (c / 32) * 1024 + (r / 2) * 64 + (c % 32) * 2 + r % 2;
      if (qpack[at] != query[r * 256 + c])
        return 1;
      ++checked;
    }
  for (unsigned i = 0; i < 512; ++i)
    bytes[i] = rng();
  for (unsigned offset : {0u, 24u, 64u, 100u, 128u, 192u, 256u, 384u, 448u}) {
    *reinterpret_cast<HVX_Vector*>(codes) =
        attention_fixed::load_codes64(bytes, offset);
    for (unsigned i = 0; i < 64; ++i) {
      if (codes[i] != bytes[offset + i]) {
        printf(
            "FAIL LOAD offset=%u i=%u got=%u expected=%u\n",
            offset,
            i,
            codes[i],
            bytes[offset + i]);
        return 2;
      }
      ++checked;
    }
  }
  for (double ratio : {.006221, .12065, .000001, 1., 1.27, 324.5}) {
    auto scale = attention_fixed::Scale::ratio(
        uint64_t(std::llround(ratio * (1ULL << 40))), 1ULL << 40);
    for (int iteration = 0; iteration < 100; ++iteration) {
      for (auto& x : input)
        x = int(rng() % 2000000) - 1000000;
      *reinterpret_cast<HVX_Vector*>(output) =
          scale.apply(*reinterpret_cast<HVX_Vector*>(input));
      for (unsigned i = 0; i < 32; ++i) {
        double exact = input[i] * ratio,
               allowed = std::abs(exact) * .00004 + 2. +
            (scale.shift < 0 ? std::ldexp(.5, -scale.shift) : 0.);
        if (std::abs(output[i] - exact) > allowed) {
          printf(
              "FAIL SCALE ratio=%g in=%d out=%d exact=%g\n",
              ratio,
              input[i],
              output[i],
              exact);
          return 3;
        }
        ++checked;
      }
    }
  }
  for (double ratio : {1e-30, 1e-9, .00001, .5, 1.}) {
    for (unsigned i = 0; i < 32; ++i)
      input[i] = i & 1 ? INT32_MIN + int(i) : INT32_MAX - int(i);
    const auto scale = attention_fixed::Scale::from_double(ratio);
    *reinterpret_cast<HVX_Vector*>(output) =
        scale.apply(*reinterpret_cast<HVX_Vector*>(input));
    for (unsigned i = 0; i < 32; ++i) {
      const double expected = input[i] * ratio;
      if (std::abs(output[i] - expected) > std::abs(expected) * .00004 + 2)
        return 4;
      ++checked;
    }
  }
  printf(
      "PASS %u HVX pack/load and integer fixed-point scale checks\n", checked);
  return 0;
}

} // namespace fixed

namespace exponential {
alignas(128) int32_t input[32];
alignas(128) uint32_t output[32];
int run() {
  unsigned checked = 0, global_error = 0;
  for (float invalid : {0.f, -1.f, 1e-30f, 1e30f})
    if (attention_exp::Parameters::make(invalid).coefficient)
      return 3;
  for (float step :
       {.00001f,
        .00003f,
        .0001f,
        .0003f,
        .001f,
        .003f,
        .004326127003878355f,
        .01f,
        .02f,
        .1f}) {
    auto p = attention_exp::Parameters::make(step);
    unsigned maxerr = 0, worst = 0;
    uint32_t previous = 65536;
    for (unsigned begin = 0; begin < 65536; begin += 32) {
      for (unsigned i = 0; i < 32; ++i)
        input[i] = begin + i;
      *reinterpret_cast<HVX_Vector*>(output) = attention_exp::codes(
          *reinterpret_cast<HVX_Vector*>(input),
          p.coefficient,
          *reinterpret_cast<const HVX_Vector*>(attention_exp::powers),
          *reinterpret_cast<const HVX_Vector*>(attention_exp::slopes),
          p.bits);
      for (unsigned i = 0; i < 32; ++i) {
        unsigned expected = unsigned(
            std::llround(65535 * std::exp(-double(step) * (begin + i))));
        unsigned error = unsigned(std::abs(int(output[i]) - int(expected)));
        if (error > maxerr) {
          maxerr = error;
          worst = begin + i;
        }
        if (output[i] > previous) {
          printf(
              "FAIL MONOTONIC step=%g delta=%u current=%u previous=%u\n",
              step,
              begin + i,
              output[i],
              previous);
          return 1;
        }
        previous = output[i];
        if (output[i] > 65535)
          return 2;
        ++checked;
      }
    }
    printf(
        "EXP step=%g coefficient=%u bits=%u max_error=%u worst_delta=%u\n",
        step,
        p.coefficient,
        p.bits,
        maxerr,
        worst);
    global_error = std::max(global_error, maxerr);
  }
  printf(
      "PASS %u exp inputs, max absolute U16-code error %u\n",
      checked,
      global_error);
  return global_error > 8;
}

} // namespace exponential

namespace shared {
int run() {
  attention_shared::Control state{};
  attention_hmx::Queue queue{};
  unsigned payload[2]{};
  std::thread workers[2];
  for (unsigned rank = 0; rank < 2; ++rank)
    workers[rank] = std::thread([&, rank] {
      for (unsigned tag = 1; tag <= 20000; ++tag) {
        unsigned slot = (tag - 1) & 1;
        if (!rank) {
          if (!attention_shared::available(
                  state, queue, slot, tag > 2 ? tag - 2 : 0))
            std::abort();
          payload[slot] = tag;
          __atomic_store_n(&state.ready[slot], tag, __ATOMIC_RELEASE);
        } else {
          while (__atomic_load_n(&state.ready[slot], __ATOMIC_ACQUIRE) != tag)
            std::this_thread::yield();
        }
        if (payload[slot] != tag)
          std::abort();
        __atomic_store_n(&state.consumed[rank][slot], tag, __ATOMIC_RELEASE);
        if ((tag + rank) % 11 == 0)
          std::this_thread::yield();
      }
    });
  for (auto& worker : workers)
    worker.join();
  printf("PASS 20000 asynchronous shared-KV stages per worker\n");
  return 0;
}

} // namespace shared

int main() {
  return fixed::run() || exponential::run() || shared::run();
}
