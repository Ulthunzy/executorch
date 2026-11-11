// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cassert>
#include <iostream>
#include <random>
#include <thread>
#include <vector>
#include "../src/ops/attention/HmxQueue.h"
#include "../src/ops/attention/GroupedScratch.h"
#include "../src/ops/attention/Reference.h"

using attention::Configuration;
using attention::Geometry;
using attention::Inputs;
using attention::Quantization;

static float grids[] = {
    1.f / 4096,
    32768,
    0.125f,
    1.f / 8192,
    32768,
    1.f / 8192,
    32768,
    1.f / 65535,
    0,
    1.f / 4096,
    32768,
    1.f / 4096,
    32768};

template <typename Code = uint16_t>
static void run_case(
    Geometry g,
    unsigned capacity,
    unsigned valid,
    unsigned first,
    unsigned history) {
  float grids[13];
  std::copy_n(::grids, 13, grids);
  if (sizeof(Code) == 1) {
    grids[9] = grids[11] = 1.f / 16;
    grids[10] = grids[12] = 128;
  }
  const unsigned bytes =
      g.batch * g.kv_heads * g.dimension * capacity * sizeof(Code);
  const int32_t kd[] = {3, int32_t(bytes | attention::exact_append_flag), 7};
  const int32_t vd[] = {4, kd[1], 7};
  const int32_t lengths[] = {int32_t(valid), int32_t(first), int32_t(history)};
  Configuration c{g};
  assert(c.configure(kd, vd, lengths, grids, sizeof(Code)));
  std::mt19937 random(capacity + history + g.dimension);
  std::vector<uint16_t> q(g.batch * g.heads * g.queries * g.dimension),
      y(q.size()), again(q.size());
  std::vector<Code> k(bytes / sizeof(Code)), v(k.size()),
      nk(g.batch * g.kv_heads * g.queries * g.dimension), nv(nk.size());
  for (auto& x : q)
    x = uint16_t(32768 + int(random() % 4096) - 2048);
  for (auto* data : {&k, &v, &nk, &nv})
    for (auto& x : *data)
      x = sizeof(Code) == 1 ? Code(128 + int(random() % 16) - 8)
                            : Code(32768 + int(random() % 4096) - 2048);
  const Quantization quant{1.f / 4096, 32768},
      cache_quant{grids[9], int(grids[10])}, outq{1.f / 16384, 32768};
  const Inputs inputs{
      q.data(),
      k.data(),
      nk.data(),
      v.data(),
      nv.data(),
      grids,
      quant,
      cache_quant,
      cache_quant,
      outq};
  std::vector<float> work(g.dimension + 256);
  for (unsigned b = 0; b < g.batch; ++b)
    for (unsigned h = 0; h < g.heads; ++h)
      for (unsigned r = 0; r < g.queries; ++r) {
        attention::reference_row(c, inputs, b, h, r, work.data(), y.data());
        const auto at = ((b * g.heads + h) * g.queries + r) * g.dimension;
        if (r >= valid) {
          for (unsigned d = 0; d < g.dimension; ++d)
            assert(y[at + d] == outq.zero);
          continue;
        }
        const unsigned keys = history + r + 1,
                       kv = b * g.kv_heads + h / (g.heads / g.kv_heads);
        std::vector<double> scores(keys), expected(g.dimension, 0);
        double maximum = -1e300;
        for (unsigned col = 0; col < keys; ++col) {
          double score = 0;
          for (unsigned d = 0; d < g.dimension; ++d) {
            const auto code = col < history
                ? k[(kv * g.dimension + d) * capacity + first + col]
                : nk[(kv * g.dimension + d) * g.queries + col - history];
            score += double(quant.dequantize(q[at + d])) *
                cache_quant.dequantize(code);
          }
          // Independent dense score matrix, double softmax and PV accumulation.
          for (unsigned stage : {0u, 3u}) {
            const double scale = grids[stage], zero = grids[stage + 1];
            score =
                (std::clamp(std::floor(score / scale + zero + .5), 0., 65535.) -
                 zero) *
                scale;
            if (stage == 0)
              score *= grids[2];
          }
          score =
              (std::clamp(
                   std::floor(score / grids[5] + grids[6] + .5), 0., 65535.) -
               grids[6]) *
              grids[5];
          maximum = std::max(maximum, scores[col] = score);
        }
        double sum = 0;
        for (unsigned col = 0; col < keys; ++col) {
          const double p = std::exp(scores[col] - maximum);
          sum += p;
          for (unsigned d = 0; d < g.dimension; ++d) {
            const auto code = col < history
                ? v[(kv * capacity + first + col) * g.dimension + d]
                : nv[(kv * g.queries + col - history) * g.dimension + d];
            expected[d] += p * cache_quant.dequantize(code);
          }
        }
        for (unsigned d = 0; d < g.dimension; ++d)
          assert(
              std::abs(
                  int(y[at + d]) - int(outq.quantize(expected[d] / sum))) <= 2);
      }
  // Poison unused physical capacity and padded queries. The result must not
  // change.
  for (unsigned head = 0; head < g.batch * g.kv_heads; ++head) {
    for (unsigned col = 0; col < capacity; ++col)
      if (col < first || col >= first + history)
        for (unsigned d = 0; d < g.dimension; ++d) {
          k[(head * g.dimension + d) * capacity + col] =
              std::numeric_limits<Code>::max();
          v[(head * capacity + col) * g.dimension + d] = 0;
        }
    for (unsigned r = valid; r < g.queries; ++r)
      for (unsigned d = 0; d < g.dimension; ++d) {
        nk[(head * g.dimension + d) * g.queries + r] =
            std::numeric_limits<Code>::max();
        nv[(head * g.queries + r) * g.dimension + d] = 0;
      }
  }
  for (unsigned b = 0; b < g.batch; ++b)
    for (unsigned h = 0; h < g.heads; ++h)
      for (unsigned r = 0; r < g.queries; ++r)
        attention::reference_row(c, inputs, b, h, r, work.data(), again.data());
  assert(y == again);
}

static void test_queue(unsigned workers, bool fail) {
  attention_hmx::Queue queue{};
  for (unsigned round = 0; round < 3; ++round) {
    queue.workers = workers;
    std::vector<std::thread> threads;
    for (unsigned worker = 0; worker < workers; ++worker) {
      threads.emplace_back([&, worker] {
        uint8_t a[8]{1, 0, 2, 0, 3, 0, 4, 0}, b = 2, c[8]{}, bias = 0;
        for (unsigned job = 0; job < 5; ++job) {
          const unsigned batches = job % 2 ? 4 : 1;
          std::fill_n(c, 8, uint8_t(0));
          if (!attention_hmx::submit(
                  queue, worker, a, &b, c, &bias, 1, 1, 1, batches, 2, 2))
            break;
          for (unsigned batch = 0; batch < 4; ++batch) {
            assert(c[batch * 2] == (batch < batches ? batch + 3 : 0));
            assert(c[batch * 2 + 1] == 0);
          }
        }
        __atomic_store_n(&queue.jobs[worker].finished, 1u, __ATOMIC_RELEASE);
      });
    }
    const bool okay =
        attention_hmx::serve(queue, [&](const attention_hmx::Job& job) {
          if (fail)
            return false;
          for (unsigned batch = 0; batch < job.batches; ++batch)
            job.c[batch * job.c_stride] = job.a[batch * job.a_stride] + *job.b;
          return true;
        });
    for (auto& thread : threads)
      thread.join();
    assert(okay != fail && queue.workers == 0 && queue.error == 0);
  }
}

int main() {
  for (unsigned workers : {1u, 2u, 4u, 8u}) {
    test_queue(workers, false);
    test_queue(workers, true);
  }
  for (unsigned valid : {1u, 7u, 17u, 27u, 37u, 100u, 120u, 127u, 128u})
    run_case({1, 4, 128, 3, 2}, 512, valid, 5, 250);
  run_case({1, 16, 1, 256, 2}, 16384, 1, 0, 0);
  run_case({1, 8, 128, 256, 1}, 2048, 1, 0, 1023);
  run_case({1, 16, 37, 128, 2}, 4096, 17, 0, 1000);
  run_case({2, 6, 7, 63, 2}, 512, 5, 3, 259);
  run_case({1, 3, 3, 65, 3}, 512, 3, 0, 257);
  run_case({1, 4, 1, 32, 1}, 16384, 1, 0, 8191);
  run_case({1, 1, 1, 1, 1}, 1, 1, 0, 0);
  run_case({1, 1, 3, 8, 1}, 99, 3, 0, 99);
  run_case({1, 8, 128, 256, 2}, 8192, 37, 0, 0);
  run_case({1, 8, 128, 256, 2}, 8192, 37, 7, 193);
  run_case({1, 8, 1, 256, 2}, 8192, 1, 0, 8191);
  run_case<uint8_t>({1, 8, 128, 256, 2}, 8192, 37, 0, 0);
  run_case<uint8_t>({1, 8, 128, 256, 2}, 8192, 37, 7, 193);
  run_case<uint8_t>({1, 8, 128, 256, 2}, 8192, 128, 0, 129);
  run_case<uint8_t>({1, 8, 1, 256, 2}, 8192, 1, 0, 8191);
  run_case<uint8_t>({1, 16, 1, 256, 2}, 64, 1, 0, 3);
  run_case<uint8_t>({2, 6, 7, 63, 2}, 512, 5, 3, 259);
  {
    Configuration packed{{1, 8, 128, 256, 2}};
    int32_t key[] = {
        3,
        int32_t(
            8388608u | attention::exact_append_flag |
            attention_cache::packed_storage_flag),
        1};
    int32_t value[] = {4, int32_t(8388608u | attention::exact_append_flag), 1};
    int32_t lengths[] = {128, 0, 0};
    assert(packed.configure(key, value, lengths, grids));
    assert(packed.capacity == 8192);
    value[1] = key[1];
    assert(!packed.configure(key, value, lengths, grids));
    assert(attention_cache::packed_rows(8192) == 8448);
    assert(attention_cache::packed_rows(16384) == 16640);
    assert(attention_cache::packed_offset(17) == 4096);
  }
  assert((!Geometry{UINT32_MAX, UINT32_MAX, UINT32_MAX, 256, 1}.valid()));
  assert((!Geometry{1, 7, 128, 128, 2}.valid()));
  assert((Geometry{1, 8, 128, 256, 2}.fast()));
  assert((!Geometry{1, 4, 128, 256, 2}.fast()));
  Configuration c{{1, 16, 128, 256, 2}};
  int32_t k[] = {3, 1024 * 16384, 1}, v[] = {4, k[1], 1}, l[] = {128, 0, 16384};
  assert(c.configure(k, v, l, grids) && c.fast());
  l[1] = 1;
  assert(!c.configure(k, v, l, grids));
  l[1] = 0;
  l[0] = 129;
  assert(!c.configure(k, v, l, grids));
  l[0] = 128;
  v[1]--;
  assert(!c.configure(k, v, l, grids));
  v[1]++;
  v[2]++;
  assert(!c.configure(k, v, l, grids));
  v[2]--;
  grids[9] = 0;
  assert(!c.configure(k, v, l, grids));
  grids[9] = 1.f / 4096;
  std::cout
      << "Attention: 26 numerical/poisoned-capacity cases and invalid-parameter checks passed\n";
}
