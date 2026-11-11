// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <thread>
#include <vector>
#include "../src/ops/attention/GroupedScratchU16.h"
#include "../src/ops/attention/HmxQueue.h"
#include "../src/ops/attention/PackedKv8.h"
namespace attention {
constexpr unsigned worker_start = 2048;
struct DmaDescriptor {
  DmaDescriptor* next;
  uint32_t control;
  const void* source;
  void* destination;
};
struct MockDma {
  std::vector<DmaDescriptor> commands;
  std::vector<DmaDescriptor*> addresses;
  const uint8_t* keys = nullptr;
  const uint8_t* values = nullptr;
  size_t bytes = 0;
  std::vector<uint8_t> key_reads, value_reads;
  unsigned launches = 0;
} mock;
void dma_start(DmaDescriptor* descriptor) {
  assert(mock.commands.empty());
  for (auto* p = descriptor; p; p = p->next) {
    mock.commands.push_back(*p);
    mock.addresses.push_back(p);
    assert(mock.commands.size() <= 32);
  }
  ++mock.launches;
}
void dma_start(
    uint8_t* destination,
    const uint8_t* source,
    unsigned bytes,
    DmaDescriptor* descriptor) {
  descriptor->next = nullptr;
  descriptor->source = source;
  descriptor->destination = destination;
  descriptor->control = bytes | (1u << 28) | (1u << 29);
  dma_start(descriptor);
}
bool dma_wait() {
  assert(!mock.commands.empty());
  for (unsigned i = 0; i < mock.commands.size(); ++i) {
    const auto& d = mock.commands[i];
    const auto& live = *mock.addresses[i];
    assert(
        d.source == live.source && d.destination == live.destination &&
        d.control == live.control && d.next == live.next);
    size_t count = d.control & 0xfffffffu;
    uintptr_t source = uintptr_t(d.source);
    auto* reads = &mock.key_reads;
    uintptr_t base = uintptr_t(mock.keys);
    if (source >= uintptr_t(mock.values) &&
        source < uintptr_t(mock.values) + mock.bytes) {
      reads = &mock.value_reads;
      base = uintptr_t(mock.values);
    }
    assert(source >= base && source + count <= base + mock.bytes);
    for (size_t byte = source - base; byte < source - base + count; ++byte)
      assert(++(*reads)[byte] == 1);
    std::memcpy(d.destination, d.source, count);
  }
  mock.commands.clear();
  mock.addresses.clear();
  return true;
}
#include "../src/ops/attention/StreamKv.h"
} // namespace attention
using namespace attention;
static void run_cache(unsigned columns, unsigned heads, unsigned readers) {
  const unsigned packed = (columns + 63) & ~63u,
                 kv_rows = (packed + 255) & ~255u;
  const size_t head_bytes = size_t(kv_rows) * 256, storage = heads * head_bytes;
  std::vector<uint8_t> allocation(storage * 4);
  const PackedKv8 packed_kv(allocation.data(), heads, kv_rows);
  auto* k = packed_kv.keys;
  auto* v = packed_kv.values;
  std::mt19937 rng(columns + readers);
  for (size_t i = 0; i < storage; ++i) {
    k[i] = rng();
    v[i] = rng();
  }
  mock = {};
  mock.keys = k;
  mock.values = v;
  mock.bytes = storage;
  mock.key_reads.resize(storage);
  mock.value_reads.resize(storage);
  std::vector<std::vector<uint8_t>> expected;
  std::vector<unsigned> visible_begin;
  for (unsigned head = 0; head < heads; ++head)
    for (unsigned begin = 0; begin < columns; begin += 1024) {
      unsigned width = std::min(1024u, columns - begin);
      for (unsigned at = begin; at < begin + width; at += 256) {
        unsigned count = std::min(256u, columns - at) * 256;
        visible_begin.push_back(at);
        expected.emplace_back(
            k + head * head_bytes + at * 256,
            k + head * head_bytes + at * 256 + count);
      }
      for (unsigned col = 0; col < 256; col += 64) {
        std::vector<uint8_t> panel(width * 64);
        for (unsigned half = 0; half < 2; ++half)
          for (unsigned offset = 0; offset < width; offset += 256) {
            unsigned at = begin + offset,
                     piece = std::min(256u, width - offset),
                     source_width = std::min(256u, packed - at / 256 * 256);
            size_t source = head * head_bytes + (at / 256) * 65536 +
                (col / 32 + half) * source_width * 32;
            std::copy_n(
                v + source,
                piece * 32,
                panel.begin() + half * width * 32 + offset * 32);
          }
        visible_begin.push_back(begin);
        expected.push_back(std::move(panel));
      }
    }
  std::unique_ptr<void, decltype(&std::free)> work(
      std::aligned_alloc(2048, worker_start + 8 * sizeof(GroupedScratchU16)),
      &std::free);
  assert(work);
  alignas(128) DmaDescriptor descriptors[64]{};
  const unsigned valid = std::min(columns, readers / 2 * 32),
                 history = columns - valid, query_tiles = (valid + 31) / 32;
  attention_stream::Cache cache(
      static_cast<uint8_t*>(work.get()),
      descriptors,
      packed_kv,
      columns,
      packed,
      heads,
      readers,
      history,
      valid);
  std::array<unsigned, 8> next{};
  next.fill(1);
  unsigned complete = 0, attempts = 0, deferrals = 0;
  bool finished[8]{};
  while (complete < readers) {
    assert(++attempts < 10000000);
    unsigned worker = attempts < 16 ? 0 : rng() % readers;
    if (finished[worker])
      continue;
    const unsigned end =
        (history + std::min(valid, (worker % query_tiles + 1) * 32) + 31) &
        ~31u;
    while (next[worker] <= expected.size() &&
           visible_begin[next[worker] - 1] >= end)
      ++next[worker];
    unsigned panel = next[worker];
    if (panel > expected.size()) {
      finished[worker] = true;
      ++complete;
      continue;
    }
    int status = int(cache.prepare(panel));
    assert(status);
    if (status == 2) {
      ++deferrals;
      continue;
    }
    auto* got =
        attention_stream::buffer(static_cast<uint8_t*>(work.get()), panel);
    assert(std::equal(
        expected[panel - 1].begin(), expected[panel - 1].end(), got));
    cache.consume(panel, worker);
    ++next[worker];
  }
  assert(cache.finish());
  assert(mock.commands.empty());
  const uint64_t unique = uint64_t(heads) * columns * 256;
  assert(cache.key_bytes == unique && cache.value_bytes == unique);
  assert(
      std::count(mock.key_reads.begin(), mock.key_reads.end(), 1) ==
      int64_t(unique));
  assert(
      std::count(mock.value_reads.begin(), mock.value_reads.end(), 1) ==
      int64_t(unique));
  assert(cache.loads == mock.launches);
  std::cout << "PASS columns=" << columns << " readers=" << readers
            << " heads=" << heads << " bytes=" << 2 * unique
            << " launches=" << cache.loads << " deferred=" << deferrals << "\n";
}
static void run_deferred_queue() {
  attention_hmx::Queue queue{};
  queue.workers = 8;
  std::thread workers[8];
  unsigned progress[8]{};
  for (unsigned w = 0; w < 8; ++w)
    workers[w] = std::thread([&, w] {
      for (unsigned step = 1; step <= 128; ++step)
        assert(attention_hmx::submit(
            queue, w, nullptr, nullptr, nullptr, nullptr, 32, step, 32));
      __atomic_store_n(&queue.jobs[w].finished, 1u, __ATOMIC_RELEASE);
    });
  unsigned deferred = 0;
  assert(attention_hmx::serve(queue, [&](const attention_hmx::Job& job) -> int {
    unsigned worker = &job - queue.jobs;
    if (worker < 7 && progress[worker + 1] < job.k) {
      ++deferred;
      return 2;
    }
    assert(progress[worker] + 1 == job.k);
    progress[worker] = job.k;
    return 1;
  }));
  for (auto& thread : workers)
    thread.join();
  for (auto x : progress)
    assert(x == 128);
  assert(deferred);
  std::cout << "PASS deferred queue resumes all eight readers\n";
}
int main() {
  for (unsigned columns :
       {32u, 64u, 96u, 128u, 288u, 640u, 1024u, 1056u, 4096u, 8192u, 16416u})
    for (unsigned readers : {2u, 4u, 6u, 8u})
      run_cache(columns, 2, readers);
  run_deferred_queue();
}
