// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <cstdint>

namespace attention_hmx {

constexpr unsigned kMaxWorkers = 8;
enum class Status { Failed, Done, Deferred };

struct alignas(128) Job {
  uint32_t pending;
  uint32_t finished;
  uint8_t* a;
  uint8_t* b;
  uint8_t* c;
  uint8_t* bias;
  unsigned m;
  unsigned k;
  unsigned n;
  unsigned batches;
  unsigned a_stride;
  unsigned c_stride;
  uint8_t* high;
  unsigned bias_stride;
  unsigned b_stride;
  // A nonzero ID selects the shared prefill KV stream.
  unsigned panel;
};

struct alignas(128) Queue {
  uint32_t workers;
  uint32_t error;
  Job jobs[kMaxWorkers];
};

inline void pause() {
#ifdef __hexagon__
  asm volatile("pause(#32)" ::: "memory");
#endif
}

// The caller initializes the queue to zero. The server resets it only after
// every worker has finished accessing its job.
inline bool submit(
    Queue& queue,
    unsigned worker,
    uint8_t* a,
    uint8_t* b,
    uint8_t* c,
    uint8_t* bias,
    unsigned m,
    unsigned k,
    unsigned n,
    unsigned batches = 1,
    unsigned a_stride = 0,
    unsigned c_stride = 0,
    uint8_t* high = nullptr,
    unsigned bias_stride = 0,
    unsigned b_stride = 0,
    unsigned panel = 0) {
  if (__atomic_load_n(&queue.error, __ATOMIC_ACQUIRE))
    return false;
  Job& job = queue.jobs[worker];
  job.a = a;
  job.b = b;
  job.c = c;
  job.bias = bias;
  job.m = m;
  job.k = k;
  job.n = n;
  job.batches = batches;
  job.a_stride = a_stride;
  job.c_stride = c_stride;
  job.high = high;
  job.bias_stride = bias_stride;
  job.b_stride = b_stride ? b_stride : k;
  job.panel = panel;
  __atomic_store_n(&job.pending, 1u, __ATOMIC_RELEASE);
  while (__atomic_load_n(&job.pending, __ATOMIC_ACQUIRE))
    pause();
  return __atomic_load_n(&queue.error, __ATOMIC_ACQUIRE) == 0;
}

template <typename Multiply>
inline bool serve(Queue& queue, Multiply multiply) {
  unsigned workers;
  while (!(workers = __atomic_load_n(&queue.workers, __ATOMIC_ACQUIRE)))
    pause();
  if (workers > kMaxWorkers)
    return false;
  for (;;) {
    unsigned finished = 0;
    bool worked = false;
    for (unsigned worker = 0; worker < workers; ++worker) {
      Job& job = queue.jobs[worker];
      if (__atomic_load_n(&job.pending, __ATOMIC_ACQUIRE)) {
        if (!__atomic_load_n(&queue.error, __ATOMIC_ACQUIRE)) {
          const int status = int(multiply(job));
          // Keep this request pending while other workers make progress.
          if (status == int(Status::Deferred))
            continue;
          if (!status)
            __atomic_store_n(&queue.error, 1u, __ATOMIC_RELEASE);
        }
        __atomic_store_n(&job.pending, 0u, __ATOMIC_RELEASE);
        worked = true;
      }
      finished += __atomic_load_n(&job.finished, __ATOMIC_ACQUIRE) != 0;
    }
    if (finished == workers)
      break;
    if (!worked)
      pause();
  }
  const bool success = __atomic_load_n(&queue.error, __ATOMIC_ACQUIRE) == 0;
  for (unsigned worker = 0; worker < workers; ++worker) {
    queue.jobs[worker].pending = 0;
    queue.jobs[worker].finished = 0;
  }
  queue.error = 0;
  __atomic_store_n(&queue.workers, 0u, __ATOMIC_RELEASE);
  return success;
}

} // namespace attention_hmx
