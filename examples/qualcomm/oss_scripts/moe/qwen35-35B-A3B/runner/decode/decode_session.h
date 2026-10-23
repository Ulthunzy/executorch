#pragma once

#include "../runtime_utils.h"
#include "../embedding.h"
#include "expert_readers.h"
#include "native_core.h"
#include <array>
#include <chrono>
#include <cmath>
#include <set>

#include "bank.h"
#include "attention.h"
#include "shared_io.h"
#include "expert_io.h"
#include "prefetch_queue.h"

struct Metrics {
  struct Io {
    int hot = 0, cold = 0, read_calls = 0;
    uint64_t bytes = 0;
    double wait = 0, sync = 0, pread_worker = 0, copy_worker = 0,
           feedback_overlap = 0, hit_native = 0, hit_overlap = 0, tail_wait = 0;
    int hit_mask = 0, split = 0;
  };

  double embedding = 0, attention = 0, router = 0, moe = 0, head = 0, read = 0,
         native = 0, feedback = 0;
  uint64_t direct = 0, logical = 0, hits = 0, requests = 0;
  int calls[5]{};
  std::array<std::array<int, 8>, 40> routes{};
  std::array<Io, 40> io{};
  std::array<std::array<double, 3>, 40> execution{};
  uint64_t prediction_bytes = 0, prediction_reads = 0, prediction_used = 0;
  double prediction_wait = 0, prediction_union = 0, prediction_overlap = 0;
};

// 一个 token 严格经过 40 层后才执行 head；context、state 和每层 LRU 跨 token
// 常驻。
struct DecodeSession {
  std::string root;
  Runtime runtime;
  NativeCore core;
  EmbeddingTable embedding;
  std::vector<std::unique_ptr<Bank>> banks;
  std::vector<std::shared_ptr<Buffer>> bank_slots;
  std::unique_ptr<ExpertCache> cache;
  std::unique_ptr<ExpertReaders> readers;
  std::shared_ptr<Buffer> ready_buffer;
  std::vector<std::unique_ptr<Attention>> attention;
  std::vector<std::unique_ptr<Graph>> router;
  std::unique_ptr<Graph> head;
  std::unique_ptr<SharedIo> shared_io;
  std::vector<uint8_t> cos, sin;
  int context_capacity = getenv("DECODE_CONTEXT_CAPACITY")
      ? atoi(getenv("DECODE_CONTEXT_CAPACITY"))
      : 1152;
  int prefix_tokens =
      getenv("PREFILL_TOKENS") ? atoi(getenv("PREFILL_TOKENS")) : 128;
  int generations[2]{}, position = 128, next_layer = 0;
  bool in_step = false;
  Metrics metrics;
  bool async_io = getenv("ASYNC_EXPERT_IO") &&
      std::string(getenv("ASYNC_EXPERT_IO")) == "1";
  bool ready_queue = getenv("EXPERT_READY_IO") &&
      std::string(getenv("EXPERT_READY_IO")) == "1";
  bool hit_first = !getenv("HIT_FIRST_EXPERTS") ||
      std::string(getenv("HIT_FIRST_EXPERTS")) == "1";
  bool overlap_feedback = getenv("OVERLAP_STATE_FEEDBACK") &&
      std::string(getenv("OVERLAP_STATE_FEEDBACK")) == "1";
  std::unique_ptr<PrefetchQueue> prefetch;
  std::vector<float> prediction_profile;
  std::array<std::array<int, 8>, 40> previous_routes{};
  std::array<std::vector<std::shared_ptr<PrefetchTask>>, 40> pending;
  std::vector<std::pair<Clock::time_point, Clock::time_point>>
      execution_windows;

  explicit DecodeSession(const std::string& path, size_t cache_mib);

  void reset();

  void warm_from_prefill();

  void embed(const int64_t* token, int64_t requested_position, float* out);

  void attend(int layer, const float* x, float* norm, float* residual);

  void
  route(int layer, const float* x, int32_t* ids, float* scores, float* shared);

  void predict(int target);

  void finish_predictions(int layer, const int32_t* actual);

  void experts(
      int layer,
      const float* x,
      const int32_t* ids,
      const float* scores,
      const float* shared,
      const float* residual,
      float* out);

  void project(const float* x, float* logits);

  ~DecodeSession();
};
