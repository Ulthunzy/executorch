#pragma once

#include "bank.h"

#include "../runtime_utils.h"
#include "expert_readers.h"
#include "native_core.h"
#include "expert_cache.h"
#include "../handoff.h"
#include "../resident_handoff.h"
#include "../embedding.h"
#include "../session_state.h"
#include <array>
#include <chrono>
#include <cstdlib>
#include <set>
#include <functional>

#include "attention.h"
#include "shared_io.h"
#include "vision_encoder.h"

// 每层保留固定容量的专家槽，缺失的 payload 通过 O_DIRECT 按需读取。

struct Metrics {
  double embedding = 0, attention = 0, router = 0, moe = 0, head = 0,
         preload = 0, read = 0, native = 0, feedback = 0;
  uint64_t direct = 0, preload_direct = 0, hits = 0, requests = 0,
           seed_hits = 0;
  int calls[5]{}, native_calls = 0, assignments = 0;
  double io_wall = 0, io_overlap = 0, io_wait = 0, staged_copy = 0;
  uint64_t layer_reuse_bytes = 0;
  double layer_seed_copy = 0;
  double batch_setup = 0, input_copy_sync = 0;
  double pread_worker = 0, payload_copy_worker = 0;
  double ready_native = 0, missing_native = 0;
  int ready_calls = 0, missing_calls = 0;
  double seed_wait = 0, seed_overlap = 0, retained_wait = 0,
         retained_overlap = 0, retained_prepare = 0, input_io_overlap = 0;
  double router_io_overlap = 0;
  double attention_io_overlap = 0;
  double gdn_attention = 0, full_attention = 0;
};

// 完整 40 层 AR128 调度；多块计划逐层处理整段输入，复用当前层的临时权重。
// Runtime 声明在资源对象之前，使其最后析构，覆盖所有 QNN/RPC 资源的生命周期。
struct PrefillSession {
  std::string root;
  Runtime runtime;
  NativeCore core;
  EmbeddingTable embedding;
  std::vector<std::unique_ptr<Bank>> banks;
  std::vector<std::shared_ptr<Buffer>> expert_slots;
  std::vector<ExpertCache> caches;
  std::vector<ExpertRead> startup_reads;
  std::unique_ptr<ExpertReaders> readers;
  std::vector<std::unique_ptr<Attention>> attention;
  std::vector<std::unique_ptr<Graph>> router;
  std::unique_ptr<Graph> head;
  std::unique_ptr<VisionEncoder> vision;
  std::unique_ptr<SharedIo> shared_io;
  std::vector<uint8_t> cos, sin;
  std::vector<float> image_features;
  size_t image_cursor = 0;
  int generations[4]{}, active_group = -1, next_layer = 0;
  size_t stride = 0;
  bool reuse =
      getenv("REUSE_EXPERTS") && std::string(getenv("REUSE_EXPERTS")) == "1";
  int capacity = 0, pool_count = 0, chunk_start = 0, prefill_window = 0,
      logical_capacity = 0;
  Metrics metrics;
  std::array<std::array<InheritedExperts::Entry, 64>, 40> layer_history{};
  std::array<uint64_t, 40> layer_clocks{};
  std::array<uint64_t, 40> stamp_offsets{};
  std::array<std::array<int32_t, 128 * 8>, 40> route_history{};
  bool async_io = getenv("ASYNC_EXPERT_IO") &&
      std::string(getenv("ASYNC_EXPERT_IO")) == "1";
  std::vector<std::shared_ptr<Buffer>> pipeline_banks;
  std::string startup_policy = getenv("STARTUP_EXPERT_PREFETCH")
      ? getenv("STARTUP_EXPERT_PREFETCH")
      : "none";
  bool startup_seed_pending = false;
  int sequence_tokens = 128, valid_tokens = 128, sequence_valid_tokens = 128,
      memo_layer = -1;
  std::shared_ptr<Buffer> layer_memo;
  std::array<bool, 256> memo_valid{};
  std::array<int, 64> memo_slot_ids{};
  enum class MemoCopy { None, Seed, Retain };
  MemoCopy memo_copy = MemoCopy::None;
  int memo_copy_layer = -1;
  std::vector<ExpertRead> memo_copies;
  std::vector<std::pair<Clock::time_point, Clock::time_point>>
      copy_compute_spans;
  std::vector<ExpertRead> route_reads;
  std::vector<int> route_read_ids;
  std::array<int, 256> route_first_use{};
  std::vector<std::pair<Clock::time_point, Clock::time_point>>
      stream_compute_spans, stream_input_spans;
  std::vector<std::pair<Clock::time_point, Clock::time_point>>
      attention_io_spans;
  int staged_router_layer = -1;
  std::vector<int32_t> staged_ids;
  std::vector<float> staged_scores, staged_shared;
  std::vector<std::pair<Clock::time_point, Clock::time_point>> route_io_spans;
  std::vector<int32_t> sequence_routes, sequence_seeds;
  std::vector<uint64_t> sequence_stamps;

  explicit PrefillSession(const std::string& path, int plan_tokens = 128);

  void prepare_layer_memo(int layer);

  void finish_memo_copy();

  void prefetch_route(int layer, const int32_t* ids);

  void finish_prefill_native();

  void prepare_decode();

  void append(
      const int64_t* tokens,
      size_t count,
      const float* features,
      size_t feature_count,
      const float* cos,
      const float* sin,
      size_t table_count,
      Qwen35SessionState& state,
      float* logits,
      const std::function<bool()>& stopped);

  void reset();

  void embed(const int64_t* tokens, float* out);

  void attend(int layer, const float* x, float* norm, float* residual);

  void
  route(int layer, const float* x, int32_t* ids, float* scores, float* shared);

  void execute_route(
      int layer,
      const float* x,
      int32_t* ids,
      float* scores,
      float* shared);

  uint64_t fill(
      int layer,
      const std::vector<int>& ids,
      const std::vector<ExpertCache::Binding>& bindings,
      bool seeded = false,
      const std::vector<ExpertRead>* prepared = nullptr);

  void experts(
      int layer,
      const float* x,
      const int32_t* ids,
      const float* scores,
      const float* shared,
      const float* residual,
      float* out,
      int tokens = 128);

  void project(const float* x, float* logits);

  ~PrefillSession();
};
