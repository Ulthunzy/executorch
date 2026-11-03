#include "prefill_session.h"
#include <exception>
#include <fstream>
#include "../../../../custom_op/lowbit/expert_stream.h"
#include "cold_start.h"

PrefillSession::PrefillSession(const std::string& path, int plan_tokens)
    : root(path),
      core(runtime, path, plan_tokens >= 512 ? plan_tokens : 128),
      embedding(path) {
  runtime.defer_contexts = true;
  if (!core.mapped || !core.fuse_residual)
    throw std::runtime_error("native mapped residual required");

  for (int layer = 0; layer < 40; layer++) {
    banks.emplace_back(new Bank(root, layer));
    stride = std::max(stride, align4096(banks.back()->maximum + 4095));
  }
  if (plan_tokens < 128 || plan_tokens > 1024 || plan_tokens % 128)
    throw std::runtime_error("invalid prefill plan length");
  sequence_tokens = plan_tokens;
  workspace_tokens = plan_tokens;
  sequence_valid_tokens = getenv("PREFILL_VALID_TOKENS")
      ? atoi(getenv("PREFILL_VALID_TOKENS"))
      : plan_tokens;
  if (sequence_tokens > 128) {
    // DSP consumes this layer's memo directly. Uncached memory avoids a
    // full-bank CPU cache flush each time the selected expert table changes.
    if (async_io) {
      layer_memo =
          std::make_shared<Buffer>(runtime, 4096 + 256 * stride, false, false);
      // Registering a delayed fd does not reserve DSP VA. Map the large bank
      // before QNN restores its contexts and IO, and keep it until Native ends.
      core.map_buffer(layer_memo, FASTRPC_MAP_FD);
    }
    sequence_routes.resize(sequence_tokens / 128 * 40 * 128 * 8);
    sequence_seeds.resize(sequence_tokens / 128 * 40 * 64, -1);
    sequence_stamps.resize(sequence_seeds.size());
    if (layer_memo) {
      staged_ids.resize(sequence_tokens * 8);
      staged_scores.resize(sequence_tokens * 8);
      staged_shared.resize(sequence_tokens * 2048);
    }
  }
  printf(
      "PREFILL_SCHEDULE plan_tokens=%d ar_len=128 layer_major=1 memo_bytes=%zu\n",
      sequence_tokens,
      layer_memo ? layer_memo->size : 0);
  capacity = configuredExpertSlots();
  pool_count = reuse ? 40 : 4;
  if (!reuse)
    throw std::runtime_error("all-LRU cache requires direct buffer reuse");

  // Expert payloads and bank headers are filled before DSP use; state buffers
  // still initialize normally.
  const bool initialize_experts =
      !(getenv("SKIP_EXPERT_ZERO") &&
        std::string(getenv("SKIP_EXPERT_ZERO")) == "1");
  for (int i = 0; i < pool_count; i++) {
    expert_slots.push_back(std::make_shared<Buffer>(
        runtime, 4096 + capacity * stride, initialize_experts));
    core.map_buffer(expert_slots.back());
    caches.emplace_back(capacity);
  }

  const int io_workers =
      getenv("EXPERT_IO_WORKERS") ? atoi(getenv("EXPERT_IO_WORKERS")) : 8;
  const int startup_workers = startup_policy == "none"
      ? io_workers
      : (getenv("STARTUP_IO_WORKERS") ? atoi(getenv("STARTUP_IO_WORKERS"))
                                      : io_workers);
  if (io_workers < 1 || io_workers > 16 || startup_workers < 1 ||
      startup_workers > 16)
    throw std::runtime_error("expert IO workers must be 1 through 16");
  readers = std::make_unique<ExpertReaders>(startup_workers, stride);
  printf(
      "PREFILL_IO_CONFIG workers=%d startup_workers=%d\n",
      io_workers,
      startup_workers);
  if (async_io && !layer_memo)
    for (int i = 0; i < 2; i++) {
      pipeline_banks.push_back(std::make_shared<Buffer>(
          runtime, 4096 + capacity * stride, initialize_experts));
      core.map_buffer(pipeline_banks.back());
    }
  printf(
      "EXPERT_BUFFER_INIT explicit_zero=%d payload_written_before_use=1\n",
      initialize_experts);
  if (startup_policy != "none" && startup_policy != "sync" &&
      startup_policy != "overlap")
    throw std::runtime_error("invalid startup prefetch policy");

  uint64_t cold_bytes = 0;
  double cold_io_ms = 0, cold_wait_ms = 0, cold_overlap_ms = 0;
  if (startup_policy != "none") {
    if (!async_io)
      throw std::runtime_error("startup preload requires pipeline");
    const char* initial_image = getenv("PREFILL_INITIAL_IMAGE");
    const bool has_initial_image = initial_image && std::string(initial_image) == "1";
    const auto table_path = cold_start_table(root, has_initial_image);
    printf(
        "STARTUP_COLD_PROFILE kind=%s table=%s\n",
        has_initial_image ? "image" : "text",
        table_path.c_str());
    const auto table = bytes(table_path);
    // 启动 profile 仍有每层 64 个候选，仅加载当前容量对应的前缀。
    if (table.size() != 40 * kMaxExpertSlots * 4)
      throw std::runtime_error("cold table dimensions");
    const auto* ids = reinterpret_cast<const int32_t*>(table.data());
    for (int layer = 0; layer < 40; layer++) {
      std::set<int> seen;
      for (int i = 0; i < capacity; i++) {
        const int expert = ids[layer * kMaxExpertSlots + i], slot = i;
        if (expert < 0 || expert >= 256 || !seen.insert(expert).second)
          throw std::runtime_error("invalid cold IDs");
        auto& bank = *banks[layer];
        const size_t offset = bank.offset(expert), length = bank.size(expert),
                     aligned = offset & ~size_t(4095);
        const size_t span = align4096(offset - aligned + length);
        cold_bytes += span;
        startup_reads.push_back(
            {bank.fd,
             0,
             aligned,
             span,
             offset - aligned,
             length,
             static_cast<uint8_t*>(expert_slots[layer]->data) + 4096 +
                 slot * stride});
        caches[layer].entries[slot] = {expert, 0, true};
      }
    }
    readers->start(startup_reads.data(), startup_reads.size());
    if (startup_policy == "sync") {
      auto waited = Clock::now();
      readers->finish();
      cold_wait_ms = elapsed(waited);
      cold_io_ms = readers->wall_ms();
    }
  }
  const auto context_started = Clock::now();

  // Grouped Native output is sequence-sized; QNN registers an AR128 tensor.
  // Give QNN its own chain so one fd is never registered at both lengths.
  const auto chain = sequence_tokens >= 512
      ? std::make_shared<Buffer>(runtime, 128 * 2048 * 4)
      : core.outbuf[0];
  shared_io = std::make_unique<SharedIo>(runtime);
  for (int layer = 0; layer < 40; layer++) {
    attention.emplace_back(
        new Attention(runtime, root, layer, layer % 4 != 3, false));
    if (!attention.back()->linear) {
      const int window = attention.back()->history + 128;
      if (prefill_window && prefill_window != window)
        throw std::runtime_error("inconsistent prefill KV windows");
      prefill_window = window;
    }
    router.emplace_back(new Graph(
        runtime, root + "/layer" + std::to_string(layer) + "/moe-front.bin"));
    // Release each layer's provisional activation IO before building the next.
    shared_io->bind_layer(*attention.back(), *router.back(), chain);
  }
  logical_capacity = getenv("DECODE_CONTEXT_CAPACITY")
      ? atoi(getenv("DECODE_CONTEXT_CAPACITY"))
      : attention[0]->input.size() == 4 ? prefill_window - 128
                                        : prefill_window;
  if (logical_capacity < 1 || sequence_tokens > prefill_window)
    throw std::runtime_error("prefill plan exceeds compiled KV window");
  // QNN IO remains AR128; grouped Native calls need separate sequence-sized IO.
  if (sequence_tokens < 512) {
    core.inbuf[0] = shared_io->hidden;
    core.inbuf[1] = shared_io->ids;
    core.inbuf[2] = shared_io->scores;
    core.inbuf[3] = shared_io->shared;
    core.residual = shared_io->residual;
  } else
    core.residual = std::make_shared<Buffer>(runtime, plan_tokens * 2048 * 4);
  plan_inputs = core.inbuf;
  plan_output = core.outbuf[0];
  plan_residual = core.residual;
  head = std::make_unique<Graph>(runtime, root + "/head.bin");
  std::vector<Graph*> graphs;
  for (int i = 0; i < 40; i++) {
    graphs.push_back(&attention[i]->graph);
    graphs.push_back(router[i].get());
  }
  graphs.push_back(head.get());
  if (const char* enabled = getenv("VISION_INPUT");
      enabled && std::string(enabled) == "1") {
    vision = std::make_unique<VisionEncoder>(runtime, root);
    graphs.push_back(&vision->graph);
  }
  ContextRestore restore;

  restore.load(runtime, graphs);
  for (auto* graph : graphs)
    graph->register_io();
  for (int i = 0; i < 4; i++)
    core.map_buffer(core.inbuf[i]);
  core.map_buffer(core.outbuf[0]);
  core.map_buffer(core.residual);

  const std::string rope_root =
      getenv("CHAT_ROPE_ROOT") ? getenv("CHAT_ROPE_ROOT") : root;
  cos = bytes(rope_root + "/cos.f32");
  sin = bytes(rope_root + "/sin.f32");
  validate_rope_tables(cos.size(), sin.size(), logical_capacity);
  const auto context_finished = Clock::now();

  if (startup_policy != "none") {
    if (startup_policy == "overlap") {
      auto waited = Clock::now();
      readers->finish();
      cold_wait_ms = elapsed(waited);
      cold_io_ms = readers->wall_ms();
      cold_overlap_ms = readers->overlap_ms(context_started, context_finished);
    }
    for (auto& buffer : expert_slots)
      core.sync_buffer(buffer, DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE);
    startup_seed_pending = true;
  }
  printf(
      "STARTUP_COLD_PRELOAD enabled=%d overlap=%d experts=%zu read_bytes=%llu io_ms=%.6f context_ms=%.6f overlap_ms=%.6f wait_ms=%.6f included_in_setup=1\n",
      startup_policy != "none",
      startup_policy == "overlap",
      startup_reads.size(),
      (unsigned long long)cold_bytes,
      cold_io_ms,
      std::chrono::duration<double, std::milli>(
          context_finished - context_started)
          .count(),
      cold_overlap_ms,
      cold_wait_ms);
  if (startup_workers != io_workers) {
    readers.reset();
    readers = std::make_unique<ExpertReaders>(io_workers, stride);
  }
}

void PrefillSession::reset() {
  drain_reads();
  image_cursor = 0;
  staged_router_layer = -1;
  memo_layer = -1;
  memo_valid.fill(false);
  core.end_output_read();
  metrics = {};
  active_group = -1;
  next_layer = 0;
  chunk_start = 0;
  sequence_start = 0;
  for (auto& a : attention)
    a->reset();
  // Startup seeds belong only to the first request; independent samples reset
  // cold entries.
  if (startup_seed_pending)
    startup_seed_pending = false;
  else
    for (auto& cache : caches)
      cache.reset();
}

void PrefillSession::embed(const int64_t* tokens, float* out) {
  core.end_output_read();
  if (chunk_start == sequence_start)
    metrics = {};
  active_group = -1;
  next_layer = 0;
  if (chunk_start < 0 || chunk_start + valid_tokens > logical_capacity)
    throw std::runtime_error("prefill chunk position");
  auto t = Clock::now();
  std::fill(out + valid_tokens * 2048, out + 128 * 2048, 0.f);
  for (int token = 0; token < valid_tokens; token++) {
    if (tokens[token] < 0 || tokens[token] >= 248320)
      throw std::runtime_error("token ID");
    if (tokens[token] == 248056) {
      if (image_cursor + 2048 > image_features.size())
        throw std::runtime_error("image tokens exceed encoded features");
      memcpy(
          out + token * 2048, image_features.data() + image_cursor, 2048 * 4);
      image_cursor += 2048;
      continue;
    }
    embedding.lookup(tokens[token], out + token * 2048);
  }
  metrics.embedding += elapsed(t);
  metrics.calls[0]++;
}

void PrefillSession::prepare_layer_memo(int layer) {
  if (!layer_memo || memo_layer == layer)
    return;
  finish_memo_copy();
  memo_valid.fill(false);
  memo_slot_ids.fill(-1);
  route_first_use.fill(-1);
  memset(
      static_cast<uint8_t*>(layer_memo->data) + expert_stream::kReadyOffset,
      0,
      256 * sizeof(uint32_t));
  std::vector<ExpertRead> copies;
  for (int slot = 0; slot < capacity; slot++) {
    const int expert = caches[layer].entries[slot].expert;
    memo_slot_ids[slot] = expert;
    if (expert < 0)
      continue;
    ExpertRead request;
    request.payload_bytes = banks[layer]->size(expert);
    request.cached = static_cast<const uint8_t*>(expert_slots[layer]->data) +
        4096 + slot * stride;
    request.destination = static_cast<uint8_t*>(layer_memo->data) + 4096 +
        expert * stride + (banks[layer]->offset(expert) & size_t(4095));
    copies.push_back(request);
    memo_valid[expert] = true;
  }
  // Seed copies overlap this layer's remaining Attention/Router work. The
  // first MoE chunk joins them before any memo payload becomes consumable.
  memo_copies = std::move(copies);
  readers->start(memo_copies.data(), memo_copies.size());
  memo_copy = MemoCopy::Seed;
  memo_copy_layer = layer;
  copy_compute_spans.clear();
  memo_layer = layer;
}

void PrefillSession::finish_memo_copy() {
  if (memo_copy == MemoCopy::None)
    return;
  const auto waited = Clock::now();
  readers->finish();
  const double wait_ms = elapsed(waited);
  double overlap_ms = 0;
  for (const auto& span : copy_compute_spans)
    overlap_ms += readers->overlap_ms(span.first, span.second);
  if (memo_copy == MemoCopy::Seed) {
    metrics.layer_seed_copy += readers->wall_ms();
    metrics.seed_wait += wait_ms;
    metrics.seed_overlap += overlap_ms;
  } else {
    metrics.staged_copy += readers->wall_ms();
    metrics.retained_overlap += overlap_ms;
    core.sync_buffer(
        expert_slots[memo_copy_layer], DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE);
    metrics.retained_wait += elapsed(waited);
  }
  metrics.read += elapsed(waited);
  memo_copy = MemoCopy::None;
  memo_copy_layer = -1;
  memo_copies.clear();
  copy_compute_spans.clear();
}

void PrefillSession::attend(
    int layer,
    const float* x,
    float* norm,
    float* residual) {
  if (layer != next_layer)
    throw std::runtime_error("prefill layer order");
  auto t = Clock::now();
  core.end_output_read();
  auto& a = *attention[layer];
  a.prefill(
      x,
      chunk_start,
      reinterpret_cast<const float*>(cos.data()),
      reinterpret_cast<const float*>(sin.data()),
      valid_tokens);
  memcpy(norm, a.hidden(), 128 * 2048 * 4);
  memcpy(residual, a.residual(), 128 * 2048 * 4);
  const auto done = Clock::now();
  const double attention_ms = elapsed(t);
  metrics.attention += attention_ms;
  if (a.linear)
    metrics.gdn_attention += attention_ms;
  else
    metrics.full_attention += attention_ms;
  metrics.calls[1]++;
  if (!route_reads.empty())
    attention_io_spans.emplace_back(t, done);
  if (memo_copy != MemoCopy::None)
    copy_compute_spans.emplace_back(t, done);
  // Let the preceding layer's final LRU copy run through this first QNN call
  // before the shared memo is reused for the new layer.
  const int offset = chunk_start - sequence_start;
  if (offset == 0)
    prepare_layer_memo(layer);
  if (layer_memo) {
    if (offset == 0)
      staged_router_layer = -1;
    execute_route(
        layer,
        norm,
        staged_ids.data() + offset * 8,
        staged_scores.data() + offset * 8,
        staged_shared.data() + offset * 2048);
    if (offset + 128 == sequence_tokens)
      staged_router_layer = layer;
  }
}

void PrefillSession::route(
    int layer,
    const float* x,
    int32_t* ids,
    float* scores,
    float* shared) {
  if (!layer_memo) {
    execute_route(layer, x, ids, scores, shared);
    return;
  }
  if (layer != next_layer || staged_router_layer != layer)
    throw std::runtime_error("prefill router handoff order");
  const auto started = Clock::now();
  const int offset = chunk_start - sequence_start;
  memcpy(ids, staged_ids.data() + offset * 8, 128 * 8 * 4);
  memcpy(scores, staged_scores.data() + offset * 8, 128 * 8 * 4);
  memcpy(shared, staged_shared.data() + offset * 2048, 128 * 2048 * 4);
  metrics.router += elapsed(started);
}

void PrefillSession::execute_route(
    int layer,
    const float* x,
    int32_t* ids,
    float* scores,
    float* shared) {
  if (layer != next_layer)
    throw std::runtime_error("router layer order");
  auto t = Clock::now();
  auto& r = *router[layer];
  memcpy(r.input(r.index(true, 0)), x, 128 * 2048 * 4);
  r.execute();
  memcpy(ids, r.output(r.index(false, 0)), 128 * 8 * 4);
  memcpy(scores, r.output(r.index(false, 1)), 128 * 8 * 4);
  memcpy(shared, r.output(r.index(false, 2)), 128 * 2048 * 4);
  metrics.router += elapsed(t);
  metrics.calls[2]++;
  if (memo_copy != MemoCopy::None)
    copy_compute_spans.emplace_back(t, Clock::now());
  if (!route_reads.empty())
    route_io_spans.emplace_back(t, Clock::now());
  if (layer_memo && chunk_start == sequence_start)
    finish_memo_copy();
  if (layer_memo)
    prefetch_route(layer, ids);
}

void PrefillSession::prefetch_route(int layer, const int32_t* ids) {
  const int chunk_offset = chunk_start - sequence_start;
  if (chunk_offset == 0) {
    if (!route_reads.empty())
      throw std::runtime_error("unconsumed route prefetch");
    // Keep the allocation stable while workers consume records already
    // published by earlier Router chunks.
    route_reads.resize(256);
    route_read_ids.clear();
    route_io_spans.clear();
    stream_compute_spans.clear();
    stream_input_spans.clear();
    attention_io_spans.clear();
  }
  auto& bank = *banks[layer];
  const size_t reads_before = route_read_ids.size();
  size_t cache_requests = 0, cache_hits = 0;
  for (int token = 0; token < valid_tokens; token++) {
    std::array<bool, 256> seen{};
    for (int rank = 0; rank < 8; rank++) {
      const int expert = ids[token * 8 + rank];
      if (expert < 0 || expert >= 256 || seen[expert])
        throw std::runtime_error("Top8 route prefetch");
      seen[expert] = true;
      if (trace_expert_cache) {
        ++cache_requests;
        // Repeated routes share one load; Native waits for ready before use.
        cache_hits += route_first_use[expert] >= 0 || memo_valid[expert];
      }
      if (route_first_use[expert] >= 0)
        continue;
      route_first_use[expert] = (chunk_offset + token) * 8 + rank;
      if (memo_valid[expert])
        continue;
      auto& request = route_reads[route_read_ids.size()];
      const size_t offset = bank.offset(expert),
                   aligned = offset & ~size_t(4095);
      request.fd = bank.fd;
      request.file_offset = aligned;
      request.prefix = offset - aligned;
      request.payload_bytes = bank.size(expert);
      request.read_bytes = align4096(request.prefix + request.payload_bytes);
      request.cache_destination = static_cast<uint8_t*>(layer_memo->data) +
          4096 + expert * stride + request.prefix;
      request.completion = reinterpret_cast<uint32_t*>(
                               static_cast<uint8_t*>(layer_memo->data) +
                               expert_stream::kReadyOffset) +
          expert;
      route_read_ids.push_back(expert);
    }
  }
  if (trace_expert_cache) {
    const size_t file_misses = route_read_ids.size() - reads_before;
    printf(
        "PREFILL_CACHE_BLOCK layer=%d start=%d window_start=%d valid_tokens=%d requests=%zu hits=%zu misses=%zu file_load_requests=%zu\n",
        layer,
        chunk_start,
        sequence_start,
        valid_tokens,
        cache_requests,
        cache_hits,
        cache_requests - cache_hits,
        file_misses);
    if (const char* path = getenv("MOE_EXPERT_CACHE_TRACE_FILE")) {
      std::ofstream trace(path, std::ios::binary | std::ios::app);
      const int32_t header[] = {
          0x4d4f4552, 1, layer, chunk_start, sequence_start,
          valid_tokens, capacity};
      trace.write(reinterpret_cast<const char*>(header), sizeof(header));
      for (const auto& slot : caches[layer].entries) {
        const int32_t expert = slot.expert;
        trace.write(reinterpret_cast<const char*>(&expert), sizeof(expert));
      }
      trace.write(
          reinterpret_cast<const char*>(ids),
          valid_tokens * 8 * sizeof(int32_t));
      trace.close();
      if (!trace)
        throw std::runtime_error("Cannot write expert cache route trace");
    }
  }
  if (chunk_offset == 0)
    readers->start(route_reads.data(), route_read_ids.size(), true);
  else
    readers->extend(route_read_ids.size());
  if (chunk_offset + 128 == sequence_tokens)
    readers->seal();
}

uint64_t PrefillSession::fill(
    int layer,
    const std::vector<int>& ids,
    const std::vector<ExpertCache::Binding>& bindings,
    bool seeded,
    const std::vector<ExpertRead>* prepared) {
  auto& bank = *banks[layer];
  auto& buffer = *expert_slots[reuse ? layer : layer % 4];
  auto* data = static_cast<uint8_t*>(buffer.data);
  memcpy(data, bank.index.data(), 4096);
  auto* table = reinterpret_cast<uint64_t*>(data + 32);
  std::fill(table, table + 256, 0);
  std::vector<ExpertRead> requests;
  uint64_t direct = 0;
  for (size_t j = 0; j < ids.size(); j++) {
    const int e = ids[j];
    table[e] = 4096 + bindings[j].slot * stride;
    if (bindings[j].hit)
      continue;
    size_t offset = bank.offset(e), length = bank.size(e),
           aligned = offset & ~size_t(4095),
           span = align4096(offset - aligned + length);
    direct += span;
    if (prepared) {
      const auto& read = prepared->at(j);
      if (read.payload_bytes != length)
        throw std::runtime_error("staged payload size");
      ExpertRead copy;
      copy.payload_bytes = length;
      copy.cached = read.raw_staging + read.prefix;
      copy.cache_destination = data + table[e];
      requests.push_back(copy);
    } else
      requests.push_back(
          {bank.fd,
           0,
           aligned,
           span,
           offset - aligned,
           length,
           data + table[e]});
    caches[reuse ? layer : layer % 4].entries[bindings[j].slot].seeded = seeded;
  }
  if (!requests.empty()) {
    auto waited = Clock::now();
    readers->read(requests.data(), requests.size());
    if (prepared)
      metrics.staged_copy += elapsed(waited);
    else {
      metrics.io_wait += elapsed(waited);
      metrics.io_wall += readers->wall_ms();
    }
  }
  core.sync_buffer(
      expert_slots[reuse ? layer : layer % 4],
      DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE);
  return direct;
}

void PrefillSession::experts(
    int layer,
    const float* x,
    const int32_t* ids,
    const float* scores,
    const float* shared,
    const float* residual,
    float* out,
    int tokens) {
  if (layer != next_layer)
    throw std::runtime_error("MoE layer order");
  if (tokens < 128 || tokens > sequence_tokens || tokens % 128 ||
      (tokens > 128 &&
       (!layer_memo || tokens != sequence_tokens ||
        chunk_start != sequence_start)))
    throw std::runtime_error("MoE token group");
  const int active_tokens =
      tokens == 128 ? valid_tokens : sequence_valid_tokens;
  if (active_tokens < 1 || active_tokens > tokens)
    throw std::runtime_error("invalid MoE valid length");
  auto t = Clock::now();
  prepare_layer_memo(layer);
  finish_memo_copy();
  if (sequence_tokens > 128 && tokens == 128) {
    const int block = (chunk_start - sequence_start) / 128;
    const int base = (block * 40 + layer) * 64;
    for (int slot = 0; slot < capacity; slot++) {
      sequence_seeds[base + slot] = caches[layer].entries[slot].expert;
      sequence_stamps[base + slot] = caches[layer].entries[slot].stamp;
    }
    std::copy(
        ids,
        ids + 128 * 8,
        sequence_routes.begin() + (block * 40 + layer) * 128 * 8);
  }
  Clock::time_point input_begin{}, input_end{};
  bool inputs_published = false;
  auto publish_inputs = [&] {
    if (inputs_published)
      return;
    input_begin = Clock::now();
    memcpy(core.input(0), x, tokens * 2048 * 4);
    memcpy(core.input(1), ids, tokens * 8 * 4);
    memcpy(core.input(2), scores, tokens * 8 * 4);
    memcpy(core.input(3), shared, tokens * 2048 * 4);
    memcpy(core.residual->data, residual, tokens * 2048 * 4);
    core.publish_inputs();
    core.sync_buffer(core.residual, DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE);
    input_end = Clock::now();
    metrics.input_copy_sync += elapsed(input_begin);
    inputs_published = true;
  };

  // 汇总当前组的 Top8 路由；Attention 的 AR128 与专家归并大小相互独立。
  std::array<int, 256> histogram{}, last_use{};
  std::copy(
      ids + (tokens - 128) * 8, ids + tokens * 8, route_history[layer].begin());
  for (int token = 0; token < active_tokens; token++) {
    std::array<bool, 256> seen{};
    for (int k = 0; k < 8; k++) {
      int e = ids[token * 8 + k];
      if (e < 0 || e >= 256 || seen[e])
        throw std::runtime_error("Top8");
      seen[e] = true;
      ++histogram[e];
      last_use[e] = token * 8 + k;
    }
  }
  auto& cache = caches[reuse ? layer : layer % 4];

  std::vector<int> cached, missing;
  std::vector<int> original_ids;
  std::vector<uint64_t> original_stamps;
  for (const auto& entry : cache.entries) {
    original_ids.push_back(entry.expert);
    original_stamps.push_back(entry.stamp);
  }
  auto stamp_of = [&](int e) -> uint64_t {
    if (histogram[e])
      return stamp_offsets[layer] + uint64_t(chunk_start - sequence_start) * 8 +
          last_use[e] + 1;
    auto found = std::find(original_ids.begin(), original_ids.end(), e);
    return found == original_ids.end()
        ? 0
        : original_stamps[found - original_ids.begin()];
  };

  for (int e = 0; e < 256; e++)
    if (histogram[e]) {
      if (layer_memo ? memo_valid[e] : cache.find(e) >= 0)
        cached.push_back(e);
      else
        missing.push_back(e);
    }
  std::sort(missing.begin(), missing.end(), [&](int a, int b) {
    return last_use[a] < last_use[b];
  });
  const int batch_capacity = capacity;

  // 命中专家先算，缺失读取与计算重叠。多块输入直接消费共享 memo；
  // 单块输入仍用两个受 LRU 槽容量约束的 pipeline bank。
  std::vector<std::vector<int>> batches;
  if (!cached.empty())
    batches.push_back(cached);
  for (size_t offset = 0; offset < missing.size(); offset += batch_capacity)
    batches.emplace_back(
        missing.begin() + offset,
        missing.begin() + std::min(missing.size(), offset + batch_capacity));
  const double initial_wall = metrics.io_wall,
               initial_overlap = metrics.io_overlap,
               initial_wait = metrics.io_wait, initial_native = metrics.native;
  std::vector<ExpertRead> prefetched;
  std::array<std::vector<int>, 2> bank_ids;

  auto stage = [&](size_t batch, std::vector<ExpertRead>& requests) {
    const int ping = batch % 2;
    const auto& selected = batches[batch];
    if (!route_reads.empty()) {
      if (selected != route_read_ids)
        throw std::runtime_error("route prefetch batch mismatch");
      requests = std::move(route_reads);
      route_read_ids.clear();
      for (int expert : selected)
        memo_valid[expert] = true;
      return;
    }
    auto& buffer = layer_memo ? *layer_memo : *pipeline_banks[ping];
    auto* data = static_cast<uint8_t*>(buffer.data);
    auto& bank = *banks[layer];
    // The next batch may load while DSP reads the current header. Publish
    // the memo's selection only after the preceding native call finishes.
    if (!layer_memo) {
      memcpy(data, bank.index.data(), 4096);
      auto* table = reinterpret_cast<uint64_t*>(data + 32);
      std::fill(table, table + 256, 0);
      for (size_t j = 0; j < selected.size(); j++)
        table[selected[j]] = 4096 + j * stride;
    }
    bank_ids[ping] = selected;
    requests.resize(selected.size());
    for (size_t j = 0; j < selected.size(); j++) {
      const int expert = selected[j];
      const size_t offset = bank.offset(expert), length = bank.size(expert),
                   aligned = offset & ~size_t(4095);
      requests[j] = {
          bank.fd,
          0,
          aligned,
          align4096(offset - aligned + length),
          offset - aligned,
          length,
          layer_memo ? nullptr : data + 4096 + j * stride};
      if (layer_memo) {
        auto* block = data + 4096 + expert * stride;
        if (memo_valid[expert]) {
          requests[j].cached = block + requests[j].prefix;
          metrics.layer_reuse_bytes += length;
        } else {
          // O_DIRECT cannot pin rpcmem pages on this device. Copy the miss
          // from reader staging once; subsequent chunks reuse it in place.
          requests[j].cache_destination = block + requests[j].prefix;
          memo_valid[expert] = true;
        }
      }
    }
    readers->start(requests.data(), requests.size());
  };
  auto record_route_overlap = [&] {
    for (const auto& span : route_io_spans)
      metrics.router_io_overlap += readers->overlap_ms(span.first, span.second);
    route_io_spans.clear();
  };
  if (layer_memo) {
    auto stamp = Clock::now();
    auto* data = static_cast<uint8_t*>(layer_memo->data);
    auto& bank = *banks[layer];
    auto* table = reinterpret_cast<uint64_t*>(data + 32);
    auto* order =
        reinterpret_cast<uint16_t*>(data + expert_stream::kOrderOffset);
    auto* ready =
        reinterpret_cast<uint32_t*>(data + expert_stream::kReadyOffset);
    memcpy(data, bank.index.data(), 32);
    std::fill(table, table + 256, 0);
    std::vector<int> selected = cached;
    std::vector<int> pending = missing;
    std::sort(pending.begin(), pending.end(), [&](int a, int b) {
      return route_first_use[a] < route_first_use[b];
    });
    selected.insert(selected.end(), pending.begin(), pending.end());
    for (size_t i = 0; i < selected.size(); i++) {
      const int expert = selected[i];
      order[i] = expert;
      table[expert] = 4096 + expert * stride + (bank.offset(expert) & 4095);
      const int slot = cache.find(expert);
      metrics.hits += slot >= 0;
      metrics.seed_hits += slot >= 0 && cache.entries[slot].seeded;
    }
    for (int expert : cached) {
      __atomic_store_n(ready + expert, expert_stream::kReady, __ATOMIC_RELEASE);
      metrics.layer_reuse_bytes += bank.size(expert);
    }
    for (int expert : missing)
      if (std::find(route_read_ids.begin(), route_read_ids.end(), expert) ==
          route_read_ids.end())
        throw std::runtime_error("expert missing from layer route queue");
    const double setup_ms = elapsed(stamp);
    metrics.read += setup_ms;
    metrics.batch_setup += setup_ms;
    int descriptor[] = {
        layer_memo->fd,
        int(layer_memo->size),
        ++generations[0],
        expert_stream::mapping_control(0, tokens, active_tokens),
        expert_stream::kPhase,
        int(selected.size()),
        active_tokens * 8,
        layer};
    memcpy(core.input(4), descriptor, sizeof(descriptor));
    Clock::time_point native_end;
    try {
      publish_inputs();
      stamp = Clock::now();
      core.execute(true);
      native_end = Clock::now();
    } catch (...) {
      if (!route_reads.empty()) {
        try {
          readers->finish();
        } catch (const std::exception& error) {
          fprintf(stderr, "pending stream read failed: %s\n", error.what());
        }
        route_reads.clear();
        route_read_ids.clear();
      }
      throw;
    }
    const double native_ms = elapsed(stamp);
    metrics.native += native_ms;
    metrics.native_calls++;
    metrics.requests += selected.size();
    metrics.assignments += active_tokens * 8;
    if (missing.empty()) {
      metrics.ready_native += native_ms;
      metrics.ready_calls++;
    } else {
      metrics.missing_native += native_ms;
      metrics.missing_calls++;
    }
    if (!route_reads.empty()) {
      stream_compute_spans.emplace_back(stamp, native_end);
      stream_input_spans.emplace_back(input_begin, input_end);
    }
    for (int expert : selected)
      memo_valid[expert] = true;
    if (!route_reads.empty() &&
        chunk_start - sequence_start + tokens == sequence_tokens) {
      auto waited = Clock::now();
      try {
        readers->finish();
      } catch (...) {
        route_reads.clear();
        route_read_ids.clear();
        throw;
      }
      const double duration = elapsed(waited);
      metrics.io_wait += duration;
      metrics.read += duration;
      metrics.io_wall += readers->wall_ms();
      for (const auto& span : stream_compute_spans)
        metrics.io_overlap += readers->overlap_ms(span.first, span.second);
      for (const auto& span : stream_input_spans)
        metrics.input_io_overlap +=
            readers->overlap_ms(span.first, span.second);
      for (const auto& span : attention_io_spans)
        metrics.attention_io_overlap +=
            readers->overlap_ms(span.first, span.second);
      record_route_overlap();
      for (size_t i = 0; i < route_read_ids.size(); i++) {
        const auto& request = route_reads[i];
        metrics.direct += request.read_bytes;
        metrics.pread_worker += request.pread_ms;
        metrics.payload_copy_worker += request.copy_ms;
      }
      route_reads.clear();
      route_read_ids.clear();
    }
  } else
    for (size_t batch = 0; batch < batches.size(); batch++) {
      auto stamp = Clock::now();
      const double wait_before = metrics.io_wait;
      const double input_before = metrics.input_copy_sync;
      auto& selected = batches[batch];
      const auto bindings = layer_memo ? std::vector<ExpertCache::Binding>{}
                                       : cache.prepare(selected);
      int assignments = 0;
      for (size_t j = 0; j < selected.size(); j++) {
        if (layer_memo) {
          const int slot = cache.find(selected[j]);
          metrics.hits += slot >= 0;
          metrics.seed_hits += slot >= 0 && cache.entries[slot].seeded;
        } else {
          metrics.hits += bindings[j].hit;
          metrics.seed_hits += bindings[j].seed_hit;
        }
        assignments += histogram[selected[j]];
      }
      const bool staged =
          async_io && (layer_memo || !(batch == 0 && !cached.empty()));

      std::shared_ptr<Buffer> current;
      int mapping_index;
      if (staged) {
        const bool memo_ready = layer_memo && batch == 0 && !cached.empty();
        if (prefetched.empty() && !memo_ready) {
          stage(batch, prefetched);
          try {
            publish_inputs();
          } catch (...) {
            // The reader borrows prefetched until finish, including error
            // paths.
            try {
              readers->finish();
            } catch (const std::exception& error) {
              fprintf(stderr, "pending expert read failed: %s\n", error.what());
            }
            throw;
          }
          auto waited = Clock::now();
          readers->finish();
          metrics.io_wait += elapsed(waited);
          metrics.io_wall += readers->wall_ms();
          record_route_overlap();
          metrics.input_io_overlap +=
              readers->overlap_ms(input_begin, input_end);
        }
        if (memo_ready)
          for (int expert : selected)
            metrics.layer_reuse_bytes += banks[layer]->size(expert);
        for (const auto& read : prefetched) {
          if (!read.cached)
            metrics.direct += read.read_bytes;
          metrics.pread_worker += read.pread_ms;
          metrics.payload_copy_worker += read.copy_ms;
        }
        current = layer_memo ? layer_memo : pipeline_banks[batch % 2];
        mapping_index = layer_memo ? 0 : int(batch % 2) + 1;
        if (layer_memo) {
          auto* data = static_cast<uint8_t*>(current->data);
          auto& bank = *banks[layer];
          memcpy(data, bank.index.data(), 4096);
          auto* table = reinterpret_cast<uint64_t*>(data + 32);
          std::fill(table, table + 256, 0);
          for (int expert : selected)
            table[expert] =
                4096 + expert * stride + (bank.offset(expert) & size_t(4095));
        }
        core.sync_buffer(current, DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE);
      } else {
        metrics.direct += fill(layer, selected, bindings);
        current = expert_slots[layer];
        mapping_index = async_io ? 3 : layer % 4;
      }
      metrics.requests += selected.size();
      const double setup_ms =
          elapsed(stamp) - (metrics.input_copy_sync - input_before);
      metrics.read += setup_ms;
      metrics.batch_setup += setup_ms - (metrics.io_wait - wait_before);
      const bool ready_batch = batch == 0 && !cached.empty();

      std::vector<ExpertRead> next_reads;
      if (async_io && batch + 1 < batches.size())
        stage(batch + 1, next_reads);
      bool last = batch + 1 == batches.size();

      int phase = batch == 0 ? (last ? 0 : 1) : (last ? 3 : 2);
      // bank fd/大小/generation/映射窗、累加阶段、专家数、路由项数、层号。
      // phase: 0=单批，1=首批，2=中间批，3=末批；仅末批合入 shared/residual。
      int descriptor[] = {
          current->fd,
          int(current->size),
          ++generations[mapping_index],
          expert_stream::mapping_control(mapping_index, tokens, active_tokens),
          phase,
          int(selected.size()),
          assignments,
          layer};
      memcpy(core.input(4), descriptor, 32);

      try {
        publish_inputs();
        stamp = Clock::now();
        core.execute(last);
      } catch (...) {
        // next_reads 的存储被 reader 借用，异常展开前也必须等读取结束。
        if (!next_reads.empty()) {
          try {
            readers->finish();
          } catch (const std::exception& error) {
            fprintf(stderr, "pending expert read failed: %s\n", error.what());
          }
        }
        throw;
      }
      const auto native_end = Clock::now();
      const double native_ms = elapsed(stamp);
      metrics.native += native_ms;
      if (ready_batch) {
        metrics.ready_native += native_ms;
        metrics.ready_calls++;
      } else {
        metrics.missing_native += native_ms;
        metrics.missing_calls++;
      }
      metrics.native_calls++;
      metrics.assignments += assignments;
      if (!next_reads.empty()) {
        auto waited = Clock::now();
        readers->finish();
        const double duration = elapsed(waited);
        metrics.io_wait += duration;
        metrics.read += duration;
        metrics.io_wall += readers->wall_ms();
        record_route_overlap();
        metrics.io_overlap += readers->overlap_ms(stamp, native_end);
        metrics.input_io_overlap += readers->overlap_ms(input_begin, input_end);
      }
      prefetched = std::move(next_reads);
    }

  // Preserve the original AR128 physical slot order and per-block handoff
  // snapshots even when arithmetic consumes all tokens together.
  if (tokens > 128) {
    for (int block = 0; block < tokens; block += 128) {
      const int seed_base = (block / 128 * 40 + layer) * 64;
      for (int slot = 0; slot < capacity; slot++) {
        sequence_seeds[seed_base + slot] = cache.entries[slot].expert;
        sequence_stamps[seed_base + slot] = cache.entries[slot].stamp;
      }
      std::copy(
          ids + block * 8,
          ids + (block + 128) * 8,
          sequence_routes.begin() + (block / 128 * 40 + layer) * 128 * 8);
      if (block + 128 < tokens)
        cache.advance(ids + block * 8, 128, block, stamp_offsets[layer]);
    }
    chunk_start = sequence_start + tokens - 128;
    histogram.fill(0);
    last_use.fill(0);
    valid_tokens = active_tokens - (tokens - 128);
    for (int i = 0; i < valid_tokens * 8; i++) {
      const int expert = ids[(tokens - 128) * 8 + i];
      ++histogram[expert];
      last_use[expert] = i;
    }
    for (int slot = 0; slot < capacity; slot++) {
      original_ids[slot] = cache.entries[slot].expert;
      original_stamps[slot] = cache.entries[slot].stamp;
    }
  }

  if (async_io) {
    auto copied = Clock::now();
    std::vector<ExpertRead> retain;
    // 计算按专家分组，缓存保留顺序仍按 token 的最后访问时间恢复。
    std::vector<int> recent;
    for (int e = 0; e < 256; e++)
      if (histogram[e] ||
          std::find(original_ids.begin(), original_ids.end(), e) !=
              original_ids.end())
        recent.push_back(e);
    std::sort(recent.begin(), recent.end(), [&](int a, int b) {
      return stamp_of(a) > stamp_of(b);
    });
    recent.resize(std::min(recent.size(), size_t(capacity)));
    std::vector<int> retained(capacity, -1);
    for (int e : recent) {
      auto found = std::find(original_ids.begin(), original_ids.end(), e);
      if (found != original_ids.end())
        retained[found - original_ids.begin()] = e;
    }
    for (int e : recent)
      if (std::find(retained.begin(), retained.end(), e) == retained.end())
        *std::find(retained.begin(), retained.end(), -1) = e;
    const bool flush_payloads =
        !layer_memo || chunk_start - sequence_start + 128 == sequence_tokens;
    for (int slot = 0; slot < capacity; slot++) {
      const int expert = retained[slot];
      cache.entries[slot] = {expert, expert < 0 ? 0 : stamp_of(expert), false};
      // Intermediate chunks update LRU metadata only. Payloads remain in the
      // current-layer memo until its final chunk; compare with physical seeds.
      const int physical_id =
          layer_memo ? memo_slot_ids[slot] : original_ids[slot];
      if (!flush_payloads || expert < 0 || physical_id == expert)
        continue;
      const uint8_t* source = layer_memo
          ? static_cast<const uint8_t*>(layer_memo->data) + 4096 +
              expert * stride + (banks[layer]->offset(expert) & size_t(4095))
          : nullptr;
      for (int ping = 0; !layer_memo && ping < 2; ping++)
        if (std::find(bank_ids[ping].begin(), bank_ids[ping].end(), expert) !=
            bank_ids[ping].end()) {
          auto* data = static_cast<uint8_t*>(pipeline_banks[ping]->data);
          const auto* table = reinterpret_cast<const uint64_t*>(data + 32);
          source = data + table[expert];
        }
      if (!source)
        throw std::runtime_error(
            "retained expert absent from final pipeline banks");
      ExpertRead request;
      request.payload_bytes = banks[layer]->size(expert);
      request.cached = source;
      request.cache_destination =
          static_cast<uint8_t*>(expert_slots[layer]->data) + 4096 +
          slot * stride;
      retain.push_back(request);
    }
    if (flush_payloads) {
      if (layer_memo) {
        // The payload memo remains untouched until the following layer's first
        // Attention finishes. Decode needs the LRU only after final prefill.
        memo_copies = std::move(retain);
        readers->start(memo_copies.data(), memo_copies.size());
        memo_copy = MemoCopy::Retain;
        memo_copy_layer = layer;
        copy_compute_spans.clear();
      } else {
        readers->read(retain.data(), retain.size());
        core.sync_buffer(
            expert_slots[layer], DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE);
        metrics.staged_copy += elapsed(copied);
      }
    }
    const double prepare_ms = elapsed(copied);
    metrics.retained_prepare += prepare_ms;
    metrics.read += prepare_ms;
  }
  for (auto& entry : cache.entries)
    if (entry.expert >= 0 && histogram[entry.expert])
      entry.stamp = stamp_of(entry.expert);

  cache.clock = stamp_offsets[layer] +
      uint64_t(chunk_start - sequence_start + valid_tokens) * 8;
  printf(
      "PREFILL_PIPE layer=%d chunk=%d group_tokens=%d ready_experts=%zu missing_experts=%zu async=%d io_wall_ms=%.6f overlap_ms=%.6f wait_ms=%.6f native_ms=%.6f\n",
      layer,
      chunk_start,
      tokens,
      cached.size(),
      missing.size(),
      async_io,
      metrics.io_wall - initial_wall,
      metrics.io_overlap - initial_overlap,
      metrics.io_wait - initial_wait,
      metrics.native - initial_native);
  if (reuse) {
    std::vector<int> recent;
    for (int e = 0; e < 256; e++)
      if (histogram[e] ||
          std::find(original_ids.begin(), original_ids.end(), e) !=
              original_ids.end())
        recent.push_back(e);
    std::sort(recent.begin(), recent.end(), [&](int a, int b) {
      return stamp_of(a) > stamp_of(b);
    });
    recent.resize(std::min(size_t(64), recent.size()));
    std::reverse(recent.begin(), recent.end());
    for (int slot = 0; slot < 64; slot++)
      layer_history[layer][slot] = {-1, 0, 0};
    for (size_t slot = 0; slot < recent.size(); slot++)
      layer_history[layer][slot] = {recent[slot], 0, stamp_of(recent[slot])};
  } else
    for (int slot = 0; slot < capacity; slot++)
      layer_history[layer][slot] = {
          cache.entries[slot].expert, 0, cache.entries[slot].stamp};
  layer_clocks[layer] = cache.clock;
  memcpy(out, core.output(0), tokens * 2048 * 4);
  auto stamp = Clock::now();
  if (sequence_tokens == 128)
    attention[layer]->feedback();
  metrics.feedback += elapsed(stamp);
  metrics.moe += elapsed(t);
  metrics.calls[3] += tokens / 128;
  if (chunk_start - sequence_start + 128 == sequence_tokens)
    next_layer++;
}

void PrefillSession::project(const float* x, float* logits) {
  if (next_layer != 40)
    throw std::runtime_error("incomplete prefill graph");
  auto t = Clock::now();
  core.end_output_read();
  int in = head->index(true, 0), out = head->index(false, 0);
  const size_t count = Graph::elements(head->inputs[in]);
  if (count != 2048 && count != 128 * 2048)
    throw std::runtime_error("prefill head shape");
  if (count == 2048)
    memcpy(head->input(in), x + (valid_tokens - 1) * 2048, count * 4);
  else {
    memcpy(head->input(in), x, count * 4);
    memcpy(
        static_cast<float*>(head->input(in)) + 127 * 2048,
        x + (valid_tokens - 1) * 2048,
        2048 * 4);
  }
  head->execute();
  memcpy(logits, head->output(out), 248320 * 4);
  metrics.head += elapsed(t);
  metrics.calls[4]++;
  if (memo_copy != MemoCopy::None)
    copy_compute_spans.emplace_back(t, Clock::now());
  finish_memo_copy();
  chunk_start += valid_tokens;
}

void PrefillSession::drain_reads() {
  std::exception_ptr failure;
  if (readers) {
    try {
      readers->drain();
    } catch (...) {
      failure = std::current_exception();
    }
  }
  if (memo_copy == MemoCopy::Retain) {
    try {
      core.sync_buffer(
          expert_slots[memo_copy_layer], DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE);
    } catch (...) {
      if (!failure)
        failure = std::current_exception();
    }
  }
  route_reads.clear();
  route_read_ids.clear();
  memo_copies.clear();
  memo_copy = MemoCopy::None;
  memo_copy_layer = -1;
  copy_compute_spans.clear();
  memo_layer = -1;
  staged_router_layer = -1;
  if (failure)
    std::rethrow_exception(failure);
}

void PrefillSession::finish_prefill_native() {
  drain_reads();
  readers.reset();
  core.close();
  layer_memo.reset();
}

PrefillSession::~PrefillSession() {
  try {
    finish_prefill_native();
    vision.reset();
    head.reset();
    router.clear();
    attention.clear();
    printf(
        "PTE_LIFECYCLE context_creates=%llu context_frees=%llu memory_registers=%llu memory_deregisters=%llu\n",
        (unsigned long long)runtime.context_creates,
        (unsigned long long)runtime.context_frees,
        (unsigned long long)runtime.memory_registers,
        (unsigned long long)runtime.memory_deregisters);
  } catch (const std::exception& e) {
    fprintf(stderr, "session cleanup failed: %s\n", e.what());
  }
}
