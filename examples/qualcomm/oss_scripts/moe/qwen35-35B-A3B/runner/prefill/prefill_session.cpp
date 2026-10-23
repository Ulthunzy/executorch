#include "prefill_session.h"

PrefillSession::PrefillSession(const std::string& path, int plan_tokens)
    : root(path), core(runtime, path), embedding(path + "/embedding.bf16") {
  runtime.defer_contexts = true;
  if (!core.mapped || !core.fuse_residual)
    throw std::runtime_error("native mapped residual required");

  if (embedding.size() != 248320ULL * 2048 * 2)
    throw std::runtime_error("embedding dimensions");
  for (int layer = 0; layer < 40; layer++) {
    banks.emplace_back(new Bank(root, layer));
    stride = std::max(stride, align4096(banks.back()->maximum + 4095));
  }
  if (plan_tokens < 128 || plan_tokens > 1024 || plan_tokens % 128)
    throw std::runtime_error("invalid prefill plan length");
  sequence_tokens = plan_tokens;
  if (sequence_tokens > 128) {
    void* memory = nullptr;
    if (posix_memalign(&memory, 4096, 256 * stride))
      throw std::runtime_error("layer expert memo allocation");
    layer_memo.reset(static_cast<uint8_t*>(memory));
    sequence_routes.resize(sequence_tokens / 128 * 40 * 128 * 8);
    sequence_seeds.resize(sequence_tokens / 128 * 40 * 64);
    sequence_stamps.resize(sequence_seeds.size());
  }
  printf(
      "PREFILL_SCHEDULE plan_tokens=%d ar_len=128 layer_major=%d memo_bytes=%zu\n",
      sequence_tokens,
      sequence_tokens > 128,
      layer_memo ? 256 * stride : 0);
  // 当前复用路径为 40×64 槽；旧四层池分支会在下方被明确拒绝。
  capacity = reuse ? kExpertSlots : 64;
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
    auto& b = *expert_slots.back();
    check(
        fastrpc_mmap(
            CDSP_DOMAIN_ID, b.fd, b.data, 0, b.size, FASTRPC_MAP_FD_DELAYED),
        "expert slot map");
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
  if (async_io)
    for (int i = 0; i < 2; i++) {
      pipeline_banks.push_back(std::make_shared<Buffer>(
          runtime, 4096 + kExpertSlots * stride, initialize_experts));
      auto& b = *pipeline_banks.back();
      check(
          fastrpc_mmap(
              CDSP_DOMAIN_ID, b.fd, b.data, 0, b.size, FASTRPC_MAP_FD_DELAYED),
          "prefill pipeline bank map");
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
    const auto table = bytes(root + "/cold-experts.i32");
    if (table.size() != 40 * kExpertSlots * 4)
      throw std::runtime_error("cold table dimensions");
    const auto* ids = reinterpret_cast<const int32_t*>(table.data());
    for (int layer = 0; layer < 40; layer++) {
      std::set<int> seen;
      for (int i = 0; i < kExpertSlots; i++) {
        const int expert = ids[layer * kExpertSlots + i], slot = i;
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

  for (int layer = 0; layer < 40; layer++) {
    attention.emplace_back(
        new Attention(runtime, root, layer, layer % 4 != 3, false));
    router.emplace_back(new Graph(
        runtime, root + "/layer" + std::to_string(layer) + "/moe-front.bin"));
  }
  shared_io =
      std::make_unique<SharedIo>(runtime, attention, router, core.outbuf[0]);
  core.inbuf[0] = shared_io->hidden;
  core.inbuf[1] = shared_io->ids;
  core.inbuf[2] = shared_io->scores;
  core.inbuf[3] = shared_io->shared;
  core.residual = shared_io->residual;
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

  cos = bytes(root + "/cos.f32");
  sin = bytes(root + "/sin.f32");
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
  image_cursor = 0;
  memo_layer = -1;
  memo_valid.fill(false);
  core.end_output_read();
  metrics = {};
  active_group = -1;
  next_layer = 0;
  chunk_start = 0;
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
  if (sequence_tokens == 128 || chunk_start == 0)
    metrics = {};
  active_group = -1;
  next_layer = 0;
  if (chunk_start < 0 || chunk_start > 896)
    throw std::runtime_error("prefill chunk position");
  auto t = Clock::now();
  auto* w = reinterpret_cast<const uint16_t*>(embedding.pointer);
  for (int token = 0; token < 128; token++) {
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
    for (int d = 0; d < 2048; d++) {
      uint32_t bits = uint32_t(w[size_t(tokens[token]) * 2048 + d]) << 16;
      memcpy(out + token * 2048 + d, &bits, 4);
    }
  }
  metrics.embedding += elapsed(t);
  metrics.calls[0]++;
}

void PrefillSession::prepare_layer_memo(int layer) {
  if (!layer_memo || memo_layer == layer)
    return;
  const auto started = Clock::now();
  memo_valid.fill(false);
  std::vector<ExpertRead> copies;
  for (int slot = 0; slot < capacity; slot++) {
    const int expert = caches[layer].entries[slot].expert;
    if (expert < 0)
      continue;
    ExpertRead request;
    request.payload_bytes = banks[layer]->size(expert);
    request.cached = static_cast<const uint8_t*>(expert_slots[layer]->data) +
        4096 + slot * stride;
    request.destination = layer_memo.get() + expert * stride +
        (banks[layer]->offset(expert) & size_t(4095));
    copies.push_back(request);
    memo_valid[expert] = true;
  }
  // 仅保留当前层的权重；各 chunk 仍使用同样的 Top8、分批次序和累加方式。
  readers->read(copies.data(), copies.size());
  memo_layer = layer;
  const double duration = elapsed(started);
  metrics.layer_seed_copy += duration;
  metrics.read += duration;
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
      reinterpret_cast<const float*>(sin.data()));
  memcpy(norm, a.hidden(), 128 * 2048 * 4);
  memcpy(residual, a.residual(), 128 * 2048 * 4);
  metrics.attention += elapsed(t);
  metrics.calls[1]++;
}

void PrefillSession::route(
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
    float* out) {
  if (layer != next_layer)
    throw std::runtime_error("MoE layer order");
  auto t = Clock::now();
  prepare_layer_memo(layer);
  if (sequence_tokens > 128) {
    const int base = (chunk_start / 128 * 40 + layer) * 64;
    for (int slot = 0; slot < capacity; slot++) {
      sequence_seeds[base + slot] = caches[layer].entries[slot].expert;
      sequence_stamps[base + slot] = caches[layer].entries[slot].stamp;
    }
    std::copy(
        ids,
        ids + 128 * 8,
        sequence_routes.begin() + (chunk_start / 128 * 40 + layer) * 128 * 8);
  }
  memcpy(core.input(0), x, 128 * 2048 * 4);
  memcpy(core.input(1), ids, 128 * 8 * 4);
  memcpy(core.input(2), scores, 128 * 8 * 4);
  memcpy(core.input(3), shared, 128 * 2048 * 4);
  memcpy(core.residual->data, residual, 128 * 2048 * 4);
  core.publish_inputs();
  core.sync_buffer(core.residual, DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE);

  // 先统计本层 128×Top8 路由，再按专家聚合 token，重复使用同一份专家权重。
  std::array<int, 256> histogram{}, last_use{};
  std::copy(ids, ids + 128 * 8, route_history[layer].begin());
  for (int token = 0; token < 128; token++) {
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
      return uint64_t(chunk_start) * 8 + last_use[e] + 1;
    auto found = std::find(original_ids.begin(), original_ids.end(), e);
    return found == original_ids.end()
        ? 0
        : original_stamps[found - original_ids.begin()];
  };

  for (int e = 0; e < 256; e++)
    if (histogram[e]) {
      if (cache.find(e) >= 0)
        cached.push_back(e);
      else
        missing.push_back(e);
    }
  std::sort(missing.begin(), missing.end(), [&](int a, int b) {
    return last_use[a] < last_use[b];
  });
  const int batch_capacity = capacity;

  // 命中专家先算；缺失按槽容量分批，用两块 pipeline bank 交叠读取与计算。
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
    auto& buffer = *pipeline_banks[ping];
    auto* data = static_cast<uint8_t*>(buffer.data);
    auto& bank = *banks[layer];
    memcpy(data, bank.index.data(), 4096);
    auto* table = reinterpret_cast<uint64_t*>(data + 32);
    std::fill(table, table + 256, 0);
    bank_ids[ping] = selected;
    requests.resize(selected.size());
    for (size_t j = 0; j < selected.size(); j++) {
      const int expert = selected[j];
      const size_t offset = bank.offset(expert), length = bank.size(expert),
                   aligned = offset & ~size_t(4095);
      table[expert] = 4096 + j * stride;
      requests[j] = {
          bank.fd,
          0,
          aligned,
          align4096(offset - aligned + length),
          offset - aligned,
          length,
          data + table[expert]};
      if (layer_memo) {
        auto* block = layer_memo.get() + expert * stride;
        if (memo_valid[expert]) {
          requests[j].cached = block + requests[j].prefix;
          metrics.layer_reuse_bytes += length;
        } else {
          // 保留已读取的有效 payload；下一块直接复制到 DMA bank，跳过 UFS。
          requests[j].cache_destination = block + requests[j].prefix;
          memo_valid[expert] = true;
        }
      }
    }
    readers->start(requests.data(), requests.size());
  };
  for (size_t batch = 0; batch < batches.size(); batch++) {
    auto stamp = Clock::now();
    auto& selected = batches[batch];
    const auto bindings = cache.prepare(selected);
    int assignments = 0;
    for (size_t j = 0; j < selected.size(); j++) {
      metrics.hits += bindings[j].hit;
      metrics.seed_hits += bindings[j].seed_hit;
      assignments += histogram[selected[j]];
    }
    const bool staged = async_io && !(batch == 0 && !cached.empty());

    std::shared_ptr<Buffer> current;
    int mapping_index;
    if (staged) {
      if (prefetched.empty()) {
        stage(batch, prefetched);
        auto waited = Clock::now();
        readers->finish();
        metrics.io_wait += elapsed(waited);
        metrics.io_wall += readers->wall_ms();
      }
      for (const auto& read : prefetched)
        if (!read.cached)
          metrics.direct += read.read_bytes;
      current = pipeline_banks[batch % 2];
      mapping_index = int(batch % 2) + 1;
      core.sync_buffer(current, DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE);
    } else {
      metrics.direct += fill(layer, selected, bindings);
      current = expert_slots[layer];
      mapping_index = async_io ? 3 : layer % 4;
    }
    metrics.requests += selected.size();
    metrics.read += elapsed(stamp);

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
        mapping_index,
        phase,
        int(selected.size()),
        assignments,
        layer};
    memcpy(core.input(4), descriptor, 32);

    stamp = Clock::now();
    try {
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
    metrics.native += elapsed(stamp);
    metrics.native_calls++;
    metrics.assignments += assignments;
    if (!next_reads.empty()) {
      auto waited = Clock::now();
      readers->finish();
      const double duration = elapsed(waited);
      metrics.io_wait += duration;
      metrics.read += duration;
      metrics.io_wall += readers->wall_ms();
      metrics.io_overlap += readers->overlap_ms(stamp, native_end);
    }
    prefetched = std::move(next_reads);
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
    for (int slot = 0; slot < capacity; slot++) {
      const int expert = retained[slot];
      cache.entries[slot] = {expert, expert < 0 ? 0 : stamp_of(expert), false};
      if (expert < 0 || original_ids[slot] == expert)
        continue;
      const uint8_t* source = nullptr;
      for (int ping = 0; ping < 2; ping++)
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
    readers->read(retain.data(), retain.size());
    core.sync_buffer(
        expert_slots[layer], DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE);
    metrics.staged_copy += elapsed(copied);
    metrics.read += elapsed(copied);
  }
  for (auto& entry : cache.entries)
    if (entry.expert >= 0 && histogram[entry.expert])
      entry.stamp = stamp_of(entry.expert);

  cache.clock = uint64_t(chunk_start + 128) * 8;
  printf(
      "PREFILL_PIPE layer=%d async=%d io_wall_ms=%.6f overlap_ms=%.6f wait_ms=%.6f native_ms=%.6f\n",
      layer,
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
    for (int slot = 0; slot < 64; slot++)
      layer_history[layer][slot] = {
          cache.entries[slot].expert, 0, cache.entries[slot].stamp};
  layer_clocks[layer] = cache.clock;
  memcpy(out, core.output(0), 128 * 2048 * 4);
  auto stamp = Clock::now();
  if (sequence_tokens == 128)
    attention[layer]->feedback();
  metrics.feedback += elapsed(stamp);
  metrics.moe += elapsed(t);
  metrics.calls[3]++;
  if (sequence_tokens == 128 || chunk_start + 128 == sequence_tokens)
    next_layer++;
}

void PrefillSession::project(const float* x, float* logits) {
  if (next_layer != 40)
    throw std::runtime_error("incomplete prefill graph");
  auto t = Clock::now();
  core.end_output_read();
  int in = head->index(true, 0), out = head->index(false, 0);
  memcpy(head->input(in), x, 128 * 2048 * 4);
  head->execute();
  memcpy(logits, head->output(out), 248320 * 4);
  metrics.head += elapsed(t);
  metrics.calls[4]++;
  chunk_start += 128;
}

void PrefillSession::finish_prefill_native() {
  readers.reset();
  layer_memo.reset();
  core.close();
  if (!native_mappings_active)
    return;
  for (auto& b : pipeline_banks)
    check(
        fastrpc_munmap(CDSP_DOMAIN_ID, b->fd, b->data, b->size),
        "pipeline unmap");
  for (auto& b : expert_slots)
    check(
        fastrpc_munmap(CDSP_DOMAIN_ID, b->fd, b->data, b->size),
        "expert unmap");
  native_mappings_active = false;
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
