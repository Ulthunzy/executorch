#include "decode_session.h"

DecodeSession::DecodeSession(const std::string& path, size_t cache_mib)
    : root(path), core(runtime, path), embedding(path) {
  runtime.defer_contexts = true;
  if (!core.mapped || !core.fuse_residual)
    throw std::runtime_error("native mapped residual required");

  if (cache_encoding.size() != 16 + 40 * 2 * sizeof(CacheEncoding) ||
      memcmp(cache_encoding.data(), "Q36CACHE", 8))
    throw std::runtime_error("source cache encodings");
  check(
      runtime.q.backendRegisterOpPackage(
          runtime.backend,
          (root + "/libQnnDeltaNetOpPackage_HTP.so").c_str(),
          "DeltaNetOpPackageInterfaceProvider",
          "HTP"),
      "register AR1 GDN V85");
  size_t maximum = 0, bank_size = 0;

  for (int layer = 0; layer < 40; layer++) {
    banks.emplace_back(new Bank(root, layer));
    maximum = std::max(maximum, align4096(banks.back()->maximum + 4095));
    bank_size = std::max(
        bank_size, align4096(4096 + 8 * align4096(banks.back()->maximum)));
  }
  for (int i = 0; i < 2; i++) {
    bank_slots.push_back(std::make_shared<Buffer>(runtime, 4096));
    auto& b = *bank_slots.back();
    check(
        fastrpc_mmap(
            CDSP_DOMAIN_ID, b.fd, b.data, 0, b.size, FASTRPC_MAP_FD_DELAYED),
        "bank slot map");
  }
  const int io_workers = getenv("DECODE_EXPERT_IO_WORKERS")
      ? atoi(getenv("DECODE_EXPERT_IO_WORKERS"))
      : 8;
  if (io_workers < 1 || io_workers > 16)
    throw std::runtime_error("decode reader count must be 1 through 16");
  cache = std::make_unique<ExpertCache>(
      runtime, 40, maximum, cache_mib * 1048576, true, root);

  readers = std::make_unique<ExpertReaders>(io_workers, maximum);
  const auto history = bytes(root + "/prefill-routes.i32");
  if (history.size() != 40 * 128 * 8 * 4)
    throw std::runtime_error("temporal route seed");
  const auto* prior = reinterpret_cast<const int32_t*>(history.data());
  for (int layer = 0; layer < 40; layer++)
    std::copy(
        prior + layer * 128 * 8 + 127 * 8,
        prior + layer * 128 * 8 + 128 * 8,
        previous_routes[layer].begin());
  if (async_io) {
    const auto profile = bytes(root + "/prediction-profile.f32");
    if (profile.size() != 40 * 16 * 9 * 4)
      throw std::runtime_error("prediction profile shape");
    prediction_profile.resize(profile.size() / 4);
    memcpy(prediction_profile.data(), profile.data(), profile.size());
    prefetch = std::make_unique<PrefetchQueue>(maximum);
  }
  execution_windows.reserve(120);

  for (auto& tasks : pending)
    tasks.reserve(2);
  for (int layer = 0; layer < 40; layer++) {
    attention.emplace_back(new Attention(runtime, root, layer, layer % 4 != 3));
    if (attention.back()->output.size() == 7)
      router.emplace_back(nullptr);
    else
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
    if (attention[i]->output.size() != 7)
      graphs.push_back(router[i].get());
  }
  graphs.push_back(head.get());
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
  if (cos.size() < 2048 * 32 * 4 || sin.size() < 2048 * 32 * 4)
    throw std::runtime_error("embedding/RoPE dimensions");
  if (ready_queue) {
    ready_buffer =
        std::make_shared<Buffer>(runtime, 4096, Buffer::CachePolicy::Uncached);
    core.map_buffer(ready_buffer);
  }
  int mapped_banks[41];
  for (int layer = 0; layer < 40; layer++)
    mapped_banks[layer] = cache->fd(layer);
  mapped_banks[40] = cache->size();
  check(
      integer_rpc_configure_banks(core.handle, mapped_banks, 41),
      "configure expert bank mapper");
  if (core.use_mailbox)
    core.start_service();
  reset();
}

void DecodeSession::reset() {
  core.end_output_read();
  cache->reset();
  for (auto& a : attention)
    a->reset();
  position = prefix_tokens;
  next_layer = 0;
  in_step = false;
  metrics = {};
}

void DecodeSession::warm_from_prefill() {
  auto begin = Clock::now();
  const auto raw_ids = bytes(root + "/prefill-history-ids.i32"),
             raw_stamps = bytes(root + "/prefill-history-stamps.u64");
  if (raw_ids.size() != 40 * 64 * 4 || raw_stamps.size() != 40 * 64 * 8)
    throw std::runtime_error("prefill history shape");
  const auto* ids = reinterpret_cast<const int32_t*>(raw_ids.data());
  const auto* stamps = reinterpret_cast<const uint64_t*>(raw_stamps.data());
  const int capacity = cache->slots_per_layer();
  if (capacity != 16 && capacity != 32 && capacity != 64)
    throw std::runtime_error(
        "warm cache requires 16, 32 or 64 slots per layer");
  std::array<std::vector<int>, 40> selected;
  std::vector<ExpertRead> requests;
  uint64_t direct_bytes = 0;
  std::vector<int32_t> saved_ids;
  for (int layer = 0; layer < 40; layer++) {
    std::vector<int> order;
    std::set<int> unique;
    for (int i = 0; i < 64; i++)
      if (ids[layer * 64 + i] >= 0) {
        if (ids[layer * 64 + i] >= 256 ||
            !unique.insert(ids[layer * 64 + i]).second)
          throw std::runtime_error("invalid prefill history");
        order.push_back(i);
      }
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
      return stamps[layer * 64 + a] > stamps[layer * 64 + b];
    });
    if (int(order.size()) < capacity)
      throw std::runtime_error("insufficient prefill history for seed");
    order.resize(capacity);
    std::reverse(order.begin(), order.end());
    auto& bank = *banks[layer];
    dma_buf_sync sync{DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE};
    if (ioctl(cache->fd(layer), DMA_BUF_IOCTL_SYNC, &sync))
      throw std::runtime_error("seed cache begin write");
    for (int slot = 0; slot < capacity; slot++) {
      const int expert = ids[layer * 64 + order[slot]];
      selected[layer].push_back(expert);
      saved_ids.push_back(expert);
      const size_t offset = bank.offset(expert), length = bank.length(expert),
                   start = offset & ~size_t(4095),
                   span = align4096(offset - start + length);
      requests.push_back(
          {bank.fd,
           0,
           start,
           span,
           offset - start,
           length,
           cache->location(layer, slot)});
      direct_bytes += span;
    }
  }
  readers->read(requests.data(), requests.size());
  for (int layer = 0; layer < 40; layer++) {
    cache->seed_layer(layer, selected[layer]);
    cache->publish(layer);
  }
  const double duration = elapsed(begin);
  save(root + "/cache-seed-ids.i32", saved_ids.data(), saved_ids.size() * 4);
  printf(
      "EXPERT_SEED layers=40 slots_per_layer=%d experts=%zu read_bytes=%llu preload_ms=%.6f scope=once_before_decode\n",
      capacity,
      requests.size(),
      (unsigned long long)direct_bytes,
      duration);
}

void DecodeSession::embed(
    const int64_t* token,
    int64_t requested_position,
    float* out) {
  if (in_step || requested_position != position || position < prefix_tokens ||
      position >= context_capacity)
    throw std::runtime_error(
        "decode position must advance sequentially from prefill");
  if (*token < 0 || *token >= 248320)
    throw std::runtime_error("token ID");
  for (auto& tasks : pending) {
    if (!tasks.empty())
      throw std::runtime_error("unfinished prediction token");
  }
  in_step = true;
  next_layer = 0;
  metrics = {};
  execution_windows.clear();
  auto t = Clock::now();
  embedding.lookup(*token, out);
  metrics.embedding += elapsed(t);
  metrics.calls[0]++;
}

void DecodeSession::attend(
    int layer,
    const float* x,
    float* norm,
    float* residual) {
  if (!in_step || layer != next_layer)
    throw std::runtime_error("attention layer order");
  auto t = Clock::now();
  core.end_output_read();
  auto& a = *attention[layer];
  a.execute(
      x,
      position,
      reinterpret_cast<const float*>(cos.data()),
      reinterpret_cast<const float*>(sin.data()));
  execution_windows.emplace_back(t, Clock::now());
  metrics.execution[layer][0] = elapsed(t);
  memcpy(norm, a.graph.output(a.output[0]), 2048 * 4);
  memcpy(residual, a.graph.output(a.output[1]), 2048 * 4);
  metrics.attention += elapsed(t);
  metrics.calls[1]++;
}

void DecodeSession::route(
    int layer,
    const float* x,
    int32_t* ids,
    float* scores,
    float* shared) {
  if (!in_step || layer != next_layer)
    throw std::runtime_error("router layer order");
  auto t = Clock::now();
  if (attention[layer]->output.size() == 7) {
    auto& a = *attention[layer];
    memcpy(ids, a.graph.output(a.output[4]), 8 * 4);
    memcpy(scores, a.graph.output(a.output[5]), 8 * 4);
    memcpy(shared, a.graph.output(a.output[6]), 2048 * 4);
  } else {
    auto& r = *router[layer];
    memcpy(r.input(r.index(true, 0)), x, 2048 * 4);
    r.execute();
    memcpy(ids, r.output(r.index(false, 0)), 8 * 4);
    memcpy(scores, r.output(r.index(false, 1)), 8 * 4);
    memcpy(shared, r.output(r.index(false, 2)), 2048 * 4);
  }
  execution_windows.emplace_back(t, Clock::now());
  metrics.execution[layer][1] = elapsed(t);
  std::copy(ids, ids + 8, metrics.routes[layer].begin());
  metrics.router += elapsed(t);
  metrics.calls[2]++;
  if (async_io && layer + 12 < 40) {
    prefetch->pause(true);
    predict(layer + 12);
  }
}

void DecodeSession::predict(int target) {
  // 可选预测只决定提前读取哪些专家；当前 Router 的真实 Top8 不受预测改变。
  // 同层上一 token 与当前 token 相隔 12/24/36 层的 ID 共同生成候选。
  if (!pending[target].empty())
    throw std::runtime_error("duplicate target prediction");
  std::array<unsigned, 256> codes{};
  std::array<int, 256> ranks;
  ranks.fill(8);
  for (int expert : previous_routes[target])
    codes[expert] |= 1;
  for (int index = 1; index <= 3; index++) {
    const int source = target - 12 * index;
    if (source < 0)
      continue;
    for (int rank = 0; rank < 8; rank++) {
      const int expert = metrics.routes[source][rank];
      codes[expert] |= 1U << index;
      if (index == 1)
        ranks[expert] = rank;
    }
  }
  std::vector<std::pair<float, int>> candidates;
  for (int expert = 0; expert < 256; expert++)
    if (codes[expert] && cache->find(target, expert) < 0) {
      float probability =
          prediction_profile[(target * 16 + codes[expert]) * 9 + ranks[expert]];
      if (probability >= .6f)
        candidates.emplace_back(probability, expert);
    }
  std::sort(candidates.begin(), candidates.end(), [](auto a, auto b) {
    return a.first != b.first ? a.first > b.first : a.second < b.second;
  });
  for (const auto& candidate : candidates) {
    if (pending[target].size() == 2)
      break;
    const int expert = candidate.second;
    const int slot =
        cache->reserve_prefetch(target, expert, previous_routes[target]);
    if (slot < 0)
      continue;
    auto task = std::make_shared<PrefetchTask>();
    task->layer = target;
    task->expert = expert;
    task->slot = slot;
    task->position = position;
    auto& bank = *banks[target];
    const size_t offset = bank.offset(expert), length = bank.length(expert),
                 start = offset & ~size_t(4095);
    task->request = {
        bank.fd,
        0,
        start,
        align4096(offset - start + length),
        offset - start,
        length,
        nullptr,
        nullptr,
        cache->location(target, slot)};
    prefetch->submit(task);
    pending[target].push_back(task);
    metrics.prediction_reads++;
    metrics.prediction_bytes += task->request.read_bytes;
    printf(
        "PREFETCH_SUBMIT step=%d target=%d source=%d expert=%d slot=%d probability=%.6f bytes=%zu\n",
        position - prefix_tokens,
        target,
        target - 12,
        expert,
        slot,
        candidate.first,
        task->request.read_bytes);
  }
}

void DecodeSession::finish_predictions(int layer, const int32_t* actual) {
  if (!prefetch)
    return;
  const auto target_ready = Clock::now();
  for (const auto& task : pending[layer]) {
    auto waited = Clock::now();
    prefetch->wait(task);
    const double residual = elapsed(waited);
    metrics.prediction_wait += residual;
    cache->finish_prefetch(layer, task->slot, task->expert);
    const bool used = std::find(actual, actual + 8, task->expert) != actual + 8;
    metrics.prediction_used += used;
    printf(
        "PREFETCH_DONE step=%d target=%d expert=%d used=%d lead_ms=%.6f io_ms=%.6f remaining_wait_ms=%.6f ready_before_target=%d\n",
        position - prefix_tokens,
        layer,
        task->expert,
        used,
        std::chrono::duration<double, std::milli>(
            target_ready - task->submitted)
            .count(),
        std::chrono::duration<double, std::milli>(
            task->completed - task->started)
            .count(),
        residual,
        task->completed <= target_ready);
  }
}

void DecodeSession::experts(
    int layer,
    const float* x,
    const int32_t* ids,
    const float* scores,
    const float* shared,
    const float* residual,
    float* out) {
  if (!in_step || layer != next_layer)
    throw std::runtime_error("MoE layer order");
  auto total = Clock::now();
  auto predicted_wait = Clock::now();

  finish_predictions(layer, ids);
  metrics.read += elapsed(predicted_wait);
  memcpy(core.input(0), x, 2048 * 4);
  memcpy(core.input(1), ids, 8 * 4);
  memcpy(core.input(2), scores, 8 * 4);
  memcpy(core.input(3), shared, 2048 * 4);
  memcpy(core.residual->data, residual, 2048 * 4);
  core.publish_inputs();
  core.sync_buffer(core.residual, DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE);

  std::copy(ids, ids + 8, metrics.routes[layer].begin());
  auto t = Clock::now();
  auto& bank = *banks[layer];
  auto& output = *bank_slots[layer % 2];
  auto* destination = static_cast<uint8_t*>(output.data);
  memcpy(destination, bank.index.data(), 4096);
  auto* offsets = reinterpret_cast<uint64_t*>(destination + 32);
  std::fill(offsets, offsets + 256, 0);
  std::array<ExpertRead, 8> requests;
  std::set<int> selected;
  for (int i = 0; i < 8; i++) {
    int e = ids[i];
    if (e < 0 || e >= 256 || !selected.insert(e).second)
      throw std::runtime_error("Top8 route");
    size_t offset = bank.offset(e), length = bank.length(e),
           start = offset & ~size_t(4095),
           span = align4096(offset - start + length);
    requests[i] = {bank.fd, 0, start, span, offset - start, length, nullptr};
    metrics.logical += length;
  }
  const auto hit = cache->prepare(layer, ids, requests.data());

  metrics.hits += hit.hot + hit.cold;
  cache->use_direct(layer, ids, requests.data(), offsets);
  for (int i = 0; i < 8; i++)
    if (!(offsets[ids[i]] >> 63))
      throw std::runtime_error("compact bank requires direct cache weights");
  auto& io = metrics.io[layer];
  io.hot = hit.hot;
  io.cold = hit.cold;
  for (auto& r : requests)
    if (!r.cached) {
      metrics.direct += r.read_bytes;
      io.bytes += r.read_bytes;
    }
  unsigned hit_mask = 0;

  for (int i = 0; i < 8; i++)
    if (requests[i].cached)
      hit_mask |= 1U << i;
  // ready 模式一次 RPC 内先消费命中，再消费陆续就绪的缺失专家。
  // split 是兼容路径：分别提交命中与缺失，最终仍按原 Top8 rank 求和。
  const bool ready = hit_first && ready_queue && hit_mask != 255;
  const bool split = hit_first && !ready_queue && hit_mask && hit_mask != 255;
  io.hit_mask = hit_mask;
  io.split = split;

  // bank fd/大小/generation/映射窗、常驻层 fd/大小、mask 或 ready-fd、phase。
  // phase: 0=整批，1=命中部分，2=缺失部分，3=ready 队列。
  int descriptor[] = {
      output.fd,
      int(output.size),
      ++generations[layer % 2],
      layer % 2,
      cache->fd(layer),
      cache->size(),
      0,
      0};
  if (split || ready) {
    // Publish before readers start: hits are pinned, and readers write only
    // disjoint miss slots.
    if (!ready)
      cache->publish(layer);
    core.sync_buffer(
        bank_slots[layer % 2], DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE);
    descriptor[6] = hit_mask;
    descriptor[7] = 1;
    if (ready) {
      auto* flags = static_cast<uint32_t*>(ready_buffer->data);
      __atomic_store_n(flags, unsigned(descriptor[2]), __ATOMIC_RELEASE);
      __atomic_store_n(flags + 1, 0U, __ATOMIC_RELEASE);
      const uint32_t ready_value = getenv("EXPERT_STAGE_READY") &&
              std::string(getenv("EXPERT_STAGE_READY")) == "1"
          ? 3U
          : 1U;
      for (int i = 0; i < 8; i++)
        __atomic_store_n(
            flags + 32 * (i + 1),
            ((hit_mask >> i) & 1U) ? ready_value : 0U,
            __ATOMIC_RELEASE);
      descriptor[6] = ready_buffer->fd;
      descriptor[7] = 3;
    }
    memcpy(core.input(4), descriptor, 32);
  }

  auto io_started = Clock::now();
  if (prefetch)
    prefetch->pause(true);
  readers->start(
      requests.data(),
      8,
      ready ? static_cast<uint32_t*>(ready_buffer->data) : nullptr,
      cache->fd(layer));
  double feedback_ms = 0;

  Clock::time_point hit_started{}, hit_done{};
  if (split) {
    hit_started = Clock::now();
    try {
      core.execute(false);
    } catch (...) {
      readers->finish();
      throw;
    }
    hit_done = Clock::now();
    io.hit_native =
        std::chrono::duration<double, std::milli>(hit_done - hit_started)
            .count();
    metrics.native += io.hit_native;
    metrics.execution[layer][2] += io.hit_native;
    execution_windows.emplace_back(hit_started, hit_done);
  }
  Clock::time_point feedback_started{}, feedback_done{};

  // 本层 state/KV 的反馈只依赖 Attention，可与专家磁盘读取交叠。
  if (overlap_feedback) {
    feedback_started = Clock::now();
    attention[layer]->feedback();
    feedback_done = Clock::now();
    feedback_ms = std::chrono::duration<double, std::milli>(
                      feedback_done - feedback_started)
                      .count();
    metrics.feedback += feedback_ms;
  }
  if (ready) {
    hit_started = Clock::now();
    try {
      core.execute();
    } catch (...) {
      readers->finish();
      throw;
    }
    hit_done = Clock::now();
    io.hit_native =
        std::chrono::duration<double, std::milli>(hit_done - hit_started)
            .count();
    metrics.native += io.hit_native;
    metrics.execution[layer][2] += io.hit_native;
    execution_windows.emplace_back(hit_started, hit_done);
  }
  const auto tail_started = Clock::now();

  readers->finish();
  io.tail_wait = elapsed(tail_started);
  if (overlap_feedback)
    io.feedback_overlap = readers->overlap_ms(feedback_started, feedback_done);
  if (split || ready)
    io.hit_overlap = readers->overlap_ms(hit_started, hit_done);
  if (prefetch)
    prefetch->pause(false);
  io.wait = elapsed(io_started) - io.hit_native;
  if (readers->disk_bytes() != io.bytes)
    throw std::runtime_error("reader chunk byte coverage mismatch");
  io.read_calls = readers->disk_calls();
  for (const auto& r : requests) {
    io.pread_worker += r.pread_ms;
    io.copy_worker += r.copy_ms;
  }
  io_started = Clock::now();
  if (!ready) {
    if (hit_mask != 255)
      cache->publish(layer);
    core.sync_buffer(
        bank_slots[layer % 2], DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE);
  }
  io.sync = elapsed(io_started);
  if (split) {
    descriptor[6] = hit_mask ^ 255;
    descriptor[7] = 2;
  }

  memcpy(core.input(4), descriptor, 32);
  metrics.read += elapsed(t) - feedback_ms - io.hit_native;
  if (!ready) {
    t = Clock::now();
    core.execute();
    execution_windows.emplace_back(t, Clock::now());
    metrics.execution[layer][2] += elapsed(t);
    metrics.native += elapsed(t);
  }
  memcpy(out, core.output(0), 2048 * 4);

  if (!overlap_feedback) {
    t = Clock::now();
    attention[layer]->feedback();
    metrics.feedback += elapsed(t);
  }
  metrics.requests += 8;
  metrics.calls[3]++;
  metrics.moe += elapsed(total);
  next_layer++;
}

void DecodeSession::project(const float* x, float* logits) {
  if (!in_step || next_layer != 40)
    throw std::runtime_error("incomplete decode graph");
  auto t = Clock::now();
  core.end_output_read();
  int input = head->index(true, 0), output = head->index(false, 0);
  if (Graph::size(head->inputs[input]) != 2048 * 4 ||
      Graph::size(head->outputs[output]) != 248320 * 4)
    throw std::runtime_error("AR1 head shape");
  memcpy(head->input(input), x, 2048 * 4);
  head->execute();
  memcpy(logits, head->output(output), 248320 * 4);
  metrics.head += elapsed(t);
  metrics.calls[4]++;
  position++;
  in_step = false;
  std::vector<std::pair<Clock::time_point, Clock::time_point>> reads;
  for (auto& tasks : pending) {
    for (const auto& task : tasks) {
      if (!task->done.load(std::memory_order_acquire))
        throw std::runtime_error("prediction not drained");
      reads.emplace_back(task->started, task->completed);
    }
    tasks.clear();
  }
  std::sort(reads.begin(), reads.end());
  std::vector<std::pair<Clock::time_point, Clock::time_point>> merged;
  for (auto interval : reads) {
    if (!merged.empty() && interval.first <= merged.back().second)
      merged.back().second = std::max(merged.back().second, interval.second);
    else
      merged.push_back(interval);
  }
  for (auto interval : merged) {
    metrics.prediction_union += std::chrono::duration<double, std::milli>(
                                    interval.second - interval.first)
                                    .count();
    for (auto execution : execution_windows)
      if (std::min(interval.second, execution.second) >
          std::max(interval.first, execution.first))
        metrics.prediction_overlap +=
            std::chrono::duration<double, std::milli>(
                std::min(interval.second, execution.second) -
                std::max(interval.first, execution.first))
                .count();
  }
  previous_routes = metrics.routes;
}

DecodeSession::~DecodeSession() {
  prefetch.reset();
  try {
    core.close();
    for (auto& b : bank_slots)
      check(
          fastrpc_munmap(CDSP_DOMAIN_ID, b->fd, b->data, b->size),
          "bank slot unmap");
    head.reset();
    router.clear();
    attention.clear();
    printf(
        "DECODE_LIFECYCLE context_creates=%llu context_frees=%llu memory_registers=%llu memory_deregisters=%llu\n",
        (unsigned long long)runtime.context_creates,
        (unsigned long long)runtime.context_frees,
        (unsigned long long)runtime.memory_registers,
        (unsigned long long)runtime.memory_deregisters);
  } catch (const std::exception& e) {
    fprintf(stderr, "session cleanup failed: %s\n", e.what());
  }
}
