#include "prefill/prefill_session.h"
#include "qnn/qnn_cache.h"
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor_ptr.h>

void PrefillSession::prepare_decode() {
  finish_memo_copy();
  if (!route_reads.empty())
    throw std::runtime_error("unfinished prefill reads at decode handoff");
  core.end_output_read();
  // DSP mappings fix each fd's length. Retain grouped IO in io_mappings and
  // reuse the AR128 buffers already registered with QNN for subsequent turns.
  core.inbuf[0] = shared_io->hidden;
  core.inbuf[1] = shared_io->ids;
  core.inbuf[2] = shared_io->scores;
  core.inbuf[3] = shared_io->shared;
  core.outbuf[0] = attention[0]->hidden_alias;
  core.residual = shared_io->residual;
  sequence_tokens = 128;
  sequence_start = 0;
  memo_layer = -1;
  staged_router_layer = -1;
}

void PrefillSession::append(
    const int64_t* tokens,
    size_t count,
    const float* features,
    size_t feature_count,
    const float* rotary_cos,
    const float* rotary_sin,
    size_t table_count,
    Qwen35SessionState& state,
    float* logits,
    const std::function<bool()>& stopped) {
  if (!tokens || !count || !logits || state.position < 0 ||
      state.position >= state.capacity ||
      count >= size_t(state.capacity - state.position) ||
      state.capacity != logical_capacity || state.slots != capacity ||
      !prefill_module || !core.handle || feature_count % 2048 ||
      (feature_count && !features))
    throw std::runtime_error("invalid resident prefill request");
  const int last_start = state.position + int((count - 1) / 128) * 128;
  if (attention[3]->history < last_start ||
      std::any_of(attention.begin(), attention.end(), [](const auto& a) {
        return a->linear && a->input.size() != 4;
      }))
    throw std::runtime_error(
        "Re-export prefill with full-context history and valid-length padding");
  validate_rope_tables(
      table_count * sizeof(float), table_count * sizeof(float), state.capacity);
  if (!rotary_cos || !rotary_sin || table_count != cos.size() / sizeof(float))
    throw std::runtime_error("prefill RoPE table geometry");
  if (stopped && stopped())
    throw std::runtime_error("prefill interrupted; reset required");

  const auto started = Clock::now();
  memcpy(cos.data(), rotary_cos, cos.size());
  memcpy(sin.data(), rotary_sin, sin.size());
  import_append_state(state);
  image_features.clear();
  if (feature_count)
    image_features.assign(features, features + feature_count);
  image_cursor = 0;
  const int begin = state.position;
  int windows = 0;
  for (size_t offset = 0; offset < count; offset += workspace_tokens) {
    const int current = int(std::min(size_t(workspace_tokens), count - offset));
    const int start = state.position;
    const auto window_started = Clock::now();
    auto result = execute_plan(tokens + offset, current, start, stopped);
    if (metrics.calls[1] != 40 * (sequence_tokens / 128) ||
        metrics.calls[3] != 40 * (sequence_tokens / 128) ||
        metrics.assignments != current * 8 * 40)
      throw std::runtime_error("incomplete layer-major prefill");
    memcpy(
        logits,
        result[0].toTensor().const_data_ptr<float>(),
        248320 * sizeof(float));
    export_append_state(state, start, current, offset + current == count);
    ++windows;
    printf(
        "PREFILL_WINDOW order=layer_major start=%d valid_tokens=%d plan_tokens=%d native_calls=%d total_ms=%.6f\n",
        start,
        current,
        sequence_tokens,
        metrics.native_calls,
        elapsed(window_started));
    if (getenv("MOE_PROFILE_TOTAL") &&
        std::string(getenv("MOE_PROFILE_TOTAL")) == "1")
      printf(
          "MOE_PHASE_PROFILE phase=prefill start=%d tokens=%d total_ms=%.6f gdn_attention_ms=%.6f full_attention_ms=%.6f attention_ms=%.6f router_ms=%.6f native_ms=%.6f read_ms=%.6f io_wall_ms=%.6f io_overlap_ms=%.6f io_wait_ms=%.6f direct_bytes=%llu\n",
          start,
          current,
          elapsed(window_started),
          metrics.gdn_attention,
          metrics.full_attention,
          metrics.attention,
          metrics.router,
          metrics.native,
          metrics.read,
          metrics.io_wall,
          metrics.io_overlap,
          metrics.io_wait,
          (unsigned long long)metrics.direct);
  }
  if (image_cursor != image_features.size())
    throw std::runtime_error("unconsumed prefill image features");
  printf(
      "CHAT_PREFILL order=layer_major start=%d tokens=%zu windows=%d total_ms=%.6f\n",
      begin,
      count,
      windows,
      elapsed(started));
}

void PrefillSession::import_append_state(const Qwen35SessionState& state) {
  drain_reads();
  core.end_output_read();
  memo_layer = -1;
  staged_router_layer = -1;
  if (state.position == 0) {
    for (auto& a : attention)
      a->reset();
  } else {
    for (int layer = 0; layer < 40; ++layer) {
      auto& a = attention[layer];
      for (int c = 0; c < 2; ++c) {
        const auto& tensor = a->cache_owner().inputs[a->cache_input[c]];
        auto destination = qnn_cache_view(tensor, a->cache_data(c));
        const auto& source = state.layer[layer].cache[c];
        if (a->linear && source.elements != destination.elements)
          throw std::runtime_error("GDN state import size mismatch");
        copy_cache_range(
            source,
            destination,
            a->linear ? CacheLayout::State
                : c   ? CacheLayout::Value
                      : CacheLayout::Key,
            a->linear || state.append_only
                ? 0
                : state.capacity - 1 - state.position,
            a->linear ? 0 : a->history - state.position,
            a->linear ? source.elements : state.position,
            !a->linear);
      }
    }
  }
  restore_expert_cache(state);
}

void PrefillSession::restore_expert_cache(const Qwen35SessionState& state) {
  for (int layer = 0; layer < 40; ++layer) {
    auto& cache = caches[layer];
    cache.clock = 0;
    for (int slot = 0; slot < capacity; ++slot) {
      cache.entries[slot] = {
          state.layer[layer].experts[slot],
          state.layer[layer].stamps[slot],
          false};
      cache.clock = std::max(cache.clock, cache.entries[slot].stamp);
    }
    stamp_offsets[layer] = cache.clock;
  }
}

std::vector<executorch::runtime::EValue> PrefillSession::execute_plan(
    const int64_t* tokens,
    int count,
    int start,
    const std::function<bool()>& stopped) {
  if (!prefill_module || !tokens || count < 1 || count > workspace_tokens ||
      start < 0 || start > logical_capacity - count)
    throw std::runtime_error("invalid prefill window");
  if (stopped && stopped())
    throw std::runtime_error("prefill interrupted; reset required");
  finish_memo_copy();
  core.end_output_read();
  core.inbuf = plan_inputs;
  core.outbuf[0] = plan_output;
  core.residual = plan_residual;
  sequence_tokens = (count + 127) / 128 * 128;
  sequence_start = start;
  sequence_valid_tokens = count;
  for (int layer = 0; layer < 40; ++layer)
    stamp_offsets[layer] = caches[layer].clock;
  std::vector<int64_t> padded(tokens, tokens + count);
  padded.resize(dynamic_plan ? sequence_tokens : workspace_tokens, 248044);
  auto input = executorch::extension::make_tensor_ptr(
      {1, int(padded.size())},
      padded.data(),
      executorch::aten::ScalarType::Long);
  stop_requested = &stopped;
  try {
    auto result = prefill_module->execute(
        "prefill_forward", std::vector<executorch::runtime::EValue>{*input});
    stop_requested = nullptr;
    if (!result.ok() || result->size() != 2)
      throw std::runtime_error("layer-major prefill PTE execution failed");
    return std::move(*result);
  } catch (...) {
    stop_requested = nullptr;
    try {
      drain_reads();
    } catch (const std::exception& error) {
      fprintf(stderr, "prefill read cleanup failed: %s\n", error.what());
    }
    throw;
  }
}

void PrefillSession::export_append_state(
    Qwen35SessionState& state, int begin, size_t count, bool final_window) {
  // Only new KV crosses back: historical decode codes retain their original
  // grid and never undergo an additional round trip through prefill encoding.
  for (int layer = 0; layer < 40; ++layer) {
    auto& a = *attention[layer];
    for (int c = 0; c < 2; ++c) {
      const auto& tensor = a.cache_owner().inputs[a.cache_input[c]];
      auto source = qnn_cache_view(tensor, a.export_data(c));
      source.elements = a.export_bytes(c) / 2;
      auto destination = state.layer[layer].cache[c];
      const auto layout = a.linear ? CacheLayout::State
          : c                      ? CacheLayout::Value
                                   : CacheLayout::Key;
      if (!a.linear && source.data == destination.data) {
        if (!final_window)
          continue;
        const size_t length = begin + count;
        // Shared FA storage changes stride at stage boundaries. Its grids
        // match, so move the entire valid prefix without re-encoding it.
        copy_cache_range(
            source,
            destination,
            layout,
            source.elements / 512 - length,
            state.append_only ? 0 : destination.elements / 512 - length,
            length,
            true);
        continue;
      }
      if (!a.linear && !state.append_only) {
        // Move old sliding history once before inserting this turn's suffix.
        std::vector<uint16_t> old(
            destination.data, destination.data + destination.elements);
        copy_cache_range(
            {old.data(), old.size(), destination.scale, destination.offset,
             destination.dtype},
            destination,
            layout,
            state.capacity - 1 - begin,
            state.capacity - 1 - begin - count,
            begin);
      }
      copy_cache_range(
          source,
          destination,
          layout,
          a.linear ? 0 : source.elements / 512 - count,
          a.linear                ? 0
              : state.append_only ? begin
                                  : state.capacity - 1 - count,
          a.linear ? source.elements : count);
    }
    for (int slot = 0; slot < capacity; ++slot) {
      state.layer[layer].experts[slot] = caches[layer].entries[slot].expert;
      state.layer[layer].stamps[slot] = caches[layer].entries[slot].stamp;
    }
    std::copy(
        route_history[layer].begin() + (valid_tokens - 1) * 8,
        route_history[layer].begin() + valid_tokens * 8,
        state.layer[layer].routes);
  }
  state.position = begin + count;
}

void PrefillSession::append_chunk_major(
    const int64_t* tokens,
    size_t count,
    const float* features,
    size_t feature_count,
    const float* rotary_cos,
    const float* rotary_sin,
    size_t table_count,
    Qwen35SessionState& state,
    float* logits,
    const std::function<bool()>& stopped) {
  fprintf(
      stderr,
      "DEPRECATED: chunk-major prefill will be removed in a future release; using the unified layer-major scheduler\n");
  append(
      tokens,
      count,
      features,
      feature_count,
      rotary_cos,
      rotary_sin,
      table_count,
      state,
      logits,
      stopped);
}

void PrefillSession::append_chunk_major_legacy(
    const int64_t* tokens,
    size_t count,
    const float* features,
    size_t feature_count,
    const float* rotary_cos,
    const float* rotary_sin,
    size_t table_count,
    Qwen35SessionState& state,
    float* logits,
    const std::function<bool()>& stopped) {
  const auto append_started = Clock::now();
  prepare_decode();
  if (!tokens || !count || !logits || count > size_t(state.capacity) ||
      state.position < 0 || state.position + count >= size_t(state.capacity) ||
      state.capacity != logical_capacity || state.slots != capacity ||
      sequence_tokens != 128 || !core.handle ||
      feature_count % 2048 || (feature_count && !features))
    throw std::runtime_error("invalid resident AR128 prefill request");
  if (attention[3]->history < state.position ||
      std::any_of(attention.begin(), attention.end(), [](const auto& a) {
        return a->linear && a->input.size() != 4;
      }))
    throw std::runtime_error(
        "Re-export prefill with full-context history and valid-length padding");
  validate_rope_tables(
      table_count * sizeof(float), table_count * sizeof(float), state.capacity);
  if (!rotary_cos || !rotary_sin || table_count != cos.size() / sizeof(float))
    throw std::runtime_error("prefill RoPE table geometry");
  memcpy(cos.data(), rotary_cos, cos.size());
  memcpy(sin.data(), rotary_sin, sin.size());
  image_features.clear();
  if (feature_count)
    image_features.assign(features, features + feature_count);
  image_cursor = 0;
  for (int layer = 0; layer < 40; ++layer) {
    auto& a = *attention[layer];
    for (int c = 0; c < 2; ++c) {
      const auto& tensor = a.cache_owner().inputs[a.cache_input[c]];
      auto destination = qnn_cache_view(tensor, a.cache_data(c));
      const auto& source = state.layer[layer].cache[c];
      if (a.linear && source.elements != destination.elements)
        throw std::runtime_error("GDN state import size mismatch");
      const auto layout = a.linear ? CacheLayout::State
          : c                      ? CacheLayout::Value
                                   : CacheLayout::Key;
      copy_cache_range(
          source,
          destination,
          layout,
          a.linear || state.append_only ? 0
                                        : state.capacity - 1 - state.position,
          a.linear ? 0 : a.history - state.position,
          a.linear ? source.elements : state.position,
          !a.linear);
    }
  }
  restore_expert_cache(state);
  const int begin = state.position;
  double total_ms = 0;
  std::array<int64_t, 128> padded{};
  for (size_t offset = 0; offset < count; offset += 128) {
    if (stopped && stopped())
      throw std::runtime_error("prefill interrupted; reset required");
    const auto started = Clock::now();
    valid_tokens = std::min(size_t(128), count - offset);
    chunk_start = begin + offset;
    sequence_start = chunk_start;
    sequence_valid_tokens = valid_tokens;
    std::copy(tokens + offset, tokens + offset + valid_tokens, padded.begin());
    float* chain = static_cast<float*>(attention[0]->hidden_alias->data);
    embed(padded.data(), chain);
    auto* normalized = static_cast<float*>(shared_io->hidden->data);
    auto* residual = static_cast<float*>(shared_io->residual->data);
    auto* ids = static_cast<int32_t*>(shared_io->ids->data);
    auto* scores = static_cast<float*>(shared_io->scores->data);
    auto* shared = static_cast<float*>(shared_io->shared->data);
    for (int layer = 0; layer < 40; ++layer) {
      stamp_offsets[layer] = caches[layer].clock;
      attend(layer, chain, normalized, residual);
      route(layer, normalized, ids, scores, shared);
      experts(layer, normalized, ids, scores, shared, residual, chain);
    }
    project(chain, logits);
    if (metrics.calls[1] != 40 || metrics.calls[3] != 40 ||
        metrics.assignments != valid_tokens * 8 * 40)
      throw std::runtime_error("incomplete AR128 prefill append");
    const double block_ms = elapsed(started);
    total_ms += block_ms;
    if (getenv("MOE_PROFILE_TOTAL") &&
        std::string(getenv("MOE_PROFILE_TOTAL")) == "1")
      printf(
          "MOE_PHASE_PROFILE phase=prefill_append chunk=%zu tokens=%d total_ms=%.6f gdn_attention_ms=%.6f full_attention_ms=%.6f attention_ms=%.6f router_ms=%.6f native_ms=%.6f read_ms=%.6f io_wall_ms=%.6f io_overlap_ms=%.6f io_wait_ms=%.6f direct_bytes=%llu\n",
          offset / 128, valid_tokens, block_ms, metrics.gdn_attention, metrics.full_attention,
          metrics.attention, metrics.router, metrics.native, metrics.read,
          metrics.io_wall, metrics.io_overlap, metrics.io_wait,
          (unsigned long long)metrics.direct);
  }
  if (image_cursor != image_features.size())
    throw std::runtime_error("unconsumed prefill image features");
  export_append_state(state, begin, count);
  printf(
      "CHAT_PREFILL tokens=%zu blocks=%zu tail_valid=%d total_ms=%.6f blocks_ms=%.6f ar1_calls=0\n",
      count,
      (count + 127) / 128,
      valid_tokens,
      elapsed(append_started),
      total_ms);
}
