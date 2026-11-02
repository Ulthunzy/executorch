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
  const auto append_started = Clock::now();
  if (initial_module) {
    initial_prefill(tokens, count, features, feature_count, rotary_cos,
                    rotary_sin, table_count, state, logits, stopped);
    return;
  }
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
      clear_cache(destination);
      const auto& source = state.layer[layer].cache[c];
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
          a.linear ? source.elements : state.position);
      if (!a.linear)
        memcpy(
            a.cache_host[c].data(), destination.data, destination.elements * 2);
    }
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

void PrefillSession::initial_prefill(
    const int64_t* tokens, size_t count, const float* features,
    size_t feature_count, const float* rotary_cos, const float* rotary_sin,
    size_t table_count, Qwen35SessionState& state, float* logits,
    const std::function<bool()>& stopped) {
  if (!tokens || !count || !logits || state.position != 0 ||
      count >= size_t(state.capacity) || state.capacity != logical_capacity ||
      state.slots != capacity || feature_count % 2048 ||
      (feature_count && !features) || !rotary_cos || !rotary_sin ||
      table_count * sizeof(float) != cos.size() || cos.size() != sin.size())
    throw std::runtime_error("invalid first resident prefill request");
  if (stopped && stopped())
    throw std::runtime_error("prefill interrupted; reset required");
  const auto started = Clock::now();
  const size_t first_count = std::min(count, size_t(workspace_tokens));
  const int plan_tokens = int((first_count + 127) / 128 * 128);
  if (initial_attempted)
    reset();
  initial_attempted = true;
  sequence_tokens = plan_tokens;
  sequence_start = 0;
  sequence_valid_tokens = first_count;
  memcpy(cos.data(), rotary_cos, cos.size());
  memcpy(sin.data(), rotary_sin, sin.size());
  image_features.clear();
  if (feature_count)
    image_features.assign(features, features + feature_count);
  std::vector<int64_t> padded(tokens, tokens + first_count);
  padded.resize(plan_tokens, 248044);
  auto input = executorch::extension::make_tensor_ptr(
      {1, plan_tokens}, padded.data(), executorch::aten::ScalarType::Long);
  auto result = initial_module->execute(
      "prefill_forward", std::vector<executorch::runtime::EValue>{*input});
  if (!result.ok())
    throw std::runtime_error("First prefill PTE execution failed");
  memcpy(logits, result->at(0).toTensor().const_data_ptr<float>(), 248320 * sizeof(float));
  export_append_state(state, 0, first_count);
  const size_t consumed_features = image_cursor;
  initial_module = nullptr;
  prepare_decode();
  if (count > first_count) {
    append(tokens + first_count, count - first_count,
           features ? features + consumed_features : nullptr,
           feature_count - consumed_features, rotary_cos, rotary_sin,
           table_count, state, logits, stopped);
  } else if (consumed_features != feature_count) {
    throw std::runtime_error("unconsumed first prefill image features");
  }
  printf("CHAT_INITIAL_PREFILL tokens=%zu plan_tokens=%d total_ms=%.6f model_load=0\n",
         count, plan_tokens, elapsed(started));
}

void PrefillSession::export_append_state(
    Qwen35SessionState& state, int begin, size_t count) {
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
          a.linear ? 0 : a.history + 128 - count,
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
