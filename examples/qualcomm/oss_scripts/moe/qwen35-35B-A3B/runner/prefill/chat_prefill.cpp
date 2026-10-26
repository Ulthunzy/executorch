#include "prefill_session.h"

void PrefillSession::prepare_decode() {
  finish_memo_copy();
  if (!route_reads.empty())
    throw std::runtime_error("unfinished prefill reads at decode handoff");
  if (sequence_tokens > 128) {
    core.release_io();
    check(integer_rpc_configure(core.handle, 128), "configure chat AR128");
    core.inbuf[0] = shared_io->hidden;
    core.inbuf[1] = shared_io->ids;
    core.inbuf[2] = shared_io->scores;
    core.inbuf[3] = shared_io->shared;
    core.outbuf[0] = attention[0]->hidden_alias;
    core.residual = shared_io->residual;
    layer_memo.reset();
    for (auto& buffer : expert_slots)
      core.map_buffer(buffer);
    for (int i = 0; i < 4; ++i)
      core.map_buffer(core.inbuf[i]);
    core.map_buffer(core.outbuf[0]);
    core.map_buffer(core.residual);
    if (async_io && pipeline_banks.empty())
      for (int i = 0; i < 2; ++i) {
        pipeline_banks.push_back(
            std::make_shared<Buffer>(runtime, 4096 + capacity * stride, false));
        core.map_buffer(pipeline_banks.back());
      }
  }
  sequence_tokens = 128;
  memo_layer = -1;
  staged_router_layer = -1;
  std::vector<int32_t>().swap(sequence_routes);
  std::vector<int32_t>().swap(sequence_seeds);
  std::vector<uint64_t>().swap(sequence_stamps);
  std::vector<int32_t>().swap(staged_ids);
  std::vector<float>().swap(staged_scores);
  std::vector<float>().swap(staged_shared);
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
  if (!tokens || !count || !logits || count > size_t(state.capacity) ||
      state.position < 0 || state.position + count >= size_t(state.capacity) ||
      state.capacity != logical_capacity || state.slots != capacity ||
      sequence_tokens != 128 || layer_memo || !core.handle ||
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
      const auto q = FIELD(tensor, quantizeParams).scaleOffsetEncoding;
      QuantizedCacheView destination{
          static_cast<uint16_t*>(a.cache_data(c)),
          Graph::elements(tensor),
          q.scale,
          q.offset};
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
    total_ms += elapsed(started);
  }
  if (image_cursor != image_features.size())
    throw std::runtime_error("unconsumed prefill image features");
  // Only new KV crosses back: historical decode codes retain their original
  // grid and never undergo an additional round trip through prefill encoding.
  for (int layer = 0; layer < 40; ++layer) {
    auto& a = *attention[layer];
    for (int c = 0; c < 2; ++c) {
      const auto& tensor = a.cache_owner().inputs[a.cache_input[c]];
      const auto q = FIELD(tensor, quantizeParams).scaleOffsetEncoding;
      QuantizedCacheView source{
          static_cast<uint16_t*>(a.export_data(c)),
          a.export_bytes(c) / 2,
          q.scale,
          q.offset};
      auto destination = state.layer[layer].cache[c];
      const auto layout = a.linear ? CacheLayout::State
          : c                      ? CacheLayout::Value
                                   : CacheLayout::Key;
      if (!a.linear && !state.append_only) {
        // Move old sliding history once before inserting this turn's suffix.
        std::vector<uint16_t> old(
            destination.data, destination.data + destination.elements);
        copy_cache_range(
            {old.data(), old.size(), destination.scale, destination.offset},
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
  printf(
      "CHAT_PREFILL tokens=%zu blocks=%zu tail_valid=%d total_ms=%.6f blocks_ms=%.6f ar1_calls=0\n",
      count,
      (count + 127) / 128,
      valid_tokens,
      elapsed(append_started),
      total_ms);
}
