#include "../decode_api.h"
#include "decoder_state.h"

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/runtime/platform/runtime.h>

using namespace executorch::runtime;

namespace {
std::unique_ptr<executorch::extension::Module> module;
std::string last_error;
bool failed = false;
bool prefill_interrupted = false;
std::array<uint64_t, 6> lifetime{};

std::array<uint64_t, 6> resources() {
  const auto& rt = session->runtime;
  return {
      rt.context_creates,
      rt.context_frees,
      rt.memory_registers,
      rt.memory_deregisters,
      rt.buffer_allocations,
      rt.buffer_frees};
}
} // namespace

extern "C" const char* qwen35_decode_error() {
  return last_error.c_str();
}

extern "C" void qwen35_decode_close() {
  try {
    if (const char* directory = getenv("DECODE_VERIFY_ROOT");
        directory && session && session->metrics.calls[4]) {
      const auto root = std::string(directory);
      const int input = session->head->index(true, 0);
      const int output = session->head->index(false, 0);
      save(root + "/hidden.f32", session->head->input(input), 2048 * 4);
      save(root + "/logits.f32", session->head->output(output), 248320 * 4);
      for (int layer = 0; layer < 40; ++layer)
        for (int cache = 0; cache < 2; ++cache)
          session->attention[layer]->save_cache(
              root + "/layer" + std::to_string(layer) + "-cache" +
                  std::to_string(cache) + ".u16",
              cache);
    }
  } catch (const std::exception& error) {
    fprintf(stderr, "decode verification snapshot failed: %s\n", error.what());
  }
  session.reset();
  module.reset();
  embedded_binaries.clear();
  failed = false;
  prefill_interrupted = false;
}

extern "C" int qwen35_decode_open(const char* root, int cache_mib) try {
  if (session || module) {
    last_error = "decode session is already open";
    return 1;
  }
  if (!root || !*root || cache_mib < 0 || cache_mib > 4096)
    throw std::runtime_error("invalid decode session settings");
  runtime_init();
  model_root = root;
  module = std::make_unique<executorch::extension::Module>(
      model_root + "/decode.pte",
      executorch::extension::Module::LoadMode::Mmap);
  const auto status = module->load_method("decode_forward");
  if (status != Error::Ok)
    throw std::runtime_error("decode PTE load " + std::to_string(int(status)));
  session = std::make_unique<DecodeSession>(model_root, cache_mib);
  for (const auto& attention : session->attention)
    if (!attention->linear &&
        attention->history + 1 != session->context_capacity)
      throw std::runtime_error("decode capacity differs from compiled graph");
  // Chat reset starts an empty conversation; it never restores this prefix.
  for (auto& attention : session->attention)
    for (auto& snapshot : attention->initial)
      std::vector<uint16_t>().swap(snapshot);
  lifetime = resources();
  int fused_routers = 0;
  for (const auto& attention : session->attention)
    fused_routers += attention->output.size() == 7;
  printf("CHAT_DECODE_READY layers=40 fused_router_layers=%d\n", fused_routers);
  last_error.clear();
  failed = false;
  return 0;
} catch (const std::exception& error) {
  last_error = error.what();
  qwen35_decode_close();
  return 1;
}

extern "C" int qwen35_decode_position() {
  return session ? session->position : -1;
}

extern "C" int qwen35_decode_export_state(Qwen35SessionState* state) try {
  if (!session || failed || prefill_interrupted || session->in_step || !state)
    throw std::runtime_error("decode state is not available for prefill");
  for (const auto& pending : session->pending)
    if (!pending.empty())
      throw std::runtime_error("unfinished decode expert prefetch");
  check(session->core.stop_service(), "pause decode mailbox");
  session->core.end_output_read();
  state->position = session->position;
  state->capacity = session->context_capacity;
  state->append_only = session->attention[3]->append_kv;
  session->cache->export_state(*state);
  for (int layer = 0; layer < 40; ++layer) {
    for (int c = 0; c < 2; ++c)
      state->layer[layer].cache[c] = session->attention[layer]->cache_view(c);
    std::copy(
        session->previous_routes[layer].begin(),
        session->previous_routes[layer].end(),
        state->layer[layer].routes);
  }
  return 0;
} catch (const std::exception& error) {
  last_error = error.what();
  return 1;
}

extern "C" int qwen35_decode_import_state(const Qwen35SessionState* state) try {
  if (!session || failed || session->in_step || !state ||
      state->capacity != session->context_capacity ||
      (!state->aborted &&
       (state->position < 1 || state->position >= state->capacity)))
    throw std::runtime_error("invalid prefill state handoff");
  session->cache->import_state(*state);
  if (state->aborted) {
    prefill_interrupted = true;
    return 0;
  }
  for (int layer = 0; layer < 40; ++layer) {
    std::copy(
        state->layer[layer].routes,
        state->layer[layer].routes + 8,
        session->previous_routes[layer].begin());
    session->attention[layer]->last_position = state->position - 1;
  }
  session->position = state->position;
  session->prefix_tokens = state->position;
  session->next_layer = 0;
  session->metrics = {};
  session->image_cursor = session->image_features.size();
  if (resources() != lifetime)
    throw std::runtime_error("prefill changed decode model resources");
  return 0;
} catch (const std::exception& error) {
  failed = true;
  last_error = error.what();
  return 1;
}

extern "C" int qwen35_decode_set_inputs(
    const float* features,
    size_t feature_count,
    const float* cos,
    const float* sin,
    size_t table_count) try {
  if (!session || failed || session->in_step || !cos || !sin ||
      table_count != session->cos.size() / sizeof(float) ||
      session->cos.size() != session->sin.size() || feature_count % 2048 ||
      (feature_count && !features) ||
      session->image_cursor != session->image_features.size())
    throw std::runtime_error(
        "Invalid multimodal inputs or unfinished image span");
  for (size_t i = 0; i < table_count; ++i)
    if (!std::isfinite(cos[i]) || !std::isfinite(sin[i]))
      throw std::runtime_error("Nonfinite RoPE table");
  for (size_t i = 0; i < feature_count; ++i)
    if (!std::isfinite(features[i]))
      throw std::runtime_error("Nonfinite image embedding");
  if (feature_count)
    session->image_features.assign(features, features + feature_count);
  else
    session->image_features.clear();
  session->image_cursor = 0;
  memcpy(session->cos.data(), cos, session->cos.size());
  memcpy(session->sin.data(), sin, session->sin.size());
  return 0;
} catch (const std::exception& error) {
  last_error = error.what();
  return 1;
}

extern "C" int
qwen35_decode_step(int64_t token, float* logits, size_t count) try {
  if (!session || !module || failed || prefill_interrupted || !logits ||
      count != 248320)
    throw std::runtime_error("invalid decode step or failed session");
  const auto started = Clock::now();
  int64_t position = session->position;
  auto input = executorch::extension::make_tensor_ptr(
      {1, 1}, &token, executorch::aten::ScalarType::Long);
  auto position_input = executorch::extension::make_tensor_ptr(
      {1}, &position, executorch::aten::ScalarType::Long);
  auto result = module->execute(
      "decode_forward", std::vector<EValue>{*input, *position_input});
  if (!result.ok() || result->size() != 2)
    throw std::runtime_error("decode execute/output failure");
  const auto& metrics = session->metrics;
  if (metrics.calls[0] != 1 || metrics.calls[1] != 40 ||
      metrics.calls[2] != 40 || metrics.calls[3] != 40 ||
      metrics.calls[4] != 1 || session->position != position + 1)
    throw std::runtime_error("incomplete decode step");
  if (resources() != lifetime)
    throw std::runtime_error("context/buffer lifetime changed during chat");
  const auto* values = result->at(0).toTensor().const_data_ptr<float>();
  for (size_t i = 0; i < count; ++i)
    if (!std::isfinite(values[i]))
      throw std::runtime_error("nonfinite logits");
  std::copy(values, values + count, logits);
  double gdn = 0, full_attention = 0;
  double pread_worker = 0, copy_worker = 0, publish_worker = 0;
  int read_calls = 0;
  for (int layer = 0; layer < 40; ++layer) {
    if (session->attention[layer]->linear)
      gdn += metrics.execution[layer][0];
    else
      full_attention += metrics.execution[layer][0];
    pread_worker += metrics.io[layer].pread_worker;
    copy_worker += metrics.io[layer].copy_worker;
    publish_worker += metrics.io[layer].publish_worker;
    read_calls += metrics.io[layer].read_calls;
  }
  // The fused graph's Router/shared compute belongs to attention_ms;
  // router_ms measures only output handoff when fused_router_layers=40.
  printf(
      "CHAT_DECODE_STEP position=%lld input_id=%lld total_ms=%.6f attention_ms=%.6f gdn_graph_ms=%.6f full_attention_graph_ms=%.6f router_ms=%.6f read_ms=%.6f native_ms=%.6f feedback_ms=%.6f head_ms=%.6f embedding_ms=%.6f direct_bytes=%llu cache_hits=%llu cache_requests=%llu pread_worker_ms=%.6f copy_worker_ms=%.6f publish_worker_ms=%.6f read_calls=%d\n",
      (long long)position,
      (long long)token,
      elapsed(started),
      metrics.attention,
      gdn,
      full_attention,
      metrics.router,
      metrics.read,
      metrics.native,
      metrics.feedback,
      metrics.head,
      metrics.embedding,
      (unsigned long long)metrics.direct,
      (unsigned long long)metrics.hits,
      (unsigned long long)metrics.requests,
      pread_worker,
      copy_worker,
      publish_worker,
      read_calls);
  return 0;
} catch (const std::exception& error) {
  failed = true;
  last_error = error.what();
  return 1;
}

extern "C" int qwen35_decode_reset() try {
  if (!session || failed || session->in_step)
    throw std::runtime_error("decode session cannot be reset");
  session->core.end_output_read();
  for (const auto& tasks : session->pending)
    if (!tasks.empty())
      throw std::runtime_error("unfinished expert prefetch during reset");
  for (auto& attention : session->attention)
    attention->clear();
  session->prefix_tokens = 0;
  session->position = 0;
  session->next_layer = 0;
  session->metrics = {};
  session->image_features.clear();
  session->image_cursor = 0;
  prefill_interrupted = false;
  if (resources() != lifetime)
    throw std::runtime_error("reset changed model resources");
  return 0;
} catch (const std::exception& error) {
  failed = true;
  last_error = error.what();
  return 1;
}
