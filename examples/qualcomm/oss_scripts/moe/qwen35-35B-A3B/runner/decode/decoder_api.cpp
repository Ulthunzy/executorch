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
  session.reset();
  module.reset();
  embedded_binaries.clear();
  failed = false;
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

extern "C" int qwen35_decode_set_inputs(
    const float* features,
    size_t feature_count,
    const float* cos,
    const float* sin,
    size_t table_count) try {
  if (!session || failed || session->in_step || !cos || !sin ||
      table_count * sizeof(float) != session->cos.size() ||
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
  if (!session || !module || failed || !logits || count != 248320)
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
  for (int layer = 0; layer < 40; ++layer) {
    if (session->attention[layer]->linear)
      gdn += metrics.execution[layer][0];
    else
      full_attention += metrics.execution[layer][0];
  }
  // The fused graph's Router/shared compute belongs to attention_ms;
  // router_ms measures only output handoff when fused_router_layers=40.
  printf(
      "CHAT_DECODE_STEP position=%lld input_id=%lld total_ms=%.6f attention_ms=%.6f gdn_graph_ms=%.6f full_attention_graph_ms=%.6f router_ms=%.6f read_ms=%.6f native_ms=%.6f feedback_ms=%.6f head_ms=%.6f embedding_ms=%.6f direct_bytes=%llu cache_hits=%llu cache_requests=%llu\n",
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
      (unsigned long long)metrics.requests);
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
  if (resources() != lifetime)
    throw std::runtime_error("reset changed model resources");
  return 0;
} catch (const std::exception& error) {
  failed = true;
  last_error = error.what();
  return 1;
}
