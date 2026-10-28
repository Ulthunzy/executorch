#include "settings.h"
#include "conversation.h"

#include <nlohmann/json.hpp>
#include <stdexcept>

namespace moe_chat {
void validate_runtime_settings(const RuntimeSettings& runtime) {
  if ((!runtime.prefill_async_io && runtime.startup_prefetch != "none") ||
      (runtime.ready_io && !runtime.hit_first) ||
      (runtime.startup_prefetch != "none" &&
       runtime.startup_prefetch != "sync" &&
       runtime.startup_prefetch != "overlap") ||
      runtime.startup_io_workers < 1 || runtime.startup_io_workers > 16 ||
      (runtime.decode_io_workers != 4 && runtime.decode_io_workers != 8 &&
       runtime.decode_io_workers != 16) ||
      (runtime.read_chunk_kib != 64 && runtime.read_chunk_kib != 128 &&
       runtime.read_chunk_kib != 256 && runtime.read_chunk_kib != 512 &&
       runtime.read_chunk_kib != 1024) ||
      (runtime.context_load_mode != "auto" &&
       runtime.context_load_mode != "async" &&
       runtime.context_load_mode != "serial"))
    throw std::runtime_error("Invalid expert IO or cache schedule");
}

void apply_model_settings(
    Options& options,
    const nlohmann::json& config,
    const std::set<std::string>& overrides) {
  if (!config.is_object() || config.value("version", 1) != 1)
    throw std::runtime_error("Expected a version 1 model configuration");
  const auto quant = config.value("quantization", nlohmann::json::object());
  const auto cache = config.value("cache", nlohmann::json::object());
  const auto context = config.value("context", nlohmann::json::object());
  const auto generation = config.value("generation", nlohmann::json::object());
  const auto schedule = config.value("schedule", nlohmann::json::object());
  auto& runtime = options.runtime;
  const auto embedding = quant.value("embedding", "int4_g32");
  if (embedding != "int4_g32" && embedding != "bf16")
    throw std::runtime_error("Unsupported embedding recipe");
  runtime.embedding_bits = embedding == "int4_g32" ? 4 : 16;
  runtime.expert_bits = quant.value("expert_bits", 2);
  if (runtime.expert_bits != 2 ||
      quant.value("expert_activation_bits", 8) != 8 ||
      quant.value("expert_gu_group", 1024) != 1024 ||
      quant.value("expert_down_group", 512) != 512)
    throw std::runtime_error(
        "Native experts require N2/A8 with groups 1024/512");
  for (const char* field : {"kv_dtype", "conv_state_dtype"})
    if (cache.value(field, "uint16") != "uint16")
      throw std::runtime_error("Native KV/conv interfaces require UINT16");
  const auto gdn_dtype = cache.value("gdn_state_dtype", "uint16");
  if (gdn_dtype != "uint16" && gdn_dtype != "float16")
    throw std::runtime_error("GDN state requires UINT16 or typed FP16");
  runtime.kv_append_only = cache.value("append_only", true);
  runtime.state_swap = cache.value("state_swap", true);
  runtime.prefill_async_io = schedule.value("prefill_async_io", true);
  runtime.startup_prefetch = schedule.value("startup_prefetch", "overlap");
  runtime.startup_io_workers = schedule.value("startup_io_workers", 4);
  if (!overrides.count("--decode-io-workers"))
    runtime.decode_io_workers = schedule.value("io_workers", 8);
  if (!overrides.count("--read-chunk-kib"))
    runtime.read_chunk_kib = schedule.value("read_chunk_kib", 256);
  runtime.hit_first = schedule.value("decode_hit_first", true);
  runtime.ready_io = schedule.value("decode_ready_io", true);
  runtime.overlap_feedback = schedule.value("overlap_feedback", true);
  runtime.context_load_mode = schedule.value("context_load_mode", "auto");
  validate_runtime_settings(runtime);
  if (!overrides.count("--max-seq-len"))
    options.capacity = context.value("max_seq_len", options.capacity);
  if (!overrides.count("--prefill-order"))
    options.prefill_order =
        schedule.value("prefill_order", options.prefill_order);
  if (!overrides.count("--cache-experts"))
    options.cache_experts =
        schedule.value("cache_experts", options.cache_experts);
  if (!overrides.count("--prefill-io-workers"))
    options.prefill_io_workers =
        schedule.value("io_workers", options.prefill_io_workers);
  if (!overrides.count("--max-new-tokens"))
    options.max_new_tokens =
        generation.value("max_new_tokens", options.max_new_tokens);
  if (!overrides.count("--temperature"))
    options.temperature = generation.value("temperature", options.temperature);
  if (!overrides.count("--ignore-eos") && !overrides.count("--no-ignore-eos"))
    options.ignore_eos = generation.value("ignore_eos", options.ignore_eos);
  if (!overrides.count("--stream") && !overrides.count("--no-stream"))
    options.stream = generation.value("stream", options.stream);
  if (!overrides.count("--interactive") && !overrides.count("--no-interactive"))
    options.interactive = generation.value("interactive", options.interactive);
  if (!overrides.count("--system-prompt"))
    options.system_prompt =
        generation.value("system_prompt", options.system_prompt);
  if (!overrides.count("--enable-thinking") &&
      !overrides.count("--no-enable-thinking"))
    options.thinking = generation.value("enable_thinking", options.thinking);
}
} // namespace moe_chat
