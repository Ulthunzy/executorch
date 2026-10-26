#pragma once

#include <nlohmann/json_fwd.hpp>
#include <set>
#include <string>

namespace moe_chat {
struct Options;

struct RuntimeSettings {
  int embedding_bits = 4;
  int expert_bits = 2;
  int decode_io_workers = 8;
  int startup_io_workers = 4;
  int read_chunk_kib = 256;
  bool prefill_async_io = true;
  bool kv_append_only = true;
  bool state_swap = true;
  bool hit_first = true;
  bool ready_io = true;
  bool overlap_feedback = true;
  std::string startup_prefetch = "overlap";
  std::string context_load_mode = "auto";
};

void apply_model_settings(
    Options& options,
    const nlohmann::json& config,
    const std::set<std::string>& overrides);
} // namespace moe_chat
