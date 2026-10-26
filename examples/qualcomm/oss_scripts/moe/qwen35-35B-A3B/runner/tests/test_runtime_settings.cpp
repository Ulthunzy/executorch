#include "../chat/conversation.h"
#include "../runtime_utils.h"
#include <array>
#include <nlohmann/json.hpp>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <unistd.h>

int main() {
  using namespace moe_chat;
  Options options;
  const nlohmann::json config = {
      {"version", 1},
      {"quantization", {{"embedding", "bf16"}}},
      {"context", {{"max_seq_len", 2048}}},
      {"cache", {{"append_only", false}, {"state_swap", false}}},
      {"schedule",
       {{"io_workers", 16},
        {"startup_prefetch", "none"},
        {"read_chunk_kib", 64},
        {"prefill_order", "chunk_major"}}},
      {"generation", {{"enable_thinking", true}, {"max_new_tokens", 64}}}};
  apply_model_settings(options, config, {});
  assert(options.capacity == 2048 && options.thinking);
  assert(options.runtime.embedding_bits == 16 && !options.runtime.state_swap);
  assert(
      !options.runtime.kv_append_only &&
      options.runtime.decode_io_workers == 16);
  options.capacity = 4096;
  options.thinking = false;
  apply_model_settings(
      options, config, {"--max-seq-len", "--no-enable-thinking"});
  assert(options.capacity == 4096 && !options.thinking);
  auto invalid = config;
  invalid["cache"]["kv_dtype"] = "uint8";
  bool rejected = false;
  try {
    apply_model_settings(options, invalid, {});
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  assert(rejected);
  validate_rope_tables(32768 * 128, 32768 * 128, 32768);
  for (const auto sizes :
       {std::array<size_t, 3>{2048 * 128, 2048 * 128, 4096},
        {2048 * 128, 2048 * 128 - 4, 1024},
        {0, 0, 1024}}) {
    rejected = false;
    try {
      validate_rope_tables(sizes[0], sizes[1], sizes[2]);
    } catch (const std::runtime_error&) {
      rejected = true;
    }
    assert(rejected);
  }
  const auto root = std::filesystem::path("/tmp") /
      ("qwen35-settings-test-" + std::to_string(getpid()));
  std::filesystem::create_directories(root / "prefill/plans");
  std::filesystem::create_directories(root / "decode");
  std::ofstream(root / "prefill/plans/PLANS.json") << nlohmann::json{
      {"version", 1},
      {"ar_len", 128},
      {"capacity", 2048},
      {"max_prefill_tokens", 1024},
      {"prefill_order", "chunk_major"},
      {"model_configuration", config}};
  std::ofstream(root / "prefill/plans/prefill-128.pte") << "plan";
  std::string directory = root.string();
  const char* arguments[] = {
      "runner",
      "--model-root",
      directory.c_str(),
      "--tokenizer-path",
      "/tmp/tokenizer.json",
      "--no-enable-thinking"};
  options = parse_options(6, const_cast<char**>(arguments));
  assert(options.capacity == 2048 && !options.thinking);
  configure_environment(options, 128, 37);
  assert(std::string(getenv("EMBEDDING_BITS")) == "16");
  assert(std::string(getenv("KV_APPEND_ONLY")) == "0");
  assert(std::string(getenv("GDN_STATE_SWAP")) == "0");
  assert(std::string(getenv("EXPERT_READ_CHUNK_KIB")) == "64");
  assert(std::string(getenv("DECODE_EXPERT_IO_WORKERS")) == "16");
  assert(std::string(getenv("PREFILL_TOKENS")) == "128");
  assert(std::string(getenv("PREFILL_VALID_TOKENS")) == "37");
  std::filesystem::remove_all(root);
  std::cout << "runtime configuration and RoPE bounds tests passed\n";
}
