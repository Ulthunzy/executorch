#include "../chat/conversation.h"
#include "../runtime_utils.h"
#include <array>
#include <nlohmann/json.hpp>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <unistd.h>
#include <vector>

int main() {
  using namespace moe_chat;
  Options options;
  assert(options.thinking);
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
      {"generation",
       {{"enable_thinking", true},
        {"max_new_tokens", 64},
        {"temperature", 0.75},
        {"ignore_eos", true},
        {"stream", false}}}};
  apply_model_settings(options, config, {});
  assert(options.capacity == 2048 && options.thinking);
  assert(options.temperature == 0.75f && options.ignore_eos && !options.stream);
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
  auto half_state = config;
  half_state["cache"]["gdn_state_dtype"] = "float16";
  apply_model_settings(options, half_state, {});
  for (const char* field : {"kv_dtype", "conv_state_dtype"}) {
    invalid = half_state;
    invalid["cache"][field] = "float16";
    rejected = false;
    try { apply_model_settings(options, invalid, {}); }
    catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
  }
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
  nlohmann::json plans = {
      {"version", 2},
      {"prefill_pte", "prefill.pte"},
      {"ar_len", 128},
      {"capacity", 2048},
      {"max_prefill_tokens", 1024},
      {"prefill_order", "chunk_major"},
      {"model_configuration", config}};
  std::ofstream(root / "prefill/plans/PLANS.json") << plans;
  std::ofstream(root / "prefill/plans/prefill.pte") << "plan";
  std::string directory = root.string();
  const char* arguments[] = {
      "runner",
      "--model-root",
      directory.c_str(),
      "--tokenizer-path",
      "/tmp/tokenizer.json",
      "--profile",
      "--no-enable-thinking"};
  options = parse_options(7, const_cast<char**>(arguments));
  assert(options.capacity == 2048 && !options.thinking && options.profile);
  configure_environment(options, 128, 37);
  assert(std::string(getenv("EMBEDDING_BITS")) == "16");
  assert(std::string(getenv("KV_APPEND_ONLY")) == "0");
  assert(std::string(getenv("GDN_STATE_SWAP")) == "0");
#if SM_MOE_DSP_ARCH == 81
  assert(std::string(getenv("EXPERT_READ_CHUNK_KIB")) == "512");
  assert(std::string(getenv("DECODE_EXPERT_IO_WORKERS")) == "8");
#else
  assert(std::string(getenv("EXPERT_READ_CHUNK_KIB")) == "64");
  assert(std::string(getenv("DECODE_EXPERT_IO_WORKERS")) == "16");
#endif
  assert(std::string(getenv("PREFILL_TOKENS")) == "128");
  assert(std::string(getenv("PREFILL_VALID_TOKENS")) == "37");
  assert(std::string(getenv("MOE_PROFILE_TOTAL")) == "1");
  assert(std::string(getenv("PREFILL_INITIAL_IMAGE")) == "0");
  options.image_path = "/tmp/actual-image.jpg";
  configure_environment(options, 256);
  assert(std::string(getenv("PREFILL_INITIAL_IMAGE")) == "1");
  options.image_path.clear();
  options.interactive = true;
  options.vision_enabled = true;
  std::ofstream(root / "prefill/vision.bin") << "resident";
  configure_environment(options, 256);
  assert(std::string(getenv("VISION_INPUT")) == "1");
  assert(std::string(getenv("PREFILL_INITIAL_IMAGE")) == "0");
  const char* sampling[] = {
      "runner",
      "--model-root",
      directory.c_str(),
      "--tokenizer-path",
      "/tmp/tokenizer.json",
      "--temperature",
      "0.25",
      "--stream",
      "--no-ignore-eos",
      "--no-interactive",
      "--no-vision",
      "--prompt",
      "test"};
  options = parse_options(13, const_cast<char**>(sampling));
  assert(options.temperature == 0.25f && options.stream && !options.ignore_eos);
  assert(!options.interactive && !options.vision_enabled);
  const std::string disabled = (root / "thinking-disabled.json").string();
  auto disabled_config = config;
  disabled_config["generation"]["enable_thinking"] = false;
  std::ofstream(disabled) << disabled_config;
  const char* thinking[] = {
      "runner",
      "--model-root",
      directory.c_str(),
      "--tokenizer-path",
      "/tmp/tokenizer.json",
      "--model-config",
      disabled.c_str(),
      "--enable-thinking"};
  options = parse_options(8, const_cast<char**>(thinking));
  assert(options.thinking);
  assert(options.prefill_io_workers == 16);
  assert(options.runtime.decode_io_workers == 16);
  assert(options.runtime.read_chunk_kib == 64);

  const auto parse = [&](std::vector<std::string> extra) {
    std::vector<std::string> args = {
        "runner", "--model-root", directory, "--tokenizer-path",
        "/tmp/tokenizer.json", "--max-seq-len", "2048"};
    args.insert(args.end(), extra.begin(), extra.end());
    std::vector<char*> argv;
    for (auto& arg : args)
      argv.push_back(arg.data());
    return parse_options(static_cast<int>(argv.size()), argv.data());
  };
  const auto check_schedule = [&](const Options& parsed, int prefill,
                                  int decode, int chunk) {
    assert(parsed.prefill_io_workers == prefill);
    assert(parsed.runtime.decode_io_workers == decode);
    assert(parsed.runtime.read_chunk_kib == chunk);
    configure_environment(parsed, 128);
    assert(std::string(getenv("EXPERT_IO_WORKERS")) == std::to_string(prefill));
    assert(std::string(getenv("DECODE_EXPERT_IO_WORKERS")) ==
           std::to_string(decode));
    assert(std::string(getenv("EXPERT_READ_CHUNK_KIB")) == std::to_string(chunk));
  };
  const auto current_recipe =
      std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() /
      "configs/model_sm8850.json";
  nlohmann::json old_config;
  std::ifstream(current_recipe) >> old_config;
  old_config["schedule"]["cache_experts"] = 32;
  old_config["schedule"]["read_chunk_kib"] = 256;
  const auto recipe = root / "old-model.json";
  std::ofstream(recipe) << old_config;
  assert(old_config.at("schedule").at("io_workers") == 8);
  assert(old_config.at("schedule").at("read_chunk_kib") == 256);
  plans["model_configuration"] = old_config;
  std::ofstream(root / "prefill/plans/PLANS.json") << plans;
#if SM_MOE_DSP_ARCH == 81
  constexpr int default_prefill = 16, default_chunk = 512;
#else
  constexpr int default_prefill = 8, default_chunk = 256;
#endif
  check_schedule(parse({}), default_prefill, 8, default_chunk);
  check_schedule(parse({"--model-config", recipe.string()}), 8, 8, 256);
  check_schedule(parse({"--prefill-io-workers", "4"}), 4, 8, default_chunk);
  const std::vector<std::string> io_overrides = {
      "--prefill-io-workers", "4", "--decode-io-workers", "16",
      "--read-chunk-kib", "64"};
  check_schedule(parse(io_overrides), 4, 16, 64);
  auto explicit_overrides = io_overrides;
  explicit_overrides.insert(
      explicit_overrides.end(), {"--model-config", recipe.string()});
  check_schedule(parse(explicit_overrides), 4, 16, 64);

  for (bool configured : {true, false}) {
    if (!configured) {
      plans.erase("model_configuration");
      std::ofstream(root / "prefill/plans/PLANS.json") << plans;
      check_schedule(parse({}), default_prefill, 8, default_chunk);
      check_schedule(parse(io_overrides), 4, 16, 64);
    }
    for (const std::vector<std::string>& invalid_io : {
             std::vector<std::string>{"--decode-io-workers", "0"},
             {"--decode-io-workers", "12"},
             {"--read-chunk-kib", "0"},
             {"--read-chunk-kib", "3"},
             {"--prefill-io-workers", "17"}}) {
      rejected = false;
      try {
        parse(invalid_io);
      } catch (const std::runtime_error&) {
        rejected = true;
      }
      assert(rejected);
    }
  }
  std::filesystem::remove_all(root);
  std::cout << "runtime configuration and RoPE bounds tests passed\n";
}
