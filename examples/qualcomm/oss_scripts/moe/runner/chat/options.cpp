#include "conversation.h"
#include "../prefill_schedule.h"

#include <cstdlib>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace moe_chat {
namespace {
int integer_option(const std::string& value) {
  size_t used = 0;
  const int result = std::stoi(value, &used);
  if (used != value.size())
    throw std::runtime_error("Invalid integer: " + value);
  return result;
}
} // namespace

Options parse_options(int argc, char** argv) {
  Options options;
  std::set<std::string> overrides;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    overrides.insert(flag);
    if (flag == "--help") {
      printf(
          "qwen35_moe_runner --model-root DIR_OR_PTE [--tokenizer-path tokenizer.json]\n"
          "  [--package-cache DIR] [--library-directory DIR]\n"
          "  [--prompt TEXT ...] [--interactive] [--system-prompt TEXT]\n"
          "  [--image-path JPG_OR_PNG] (image files are decoded on device)\n"
          "  [--max-new-tokens 128] [--temperature 0] [--enable-thinking|--no-enable-thinking]\n"
          "  [--stream|--no-stream] [--vision|--no-vision]\n"
          "  [--prompt-tokens N] [--ignore-eos] (fixed-length performance runs)\n"
          "  [--max-seq-len 1152] [--cache-experts 32] [--shared-model DIR]\n"
          "  [--prefill-order layer_major] (chunk_major is deprecated and uses layer_major)\n"
          "  [--prefill-io-workers N] [--decode-io-workers 4|8|16]\n"
          "  [--read-chunk-kib 64|128|256|512|1024]\n"
          "  [--model-config FILE] (deployed plan settings; V81 defaults to IO 16/8, chunk 512)\n"
          "  [--log-path FILE] [--tokenize-only]\n"
          "  [--profile] (cumulative DSP timings; requires a profiling build)\n"
          "Interactive commands: /image PATH QUESTION, /reset, /exit.\n");
      options.help_requested = true;
      return options;
    }
    if (flag == "--interactive")
      options.interactive = true;
    else if (flag == "--no-interactive")
      options.interactive = false;
    else if (flag == "--enable-thinking")
      options.thinking = true;
    else if (flag == "--no-enable-thinking")
      options.thinking = false;
    else if (flag == "--tokenize-only")
      options.tokenize_only = true;
    else if (flag == "--ignore-eos")
      options.ignore_eos = true;
    else if (flag == "--profile")
      options.profile = true;
    else if (flag == "--no-ignore-eos")
      options.ignore_eos = false;
    else if (flag == "--stream" || flag == "--no-stream")
      options.stream = flag == "--stream";
    else if (flag == "--vision" || flag == "--no-vision")
      options.vision_enabled = flag == "--vision";
    else {
      if (i + 1 == argc)
        throw std::runtime_error("Missing value for " + flag);
      const std::string value = argv[++i];
      if (flag == "--model-family")
        options.model_family = value;
      else if (flag == "--model-root")
        options.model_root = value;
      else if (flag == "--tokenizer-path")
        options.tokenizer_path = value;
      else if (flag == "--shared-model")
        options.shared_model = value;
      else if (flag == "--package-cache")
        options.package_cache = value;
      else if (flag == "--library-directory")
        options.library_directory = value;
      else if (flag == "--model-config")
        options.model_config = value;
      else if (flag == "--prompt")
        options.prompts.push_back(value);
      else if (flag == "--image-path")
        options.image_path = value;
      else if (flag == "--prefill-order")
        options.prefill_order = value;
      else if (flag == "--system-prompt")
        options.system_prompt = value;
      else if (flag == "--log-path")
        options.log_path = value;
      else if (flag == "--max-new-tokens")
        options.max_new_tokens = integer_option(value);
      else if (flag == "--prompt-tokens")
        options.prompt_tokens = integer_option(value);
      else if (flag == "--max-seq-len")
        options.capacity = integer_option(value);
      else if (flag == "--cache-experts")
        options.cache_experts = integer_option(value);
      else if (flag == "--prefill-io-workers")
        options.prefill_io_workers = integer_option(value);
      else if (flag == "--decode-io-workers")
        options.runtime.decode_io_workers = integer_option(value);
      else if (flag == "--read-chunk-kib")
        options.runtime.read_chunk_kib = integer_option(value);
      else if (flag == "--temperature") {
        size_t used = 0;
        options.temperature = std::stof(value, &used);
        if (used != value.size())
          throw std::runtime_error("Invalid temperature: " + value);
      } else
        throw std::runtime_error("Unknown argument: " + flag);
    }
  }
  if (std::filesystem::is_regular_file(options.model_root)) {
    const auto assets = std::filesystem::path(
        unpack_model_package(options.model_root, options.package_cache));
    options.model_root = (assets / "model").string();
    options.shared_model = (assets / "shared-model").string();
    options.tokenizer_path = (assets / "tokenizer.json").string();
    options.package_lock = lock_package_model(options.model_root);
    if (options.library_directory.empty())
      options.library_directory =
          std::filesystem::canonical("/proc/self/exe").parent_path().string();
    link_model_libraries(options.model_root, options.library_directory);
  }
  if (options.model_root.empty() || options.tokenizer_path.empty())
    throw std::runtime_error("--model-root and --tokenizer-path are required");
  if (!options.vision_enabled && !options.image_path.empty())
    throw std::runtime_error("--image-path requires vision");
  options.model_root = std::filesystem::canonical(options.model_root).string();
  if (!options.shared_model.empty())
    options.shared_model =
        std::filesystem::path(options.shared_model).lexically_normal().string();
  nlohmann::json plans;
  std::ifstream manifest(options.model_root + "/prefill/plans/PLANS.json");
  if (!manifest)
    throw std::runtime_error(
        "Missing prefill plan bundle; redeploy with run_chat.py");
  manifest >> plans;
  nlohmann::json configuration =
      plans.value("model_configuration", nlohmann::json::object());
  if (!options.model_config.empty()) {
    std::ifstream settings(options.model_config);
    if (!settings)
      throw std::runtime_error("Cannot open model configuration");
    settings >> configuration;
  }
#if defined(SM_MOE_DSP_ARCH)
  const auto target = configuration.value("target", nlohmann::json::object());
  if (target.contains("dsp_arch") &&
      target.at("dsp_arch") != "v" + std::to_string(SM_MOE_DSP_ARCH))
    throw std::runtime_error("Model DSP architecture differs from this runtime");
#endif
  options.model_family = configuration.value("model_family", options.model_family);
  try { moe::create_model_adapter(options.model_family); }
  catch (const std::exception&) { throw std::runtime_error("Unsupported model_family: " + options.model_family); }
  if (!configuration.empty())
    apply_model_settings(options, configuration, overrides);
  else if (
      !std::filesystem::is_regular_file(
          options.model_root + "/prefill/embedding.int4") &&
      std::filesystem::is_regular_file(
          options.model_root + "/prefill/embedding.bf16"))
    options.runtime.embedding_bits = 16;
#if defined(SM_MOE_DSP_ARCH) && SM_MOE_DSP_ARCH == 81
  if (!overrides.count("--model-config")) {
    if (!overrides.count("--prefill-io-workers"))
      options.prefill_io_workers = 16;
    if (!overrides.count("--decode-io-workers"))
      options.runtime.decode_io_workers = 8;
    if (!overrides.count("--read-chunk-kib"))
      options.runtime.read_chunk_kib = 512;
  }
#endif
  validate_runtime_settings(options.runtime);
  if ((plans.at("version") != 2 && plans.at("version") != 3) ||
      plans.value("prefill_pte", "") != "prefill.pte" ||
      plans.at("ar_len") != 128 || plans.at("capacity") != options.capacity)
    throw std::runtime_error(
        "Prefill plan bundle does not match the compiled context");
  if (options.prefill_order.empty())
    options.prefill_order = plans.at("prefill_order");
  options.prefill_workspace_tokens = plans.at("version") == 3
      ? plans.at("workspace_tokens").get<int>()
      : plans.value("max_prefill_tokens", 1024);
  if (options.runtime.context_load_mode == "serial")
    throw std::runtime_error(
        "Shared contexts require batched initialization; serial loading is unsupported");
  if (options.prefill_order != "layer_major" &&
      options.prefill_order != "chunk_major")
    throw std::runtime_error("Invalid prefill order");
  if (options.prefill_order == "chunk_major") {
    fprintf(
        stderr,
        "DEPRECATED: chunk_major will be removed in a future release; using the unified layer_major scheduler\n");
    options.prefill_order = "layer_major";
  }
  options.tokenizer_path = std::filesystem::absolute(options.tokenizer_path)
                               .lexically_normal()
                               .string();
  if (options.max_new_tokens < 1 || options.max_new_tokens > options.capacity ||
      options.capacity < 1024 || options.capacity > 32768 ||
      options.capacity % 128 || options.prefill_workspace_tokens < 128 ||
      options.prefill_workspace_tokens > 1024 ||
      options.prefill_workspace_tokens % 128 || options.prompt_tokens < 0 ||
      options.prompt_tokens >= options.capacity ||
      options.prompt_tokens % 128 ||
      (options.cache_experts != 16 && options.cache_experts != 32 &&
       options.cache_experts != 64) ||
      options.prefill_io_workers < 1 || options.prefill_io_workers > 16 ||
      !std::isfinite(options.temperature) || options.temperature < 0)
    throw std::runtime_error(
        "Invalid generation, context or expert cache setting");
  if (options.shared_model.empty())
    options.shared_model =
        (std::filesystem::path(options.model_root).parent_path() /
         "shared-model")
            .string();
  else
    options.shared_model = std::filesystem::absolute(options.shared_model)
                               .lexically_normal()
                               .string();
  if (options.log_path.empty())
    options.log_path = options.model_root + "/chat.log";
  if (options.prompts.empty())
    options.interactive = true;
  return options;
}

void configure_environment(
    const Options& o,
    int prefix_tokens,
    int valid_tokens) {
  setenv("MOE_MODEL_FAMILY", o.model_family.c_str(), 1);
  if (valid_tokens < 0)
    valid_tokens = prefix_tokens;
  const std::string prefill = o.model_root + "/prefill",
                    decode = o.model_root + "/decode";
  const int plan_tokens =
      prefill_plan_tokens(prefix_tokens, o.prefill_workspace_tokens);
  const std::string plan = prefill + "/plans/prefill.pte";
  if (!o.tokenize_only && !std::filesystem::is_regular_file(plan))
    throw std::runtime_error("Missing prefill plan: " + plan);
  // Interactive sessions keep vision resident for later /image turns.
  const bool vision_enabled = o.vision_enabled &&
      (!o.image_path.empty() || o.interactive) &&
      std::filesystem::is_regular_file(prefill + "/vision.bin");
  printf(
      "CHAT_PREFILL_PLAN order=%s prompt_tokens=%d plan_tokens=%d path=%s vision_enabled=%d\n",
      "layer_major",
      prefix_tokens,
      plan_tokens,
      plan.c_str(),
      vision_enabled);
  for (const auto& item : std::vector<std::pair<std::string, std::string>>{
           {"CHAT_PREFILL_PLAN", plan},
           {"LD_LIBRARY_PATH", prefill},
           {"MOE_QNN_LIBRARY_PATH", prefill},
           {"ADSP_LIBRARY_PATH",
            prefill + ";" + decode +
                ";/odm/lib64/rfsa;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/dsp"},
           {"SHARED_MODEL_ROOT", o.shared_model},
           {"SHARED_OP_PACKAGE", decode + "/libet_qnnop.so"},
           {"HANDOFF_DECODE_ROOT", decode},
           {"HANDOFF_MAX_TOKENS", std::to_string(o.max_new_tokens)},
           {"HANDOFF_EOS_IDS", "248046,248044"},
           {"PREFILL_TOKENS", std::to_string(prefix_tokens)},
           {"PREFILL_VALID_TOKENS", std::to_string(valid_tokens)},
           {"DECODE_CONTEXT_CAPACITY", std::to_string(o.capacity)},
           {"EXPERT_CACHE_SLOTS", std::to_string(o.cache_experts)},
           {"EMBEDDING_BITS", std::to_string(o.runtime.embedding_bits)},
           {"EXPERT_BITS", std::to_string(o.runtime.expert_bits)},
           {"NATIVE_MAPPED_IO", "1"},
           {"NATIVE_RESIDUAL", "1"},
           {"REUSE_EXPERTS", "1"},
           {"INHERIT_EXPERT_CACHE", "1"},
           {"ASYNC_EXPERT_IO", o.runtime.prefill_async_io ? "1" : "0"},
           {"ASYNC_DECODE_IO", "0"},
           {"EXPERT_IO_WORKERS", std::to_string(o.prefill_io_workers)},
           {"DECODE_EXPERT_IO_WORKERS",
            std::to_string(o.runtime.decode_io_workers)},
           {"STARTUP_EXPERT_PREFETCH", o.runtime.startup_prefetch},
           {"STARTUP_IO_WORKERS", std::to_string(o.runtime.startup_io_workers)},
           {"SKIP_EXPERT_ZERO", "1"},
           {"EXPERT_READ_CHUNK_KIB", std::to_string(o.runtime.read_chunk_kib)},
           {"KV_APPEND_ONLY", o.runtime.kv_append_only ? "1" : "0"},
           {"GDN_STATE_SWAP", o.runtime.state_swap ? "1" : "0"},
           {"NATIVE_MAILBOX", "1"},
           {"HIT_FIRST_EXPERTS", o.runtime.hit_first ? "1" : "0"},
           {"EXPERT_READY_IO", o.runtime.ready_io ? "1" : "0"},
           {"OVERLAP_STATE_FEEDBACK", o.runtime.overlap_feedback ? "1" : "0"},
           {"MAILBOX_BACKOFF_US", "0"},
           {"MAILBOX_POLL_DELAY_NS", "0"},
           {"SEED_PREFILL_LRU", "0"},
           {"RESIDENT_PREFAULT", "0"},
           {"EXPERT_STAGE_READY", "0"},
           {"SAVE_ALL_LOGITS", "0"},
           {"DEVICE_CHAT", "1"},
           {"MOE_PROFILE_TOTAL", o.profile ? "1" : "0"},
           {"CHAT_ROPE_ROOT", o.model_root},
           {"PREFILL_INITIAL_IMAGE", o.image_path.empty() ? "0" : "1"},
           {"VISION_INPUT", vision_enabled ? "1" : "0"}})
    if (setenv(item.first.c_str(), item.second.c_str(), 1))
      throw std::runtime_error("Cannot set " + item.first);
  for (const char* flag :
       {"BENCH_TOKENS_PATH",
        "BENCH_IGNORE_EOS",
        "FRONT_OVERRIDE",
        "DECODE_TRACE_LAYERS",
        "QNN_SERIAL_CONTEXTS"})
    unsetenv(flag);
}
} // namespace moe_chat
