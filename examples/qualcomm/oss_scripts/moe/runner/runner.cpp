#include "runner.h"
#include <nlohmann/json.hpp>
#include "model_adapter.h"
#include "chat/terminal.h"

#include <dlfcn.h>
#include <linux/memfd.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <cstdlib>
#include <fstream>
#include <stdexcept>

int qwen35_moe_run_client(moe_chat::Options, moe_chat::Client&);

namespace moe {
namespace {
namespace fs = std::filesystem;

bool transient(const std::string& name) {
  return name == "cos.f32" || name == "sin.f32" || name == "tokens.i64" ||
      name == "pixels.f32" || name == "resident-handoff.bin" ||
      name == "hot-masks.u32" || name == "cache-prefill" ||
      name == "image-features.f32" || name.rfind("pte-", 0) == 0 ||
      name.rfind("prefill-", 0) == 0 || name.rfind("reused-", 0) == 0 ||
      name.rfind("decode-pte-", 0) == 0 || name.rfind("chat", 0) == 0;
}

void link_assets(const fs::path& source, const fs::path& target) {
  fs::create_directory(target);
  for (const auto& file : fs::directory_iterator(source)) {
    const auto name = file.path().filename().string();
    if (!transient(name))
      fs::create_symlink(fs::absolute(file.path()), target / name);
  }
}

// configure_environment is shared with the terminal runner. Restore only its
// keys on exit, never replace environ or redirect the application's stdout.
class Environment {
 public:
  Environment() {
    for (const char* key :
         {"MOE_QNN_LIBRARY_PATH",
          "MOE_MODEL_FAMILY",
          "CHAT_PREFILL_PLAN",
          "LD_LIBRARY_PATH",
          "ADSP_LIBRARY_PATH",
          "SHARED_MODEL_ROOT",
          "SHARED_OP_PACKAGE",
          "HANDOFF_DECODE_ROOT",
          "HANDOFF_MAX_TOKENS",
          "HANDOFF_EOS_IDS",
          "PREFILL_TOKENS",
          "PREFILL_VALID_TOKENS",
          "PREFILL_INITIAL_IMAGE",
          "DECODE_CONTEXT_CAPACITY",
          "EXPERT_CACHE_SLOTS",
          "EMBEDDING_BITS",
          "EXPERT_BITS",
          "NATIVE_MAPPED_IO",
          "NATIVE_RESIDUAL",
          "REUSE_EXPERTS",
          "INHERIT_EXPERT_CACHE",
          "ASYNC_EXPERT_IO",
          "ASYNC_DECODE_IO",
          "EXPERT_IO_WORKERS",
          "DECODE_EXPERT_IO_WORKERS",
          "STARTUP_EXPERT_PREFETCH",
          "STARTUP_IO_WORKERS",
          "SKIP_EXPERT_ZERO",
          "EXPERT_READ_CHUNK_KIB",
          "KV_APPEND_ONLY",
          "GDN_STATE_SWAP",
          "NATIVE_MAILBOX",
          "HIT_FIRST_EXPERTS",
          "EXPERT_READY_IO",
          "OVERLAP_STATE_FEEDBACK",
          "MAILBOX_BACKOFF_US",
          "MAILBOX_POLL_DELAY_NS",
          "SEED_PREFILL_LRU",
          "RESIDENT_PREFAULT",
          "EXPERT_STAGE_READY",
          "SAVE_ALL_LOGITS",
          "DEVICE_CHAT",
          "MOE_PROFILE_TOTAL",
          "CHAT_ROPE_ROOT",
          "VISION_INPUT",
          "BENCH_TOKENS_PATH",
          "BENCH_IGNORE_EOS",
          "FRONT_OVERRIDE",
          "DECODE_TRACE_LAYERS",
          "QNN_SERIAL_CONTEXTS",
          "QWEN35_TOKENIZER_CACHE"}) {
      const char* previous = getenv(key);
      values_.emplace(
          key, previous ? std::optional<std::string>(previous) : std::nullopt);
    }
  }
  ~Environment() {
    for (const auto& item : values_)
      if (item.second)
        setenv(item.first.c_str(), item.second->c_str(), 1);
      else
        unsetenv(item.first.c_str());
  }

 private:
  std::map<std::string, std::optional<std::string>> values_;
};
} // namespace

Workspace::Workspace(
    const std::string& model,
    const std::string& cache,
    const std::string& libraries) {
  if (fs::is_regular_file(model))
    package_root_ = unpack_model_package(model, cache);
  const auto source = fs::canonical(
      package_root_.empty() ? fs::path(model)
                            : fs::path(package_root_) / "model");
  const auto parent = fs::canonical(cache);
  std::string pattern = (parent / "qwen35-moe-XXXXXX").string();
  if (!mkdtemp(pattern.data()))
    throw std::runtime_error("Cannot create private MoE workspace");
  root_ = pattern;
  try {
    for (const auto& file : fs::directory_iterator(source)) {
      const auto name = file.path().filename().string();
      if (name != "prefill" && name != "decode" && !transient(name))
        fs::create_symlink(file.path(), fs::path(root_) / name);
    }
    for (const char* stage : {"prefill", "decode"})
      link_assets(source / stage, fs::path(root_) / stage);
    // A plan bundle is immutable, despite its prefill-* filename prefix.
    if (!fs::exists(fs::path(root_) / "prefill/plans"))
      fs::create_directory_symlink(
          source / "prefill/plans", fs::path(root_) / "prefill/plans");
    const auto decode = fs::path(root_) / "decode";
    fs::create_directory(decode / "cache-prefill");
    handoff_fds_.reserve(80);
    std::string family = "qwen35";
    std::ifstream configuration(source / "MODEL_CONFIG.json");
    if (configuration) {
      nlohmann::json metadata;
      configuration >> metadata;
      family = metadata.value("model_family", family);
    }
    const auto model = create_model_adapter(family);
    for (int layer = 0; layer < model->info().layers; ++layer)
      for (int c = 0; c < 2; ++c) {
        const auto name = "layer" + std::to_string(layer) + "-cache" +
            std::to_string(c) + ".u16";
        // These snapshots only bridge resident stages, so keep them off UFS.
        const int fd = syscall(SYS_memfd_create, "qwen35-cache", MFD_CLOEXEC);
        if (fd < 0)
          throw std::runtime_error("Cannot create in-memory cache handoff");
        handoff_fds_.push_back(fd);
        fs::create_symlink(
            "/proc/self/fd/" + std::to_string(fd),
            fs::path(root_) / "prefill" / ("pte-0-" + name));
        fs::create_symlink(
            fs::path(root_) / "prefill" / ("pte-0-" + name),
            decode / "cache-prefill" / name);
      }
    for (const char* name :
         {"prefill-routes.i32",
          "prefill-history-ids.i32",
          "prefill-history-stamps.u64"})
      fs::create_symlink(fs::path(root_) / "prefill" / name, decode / name);
    if (!package_root_.empty())
      link_model_libraries(root_, libraries);
    if (!libraries.empty()) {
      const auto libs = fs::canonical(libraries);
      for (const char* stage : {"prefill", "decode"})
        for (const char* name : {"libet_infer.so", "libet_qnnop.so"}) {
          const auto destination = fs::path(root_) / stage / name;
          if (!fs::is_regular_file(libs / name))
            throw std::runtime_error("Missing runtime library: " + (libs / name).string());
          fs::remove(destination);
          fs::create_symlink(libs / name, destination);
        }
    }

  } catch (...) {
    for (const int fd : handoff_fds_)
      ::close(fd);
    fs::remove_all(root_);
    throw;
  }
}

Workspace::~Workspace() {
  release_handoff();
  std::error_code error;
  fs::remove_all(root_, error);
}

void Workspace::release_handoff() {
  for (const int fd : handoff_fds_)
    ::close(fd);
  handoff_fds_.clear();
}

void Runner::ready() {
  loaded_.store(true);
  workspace_->release_handoff();
}

Runner::Runner(moe_chat::Options options, std::unique_ptr<Workspace> workspace)
    : options_(std::move(options)), workspace_(std::move(workspace)) {}

Runner::~Runner() {
  shutdown();
}

void Runner::stop() {
  stop_.store(true);
}
bool Runner::stopped() const {
  return stop_.load();
}

void Runner::shutdown() {
  std::lock_guard<std::mutex> join(shutdown_);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    closing_ = true;
    stop();
    changed_.notify_all();
  }
  if (worker_.joinable())
    worker_.join();
  loaded_.store(false);
}

moe_chat::Request Runner::next() {
  std::unique_lock<std::mutex> lock(mutex_);
  changed_.wait(lock, [&] { return request_.has_value() || closing_; });
  if (closing_) {
    moe_chat::Request close;
    close.kind = moe_chat::Request::Kind::Close;
    return close;
  }
  auto result = std::move(*request_);
  request_.reset();
  return result;
}

void Runner::token(const std::string& text) {
  std::lock_guard<std::mutex> lock(mutex_);
  output_.push_back(text);
  changed_.notify_all();
}

void Runner::complete(const std::string& stats) {
  std::lock_guard<std::mutex> lock(mutex_);

  stats_ = stats;
  done_ = true;
  changed_.notify_all();
}

void Runner::fail(const std::string& error) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (error_.empty())
    error_ = error;
  done_ = true;
  changed_.notify_all();
}

void Runner::run() {
  Environment restore;
  try {
    auto request = next();
    if (request.kind != moe_chat::Request::Kind::Close) {
      options_.prompts = {request.prompt};
      options_.image_path = request.image_path;
      options_.max_new_tokens = request.max_new_tokens;
      options_.temperature = request.temperature;
      options_.thinking = request.thinking;
      if (request.ignore_eos)
        options_.ignore_eos = *request.ignore_eos;
      setenv("QWEN35_TOKENIZER_CACHE", workspace_->root().c_str(), 1);
      const int status = qwen35_moe_run_client(options_, *this);
      if (status)
        fail("MoE model execution failed; close and recreate the module");
    }
  } catch (const std::exception& error) {
    fail(error.what());
  }
  std::lock_guard<std::mutex> lock(mutex_);
  loaded_.store(false);
  exited_ = true;
  changed_.notify_all();
}

std::string Runner::invoke(
    moe_chat::Request request,
    const std::function<void(const std::string&)>& callback) {
  std::unique_lock<std::mutex> operation(invocation_, std::try_to_lock);
  if (!operation.owns_lock())
    throw std::runtime_error("Another generation or reset is in progress");
  std::unique_lock<std::mutex> lock(mutex_);
  if (closing_ || exited_)
    throw std::runtime_error("MoE session is closed or failed");
  // Reset before the first prompt does not load several GiB of model weights.
  if (!worker_.joinable() && request.kind == moe_chat::Request::Kind::Reset)
    return "{\"reset\":true}";
  if (request.kind == moe_chat::Request::Kind::Generate &&
      ((!options_.vision_enabled && !request.image_path.empty()) ||
       request.max_new_tokens > options_.capacity))
    throw std::invalid_argument(
        "Image input is disabled or output exceeds context capacity");
  stop_.store(false);
  error_.clear();
  stats_.clear();
  output_.clear();
  done_ = false;
  budget_ = request.budget;
  request_ = std::move(request);
  if (!worker_.joinable())
    worker_ = std::thread(&Runner::run, this);
  changed_.notify_all();
  std::exception_ptr callback_error;
  while (true) {
    changed_.wait(lock, [&] { return !output_.empty() || done_ || exited_; });
    while (!output_.empty()) {
      auto text = std::move(output_.front());
      output_.pop_front();
      lock.unlock();
      if (!callback_error) {
        try {
          callback(text);
        } catch (...) {
          callback_error = std::current_exception();
          stop();
        }
      }
      lock.lock();
    }
    if (done_ || exited_)
      break;
  }
  if (callback_error)
    std::rethrow_exception(callback_error);
  if (!error_.empty())
    throw std::runtime_error(error_);
  if (!done_)
    throw std::runtime_error("MoE session closed during generation");
  return stats_;
}
} // namespace moe

namespace moe {
bool Runner::is_loaded() const {
  return loaded_.load();
}

executorch::runtime::Error Runner::load() {
  // QNN contexts remain lazy until the first input.
  std::lock_guard<std::mutex> operation(invocation_);
  std::lock_guard<std::mutex> lock(mutex_);
  if (closing_ || exited_)
    return executorch::runtime::Error::InvalidState;
  loaded_.store(true);
  return executorch::runtime::Error::Ok;
}
void Runner::reset() {
  moe_chat::Request request;
  request.kind = moe_chat::Request::Kind::Reset;
  invoke(std::move(request), [](const std::string&) {});
}
executorch::runtime::Error Runner::generate(
    const std::string& prompt,
    const executorch::extension::llm::GenerationConfig& config,
    std::function<void(const std::string&)> token,
    std::function<void(const executorch::llm::Stats&)> stats_callback) {
  if (!config.grammar.empty() || !config.grammar_type.empty() ||
      config.num_bos || config.num_eos)
    return executorch::runtime::Error::NotSupported;
  moe_chat::Request request;
  request.prompt = prompt;
  request.max_new_tokens = config.max_new_tokens < 0
      ? capacity()
      : std::min(config.max_new_tokens, capacity());
  request.budget =
      moe_chat::GenerationBudget{config.max_new_tokens, config.seq_len};
  request.temperature = config.temperature;
  request.thinking = options_.thinking;
  request.ignore_eos = config.ignore_eos;
  try {
    if (config.echo && token)
      token(prompt);
    const auto raw =
        invoke(std::move(request), token ? token : [](const std::string&) {});
    if (stats_callback) {
      const auto data = nlohmann::json::parse(raw);
      executorch::llm::Stats stats;
      stats.model_load_start_ms = data.value("model_load_start_ms", 0L);
      stats.model_load_end_ms = data.value("model_load_end_ms", 0L);
      stats.inference_start_ms = data.value("inference_start_ms", 0L);
      stats.inference_end_ms = data.value("inference_end_ms", 0L);
      stats.prompt_eval_end_ms = data.value("prompt_eval_end_ms", 0L);
      stats.first_token_ms = data.value("first_token_ms", 0L);
      stats.token_encode_end_ms = data.value("token_encode_end_ms", 0L);
      stats.model_execution_start_ms =
          data.value("model_execution_start_ms", 0L);
      stats.model_execution_end_ms = data.value("model_execution_end_ms", 0L);
      stats.num_prompt_tokens = data.value("prompt_tokens", int64_t(0));
      stats.num_generated_tokens = data.value("generated_tokens", int64_t(0));
      stats.aggregate_sampling_time_ms =
          data.value("aggregate_sampling_time_ms", 0L);
      stats.aggregate_model_execution_time_ms =
          data.value("aggregate_model_execution_time_ms", 0L);
      stats_callback(stats);
    }
    return executorch::runtime::Error::Ok;
  } catch (const std::invalid_argument&) {
    return executorch::runtime::Error::InvalidArgument;
  } catch (const std::exception&) {
    return executorch::runtime::Error::Internal;
  }
}
int Runner::run_terminal(int argc, char** argv, moe_chat::PrefillMain prefill) {
  SessionLease lease;
  moe_chat::ScopedConsoleOutput restore_output;
  Environment restore;
  return moe_chat::main(argc, argv, prefill);
}
} // namespace moe

namespace moe {
int Runner::output_limit(int64_t occupied, int requested) const {
  if (!budget_)
    return requested;
  executorch::extension::llm::GenerationConfig config;
  config.max_new_tokens = budget_->max_new_tokens;
  config.seq_len = budget_->seq_len;
  return config.resolve_max_new_tokens(capacity(), occupied);
}
} // namespace moe
