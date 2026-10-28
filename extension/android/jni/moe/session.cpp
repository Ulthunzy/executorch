#include "session.h"

#include <dlfcn.h>
#include <linux/memfd.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <cstdlib>
#include <stdexcept>

int qwen35_moe_run_client(moe_chat::Options, moe_chat::Client&);

namespace qwen35_jni {
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
         {"CHAT_PREFILL_PLAN",
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
    for (int layer = 0; layer < 40; ++layer)
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
    // Packaged APK libraries have unique basenames for the two DSP skeletons.
    if (!libraries.empty()) {
      const fs::path libs = fs::canonical(libraries);
      for (const char* stage : {"prefill", "decode"}) {
        const auto destination = fs::path(root_) / stage;
        const auto skel =
            libs / (std::string("libqwen35_") + stage + "_skel.so");
        if (fs::exists(skel)) {
          const auto filename = std::string(stage) == "decode"
              ? "libinteger_decode_rpc_skel.so"
              : "libinteger_rpc_skel.so";
          fs::remove(destination / filename);
          fs::create_symlink(skel, destination / filename);
        }
        for (const char* name :
             {"libmoe_decode.so",
              "libqwen35_shared.so",
              "libQnnDeltaNetOpPackage_HTP.so"})
          if (fs::exists(libs / name)) {
            fs::remove(destination / name);
            fs::create_symlink(libs / name, destination / name);
          }
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

void Session::ready() {
  workspace_->release_handoff();
}

Session::Session(
    moe_chat::Options options,
    std::unique_ptr<Workspace> workspace)
    : options_(std::move(options)), workspace_(std::move(workspace)) {}

Session::~Session() {
  shutdown();
}

void Session::stop() {
  stop_.store(true);
}
bool Session::stopped() const {
  return stop_.load();
}

void Session::shutdown() {
  std::lock_guard<std::mutex> join(shutdown_);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    closing_ = true;
    stop();
    changed_.notify_all();
  }
  if (worker_.joinable())
    worker_.join();
}

moe_chat::Request Session::next() {
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

void Session::token(const std::string& text) {
  std::lock_guard<std::mutex> lock(mutex_);
  output_.push_back(text);
  changed_.notify_all();
}

void Session::complete(const std::string& stats) {
  std::lock_guard<std::mutex> lock(mutex_);
  stats_ = stats;
  done_ = true;
  changed_.notify_all();
}

void Session::fail(const std::string& error) {
  std::lock_guard<std::mutex> lock(mutex_);
  error_ = error;
  done_ = true;
  changed_.notify_all();
}

void Session::run() {
  Environment restore;
  try {
    auto request = next();
    if (request.kind != moe_chat::Request::Kind::Close) {
      options_.prompts = {request.prompt};
      options_.image_path = request.image_path;
      options_.max_new_tokens = request.max_new_tokens;
      options_.temperature = request.temperature;
      options_.thinking = request.thinking;
      setenv("QWEN35_TOKENIZER_CACHE", workspace_->root().c_str(), 1);
      const int status = qwen35_moe_run_client(options_, *this);
      if (status)
        fail("MoE model execution failed; close and recreate the module");
    }
  } catch (const std::exception& error) {
    fail(error.what());
  }
  std::lock_guard<std::mutex> lock(mutex_);
  exited_ = true;
  changed_.notify_all();
}

std::string Session::invoke(
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
      ((!options_.interactive && !request.image_path.empty()) ||
       request.max_new_tokens > options_.capacity))
    throw std::invalid_argument(
        "Image input is disabled or output exceeds context capacity");
  stop_.store(false);
  error_.clear();
  stats_.clear();
  output_.clear();
  done_ = false;
  request_ = std::move(request);
  if (!worker_.joinable())
    worker_ = std::thread(&Session::run, this);
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
} // namespace qwen35_jni
