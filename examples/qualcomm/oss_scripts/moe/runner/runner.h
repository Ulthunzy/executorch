#pragma once

#include <executorch/examples/qualcomm/oss_scripts/moe/runner/chat/model_session.h>
#include <executorch/extension/llm/runner/irunner.h>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>
#include "chat/chat.h"
#include "session_lease.h"

namespace moe {
// Only this adapter uses the application workspace. Exported model assets are
// read-only; fd handoff, RoPE and runtime snapshots never overwrite them.
class Workspace {
 public:
  Workspace(
      const std::string& model,
      const std::string& cache,
      const std::string& libraries);
  ~Workspace();
  Workspace(const Workspace&) = delete;
  Workspace& operator=(const Workspace&) = delete;
  void release_handoff();
  const std::string& root() const {
    return root_;
  }
  const std::string& package_root() const {
    return package_root_;
  }

 private:
  std::string root_;
  std::string package_root_;
  std::vector<int> handoff_fds_;
};

class Runner final : public executorch::extension::llm::IRunner,
                     private moe_chat::Client {
 public:
  Runner(moe_chat::Options options, std::unique_ptr<Workspace> workspace);
  ~Runner() override;
  std::string invoke(
      moe_chat::Request request,
      const std::function<void(const std::string&)>& token);
  bool is_loaded() const override;
  executorch::runtime::Error load() override;
  executorch::runtime::Error generate(
      const std::string& prompt,
      const executorch::extension::llm::GenerationConfig& config,
      std::function<void(const std::string&)> token,
      std::function<void(const executorch::llm::Stats&)> stats) override;
  void reset() override;
  void stop() override;
  static int run_terminal(int argc, char** argv, moe_chat::PrefillMain prefill);

  void shutdown();
  int capacity() const {
    return options_.capacity;
  }

 private:
  moe_chat::Request next() override;
  bool stopped() const override;
  int output_limit(int64_t occupied, int requested) const override;
  void ready() override;
  void token(const std::string& text) override;
  void complete(const std::string& stats) override;
  void fail(const std::string& error) override;
  void run();
  SessionLease session_lease_;
  moe_chat::Options options_;
  std::unique_ptr<Workspace> workspace_;
  std::thread worker_;
  std::mutex mutex_, invocation_, shutdown_;
  std::condition_variable changed_;
  std::optional<moe_chat::Request> request_;
  std::deque<std::string> output_;
  std::atomic<bool> stop_{false}, loaded_{false};
  bool closing_ = false, exited_ = false, done_ = false;
  std::string stats_, error_;
  std::optional<moe_chat::GenerationBudget> budget_;
};
} // namespace moe
