#pragma once

#include <executorch/examples/qualcomm/oss_scripts/moe/qwen35-35B-A3B/runner/chat/conversation.h>
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

namespace qwen35_jni {
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

class Session final : public moe_chat::Client {
 public:
  Session(moe_chat::Options options, std::unique_ptr<Workspace> workspace);
  ~Session() override;
  std::string invoke(
      moe_chat::Request request,
      const std::function<void(const std::string&)>& token);
  void stop();
  void shutdown();
  int capacity() const {
    return options_.capacity;
  }

 private:
  moe_chat::Request next() override;
  bool stopped() const override;
  void ready() override;
  void token(const std::string& text) override;
  void complete(const std::string& stats) override;
  void fail(const std::string& error) override;
  void run();
  moe_chat::Options options_;
  std::unique_ptr<Workspace> workspace_;
  std::thread worker_;
  std::mutex mutex_, invocation_, shutdown_;
  std::condition_variable changed_;
  std::optional<moe_chat::Request> request_;
  std::deque<std::string> output_;
  std::atomic<bool> stop_{false};
  bool closing_ = false, exited_ = false, done_ = false;
  std::string stats_, error_;
};
} // namespace qwen35_jni
