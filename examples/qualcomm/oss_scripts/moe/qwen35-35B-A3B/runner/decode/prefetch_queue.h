#pragma once

#include "expert_read.h"
#include "../runtime_utils.h"
#include <atomic>
#include <memory>
#include <vector>
#include <thread>
#include <condition_variable>
#include <deque>
#include <mutex>

// 预测读取只填预留槽，实际路由仍由当前层 Router 决定，不改变模型 Top8。
struct PrefetchTask {
  int layer = 0, expert = 0, slot = 0, position = 0;
  ExpertRead request;
  Clock::time_point submitted{}, started{}, completed{};
  std::atomic<bool> done{false};
  bool required = false;
};

// 可选的预测预取线程池；required 任务优先，pause 暂停尚未取走的预测任务。
// 默认入口关闭预测路径，普通缺失读取由 ExpertReaders 承担。
class PrefetchQueue {
  std::mutex mutex;
  std::condition_variable changed;
  std::deque<std::shared_ptr<PrefetchTask>> jobs;
  std::vector<std::thread> workers;
  std::array<void*, 2> staging{};
  bool stopping = false, paused = false;

  void work(int index);

 public:
  explicit PrefetchQueue(size_t maximum);

  void stop();

  ~PrefetchQueue();

  void submit(const std::shared_ptr<PrefetchTask>& task);

  void pause(bool value);

  void wait(const std::shared_ptr<PrefetchTask>& task);
};
