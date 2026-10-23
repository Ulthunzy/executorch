#pragma once

#include "expert_read.h"
#include "../runtime_utils.h"
#include <vector>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <mutex>
#include <thread>

// 将 Top8 缺失权重拆成互不重叠的读取块，在后台写入已经预留的 LRU 槽。
// start 借用请求数组，finish 前调用者不能销毁或复用这些请求和目标槽。
class ExpertReaders {
  std::vector<std::thread> threads;
  std::vector<void*> staging;
  std::mutex mutex;
  std::condition_variable changed;
  std::atomic<int> next{0};
  ExpertRead* requests = nullptr;
  ExpertRead* original = nullptr;
  std::vector<ExpertRead> chunks;
  std::vector<int> owners;
  size_t chunk_bytes = 0;
  int original_count = 0;
  int count = 0, epoch = 0, finished = 0;
  bool stopping = false, pending = false;
  // uncached 控制页：word 0=generation，word 1=错误，每隔 32 个 word
  // 一个专家标志。 payload 先完成 DMA 同步，再用 release 发布，DSP 用 acquire
  // 观察。
  uint32_t* ready_words = nullptr;
  int cache_fd = -1;
  std::atomic<int> remaining[8];
  std::atomic<uint32_t> completed_chunks[8], gu_published[8];
  bool stage_ready = getenv("EXPERT_STAGE_READY") &&
      std::string(getenv("EXPERT_STAGE_READY")) == "1";
  Clock::time_point started{}, completed{};

  static void perform(ExpertRead& request, void* buffer);

  void worker(int index);

 public:
  explicit ExpertReaders(int workers, size_t maximum);

  void stop();

  ~ExpertReaders();

  void start(
      ExpertRead* jobs,
      int size,
      uint32_t* readiness = nullptr,
      int destination_fd = -1);

  void finish();

  double wall_ms() const;

  uint64_t disk_bytes() const;

  int disk_calls() const;

  double overlap_ms(Clock::time_point begin, Clock::time_point end) const;

  void read(ExpertRead* jobs, int size);
};
