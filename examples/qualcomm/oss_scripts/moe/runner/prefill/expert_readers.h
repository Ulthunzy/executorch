#pragma once

#include "expert_read.h"
#include "../runtime_utils.h"
#include <vector>
#include <cerrno>
#include <condition_variable>
#include <mutex>
#include <thread>

// read_bytes/file_offset 是 O_DIRECT 对齐范围；prefix/payload_bytes
// 描述其中的专家。 cached 跳过磁盘读取，raw_staging 可指定流水线自己的临时区。

// 固定线程池读取一个批次。start 借用 requests，调用者须保留它直到 finish 返回。
class ExpertReaders {
  std::vector<std::thread> threads;
  std::vector<void*> staging;
  std::mutex mutex;
  std::condition_variable changed;
  int next = 0;
  ExpertRead* requests = nullptr;
  int count = 0, epoch = 0, finished = 0;
  int active = 0;
  bool stopping = false, pending = false, sealed = true;
  Clock::time_point started{}, completed{};
  Clock::time_point active_started{};
  std::vector<std::pair<Clock::time_point, Clock::time_point>> active_spans;

  static void perform(ExpertRead& request, void* buffer);

  void worker(int index);

 public:
  explicit ExpertReaders(int workers, size_t maximum);

  void stop();

  ~ExpertReaders();

  void start(ExpertRead* jobs, int size, bool growable = false);

  // Appended records must be initialized in the original, stable allocation
  // before extend publishes the new count to workers.
  void extend(int size);

  void seal();

  void finish();

  void drain();

  double wall_ms() const;

  double overlap_ms(Clock::time_point begin, Clock::time_point end) const;

  void read(ExpertRead* jobs, int size);
};
