#pragma once
#include "bank_mapping.h"
#include "qurt.h"
#include "qurt_sem.h"

namespace moe::dsp::decode {

// Map the next layer while QNN executes. Two windows bound DSP virtual memory
// without copying weights or changing the ARM expert cache's ownership.
class BankMapper {
  // 仅两个 DSP 虚拟地址窗：当前层占一个，后台线程准备下一层占另一个。
  // schedule/wait 每次只允许一项在途请求，避免覆盖仍在计算的映射。
  int fds[40]{}, bytes = 0, requested = 0, status = 0;
  BankMapping windows[2];
  qurt_sem_t ready, done;
  qurt_thread_t thread{};
  volatile uint32_t stop = 0;
  void* stack = nullptr;
  bool started = false, pending = false;
  uint64_t map_ticks = 0;

  static void work(void* opaque);

 public:
  BankMapper();

  int configure(const int* descriptors, int count);

  bool enabled() const {
    return started;
  }

  int wait(int fd, int length, const uint8_t*& pointer, int& layer);

  uint64_t duration() const {
    return map_ticks;
  }

  void schedule(int layer);

  void close();

  ~BankMapper();
};

} // namespace moe::dsp::decode
