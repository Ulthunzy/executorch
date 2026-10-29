#include "worker_start.h"
#ifndef __hexagon__
#include <thread>
#endif

WorkerStart::WorkerStart(uint32_t participants) : participants_(participants) {}

int WorkerStart::arrive(int lock_error) {
  if (lock_error) {
    int expected = 0;
    __atomic_compare_exchange_n(
        &error_,
        &expected,
        lock_error,
        false,
        __ATOMIC_RELAXED,
        __ATOMIC_RELAXED);
  }
  // RMW 的 release/acquire 链把每个 worker 的锁错误传递给全组。
  __atomic_fetch_add(&arrived_, 1U, __ATOMIC_ACQ_REL);
  while (__atomic_load_n(&arrived_, __ATOMIC_ACQUIRE) != participants_) {
#ifdef __hexagon__
    asm volatile("pause(#32)");
#else
    std::this_thread::yield();
#endif
  }
  return __atomic_load_n(&error_, __ATOMIC_ACQUIRE);
}
