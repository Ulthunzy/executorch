#pragma once
#include <cstdint>

// 每次 RPC 独立的启动屏障。所有 worker 均报告锁申请结果后，才能进入计算屏障。
class WorkerStart {
 public:
  explicit WorkerStart(uint32_t participants);
  WorkerStart(const WorkerStart&) = delete;
  WorkerStart& operator=(const WorkerStart&) = delete;
  int arrive(int lock_error);

 private:
  const uint32_t participants_;
  uint32_t arrived_ = 0;
  int error_ = 0;
};
