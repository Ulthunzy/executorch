#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>

class PrefetchQueue;
struct PrefetchTask;

// 对齐磁盘范围与其中的有效专家 payload；目标槽需保留到读取完成。
struct ExpertRead {
  int fd = -1, error = 0;
  size_t file_offset = 0, read_bytes = 0, prefix = 0, payload_bytes = 0;
  uint8_t* destination = nullptr;
  const uint8_t* cached = nullptr;
  uint8_t* cache_destination = nullptr;
  double pread_ms = 0, copy_ms = 0, publish_ms = 0;
  PrefetchQueue* source_queue = nullptr;
  std::shared_ptr<PrefetchTask> source;
};
