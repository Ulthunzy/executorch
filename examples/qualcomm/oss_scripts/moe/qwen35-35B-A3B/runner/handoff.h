#pragma once
#include <cstdint>

// prefill 导出的历史 LRU 顺序，以及旧四槽组的交接元数据。
// 这是原始二进制 ABI，字段顺序/宽度必须与写入和读取两端一致。
struct InheritedExperts {
  struct Entry {
    int32_t expert;
    uint32_t reserved;
    uint64_t stamp;
  };

  char magic[8];
  uint64_t stride;
  uint64_t buffer_bytes[4];
  uint64_t clocks[4];
  uint64_t fingerprints[4];
  int32_t fds[4];
  int32_t group;
  int32_t capacity;
  Entry entries[4][64];
  uint64_t layer_clocks[40];
  Entry layer_entries[40][64];
};

// 抽样检查共享权重是否仍指向同一内容；不是全量校验和。
uint64_t expertFingerprint(const uint8_t* data, uint64_t stride);
