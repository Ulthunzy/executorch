#pragma once
#include <cstdint>
#include <initializer_list>
constexpr int kExpertSlots = 64;

// 同进程 prefill→decode 的 40 层常驻专家交接，传 fd、槽 ID 和 LRU stamp。
// 两个阶段在同一地址空间运行；导入端关闭 dup fd，源 buffer 仍由模型持有。
struct ResidentExperts {
  struct Layer {
    int32_t fd;
    uint32_t hot_mask;
    uint64_t bytes, fingerprint;
    int32_t ids[kExpertSlots];
    uint64_t stamps[kExpertSlots];
  };

  char magic[8];
  uint64_t stride, data_offset;
  uint32_t capacity, layers;
  Layer layer[40];
};

// 每槽仅取固定采样点，用来检测交接错位，不保证发现任意权重损坏。
uint64_t
residentFingerprint(const uint8_t* data, uint64_t stride, uint64_t offset);
