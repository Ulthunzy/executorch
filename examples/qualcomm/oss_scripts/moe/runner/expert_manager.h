#pragma once

#include "decode/expert_read.h"
#include "qnn/qnn_front.h"
#include "runtime_utils.h"
#include <array>
#include <set>
#include <atomic>
#include <cerrno>
#include <thread>
#include "resident_handoff.h"
#include "session_state.h"

namespace moe::decode {

// 一个专家的对齐磁盘范围和有效 payload；cached 与 cache_destination
// 区分命中/缺失。

// 40 层各有独立的 LRU，层号是缓存键的一部分。
// 相同 expert ID 在不同层对应不同权重，不能跨层共享一个槽。
class ExpertCache {
  struct Entry {
    int expert = -1;
    uint64_t stamp = 0;
    bool prefetched = false, pending = false;
  };

  int layers, capacity;
  size_t stride, data_offset = 0;
  uint8_t* data = nullptr;
  uint64_t clock = 0;
  std::vector<Entry> entries;
  std::vector<std::shared_ptr<Buffer>> buffers;
  bool mapped;
  ResidentExperts transfer{};

 public:
  struct Hits {
    int hot = 0, cold = 0, predicted = 0;
  };

  size_t bytes = 0;
  bool imported = false;

  ExpertCache(
      Runtime& runtime,
      int count,
      size_t maximum,
      size_t budget,
      bool direct,
      const std::string& root);

  ~ExpertCache();

  uint8_t* location(int layer, int slot);

  int fd(int layer) const {
    return buffers.empty() ? -1 : buffers[layer]->fd;
  }

  int size() const {
    return data_offset + capacity * stride;
  }

  void publish(int layer);

  void use_direct(
      int layer,
      const int* ids,
      ExpertRead* requests,
      uint64_t* offsets);

  int slots_per_layer() const {
    return capacity;
  }

  int expert_at(int layer, int slot) const {
    return entries[layer * capacity + slot].expert;
  }

  int find(int layer, int expert) const;

  int reserve_prefetch(
      int layer,
      int expert,
      const std::array<int, 8>& previous);

  void finish_prefetch(int layer, int slot, int expert);

  void verify_cache();

  void seed_layer(int layer, const std::vector<int>& ids);

  void reset();

  Hits prepare(int layer, const int* ids, ExpertRead* requests);

  void export_state(Qwen35SessionState& state) const;
  void import_state(const Qwen35SessionState& state);
};

} // namespace moe::decode
