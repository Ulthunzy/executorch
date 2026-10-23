#pragma once

#include <cstdint>
#include <vector>

// 单层专家槽的 LRU 元数据；权重内存由 PrefillSession 管理。
// stamp 是访问顺序，不是墙上时间；seeded 仅用于统计启动预加载命中。
struct ExpertCache {
  struct Entry {
    int expert = -1;
    uint64_t stamp = 0;
    bool seeded = false;
  };

  struct Binding {
    int slot;
    bool hit;
    bool seed_hit = false;
  };

  std::vector<Entry> entries;
  uint64_t clock = 0;

  explicit ExpertCache(int capacity);

  void reset();

  int find(int expert) const;

  std::vector<Binding> prepare(const std::vector<int>& experts);
};
