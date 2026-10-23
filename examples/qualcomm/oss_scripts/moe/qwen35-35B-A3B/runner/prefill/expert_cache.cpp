#include "expert_cache.h"

#include <algorithm>
#include <stdexcept>

ExpertCache::ExpertCache(int capacity) : entries(capacity) {}

void ExpertCache::reset() {
  std::fill(entries.begin(), entries.end(), Entry{});
  clock = 0;
}

int ExpertCache::find(int expert) const {
  for (size_t slot = 0; slot < entries.size(); slot++)
    if (entries[slot].expert == expert)
      return slot;
  return -1;
}

std::vector<ExpertCache::Binding> ExpertCache::prepare(
    const std::vector<int>& experts) {
  if (experts.size() > entries.size())
    throw std::runtime_error("expert cache batch exceeds capacity");
  std::vector<Binding> result(experts.size(), {-1, false});
  std::vector<bool> pinned(entries.size());
  // 先保护本批所有命中，再为缺失专家挑槽，防止覆盖稍后仍要计算的命中权重。
  for (size_t i = 0; i < experts.size(); i++) {
    int slot = find(experts[i]);
    if (slot < 0)
      continue;
    if (pinned[slot])
      throw std::runtime_error("duplicate expert in cache batch");
    result[i] = {slot, true, entries[slot].seeded};
    pinned[slot] = true;
    entries[slot].stamp = ++clock;
    entries[slot].seeded = false;
  }
  // pinned 只在当前批有效；未受保护的最旧槽才能被替换。
  for (size_t i = 0; i < experts.size(); i++)
    if (result[i].slot < 0) {
      int victim = -1;
      for (size_t slot = 0; slot < entries.size(); slot++)
        if (!pinned[slot] &&
            (victim < 0 || entries[slot].stamp < entries[victim].stamp))
          victim = slot;
      if (victim < 0)
        throw std::runtime_error("no unpinned expert cache slot");
      result[i] = {victim, false};
      pinned[victim] = true;
      entries[victim] = {experts[i], ++clock};
    }
  return result;
}
