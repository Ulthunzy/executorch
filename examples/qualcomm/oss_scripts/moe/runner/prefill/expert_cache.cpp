#include "expert_cache.h"

#include <algorithm>
#include <array>
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

void ExpertCache::advance(
    const int32_t* ids,
    int tokens,
    int first_token,
    uint64_t stamp_offset) {
  std::array<uint64_t, 256> stamps{};
  std::array<bool, 256> present{};
  for (const auto& entry : entries)
    if (entry.expert >= 0) {
      present[entry.expert] = true;
      stamps[entry.expert] = entry.stamp;
    }
  for (int i = 0; i < tokens * 8; i++) {
    const int expert = ids[i];
    if (expert < 0 || expert >= 256)
      throw std::runtime_error("LRU route ID");
    present[expert] = true;
    stamps[expert] = stamp_offset + uint64_t(first_token) * 8 + i + 1;
  }
  std::vector<int> recent;
  for (int e = 0; e < 256; e++)
    if (present[e])
      recent.push_back(e);
  std::sort(recent.begin(), recent.end(), [&](int a, int b) {
    return stamps[a] > stamps[b];
  });
  recent.resize(std::min(recent.size(), entries.size()));
  std::vector<int> retained(entries.size(), -1);
  for (int expert : recent) {
    const int slot = find(expert);
    if (slot >= 0)
      retained[slot] = expert;
  }
  for (int expert : recent)
    if (std::find(retained.begin(), retained.end(), expert) == retained.end())
      *std::find(retained.begin(), retained.end(), -1) = expert;
  for (size_t slot = 0; slot < entries.size(); slot++) {
    const int expert = retained[slot];
    entries[slot] = {expert, expert < 0 ? 0 : stamps[expert], false};
  }
  clock = stamp_offset + uint64_t(first_token + tokens) * 8;
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
