#include "expert_io.h"

void ExpertCache::export_state(Qwen35SessionState& state) const {
  state.slots = capacity;
  for (int layer = 0; layer < layers; ++layer)
    for (int slot = 0; slot < capacity; ++slot) {
      const auto& entry = entries[layer * capacity + slot];
      if (entry.pending)
        throw std::runtime_error("unfinished expert load during state handoff");
      state.layer[layer].experts[slot] = entry.expert;
      state.layer[layer].stamps[slot] = entry.stamp;
    }
}

void ExpertCache::import_state(const Qwen35SessionState& state) {
  if (state.slots != capacity)
    throw std::runtime_error("expert cache capacity changed across prefill");
  for (int layer = 0; layer < layers; ++layer) {
    std::set<int> seen;
    for (int slot = 0; slot < capacity; ++slot) {
      const int expert = state.layer[layer].experts[slot];
      if (expert < -1 || expert >= 256 ||
          (expert >= 0 && !seen.insert(expert).second))
        throw std::runtime_error("invalid prefill expert slot metadata");
    }
  }
  for (int layer = 0; layer < layers; ++layer)
    for (int slot = 0; slot < capacity; ++slot) {
      auto& entry = entries[layer * capacity + slot];
      entry = {
          state.layer[layer].experts[slot],
          state.layer[layer].stamps[slot],
          false,
          false};
      clock = std::max(clock, entry.stamp);
    }
}
