#pragma once

#include "state_manager.h"
#include <cstdint>

// Views borrow live decode buffers. Expert payloads are already shared; only
// slot identity and LRU metadata cross the prefill/decode DSO boundary.
struct Qwen35LayerState {
  QuantizedCacheView cache[2];
  int32_t experts[64];
  uint64_t stamps[64];
  int32_t routes[8];
};

struct Qwen35SessionState {
  int position = 0, capacity = 0, slots = 0;
  bool append_only = true;
  bool aborted = false;
  Qwen35LayerState layer[40];
};
