#include "resident_handoff.h"
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <stdexcept>

int configuredExpertSlots() {
  const char* value = std::getenv("EXPERT_CACHE_SLOTS");
  if (!value || std::strcmp(value, "64") == 0)
    return 64;
  if (std::strcmp(value, "32") == 0)
    return 32;
  if (std::strcmp(value, "16") == 0)
    return 16;
  throw std::runtime_error("EXPERT_CACHE_SLOTS must be 16, 32 or 64");
}

uint64_t residentFingerprint(
    const uint8_t* data,
    uint64_t stride,
    uint64_t offset,
    int capacity,
    const int32_t* ids) {
  uint64_t hash = 14695981039346656037ULL;
  for (int slot = 0; slot < capacity; slot++) {
    if (ids[slot] < 0)
      continue;
    for (int sample : {0, 32, 256, 1024, 8192, 65536})
      for (int byte = 0; byte < 16; byte++)
        hash = (hash ^ data[offset + slot * stride + sample + byte]) *
            1099511628211ULL;
  }
  return hash;
}
