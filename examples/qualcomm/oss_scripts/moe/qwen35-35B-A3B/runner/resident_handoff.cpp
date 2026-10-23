#include "resident_handoff.h"
#include <initializer_list>

uint64_t
residentFingerprint(const uint8_t* data, uint64_t stride, uint64_t offset) {
  uint64_t hash = 14695981039346656037ULL;
  for (int slot = 0; slot < kExpertSlots; slot++)
    for (int sample : {0, 32, 256, 1024, 8192, 65536})
      for (int byte = 0; byte < 16; byte++)
        hash = (hash ^ data[offset + slot * stride + sample + byte]) *
            1099511628211ULL;
  return hash;
}
