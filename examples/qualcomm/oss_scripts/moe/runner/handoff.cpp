#include "handoff.h"
#include <initializer_list>

uint64_t expertFingerprint(const uint8_t* data, uint64_t stride) {
  uint64_t hash = 14695981039346656037ULL;
  for (int slot = 0; slot < 64; slot++)
    for (int offset : {0, 32, 256, 1024, 8192, 65536})
      for (int byte = 0; byte < 16; byte++)
        hash = (hash ^ data[4096 + slot * stride + offset + byte]) *
            1099511628211ULL;
  return hash;
}
