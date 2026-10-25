#include "HAP_mem.h"
#include <cstddef>

extern "C" void* __real_HAP_mmap2(void*, size_t, int, int, int, long);

extern "C" void* __wrap_HAP_mmap2(
    void* address,
    size_t size,
    int protection,
    int flags,
    int fd,
    long offset) {
  // Fail one bank map, then let the same handle retry the real SDK mapping.
  static bool failed = false;
  if (size == 8192 && !failed) {
    failed = true;
    return reinterpret_cast<void*>(-1);
  }
  return __real_HAP_mmap2(address, size, protection, flags, fd, offset);
}
