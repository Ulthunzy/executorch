#include "runtime_utils.h"
#include <fstream>
#include <string>

uint64_t physical_reads() {
  std::ifstream file("/proc/self/io");
  std::string key;
  uint64_t value;
  while (file >> key >> value)
    if (key == "read_bytes:")
      return value;
  return 0;
}
