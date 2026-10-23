#pragma once
#include <chrono>
#include <cstddef>
#include <cstdint>

using Clock = std::chrono::steady_clock;

inline double elapsed(Clock::time_point started) {
  return std::chrono::duration<double, std::milli>(Clock::now() - started)
      .count();
}

inline size_t align4096(size_t size) {
  return (size + 4095) & ~size_t(4095);
}

uint64_t physical_reads();
