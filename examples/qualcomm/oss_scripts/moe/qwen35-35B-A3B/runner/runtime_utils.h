#pragma once
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

inline double elapsed(Clock::time_point started) {
  return std::chrono::duration<double, std::milli>(Clock::now() - started)
      .count();
}

inline size_t align4096(size_t size) {
  return (size + 4095) & ~size_t(4095);
}

uint64_t physical_reads();
void validate_rope_tables(size_t cos_bytes, size_t sin_bytes, size_t positions);

std::vector<uint8_t> bytes(const std::string& name);
void save(const std::string& name, const void* data, size_t size);
