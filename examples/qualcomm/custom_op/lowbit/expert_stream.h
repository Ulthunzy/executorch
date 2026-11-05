#pragma once

#include <cstdint>

namespace expert_stream {

constexpr int kPhase = 4;
constexpr int kResidentPhase = 5;
constexpr unsigned kOrderOffset = 2112;
constexpr unsigned kResidentOffset = 2624;
constexpr unsigned kReadyOffset = 3072;
constexpr uint64_t kResident = uint64_t(1) << 63;
constexpr uint32_t kReady = 1;
constexpr uint32_t kFailed = 2;
constexpr unsigned kTokenShift = 8;

// Low 8 bits select a mapping window. Separate physical and valid lengths
// let AR128 padding skip expert loads and arithmetic without changing IO sizes.
inline int token_count(int mapping_control) {
  const unsigned encoded = (unsigned(mapping_control) >> kTokenShift) & 2047;
  return encoded ? int(encoded) : 128;
}

inline int valid_token_count(int control) {
  const unsigned valid = unsigned(control) >> 20;
  return valid ? int(valid) : token_count(control);
}

inline int mapping_control(int window, int physical, int valid) {
  return window | (physical == 128 ? 0 : physical << kTokenShift) |
      (valid == physical ? 0 : valid << 20);
}

// The first 4 KiB remains the bank header. ARM publishes readiness only after
// copying a payload to uncached rpcmem; DSP polls it with cache-bypassing DMA.
static_assert(kOrderOffset >= 32 + 256 * sizeof(uint64_t));
static_assert(kOrderOffset + 256 * sizeof(uint16_t) <= kResidentOffset);
static_assert(kResidentOffset + 2 * sizeof(int32_t) <= kReadyOffset);
static_assert(kReadyOffset + 256 * sizeof(uint32_t) == 4096);

} // namespace expert_stream
