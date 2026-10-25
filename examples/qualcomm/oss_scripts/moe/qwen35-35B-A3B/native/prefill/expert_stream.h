#pragma once

#include <cstdint>

namespace expert_stream {

constexpr int kPhase = 4;
constexpr unsigned kOrderOffset = 2112;
constexpr unsigned kReadyOffset = 3072;
constexpr uint32_t kReady = 1;
constexpr uint32_t kFailed = 2;
constexpr unsigned kTokenShift = 8;

// Low 8 bits select a mapping window; zero upper bits retain legacy AR128.
inline int token_count(int mapping_control) {
  const unsigned encoded = unsigned(mapping_control) >> kTokenShift;
  return encoded ? int(encoded) : 128;
}

// The first 4 KiB remains the bank header. ARM publishes readiness only after
// copying a payload to uncached rpcmem; DSP polls it with cache-bypassing DMA.
static_assert(kOrderOffset >= 32 + 256 * sizeof(uint64_t));
static_assert(kOrderOffset + 256 * sizeof(uint16_t) <= kReadyOffset);
static_assert(kReadyOffset + 256 * sizeof(uint32_t) == 4096);

} // namespace expert_stream
