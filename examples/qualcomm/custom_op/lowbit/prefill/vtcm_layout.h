#pragma once

#include "../../htp_arch.h"

#include <cstddef>
#include <cstdint>

namespace moe_prefill_vtcm {

#if QCOM_HTP_V81
constexpr bool kCompact = true;
#else
constexpr bool kCompact = false;
#endif

constexpr std::size_t kMiB = 1024 * 1024;
constexpr unsigned kWorkers = 8;
constexpr std::size_t kWorkerBytes = 458752;
constexpr std::size_t kInputCodesOffset = kWorkers * kWorkerBytes;
constexpr std::size_t kWorkspaceAlignment = 2048;
constexpr std::size_t kAr128Bytes = (kCompact ? 3 : 12) * kMiB;
constexpr std::size_t kGroupedBytes = 8 * kMiB;
constexpr std::size_t kPartitionedBytes = (kCompact ? 3 : 5) * kMiB;
constexpr bool kRouteInVtcm = !kCompact;
constexpr std::size_t kRouteOffset = 4 * kMiB;
constexpr std::size_t kAr128RouteBytes = 128 * 8 * 2048 * sizeof(float);

constexpr unsigned workersForTokens(unsigned tokens) {
  return kCompact && tokens == 128 ? 4 : kWorkers;
}

constexpr std::size_t inputCodesOffset(unsigned tokens) {
  return workersForTokens(tokens) * kWorkerBytes;
}

constexpr std::size_t scratchBytes(unsigned tokens) {
  return inputCodesOffset(tokens) + tokens * 2048 * sizeof(std::uint16_t) +
      kWorkspaceAlignment - 1;
}

static_assert(kWorkers <= 8 && (128 / kWorkers) % 16 == 0);
static_assert(kWorkerBytes % kWorkspaceAlignment == 0);
static_assert(131072 + 65536 == 196608);
static_assert(196608 + 2 * 65536 == 327680);
static_assert(327680 + 40960 == 368640);
static_assert(368640 + 8 * 1024 * sizeof(std::uint16_t) == 385024);
static_assert(385024 + 4096 == 389120);
static_assert(398592 + 1024 * 8 * sizeof(std::uint16_t) <= 415104);
static_assert(415616 + 256 * sizeof(std::uint32_t) <= kWorkerBytes);
static_assert(447488 + 1024 * 2 * sizeof(float) <= kWorkerBytes);
static_assert(scratchBytes(1024) <= kGroupedBytes);
static_assert(scratchBytes(128) <= kPartitionedBytes);
static_assert(kRouteInVtcm || scratchBytes(128) <= kAr128Bytes);
static_assert(!kRouteInVtcm || kRouteOffset + kAr128RouteBytes <= kAr128Bytes);

} // namespace moe_prefill_vtcm
