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
constexpr std::size_t kWorkspaceAlignment = 2048;
constexpr std::size_t kAr128Bytes = (kCompact ? 3 : 12) * kMiB;
constexpr std::size_t kGroupedBytes = 8 * kMiB;
constexpr std::size_t kPartitionedBytes = (kCompact ? 3 : 5) * kMiB;
constexpr bool kRouteInVtcm = !kCompact;
constexpr std::size_t kRouteOffset = 4 * kMiB;
constexpr std::size_t kAr128RouteBytes = 128 * 8 * 2048 * sizeof(float);

template <bool Grouped>
struct Layout {
  static constexpr bool kWide = kCompact && Grouped;
  static constexpr unsigned kBatchRows = kWide ? 64 : 32;
  static constexpr unsigned kRowTiles = kBatchRows / 32;
  static constexpr unsigned kInputCodeBytes = kWide ? 1 : 2;
  // Wide groups keep HMX operands in the leading arena and HVX data after it.
  static constexpr std::size_t kGuOffset = kWide ? 0 : 131072;
  static constexpr std::size_t kWeightsOffset = kWide ? 262144 : 196608;
  static constexpr std::size_t kRawOffset = kWide ? 393216 : 368640;
  static constexpr std::size_t kBiasOffset = kWide ? 425984 : 385024;
  static constexpr std::size_t kMetaOffset = kWide ? 131072 : 327680;
  static constexpr std::size_t kHeaderOffset =
      kWide ? kMetaOffset + 40960 : kBiasOffset + 4096;
  static constexpr std::size_t kIdsOffset = kHeaderOffset + 128;
  static constexpr std::size_t kScoresOffset = kIdsOffset + 4096;
  static constexpr std::size_t kRouteOffsetsOffset = kScoresOffset + 4096;
  static constexpr std::size_t kAssignmentsOffset = kRouteOffsetsOffset + 1152;
  static constexpr std::size_t kOrderOffset = kAssignmentsOffset + 16384 + 128;
  static constexpr std::size_t kReadyOffset = kOrderOffset + 512;
  static constexpr std::size_t kDownMetaOffset = kReadyOffset + 1024;
  static constexpr std::size_t kInputScalesOffset = kDownMetaOffset + 30848;
  static constexpr std::size_t kAuxWorkerBytes = kInputScalesOffset + 11264;
  static constexpr std::size_t kHmxWorkerBytes =
      kWide ? 458752 : kAuxWorkerBytes;
  static constexpr std::size_t kWorkerBytes =
      kWide ? kHmxWorkerBytes + kAuxWorkerBytes : kHmxWorkerBytes;

  static_assert(kWorkerBytes % kWorkspaceAlignment == 0);
  static_assert(
      kGuOffset + kBatchRows * 512 * 4 <=
      (kWide ? kMetaOffset : kWeightsOffset));
  static_assert(kRawOffset + kRowTiles * 16384 <= kBiasOffset);
  static_assert(kBiasOffset + 4096 <= kHmxWorkerBytes);
  static_assert(
      kAssignmentsOffset + 1024 * 8 * sizeof(std::uint16_t) <= kOrderOffset);
  static_assert(kDownMetaOffset + 24576 <= kInputScalesOffset);
  static_assert(
      kInputScalesOffset + 1024 * 2 * sizeof(float) <= kAuxWorkerBytes);
  static_assert(!kWide || kWorkers * kHmxWorkerBytes <= 4 * kMiB);
};

constexpr unsigned workersForTokens(unsigned tokens) {
  return kCompact && tokens == 128 ? 4 : kWorkers;
}

constexpr std::size_t inputCodesOffset(unsigned tokens) {
  return workersForTokens(tokens) *
      (tokens > 128 ? Layout<true>::kWorkerBytes : Layout<false>::kWorkerBytes);
}

constexpr std::size_t scratchBytes(unsigned tokens) {
  const unsigned code_bytes = tokens > 128 ? Layout<true>::kInputCodeBytes
                                           : Layout<false>::kInputCodeBytes;
  return inputCodesOffset(tokens) + tokens * 2048 * code_bytes +
      kWorkspaceAlignment - 1;
}

static_assert(kWorkers <= 8 && (128 / kWorkers) % 16 == 0);
static_assert(Layout<false>::kWorkerBytes == 458752);
static_assert(Layout<false>::kInputScalesOffset == 447488);
static_assert(scratchBytes(1024) <= kGroupedBytes);
static_assert(scratchBytes(128) <= kPartitionedBytes);
static_assert(kRouteInVtcm || scratchBytes(128) <= kAr128Bytes);
static_assert(!kRouteInVtcm || kRouteOffset + kAr128RouteBytes <= kAr128Bytes);

} // namespace moe_prefill_vtcm
