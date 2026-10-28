#include "../native/prefill/vtcm_layout.h"

#include <array>
#include <cassert>
#include <cstdio>
#include <vector>

int main() {
  using namespace moe_prefill_vtcm;
  struct Region {
    std::size_t begin, end;
  };
  constexpr std::array<Region, 14> regions{{
      {0, 131072},
      {131072, 196608},
      {196608, 327680},
      {327680, 368640},
      {368640, 385024},
      {385024, 389120},
      {389120, 389248},
      {389248, 393344},
      {393344, 397440},
      {397440, 398468},
      {398592, 414976},
      {415104, 415616},
      {415616, 416640},
      {447488, 455680},
  }};
  assert(kWorkers == 8 && kWorkerBytes == 458752);
  assert(kGroupedBytes == 8 * kMiB);
#if SM_MOE_DSP_ARCH == 81
  assert(kCompact && !kRouteInVtcm);
  assert(kAr128Bytes == 3 * kMiB && kPartitionedBytes == 3 * kMiB);
  assert(workersForTokens(128) == 4);
  assert(inputCodesOffset(128) < kInputCodesOffset);
  assert(scratchBytes(128) == 2361343);
#else
  assert(!kCompact && kRouteInVtcm);
  assert(kAr128Bytes == 12 * kMiB && kPartitionedBytes == 5 * kMiB);
  assert(workersForTokens(128) == 8);
  assert(inputCodesOffset(128) == kInputCodesOffset);
#endif
  assert(scratchBytes(1024) == 7866367);
  for (std::size_t i = 0; i < regions.size(); ++i) {
    assert(regions[i].begin < regions[i].end);
    assert(regions[i].end <= kWorkerBytes);
    for (std::size_t j = i + 1; j < regions.size(); ++j)
      assert(
          regions[i].end <= regions[j].begin ||
          regions[j].end <= regions[i].begin);
  }
  for (unsigned tokens = 128; tokens <= 1024; tokens += 128) {
    const auto workers = workersForTokens(tokens);
    const auto codes = inputCodesOffset(tokens);
    if (tokens > 128)
      assert(workers == 8 && codes == kInputCodesOffset);
    assert(codes == workers * kWorkerBytes);
    assert(tokens % workers == 0 && (tokens / workers) % 16 == 0);
    std::vector<unsigned> visits(tokens);
    for (unsigned worker = 0; worker < workers; ++worker) {
      for (const auto& region : regions)
        assert(worker * kWorkerBytes + region.end <= codes);
      for (unsigned first = worker * (tokens / workers);
           first < (worker + 1) * (tokens / workers);
           first += 16)
        for (unsigned row = first; row < first + 16; ++row)
          ++visits.at(row);
    }
    for (auto count : visits)
      assert(count == 1);
    std::size_t worst = 0;
    for (std::size_t base = 0; base < kWorkspaceAlignment; ++base) {
      const auto aligned =
          (base + kWorkspaceAlignment - 1) & ~(kWorkspaceAlignment - 1);
      const auto end = aligned + codes + tokens * 2048 * sizeof(std::uint16_t);
      const auto used = end - base;
      if (used > worst)
        worst = used;
      assert(used <= scratchBytes(tokens));
      assert(used <= kGroupedBytes);
      for (unsigned worker = 0; worker < workers; ++worker)
        assert((aligned + worker * kWorkerBytes) % kWorkspaceAlignment == 0);
      if (tokens == 128) {
        assert(used <= kPartitionedBytes);
        if (!kRouteInVtcm)
          assert(used <= kAr128Bytes);
      }
    }
    assert(worst == scratchBytes(tokens));
    if (tokens == 128 && kRouteInVtcm) {
      assert(codes + tokens * 2048 * sizeof(std::uint16_t) <= kRouteOffset);
      assert(kRouteOffset + kAr128RouteBytes <= kAr128Bytes);
    }
  }
  for (unsigned width : {128u, 256u, 384u, 512u})
    for (bool gu : {false, true}) {
      const auto padded_k = gu ? 2048u : width;
      const auto grid = gu ? 1024u : 512u;
      const auto groups = (padded_k + grid - 1) / grid;
      const auto transfer_blocks = 65536 / (padded_k * 16);
      const auto blocks = transfer_blocks < 8 ? transfer_blocks : 8;
      assert(blocks * groups * 1024 * 2 <= 385024 - 368640);
      assert(blocks * groups * 64 * 4 <= 389120 - 385024);
    }
  std::printf(
      "PASS arch=%d active128=%u active1024=%u ar128=%zu grouped=%zu scratch128=%zu scratch1024=%zu\n",
      SM_MOE_DSP_ARCH,
      workersForTokens(128),
      workersForTokens(1024),
      kAr128Bytes,
      kGroupedBytes,
      scratchBytes(128),
      scratchBytes(1024));
}
