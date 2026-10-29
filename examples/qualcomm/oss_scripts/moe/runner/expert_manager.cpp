#include "expert_manager.h"

#include "remote.h"
#include <sys/ioctl.h>
#include <linux/dma-buf.h>

namespace moe::decode {

ExpertCache::ExpertCache(
    Runtime& runtime,
    int count,
    size_t maximum,
    size_t budget,
    bool direct,
    const std::string& root)
    : layers(count),
      capacity(configuredExpertSlots()),
      stride(maximum),
      entries(count * capacity),
      mapped(direct) {
  bytes = size_t(layers) * capacity * stride;
  if (bytes > budget)
    throw std::runtime_error("expert cache exceeds configured memory budget");
  imported = getenv("INHERIT_EXPERT_CACHE") &&
      std::string(getenv("INHERIT_EXPERT_CACHE")) == "1";
  if (imported) {
    // prefill 与 decode 同进程交接：映射已有 fd，校验内容，不重新读 UFS
    // 或复制权重。
    const auto started = Clock::now();
    const auto raw = ::bytes(root + "/resident-handoff.bin");
    if (raw.size() != sizeof(transfer))
      throw std::runtime_error("resident handoff size");
    memcpy(&transfer, raw.data(), sizeof(transfer));
    if (memcmp(transfer.magic, "Q36DATA2", 8) || transfer.layers != 40 ||
        transfer.capacity != uint32_t(capacity) || stride != transfer.stride ||
        !mapped)
      throw std::runtime_error("resident handoff dimensions");
    if (transfer.data_offset != 4096)
      throw std::runtime_error("reuse data offset");
    data_offset = transfer.data_offset;
    bytes = size_t(layers) * (data_offset + capacity * stride);
    if (bytes > budget)
      throw std::runtime_error("expert cache exceeds configured memory budget");
    for (int layer = 0; layer < layers; layer++) {
      const auto& metadata = transfer.layer[layer];
      std::set<int> unique;
      if (metadata.hot_mask)
        throw std::runtime_error("all-LRU cache forbids hot slots");
      for (int slot = 0; slot < capacity; slot++)
        if (int id = metadata.ids[slot]; true)
          if (id < -1 || id >= 256 || (id >= 0 && !unique.insert(id).second) ||
              (id == -1 && metadata.stamps[slot]))
            throw std::runtime_error("resident seed IDs");
      if (metadata.bytes != data_offset + capacity * stride)
        throw std::runtime_error("resident pool size");
      buffers.push_back(
          std::make_shared<Buffer>(runtime, Buffer::ImportedFd{metadata.fd}, metadata.bytes));
      auto& b = *buffers.back();
      dma_buf_sync sync{DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ};
      if (ioctl(b.fd, DMA_BUF_IOCTL_SYNC, &sync))
        throw std::runtime_error("resident begin read");
      if (residentFingerprint(
              static_cast<const uint8_t*>(b.data),
              stride,
              data_offset,
              capacity,
              metadata.ids) != metadata.fingerprint)
        throw std::runtime_error("resident weight fingerprint");
      sync.flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ;
      if (ioctl(b.fd, DMA_BUF_IOCTL_SYNC, &sync))
        throw std::runtime_error("resident end read");
      check(
          fastrpc_mmap(
              CDSP_DOMAIN_ID, b.fd, b.data, 0, b.size, FASTRPC_MAP_FD_DELAYED),
          "map resident layer cache");
    }
    printf(
        "EXPERT_RESIDENT_IMPORT layers=40 experts=%d buffer_bytes=%zu fingerprints_verified=40 read_bytes=0 copy_bytes=0 import_ms=%.6f\n",
        layers * capacity,
        bytes,
        elapsed(started));
  } else if (bytes && mapped) {
    for (int i = 0; i < layers; i++) {
      buffers.push_back(std::make_shared<Buffer>(runtime, capacity * stride));
      auto& b = *buffers.back();
      check(
          fastrpc_mmap(
              CDSP_DOMAIN_ID, b.fd, b.data, 0, b.size, FASTRPC_MAP_FD_DELAYED),
          "map layer expert cache");
    }
  } else if (
      bytes && posix_memalign(reinterpret_cast<void**>(&data), 4096, bytes))
    throw std::runtime_error("expert cache allocation");
}

ExpertCache::~ExpertCache() {
  for (auto& b : buffers) {
    const int status = fastrpc_munmap(CDSP_DOMAIN_ID, b->fd, b->data, b->size);
    if (status)
      fprintf(stderr, "cache unmap failed fd=%d status=%d\n", b->fd, status);
  }
  free(data);
}

uint8_t* ExpertCache::location(int layer, int slot) {
  return mapped ? static_cast<uint8_t*>(buffers[layer]->data) + data_offset +
          slot * stride
                : data + (layer * capacity + slot) * stride;
}

void ExpertCache::publish(int layer) {
  if (buffers.empty())
    return;
  dma_buf_sync sync{DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE};
  if (ioctl(fd(layer), DMA_BUF_IOCTL_SYNC, &sync))
    throw std::runtime_error("expert cache sync");
}

void ExpertCache::use_direct(
    int layer,
    const int* ids,
    ExpertRead* requests,
    uint64_t* offsets) {
  // offset 的最高位表示地址来自当前层常驻 cache，而非临时 bank 的 payload
  // 区。
  if (buffers.empty())
    return;
  auto* base = static_cast<uint8_t*>(buffers[layer]->data);
  for (int i = 0; i < 8; i++) {
    const uint8_t* pointer = requests[i].cache_destination
        ? requests[i].cache_destination
        : requests[i].cached;
    if (pointer) {
      offsets[ids[i]] = (1ULL << 63) | uint64_t(pointer - base);
      requests[i].destination = nullptr;
    }
  }
}

int ExpertCache::find(int layer, int expert) const {
  for (int slot = 0; slot < capacity; slot++)
    if (entries[layer * capacity + slot].expert == expert)
      return slot;
  return -1;
}

int ExpertCache::reserve_prefetch(
    int layer,
    int expert,
    const std::array<int, 8>& previous) {
  if (find(layer, expert) >= 0)
    return -1;
  int victim = -1;
  for (int slot = 0; slot < capacity; slot++) {
    const auto& entry = entries[layer * capacity + slot];
    if (entry.pending ||
        std::find(previous.begin(), previous.end(), entry.expert) !=
            previous.end())
      continue;
    if (victim < 0 || entry.stamp < entries[layer * capacity + victim].stamp)
      victim = slot;
  }
  if (victim >= 0)
    entries[layer * capacity + victim] = {expert, 0, true, true};
  return victim;
}

void ExpertCache::finish_prefetch(int layer, int slot, int expert) {
  auto& entry = entries[layer * capacity + slot];
  if (!entry.pending || entry.expert != expert)
    throw std::runtime_error("prefetch slot ownership");
  entry.pending = false;
}

void ExpertCache::verify_cache() {
  for (int layer = 0; layer < layers; layer++) {
    std::set<int> unique;
    for (int slot = 0; slot < capacity; slot++) {
      const auto& entry = entries[layer * capacity + slot];
      if (entry.pending || entry.expert < -1 || entry.expert >= 256 ||
          (entry.expert >= 0 && !unique.insert(entry.expert).second) ||
          (entry.expert == -1 && entry.stamp))
        throw std::runtime_error("invalid final cold cache");
    }
  }
  printf(
      "EXPERT_CACHE_VERIFY layers=%d hot_slots=0 mutable_slots=%d\n",
      layers,
      layers * capacity);
}

void ExpertCache::seed_layer(int layer, const std::vector<int>& ids) {
  if (int(ids.size()) > capacity)
    throw std::runtime_error("seed exceeds layer cache capacity");
  std::set<int> unique;
  for (size_t i = 0; i < ids.size(); i++) {
    if (ids[i] < 0 || ids[i] >= 256 || !unique.insert(ids[i]).second)
      throw std::runtime_error("invalid seed expert");
    entries[layer * capacity + i] = {ids[i], ++clock};
  }
}

void ExpertCache::reset() {
  std::fill(entries.begin(), entries.end(), Entry{});
  clock = 0;
  if (imported)
    for (int layer = 0; layer < layers; layer++)
      for (int slot = 0; slot < capacity; slot++) {
        const auto& metadata = transfer.layer[layer];
        entries[layer * capacity + slot] = {
            metadata.ids[slot], metadata.stamps[slot]};
        clock = std::max(clock, metadata.stamps[slot]);
      }
}

ExpertCache::Hits
ExpertCache::prepare(int layer, const int* ids, ExpertRead* requests) {
  // 命中先钉住；缺失只预留其他槽，真正可用要等 reader 发布 ready 标志。
  std::array<bool, 256> pinned{};
  Hits hits;
  for (int i = 0; i < 8; i++)
    for (int slot = 0; slot < capacity; slot++) {
      auto& entry = entries[layer * capacity + slot];
      if (entry.expert == ids[i]) {
        if (entry.pending)
          throw std::runtime_error("expert still loading");
        hits.predicted += entry.prefetched;
        entry.prefetched = false;
        requests[i].cached = location(layer, slot);
        entry.stamp = ++clock;
        pinned[slot] = true;
        ++hits.cold;
        break;
      }
    }
  // Partition by layer so a 40-layer sweep cannot evict the preceding token's
  // entire working set.
  for (int i = 0; i < 8; i++)
    if (!requests[i].cached) {
      int victim = -1;
      for (int slot = 0; slot < capacity; slot++)
        if (!pinned[slot] && !entries[layer * capacity + slot].pending &&
            (victim < 0 ||
             entries[layer * capacity + slot].stamp <
                 entries[layer * capacity + victim].stamp))
          victim = slot;
      if (victim < 0)
        throw std::runtime_error("no unpinned cold expert slot");
      auto& entry = entries[layer * capacity + victim];
      entry = {ids[i], ++clock};
      pinned[victim] = true;
      requests[i].cache_destination = location(layer, victim);
    }
  return hits;
}

} // namespace moe::decode
