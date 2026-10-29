#include "bank_mapping.h"
#include "HAP_mem.h"
#include "qurt_memory.h"

namespace moe::dsp::decode {

static BankMapping selected_mappings[2], cached_mapping;

int release_expert_mappings() {
  int error = 0;
  for (auto& mapping : selected_mappings) {
    if (mapping.pointer && HAP_munmap(mapping.pointer, mapping.bytes))
      error = 52;
    mapping = BankMapping{};
  }
  if (cached_mapping.pointer &&
      HAP_munmap(cached_mapping.pointer, cached_mapping.bytes))
    error = 52;
  cached_mapping = BankMapping{};
  return error;
}

void prepare_integer_bank(Sync* sync, const int32_t* descriptor) {
  auto mapping_stamp = tick();
  BankMapping* mapping = nullptr;
  for (auto& m : selected_mappings)
    if (m.fd == descriptor[0]) {
      mapping = &m;
      break;
    }
  if (!mapping)
    for (auto& m : selected_mappings)
      if (m.fd == -1) {
        mapping = &m;
        break;
      }
  if (!mapping)
    sync->error = 24;
  else {
    if (mapping->fd == -1) {
      mapping->pointer = static_cast<uint8_t*>(HAP_mmap2(
          nullptr,
          descriptor[1],
          HAP_PROT_READ | HAP_PROT_WRITE,
          0,
          descriptor[0],
          0));
      mapping->fd = descriptor[0];
      mapping->bytes = descriptor[1];
    }
    if (!mapping->pointer || mapping->pointer == (void*)-1)
      sync->error = 25;
    else {
      // DMA reads bypass DSP caches. Only the global header/offset table is
      // read by the CPU.
      if (mapping->generation != descriptor[2])
        qurt_mem_cache_clean(
            uintptr_t(mapping->pointer),
            4096,
            QURT_MEM_CACHE_INVALIDATE,
            QURT_MEM_DCACHE);
      mapping->generation = descriptor[2];
      sync->bank = mapping->pointer;
      if (*reinterpret_cast<const uint32_t*>(sync->bank) !=
          (SM_EXPERT_BITS == 2 ? 0x4e32424b : 0x4e34424b))
        sync->error = 26;
    }
  }
  metrics[10][1] = tick() - mapping_stamp;
  mapping_stamp = tick();
  if (descriptor[4] >= 0 && !sync->cached_bank) {
    if (cached_mapping.fd != descriptor[4]) {
      if (cached_mapping.pointer)
        HAP_munmap(cached_mapping.pointer, cached_mapping.bytes);
      cached_mapping = {
          descriptor[4],
          0,
          static_cast<uint8_t*>(HAP_mmap2(
              nullptr,
              descriptor[5],
              HAP_PROT_READ | HAP_PROT_WRITE,
              0,
              descriptor[4],
              0)),
          unsigned(descriptor[5])};
    }
    if (!cached_mapping.pointer || cached_mapping.pointer == (void*)-1)
      sync->error = 25;
    sync->cached_bank = cached_mapping.pointer;
  }
  metrics[10][2] = tick() - mapping_stamp;
}

} // namespace moe::dsp::decode
