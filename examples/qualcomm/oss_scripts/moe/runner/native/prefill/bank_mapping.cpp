#include "bank_mapping.h"
#include "../../../../../custom_op/lowbit/expert_stream.h"
#include "HAP_mem.h"
#include "qurt_memory.h"

// Only the current call uses its bank. Two DSP VA windows suffice while the
// 40-layer ARM cache keeps every resident payload.
static BankMapping mappings[2];

int release_banks() {
  int error = 0;
  for (auto& m : mappings) {
    if (m.pointer && HAP_munmap(m.pointer, m.bytes))
      error = 52;
    m = BankMapping{};
  }
  return error;
}

void prepare_integer_bank(Sync* sync, const int32_t* descriptor) {
  BankMapping* mapping = nullptr;
  for (auto& m : mappings)
    if (m.fd == descriptor[0]) {
      mapping = &m;
      break;
    }
  if (!mapping)
    for (auto& m : mappings)
      if (m.fd == -1) {
        mapping = &m;
        break;
      }
  if (!mapping) {
    auto& victim = mappings[unsigned(descriptor[3]) % 2];
    if (victim.pointer && HAP_munmap(victim.pointer, victim.bytes))
      sync->error = 52;
    else {
      victim = BankMapping{};
      mapping = &victim;
    }
  }
  if (!mapping)
    sync->error = 24;
  else {
    if (mapping->fd == -1) {
      auto* pointer = static_cast<uint8_t*>(HAP_mmap2(
          nullptr,
          descriptor[1],
          HAP_PROT_READ | HAP_PROT_WRITE,
          0,
          descriptor[0],
          0));
      if (pointer && pointer != reinterpret_cast<void*>(-1))
        *mapping = {descriptor[0], -1, pointer, unsigned(descriptor[1])};
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
      if (*reinterpret_cast<const uint32_t*>(sync->bank) != 0x4e32424b)
        sync->error = 26;
      if (!sync->error && descriptor[4] == expert_stream::kResidentPhase) {
        const auto* resident = reinterpret_cast<const int32_t*>(
            sync->bank + expert_stream::kResidentOffset);
        auto& other = mappings[mapping == &mappings[0] ? 1 : 0];
        if (resident[0] < 0 || resident[1] < 4096 ||
            resident[0] == descriptor[0]) {
          sync->error = 26;
          return;
        }
        if (other.fd != resident[0] || other.bytes != unsigned(resident[1])) {
          if (other.pointer && HAP_munmap(other.pointer, other.bytes)) {
            sync->error = 52;
            return;
          }
          other = BankMapping{};
          auto* pointer = static_cast<uint8_t*>(HAP_mmap2(
              nullptr, resident[1], HAP_PROT_READ | HAP_PROT_WRITE,
              0, resident[0], 0));
          if (!pointer || pointer == reinterpret_cast<void*>(-1)) {
            sync->error = 25;
            return;
          }
          other = {resident[0], -1, pointer, unsigned(resident[1])};
        }
        sync->resident_bank = other.pointer;
        const auto* offsets = reinterpret_cast<const uint64_t*>(sync->bank + 32);
        for (int e = 0; e < 256; ++e) {
          const uint64_t offset = offsets[e] & ~expert_stream::kResident;
          const size_t bytes = offsets[e] & expert_stream::kResident
              ? other.bytes : mapping->bytes;
          if (offset && (offset < 4096 || offset > bytes - 128))
            sync->error = 26;
        }
      }
    }
  }
}
