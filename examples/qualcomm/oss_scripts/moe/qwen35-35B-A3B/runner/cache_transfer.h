#pragma once

#include <cstddef>
#include <cstdint>

enum class CacheDType : uint32_t { AffineUint16 = 0, Float16 = 1 };

struct QuantizedCacheView {
  uint16_t* data;
  size_t elements;
  float scale;
  int32_t offset;
  CacheDType dtype = CacheDType::AffineUint16;
};

enum class CacheLayout { State, Key, Value };

void clear_cache(const QuantizedCacheView& cache);
void copy_cache_range(
    const QuantizedCacheView& source,
    const QuantizedCacheView& destination,
    CacheLayout layout,
    size_t source_begin,
    size_t destination_begin,
    size_t count);

void append_prefill_kv(
    const QuantizedCacheView& history,
    const uint16_t* block,
    const uint16_t* encoding,
    uint16_t* handoff,
    CacheLayout layout,
    size_t valid_tokens);
