#pragma once

#include "cache_transfer.h"
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

struct CacheEncoding {
  float scale;
  int32_t offset;
  uint64_t elements;
  CacheDType dtype = CacheDType::AffineUint16;
};

inline uint32_t cache_encoding_version(const std::vector<uint8_t>& data) {
  if (data.size() < 16 || std::memcmp(data.data(), "Q36CACHE", 8))
    throw std::runtime_error("Invalid cache encoding header");
  uint32_t version, layers;
  std::memcpy(&version, data.data() + 8, 4);
  std::memcpy(&layers, data.data() + 12, 4);
  if ((version != 1 && version != 2) || layers != 40 ||
      data.size() != 16 + 80 * (version == 1 ? 16 : 24))
    throw std::runtime_error("Unsupported cache encoding version or size");
  return version;
}

inline CacheEncoding read_cache_encoding(
    const std::vector<uint8_t>& data, size_t index) {
  const uint32_t version = cache_encoding_version(data);
  if (index >= 80)
    throw std::runtime_error("Invalid cache encoding index");
  const auto* entry = data.data() + 16 + index * (version == 1 ? 16 : 24);
  CacheEncoding result{};
  std::memcpy(&result.scale, entry, 4);
  std::memcpy(&result.offset, entry + 4, 4);
  std::memcpy(&result.elements, entry + 8, 8);
  if (version == 2) {
    uint32_t dtype, reserved;
    std::memcpy(&dtype, entry + 16, 4);
    std::memcpy(&reserved, entry + 20, 4);
    if (dtype > uint32_t(CacheDType::Float16) || reserved)
      throw std::runtime_error("Invalid typed cache encoding");
    result.dtype = CacheDType(dtype);
  }
  if (!result.elements ||
      (result.dtype == CacheDType::Float16
           ? result.scale != 0 || result.offset != 0
           : !std::isfinite(result.scale) || result.scale <= 0 ||
                 result.offset > 0 || result.offset < -65535))
    throw std::runtime_error("Invalid cache encoding values");
  if (result.dtype == CacheDType::Float16 && (index % 2 != 1 || index / 2 % 4 == 3))
    throw std::runtime_error("Only GDN recurrent state may use FP16 cache");
  return result;
}

inline void validate_cache_encoding_payload(
    const std::vector<uint8_t>& data, const uint8_t* embedded, size_t size) {
  cache_encoding_version(data);
  for (size_t index = 0; index < 80; ++index)
    read_cache_encoding(data, index);
  if (!embedded || size != data.size() || std::memcmp(data.data(), embedded, size))
    throw std::runtime_error("Embedded and runtime cache encodings differ");
}
