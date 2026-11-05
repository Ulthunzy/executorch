#include "state_manager.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {
void validate(const QuantizedCacheView& view) {
  if (!view.data || !view.elements)
    throw std::runtime_error("Invalid cache storage");
  if (view.dtype == CacheDType::Float16) {
    if (view.scale != 0 || view.offset != 0)
      throw std::runtime_error("FP16 cache must not carry affine encoding");
    return;
  }
  if (view.dtype != CacheDType::AffineUint16 || !std::isfinite(view.scale) ||
      view.scale <= 0 || view.offset > 0 || view.offset < -65535)
    throw std::runtime_error("Invalid UINT16 cache view");
}
} // namespace

void clear_cache(const QuantizedCacheView& cache) {
  validate(cache);
  std::fill(cache.data, cache.data + cache.elements,
            cache.dtype == CacheDType::Float16 ? uint16_t(0) : uint16_t(-cache.offset));
}

void copy_cache_range(
    const QuantizedCacheView& source,
    const QuantizedCacheView& destination,
    CacheLayout layout,
    size_t source_begin,
    size_t destination_begin,
    size_t count,
    bool clear_unused) {
  validate(source);
  validate(destination);
  if (source.dtype != destination.dtype)
    throw std::runtime_error("Cache transfer requires matching storage dtypes");
  if (source.dtype == CacheDType::Float16 && layout != CacheLayout::State)
    throw std::runtime_error("FP16 cache is supported only for GDN state");
  const size_t width = layout == CacheLayout::State ? 1 : 512;
  if (source.elements % width || destination.elements % width ||
      source_begin > source.elements / width ||
      count > source.elements / width - source_begin ||
      destination_begin > destination.elements / width ||
      count > destination.elements / width - destination_begin)
    throw std::runtime_error("Cache transfer exceeds tensor dimensions");
  const bool identity =
      source.scale == destination.scale && source.offset == destination.offset;
  const bool aliased = source.data == destination.data;
  if (aliased && source.elements != destination.elements &&
      (!identity ||
       (destination.elements > source.elements &&
        destination_begin < source_begin) ||
       (destination.elements < source.elements &&
        destination_begin > source_begin)))
    throw std::runtime_error("Unsupported overlapping cache transfer");
  std::vector<uint16_t> mapping;
  if (!identity) {
    mapping.resize(65536);
    for (size_t i = 0; i < mapping.size(); ++i)
      mapping[i] = std::lround(std::clamp(
          (double(i) + source.offset) * source.scale / destination.scale -
              destination.offset,
          0.,
          65535.));
  }
  auto copy = [&](size_t from, size_t to, size_t length) {
    if (identity)
      std::memmove(destination.data + to, source.data + from, length * 2);
    else {
      const bool backwards = aliased && to > from;
      for (size_t j = 0; j < length; ++j) {
        const size_t i = backwards ? length - 1 - j : j;
        destination.data[to + i] = mapping[source.data[from + i]];
      }
    }
  };
  if (layout == CacheLayout::State) {
    copy(source_begin, destination_begin, count);
  } else if (layout == CacheLayout::Key) {
    for (size_t i = 0; i < 512; ++i) {
      const size_t row = aliased && destination.elements > source.elements
          ? 511 - i : i;
      copy(
          row * (source.elements / 512) + source_begin,
          row * (destination.elements / 512) + destination_begin,
          count);
    }
  } else {
    for (size_t i = 0; i < 2; ++i) {
      const size_t head = aliased && destination.elements > source.elements
          ? 1 - i : i;
      copy(
          head * (source.elements / 2) + source_begin * 256,
          head * (destination.elements / 2) + destination_begin * 256,
          count * 256);
    }
  }
  if (clear_unused) {
    const size_t rows = layout == CacheLayout::State ? 1
        : layout == CacheLayout::Key ? 512 : 2;
    const size_t step = layout == CacheLayout::Value ? 256 : 1;
    const size_t row_elements = destination.elements / rows;
    const uint16_t zero = destination.dtype == CacheDType::Float16
        ? uint16_t(0) : uint16_t(-destination.offset);
    for (size_t row = 0; row < rows; ++row) {
      auto* data = destination.data + row * row_elements;
      std::fill(data, data + destination_begin * step, zero);
      std::fill(data + (destination_begin + count) * step,
                data + row_elements, zero);
    }
  }
}

void append_prefill_kv(
    const QuantizedCacheView& history,
    const uint16_t* block,
    const uint16_t* encoding,
    uint16_t* handoff,
    CacheLayout layout,
    size_t valid_tokens) {
  validate(history);
  if (history.dtype != CacheDType::AffineUint16 || history.elements % 512 || !block ||
      layout == CacheLayout::State || valid_tokens < 1 || valid_tokens > 128)
    throw std::runtime_error("invalid padded KV block");
  const size_t width = layout == CacheLayout::Key ? 1 : 256;
  const size_t rows = 512 / width;
  const size_t past = history.elements / rows, window = past + 128 * width;
  for (size_t row = 0; row < rows; ++row) {
    auto* retained = history.data + row * past;
    if (!handoff) {
      const size_t appended = valid_tokens * width;
      const size_t kept = std::min(past, appended);
      std::memmove(retained, retained + kept, (past - kept) * 2);
      const auto* from = block + row * 128 * width + appended - kept;
      if (encoding)
        for (size_t i = 0; i < kept; ++i)
          retained[past - kept + i] = encoding[from[i]];
      else
        std::memcpy(retained + past - kept, from, kept * 2);
      continue;
    }
    auto* full = handoff + row * window;
    std::memcpy(full, retained, past * 2);
    if (encoding)
      for (size_t i = 0; i < valid_tokens * width; ++i)
        full[past + i] = encoding[block[row * 128 * width + i]];
    else
      std::memcpy(full + past, block + row * 128 * width,
                  valid_tokens * width * 2);
    std::memcpy(retained, full + valid_tokens * width, past * 2);
    if (valid_tokens != 128) {
      const size_t padding = (128 - valid_tokens) * width;
      std::memmove(full + padding, full, (past + valid_tokens * width) * 2);
      std::fill(full, full + padding, uint16_t(-history.offset));
    }
  }
}
