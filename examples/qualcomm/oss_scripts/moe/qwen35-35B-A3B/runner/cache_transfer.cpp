#include "cache_transfer.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {
void validate(const QuantizedCacheView& view) {
  if (!view.data || !view.elements || !std::isfinite(view.scale) ||
      view.scale <= 0 || view.offset > 0 || view.offset < -65535)
    throw std::runtime_error("Invalid UINT16 cache view");
}
} // namespace

void clear_cache(const QuantizedCacheView& cache) {
  validate(cache);
  std::fill(cache.data, cache.data + cache.elements, uint16_t(-cache.offset));
}

void copy_cache_range(
    const QuantizedCacheView& source,
    const QuantizedCacheView& destination,
    CacheLayout layout,
    size_t source_begin,
    size_t destination_begin,
    size_t count) {
  validate(source);
  validate(destination);
  const size_t width = layout == CacheLayout::State ? 1 : 512;
  if (source.elements % width || destination.elements % width ||
      source_begin > source.elements / width ||
      count > source.elements / width - source_begin ||
      destination_begin > destination.elements / width ||
      count > destination.elements / width - destination_begin)
    throw std::runtime_error("Cache transfer exceeds tensor dimensions");
  if (source.data == destination.data &&
      (source.elements != destination.elements ||
       source_begin != destination_begin))
    throw std::runtime_error(
        "Overlapping cache layouts require separate storage");

  const bool identity =
      source.scale == destination.scale && source.offset == destination.offset;
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
    else
      for (size_t i = 0; i < length; ++i)
        destination.data[to + i] = mapping[source.data[from + i]];
  };
  if (layout == CacheLayout::State) {
    copy(source_begin, destination_begin, count);
  } else if (layout == CacheLayout::Key) {
    for (size_t row = 0; row < 512; ++row)
      copy(
          row * (source.elements / 512) + source_begin,
          row * (destination.elements / 512) + destination_begin,
          count);
  } else {
    for (size_t head = 0; head < 2; ++head)
      copy(
          head * (source.elements / 2) + source_begin * 256,
          head * (destination.elements / 2) + destination_begin * 256,
          count * 256);
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
  if (history.elements % 512 || !block || !encoding || !handoff ||
      layout == CacheLayout::State || valid_tokens < 1 || valid_tokens > 128)
    throw std::runtime_error("invalid padded KV block");
  const size_t width = layout == CacheLayout::Key ? 1 : 256;
  const size_t rows = 512 / width;
  const size_t past = history.elements / rows, window = past + 128 * width;
  for (size_t row = 0; row < rows; ++row) {
    auto* full = handoff + row * window;
    auto* retained = history.data + row * past;
    std::memcpy(full, retained, past * 2);
    for (size_t i = 0; i < valid_tokens * width; ++i)
      full[past + i] = encoding[block[row * 128 * width + i]];
    std::memcpy(retained, full + valid_tokens * width, past * 2);
    if (valid_tokens != 128) {
      const size_t padding = (128 - valid_tokens) * width;
      std::memmove(full + padding, full, (past + valid_tokens * width) * 2);
      std::fill(full, full + padding, uint16_t(-history.offset));
    }
  }
}
