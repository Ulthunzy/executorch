// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
namespace attention_stream {
constexpr unsigned slots = 8;
inline uint8_t* buffer(uint8_t* work, unsigned panel) {
  return reinterpret_cast<GroupedScratchU16*>(
             work + worker_start +
             ((panel - 1) % slots) * sizeof(GroupedScratchU16))
      ->kv;
}
// The HMX server owns the DMA stream and all slot lifetimes.
class Cache {
 public:
  Cache(
      uint8_t* work,
      DmaDescriptor* descriptors,
      PackedKv8 packed,
      unsigned columns,
      unsigned packed_columns,
      unsigned heads,
      unsigned readers,
      unsigned history,
      unsigned valid)
      : work_(work),
        descriptors_(descriptors),
        packed_(packed),
        columns_(columns),
        packed_columns_(packed_columns),
        readers_((1u << std::min(readers, slots)) - 1),
        panels_per_head_(
            (columns / 1024) * 8 +
            (columns % 1024 ? (columns % 1024 + 255) / 256 + 4 : 0)),
        total_(panels_per_head_ * heads) {
    minimum_end_ = (history + std::min(32u, valid) + 31) & ~31u;
    const unsigned query_tiles = (valid + 31) / 32;
    for (unsigned slot = 0; slot < slots; ++slot) {
      consumed_[slot] = readers_;
      const unsigned begin = (slot % query_tiles) * 32;
      query_end_[slot] =
          (history + begin + std::min(32u, valid - begin) + 31) & ~31u;
    }
  }
  // Defer a blocked slot so other pending HMX jobs can release it.
  attention_hmx::Status prepare(unsigned panel) {
    const unsigned slot = (panel - 1) % slots;
    if (tags_[slot] != panel) {
      start();
      if (tags_[slot] != panel)
        return attention_hmx::Status::Deferred;
    }
    if (pending_ == panel) {
      if (!dma_wait())
        return attention_hmx::Status::Failed;
      pending_ = 0;
    }
    start();
    return attention_hmx::Status::Done;
  }
  void consume(unsigned panel, unsigned worker) {
    consumed_[(panel - 1) % slots] |= 1u << worker;
  }
  bool finish() {
    return !pending_ || dma_wait();
  }
  uint32_t key_bytes = 0, value_bytes = 0, loads = 0;

 private:
  void start() {
    const unsigned panel = next_panel_;
    if (pending_ || panel > total_)
      return;
    const unsigned slot = (panel - 1) % slots;
    if (tags_[slot] == panel || consumed_[slot] != readers_)
      return;
    const unsigned head = (panel - 1) / panels_per_head_,
                   index = (panel - 1) % panels_per_head_,
                   begin = index / 8 * 1024,
                   width = std::min(1024u, columns_ - begin),
                   key_panels = (width + 255) / 256, part = index % 8;
    auto* destination = buffer(work_, panel);
    if (part < key_panels) {
      const unsigned at = begin + part * 256,
                     bytes = std::min(256u, columns_ - at) * 256;
      dma_start(destination, packed_.key_panel(head, at), bytes, descriptors_);
      key_bytes += bytes;
    } else {
      const unsigned output_col = (part - key_panels) * 64;
      unsigned count = 0;
      for (unsigned col = 0; col < 2; ++col)
        for (unsigned offset = 0; offset < width; offset += 256) {
          const unsigned at = begin + offset,
                         piece = std::min(256u, width - offset);
          auto& descriptor = descriptors_[count];
          descriptor.next = descriptors_ + count + 1;
          descriptor.control = piece * 32 | (1u << 28) | (1u << 29);
          descriptor.source = packed_.value_column(
              head, at, output_col + col * 32, packed_columns_);
          descriptor.destination = destination + col * width * 32 + offset * 32;
          ++count;
        }
      descriptors_[count - 1].next = nullptr;
      dma_start(descriptors_);
      value_bytes += width * 64;
    }
    tags_[slot] = panel;
    consumed_[slot] = 0;
    // Readers whose query tile cannot see this panel never submit a job.
    const unsigned visible_begin = begin + (part < key_panels ? part * 256 : 0);
    if (visible_begin >= minimum_end_) {
      for (unsigned worker = 0; worker < slots; ++worker)
        if (query_end_[worker] <= visible_begin)
          consumed_[slot] |= readers_ & (1u << worker);
    }
    pending_ = panel;
    ++next_panel_;
    ++loads;
  }
  uint8_t* work_;
  DmaDescriptor* descriptors_;
  PackedKv8 packed_;
  unsigned columns_, packed_columns_, readers_, panels_per_head_, total_;
  unsigned tags_[slots]{}, consumed_[slots]{}, query_end_[slots]{},
      minimum_end_ = 0, pending_ = 0, next_panel_ = 1;
};
} // namespace attention_stream
