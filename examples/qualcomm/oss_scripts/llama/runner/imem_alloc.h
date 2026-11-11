/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <executorch/runtime/executor/method_meta.h>
#include <cstddef>
#include <cstdint>

namespace example {

inline size_t align_shared_buffer_allocation_size(size_t data_size) {
  constexpr size_t kSharedBufferAlignment = 4096;
  return ((data_size + kSharedBufferAlignment - 1) / kSharedBufferAlignment) *
      kSharedBufferAlignment;
}

/**
 * @class IMemAlloc
 * @brief Interface for buffer allocation.
 */
class IMemAlloc {
 public:
  struct MappedBuffer {
    std::byte* data;
    int32_t fd;
  };
  IMemAlloc(){};
  virtual ~IMemAlloc(){};
  virtual std::byte* allocate(size_t data_size) = 0;
  virtual MappedBuffer allocate_mapped(size_t data_size) {
    return {nullptr, -1};
  }
  virtual void add_memory_info(
      void* data_ptr,
      size_t data_size,
      executorch::runtime::TensorInfo tensor_info) = 0;
};

} // namespace example
