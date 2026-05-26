/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/qualcomm/runtime/QnnExecuTorch.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/rpc_mem.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/platform/assert.h>
using executorch::runtime::MemoryAllocator;
using executorch::runtime::TensorInfo;

namespace example {
RpcMem::RpcMem(
    const size_t total_cache_size,
    const size_t total_prompt_processor_io_size,
    const size_t total_token_generator_io_size)
    : calculated_offsets_(0) {
  size_t total_bytes = total_cache_size + total_prompt_processor_io_size +
      total_token_generator_io_size;
  total_bytes_ = align_shared_buffer_allocation_size(total_bytes);
  shared_buffer_base_ptr_ = QnnExecuTorchAllocCustomMem(
      total_bytes_, MemoryAllocator::kDefaultAlignment);
}
RpcMem::RpcMem(
    const size_t total_cache_size,
    const size_t total_prompt_processor_io_size,
    const size_t total_token_generator_io_size,
    const size_t total_embedding_processor_io_size,
    const size_t total_embedding_generator_io_size)
    : calculated_offsets_(0) {
  size_t total_bytes = total_cache_size + total_prompt_processor_io_size +
      total_token_generator_io_size + total_embedding_processor_io_size +
      total_embedding_generator_io_size;
  total_bytes_ = align_shared_buffer_allocation_size(total_bytes);
  shared_buffer_base_ptr_ = QnnExecuTorchAllocCustomMem(
      total_bytes_, MemoryAllocator::kDefaultAlignment);
}
RpcMem::~RpcMem() {
  QnnExecuTorchFreeCustomMem(shared_buffer_base_ptr_);
}

std::byte* RpcMem::allocate(size_t data_size) {
  const size_t aligned_offset =
      align_shared_buffer_allocation_size(calculated_offsets_);
  ET_CHECK_MSG(
      aligned_offset + data_size <= total_bytes_,
      "RpcMem overflow: request=%zu aligned_offset=%zu total_bytes=%zu",
      data_size,
      aligned_offset,
      total_bytes_);
  std::byte* data_ptr = static_cast<std::byte*>(shared_buffer_base_ptr_);
  data_ptr += aligned_offset;
  // Record the position of the data pointer
  io_pos_map_[data_ptr] = aligned_offset;
  calculated_offsets_ = aligned_offset + data_size;
  return data_ptr;
}

void RpcMem::add_memory_info(
    void* data_ptr,
    size_t data_size,
    TensorInfo tensor_info) {
  if (auto it = io_pos_map_.find(static_cast<std::byte*>(data_ptr));
      it == io_pos_map_.end()) {
    ET_LOG(Error, "Shared buffer pointer %p is not found", data_ptr);
  }
  if (binded_tensor_addr_set_.find(data_ptr) == binded_tensor_addr_set_.end()) {
    QnnExecuTorchAddCustomMemTensorAddr(data_ptr, shared_buffer_base_ptr_);
    binded_tensor_addr_set_.insert(data_ptr);
  }
};

} // namespace example
