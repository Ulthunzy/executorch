/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dlfcn.h>
#include <executorch/backends/qualcomm/runtime/QnnExecuTorch.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/rpc_mem.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/platform/assert.h>
#include <limits>
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
  for (const auto& buffer : mapped_buffers_) {
    const int status =
        mapped_unmap_(mapped_domain_, buffer.fd, buffer.data, buffer.bytes);
    if (status) {
      ET_LOG(Error, "Cannot unmap KV cache FD %d: %d", buffer.fd, status);
    }
    mapped_free_(buffer.data);
  }
  if (rpc_library_) {
    dlclose(rpc_library_);
  }
  QnnExecuTorchFreeCustomMem(shared_buffer_base_ptr_);
}

IMemAlloc::MappedBuffer RpcMem::allocate_mapped(size_t size) {
  if (!rpc_library_) {
    rpc_library_ = dlopen("libcdsprpc.so", RTLD_NOW | RTLD_LOCAL);
    ET_CHECK_MSG(
        rpc_library_, "Cannot load libcdsprpc.so for mapped KV caches");
    mapped_free_ =
        reinterpret_cast<void (*)(void*)>(dlsym(rpc_library_, "rpcmem_free"));
    mapped_unmap_ = reinterpret_cast<int (*)(int, int, void*, size_t)>(
        dlsym(rpc_library_, "fastrpc_munmap"));
    auto session_control = reinterpret_cast<int (*)(uint32_t, void*, uint32_t)>(
        dlsym(rpc_library_, "remote_session_control"));
    ET_CHECK_MSG(
        session_control && mapped_unmap_, "Cannot load FastRPC mapping API");
    // FastRPC remote.h ABI; keep the runner independent of Hexagon headers.
    constexpr uint32_t kGetEffectiveDomainId = 14;
    char domain_name[] = "cdsp";
    struct {
      char* name;
      uint32_t name_length, session, domain;
    } request{domain_name, 4, 0, 0};
    ET_CHECK_MSG(
        session_control(kGetEffectiveDomainId, &request, sizeof(request)) == 0,
        "Cannot resolve the default CDSP session");
    mapped_domain_ = static_cast<int>(request.domain);
  }
  auto allocate = reinterpret_cast<void* (*)(int, unsigned, int)>(
      dlsym(rpc_library_, "rpcmem_alloc"));
  auto get_fd =
      reinterpret_cast<int (*)(void*)>(dlsym(rpc_library_, "rpcmem_to_fd"));
  auto map_buffer =
      reinterpret_cast<int (*)(int, int, void*, int, size_t, int)>(
          dlsym(rpc_library_, "fastrpc_mmap"));
  const size_t bytes = align_shared_buffer_allocation_size(size);
  ET_CHECK_MSG(
      allocate && get_fd && mapped_free_ && map_buffer &&
          bytes <= static_cast<size_t>(std::numeric_limits<int>::max()),
      "Invalid mapped KV allocation");
  // These buffers are read through DSP descriptors, outside QNN IO cache sync.
  void* buffer = allocate(25, 0, static_cast<int>(bytes));
  ET_CHECK_MSG(buffer, "Cannot allocate mapped KV cache (%zu bytes)", bytes);
  const int fd = get_fd(buffer);
  ET_CHECK_MSG(fd >= 0, "Cannot obtain mapped KV cache FD");
  // FASTRPC_MAP_FD is paired with HAP_mmap_get in the op.
  constexpr int kMapFd = 2;
  ET_CHECK_MSG(
      map_buffer(mapped_domain_, fd, buffer, 0, bytes, kMapFd) == 0,
      "Cannot map KV cache FD %d to CDSP",
      fd);
  mapped_buffers_.push_back({buffer, fd, bytes});
  return {static_cast<std::byte*>(buffer), fd};
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
