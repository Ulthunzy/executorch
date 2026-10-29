#pragma once

#include "QnnInterface.h"
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

struct Runtime;
struct Buffer;

// The virtual image borrows file-backed weight pages; QNN owns the lifetime of
// the DMA allocations through its provide/release callbacks.
class WeightSource {
 public:
  WeightSource(Runtime& runtime, const std::string& directory);
  ~WeightSource();
  WeightSource(const WeightSource&) = delete;
  WeightSource& operator=(const WeightSource&) = delete;

  const void* image() const;
  uint64_t image_size() const;
  const Qnn_ContextBinaryCallback_t* callback() const;
  size_t live_allocations() const;

  uint64_t provides = 0, releases = 0, allocated_bytes = 0;

 private:
  struct Section {
    uint64_t offset, size, file_offset;
  };

  Runtime& runtime_;
  void* image_ = nullptr;
  uint64_t image_size_ = 0;
  int weights_fd_ = -1;
  std::vector<Section> weights_;
  std::map<int, std::shared_ptr<Buffer>> allocations_;
  Qnn_ContextBinaryCallback_t callback_{};

  static Qnn_ErrorHandle_t provide(
      Qnn_ContextBinaryDataRequest_t request,
      Qnn_ContextBinaryDmaDataResponse_t* response,
      void* opaque);
  static Qnn_ErrorHandle_t release(
      Qnn_ContextBinaryDmaDataMem_t memory,
      void* opaque);
};
