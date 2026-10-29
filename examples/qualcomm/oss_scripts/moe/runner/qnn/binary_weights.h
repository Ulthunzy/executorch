#pragma once

#include "QnnInterface.h"
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>

struct Runtime;
struct Buffer;

// Read opaque weight sections directly into QNN-owned DMA allocations.
class BinaryWeights {
 public:
  BinaryWeights(Runtime& runtime, const std::string& path);
  ~BinaryWeights();
  BinaryWeights(const BinaryWeights&) = delete;
  BinaryWeights& operator=(const BinaryWeights&) = delete;

  const Qnn_ContextBinaryCallback_t* callback() const;
  void discard_file_cache() const;

 private:
  Runtime& runtime_;
  int fd_ = -1;
  uint64_t size_ = 0;
  uint64_t provides_ = 0, releases_ = 0;
  std::mutex mutex_;
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
