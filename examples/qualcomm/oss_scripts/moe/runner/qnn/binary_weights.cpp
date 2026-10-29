#include "binary_weights.h"
#include "qnn_front.h"
#include <cerrno>
#include <limits>

BinaryWeights::BinaryWeights(Runtime& runtime, const std::string& path)
    : runtime_(runtime) {
  fd_ = open(path.c_str(), O_RDONLY);
  if (fd_ < 0)
    throw std::runtime_error("open binary weights " + path);

  struct stat info {};

  if (fstat(fd_, &info) || info.st_size <= 0) {
    close(fd_);
    throw std::runtime_error("stat binary weights " + path);
  }
  size_ = info.st_size;
  callback_.type = QNN_CONTEXT_CALLBACK_DMA_BUFFER;
  callback_.dmaBufferCallback.version =
      QNN_CONTEXT_CALLBACK_DMA_BUFFER_VERSION_1;
  callback_.dmaBufferCallback.v1 = {provide, release, this};
}

BinaryWeights::~BinaryWeights() {
  fprintf(
      stderr,
      "BINARY_WEIGHTS_RELEASE provides=%llu releases=%llu live=%zu\n",
      (unsigned long long)provides_,
      (unsigned long long)releases_,
      allocations_.size());
  close(fd_);
}

const Qnn_ContextBinaryCallback_t* BinaryWeights::callback() const {
  return &callback_;
}

void BinaryWeights::discard_file_cache() const {
  const int status = posix_fadvise(fd_, 0, 0, POSIX_FADV_DONTNEED);
  if (status)
    fprintf(stderr, "binary weights cache advice failed: %d\n", status);
}

Qnn_ErrorHandle_t BinaryWeights::provide(
    Qnn_ContextBinaryDataRequest_t request,
    Qnn_ContextBinaryDmaDataResponse_t* response,
    void* opaque) {
  auto& self = *static_cast<BinaryWeights*>(opaque);
  std::lock_guard<std::mutex> lock(self.mutex_);
  try {
    if (!response || !request.size || request.offset > self.size_ ||
        request.size > self.size_ - request.offset ||
        request.size > std::numeric_limits<int>::max() - 4095)
      return QNN_CONTEXT_ERROR_INVALID_ARGUMENT;
    auto buffer = std::make_shared<Buffer>(self.runtime_, request.size, false);
    size_t done = 0;
    while (done < request.size) {
      const auto count = pread(
          self.fd_,
          static_cast<char*>(buffer->data) + done,
          request.size - done,
          request.offset + done);
      if (count < 0 && errno == EINTR)
        continue;
      if (count <= 0)
        throw std::runtime_error("read binary weight section");
      done += count;
    }
    response->dmaBuffer.fd = buffer->fd;
    response->dmaBuffer.data = buffer->data;
    response->dataStartOffset = 0;
    response->alignedSize = buffer->size;
    self.allocations_.emplace(buffer->fd, buffer);
    ++self.provides_;
    fprintf(
        stderr,
        "BINARY_WEIGHTS_PROVIDE fd=%d bytes=%zu offset=%llu\n",
        buffer->fd,
        buffer->size,
        (unsigned long long)request.offset);
    return QNN_SUCCESS;
  } catch (const std::exception& error) {
    fprintf(stderr, "binary weight callback: %s\n", error.what());
    return QNN_CONTEXT_ERROR_MEM_ALLOC;
  }
}

Qnn_ErrorHandle_t BinaryWeights::release(
    Qnn_ContextBinaryDmaDataMem_t memory,
    void* opaque) {
  auto& self = *static_cast<BinaryWeights*>(opaque);
  std::lock_guard<std::mutex> lock(self.mutex_);
  const auto entry = self.allocations_.find(memory.dmaBuffer.fd);
  if (entry == self.allocations_.end() ||
      entry->second->data != memory.dmaBuffer.data)
    return QNN_CONTEXT_ERROR_INVALID_ARGUMENT;
  self.allocations_.erase(entry);
  ++self.releases_;
  return QNN_SUCCESS;
}
