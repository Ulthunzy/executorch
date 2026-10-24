#include "../runner/decode/qnn_front.h"
#include <cerrno>
#include <mutex>

namespace {
struct Provider {
  Runtime& runtime;
  int source;
  FILE* trace;
  size_t file_size;
  std::mutex mutex;
  std::map<int, std::shared_ptr<Buffer>> allocations;
  size_t calls = 0, releases = 0;

  ~Provider() {
    if (trace)
      fclose(trace);
    if (source >= 0)
      close(source);
  }

  static Qnn_ErrorHandle_t provide(
      Qnn_ContextBinaryDataRequest_t request,
      Qnn_ContextBinaryDmaDataResponse_t* response,
      void* opaque) {
    auto& self = *static_cast<Provider*>(opaque);
    std::lock_guard<std::mutex> lock(self.mutex);
    try {
      if (request.offset > self.file_size ||
          request.size > self.file_size - request.offset)
        throw std::runtime_error("weight callback range");
      auto buffer = std::make_shared<Buffer>(self.runtime, request.size);
      size_t done = 0;
      while (done < request.size) {
        const auto count = pread(
            self.source,
            static_cast<char*>(buffer->data) + done,
            request.size - done,
            request.offset + done);
        if (count < 0 && errno == EINTR)
          continue;
        if (count <= 0)
          throw std::runtime_error("weight callback pread");
        done += count;
      }
      response->dmaBuffer.data = buffer->data;
      response->dmaBuffer.fd = buffer->fd;
      response->dataStartOffset = 0;
      response->alignedSize = buffer->size;
      self.allocations.emplace(buffer->fd, buffer);
      if (fprintf(
              self.trace,
              "%llu %llu\n",
              (unsigned long long)request.offset,
              (unsigned long long)request.size) < 0 ||
          fflush(self.trace))
        throw std::runtime_error("write callback trace");
      ++self.calls;
      return QNN_SUCCESS;
    } catch (const std::exception& error) {
      fprintf(stderr, "WEIGHT_CAPTURE_ERROR %s\n", error.what());
      return QNN_CONTEXT_ERROR_MEM_ALLOC;
    }
  }

  static Qnn_ErrorHandle_t release(
      Qnn_ContextBinaryDmaDataMem_t memory,
      void* opaque) {
    auto& self = *static_cast<Provider*>(opaque);
    std::lock_guard<std::mutex> lock(self.mutex);
    if (self.allocations.erase(memory.dmaBuffer.fd) != 1)
      return QNN_CONTEXT_ERROR_INVALID_ARGUMENT;
    ++self.releases;
    return QNN_SUCCESS;
  }
};

struct Context {
  Runtime& runtime;
  Qnn_ContextHandle_t handle = nullptr;

  ~Context() {
    if (handle)
      runtime.q.contextFree(handle, nullptr);
  }
};
} // namespace

int main(int argc, char** argv) try {
  if (argc != 4)
    throw std::runtime_error("capture_weights CONTEXT OP_PACKAGE REQUESTS");
  Runtime runtime;
  check(
      runtime.q.backendRegisterOpPackage(
          runtime.backend,
          argv[2],
          "DeltaNetOpPackageInterfaceProvider",
          "HTP"),
      "register GDN");
  MappedBinary binary(argv[1]);
  Provider provider{
      runtime, open(argv[1], O_RDONLY), fopen(argv[3], "w"), binary.size()};
  if (provider.source < 0 || !provider.trace)
    throw std::runtime_error("open capture inputs/output");
  Qnn_ContextBinaryCallback_t callback{};
  callback.type = QNN_CONTEXT_CALLBACK_DMA_BUFFER;
  callback.dmaBufferCallback.version =
      QNN_CONTEXT_CALLBACK_DMA_BUFFER_VERSION_1;
  callback.dmaBufferCallback.v1 = {
      Provider::provide, Provider::release, &provider};
  Context context{runtime};
  check(
      runtime.q.contextCreateFromBinaryWithCallback(
          runtime.backend,
          runtime.device,
          nullptr,
          &callback,
          binary.data(),
          binary.size(),
          &context.handle,
          nullptr,
          nullptr),
      "capture context restore");
  check(runtime.q.contextFree(context.handle, nullptr), "free capture context");
  context.handle = nullptr;
  printf(
      "WEIGHT_CAPTURE calls=%zu releases=%zu live=%zu\n",
      provider.calls,
      provider.releases,
      provider.allocations.size());
  return provider.calls && provider.allocations.empty() ? 0 : 1;
} catch (const std::exception& error) {
  fprintf(stderr, "FATAL %s\n", error.what());
  return 1;
}
