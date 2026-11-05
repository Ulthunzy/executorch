#include "weight_source.h"
#include "../qnn_front.h"
#include <cerrno>
#include <limits>
#include <mutex>

namespace {
// Batch context initialization can invoke different providers concurrently;
// their rpcmem allocations update the same Runtime accounting.
std::mutex weight_callbacks;

struct Header {
  char magic[8];
  uint64_t version, image_size, graph_count, weight_count;
};

void read_exact(int fd, void* destination, size_t size, uint64_t offset) {
  size_t done = 0;
  while (done < size) {
    const auto count = pread(
        fd, static_cast<char*>(destination) + done, size - done, offset + done);
    if (count < 0 && errno == EINTR)
      continue;
    if (count <= 0)
      throw std::runtime_error("shared context pread");
    done += count;
  }
}
} // namespace

WeightSource::WeightSource(Runtime& runtime, const std::string& directory)
    : runtime_(runtime) {
  const int graph_fd = open((directory + "/graph.qnn").c_str(), O_RDONLY);
  if (graph_fd < 0)
    throw std::runtime_error("open graph.qnn");
  try {
    Header header{};
    read_exact(graph_fd, &header, sizeof(header), 0);
    if (memcmp(header.magic, "Q35CTX01", 8) || header.version != 1 ||
        header.graph_count > 4096 || header.weight_count > 4096 ||
        !header.graph_count || !header.weight_count || !header.image_size ||
        header.image_size > std::numeric_limits<size_t>::max())
      throw std::runtime_error("shared context header");
    std::vector<Section> graph(header.graph_count);
    weights_.resize(header.weight_count);
    read_exact(
        graph_fd, graph.data(), graph.size() * sizeof(Section), sizeof(header));
    read_exact(
        graph_fd,
        weights_.data(),
        weights_.size() * sizeof(Section),
        sizeof(header) + graph.size() * sizeof(Section));
    weights_fd_ = open((directory + "/weights.bin").c_str(), O_RDONLY);
    if (weights_fd_ < 0)
      throw std::runtime_error("open weights.bin");

    struct stat graph_stat {};

    struct stat weights_stat {};

    if (fstat(graph_fd, &graph_stat) || fstat(weights_fd_, &weights_stat))
      throw std::runtime_error("shared context stat");
    std::vector<Section> coverage = graph;
    coverage.insert(coverage.end(), weights_.begin(), weights_.end());
    std::sort(coverage.begin(), coverage.end(), [](auto& a, auto& b) {
      return a.offset < b.offset;
    });
    uint64_t end = 0;
    for (const auto& section : coverage) {
      if (section.offset != end || !section.size || section.offset % 4096 ||
          section.file_offset % 4096 || section.size > header.image_size - end)
        throw std::runtime_error("shared context section coverage");
      end += section.size;
    }
    if (end != header.image_size)
      throw std::runtime_error("incomplete shared context image");
    image_size_ = header.image_size;
    image_ = mmap(
        nullptr, image_size_, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (image_ == MAP_FAILED) {
      image_ = nullptr;
      throw std::runtime_error("reserve shared context image");
    }
    auto map_sections =
        [&](int fd, const std::vector<Section>& sections, uint64_t file_size) {
          for (const auto& section : sections) {
            if (section.file_offset > file_size ||
                section.size > file_size - section.file_offset)
              throw std::runtime_error("truncated shared context section");
            void* target = static_cast<char*>(image_) + section.offset;
            if (mmap(
                    target,
                    section.size,
                    PROT_READ,
                    MAP_PRIVATE | MAP_FIXED,
                    fd,
                    section.file_offset) == MAP_FAILED)
              throw std::runtime_error("map shared context section");
          }
        };
    map_sections(graph_fd, graph, graph_stat.st_size);
    map_sections(weights_fd_, weights_, weights_stat.st_size);
    close(graph_fd);
    callback_.type = QNN_CONTEXT_CALLBACK_DMA_BUFFER;
    callback_.dmaBufferCallback.version =
        QNN_CONTEXT_CALLBACK_DMA_BUFFER_VERSION_1;
    callback_.dmaBufferCallback.v1 = {provide, release, this};
  } catch (...) {
    close(graph_fd);
    if (weights_fd_ >= 0)
      close(weights_fd_);
    if (image_)
      munmap(image_, image_size_);
    throw;
  }
}

WeightSource::~WeightSource() {
  if (!allocations_.empty())
    fprintf(stderr, "WeightSource destroyed with live QNN weight buffers\n");
  if (image_)
    munmap(image_, image_size_);
  if (weights_fd_ >= 0)
    close(weights_fd_);
}

const void* WeightSource::image() const {
  return image_;
}

uint64_t WeightSource::image_size() const {
  return image_size_;
}

const Qnn_ContextBinaryCallback_t* WeightSource::callback() const {
  return &callback_;
}

size_t WeightSource::live_allocations() const {
  return allocations_.size();
}

void WeightSource::discard_file_cache() {
  // Only weights.bin is discarded. Graph metadata and expert IO source files
  // remain available; QNN inference reads the provided DMA allocations.
  struct stat info {};
  const long system_page_size = sysconf(_SC_PAGESIZE);
  if (fstat(weights_fd_, &info) || info.st_size <= 0 || system_page_size <= 0) {
    fprintf(stderr, "shared weights cache stat failed: %d\n", errno);
    return;
  }
  const size_t page_size = system_page_size;
  const size_t length = info.st_size;
  std::vector<unsigned char> pages((length + page_size - 1) / page_size);
  void* mapping = mmap(
      nullptr, length, PROT_READ, MAP_PRIVATE, weights_fd_, 0);
  auto resident_bytes = [&]() -> uint64_t {
    if (mapping == MAP_FAILED || mincore(mapping, length, pages.data()))
      return 0;
    return std::count_if(pages.begin(), pages.end(), [](unsigned char value) {
      return value & 1;
    }) * page_size;
  };
  const auto before = resident_bytes();
  for (const auto& section : weights_)
    if (madvise(
            static_cast<char*>(image_) + section.offset,
            section.size,
            MADV_DONTNEED))
      fprintf(stderr, "shared weights page advice failed: %d\n", errno);
  const int status = posix_fadvise(weights_fd_, 0, 0, POSIX_FADV_DONTNEED);
  if (status)
    fprintf(stderr, "shared weights file advice failed: %d\n", status);
  const auto after = resident_bytes();
  if (mapping != MAP_FAILED)
    munmap(mapping, length);
  fprintf(
      stderr,
      "SHARED_WEIGHT_FILE_CACHE dma_bytes=%llu source_bytes=%zu resident_before=%llu resident_after=%llu advice_status=%d\n",
      (unsigned long long)allocated_bytes,
      length,
      (unsigned long long)before,
      (unsigned long long)after,
      status);
}

Qnn_ErrorHandle_t WeightSource::provide(
    Qnn_ContextBinaryDataRequest_t request,
    Qnn_ContextBinaryDmaDataResponse_t* response,
    void* opaque) {
  auto& self = *static_cast<WeightSource*>(opaque);
  std::lock_guard<std::mutex> guard(weight_callbacks);
  try {
    const auto found = std::find_if(
        self.weights_.begin(), self.weights_.end(), [&](const auto& section) {
          return request.offset >= section.offset &&
              request.offset - section.offset <= section.size &&
              request.size <= section.size - (request.offset - section.offset);
        });
    if (!response || !request.size || found == self.weights_.end() ||
        request.size > std::numeric_limits<int>::max() - 4095)
      return QNN_CONTEXT_ERROR_INVALID_ARGUMENT;
    auto buffer = std::make_shared<Buffer>(self.runtime_, request.size, false);
    read_exact(
        self.weights_fd_,
        buffer->data,
        request.size,
        found->file_offset + request.offset - found->offset);
    if (buffer->size > request.size)
      memset(
          static_cast<char*>(buffer->data) + request.size,
          0,
          buffer->size - request.size);
    response->dmaBuffer.fd = buffer->fd;
    response->dmaBuffer.data = buffer->data;
    response->dataStartOffset = 0;
    response->alignedSize = buffer->size;
    self.allocations_.emplace(buffer->fd, buffer);
    ++self.provides;
    self.allocated_bytes += buffer->size;
    // The callback supplies a DMA copy; avoid retaining another file-backed
    // copy until every context has finished loading.
    posix_fadvise(
        self.weights_fd_,
        found->file_offset + request.offset - found->offset,
        request.size,
        POSIX_FADV_DONTNEED);
    return QNN_SUCCESS;
  } catch (const std::exception& error) {
    fprintf(stderr, "weight source: %s\n", error.what());
    return QNN_CONTEXT_ERROR_MEM_ALLOC;
  }
}

Qnn_ErrorHandle_t WeightSource::release(
    Qnn_ContextBinaryDmaDataMem_t memory,
    void* opaque) {
  auto& self = *static_cast<WeightSource*>(opaque);
  std::lock_guard<std::mutex> guard(weight_callbacks);
  const auto found = self.allocations_.find(memory.dmaBuffer.fd);
  if (found == self.allocations_.end() ||
      found->second->data != memory.dmaBuffer.data)
    return QNN_CONTEXT_ERROR_INVALID_ARGUMENT;
  self.allocations_.erase(found);
  ++self.releases;
  return QNN_SUCCESS;
}
