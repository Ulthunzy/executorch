#include "embedding.h"

#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

EmbeddingTable::EmbeddingTable(const std::string& directory) {
  const char* bits = getenv("EMBEDDING_BITS");
  if (bits && strcmp(bits, "4") && strcmp(bits, "16"))
    throw std::runtime_error("EMBEDDING_BITS must be 4 or 16");
  packed_ = bits && !strcmp(bits, "4");
  const std::string path =
      directory + (packed_ ? "/embedding.int4" : "/embedding.bf16");
  const int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0)
    throw std::runtime_error("open embedding: " + path);

  struct stat metadata {};

  if (fstat(fd, &metadata)) {
    close(fd);
    throw std::runtime_error("embedding fstat");
  }
  size_ = metadata.st_size;
  const size_t expected = packed_ ? 64 + 248320ULL * 1280 : 248320ULL * 4096;
  if (size_ != expected) {
    close(fd);
    throw std::runtime_error("embedding dimensions");
  }
  void* mapped = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (mapped == MAP_FAILED)
    throw std::runtime_error("embedding mmap");
  data_ = static_cast<const uint8_t*>(mapped);
  if (packed_) {
    uint32_t header[4];
    memcpy(header, data_ + 8, sizeof(header));
    if (memcmp(data_, "Q35EMB4", 8) || header[0] != 1 || header[1] != 248320 ||
        header[2] != 2048 || header[3] != 32) {
      munmap(mapped, size_);
      data_ = nullptr;
      throw std::runtime_error("INT4 embedding header");
    }
  }
}

EmbeddingTable::~EmbeddingTable() {
  if (data_)
    munmap(const_cast<uint8_t*>(data_), size_);
}

void EmbeddingTable::lookup(int64_t token, float* output) const {
  if (token < 0 || token >= 248320)
    throw std::runtime_error("embedding token ID");
  if (!packed_) {
    const auto* row = reinterpret_cast<const uint16_t*>(data_) + token * 2048;
    for (int i = 0; i < 2048; i++) {
      uint32_t bits = uint32_t(row[i]) << 16;
      memcpy(output + i, &bits, sizeof(bits));
    }
    return;
  }
  const auto* row = data_ + 64 + token * 1280;
  for (int group = 0; group < 64; group++) {
    const auto* block = row + group * 20;
    _Float16 scale, bias;
    memcpy(&scale, block + 16, 2);
    memcpy(&bias, block + 18, 2);
    for (int i = 0; i < 32; i++) {
      const int code = (block[i / 2] >> ((i % 2) * 4)) & 15;
      output[group * 32 + i] = code * float(scale) + float(bias);
    }
  }
}
