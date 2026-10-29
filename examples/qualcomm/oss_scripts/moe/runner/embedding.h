#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

class EmbeddingTable {
 public:
  explicit EmbeddingTable(const std::string& directory);
  ~EmbeddingTable();
  EmbeddingTable(const EmbeddingTable&) = delete;
  EmbeddingTable& operator=(const EmbeddingTable&) = delete;

  void lookup(int64_t token, float* output) const;

 private:
  const uint8_t* data_ = nullptr;
  size_t size_ = 0;
  bool packed_ = false;
};
