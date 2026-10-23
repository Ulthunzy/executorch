#pragma once

#include "qnn_front.h"

// 每层单独的外存专家文件：索引存 256 个专家的 offset/length，不把整文件读入
// DRAM。 O_DIRECT 的页对齐读取由 ExpertReaders 处理，length 仍表示实际 payload
// 大小。
struct Bank {
  int fd;
  std::vector<uint8_t> index;
  size_t maximum = 0, size = 0;

  Bank(const std::string& root, int layer);

  size_t offset(int e) const;

  size_t length(int e) const;

  ~Bank();
};
