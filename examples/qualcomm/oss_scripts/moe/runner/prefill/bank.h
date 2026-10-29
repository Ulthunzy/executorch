#pragma once
#include "../qnn/qnn_front.h"

// 仅常驻专家索引，payload 由 reader 按需读取。
struct Bank {
  int fd = -1;
  std::vector<uint8_t> index;
  size_t maximum = 0;

  Bank(const std::string& root, int layer);

  size_t offset(int e) const;

  size_t size(int e) const;

  ~Bank();
};
