#pragma once
#include <cstdint>

// 原地 N2→N4 展开；128 B 对齐，容量至少为 compressed_bytes 的两倍。
void n2_expand_in_place(uint8_t* buffer, int compressed_bytes);
