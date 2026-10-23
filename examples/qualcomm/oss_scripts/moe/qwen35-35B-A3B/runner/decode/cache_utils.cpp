#include "cache_utils.h"
#include <cmath>

std::vector<uint16_t> grid_map(
    const Qnn_Tensor_t& from,
    const Qnn_Tensor_t& to) {
  // 整数存储类型相同不代表 scale/offset 相同；预先建立下一步反馈用的 LUT。
  auto a = FIELD(from, quantizeParams).scaleOffsetEncoding,
       b = FIELD(to, quantizeParams).scaleOffsetEncoding;
  std::vector<uint16_t> map(65536);
  for (int i = 0; i < 65536; i++)
    map[i] = std::clamp(
        long(lround((i + a.offset) * double(a.scale) / b.scale - b.offset)),
        0L,
        65535L);
  return map;
}
