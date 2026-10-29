#include "cache_utils.h"
#include "../qnn/qnn_cache.h"
#include <cmath>

std::vector<uint16_t> remap(
    const Qnn_Tensor_t& source,
    const Qnn_Tensor_t& destination) {
  // 构造整张 UINT16→UINT16 LUT，转换输出网格到下一批 state 输入网格。
  auto a = qnn_cache_view(source, nullptr),
       b = qnn_cache_view(destination, nullptr);
  if (a.dtype != b.dtype)
    throw std::runtime_error("Cache feedback requires matching dtypes");
  if (a.dtype == CacheDType::Float16)
    return {};
  std::vector<uint16_t> result(65536);
  for (int i = 0; i < 65536; i++)
    result[i] = std::clamp(
        long(lround((i + a.offset) * double(a.scale) / b.scale - b.offset)),
        0L,
        65535L);
  return result;
}

void reset_quant(Graph& graph, int index) {
  clear_cache(qnn_cache_view(graph.inputs[index], graph.input(index)));
}
