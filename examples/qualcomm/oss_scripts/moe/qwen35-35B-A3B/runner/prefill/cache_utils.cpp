#include "cache_utils.h"
#include <cmath>

std::vector<uint16_t> remap(
    const Qnn_Tensor_t& source,
    const Qnn_Tensor_t& destination) {
  // 构造整张 UINT16→UINT16 LUT，转换输出网格到下一批 state 输入网格。
  auto a = FIELD(source, quantizeParams).scaleOffsetEncoding,
       b = FIELD(destination, quantizeParams).scaleOffsetEncoding;
  std::vector<uint16_t> result(65536);
  for (int i = 0; i < 65536; i++)
    result[i] = std::clamp(
        long(lround((i + a.offset) * double(a.scale) / b.scale - b.offset)),
        0L,
        65535L);
  return result;
}

void reset_quant(Graph& graph, int index) {
  auto& t = graph.inputs[index];
  auto q = FIELD(t, quantizeParams).scaleOffsetEncoding;
  auto* p = static_cast<uint16_t*>(graph.input(index));
  std::fill(p, p + Graph::elements(t), uint16_t(-q.offset));
}
