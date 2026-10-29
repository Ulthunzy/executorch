#pragma once

#include "attention.h"

namespace moe::decode {

// 串行层间共用激活和路由内存；链式 hidden 直接接 Native 输出。
// 融合图的额外三个输出直接绑定 ids/scores/shared，避免再执行 Router 图。
struct SharedIo {
  std::shared_ptr<Buffer> hidden, residual, ids, scores, shared, cos, sin, mask;
  size_t bytes = 0;

  static void bind(
      Graph& graph,
      bool input,
      int index,
      const std::shared_ptr<Buffer>& buffer);

  SharedIo(
      Runtime& runtime,
      std::vector<std::unique_ptr<Attention>>& attention,
      std::vector<std::unique_ptr<Graph>>& router,
      const std::shared_ptr<Buffer>& chain);
};

} // namespace moe::decode
