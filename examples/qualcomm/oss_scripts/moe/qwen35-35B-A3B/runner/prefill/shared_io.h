#pragma once

#include "attention.h"

// 40 层顺序执行，复用同一套激活/路由 IO；每层的 state/KV 不参与这套别名复用。
// bind 同时更新 C++ Buffer 所有权和 QNN tensor 的内存绑定。
struct SharedIo {
  std::shared_ptr<Buffer> hidden, residual, ids, scores, shared, cos, sin, mask,
      last_token_index;
  size_t bytes = 0;

  static void bind(
      Graph& graph,
      bool input,
      int index,
      const std::shared_ptr<Buffer>& buffer);

  explicit SharedIo(Runtime& runtime);

  void bind_layer(
      Attention& attention,
      Graph& router,
      const std::shared_ptr<Buffer>& chain);
};
