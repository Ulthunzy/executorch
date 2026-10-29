#pragma once
#include <cstddef>

namespace moe::decode {

// temperature=0 的 greedy 采样，拒绝非有限 logits，保留首个最大值。
size_t checked_argmax(const float* values, size_t count);

} // namespace moe::decode
