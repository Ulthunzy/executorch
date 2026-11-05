#pragma once

#include "qnn_front.h"

namespace moe {

// The resident Prefill session outlives Decode. Only serial stages may borrow
// these buffers. FA may change its history stride in place when grids match;
// incompatible layouts keep their private allocation.
std::shared_ptr<Buffer> prefill_state_buffer(
    int layer, int cache, bool output, const Qnn_Tensor_t& target);
std::shared_ptr<Buffer> prefill_head_buffer(
    bool output, const Qnn_Tensor_t& target);

} // namespace moe
