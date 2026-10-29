#pragma once
#include "../qnn/qnn_front.h"

namespace moe::decode {

std::vector<uint16_t> grid_map(
    const Qnn_Tensor_t& from,
    const Qnn_Tensor_t& to);

} // namespace moe::decode
