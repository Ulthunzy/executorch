#pragma once
#include "qnn_front.h"

std::vector<uint16_t> grid_map(
    const Qnn_Tensor_t& from,
    const Qnn_Tensor_t& to);
