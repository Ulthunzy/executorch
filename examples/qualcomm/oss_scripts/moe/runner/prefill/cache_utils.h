#pragma once
#include "../qnn/qnn_front.h"

std::vector<uint16_t> remap(
    const Qnn_Tensor_t& source,
    const Qnn_Tensor_t& destination);
void reset_quant(Graph& graph, int index);
