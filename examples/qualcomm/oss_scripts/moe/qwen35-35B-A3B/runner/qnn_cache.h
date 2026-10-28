#pragma once

#include "cache_transfer.h"
#include <QnnTypes.h>
#include <stdexcept>

inline QuantizedCacheView qnn_cache_view(const Qnn_Tensor_t& tensor, void* data) {
  if (tensor.version != QNN_TENSOR_VERSION_1 && tensor.version != QNN_TENSOR_VERSION_2)
    throw std::runtime_error("Unsupported QNN cache tensor version");
  const auto dtype = tensor.version == QNN_TENSOR_VERSION_1
      ? tensor.v1.dataType : tensor.v2.dataType;
  const auto quant = tensor.version == QNN_TENSOR_VERSION_1
      ? tensor.v1.quantizeParams : tensor.v2.quantizeParams;
  const auto rank = tensor.version == QNN_TENSOR_VERSION_1
      ? tensor.v1.rank : tensor.v2.rank;
  const auto* dimensions = tensor.version == QNN_TENSOR_VERSION_1
      ? tensor.v1.dimensions : tensor.v2.dimensions;
  size_t elements = 1;
  for (unsigned i = 0; i < rank; ++i)
    elements *= dimensions[i];
  if (dtype == QNN_DATATYPE_FLOAT_16 &&
      quant.quantizationEncoding == QNN_QUANTIZATION_ENCODING_UNDEFINED)
    return {static_cast<uint16_t*>(data), elements, 0, 0, CacheDType::Float16};
  if (dtype != QNN_DATATYPE_UFIXED_POINT_16 ||
      quant.quantizationEncoding != QNN_QUANTIZATION_ENCODING_SCALE_OFFSET)
    throw std::runtime_error("Expected affine UINT16 or unquantized FP16 cache");
  return {static_cast<uint16_t*>(data), elements,
          quant.scaleOffsetEncoding.scale, quant.scaleOffsetEncoding.offset};
}
