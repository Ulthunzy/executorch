#include "../cache_encoding.h"
#include "../qnn_cache.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <limits>

int main() {
  for (uint32_t version : {1U, 2U}) {
    const size_t stride = version == 1 ? 16 : 24;
    std::vector<uint8_t> bytes(16 + 80 * stride);
    std::memcpy(bytes.data(), "Q36CACHE", 8);
    const uint32_t header[] = {version, 40};
    std::memcpy(bytes.data() + 8, header, 8);
    for (size_t index = 0; index < 80; ++index) {
      const float scale = version == 2 && index == 1 ? 0.f : 0.25f;
      const int32_t offset = scale == 0 ? 0 : -100;
      const uint64_t elements = 512;
      auto* entry = bytes.data() + 16 + index * stride;
      std::memcpy(entry, &scale, 4);
      std::memcpy(entry + 4, &offset, 4);
      std::memcpy(entry + 8, &elements, 8);
      if (version == 2) {
        const uint32_t dtype = index == 1 ? 1 : 0;
        std::memcpy(entry + 16, &dtype, 4);
      }
      const auto encoding = read_cache_encoding(bytes, index);
      assert(encoding.elements == elements && encoding.scale == scale);
      assert(encoding.offset == offset);
      assert(encoding.dtype == (scale == 0 ? CacheDType::Float16 : CacheDType::AffineUint16));
    }
    for (const auto mutation : {0, 1, 2, 3}) {
      auto invalid = bytes;
      if (mutation == 0)
        invalid.pop_back();
      else if (mutation == 1)
        invalid[8] = 3;
      else if (mutation == 2)
        invalid[12] = 39;
      else {
        const float scale = std::numeric_limits<float>::quiet_NaN();
        std::memcpy(invalid.data() + 16, &scale, 4);
      }
      bool rejected = false;
      try { read_cache_encoding(invalid, 0); }
      catch (const std::runtime_error&) { rejected = true; }
      assert(rejected);
    }
    if (version == 2) {
      for (size_t index : {0, 7}) {
        auto invalid = bytes;
        std::copy_n(bytes.data() + 16 + stride, stride, invalid.data() + 16 + index * stride);
        bool rejected = false;
        try { read_cache_encoding(invalid, index); }
        catch (const std::runtime_error&) { rejected = true; }
        assert(rejected);
      }
    }
  }
  uint16_t data[512]{};
  uint32_t dims[] = {1, 2, 16, 16};
  Qnn_Tensor_t tensor = QNN_TENSOR_INIT;
  tensor.version = QNN_TENSOR_VERSION_1;
  tensor.v1.rank = 4;
  tensor.v1.dimensions = dims;
  tensor.v1.dataType = QNN_DATATYPE_UFIXED_POINT_16;
  tensor.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
  tensor.v1.quantizeParams.scaleOffsetEncoding = {0.25f, -100};
  auto view = qnn_cache_view(tensor, data);
  assert(view.dtype == CacheDType::AffineUint16 && view.scale == 0.25f && view.offset == -100);
  tensor.v1.dataType = QNN_DATATYPE_FLOAT_16;
  tensor.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
  view = qnn_cache_view(tensor, data);
  assert(view.dtype == CacheDType::Float16 && view.scale == 0 && view.offset == 0 && view.elements == 512);
  tensor.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
  bool rejected = false;
  try { qnn_cache_view(tensor, data); }
  catch (const std::runtime_error&) { rejected = true; }
  assert(rejected);
  std::cout << "legacy/typed cache encodings and QNN physical dtype tests passed\n";
}
