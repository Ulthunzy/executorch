#pragma once

#include "HTP/core/qhpi.h"
#include <cstdint>

#ifdef NATIVE_RPC
struct QHPI_RuntimeHandle {
  uint32_t resource, slices, slice;
  void* sync;
};

struct QHPI_Tensor {
  void* data;
};
#endif

namespace native_qhpi {
inline void* tensor_raw_data(const QHPI_Tensor* tensor) {
#ifdef NATIVE_RPC
  return tensor->data;
#else
  return qhpi_tensor_raw_data(tensor);
#endif
}

inline uint32_t thread_resources(QHPI_RuntimeHandle* handle) {
#ifdef NATIVE_RPC
  return handle->resource;
#else
  return qhpi_thread_resources(handle);
#endif
}

inline uint32_t num_slices(QHPI_RuntimeHandle* handle) {
#ifdef NATIVE_RPC
  return handle->slices;
#else
  return qhpi_num_slices(handle);
#endif
}

inline uint32_t slice_number(QHPI_RuntimeHandle* handle) {
#ifdef NATIVE_RPC
  return handle->slice;
#else
  return qhpi_slice_number(handle);
#endif
}

inline void* sync_block(QHPI_RuntimeHandle* handle) {
#ifdef NATIVE_RPC
  return handle->sync;
#else
  return qhpi_sync_block(handle);
#endif
}
} // namespace native_qhpi
