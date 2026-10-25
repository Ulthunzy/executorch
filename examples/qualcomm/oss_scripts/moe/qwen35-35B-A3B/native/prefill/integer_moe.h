#pragma once
#include "HTP/core/qhpi.h"
#include <cstdint>

#ifdef NATIVE_RPC
// FastRPC 的轻量 QHPI 适配器；与 opaque QHPI 类型隔离，计算入口按同一接口访问。
struct QHPI_RuntimeHandle {
  uint32_t resource, slices, slice;
  void* sync;
};

struct QHPI_Tensor {
  void* data;
};
#endif

uint32_t execute_integer_moe(
    QHPI_RuntimeHandle* handle,
    uint32_t output_count,
    QHPI_Tensor** outputs,
    uint32_t input_count,
    const QHPI_Tensor* const* inputs);

#ifdef __hexagon__
#include "HAP_perf.h"
#include "qurt_pimutex.h"

constexpr int kWorkers = 8;

struct Request {
  volatile uint32_t ready;
  uint8_t *a, *w;
  uint32_t* bias;
  uint16_t* raw;
  uint32_t k, groups, worker, nblocks, block_bytes;
};

struct Sync {
  volatile uint32_t workers, arrived, departed, done, go, next, combined, error,
      prepared, executed, assignments;
  volatile uint32_t waiting, done_mask;
  uint64_t ready_wait[8];
  const uint8_t* bank;
  qurt_mutex_t* token_locks;
  const float* residual;
  Request requests[8];
};

struct BankMapping {
  int fd = -1, generation = -1;
  uint8_t* pointer = nullptr;
  unsigned bytes = 0;
};

extern uint64_t metrics[10][8];
extern uint32_t execution_counts[4];
int release_banks();

inline uint64_t tick() {
#if SM_MOE_PROFILE
  return HAP_perf_get_qtimer_count();
#else
  return 0;
#endif
}

inline uint32_t read(const volatile uint32_t* p) {
  return __atomic_load_n(p, __ATOMIC_ACQUIRE);
}

inline void write(volatile uint32_t* p, uint32_t v) {
  __atomic_store_n(p, v, __ATOMIC_RELEASE);
}

inline uint32_t increment(volatile uint32_t* p) {
  return __atomic_fetch_add(p, 1, __ATOMIC_ACQ_REL);
}
#endif
