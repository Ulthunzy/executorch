#include <cstddef>
#include "qurt_hvx.h"
#include <cstdint>

// 仅在故障测试库中用 --wrap 链接，正式 Native 库没有注入分支。
#ifdef MOE_TEST_HMX_FAILURE
extern "C" int __real_compute_resource_hmx_lock(unsigned int)
    __attribute__((weak));

extern "C" int __wrap_compute_resource_hmx_lock(unsigned int context) {
  static uint32_t calls = 0;
  if (__atomic_fetch_add(&calls, 1U, __ATOMIC_RELAXED) == 0)
    return 73;
  return __real_compute_resource_hmx_lock
      ? __real_compute_resource_hmx_lock(context)
      : -1;
}
#else
extern "C" int __real_qurt_hvx_lock(qurt_hvx_mode_t);

extern "C" int __wrap_qurt_hvx_lock(qurt_hvx_mode_t mode) {
  static uint32_t calls = 0;
  // AR1 的 open 先执行一次 N2 自检，跳过这次调用再注入 worker 锁错误。
  if (__atomic_fetch_add(&calls, 1U, __ATOMIC_RELAXED) == MOE_TEST_HVX_SKIP)
    return 73;
  return __real_qurt_hvx_lock(mode);
}
#endif
