#include "integer_rpc.h"
#include "remote.h"
#include <array>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

int main() {
  remote_rpc_control_unsigned_module request{CDSP_DOMAIN_ID, 1};
  int error = remote_session_control(
      DSPRPC_CONTROL_UNSIGNED_MODULE, &request, sizeof(request));
  if (error)
    return error;
  remote_handle64 handle = 0;
  error = integer_rpc_open(
      (std::string(integer_rpc_URI) + "&_dom=cdsp").c_str(), &handle);
  if (error)
    return error;
  constexpr int tokens = MOE_TEST_AR;
  std::vector<float> hidden(tokens * 2048), scores(tokens * 8),
      shared(tokens * 2048), result(tokens * 2048);
  std::vector<int> ids(tokens * 8);
  std::array<int, 8> descriptor{-1, 0, 0, 0, 0, 1, tokens == 128 ? 1024 : 0, 0};
  auto run = [&] {
    return integer_rpc_run(
        handle,
        hidden.data(),
        hidden.size(),
        ids.data(),
        ids.size(),
        scores.data(),
        scores.size(),
        shared.data(),
        shared.size(),
        descriptor.data(),
        descriptor.size(),
        result.data(),
        result.size());
  };
  const auto started = std::chrono::steady_clock::now();
  const int failed = run();
  // 第二次调用沿用同一 handle/worker
  // pool，查询模式验证失败后的锁与线程均可复用。
  descriptor[2] = -1;
  const int recovered = run();
  const int closed = integer_rpc_close(handle);
  const double elapsed = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - started)
                             .count();
  std::printf(
      "LOCK_PROBE ar=%d failed=%d recovered=%d close=%d elapsed_ms=%.3f\n",
      tokens,
      failed,
      recovered,
      closed,
      elapsed);
  return failed == 1073 && !recovered && !closed ? 0 : 1;
}
