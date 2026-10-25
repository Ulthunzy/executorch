#include "integer_rpc.h"
#include "remote.h"
#include "rpcmem.h"
#include <array>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

int main() {
  remote_rpc_control_unsigned_module request{CDSP_DOMAIN_ID, 1};
  if (remote_session_control(
          DSPRPC_CONTROL_UNSIGNED_MODULE, &request, sizeof(request)))
    return 1;
  remote_handle64 handle = 0;
  if (integer_rpc_open(
          (std::string(integer_rpc_URI) + "&_dom=cdsp").c_str(), &handle))
    return 2;
  auto* bank = rpcmem_alloc(25, 0, 8192);
  if (!bank)
    return 3;
  memset(bank, 0, 8192);
  const int fd = rpcmem_to_fd(bank);
  if (fastrpc_mmap(CDSP_DOMAIN_ID, fd, bank, 0, 8192, FASTRPC_MAP_FD_DELAYED))
    return 4;
  constexpr int tokens = MOE_TEST_AR;
  if (integer_rpc_configure(handle, tokens))
    return 5;
  std::vector<float> hidden(tokens * 2048), scores(tokens * 8),
      shared(tokens * 2048), output(tokens * 2048);
  std::vector<int> ids(tokens * 8);
  for (int i = 0; i < tokens * 8; ++i)
    ids[i] = i % 8;
  std::array<int, 8> descriptor{
      fd, 8192, 1, tokens == 128 ? 0 : tokens << 8, 0, 8, tokens * 8, 0};
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
        output.data(),
        output.size());
  };
  const int failed = run();
  // A real retry reaches header validation (26), rather than reusing -1 (25).
  const int retried = run();
  descriptor[2] = -1;
  const int recovered = run();
  const int closed = integer_rpc_close(handle);
  const int unmapped = fastrpc_munmap(CDSP_DOMAIN_ID, fd, bank, 8192);
  rpcmem_free(bank);
  printf(
      "MAP_FAILURE_PROBE ar=%d failed=%d retried=%d recovered=%d closed=%d unmapped=%d\n",
      tokens,
      failed,
      retried,
      recovered,
      closed,
      unmapped);
  return failed == 25 && retried == 26 && !recovered && !closed && !unmapped
      ? 0
      : 6;
}
