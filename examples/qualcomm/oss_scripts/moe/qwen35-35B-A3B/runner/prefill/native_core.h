#pragma once

#include "../runtime_utils.h"
#include "qnn_front.h"
#include "integer_rpc.h"
#include "remote.h"
#include "AEEStdErr.h"
#include <sys/ioctl.h>
#include <linux/dma-buf.h>

// ARM 侧 Native MoE RPC 包装：QNN 与 DSP 共用已映射的输入/输出 DMA buffer。
// publish_inputs/end_output_read 配对处理 CPU 与 DSP 间的缓存可见性。
struct NativeCore {
  Runtime& runtime;
  remote_handle64 handle = 0;
  bool mapped = getenv("NATIVE_MAPPED_IO"),
       fuse_residual = getenv("NATIVE_RESIDUAL"), output_read = false;
  std::shared_ptr<Buffer> residual;
  std::map<int, std::shared_ptr<Buffer>> io_mappings;
  std::vector<int> owned_mappings;
  std::vector<std::shared_ptr<Buffer>> inbuf, outbuf;

  NativeCore(Runtime& r, const std::string&, int tokens = 128);

  int named(bool input, const char* name);

  void* input(int i) {
    return inbuf[i]->data;
  }

  void* output(int) {
    return outbuf[0]->data;
  }

  void sync_buffer(const std::shared_ptr<Buffer>& buffer, uint64_t flags);

  void publish_inputs();

  void map_buffer(
      const std::shared_ptr<Buffer>& buffer,
      fastrpc_map_flags flags = FASTRPC_MAP_FD_DELAYED);

  void end_output_read();

  void execute(bool read_output = true);

  void release_io();

  void close();

  ~NativeCore();
};
