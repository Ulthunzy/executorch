#pragma once

#include "../runtime_utils.h"
#include "qnn_front.h"
#include "integer_rpc.h"
#include "remote.h"
#include "AEEStdErr.h"
#include <sys/ioctl.h>
#include <linux/dma-buf.h>
#include <atomic>
#include <chrono>
#include <thread>

// 持久 DSP worker、映射和 mailbox；每层只提交描述符，不重复建立整套 FastRPC
// 状态。
struct NativeCore {
  Runtime& runtime;
  remote_handle64 handle = 0;
  bool mapped = getenv("NATIVE_MAPPED_IO"),
       fuse_residual = getenv("NATIVE_RESIDUAL"), output_read = false;
  std::shared_ptr<Buffer> residual;
  std::map<int, std::shared_ptr<Buffer>> io_mappings;
  std::vector<int> owned_mappings;
  std::vector<std::shared_ptr<Buffer>> inbuf, outbuf;
  std::shared_ptr<Buffer> mailbox;
  std::thread service;
  std::atomic<bool> service_done{true};
  std::mutex service_mutex;
  std::condition_variable service_changed;
  bool service_pending = false, service_stopping = false;
  std::atomic<int> service_status{0};
  uint32_t generation = 0;
  uint64_t service_invocations = 0, mailbox_commands = 0,
           direct_invocations = 0, service_threads = 0;
  bool use_mailbox =
      getenv("NATIVE_MAILBOX") && std::string(getenv("NATIVE_MAILBOX")) == "1";

  uint32_t* box() {
    return static_cast<uint32_t*>(mailbox->data);
  }

  // 控制页以 128 B 隔开请求、完成和启动状态；acquire/release
  // 保证描述符先于通知可见。
  void await_mailbox(int word, uint32_t expected);

  void start_service();

  int stop_service();

  int submit(const int* descriptors, bool final_command);

  NativeCore(Runtime& r, const std::string&);

  int named(bool input, const char* name);

  void* input(int i) {
    return inbuf[i]->data;
  }

  void* output(int) {
    return outbuf[0]->data;
  }

  void sync_buffer(const std::shared_ptr<Buffer>& buffer, uint64_t flags);

  void publish_inputs();

  void map_buffer(const std::shared_ptr<Buffer>& buffer);

  void end_output_read();

  void execute(bool read_output = true, bool final_command = false);

  void retire_inputs();

  void release_io();

  void close();

  ~NativeCore();
};
