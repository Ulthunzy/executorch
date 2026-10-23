#pragma once

// QNN 图、context 与 rpcmem buffer 的生命周期管理；层间共享 IO，state
// 各自保留。
#include "QnnInterface.h"
#include "System/QnnSystemInterface.h"
#include "HTP/QnnHtpDevice.h"
#include "HTP/QnnHtpGraph.h"
#include "HTP/QnnHtpContext.h"
#include <condition_variable>
#include <mutex>
#include <algorithm>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#define FIELD(t, f) ((t).version == QNN_TENSOR_VERSION_1 ? (t).v1.f : (t).v2.f)

void check(Qnn_ErrorHandle_t status, const char* action);
std::vector<uint8_t> bytes(const std::string& name);
void save(const std::string& name, const void* data, size_t size);

// 一次 session 共享 backend/device，context 和 IO 注册次数用于检查循环内重建。
struct Runtime {
  bool defer_contexts = false, defer_bindings = true;
  uint64_t context_creates = 0, context_frees = 0, live_contexts = 0,
           peak_contexts = 0, memory_registers = 0, memory_deregisters = 0;
  uint64_t buffer_allocations = 0, buffer_frees = 0, buffer_bytes = 0,
           peak_buffer_bytes = 0;
  QNN_INTERFACE_VER_TYPE q;
  QNN_SYSTEM_INTERFACE_VER_TYPE s;

  Qnn_BackendHandle_t backend = nullptr;
  Qnn_DeviceHandle_t device = nullptr;
  Qnn_LogHandle_t log = nullptr;

  void* (*allocate)(int, unsigned, int);
  void (*release)(void*);
  int (*get_fd)(void*);

  QnnHtpDevice_PerfInfrastructure_t perf{};
  uint32_t power_id = 0;

  static void
  logger(const char* fmt, QnnLog_Level_t level, uint64_t, va_list args);

  Runtime();

  void package(const char* path);

  ~Runtime();
};

// 4 KiB 对齐的 rpcmem 分配；fd 用于 QNN 注册和 FastRPC 映射。
// Runtime 必须比所有引用它的 Buffer 活得更久。
struct Buffer {
  Runtime* runtime;
  void* data;
  size_t size;
  int fd;

  Buffer(Runtime& r, size_t n, bool initialize = true);

  ~Buffer();
};

struct EmbeddedBinary {
  const uint8_t* pointer;
  size_t length;
};

extern std::map<std::string, EmbeddedBinary> embedded_binaries;

// PTE 内嵌 binary 只借用地址；外部 binary 才由本对象 mmap/munmap。
struct MappedBinary {
  uint8_t* pointer = nullptr;
  size_t length = 0;
  bool owned = false;

  explicit MappedBinary(const std::string& path);

  MappedBinary(const MappedBinary&) = delete;
  MappedBinary& operator=(const MappedBinary&) = delete;

  uint8_t* data() {
    return pointer;
  }

  size_t size() const {
    return length;
  }

  ~MappedBinary();
};

// System context 持有 tensor 元数据，不能在 inputs/outputs 使用期间释放。
// QNN 内存句柄属于各自的 context，即使多张图绑定的是同一个 Buffer。
struct Graph {
  Runtime& runtime;
  Qnn_ContextHandle_t context = nullptr;
  Qnn_GraphHandle_t graph = nullptr;
  QnnSystemContext_Handle_t system = nullptr;

  MappedBinary binary;
  std::vector<Qnn_Tensor_t> inputs, outputs;
  std::vector<std::shared_ptr<Buffer>> inbuf, outbuf;

  std::map<std::pair<void*, uint32_t>, Qnn_MemHandle_t> registered;
  std::string name, graph_name;
  std::map<uint32_t, std::shared_ptr<Buffer>> bindings;

  static size_t elements(const Qnn_Tensor_t& t);

  static size_t size(const Qnn_Tensor_t& t);

  Graph(Runtime& r, const std::string& path, bool metadata_only = false);

  void activate();

  void deactivate();

  void bind(Qnn_Tensor_t& t, const std::shared_ptr<Buffer>& buffer);

  void register_io();

  void unregister_io();

  void register_tensor(Qnn_Tensor_t& t, const std::shared_ptr<Buffer>& buffer);

  int index(bool input, int argument) const;

  int named(bool input, const char* value) const;

  void* input(int i) {
    return inbuf.at(i)->data;
  }

  void* output(int i) {
    return outbuf.at(i)->data;
  }

  void profile_event(QnnProfile_EventId_t event, int depth);

  void execute();

  ~Graph();
};

// 异步恢复一组常驻 context；load 返回前等待回调完成，保证回调参数仍然有效。
struct ContextRestore {
  struct Entry {
    ContextRestore* owner;
    Graph* target;
    Qnn_ContextHandle_t context = nullptr;
    bool completed = false;
  };

  std::mutex mutex;
  std::condition_variable changed;
  int complete = 0;
  Qnn_ErrorHandle_t error = 0;

  static void notify(
      Qnn_ContextHandle_t context,
      Qnn_GraphHandle_t,
      const char*,
      QnnContext_createFromBinaryAsyncNotifyType_t type,
      void* data,
      Qnn_ErrorHandle_t status);

  void load(Runtime& runtime, const std::vector<Graph*>& graphs);
};
