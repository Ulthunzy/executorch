#pragma once

// AR1 使用常驻 QNN context；共享 IO 和专家 DMA buffer 在整轮 decode 中复用。
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

// 一次 session 共享 backend/device，计数器用于检查逐 token 路径是否重建资源。
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

// 支持新分配 rpcmem，以及接管同进程 prefill 传来的 DMA fd。
// 析构方式取决于创建方式；Runtime 的生命周期必须覆盖所有 Buffer。
struct Buffer {
  enum class CachePolicy { Cached, Uncached };
  Runtime* runtime;
  void* data;
  size_t size;
  int fd;
  bool inherited = false;

  Buffer(Runtime& r, size_t n, CachePolicy policy = CachePolicy::Cached);

  Buffer(Runtime& r, int imported_fd, size_t n);

  ~Buffer();
};

struct EmbeddedBinary {
  const uint8_t* pointer;
  size_t length;
};

extern std::map<std::string, EmbeddedBinary> embedded_binaries;

// 内嵌 binary 借用 PTE 内存，外部 binary 则拥有独立的文件映射。
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

// tensor 元数据依赖 System context；memHandle 则属于执行它的 QNN context。
// bind 允许多层复用 IO 内存，但每层的 state/KV 仍绑定各自的 Buffer。
struct Graph {
  unsigned profile_calls = 0;
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

// 集中恢复常驻 context，并等待异步回调后再注册共享 IO。
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
