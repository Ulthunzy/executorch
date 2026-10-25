#include "context_api.h"
#include "weight_source.h"
#include "../decode/qnn_front.h"
#include <tuple>

namespace {
std::mutex registry_mutex;

struct Context {
  Runtime& runtime;
  std::unique_ptr<WeightSource> weights;
  Qnn_ContextHandle_t handle = nullptr;
  QnnSystemContext_Handle_t system = nullptr;
  const QnnSystemContext_BinaryInfo_t* metadata = nullptr;
  size_t references = 0;
  bool initialized = false;
  std::mutex* init_mutex = nullptr;
  std::condition_variable* init_changed = nullptr;
  Qnn_ErrorHandle_t init_error = QNN_SUCCESS;
  using MemoryKey = std::tuple<int, Qnn_DataType_t, std::vector<uint32_t>>;

  struct Memory {
    Qnn_MemHandle_t handle;
    size_t references;
  };

  std::map<MemoryKey, Memory> memory;

  Context(Runtime& r, const std::string& directory) : runtime(r) {
    const void* image;
    size_t size;
    weights = std::make_unique<WeightSource>(r, directory);
    image = weights->image();
    size = weights->image_size();
    try {
      check(r.s.systemContextCreate(&system), "shared metadata create");
      Qnn_ContextBinarySize_t length = 0;
      check(
          r.s.systemContextGetBinaryInfo(
              system, const_cast<void*>(image), size, &metadata, &length),
          "shared metadata");
      if (metadata->version != QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3)
        throw std::runtime_error("shared binary metadata version");
    } catch (...) {
      if (handle)
        r.q.contextFree(handle, nullptr);
      if (system)
        r.s.systemContextFree(system);
      throw;
    }
  }

  ~Context() {
    if (!memory.empty())
      fprintf(stderr, "SHARED_CONTEXT_LIVE_IO count=%zu\n", memory.size());
    const auto status =
        handle ? runtime.q.contextFree(handle, nullptr) : QNN_SUCCESS;
    if (status)
      fprintf(
          stderr,
          "SHARED_CONTEXT_FREE_FAILED %llu\n",
          (unsigned long long)status);
    runtime.s.systemContextFree(system);
    fprintf(
        stderr,
        "SHARED_CONTEXT_RELEASE provides=%llu releases=%llu live=%zu\n",
        (unsigned long long)(weights ? weights->provides : 0),
        (unsigned long long)(weights ? weights->releases : 0),
        weights ? weights->live_allocations() : 0);
  }
};

struct Store {
  Runtime runtime;
  std::map<std::string, std::unique_ptr<Context>> contexts;

  static void notify(
      Qnn_ContextHandle_t handle,
      Qnn_GraphHandle_t,
      const char*,
      QnnContext_createFromBinaryAsyncNotifyType_t type,
      void* opaque,
      Qnn_ErrorHandle_t status) {
    auto& entry = *static_cast<Context*>(opaque);
    std::lock_guard<std::mutex> lock(*entry.init_mutex);
    if (handle)
      entry.handle = handle;
    if (status)
      entry.init_error = status;
    if (type == QNN_CONTEXT_NOTIFY_TYPE_CONTEXT_INIT)
      entry.initialized = true;
    entry.init_changed->notify_all();
  }

  void load(const std::string& root) {
    std::mutex mutex;
    std::condition_variable changed;
    std::vector<Context*> entries;
    for (int group = 0; group <= 10; ++group) {
      const std::string path =
          root + (group == 10 ? "/head" : "/group" + std::to_string(group));
      auto entry = std::make_unique<Context>(runtime, path);
      entry->init_mutex = &mutex;
      entry->init_changed = &changed;
      entries.push_back(entry.get());
      contexts.emplace(path, std::move(entry));
    }
    std::vector<QnnContext_ParamsV2_t> details;
    details.reserve(entries.size());
    std::vector<QnnContext_Params_t> parameters;
    parameters.reserve(entries.size());
    std::vector<const QnnContext_Params_t*> list;
    for (size_t i = 0; i < entries.size(); ++i) {
      auto& entry = *entries[i];
      details.push_back(
          {nullptr,
           entry.weights->image(),
           entry.weights->image_size(),
           nullptr,
           notify,
           &entry,
           entry.weights->callback()});
      parameters.push_back(QnnContext_Params_t{});
      parameters[i].version = QNN_CONTEXT_PARAMS_VERSION_2;
      parameters[i].v2 = &details.back();
      list.push_back(&parameters[i]);
    }
    list.push_back(nullptr);
    QnnHtpContext_CustomConfig_t custom[3]{};
    custom[0].option = QNN_HTP_CONTEXT_CONFIG_OPTION_SHARE_RESOURCES;
    custom[0].shareResources = true;
    custom[1].option =
        QNN_HTP_CONTEXT_CONFIG_OPTION_SHARE_RESOURCES_OPTIMIZATION_TYPE;
    custom[1].shareResOptType = SEQUENTIAL_WITHOUT_VA_OPTIMIZATION;
    custom[2].option = QNN_HTP_CONTEXT_CONFIG_OPTION_REUSED_IO_LIMIT;
    size_t cache_bytes = 0;
    for (const auto* entry : entries) {
      const auto& info = entry->metadata->contextBinaryInfoV3;
      for (uint32_t i = 0; i < info.numGraphs; ++i) {
        const auto& graph = info.graphs[i].graphInfoV1;
        for (uint32_t j = 0; j < graph.numGraphInputs; ++j)
          if (FIELD(graph.graphInputs[j], dataType) ==
              QNN_DATATYPE_UFIXED_POINT_16)
            cache_bytes += Graph::size(graph.graphInputs[j]);
        for (uint32_t j = 0; j < graph.numGraphOutputs; ++j)
          if (FIELD(graph.graphOutputs[j], dataType) ==
              QNN_DATATYPE_UFIXED_POINT_16)
            cache_bytes += Graph::size(graph.graphOutputs[j]);
      }
    }
    // Cache IO is private to each layer/phase; activation IO is shared. Round
    // its allowance to 16 MiB without reserving a second copy of every tensor.
    custom[2].reusedIoLimitMb = std::max<size_t>(
        192, ((cache_bytes + 31 * 1048576) / (16 * 1048576)) * 16);
    QnnContext_Config_t config[3]{};
    const QnnContext_Config_t* configs[4]{};
    for (int i = 0; i < 3; ++i) {
      config[i].option = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
      config[i].customConfig = &custom[i];
      configs[i] = &config[i];
    }
    check(
        runtime.q.contextCreateFromBinaryListAsync(
            runtime.backend, runtime.device, list.data(), configs, nullptr),
        "shared asynchronous contexts");
    std::unique_lock<std::mutex> lock(mutex);
    changed.wait(lock, [&] {
      return std::all_of(entries.begin(), entries.end(), [](const auto* e) {
        return e->initialized;
      });
    });
    for (auto* entry : entries) {
      check(entry->init_error, "shared asynchronous context callback");
      fprintf(
          stderr,
          "SHARED_CONTEXT_READY graphs=%u weight_bytes=%llu provides=%llu\n",
          entry->metadata->contextBinaryInfoV3.numGraphs,
          (unsigned long long)entry->weights->allocated_bytes,
          (unsigned long long)entry->weights->provides);
    }
    fprintf(
        stderr,
        "SHARED_CONTEXT_BATCH contexts=%zu resource_sharing=1 va_optimization=0 io_budget_mb=%llu\n",
        entries.size(),
        (unsigned long long)custom[2].reusedIoLimitMb);
  }

  Store() {
    // Limit heap growth slack while keeping every context resident.
    QnnHtpPerfInfrastructure_MemoryConfig_t memory{};
    memory.option = QNN_HTP_PERF_INFRASTRUCTURE_MEMORY_CONFIGOPTION_GROW_SIZE;
    memory.memGrowSizeConfig = 4 * 1048576;
    const QnnHtpPerfInfrastructure_MemoryConfig_t* configs[] = {
        &memory, nullptr};
    check(runtime.perf.setMemoryConfig(0, 0, configs), "shared heap grow size");
    const char* package = getenv("SHARED_OP_PACKAGE");
    if (!package)
      throw std::runtime_error("SHARED_OP_PACKAGE is required");
    check(
        runtime.q.backendRegisterOpPackage(
            runtime.backend,
            package,
            "DeltaNetOpPackageInterfaceProvider",
            "HTP"),
        "shared GDN package");
  }
};

Store& store() {
  static Store value;
  return value;
}
} // namespace

extern "C" __attribute__((visibility("default"))) int qwen35_shared_acquire(
    const char* directory,
    const char* graph_name,
    Qnn_ContextHandle_t* context,
    Qnn_GraphHandle_t* graph,
    const QnnSystemContext_GraphInfoV1_t** metadata) {
  try {
    std::lock_guard<std::mutex> guard(registry_mutex);
    auto& owner = store();
    if (owner.contexts.empty()) {
      const auto root = std::string(directory).substr(
          0, std::string(directory).find_last_of('/'));
      owner.load(root);
    }
    auto& entry = owner.contexts.at(directory);
    const auto& info = entry->metadata->contextBinaryInfoV3;
    for (uint32_t i = 0; i < info.numGraphs; ++i) {
      const auto& candidate = info.graphs[i].graphInfoV1;
      if (!*graph_name || std::string(candidate.graphName) == graph_name) {
        check(
            owner.runtime.q.graphRetrieve(
                entry->handle, candidate.graphName, graph),
            "shared graph retrieve");
        *context = entry->handle;
        *metadata = &candidate;
        ++entry->references;
        return 0;
      }
    }
    throw std::runtime_error(
        "shared graph not found: " + std::string(graph_name));
  } catch (const std::exception& error) {
    fprintf(stderr, "SHARED_ACQUIRE_FAILED %s\n", error.what());
    return 1;
  }
}

extern "C" __attribute__((visibility("default"))) void qwen35_shared_release(
    Qnn_ContextHandle_t context) {
  std::lock_guard<std::mutex> guard(registry_mutex);
  auto& entries = store().contexts;
  for (auto it = entries.begin(); it != entries.end(); ++it)
    if (it->second && it->second->handle == context) {
      if (!--it->second->references)
        entries.erase(it);
      return;
    }
}

extern "C" __attribute__((visibility("default"))) int qwen35_shared_register(
    Qnn_ContextHandle_t context,
    const Qnn_MemDescriptor_t* descriptor,
    Qnn_MemHandle_t* handle) {
  try {
    std::lock_guard<std::mutex> guard(registry_mutex);
    auto& owner = store();
    for (auto& pair : owner.contexts) {
      auto& entry = *pair.second;
      if (entry.handle != context)
        continue;
      Context::MemoryKey key{
          descriptor->ionInfo.fd,
          descriptor->dataType,
          std::vector<uint32_t>(
              descriptor->memShape.dimSize,
              descriptor->memShape.dimSize + descriptor->memShape.numDim)};
      auto found = entry.memory.find(key);
      if (found == entry.memory.end()) {
        check(
            owner.runtime.q.memRegister(context, descriptor, 1, handle),
            "shared IO register");
        entry.memory.emplace(std::move(key), Context::Memory{*handle, 1});
      } else {
        *handle = found->second.handle;
        ++found->second.references;
      }
      return 0;
    }
  } catch (const std::exception& error) {
    fprintf(stderr, "SHARED_REGISTER_FAILED %s\n", error.what());
  }
  return 1;
}

extern "C" __attribute__((visibility("default"))) int qwen35_shared_unregister(
    Qnn_ContextHandle_t context,
    Qnn_MemHandle_t handle) {
  std::lock_guard<std::mutex> guard(registry_mutex);
  auto& owner = store();
  for (auto& pair : owner.contexts) {
    auto& entry = *pair.second;
    if (entry.handle != context)
      continue;
    for (auto it = entry.memory.begin(); it != entry.memory.end(); ++it)
      if (it->second.handle == handle) {
        if (!--it->second.references) {
          const auto status = owner.runtime.q.memDeRegister(&handle, 1);
          if (status)
            return int(status);
          entry.memory.erase(it);
        }
        return 0;
      }
  }
  return 1;
}

extern "C" __attribute__((visibility("default"))) uint64_t
qwen35_shared_context_count() {
  std::lock_guard<std::mutex> guard(registry_mutex);
  return store().contexts.size();
}
