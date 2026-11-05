#include "shared/graph_binding.h"
#include "qnn_front.h"
#include <exception>
#include "binary_weights.h"
#include <cerrno>

std::map<std::string, EmbeddedBinary> embedded_binaries;

void check(Qnn_ErrorHandle_t status, const char* action) {
  if (status)
    throw std::runtime_error(
        std::string(action) + ": " + std::to_string(status));
}

void Runtime::logger(
    const char* fmt,
    QnnLog_Level_t level,
    uint64_t,
    va_list args) {
  if (level <= QNN_LOG_LEVEL_WARN || getenv("QNN_DEBUG")) {
    vfprintf(stderr, fmt, args);
    fputc('\n', stderr);
  }
}

Runtime::Runtime(Phase selected_phase, SharedGraphResolver resolver)
    : phase(selected_phase), shared_graphs(resolver) {
  try {
    const char* directory = getenv("MOE_QNN_LIBRARY_PATH");
    const std::string prefix = directory && *directory ? std::string(directory) + "/" : "";
    htp_library.reset(dlopen((prefix + "libQnnHtp.so").c_str(), RTLD_NOW | RTLD_GLOBAL));
    void* h = htp_library.get();
    if (!h)
      throw std::runtime_error(dlerror());
    const QnnInterface_t** providers = nullptr;
    uint32_t n = 0;
    auto get = (Qnn_ErrorHandle_t(*)(const QnnInterface_t***, uint32_t*))dlsym(
        h, "QnnInterface_getProviders");
    if (!get)
      throw std::runtime_error("QnnInterface_getProviders is missing from libQnnHtp.so");
    check(get(&providers, &n), "getProviders");
    const QnnInterface_t* selected = nullptr;
    for (uint32_t i = 0; providers && i < n; ++i)
      if (providers[i] &&
          providers[i]->apiVersion.coreApiVersion.major == QNN_API_VERSION_MAJOR &&
          providers[i]->apiVersion.coreApiVersion.minor >= QNN_API_VERSION_MINOR) {
        selected = providers[i];
        break;
      }
    if (!selected) {
      std::string versions;
      for (uint32_t i = 0; providers && i < n; ++i)
        if (providers[i]) {
          const auto& version = providers[i]->apiVersion.coreApiVersion;
          versions += " " + std::to_string(version.major) + "." + std::to_string(version.minor);
        }
      throw std::runtime_error("No compatible QNN HTP interface provider; check SDK libraries; expected=" +
          std::to_string(QNN_API_VERSION_MAJOR) + "." + std::to_string(QNN_API_VERSION_MINOR) +
          " count=" + std::to_string(n) + " received=" + versions);
    }
    q = selected->QNN_INTERFACE_VER_NAME;

    system_library.reset(dlopen((prefix + "libQnnSystem.so").c_str(), RTLD_NOW | RTLD_GLOBAL));
    void* sys = system_library.get();
    if (!sys)
      throw std::runtime_error(dlerror());
    const QnnSystemInterface_t** sp = nullptr;
    auto sg =
        (Qnn_ErrorHandle_t(*)(const QnnSystemInterface_t***, uint32_t*))dlsym(
            sys, "QnnSystemInterface_getProviders");
    if (!sg)
      throw std::runtime_error("QnnSystemInterface_getProviders is missing from libQnnSystem.so");
    n = 0;
    check(sg(&sp, &n), "systemProviders");
    const QnnSystemInterface_t* selected_system = nullptr;
    for (uint32_t i = 0; sp && i < n; ++i)
      if (sp[i] &&
          sp[i]->systemApiVersion.major == QNN_SYSTEM_API_VERSION_MAJOR &&
          sp[i]->systemApiVersion.minor >= QNN_SYSTEM_API_VERSION_MINOR) {
        selected_system = sp[i];
        break;
      }
    if (!selected_system)
      throw std::runtime_error("No compatible QNN system interface provider; check SDK libraries");
    s = selected_system->QNN_SYSTEM_INTERFACE_VER_NAME;

    rpc_library.reset(dlopen("libcdsprpc.so", RTLD_NOW | RTLD_GLOBAL));
    if (!rpc_library)
      rpc_library.reset(dlopen("libadsprpc.so", RTLD_NOW | RTLD_GLOBAL));
    void* rpc = rpc_library.get();
    if (!rpc)
      throw std::runtime_error(dlerror());
    allocate = (decltype(allocate))dlsym(rpc, "rpcmem_alloc");
    release = (decltype(release))dlsym(rpc, "rpcmem_free");
    get_fd = (decltype(get_fd))dlsym(rpc, "rpcmem_to_fd");
    if (!allocate || !release || !get_fd)
      throw std::runtime_error("rpcmem symbols");

    check(
        q.logCreate(
            logger,
            getenv("QNN_DEBUG") ? QNN_LOG_LEVEL_DEBUG : QNN_LOG_LEVEL_WARN,
            &log),
        "logCreate");
    check(q.backendCreate(log, nullptr, &backend), "backendCreate");
    check(q.deviceCreate(log, nullptr, &device), "deviceCreate");

    QnnDevice_Infrastructure_t infrastructure = nullptr;
    check(
        q.deviceGetInfrastructure(&infrastructure), "deviceGetInfrastructure");
    perf = ((QnnHtpDevice_Infrastructure_t*)infrastructure)->perfInfra;
    set_power_active(true);
  } catch (...) {
    release_handles();
    throw;
  }
}

void Runtime::set_power_active(bool active) {
  if (!active) {
    if (power_id) {
      check(perf.destroyPowerConfigId(power_id), "destroyPowerConfigId");
      power_id = 0;
    }
    return;
  }
  if (power_id)
    return;
  try {
    check(perf.createPowerConfigId(0, 0, &power_id), "createPowerConfigId");

    QnnHtpPerfInfrastructure_PowerConfig_t dcvs{}, latency{}, poll{};
    dcvs.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
    auto& d = dcvs.dcvsV3Config;
    d.contextId = power_id;
    d.setDcvsEnable = 1;
    d.dcvsEnable = 0;
    d.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
    d.setSleepLatency = 1;
    d.sleepLatency = 40;
    d.setSleepDisable = 1;
    d.sleepDisable = 1;
    d.setBusParams = 1;
    d.setCoreParams = 1;
    d.busVoltageCornerMin = d.busVoltageCornerTarget = d.busVoltageCornerMax =
        DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    d.coreVoltageCornerMin = d.coreVoltageCornerTarget =
        d.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    latency.option =
        QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY;
    latency.rpcControlLatencyConfig = 100;
    poll.option =
        QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME;
    poll.rpcPollingTimeConfig = 9999;
    const QnnHtpPerfInfrastructure_PowerConfig_t* config[] = {
        &dcvs, &latency, &poll, nullptr};
    check(perf.setPowerConfig(power_id, config), "setPowerConfig");
  } catch (...) {
    if (power_id)
      perf.destroyPowerConfigId(power_id);
    power_id = 0;
    throw;
  }
}

void Runtime::package(const char* path) {
  check(
      q.backendRegisterOpPackage(
          backend, path, "IntegerMoEOpPackageInterfaceProvider", "HTP"),
      "registerOpPackage");
}

Runtime::~Runtime() {
  release_handles();
}

void Runtime::release_handles() {
  if (power_id)
    perf.destroyPowerConfigId(power_id);
  if (device)
    q.deviceFree(device);
  if (backend)
    q.backendFree(backend);
  if (log)
    q.logFree(log);
  power_id = 0;
  device = nullptr;
  backend = nullptr;
  log = nullptr;
}

Buffer::Buffer(Runtime& r, size_t n, bool initialize, bool cached)
    : runtime(&r), size((n + 4095) & ~size_t(4095)) {
  data = r.allocate(25, cached ? 1 : 0, size);
  if (!data)
    throw std::runtime_error("rpcmem_alloc " + std::to_string(size));
  fd = r.get_fd(data);
  if (fd < 0) {
    r.release(data);
    data = nullptr;
    throw std::runtime_error("rpcmem_to_fd");
  }
  if (initialize)
    memset(data, 0, size);
  ++r.buffer_allocations;
  r.buffer_bytes += size;
  r.peak_buffer_bytes = std::max(r.peak_buffer_bytes, r.buffer_bytes);
}

Buffer::Buffer(Runtime& r, size_t n, CachePolicy policy)
    : Buffer(r, n, true, policy == CachePolicy::Cached) {}

Buffer::Buffer(Runtime& r, ImportedFd imported, size_t n)
    : inherited(true), runtime(&r), size(n), fd(imported.value) {
  if (fcntl(fd, F_GETFD) < 0)
    throw std::runtime_error("inherited fd closed");
  data = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (data == MAP_FAILED) {
    const int error = errno;
    close(fd);
    throw std::runtime_error("mmap resident fd errno=" + std::to_string(error));
  }
  ++r.buffer_allocations;
  r.buffer_bytes += size;
  r.peak_buffer_bytes = std::max(r.peak_buffer_bytes, r.buffer_bytes);
}

Buffer::~Buffer() {
  if (inherited) {
    munmap(data, size);
    close(fd);
  } else
    runtime->release(data);
  ++runtime->buffer_frees;
  runtime->buffer_bytes -= size;
}

MappedBinary::MappedBinary(const std::string& path) {
  if (path.empty())
    return;
  auto it = embedded_binaries.find(path);
  if (it != embedded_binaries.end()) {
    pointer = const_cast<uint8_t*>(it->second.pointer);
    length = it->second.length;
    return;
  }
  owned = true;
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0)
    throw std::runtime_error("open " + path);

  struct stat statbuf {};

  if (fstat(fd, &statbuf)) {
    close(fd);
    throw std::runtime_error("fstat " + path);
  }
  length = statbuf.st_size;
  void* mapped = mmap(nullptr, length, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (mapped == MAP_FAILED)
    throw std::runtime_error("mmap " + path);
  pointer = static_cast<uint8_t*>(mapped);
}

MappedBinary::~MappedBinary() {
  if (pointer && owned)
    munmap(pointer, length);
}

void MappedBinary::discard_pages() {
  // Keep the mapping and metadata addresses valid; clean file pages can fault
  // back in if the SDK needs them. Embedded PTE memory is never discarded.
  if (owned && pointer && madvise(pointer, length, MADV_DONTNEED))
    fprintf(stderr, "binary page advice failed: %d\n", errno);
}

size_t Graph::elements(const Qnn_Tensor_t& t) {
  size_t n = 1;
  for (unsigned i = 0; i < FIELD(t, rank); i++)
    n *= FIELD(t, dimensions)[i];
  return n;
}

size_t Graph::size(const Qnn_Tensor_t& t) {
  unsigned code = uint32_t(FIELD(t, dataType)) & 255;
  return elements(t) * (((code >> 4) * 10 + (code & 15)) / 8);
}

bool Graph::same_encoding_and_layout(
    const Qnn_Tensor_t& first, const Qnn_Tensor_t& second) {
  if (FIELD(first, dataType) != FIELD(second, dataType) ||
      FIELD(first, rank) != FIELD(second, rank) ||
      !std::equal(
          FIELD(first, dimensions),
          FIELD(first, dimensions) + FIELD(first, rank),
          FIELD(second, dimensions)))
    return false;
  const auto a = FIELD(first, quantizeParams);
  const auto b = FIELD(second, quantizeParams);
  if (a.encodingDefinition != b.encodingDefinition ||
      a.quantizationEncoding != b.quantizationEncoding)
    return false;
  if (a.quantizationEncoding == QNN_QUANTIZATION_ENCODING_UNDEFINED)
    return true;
  return a.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET &&
      a.scaleOffsetEncoding.scale == b.scaleOffsetEncoding.scale &&
      a.scaleOffsetEncoding.offset == b.scaleOffsetEncoding.offset;
}

Graph::Graph(Runtime& r, const std::string& path, bool metadata_only,
             bool callback_weights)
    : runtime(r),
      binary(r.shared_graphs && r.shared_graphs(path) ? "" : path),
      name(path) {
  try {
    const auto binding = r.shared_graphs ? r.shared_graphs(path) : std::nullopt;
    borrowed_context = binding.has_value();
    const QnnSystemContext_GraphInfoV1_t* selected = nullptr;
    if (borrowed_context) {
      selected = borrow_shared_graph(*binding, &context, &graph);
    } else {
      check(r.s.systemContextCreate(&system), "systemContextCreate");
      const QnnSystemContext_BinaryInfo_t* info = nullptr;
      Qnn_ContextBinarySize_t n = 0;
      check(
          r.s.systemContextGetBinaryInfo(
              system, binary.data(), binary.size(), &info, &n),
          "getBinaryInfo");
      QnnSystemContext_GraphInfo_t* g =
          info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1
          ? info->contextBinaryInfoV1.graphs
          : info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2
          ? info->contextBinaryInfoV2.graphs
          : info->contextBinaryInfoV3.graphs;
      selected = &g[0].graphInfoV1;
    }
    const auto& gi = *selected;
    graph_name = gi.graphName;
    inputs.assign(gi.graphInputs, gi.graphInputs + gi.numGraphInputs);
    outputs.assign(gi.graphOutputs, gi.graphOutputs + gi.numGraphOutputs);
    if (!borrowed_context && callback_weights) {
      weights = std::make_unique<BinaryWeights>(r, path);
      binary.discard_pages();
      weights->discard_file_cache();
    }

    for (auto& t : inputs) {
      auto b = std::make_shared<Buffer>(r, size(t));
      inbuf.push_back(b);
      if (!metadata_only)
        bindings[FIELD(t, id)] = b;
    }
    for (auto& t : outputs) {
      if (metadata_only) {
        outbuf.push_back(nullptr);
        continue;
      }
      auto b = std::make_shared<Buffer>(r, size(t));
      outbuf.push_back(b);
      bindings[FIELD(t, id)] = b;
    }
    if (!metadata_only && !r.defer_contexts)
      activate();

    for (int direction = 0; direction < 2; direction++) {
      auto& tensors = direction ? outputs : inputs;
      for (size_t i = 0; i < tensors.size(); i++) {
        auto& t = tensors[i];
        fprintf(
            stderr,
            "IO %s %s %zu %s dt=%x dims=",
            path.c_str(),
            direction ? "out" : "in",
            i,
            FIELD(t, name),
            FIELD(t, dataType));
        for (unsigned j = 0; j < FIELD(t, rank); j++)
          fprintf(stderr, "%u,", FIELD(t, dimensions)[j]);
        auto qp = FIELD(t, quantizeParams);
        if (qp.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET)
          fprintf(
              stderr,
              " scale=%g offset=%d",
              qp.scaleOffsetEncoding.scale,
              qp.scaleOffsetEncoding.offset);
        fprintf(stderr, "\n");
      }
    }
  } catch (...) {
    try {
      deactivate();
    } catch (const std::exception& error) {
      fprintf(stderr, "graph construction cleanup failed: %s\n", error.what());
    }
    if (system)
      runtime.s.systemContextFree(system);
    throw;
  }
}

void Graph::activate() {
  // Borrowed graphs already have a context but still need run configuration.
  if (context) {
    if (runtime.phase == Runtime::Phase::Prefill)
      configure_threads();
    return;
  }
  if (weights) {
    check(
        runtime.q.contextCreateFromBinaryWithCallback(
            runtime.backend,
            runtime.device,
            nullptr,
            weights->callback(),
            binary.data(),
            binary.size(),
            &context,
            nullptr,
            nullptr),
        "contextCreateFromBinaryWithCallback");
    binary.discard_pages();
    weights->discard_file_cache();
  } else {
    QnnHtpContext_CustomConfig_t custom{};
    custom.option = QNN_HTP_CONTEXT_CONFIG_OPTION_IO_MEM_ESTIMATION;
    custom.ioMemEstimation = true;
    QnnContext_Config_t option = QNN_CONTEXT_CONFIG_INIT;
    option.option = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
    option.customConfig = &custom;
    const QnnContext_Config_t* config[] = {&option, nullptr};
    check(
        runtime.q.contextCreateFromBinary(
            runtime.backend,
            runtime.device,
            runtime.phase == Runtime::Phase::Decode && getenv("QNN_SERIAL_CONTEXTS") ? config : nullptr,
            binary.data(),
            binary.size(),
            &context,
            nullptr),
        "contextCreateFromBinary");
  }
  ++runtime.context_creates;
  ++runtime.live_contexts;
  runtime.peak_contexts =
      std::max(runtime.peak_contexts, runtime.live_contexts);
  check(
      runtime.q.graphRetrieve(context, graph_name.c_str(), &graph),
      "graphRetrieve");
  if (runtime.phase == Runtime::Phase::Prefill)
    configure_threads();
  if (!runtime.defer_bindings)
    register_io();
}

void Graph::configure_threads() {
  if (const char* threads = getenv("QNN_HVX_THREADS")) {
    QnnHtpGraph_CustomConfig_t custom{};
    custom.option = QNN_HTP_GRAPH_CONFIG_OPTION_NUM_HVX_THREADS;
    custom.numHvxThreads = atoi(threads);
    QnnGraph_Config_t option = QNN_GRAPH_CONFIG_INIT;
    option.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
    option.customConfig = &custom;
    const QnnGraph_Config_t* config[] = {&option, nullptr};
    check(
        runtime.q.graphSetConfig(graph, config), "graphSetConfig HVX threads");
  }
}

void Graph::deactivate() {
  if (!context)
    return;
  std::exception_ptr unregister_error;
  try {
    unregister_io();
  } catch (...) {
    unregister_error = std::current_exception();
  }
  if (borrowed_context) {
    release_shared_graph(context);
    context = nullptr;
    graph = nullptr;
    if (unregister_error)
      std::rethrow_exception(unregister_error);
    return;
  }
  check(runtime.q.contextFree(context, nullptr), "contextFree");
  ++runtime.context_frees;
  --runtime.live_contexts;
  context = nullptr;
  graph = nullptr;
  if (unregister_error)
    std::rethrow_exception(unregister_error);
}

void Graph::bind(Qnn_Tensor_t& t, const std::shared_ptr<Buffer>& buffer) {
  bindings[FIELD(t, id)] = buffer;
  if (!context || runtime.defer_bindings)
    return;
  register_tensor(t, buffer);
}

void Graph::register_io() {
  if (!context)
    throw std::runtime_error("register IO without context");
  if (runtime.phase == Runtime::Phase::Decode)
    configure_threads();
  for (auto& t : inputs)
    register_tensor(t, bindings.at(FIELD(t, id)));
  for (auto& t : outputs)
    register_tensor(t, bindings.at(FIELD(t, id)));
}

void Graph::unregister_io() {
  if (registered.empty())
    return;
  std::vector<Qnn_MemHandle_t> handles;
  for (auto& item : registered)
    handles.push_back(item.second);
  if (borrowed_context) {
    for (auto handle : handles)
      check(unregister_shared_io(context, handle), "shared IO unregister");
  } else {
    check(
        runtime.q.memDeRegister(handles.data(), handles.size()),
        "memDeRegister IO");
  }
  runtime.memory_deregisters += handles.size();
  registered.clear();
}

void Graph::register_tensor(
    Qnn_Tensor_t& t,
    const std::shared_ptr<Buffer>& buffer) {
  // 同一 buffer 可能对应不同 tensor 描述，注册缓存同时按地址和 tensor ID
  // 区分。
  const auto key = std::make_pair(buffer->data, FIELD(t, id));
  auto found = registered.find(key);
  Qnn_MemHandle_t handle;
  if (found == registered.end()) {
    Qnn_MemDescriptor_t descriptor = QNN_MEM_DESCRIPTOR_INIT;
    descriptor.memShape = {FIELD(t, rank), FIELD(t, dimensions), nullptr};
    descriptor.dataType = FIELD(t, dataType);
    if (buffer->size < size(t))
      throw std::runtime_error("QNN tensor exceeds buffer allocation");
    descriptor.memType = QNN_MEM_TYPE_ION;
    descriptor.ionInfo.fd = buffer->fd;
    QnnMemHtp_Descriptor_t shared{};
    if (buffer->shared_views ||
        buffer->size > ((size(t) + 4095) & ~size_t(4095))) {
      // All views of this fd must use one mapping type and allocation size.
      shared.type = QNN_HTP_MEM_SHARED_BUFFER;
      shared.size = buffer->size;
      shared.sharedBufferConfig = {buffer->fd, 0};
      descriptor.memType = QNN_MEM_TYPE_CUSTOM;
      descriptor.customInfo = &shared;
    }
    check(
        borrowed_context
            ? register_shared_io(context, &descriptor, &handle)
            : runtime.q.memRegister(context, &descriptor, 1, &handle),
        "memRegister");
    ++runtime.memory_registers;
    registered[key] = handle;
  } else
    handle = found->second;
  FIELD(t, memType) = QNN_TENSORMEMTYPE_MEMHANDLE;
  FIELD(t, memHandle) = handle;
}

int Graph::index(bool input, int argument) const {
  const auto& tensors = input ? inputs : outputs;
  const std::string prefix =
      (input ? "input_" : "output_") + std::to_string(argument) + "_";
  for (size_t i = 0; i < tensors.size(); i++)
    if (std::string(FIELD(tensors[i], name)).rfind(prefix, 0) == 0)
      return i;
  if (!input && argument < int(tensors.size()))
    return argument;
  if (tensors.size() == 1)
    return 0;
  throw std::runtime_error("missing " + prefix + " in " + name);
}

int Graph::named(bool input, const char* value) const {
  const auto& ts = input ? inputs : outputs;
  for (size_t i = 0; i < ts.size(); i++)
    if (std::string(FIELD(ts[i], name)) == value)
      return i;
  throw std::runtime_error("missing tensor " + std::string(value));
}

void Graph::profile_event(QnnProfile_EventId_t event, int depth) {
  QnnProfile_EventData_t data = QNN_PROFILE_EVENT_DATA_INIT;
  const int status = runtime.q.profileGetEventData(event, &data);
  if (!status)
    printf(
        "QNN_EVENT graph=%s depth=%d type=%u unit=%u value=%llu name=%s\n",
        name.c_str(),
        depth,
        data.type,
        data.unit,
        (unsigned long long)data.value,
        data.identifier ? data.identifier : "");
  const QnnProfile_EventId_t* events = nullptr;
  uint32_t count = 0;
  check(
      runtime.q.profileGetSubEvents(event, &events, &count),
      "profileGetSubEvents");
  for (unsigned i = 0; i < count; i++)
    profile_event(events[i], depth + 1);
}

void Graph::execute() {
  Qnn_ProfileHandle_t profile = nullptr;
  const char* pattern = getenv("QNN_PROFILE");
  if (pattern &&
      (std::string(pattern) == "all" ||
       name.find(pattern) != std::string::npos))
    check(
        runtime.q.profileCreate(
            runtime.backend, QNN_PROFILE_LEVEL_DETAILED, &profile),
        "profileCreate");
  if (runtime.phase == Runtime::Phase::Decode && getenv("QNN_BASIC_ONCE") &&
      profile_calls++ == 8 && !profile)
    check(runtime.q.profileCreate(runtime.backend, QNN_PROFILE_LEVEL_BASIC, &profile),
          "profileCreate basic");
  check(
      runtime.q.graphExecute(
          graph,
          inputs.data(),
          inputs.size(),
          outputs.data(),
          outputs.size(),
          profile,
          nullptr),
      "graphExecute");
  if (profile) {
    const QnnProfile_EventId_t* events = nullptr;
    uint32_t count = 0;
    check(
        runtime.q.profileGetEvents(profile, &events, &count),
        "profileGetEvents");
    for (unsigned i = 0; i < count; i++)
      profile_event(events[i], 0);
    check(runtime.q.profileFree(profile), "profileFree");
  }
}

Graph::~Graph() {
  try {
    deactivate();
  } catch (const std::exception& error) {
    fprintf(stderr, "graph cleanup failed: %s\n", error.what());
  }
  if (system)
    runtime.s.systemContextFree(system);
}

void ContextRestore::notify(
    Qnn_ContextHandle_t context,
    Qnn_GraphHandle_t,
    const char*,
    QnnContext_createFromBinaryAsyncNotifyType_t type,
    void* data,
    Qnn_ErrorHandle_t status) {
  auto& entry = *static_cast<Entry*>(data);
  auto& owner = *entry.owner;
  std::lock_guard<std::mutex> guard(owner.mutex);
  if (status && !owner.error)
    owner.error = status;
  if (context)
    entry.context = context;
  if (type == QNN_CONTEXT_NOTIFY_TYPE_CONTEXT_INIT && !entry.completed) {
    entry.completed = true;
    ++owner.complete;
  }
  owner.changed.notify_all();
}

void ContextRestore::load(Runtime& runtime, const std::vector<Graph*>& graphs) {
  if (getenv("SHARED_MODEL_ROOT")) {
    for (auto* graph : graphs)
      graph->activate();
    return;
  }
  if (runtime.phase == Runtime::Phase::Decode && getenv("QNN_SERIAL_CONTEXTS")) {
    for (auto* graph : graphs)
      graph->activate();
    return;
  }
  std::vector<Entry> entries;
  entries.reserve(graphs.size());
  std::vector<QnnContext_Params_t> parameters;
  parameters.reserve(graphs.size());
  std::vector<const QnnContext_Params_t*> list;
  for (auto* graph : graphs) {
    entries.push_back({this, graph});
    parameters.push_back(
        {QNN_CONTEXT_PARAMS_VERSION_1,
         {{nullptr,
           graph->binary.data(),
           graph->binary.size(),
           nullptr,
           notify,
           &entries.back()}}});
  }
  for (auto& param : parameters)
    list.push_back(&param);
  list.push_back(nullptr);
  QnnHtpContext_CustomConfig_t custom[3]{};
  custom[0].option = QNN_HTP_CONTEXT_CONFIG_OPTION_SHARE_RESOURCES;
  custom[0].shareResources = true;
  custom[1].option =
      QNN_HTP_CONTEXT_CONFIG_OPTION_SHARE_RESOURCES_OPTIMIZATION_TYPE;
  custom[1].shareResOptType = SEQUENTIAL_WITHOUT_VA_OPTIMIZATION;
  custom[2].option = QNN_HTP_CONTEXT_CONFIG_OPTION_REUSED_IO_LIMIT;
  std::map<int, size_t> io_buffers;
  for (auto* graph : graphs)
    for (auto& binding : graph->bindings)
      io_buffers[binding.second->fd] = binding.second->size;
  size_t io_bytes = 0;
  for (auto& buffer : io_buffers)
    io_bytes += buffer.second;
  custom[2].reusedIoLimitMb =
      std::max(size_t(16), (io_bytes + 1048575) / 1048576);
  QnnContext_Config_t config[3]{};
  const QnnContext_Config_t* configs[4]{};
  for (int i = 0; i < 3; i++) {
    config[i].option = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
    config[i].customConfig = &custom[i];
    configs[i] = config + i;
  }
  check(
      runtime.q.contextCreateFromBinaryListAsync(
          runtime.backend, runtime.device, list.data(), configs, nullptr),
      "contextCreateFromBinaryListAsync sequential resource sharing");
  {
    std::unique_lock<std::mutex> guard(mutex);
    changed.wait(guard, [&] {
      return complete == int(graphs.size());
    });
  }
  for (auto& entry : entries)
    if (entry.context) {
      entry.target->context = entry.context;
      ++runtime.context_creates;
      ++runtime.live_contexts;
      runtime.peak_contexts =
          std::max(runtime.peak_contexts, runtime.live_contexts);
    }
  check(error, "context initialization callback");
  for (auto* graph : graphs) {
    check(
        runtime.q.graphRetrieve(
            graph->context, graph->graph_name.c_str(), &graph->graph),
        "graphRetrieve initialized context");
    if (runtime.phase == Runtime::Phase::Prefill)
      graph->configure_threads();
  }
}
