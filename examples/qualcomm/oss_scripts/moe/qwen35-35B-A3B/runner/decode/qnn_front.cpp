#include "../shared/graph_binding.h"
#include "qnn_front.h"

std::map<std::string, EmbeddedBinary> embedded_binaries;

void check(Qnn_ErrorHandle_t status, const char* action) {
  if (status)
    throw std::runtime_error(
        std::string(action) + ": " + std::to_string(status));
}

std::vector<uint8_t> bytes(const std::string& name) {
  std::ifstream f(name, std::ios::binary | std::ios::ate);
  if (!f)
    throw std::runtime_error("open " + name);
  size_t n = f.tellg();
  std::vector<uint8_t> b(n);
  f.seekg(0);
  f.read((char*)b.data(), n);
  return b;
}

void save(const std::string& name, const void* p, size_t n) {
  FILE* f = fopen(name.c_str(), "wb");
  if (!f)
    throw std::runtime_error("save " + name);
  if (fwrite(p, 1, n, f) != n)
    throw std::runtime_error("write " + name);
  fflush(f);
  fsync(fileno(f));
  fclose(f);
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

Runtime::Runtime() {
  void* h = dlopen("libQnnHtp.so", RTLD_NOW | RTLD_GLOBAL);
  if (!h)
    throw std::runtime_error(dlerror());
  const QnnInterface_t** providers = nullptr;
  uint32_t n = 0;
  auto get = (Qnn_ErrorHandle_t(*)(const QnnInterface_t***, uint32_t*))dlsym(
      h, "QnnInterface_getProviders");
  check(get(&providers, &n), "getProviders");
  q = providers[0]->QNN_INTERFACE_VER_NAME;

  void* sys = dlopen("libQnnSystem.so", RTLD_NOW | RTLD_GLOBAL);
  if (!sys)
    throw std::runtime_error(dlerror());
  const QnnSystemInterface_t** sp = nullptr;
  auto sg =
      (Qnn_ErrorHandle_t(*)(const QnnSystemInterface_t***, uint32_t*))dlsym(
          sys, "QnnSystemInterface_getProviders");
  check(sg(&sp, &n), "systemProviders");
  s = sp[0]->QNN_SYSTEM_INTERFACE_VER_NAME;

  void* rpc = dlopen("libcdsprpc.so", RTLD_NOW | RTLD_GLOBAL);
  if (!rpc)
    rpc = dlopen("libadsprpc.so", RTLD_NOW | RTLD_GLOBAL);
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
  check(q.deviceGetInfrastructure(&infrastructure), "deviceGetInfrastructure");
  perf = ((QnnHtpDevice_Infrastructure_t*)infrastructure)->perfInfra;
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
  d.coreVoltageCornerMin = d.coreVoltageCornerTarget = d.coreVoltageCornerMax =
      DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  latency.option =
      QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY;
  latency.rpcControlLatencyConfig = 100;
  poll.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME;
  poll.rpcPollingTimeConfig = 9999;
  const QnnHtpPerfInfrastructure_PowerConfig_t* config[] = {
      &dcvs, &latency, &poll, nullptr};
  check(perf.setPowerConfig(power_id, config), "setPowerConfig");
}

void Runtime::package(const char* path) {
  check(
      q.backendRegisterOpPackage(
          backend, path, "IntegerMoEOpPackageInterfaceProvider", "HTP"),
      "registerOpPackage");
}

Runtime::~Runtime() {
  if (power_id)
    perf.destroyPowerConfigId(power_id);
  if (device)
    q.deviceFree(device);
  if (backend)
    q.backendFree(backend);
  if (log)
    q.logFree(log);
}

Buffer::Buffer(Runtime& r, size_t n, CachePolicy policy)
    : runtime(&r), size((n + 4095) & ~size_t(4095)) {
  data = r.allocate(25, policy == CachePolicy::Cached ? 1 : 0, size);
  if (!data)
    throw std::runtime_error("rpcmem_alloc " + std::to_string(size));
  fd = r.get_fd(data);
  if (fd < 0)
    throw std::runtime_error("rpcmem_to_fd");
  memset(data, 0, size);
  ++r.buffer_allocations;
  r.buffer_bytes += size;
  r.peak_buffer_bytes = std::max(r.peak_buffer_bytes, r.buffer_bytes);
}

Buffer::Buffer(Runtime& r, int imported_fd, size_t n)
    : runtime(&r), size(n), fd(imported_fd), inherited(true) {
  if (fcntl(fd, F_GETFD) < 0)
    throw std::runtime_error("inherited fd closed");
  data = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (data == MAP_FAILED)
    throw std::runtime_error("mmap resident fd errno=" + std::to_string(errno));
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

Graph::Graph(Runtime& r, const std::string& path, bool metadata_only)
    : runtime(r), binary(shared_graph_enabled(path) ? "" : path), name(path) {
  borrowed_context = shared_graph_enabled(path);
  const QnnSystemContext_GraphInfoV1_t* selected = nullptr;
  if (borrowed_context) {
    selected = borrow_shared_graph(path, &context, &graph);
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
}

void Graph::activate() {
  if (context)
    return;
  // Estimate IO before the SDK selects a process domain for a resident graph.
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
          getenv("QNN_SERIAL_CONTEXTS") ? config : nullptr,
          binary.data(),
          binary.size(),
          &context,
          nullptr),
      "contextCreateFromBinary");
  ++runtime.context_creates;
  ++runtime.live_contexts;
  runtime.peak_contexts =
      std::max(runtime.peak_contexts, runtime.live_contexts);
  check(
      runtime.q.graphRetrieve(context, graph_name.c_str(), &graph),
      "graphRetrieve");
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
  if (!runtime.defer_bindings)
    register_io();
}

void Graph::deactivate() {
  if (!context)
    return;
  unregister_io();
  if (borrowed_context) {
    release_shared_graph(context);
    context = nullptr;
    graph = nullptr;
    return;
  }
  check(runtime.q.contextFree(context, nullptr), "contextFree");
  ++runtime.context_frees;
  --runtime.live_contexts;
  context = nullptr;
  graph = nullptr;
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
  // 缓存已注册句柄，避免 token 循环中重复 memRegister。
  const auto key = std::make_pair(buffer->data, FIELD(t, id));
  auto found = registered.find(key);
  Qnn_MemHandle_t handle;
  if (found == registered.end()) {
    Qnn_MemDescriptor_t descriptor = QNN_MEM_DESCRIPTOR_INIT;
    descriptor.memShape = {FIELD(t, rank), FIELD(t, dimensions), nullptr};
    descriptor.dataType = FIELD(t, dataType);
    descriptor.memType = QNN_MEM_TYPE_ION;
    descriptor.ionInfo.fd = buffer->fd;
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
  if (getenv("QNN_BASIC_ONCE") && profile_calls++ == 8 && !profile)
    check(
        runtime.q.profileCreate(
            runtime.backend, QNN_PROFILE_LEVEL_BASIC, &profile),
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
  if (getenv("QNN_SERIAL_CONTEXTS")) {
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
  for (auto* graph : graphs)
    check(
        runtime.q.graphRetrieve(
            graph->context, graph->graph_name.c_str(), &graph->graph),
        "graphRetrieve initialized context");
}
