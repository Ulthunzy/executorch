#include "../shared/context_store.cpp"
#include <cassert>
#include <iostream>

static QnnSystemContext_BinaryInfo_t fake_metadata{};
static QnnSystemContext_GraphInfo_t fake_graph{};
static int freed_contexts = 0, metadata_calls = 0;
static bool fail_metadata = false, fail_async = false, fail_graph = false;
static uint64_t io_limit_mb = 0;
static uint32_t activation_elements = 20 * 1048576 / sizeof(float);
static Qnn_Tensor_t activation = QNN_TENSOR_INIT;

void check(Qnn_ErrorHandle_t status, const char* action) {
  if (status)
    throw std::runtime_error(action);
}

size_t Graph::size(const Qnn_Tensor_t& tensor) {
  return FIELD(tensor, dimensions)[0] * sizeof(float);
}

Runtime::Runtime() {
  fake_metadata.version = QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3;
  fake_metadata.contextBinaryInfoV3.numGraphs = 1;
  fake_metadata.contextBinaryInfoV3.graphs = &fake_graph;
  fake_graph.graphInfoV1.graphName = "layer0_prefill_attention";
  FIELD(activation, dataType) = QNN_DATATYPE_FLOAT_32;
  FIELD(activation, rank) = 1;
  FIELD(activation, dimensions) = &activation_elements;
  fake_graph.graphInfoV1.numGraphInputs = 1;
  fake_graph.graphInfoV1.graphInputs = &activation;
  s.systemContextCreate = [](auto* value) -> Qnn_ErrorHandle_t {
    *value = reinterpret_cast<QnnSystemContext_Handle_t>(1);
    return 0;
  };
  s.systemContextFree = [](auto) -> Qnn_ErrorHandle_t {
    return 0;
  };
  s.systemContextGetBinaryInfo =
      [](auto, auto, auto, auto** result, auto* length) -> Qnn_ErrorHandle_t {
    if (fail_metadata && ++metadata_calls == 3)
      return 1;
    *result = &fake_metadata;
    *length = sizeof(fake_metadata);
    return 0;
  };
  perf.setMemoryConfig = [](auto, auto, auto) -> Qnn_ErrorHandle_t {
    return 0;
  };
  q.backendRegisterOpPackage = [](auto, auto, auto, auto) -> Qnn_ErrorHandle_t {
    return 0;
  };
  q.contextCreateFromBinaryListAsync =
      [](auto, auto, auto list, auto configs, auto) -> Qnn_ErrorHandle_t {
    for (int index = 0; configs[index]; ++index) {
      auto* option = static_cast<const QnnHtpContext_CustomConfig_t*>(
          configs[index]->customConfig);
      if (option->option == QNN_HTP_CONTEXT_CONFIG_OPTION_REUSED_IO_LIMIT)
        io_limit_mb = option->reusedIoLimitMb;
    }
    for (int index = 0; list[index]; ++index) {
      const auto& item = *list[index]->v2;
      item.notifyFunc(
          reinterpret_cast<Qnn_ContextHandle_t>(uintptr_t(index + 1)),
          nullptr,
          nullptr,
          QNN_CONTEXT_NOTIFY_TYPE_CONTEXT_INIT,
          item.notifyParam,
          fail_async && index == 3 ? 1 : 0);
    }
    return 0;
  };
  q.graphRetrieve = [](auto context, auto, auto* graph) -> Qnn_ErrorHandle_t {
    if (fail_graph)
      return 1;
    *graph = reinterpret_cast<Qnn_GraphHandle_t>(context);
    return 0;
  };
  q.contextFree = [](auto, auto) -> Qnn_ErrorHandle_t {
    ++freed_contexts;
    return 0;
  };
}

Runtime::~Runtime() = default;

WeightSource::WeightSource(Runtime& runtime, const std::string&)
    : runtime_(runtime) {}

WeightSource::~WeightSource() = default;

const void* WeightSource::image() const {
  return this;
}

uint64_t WeightSource::image_size() const {
  return 4096;
}

const Qnn_ContextBinaryCallback_t* WeightSource::callback() const {
  return nullptr;
}

size_t WeightSource::live_allocations() const {
  return 0;
}

Buffer::~Buffer() = default;

int main() {
  setenv("SHARED_OP_PACKAGE", "fake", 1);
  Qnn_ContextHandle_t context{}, second{}, third{};
  Qnn_GraphHandle_t graph{};
  const QnnSystemContext_GraphInfoV1_t* metadata{};
  auto acquire = [&](const char* group, Qnn_ContextHandle_t* out) {
    return qwen35_shared_acquire(
        group, "layer0_prefill_attention", out, &graph, &metadata);
  };
  assert(acquire("/model/group0", &context) == 0);
  assert(io_limit_mb >= 11 * 20 && io_limit_mb % 16 == 0);
  assert(qwen35_shared_context_count() == 11);
  qwen35_shared_release(context);
  assert(qwen35_shared_context_count() == 0 && freed_contexts == 11);
  assert(acquire("/model/group0", &context) == 0);
  assert(acquire("/model/group1", &second) == 0);
  qwen35_shared_release(context);
  assert(qwen35_shared_context_count() == 11);
  assert(acquire("/model/group0", &third) == 0 && third == context);
  qwen35_shared_release(second);
  qwen35_shared_release(third);
  assert(qwen35_shared_context_count() == 0 && freed_contexts == 22);
  for (auto* fault : {&fail_metadata, &fail_async, &fail_graph}) {
    *fault = true;
    metadata_calls = 0;
    assert(acquire("/model/group0", &context) != 0);
    assert(qwen35_shared_context_count() == 0);
    *fault = false;
    assert(acquire("/model/group0", &context) == 0);
    qwen35_shared_release(context);
    assert(qwen35_shared_context_count() == 0);
  }
  assert(acquire("/model/group0", &context) == 0);
  assert(acquire("/different/group0", &third) != 0);
  assert(qwen35_shared_context_count() == 11);
  qwen35_shared_release(context);
  assert(qwen35_shared_context_count() == 0);
  std::cout << "context rollback/retry tests passed\n";
}
