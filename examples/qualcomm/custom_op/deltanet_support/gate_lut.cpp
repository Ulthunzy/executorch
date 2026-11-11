// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <vector>
#include "HTP/QnnHtpDevice.h"
#include "QnnInterface.h"
#include "System/QnnSystemInterface.h"

#define FIELD(t, f) ((t).version == QNN_TENSOR_VERSION_1 ? (t).v1.f : (t).v2.f)

namespace {
void check(Qnn_ErrorHandle_t status, const char* operation) {
  if (status != QNN_SUCCESS)
    throw std::runtime_error(
        std::string(operation) + ": " + std::to_string(status));
}

struct Session {
  QNN_INTERFACE_VER_TYPE q{};
  QNN_SYSTEM_INTERFACE_VER_TYPE s{};
  Qnn_BackendHandle_t backend = nullptr;
  Qnn_DeviceHandle_t device = nullptr;
  Qnn_ContextHandle_t context = nullptr;
  QnnSystemContext_Handle_t system = nullptr;
  QnnHtpDevice_PerfInfrastructure_t perf{};
  uint32_t power_id = 0;
  bool power_active = false;

  ~Session() {
    if (context)
      q.contextFree(context, nullptr);
    if (system)
      s.systemContextFree(system);
    if (power_active)
      perf.destroyPowerConfigId(power_id);
    if (device)
      q.deviceFree(device);
    if (backend)
      q.backendFree(backend);
  }

  void generate(const char* path, const char* output) {
    const QnnInterface_t** providers = nullptr;
    uint32_t count = 0;
    check(QnnInterface_getProviders(&providers, &count), "getProviders");
    for (uint32_t i = 0; i < count; ++i)
      if (providers[i]->apiVersion.coreApiVersion.major ==
              QNN_API_VERSION_MAJOR &&
          providers[i]->apiVersion.coreApiVersion.minor >=
              QNN_API_VERSION_MINOR) {
        q = providers[i]->QNN_INTERFACE_VER_NAME;
        break;
      }
    const QnnSystemInterface_t** system_providers = nullptr;
    check(
        QnnSystemInterface_getProviders(&system_providers, &count),
        "systemProviders");
    for (uint32_t i = 0; i < count; ++i)
      if (system_providers[i]->systemApiVersion.major ==
              QNN_SYSTEM_API_VERSION_MAJOR &&
          system_providers[i]->systemApiVersion.minor >=
              QNN_SYSTEM_API_VERSION_MINOR) {
        s = system_providers[i]->QNN_SYSTEM_INTERFACE_VER_NAME;
        break;
      }
    if (!q.backendCreate || !s.systemContextCreate)
      throw std::runtime_error("Incompatible QNN SDK interfaces");
    check(q.backendCreate(nullptr, nullptr, &backend), "backendCreate");
    check(q.deviceCreate(nullptr, nullptr, &device), "deviceCreate");
    QnnDevice_Infrastructure_t infrastructure = nullptr;
    check(
        q.deviceGetInfrastructure(&infrastructure), "deviceGetInfrastructure");
    if (!infrastructure)
      throw std::runtime_error("Missing HTP infrastructure");
    perf =
        static_cast<QnnHtpDevice_Infrastructure_t*>(infrastructure)->perfInfra;
    check(perf.createPowerConfigId(0, 0, &power_id), "createPowerConfigId");
    power_active = true;
    QnnHtpPerfInfrastructure_PowerConfig_t dcvs{}, latency{}, poll{};
    dcvs.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
    auto& config = dcvs.dcvsV3Config;
    config.contextId = power_id;
    config.setDcvsEnable = config.setSleepDisable = config.sleepDisable = 1;
    config.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
    config.setBusParams = config.setCoreParams = 1;
    config.busVoltageCornerMin = config.busVoltageCornerTarget =
        config.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    config.coreVoltageCornerMin = config.coreVoltageCornerTarget =
        config.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    latency.option =
        QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY;
    latency.rpcControlLatencyConfig = 100;
    poll.option =
        QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME;
    poll.rpcPollingTimeConfig = 9999;
    const QnnHtpPerfInfrastructure_PowerConfig_t* power[] = {
        &dcvs, &latency, &poll, nullptr};
    check(perf.setPowerConfig(power_id, power), "setPowerConfig");
    check(s.systemContextCreate(&system), "systemContextCreate");
    std::ifstream file(path, std::ios::binary);
    if (!file)
      throw std::runtime_error("Cannot read gate context");
    std::vector<char> binary(
        (std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>());
    const QnnSystemContext_BinaryInfo_t* info = nullptr;
    Qnn_ContextBinarySize_t size = 0;
    check(
        s.systemContextGetBinaryInfo(
            system, binary.data(), binary.size(), &info, &size),
        "getBinaryInfo");
    QnnSystemContext_GraphInfo_t* graphs = nullptr;
    uint32_t graph_count = 0;
    switch (info->version) {
      case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1:
        graphs = info->contextBinaryInfoV1.graphs;
        graph_count = info->contextBinaryInfoV1.numGraphs;
        break;
      case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2:
        graphs = info->contextBinaryInfoV2.graphs;
        graph_count = info->contextBinaryInfoV2.numGraphs;
        break;
      case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3:
        graphs = info->contextBinaryInfoV3.graphs;
        graph_count = info->contextBinaryInfoV3.numGraphs;
        break;
      default:
        throw std::runtime_error("Unsupported context metadata version");
    }
    if (graph_count != 1)
      throw std::runtime_error("Expected one gate graph");
    const char* name = nullptr;
    Qnn_Tensor_t input{}, result{};
    auto select = [&](const auto& graph) {
      if (graph.numGraphInputs != 1 || graph.numGraphOutputs != 1)
        throw std::runtime_error("Expected one gate input/output");
      name = graph.graphName;
      input = graph.graphInputs[0];
      result = graph.graphOutputs[0];
    };
    switch (graphs[0].version) {
      case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1:
        select(graphs[0].graphInfoV1);
        break;
      case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2:
        select(graphs[0].graphInfoV2);
        break;
      case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3:
        select(graphs[0].graphInfoV3);
        break;
      default:
        throw std::runtime_error("Unsupported graph metadata version");
    }
    auto elements = [](const Qnn_Tensor_t& tensor) {
      size_t n = 1;
      for (uint32_t i = 0; i < FIELD(tensor, rank); ++i)
        n *= FIELD(tensor, dimensions)[i];
      return n;
    };
    const size_t heads = elements(input);
    if (!heads || heads > 256 || elements(result) != heads ||
        FIELD(input, dataType) != QNN_DATATYPE_UFIXED_POINT_16 ||
        FIELD(result, dataType) != QNN_DATATYPE_UFIXED_POINT_16)
      throw std::runtime_error("Expected AR1 U16 gate IO");
    std::vector<uint16_t> codes(heads), values(heads), table(heads * 65536);
    FIELD(input, memType) = FIELD(result, memType) = QNN_TENSORMEMTYPE_RAW;
    FIELD(input, clientBuf).data = codes.data();
    FIELD(input, clientBuf).dataSize = heads * 2;
    FIELD(result, clientBuf).data = values.data();
    FIELD(result, clientBuf).dataSize = heads * 2;
    check(
        q.contextCreateFromBinary(
            backend,
            device,
            nullptr,
            binary.data(),
            binary.size(),
            &context,
            nullptr),
        "contextCreateFromBinary");
    Qnn_GraphHandle_t graph = nullptr;
    check(q.graphRetrieve(context, name, &graph), "graphRetrieve");
    // Preserve AR1 geometry: a larger batch can select different SDK rounding.
    for (uint32_t code = 0; code < 65536; ++code) {
      std::fill(codes.begin(), codes.end(), uint16_t(code));
      check(
          q.graphExecute(graph, &input, 1, &result, 1, nullptr, nullptr),
          "graphExecute");
      for (size_t head = 0; head < heads; ++head)
        table[head * 65536 + code] = values[head];
      if ((code + 1) % 8192 == 0)
        std::fprintf(stderr, "AR1 gate codes: %u/65536\n", code + 1);
    }
    std::ofstream target(output, std::ios::binary);
    target.write(reinterpret_cast<const char*>(table.data()), table.size() * 2);
    if (!target)
      throw std::runtime_error("Cannot write gate table");
    std::printf("Generated %zu SDK AR1 gate codes\n", table.size());
  }
};
} // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "Usage: gate_lut CONTEXT OUTPUT\n");
    return 2;
  }
  try {
    Session session;
    session.generate(argv[1], argv[2]);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
