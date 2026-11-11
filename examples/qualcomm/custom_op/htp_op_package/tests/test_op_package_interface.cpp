// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "OpRegistration.h"
#include "QnnOpPackage.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <initializer_list>

extern "C" Qnn_ErrorHandle_t DeltaNetOpPackageInterfaceProvider(
    QnnOpPackage_Interface_t* interface);
extern "C" const char* qhpi_init();

// Test the public package ABI without requiring the QHPI runtime.
static unsigned registrations[4]{};
static unsigned cache_releases = 0;
void register_chunk_ops() {
  ++registrations[0];
}
void register_chunk_state_qhpi_ops() {
  ++registrations[1];
}
void register_deltanet_qhpi_ops() {
  ++registrations[2];
}
void register_attention_qhpi_ops() {
  ++registrations[3];
}
extern "C" void release_attention_cache_maps() {
  ++cache_releases;
}

int main() {
  assert(std::strcmp(qhpi_init(), "DeltaNetOpPackage") == 0);
  for (auto count : registrations)
    assert(count == 1);
  QnnOpPackage_Interface_t interface{};
  assert(DeltaNetOpPackageInterfaceProvider(nullptr) != QNN_SUCCESS);
  assert(DeltaNetOpPackageInterfaceProvider(&interface) == QNN_SUCCESS);
  const auto& api = interface.v1_4;
  const QnnOpPackage_Info_t* info = nullptr;
  assert(api.getInfo(&info) == QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED);
  assert(api.init(nullptr) == QNN_SUCCESS);
  assert(api.getInfo(&info) == QNN_SUCCESS);
  assert(std::strcmp(info->packageName, "DeltaNetOpPackage") == 0);
  assert(info->numOperations == 19);
  for (uint32_t i = 0; i < info->numOperations; ++i) {
    assert(std::strstr(info->operationNames[i], "ChunkReadout") == nullptr);
    assert(std::strstr(info->operationNames[i], "QuantizedGemv") == nullptr);
    for (uint32_t j = 0; j < i; ++j)
      assert(std::strcmp(info->operationNames[i], info->operationNames[j]));
  }

  Qnn_OpConfig_t config = QNN_OPCONFIG_INIT;
  config.v1.packageName = "DeltaNetOpPackage";
  config.v1.typeName = "ChunkExecute";
  config.v1.numOfInputs = 8;
  config.v1.numOfOutputs = 2;
  assert(api.validateOpConfig(config) == QNN_SUCCESS);
  config.v1.numOfInputs = 7;
  assert(api.validateOpConfig(config) != QNN_SUCCESS);
  config.v1.numOfInputs = 8;
  config.v1.numOfOutputs = 1;
  assert(api.validateOpConfig(config) != QNN_SUCCESS);
  config.v1.typeName = "ChunkExecute";
  config.v1.numOfInputs = 8;
  config.v1.numOfOutputs = 2;
  assert(api.validateOpConfig(config) == QNN_SUCCESS);
  config.v1.typeName = "AttentionExecute";
  config.v1.numOfInputs = 9;
  config.v1.numOfOutputs = 1;
  assert(api.validateOpConfig(config) == QNN_SUCCESS);
  config.v1.typeName = "DeltaNetCore";
  config.v1.numOfInputs = 6;
  config.v1.numOfOutputs = 2;
  assert(api.validateOpConfig(config) == QNN_SUCCESS);
  config.v1.typeName = "DeltaNetCoreValidVtcm";
  config.v1.numOfInputs = 7;
  assert(api.validateOpConfig(config) == QNN_SUCCESS);
  for (const char* removed :
       {"ChunkRound2",
        "ChunkReadout",
        "QuantizedGemv",
        "GdnPreparedCached",
        "GdnTriangle",
        "GdnGatedNorm",
        "GdnWindowLookup",
        "GdnLosslessLookup",
        "Unknown"}) {
    config.v1.typeName = removed;
    assert(api.validateOpConfig(config) != QNN_SUCCESS);
  }
  config.v1.typeName = "ChunkExecute";
  config.v1.numOfInputs = 8;
  config.v1.numOfOutputs = 2;
  config.v1.numOfParams = 1;
  assert(api.validateOpConfig(config) != QNN_SUCCESS);
  config.v1.numOfParams = 0;
  config.v1.typeName = nullptr;
  assert(api.validateOpConfig(config) != QNN_SUCCESS);
  config.v1.typeName = "ChunkExecute";
  config.v1.packageName = nullptr;
  assert(api.validateOpConfig(config) != QNN_SUCCESS);
  config.v1.packageName = "OtherPackage";
  assert(api.validateOpConfig(config) != QNN_SUCCESS);
  config.v1.packageName = "DeltaNetOpPackage";
  config.version = static_cast<Qnn_OpConfigVersion_t>(0);
  assert(api.validateOpConfig(config) != QNN_SUCCESS);
  assert(api.terminate() == QNN_SUCCESS);
  assert(cache_releases == 1);
  assert(api.getInfo(&info) == QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED);
  std::puts(
      "PASS package lifecycle, public contracts, removed ops and invalid configs");
}
