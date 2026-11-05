//==============================================================================
// Shared HTP package interface; the legacy package name preserves PTE compatibility.
//==============================================================================

#include "HTP/QnnHtpCommon.h"
#include "QnnOpPackage.h"
#include "QnnSdkBuildId.h"

#include <array>
#include <string>

#define STRINGIZE_DETAIL(X) #X
#define STRINGIZE(X) STRINGIZE_DETAIL(X)
#define THIS_PKG_NAME_STR STRINGIZE(THIS_PKG_NAME)

static constexpr auto sg_packageName = THIS_PKG_NAME_STR;
static std::array<const char*, 10> sg_opNames{{
    "DeltaNetCore",
    "DeltaNetCoreStream",
    "DeltaNetCoreVtcm",
    "DeltaNetCoreFp16Accum",
    "DeltaNetCoreFp16State",
    "DeltaNetCoreValid",
    "DeltaNetCoreValidVtcm",
    "AttentionValid",
    "AttentionInitialize",
    "AttentionExecute",
}};

static Qnn_ApiVersion_t sg_sdkApiVersion = QNN_HTP_API_VERSION_INIT;
static QnnOpPackage_Info_t sg_packageInfo = QNN_OP_PACKAGE_INFO_INIT;
static QnnOpPackage_GlobalInfrastructure_t sg_globalInfra = nullptr;
static bool sg_packageInitialized = false;

static QnnLog_Callback_t sg_logCallback = nullptr;
static QnnLog_Level_t sg_maxLogLevel = (QnnLog_Level_t)0;
static bool sg_logInitialized = false;

Qnn_ErrorHandle_t DeltaNetOpPackageInit(
    QnnOpPackage_GlobalInfrastructure_t infrastructure) {
  if (sg_packageInitialized) {
    return QNN_OP_PACKAGE_ERROR_LIBRARY_ALREADY_INITIALIZED;
  }

  sg_globalInfra = infrastructure;
  sg_packageInitialized = true;
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t DeltaNetOpPackageGetInfo(const QnnOpPackage_Info_t** info) {
  if (!sg_packageInitialized) {
    return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
  }
  if (!info) {
    return QNN_OP_PACKAGE_ERROR_INVALID_INFO;
  }

  sg_packageInfo = QNN_OP_PACKAGE_INFO_INIT;
  sg_packageInfo.packageName = sg_packageName;
  sg_packageInfo.operationNames = sg_opNames.data();
  sg_packageInfo.numOperations = sg_opNames.size();
  sg_packageInfo.sdkBuildId = QNN_SDK_BUILD_ID;
  sg_packageInfo.sdkApiVersion = &sg_sdkApiVersion;

  *info = &sg_packageInfo;
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t DeltaNetOpPackageLogInitialize(
    QnnLog_Callback_t callback,
    QnnLog_Level_t maxLogLevel) {
  if (sg_logInitialized) {
    return QNN_OP_PACKAGE_ERROR_LIBRARY_ALREADY_INITIALIZED;
  }
  if (!callback) {
    return QNN_LOG_ERROR_INVALID_ARGUMENT;
  }
  if (maxLogLevel < QNN_LOG_LEVEL_ERROR) {
    return QNN_LOG_ERROR_INVALID_ARGUMENT;
  }
  sg_logCallback = callback;
  sg_maxLogLevel = maxLogLevel;
  sg_logInitialized = true;
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t DeltaNetOpPackageLogSetLevel(QnnLog_Level_t maxLogLevel) {
  if (maxLogLevel < QNN_LOG_LEVEL_ERROR) {
    return QNN_LOG_ERROR_INVALID_ARGUMENT;
  }
  sg_maxLogLevel = maxLogLevel;
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t DeltaNetOpPackageLogTerminate() {
  if (!sg_logInitialized) {
    return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
  }
  sg_logCallback = nullptr;
  sg_maxLogLevel = (QnnLog_Level_t)0;
  sg_logInitialized = false;
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t DeltaNetOpPackageValidateOpConfig(Qnn_OpConfig_t opConfig) {
  if (std::string(sg_packageName) != opConfig.v1.packageName) {
    return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
  }

  const std::string type(opConfig.v1.typeName);
  if (type == "AttentionValid" || type == "AttentionInitialize" ||
      type == "AttentionExecute") {
    const unsigned inputs = type == "AttentionExecute" ? 9 : 7;
    const unsigned outputs = type == "AttentionInitialize" ? 2 : 1;
    return opConfig.v1.numOfParams == 0 && opConfig.v1.numOfInputs == inputs &&
            opConfig.v1.numOfOutputs == outputs
        ? QNN_SUCCESS
        : QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
  }
  if (std::string(opConfig.v1.typeName) == "DeltaNetCoreValid" ||
      std::string(opConfig.v1.typeName) == "DeltaNetCoreValidVtcm") {
    if (opConfig.v1.numOfParams != 0 || opConfig.v1.numOfInputs != 7 ||
        opConfig.v1.numOfOutputs != 2) {
      return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    }
  } else if (std::string(opConfig.v1.typeName) == "DeltaNetCore" ||
      std::string(opConfig.v1.typeName) == "DeltaNetCoreStream" ||
      std::string(opConfig.v1.typeName) == "DeltaNetCoreVtcm" ||
      std::string(opConfig.v1.typeName) == "DeltaNetCoreFp16Accum" ||
      std::string(opConfig.v1.typeName) == "DeltaNetCoreFp16State") {
    if (opConfig.v1.numOfParams != 0 || opConfig.v1.numOfInputs != 6 ||
        opConfig.v1.numOfOutputs != 2) {
      return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    }
  } else {
    return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
  }

  return QNN_SUCCESS;
}

extern "C" void release_attention_cache_maps();

Qnn_ErrorHandle_t DeltaNetOpPackageTerminate() {
  if (!sg_packageInitialized) {
    return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
  }

  release_attention_cache_maps();
  sg_globalInfra = nullptr;
  sg_packageInitialized = false;
  return QNN_SUCCESS;
}

void register_deltanet_qhpi_ops();
void register_attention_qhpi_ops();

#ifdef __cplusplus
extern "C" {
#endif

Qnn_ErrorHandle_t DeltaNetOpPackageInterfaceProvider(
    QnnOpPackage_Interface_t* interface) {
  if (!interface) {
    return QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT;
  }
  interface->interfaceVersion = {1, 4, 0};
  interface->v1_4.init = DeltaNetOpPackageInit;
  interface->v1_4.terminate = DeltaNetOpPackageTerminate;
  interface->v1_4.getInfo = DeltaNetOpPackageGetInfo;
  interface->v1_4.validateOpConfig = DeltaNetOpPackageValidateOpConfig;
  interface->v1_4.createOpImpl = nullptr;
  interface->v1_4.freeOpImpl = nullptr;
  interface->v1_4.logInitialize = DeltaNetOpPackageLogInitialize;
  interface->v1_4.logSetLevel = DeltaNetOpPackageLogSetLevel;
  interface->v1_4.logTerminate = DeltaNetOpPackageLogTerminate;
  return QNN_SUCCESS;
}

const char* qhpi_init() {
  register_deltanet_qhpi_ops();
  register_attention_qhpi_ops();
  return THIS_PKG_NAME_STR;
}

#ifdef __cplusplus
}
#endif
