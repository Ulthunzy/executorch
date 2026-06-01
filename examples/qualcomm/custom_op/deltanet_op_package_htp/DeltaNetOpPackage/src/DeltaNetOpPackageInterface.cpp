//==============================================================================
// Auto Generated Code for DeltaNetOpPackage
//==============================================================================

#include "HTP/QnnHtpCommon.h"
#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"
#include "HTP/core/unique_types.h"
#include "QnnOpPackage.h"
#include "QnnSdkBuildId.h"

#include <array>
#include <string>

DEFINE_UNIQ_TY()
BEGIN_PKG_OPS_OPTS_LIST()

DECLARE_PKG_OPS_OPTS_LIST(PKG_DeltaNet)

END_PKG_OPS_OPTS_LIST()

static constexpr auto sg_packageName = THIS_PKG_NAME_STR;
static std::array<const char*, 7> sg_opNames{{"DeltaNet",
                                              "DeltaNetPacked",
                                              "DeltaNetCore",
                                              "DeltaNetCoreFp16Accum",
                                              "DeltaNetIdentity",
                                              "DeltaNetDecayState",
                                              "DeltaNetStatePassthrough"}};

static Qnn_ApiVersion_t sg_sdkApiVersion = QNN_HTP_API_VERSION_INIT;
static QnnOpPackage_Info_t sg_packageInfo = QNN_OP_PACKAGE_INFO_INIT;
static QnnOpPackage_GlobalInfrastructure_t sg_globalInfra = nullptr;
static bool sg_packageInitialized = false;

static QnnLog_Callback_t sg_logCallback = nullptr;
static QnnLog_Level_t sg_maxLogLevel = (QnnLog_Level_t)0;
static bool sg_logInitialized = false;

INIT_PACKAGE_OP_DEF()
INIT_PACKAGE_OPTIMIZATION_DEF()
INIT_PACKAGE_PARAM_ORDER_DEF()
INIT_PKG_CORE_INIT_FUNC()

Qnn_ErrorHandle_t DeltaNetOpPackageInit(
    QnnOpPackage_GlobalInfrastructure_t infrastructure) {
  if (sg_packageInitialized) {
    return QNN_OP_PACKAGE_ERROR_LIBRARY_ALREADY_INITIALIZED;
  }

  REGISTER_PACKAGE_PARAM_ORDERS()
  REGISTER_PACKAGE_AXIS_PARAMS()
  REGISTER_PACKAGE_PER_CHANNEL_QUANTIZED_OPS()

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

  if (std::string(opConfig.v1.typeName) == "DeltaNet") {
    if (opConfig.v1.numOfParams != 2 || opConfig.v1.numOfInputs != 6 ||
        opConfig.v1.numOfOutputs != 2) {
      return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    }
  } else if (std::string(opConfig.v1.typeName) == "DeltaNetPacked") {
    if (opConfig.v1.numOfParams != 0 || opConfig.v1.numOfInputs != 6 ||
        opConfig.v1.numOfOutputs != 1) {
      return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    }
  } else if (std::string(opConfig.v1.typeName) == "DeltaNetCore") {
    if (opConfig.v1.numOfParams != 0 || opConfig.v1.numOfInputs != 6 ||
        opConfig.v1.numOfOutputs != 2) {
      return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    }
  } else if (std::string(opConfig.v1.typeName) == "DeltaNetCoreFp16Accum") {
    if (opConfig.v1.numOfParams != 0 || opConfig.v1.numOfInputs != 6 ||
        opConfig.v1.numOfOutputs != 2) {
      return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    }
  } else if (std::string(opConfig.v1.typeName) == "DeltaNetIdentity") {
    if (opConfig.v1.numOfParams != 0 || opConfig.v1.numOfInputs != 1 ||
        opConfig.v1.numOfOutputs != 1) {
      return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    }
  } else if (std::string(opConfig.v1.typeName) == "DeltaNetDecayState") {
    if (opConfig.v1.numOfParams != 0 || opConfig.v1.numOfInputs != 2 ||
        opConfig.v1.numOfOutputs != 1) {
      return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    }
  } else if (
      std::string(opConfig.v1.typeName) == "DeltaNetStatePassthrough") {
    if (opConfig.v1.numOfParams != 0 || opConfig.v1.numOfInputs != 2 ||
        opConfig.v1.numOfOutputs != 1) {
      return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    }
  } else {
    return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
  }

  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t DeltaNetOpPackageTerminate() {
  if (!sg_packageInitialized) {
    return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
  }

  sg_globalInfra = nullptr;
  sg_packageInitialized = false;
  return QNN_SUCCESS;
}

void register_deltanet_qhpi_ops();

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
  return THIS_PKG_NAME_STR;
}

#ifdef __cplusplus
}
#endif
