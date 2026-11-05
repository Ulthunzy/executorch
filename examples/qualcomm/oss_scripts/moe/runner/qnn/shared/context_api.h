#pragma once

#include "QnnInterface.h"
#include "System/QnnSystemContext.h"

// Graph views borrow metadata and a context until their matching release.
// Configure before the first graph lease; nullptr keeps vision out of the batch.
extern "C" int qwen35_shared_prepare_vision(const char* path);
extern "C" int qwen35_shared_acquire(
    const char* directory,
    const char* graph_name,
    Qnn_ContextHandle_t* context,
    Qnn_GraphHandle_t* graph,
    const QnnSystemContext_GraphInfoV1_t** metadata);
extern "C" void qwen35_shared_release(Qnn_ContextHandle_t context);
extern "C" int qwen35_shared_register(
    Qnn_ContextHandle_t context,
    const Qnn_MemDescriptor_t* descriptor,
    Qnn_MemHandle_t* handle);
extern "C" int qwen35_shared_unregister(
    Qnn_ContextHandle_t context,
    Qnn_MemHandle_t handle);
extern "C" uint64_t qwen35_shared_context_count();

extern "C" int qwen35_shared_set_power_active(int active);
