#pragma once

#include "context_api.h"
#include <string>

bool shared_graph_enabled(const std::string& path);
const QnnSystemContext_GraphInfoV1_t* borrow_shared_graph(
    const std::string& path,
    Qnn_ContextHandle_t* context,
    Qnn_GraphHandle_t* graph);
void release_shared_graph(Qnn_ContextHandle_t context);
int register_shared_io(
    Qnn_ContextHandle_t context,
    const Qnn_MemDescriptor_t* descriptor,
    Qnn_MemHandle_t* handle);
int unregister_shared_io(Qnn_ContextHandle_t context, Qnn_MemHandle_t handle);
uint64_t shared_context_count();
