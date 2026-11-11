// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#if defined(__hexagon__) || !defined(ANDROID)
#include "../kernels/Chunk.h"
#include <algorithm>
#include <cstring>
#include "../utils/QhpiUtils.h"
#include "HTP/core/qhpi.h"
#include "OpRegistration.h"
#if defined(__hexagon__) && QCOM_HTP_V81
#include "HAP_compute_res.h"
#endif

namespace {
bool valid(const QHPI_Shape* inputs, const QHPI_Shape* outputs) {
  const unsigned batch = inputs[0].dims[0], heads = inputs[0].dims[1];
  if (!batch || !heads || uint64_t(batch) * heads > 256)
    return false;
  for (unsigned i : {0u, 1u, 2u, 5u, 6u})
    if (!op_utils::hasShape(inputs[i], batch, heads, 128, 128))
      return false;
  for (unsigned i : {3u, 4u})
    if (!op_utils::hasShape(inputs[i], batch, heads, 1, 128))
      return false;
  return op_utils::hasShape(outputs[0], batch, heads, 128, 128) &&
      op_utils::hasShape(outputs[1], batch, heads, 128, 128);
}

const QHPI_Op* rewrite(const QHPI_Op* op) {
  QHPI_OpRef inputs[8];
  QHPI_Shape shapes[7], outputs[2];
  QHPI_OutputDef defs[2];
  for (unsigned i = 0; i < 7; ++i) {
    inputs[i] = qhpi_op_input(op, i);
    shapes[i] = qhpi_op_output(inputs[i].op, inputs[i].output_number).shape;
  }
  for (unsigned i = 0; i < 2; ++i) {
    defs[i] = qhpi_op_output(op, i);
    outputs[i] = defs[i].shape;
  }
  if (!valid(shapes, outputs))
    return nullptr;
  QHPI_OutputDef scratch{};
  scratch.type = QHPI_FLOAT16;
  scratch.shape = {4, {1, 1, 1, gdn_chunk::workspace_bytes / 2}};
  const auto* reserve = qhpi_op_create(
      op, "DeltaNetOpPackage::ChunkReserve", 7, inputs, 1, &scratch);
  inputs[7] = qhpi_op_reference(reserve, 0);
  return qhpi_op_create(
      op, "DeltaNetOpPackage::ChunkExecute", 8, inputs, 2, defs);
}

uint32_t reserve(
    QHPI_RuntimeHandle*,
    uint32_t,
    QHPI_Tensor** outputs,
    uint32_t,
    const QHPI_Tensor* const*) {
  std::memset(qhpi_tensor_raw_data(outputs[0]), 0, 2048);
  return QHPI_SUCCESS;
}

uint32_t execute(
    QHPI_RuntimeHandle* handle,
    uint32_t num_outputs,
    QHPI_Tensor** outputs,
    uint32_t num_inputs,
    const QHPI_Tensor* const* inputs) {
  if (num_inputs != 8 || num_outputs != 2)
    return QHPI_ERROR_FATAL;
  QHPI_Shape shapes[7], results[2];
  for (unsigned i = 0; i < 7; ++i)
    shapes[i] = qhpi_tensor_shape(inputs[i]);
  for (unsigned i = 0; i < 2; ++i)
    results[i] = qhpi_tensor_shape(outputs[i]);
  if (!valid(shapes, results) ||
      !op_utils::hasShape(
          qhpi_tensor_shape(inputs[7]),
          1,
          1,
          1,
          gdn_chunk::workspace_bytes / 2))
    return QHPI_UNSUPPORTED;
#if defined(__hexagon__) && QCOM_HTP_V81
  auto* scratch = static_cast<uint8_t*>(qhpi_tensor_raw_data(inputs[7]));
  if (reinterpret_cast<uintptr_t>(scratch) & 127)
    return QHPI_UNSUPPORTED;
  auto& queue = *reinterpret_cast<attention_hmx::Queue*>(scratch);
  const auto resource = qhpi_thread_resources(handle);
  if (resource == 0) {
    const auto context = qhpi_srm_context(handle);
    const bool locked = context && HAP_compute_res_hmx_lock(context) == 0;
    if (!locked)
      __atomic_store_n(&queue.error, 1u, __ATOMIC_RELEASE);
    const bool okay = gdn_chunk::serve(queue);
    if (locked)
      HAP_compute_res_hmx_unlock(context);
    return locked && okay ? QHPI_SUCCESS : QHPI_ERROR_FATAL;
  }
  if (resource != QHPI_RESOURCE_HVX)
    return QHPI_SUCCESS;
  const unsigned slice = qhpi_slice_number(handle);
  const unsigned available = qhpi_num_slices(handle);
  if (!available || available > attention_hmx::kMaxWorkers ||
      slice >= available) {
    __atomic_store_n(&queue.error, 1u, __ATOMIC_RELEASE);
    if (!available)
      __atomic_store_n(&queue.jobs[0].finished, 1u, __ATOMIC_RELEASE);
    __atomic_store_n(
        &queue.workers, available ? available : 1u, __ATOMIC_RELEASE);
    return QHPI_UNSUPPORTED;
  }
  const unsigned slices = std::min(available, gdn_chunk::workers);
  if (slice >= slices)
    return QHPI_SUCCESS;
  const uint16_t* data[7];
  for (unsigned i = 0; i < 7; ++i)
    data[i] = static_cast<const uint16_t*>(qhpi_tensor_raw_data(inputs[i]));
  const bool okay = gdn_chunk::run(
      data,
      static_cast<uint16_t*>(qhpi_tensor_raw_data(outputs[0])),
      static_cast<uint16_t*>(qhpi_tensor_raw_data(outputs[1])),
      scratch,
      shapes[0].dims[0] * shapes[0].dims[1],
      slice,
      slices,
      queue);
  return okay ? QHPI_SUCCESS : QHPI_ERROR_FATAL;
#else
  return QHPI_UNSUPPORTED;
#endif
}

#define F(LOC) \
  { QHPI_FLOAT16, QHPI_LAYOUT_FLAT_4, QHPI_STORAGE_DIRECT, LOC }
#define D F(QHPI_MEM_LOC_DDR_OR_TCM)
#define T F(QHPI_MEM_LOC_TCM_ONLY)
QHPI_Tensor_Signature_v1 inputs[] = {D, D, D, D, D, D, D};
QHPI_Tensor_Signature_v1 execute_inputs[] = {D, D, D, D, D, D, D, T};
QHPI_Tensor_Signature_v1 outputs[] = {D, D}, scratch[] = {T};
QHPI_Kernel_v1 kernels[] = {
    {.function_name = "DeltaNetOpPackage::ChunkReserve.v1",
     .function = reserve,
     .resources = QHPI_RESOURCE_HVX,
     .min_inputs = 7,
     .input_signature = inputs,
     .min_outputs = 1,
     .output_signature = scratch},
    {.function_name = "DeltaNetOpPackage::ChunkExecute.v1",
     .function = execute,
     .resources = QHPI_RESOURCE_EXCLUSIVE,
     .multithreaded = true,
     .min_inputs = 8,
     .input_signature = execute_inputs,
     .min_outputs = 2,
     .output_signature = outputs,
     .sync_block_size = 0}};
QHPI_OpInfo_v1 ops[] = {
    {.name = "DeltaNetOpPackage::Chunk", .early_rewrite = rewrite},
    {.name = "DeltaNetOpPackage::ChunkReserve",
     .num_kernels = 1,
     .kernels = kernels},
    {.name = "DeltaNetOpPackage::ChunkExecute",
     .num_kernels = 1,
     .kernels = kernels + 1}};
#undef T
#undef D
#undef F
} // namespace

void register_chunk_ops() {
  qhpi_register_ops_v1(sizeof(ops) / sizeof(ops[0]), ops, "DeltaNetOpPackage");
}
#else
void register_chunk_ops() {}
#endif
