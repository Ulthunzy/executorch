// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#if defined(__hexagon__) || !defined(ANDROID)
#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstring>
#include "../utils/DeltaNetConfig.h"
#include "../utils/QhpiUtils.h"
#include "HTP/core/qhpi.h"
#include "HvxUtils.h"
#include "OpRegistration.h"

#if defined(__hexagon__) && QCOM_HTP_V81
#include "HAP_compute_res.h"
#include "HmxDispatch.h"
static_assert(sizeof(attention_hmx::Queue) <= 2048);
extern "C" int32_t qhblas_hmx_transfer_row_ahf(__fp16*, uint8_t*, uint32_t);
extern "C" int32_t qhblas_hmx_matrix_matrix_mpy_hf_sm(
    uint8_t*,
    uint32_t,
    uint8_t*,
    uint32_t,
    uint8_t*,
    uint32_t,
    uint8_t*);
#endif

namespace chunk_state {
constexpr unsigned workers = 8;
struct Geometry {
  unsigned batch, heads, rows, columns, inner;
  unsigned b_bytes() const {
    return inner * columns * 2;
  }
  unsigned a_offset(unsigned batch = 1) const {
    return batch * b_bytes();
  }
  unsigned c_offset(unsigned batch = 1) const {
    return a_offset(batch) + batch * rows * inner * 2;
  }
  unsigned bias_offset(unsigned batch = 1) const {
    return c_offset(batch) + batch * rows * columns * 2;
  }
  unsigned worker_bytes(unsigned batch = 1) const {
    return (bias_offset(batch) + 256 + 2047) & ~2047u;
  }
};
static bool geometry(const QHPI_Shape* s, Geometry& g) {
  if (s[0].rank != 4 || s[1].rank != 4)
    return false;
  g = {s[0].dims[0], s[0].dims[1], s[0].dims[2], s[0].dims[3], s[1].dims[3]};
  return g.batch && g.heads && g.rows && g.rows <= 256 && !(g.rows % 32) &&
      g.columns && g.columns <= 256 && !(g.columns % 64) && g.inner &&
      g.inner <= 128 && !(g.inner % 32) &&
      uint64_t(g.batch) * g.heads * g.rows * g.columns <= INT32_MAX &&
      op_utils::hasShape(s[1], g.batch, g.heads, g.rows, g.inner) &&
      op_utils::hasShape(s[2], g.batch, g.heads, g.inner, g.columns) &&
      op_utils::hasShape(s[3], g.batch, g.heads, g.inner, 1) &&
      op_utils::hasShape(s[4], g.batch, g.heads, 1, 1);
}
template <unsigned Batch>
static const QHPI_Op* rewrite(const QHPI_Op* op) {
  QHPI_OpRef input[5];
  QHPI_Shape shapes[5];
  for (unsigned i = 0; i < 5; ++i) {
    input[i] = qhpi_op_input(op, i);
    shapes[i] = qhpi_op_output(input[i].op, input[i].output_number).shape;
  }
  Geometry g{};
  if (!geometry(shapes, g))
    return nullptr;
  auto output = qhpi_op_output(op, 0);
  if (!op_utils::hasShape(output.shape, g.batch, g.heads, g.rows, g.columns))
    return nullptr;
  QHPI_OutputDef scratch{};
  scratch.type = QHPI_FLOAT16;
  scratch.shape = {4, {1, 1, 1, (2048 + workers * g.worker_bytes(Batch)) / 2}};
  const auto* reserve = qhpi_op_create(
      op, "DeltaNetOpPackage::ChunkStateReserve", 5, input, 1, &scratch);
  QHPI_OpRef args[6] = {
      qhpi_op_reference(reserve, 0),
      input[0],
      input[1],
      input[2],
      input[3],
      input[4]};
  return qhpi_op_create(
      op,
      Batch == 1 ? "DeltaNetOpPackage::ChunkStateExecute"
                 : "DeltaNetOpPackage::ChunkStateExecuteBatch",
      6,
      args,
      1,
      &output);
}
static uint32_t reserve(
    QHPI_RuntimeHandle*,
    uint32_t,
    QHPI_Tensor** out,
    uint32_t,
    const QHPI_Tensor* const*) {
  std::memset(qhpi_tensor_raw_data(out[0]), 0, 2048);
  return QHPI_SUCCESS;
}
template <unsigned Batch>
static uint32_t execute(
    QHPI_RuntimeHandle* handle,
    uint32_t no,
    QHPI_Tensor** out,
    uint32_t ni,
    const QHPI_Tensor* const* in) {
  if (ni != 6 || no != 1)
    return QHPI_ERROR_FATAL;
  QHPI_Shape shapes[5];
  for (unsigned i = 0; i < 5; ++i)
    shapes[i] = qhpi_tensor_shape(in[i + 1]);
  Geometry g{};
  if (!geometry(shapes, g) ||
      !op_utils::hasShape(
          qhpi_tensor_shape(out[0]), g.batch, g.heads, g.rows, g.columns))
    return QHPI_UNSUPPORTED;
  const auto work_shape = qhpi_tensor_shape(in[0]);
  if (work_shape.rank != 4 ||
      work_shape.dims[3] * 2 < 2048 + workers * g.worker_bytes(Batch))
    return QHPI_UNSUPPORTED;
  const auto* state = static_cast<const Float16*>(qhpi_tensor_raw_data(in[1]));
  const auto* key = static_cast<const Float16*>(qhpi_tensor_raw_data(in[2]));
  const auto* correction =
      static_cast<const Float16*>(qhpi_tensor_raw_data(in[3]));
  const auto* weight = static_cast<const Float16*>(qhpi_tensor_raw_data(in[4]));
  const auto* decay = static_cast<const Float16*>(qhpi_tensor_raw_data(in[5]));
  auto* result = static_cast<Float16*>(qhpi_tensor_raw_data(out[0]));
#if defined(__hexagon__) && QCOM_HTP_V81
  auto& queue =
      *static_cast<attention_hmx::Queue*>(qhpi_tensor_raw_data(in[0]));
  const auto resource = qhpi_thread_resources(handle);
  if (resource == 0) {
    const auto context = qhpi_srm_context(handle);
    const bool locked = context && HAP_compute_res_hmx_lock(context) == 0;
    if (!locked)
      __atomic_store_n(&queue.error, 1u, __ATOMIC_RELEASE);
    const bool okay =
        attention_hmx::serve(queue, [](const attention_hmx::Job& job) {
          const auto matrix = job;
          for (unsigned batch = 0; batch < matrix.batches; ++batch)
            for (unsigned c = 0; c < matrix.n; c += 32)
              qhblas_hmx_matrix_matrix_mpy_hf_sm(
                  matrix.a + batch * matrix.a_stride,
                  ((matrix.k / 32) << 11) | ((matrix.m - 1) << 2),
                  matrix.b + (batch / matrix.panel) * matrix.b_stride +
                      c * matrix.k * 2,
                  ((matrix.k / 32) << 11) - 1,
                  matrix.c + batch * matrix.c_stride + c * 64,
                  0,
                  matrix.bias);
          return attention_hmx::Status::Done;
        });
    if (locked)
      HAP_compute_res_hmx_unlock(context);
    return locked && okay ? QHPI_SUCCESS : QHPI_ERROR_FATAL;
  }
  if (resource != QHPI_RESOURCE_HVX)
    return QHPI_SUCCESS;
  const unsigned slice = qhpi_slice_number(handle),
                 slices = qhpi_num_slices(handle);
  __atomic_store_n(&queue.workers, slices, __ATOMIC_RELEASE);
  if (!slices || slices > workers || slice >= slices) {
    __atomic_store_n(&queue.error, 1u, __ATOMIC_RELEASE);
    return QHPI_UNSUPPORTED;
  }
  auto* scratch = static_cast<uint8_t*>(qhpi_tensor_raw_data(in[0])) + 2048 +
      slice * g.worker_bytes(Batch);
  auto* packed_b = scratch;
  auto* packed_a = scratch + g.a_offset(Batch);
  auto* product = reinterpret_cast<uint16_t*>(scratch + g.c_offset(Batch));
  auto* bias = scratch + g.bias_offset(Batch);
  *reinterpret_cast<HVX_Vector*>(bias) = Q6_V_vsplat_R(0x3c00);
  *reinterpret_cast<HVX_Vector*>(bias + 128) = Q6_V_vzero();
  bool okay = true;
  for (unsigned first = slice; first < g.batch * g.heads;
       first += slices * Batch) {
    const unsigned count =
        std::min<unsigned>(Batch, (g.batch * g.heads - 1 - first) / slices + 1);
    for (unsigned part = 0; part < count; ++part) {
      const unsigned head = first + part * slices;
      for (unsigned t = 0; t < g.inner; t += 2) {
        const auto w0 = Q6_Vh_vsplat_R(weight[head * g.inner + t].raw());
        const auto w1 = Q6_Vh_vsplat_R(weight[head * g.inner + t + 1].raw());
        for (unsigned c = 0; c < g.columns; c += 64) {
          HVX_Vector a =
              vmemu(correction + (head * g.inner + t) * g.columns + c);
          HVX_Vector b =
              vmemu(correction + (head * g.inner + t + 1) * g.columns + c);
          a = Q6_Vhf_vmpy_VhfVhf(a, w0);
          b = Q6_Vhf_vmpy_VhfVhf(b, w1);
          const auto pair = Q6_W_vshuff_VVR(b, a, -2);
          *reinterpret_cast<HVX_Vector*>(
              packed_b + part * g.b_bytes() + (c / 32) * g.inner * 64 +
              (t / 2) * 128) = Q6_V_lo_W(pair);
          *reinterpret_cast<HVX_Vector*>(
              packed_b + part * g.b_bytes() + (c / 32 + 1) * g.inner * 64 +
              (t / 2) * 128) = Q6_V_hi_W(pair);
        }
      }
      for (unsigned row = 0; row < g.rows; row += 32)
        qhblas_hmx_transfer_row_ahf(
            (__fp16*)(key + (head * g.rows + row) * g.inner),
            packed_a + (part * g.rows + row) * g.inner * 2,
            g.inner);
    }
    if (!attention_hmx::submit(
            queue,
            slice,
            packed_a,
            packed_b,
            reinterpret_cast<uint8_t*>(product),
            bias,
            32,
            g.inner,
            g.columns,
            count * g.rows / 32,
            32 * g.inner * 2,
            32 * g.columns * 2,
            nullptr,
            0,
            g.b_bytes(),
            g.rows / 32)) {
      okay = false;
      break;
    }
    for (unsigned part = 0; part < count; ++part) {
      const unsigned head = first + part * slices;
      const auto d = Q6_Vh_vsplat_R(decay[head].raw());
      for (unsigned row = 0; row < g.rows; row += 32) {
        const auto* tile = product + (part * g.rows + row) * g.columns;
        for (unsigned r = 0; r < 32; r += 2)
          for (unsigned c = 0; c < g.columns; c += 64) {
            const auto a = *reinterpret_cast<const HVX_Vector*>(
                tile + (c / 32) * 1024 + (r / 2) * 64);
            const auto b = *reinterpret_cast<const HVX_Vector*>(
                tile + (c / 32 + 1) * 1024 + (r / 2) * 64);
            const auto pair = Q6_W_vdeal_VVR(b, a, -2);
            const unsigned at = (head * g.rows + row + r) * g.columns + c;
            // Preserve separate FP16 multiply/add rounding rather than FMA.
            const auto s0 = Q6_Vhf_vmpy_VhfVhf(vmemu(state + at), d);
            const auto s1 =
                Q6_Vhf_vmpy_VhfVhf(vmemu(state + at + g.columns), d);
            vmemu(result + at) = Q6_Vhf_vadd_VhfVhf(s0, Q6_V_lo_W(pair));
            vmemu(result + at + g.columns) =
                Q6_Vhf_vadd_VhfVhf(s1, Q6_V_hi_W(pair));
          }
      }
    }
    if (!okay)
      break;
  }
  __atomic_store_n(&queue.jobs[slice].finished, 1u, __ATOMIC_RELEASE);
  return okay ? QHPI_SUCCESS : QHPI_ERROR_FATAL;
#else
  if (qhpi_thread_resources(handle) != QHPI_RESOURCE_HVX)
    return QHPI_SUCCESS;
  const unsigned slices = std::max<uint32_t>(1, qhpi_num_slices(handle)),
                 slice = qhpi_slice_number(handle);
  for (unsigned head = slice; head < g.batch * g.heads; head += slices)
    for (unsigned r = 0; r < g.rows; ++r)
      for (unsigned c = 0; c < g.columns; ++c) {
        float sum = 0;
        for (unsigned t = 0; t < g.inner; ++t) {
          const auto weighted = Float16(
              float(correction[(head * g.inner + t) * g.columns + c]) *
              float(weight[head * g.inner + t]));
          sum +=
              float(key[(head * g.rows + r) * g.inner + t]) * float(weighted);
        }
        const auto old = Float16(
            float(state[(head * g.rows + r) * g.columns + c]) *
            float(decay[head]));
        result[(head * g.rows + r) * g.columns + c] =
            Float16(float(old) + float(Float16(sum)));
      }
  return QHPI_SUCCESS;
#endif
}
#define F(LOC) \
  { QHPI_FLOAT16, QHPI_LAYOUT_FLAT_4, QHPI_STORAGE_DIRECT, LOC }
static QHPI_Tensor_Signature_v1 inputs[] = {
    F(QHPI_MEM_LOC_DDR_OR_TCM),
    F(QHPI_MEM_LOC_DDR_OR_TCM),
    F(QHPI_MEM_LOC_DDR_OR_TCM),
    F(QHPI_MEM_LOC_DDR_OR_TCM),
    F(QHPI_MEM_LOC_DDR_OR_TCM)};
static QHPI_Tensor_Signature_v1 scratch_output[] = {F(QHPI_MEM_LOC_TCM_ONLY)};
static QHPI_Tensor_Signature_v1 exec_inputs[] = {
    F(QHPI_MEM_LOC_TCM_ONLY),
    F(QHPI_MEM_LOC_DDR_OR_TCM),
    F(QHPI_MEM_LOC_DDR_OR_TCM),
    F(QHPI_MEM_LOC_DDR_OR_TCM),
    F(QHPI_MEM_LOC_DDR_OR_TCM),
    F(QHPI_MEM_LOC_DDR_OR_TCM)};
static QHPI_Tensor_Signature_v1 outputs[] = {F(QHPI_MEM_LOC_DDR_OR_TCM)};
static QHPI_Kernel_v1 reserve_kernel[] = {
    {.function_name = "DeltaNetOpPackage::ChunkStateReserve.hvx",
     .function = reserve,
     .resources = QHPI_RESOURCE_HVX,
     .source_destructive = false,
     .multithreaded = false,
     .min_inputs = 5,
     .input_signature = inputs,
     .min_outputs = 1,
     .output_signature = scratch_output}};
static QHPI_Kernel_v1 exec_kernel[] = {
    {.function_name = "DeltaNetOpPackage::ChunkStateExecute.hmx",
     .function = execute<1>,
     .resources = QHPI_RESOURCE_EXCLUSIVE,
     .source_destructive = false,
     .multithreaded = true,
     .min_inputs = 6,
     .input_signature = exec_inputs,
     .min_outputs = 1,
     .output_signature = outputs,
     .sync_block_size = 0}};
static QHPI_Kernel_v1 batch_kernel[] = {
    {.function_name = "DeltaNetOpPackage::ChunkStateExecuteBatch.hmx",
     .function = execute<2>,
     .resources = QHPI_RESOURCE_EXCLUSIVE,
     .multithreaded = true,
     .min_inputs = 6,
     .input_signature = exec_inputs,
     .min_outputs = 1,
     .output_signature = outputs}};
static QHPI_OpInfo_v1 ops[] = {
    {.name = "DeltaNetOpPackage::ChunkStateUpdateBatch",
     .early_rewrite = rewrite<2>},
    {.name = "DeltaNetOpPackage::ChunkStateExecuteBatch",
     .num_kernels = 1,
     .kernels = batch_kernel},
    {.name = "DeltaNetOpPackage::ChunkStateUpdate",
     .early_rewrite = rewrite<1>},
    {.name = "DeltaNetOpPackage::ChunkStateReserve",
     .num_kernels = 1,
     .kernels = reserve_kernel},
    {.name = "DeltaNetOpPackage::ChunkStateExecute",
     .num_kernels = 1,
     .kernels = exec_kernel}};
} // namespace chunk_state
void register_chunk_state_qhpi_ops() {
  qhpi_register_ops_v1(
      sizeof(chunk_state::ops) / sizeof(chunk_state::ops[0]),
      chunk_state::ops,
      "DeltaNetOpPackage");
}
#else
void register_chunk_state_qhpi_ops() {}
#endif
