// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#if defined(__hexagon__) || !defined(ANDROID)
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <new>
#include "../../../../htp_arch.h"
#include "HTP/core/qhpi.h"
#include "attention/AttentionConfig.h"
#include "attention/Reference.h"
#ifdef __hexagon__
#include "HvxUtils.h"
#include "HAP_compute_res.h"
#include "HAP_mem.h"
#include "attention/HmxApi.h"
#include "attention/HmxQueue.h"
static_assert(sizeof(attention_hmx::Queue) <= 1536);
#include "qurt_memory.h"
#include "qurt_pimutex.h"
#endif
namespace attention {
struct InlineSync {
  uint32_t prepared, ready, departed, status, expected;
};
constexpr unsigned workers = max_workers, rows = 32, stride = fast_kv_rows;
[[maybe_unused]] constexpr unsigned tile = 256;
constexpr unsigned per_worker = 262144, header = 2048, worker_start = header;
[[maybe_unused]] constexpr unsigned kv_rows = fast_kv_rows;
constexpr unsigned score_bytes = rows * stride * 2;
static bool shape_is(
    const QHPI_Shape& shape,
    unsigned a,
    unsigned b,
    unsigned c,
    unsigned d) {
  return shape.rank == 4 && shape.dims[0] == a && shape.dims[1] == b &&
      shape.dims[2] == c && shape.dims[3] == d;
}
static bool geometry(
    const QHPI_Shape& q,
    const QHPI_Shape& nk,
    const QHPI_Shape& nv,
    Geometry& g) {
  if (q.rank != 4 || nk.rank != 4 || nv.rank != 4)
    return false;
  g = {q.dims[0], q.dims[1], q.dims[2], q.dims[3], nk.dims[1]};
  return g.valid() &&
      shape_is(nk, g.batch, g.kv_heads, g.dimension, g.queries) &&
      shape_is(nv, g.batch, g.kv_heads, g.queries, g.dimension);
}
static bool
configuration(uint32_t n, const QHPI_Tensor* const* in, Configuration& c) {
  if (n != 7 ||
      !geometry(
          qhpi_tensor_shape(in[0]),
          qhpi_tensor_shape(in[2]),
          qhpi_tensor_shape(in[4]),
          c.geometry) ||
      !shape_is(qhpi_tensor_shape(in[1]), 1, 1, 1, 3) ||
      !shape_is(qhpi_tensor_shape(in[3]), 1, 1, 1, 3) ||
      !shape_is(qhpi_tensor_shape(in[5]), 1, 1, 1, 3) ||
      !shape_is(qhpi_tensor_shape(in[6]), 1, 1, 1, 13))
    return false;
  for (unsigned i : {0u, 2u, 4u}) {
    const auto q = qhpi_tensor_quant_parameters(in[i]);
    if (!Quantization{q.stepsize, q.zero_offset}.valid())
      return false;
  }
  return c.configure(
      static_cast<const int32_t*>(qhpi_tensor_raw_data(in[1])),
      static_cast<const int32_t*>(qhpi_tensor_raw_data(in[3])),
      static_cast<const int32_t*>(qhpi_tensor_raw_data(in[5])),
      static_cast<const float*>(qhpi_tensor_raw_data(in[6])));
}
static QHPI_OutputDef scratch(unsigned bytes) {
  QHPI_OutputDef d{};
  d.type = QHPI_FLOAT16;
  d.shape = {4, {1, 1, 1, bytes / 2}};
  return d;
}
static const QHPI_Op* rewrite(const QHPI_Op* op) {
  QHPI_OpRef inputs[7];
  for (unsigned i = 0; i < 7; ++i)
    inputs[i] = qhpi_op_input(op, i);
  auto shape = [&](unsigned i) {
    return qhpi_op_output(inputs[i].op, inputs[i].output_number).shape;
  };
  Geometry g{};
  if (!geometry(shape(0), shape(2), shape(4), g) ||
      !shape_is(shape(1), 1, 1, 1, 3) || !shape_is(shape(3), 1, 1, 1, 3) ||
      !shape_is(shape(5), 1, 1, 1, 3) || !shape_is(shape(6), 1, 1, 1, 13))
    return nullptr;
  const auto y = qhpi_op_output(op, 0);
  if (!shape_is(y.shape, g.batch, g.heads, g.queries, g.dimension))
    return nullptr;
  const bool decode = g.queries == 1, fast = g.fast();
  const unsigned work_bytes = fast
      ? worker_start + (decode ? g.kv_heads : workers) * per_worker
      : workers * ((g.dimension + 256 + 31) & ~31u) * sizeof(float);
  QHPI_OutputDef storage[2] = {
      scratch(work_bytes),
      scratch(
          fast ? (decode ? g.kv_heads * 8 * stride * 2 : workers * score_bytes)
               : 128)};
  const auto* init = qhpi_op_create(
      op, "DeltaNetOpPackage::AttentionInitialize", 7, inputs, 2, storage);
  QHPI_OpRef args[9] = {
      qhpi_op_reference(init, 0),
      inputs[0],
      inputs[1],
      inputs[2],
      inputs[3],
      inputs[4],
      inputs[5],
      inputs[6],
      qhpi_op_reference(init, 1)};
  return qhpi_op_create(
      op, "DeltaNetOpPackage::AttentionExecute", 9, args, 1, &y);
}
static uint32_t initialize(
    QHPI_RuntimeHandle*,
    uint32_t no,
    QHPI_Tensor** out,
    uint32_t ni,
    const QHPI_Tensor* const* in);
#ifdef __hexagon__
struct CacheMappings {
  struct Slot {
    int fd = -1;
    unsigned bytes = 0, packed_bytes = 0;
    uint16_t* p = nullptr;
    uint8_t* packed = nullptr;
    Slot* next = nullptr;
    unsigned history = 0, last_valid = 0;
    bool last_exact = false, short_dense = false, mapped_by_get = false;
    uint32_t epoch = 0, coherent_epoch = 0;
    unsigned coherent_history = 0;
    alignas(128) uint16_t pending_k[2048]{}, pending_v[2048]{};
    alignas(128) uint32_t dma_words[8][32]{};
    alignas(16) InlineSync inline_sync{};
  };
  qurt_mutex_t lock;
  Slot* entries = nullptr;
  CacheMappings() {
    qurt_pimutex_init(&lock);
  }
};
static CacheMappings& cache_mappings() {
  static CacheMappings maps;
  return maps;
}
static uint16_t* cache_pointer(const int32_t* descriptor) {
  auto& maps = cache_mappings();
  const unsigned bytes =
      ((uint32_t(descriptor[1]) & logical_size_mask) + 4095u) & ~4095u;
  for (auto* slot = __atomic_load_n(&maps.entries, __ATOMIC_ACQUIRE); slot;
       slot = slot->next)
    if (slot->fd == descriptor[0])
      return slot->bytes >= bytes ? slot->p : nullptr;
  qurt_pimutex_lock(&maps.lock);
  uint16_t* pointer = nullptr;
  bool found = false;
  for (auto* slot = maps.entries; slot; slot = slot->next) {
    if (slot->fd == descriptor[0]) {
      pointer = slot->bytes >= bytes ? slot->p : nullptr;
      found = true;
      break;
    }
  }
  if (!found) {
    void* raw = nullptr;
    uint64 physical = 0;
    const bool by_get = HAP_mmap_get(descriptor[0], &raw, &physical) == 0;
    if (!by_get)
      raw = HAP_mmap2(
          nullptr, bytes, HAP_PROT_READ | HAP_PROT_WRITE, 0, descriptor[0], 0);
    if (raw && raw != (void*)-1) {
      auto* memory =
          memalign(alignof(CacheMappings::Slot), sizeof(CacheMappings::Slot));
      if (memory) {
        auto* slot = new (memory) CacheMappings::Slot{};
        slot->fd = descriptor[0];
        slot->bytes = bytes;
        slot->p = static_cast<uint16_t*>(raw);
        slot->mapped_by_get = by_get;
        slot->next = maps.entries;
        __atomic_store_n(&maps.entries, slot, __ATOMIC_RELEASE);
        pointer = slot->p;
      } else if (by_get)
        HAP_mmap_put(descriptor[0]);
      else
        HAP_munmap(raw, bytes);
    }
  }
  qurt_pimutex_unlock(&maps.lock);
  return pointer;
}
static CacheMappings::Slot* packed_cache(int fd, unsigned kv_bytes) {
  auto& maps = cache_mappings();
  for (auto* slot = __atomic_load_n(&maps.entries, __ATOMIC_ACQUIRE); slot;
       slot = slot->next) {
    if (slot->fd == fd && __atomic_load_n(&slot->packed, __ATOMIC_ACQUIRE))
      return slot->packed_bytes == 2 * kv_bytes ? slot : nullptr;
  }
  qurt_pimutex_lock(&maps.lock);
  CacheMappings::Slot* found = nullptr;
  for (auto* slot = maps.entries; slot; slot = slot->next) {
    if (slot->fd != fd)
      continue;
    if (slot->packed && slot->packed_bytes != 2 * kv_bytes)
      break;
    if (!slot->packed) {
      slot->packed_bytes = 2 * kv_bytes;
      __atomic_store_n(
          &slot->packed,
          static_cast<uint8_t*>(memalign(2048, 2 * kv_bytes)),
          __ATOMIC_RELEASE);
    }
    if (slot->packed)
      found = slot;
    break;
  }
  qurt_pimutex_unlock(&maps.lock);
  return found;
}
static void invalidate(uint16_t* p, unsigned offset, unsigned count) {
  if (!count)
    return;
  uintptr_t begin = uintptr_t(p + offset) & ~uintptr_t(127);
  uintptr_t end = (uintptr_t(p + offset + count) + 127) & ~uintptr_t(127);
  qurt_mem_cache_clean(
      begin, end - begin, QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
}
static uint32_t generic_execute(
    QHPI_RuntimeHandle* handle,
    QHPI_Tensor** out,
    const QHPI_Tensor* const* in,
    const Configuration& c) {
  if (qhpi_thread_resources(handle) != QHPI_RESOURCE_HVX)
    return QHPI_SUCCESS;
  const unsigned slice = qhpi_slice_number(handle),
                 slices = qhpi_num_slices(handle);
  if (!slices || slices > workers || slice >= slices)
    return QHPI_UNSUPPORTED;
  auto* key =
      cache_pointer(static_cast<const int32_t*>(qhpi_tensor_raw_data(in[2])));
  auto* value =
      cache_pointer(static_cast<const int32_t*>(qhpi_tensor_raw_data(in[4])));
  if (!key || !value)
    return QHPI_ERROR_FATAL;
  const auto& g = c.geometry;
  for (unsigned head = 0; head < g.batch * g.kv_heads; ++head) {
    for (unsigned d = 0; d < g.dimension; ++d)
      invalidate(
          key, (head * g.dimension + d) * c.capacity + c.first, c.history);
    invalidate(
        value,
        (head * c.capacity + c.first) * g.dimension,
        c.history * g.dimension);
  }
  const auto qq = qhpi_tensor_quant_parameters(in[1]),
             kq = qhpi_tensor_quant_parameters(in[3]),
             vq = qhpi_tensor_quant_parameters(in[5]),
             oq = qhpi_tensor_quant_parameters(out[0]);
  if (!Quantization{oq.stepsize, oq.zero_offset}.valid())
    return QHPI_UNSUPPORTED;
  Inputs args{
      static_cast<const uint16_t*>(qhpi_tensor_raw_data(in[1])),
      key,
      static_cast<const uint16_t*>(qhpi_tensor_raw_data(in[3])),
      value,
      static_cast<const uint16_t*>(qhpi_tensor_raw_data(in[5])),
      static_cast<const float*>(qhpi_tensor_raw_data(in[7])),
      {qq.stepsize, qq.zero_offset},
      {kq.stepsize, kq.zero_offset},
      {vq.stepsize, vq.zero_offset},
      {oq.stepsize, oq.zero_offset}};
  auto* work = static_cast<float*>(qhpi_tensor_raw_data(in[0])) +
      slice * ((g.dimension + 256 + 31) & ~31u);
  auto* y = static_cast<uint16_t*>(qhpi_tensor_raw_data(out[0]));
  for (unsigned task = slice; task < g.batch * g.heads * g.queries;
       task += slices) {
    const unsigned row = task % g.queries, head = task / g.queries % g.heads,
                   batch = task / g.queries / g.heads;
    reference_row(c, args, batch, head, row, work, y);
  }
  return QHPI_SUCCESS;
}
using op_utils::hvxAddF32;
using op_utils::hvxMulF32;
using op_utils::hvxSplatF32;
static HVX_Vector qdq(HVX_Vector x, float scale, int zero) {
  x = hvxAddF32(
      hvxMulF32(x, hvxSplatF32(1.f / scale)), hvxSplatF32(float(zero) + .5f));
  auto code = Q6_Vw_equals_Vsf(Q6_Vsf_vmin_VsfVsf(
      Q6_Vsf_vmax_VsfVsf(x, Q6_V_vzero()), hvxSplatF32(65535.f)));
  return hvxMulF32(
      Q6_Vsf_equals_Vw(Q6_Vw_vsub_VwVw(code, Q6_V_vsplat_R(zero))),
      hvxSplatF32(scale));
}
static HVX_Vector expv(HVX_Vector x) {
  x = Q6_Vsf_vmax_VsfVsf(x, hvxSplatF32(-87.f));
  auto z = hvxMulF32(x, hvxSplatF32(1.4426950408889634f));
  auto n = Q6_Vw_equals_Vsf(z);
  n = Q6_Vw_vsub_VwVw(
      n,
      Q6_V_vmux_QVV(
          Q6_Q_vcmp_gt_VsfVsf(Q6_Vsf_equals_Vw(n), z),
          Q6_V_vsplat_R(1),
          Q6_V_vzero()));
  auto r = Q6_Vsf_vsub_VsfVsf(
      x, hvxMulF32(Q6_Vsf_equals_Vw(n), hvxSplatF32(.6931471805599453f)));
  auto p = hvxSplatF32(1.f / 40320.f);
  for (float c :
       {1.f / 5040.f,
        1.f / 720.f,
        1.f / 120.f,
        1.f / 24.f,
        1.f / 6.f,
        .5f,
        1.f,
        1.f})
    p = hvxAddF32(hvxMulF32(p, r), hvxSplatF32(c));
  return hvxMulF32(
      p, Q6_Vw_vasl_VwR(Q6_Vw_vadd_VwVw(n, Q6_V_vsplat_R(127)), 23));
}
static float reduce(HVX_Vector v, bool maximum) {
  for (int shift = 64; shift >= 4; shift /= 2) {
    auto r = Q6_V_vror_VR(v, shift);
    v = maximum ? Q6_Vsf_vmax_VsfVsf(v, r) : hvxAddF32(v, r);
  }
  union {
    uint32_t u;
    float f;
  } value{uint32_t(Q6_R_vextract_VR(v, 0))};
  return value.f;
}
static void to_half(
    uint16_t* dst,
    const uint16_t* src,
    unsigned count,
    QHPI_Quant_Parameters q,
    bool is_signed = false) {
  for (unsigned i = 0; i < count; i += 64) {
    const auto raw = vmemu(src + i);
    auto u = is_signed ? Q6_Ww_vunpack_Vh(raw) : Q6_Wuw_vunpack_Vuh(raw);
    auto a = hvxMulF32(
        Q6_Vsf_equals_Vw(
            Q6_Vw_vsub_VwVw(Q6_V_lo_W(u), Q6_V_vsplat_R(q.zero_offset))),
        hvxSplatF32(q.stepsize));
    auto b = hvxMulF32(
        Q6_Vsf_equals_Vw(
            Q6_Vw_vsub_VwVw(Q6_V_hi_W(u), Q6_V_vsplat_R(q.zero_offset))),
        hvxSplatF32(q.stepsize));
    *reinterpret_cast<HVX_Vector*>(dst + i) =
        Q6_Vh_vdeal_Vh(Q6_Vhf_vcvt_VsfVsf(a, b));
  }
}
static void
unpack_row(float* dst, const uint16_t* c, unsigned row, unsigned n) {
  for (unsigned col = 0; col < n; col += 32) {
    auto raw = *reinterpret_cast<const HVX_Vector*>(
        c + (col / 32) * 1024 + (row / 2) * 64);
    auto f = Q6_Wsf_vcvt_Vhf(raw);
    *reinterpret_cast<HVX_Vector*>(dst + col) =
        row % 2 ? Q6_V_hi_W(f) : Q6_V_lo_W(f);
  }
}
struct alignas(16) DmaDescriptor {
  DmaDescriptor* next;
  uint32_t control;
  const void* source;
  void* destination;
};
static_assert(sizeof(DmaDescriptor) == 16);
static void dma_start(
    uint8_t* destination,
    const uint8_t* source,
    unsigned count,
    DmaDescriptor* descriptor) {
  descriptor->next = nullptr;
  descriptor->control = count | (1u << 28) | (1u << 29);
  descriptor->source = source;
  descriptor->destination = destination;
  asm volatile("release(%0):at" ::"r"(descriptor) : "memory");
  asm volatile("dmstart(%0)" ::"r"(descriptor) : "memory");
}
static bool dma_wait() {
  const auto result = Q6_R_dmwait();
  asm volatile("" ::: "memory");
  return (result & 2) == 0;
}
static bool dma_copy(
    uint8_t* destination,
    const uint8_t* source,
    unsigned count,
    DmaDescriptor* descriptor) {
  dma_start(destination, source, count, descriptor);
  return dma_wait();
}
static void
pack_decode_a(const uint16_t* source, uint8_t* destination, unsigned width) {
  for (unsigned row = 0; row < 8; row += 2)
    for (unsigned col = 0; col < width; col += 64) {
      const auto a =
          *reinterpret_cast<const HVX_Vector*>(source + row * width + col);
      const auto b = *reinterpret_cast<const HVX_Vector*>(
          source + (row + 1) * width + col);
      const auto pair = Q6_W_vshuff_VVR(b, a, -2);
      *reinterpret_cast<HVX_Vector*>(
          destination + (col / 32) * 2048 + (row / 2) * 128) = Q6_V_lo_W(pair);
      *reinterpret_cast<HVX_Vector*>(
          destination + (col / 32 + 1) * 2048 + (row / 2) * 128) =
          Q6_V_hi_W(pair);
    }
}
#include "attention/ShortAttention.h"
#endif
static uint32_t initialize(
    QHPI_RuntimeHandle* handle,
    uint32_t no,
    QHPI_Tensor** out,
    uint32_t ni,
    const QHPI_Tensor* const* in) {
#ifdef __hexagon__
  if (no != 2 || ni != 7)
    return QHPI_ERROR_FATAL;
  const auto shape = qhpi_tensor_shape(in[0]);
  const unsigned qheads = shape.dims[1],
                 kvheads = qhpi_tensor_shape(in[2]).dims[1];
  const unsigned kv_bytes = kvheads * kv_rows * 256 * 2;
  Configuration config{};
  if (!configuration(ni, in, config))
    return QHPI_UNSUPPORTED;
  if (!config.fast() || !QCOM_HTP_V81)
    return QHPI_SUCCESS;
  unsigned slice = qhpi_slice_number(handle), slices = qhpi_num_slices(handle);
  if (!slices || slices > max_workers || slice >= slices)
    return QHPI_UNSUPPORTED;

  auto* work = static_cast<uint8_t*>(qhpi_tensor_raw_data(out[0]));
  if (slice == 0) {
    for (unsigned i = 0; i < header / 128; ++i)
      reinterpret_cast<HVX_Vector*>(work)[i] = Q6_V_vzero();
    asm volatile("" ::: "memory");
    __atomic_store_n(
        &reinterpret_cast<attention_hmx::Queue*>(work)->workers,
        slices,
        __ATOMIC_RELEASE);
  }
  const auto* l = static_cast<const int32_t*>(qhpi_tensor_raw_data(in[5]));
  const unsigned qrows = qhpi_tensor_shape(in[0]).dims[2],
                 columns = (unsigned(static_cast<const int32_t*>(
                                qhpi_tensor_raw_data(in[1]))[1]) &
                            logical_size_mask) /
      (kvheads * 256 * 2),
                 capacity = columns;
  if (!config.fast() || l[0] < 1 || unsigned(l[0]) > qrows || l[1] < 0 ||
      l[2] < 0 || unsigned(l[1] + l[2]) > capacity)
    return QHPI_UNSUPPORTED;
  unsigned keys = l[0] + l[2], padded = (keys + 63) & ~63u;
  auto* stage = reinterpret_cast<uint16_t*>(
      work + worker_start + slice * (qrows == 1 ? 32768u : per_worker));
  uint8_t* kpack = nullptr;
  uint8_t* vpack = nullptr;
  auto* k =
      cache_pointer(static_cast<const int32_t*>(qhpi_tensor_raw_data(in[1])));
  const auto* meta = static_cast<const float*>(qhpi_tensor_raw_data(in[6]));
  QHPI_Quant_Parameters kq{int(meta[10]), meta[9]};
  auto* v =
      cache_pointer(static_cast<const int32_t*>(qhpi_tensor_raw_data(in[3])));
  QHPI_Quant_Parameters vq{int(meta[12]), meta[11]};
  const auto* nk = static_cast<const uint16_t*>(qhpi_tensor_raw_data(in[2]));
  auto nkq = qhpi_tensor_quant_parameters(in[2]);
  const auto* nv = static_cast<const uint16_t*>(qhpi_tensor_raw_data(in[4]));
  auto nvq = qhpi_tensor_quant_parameters(in[4]);
  if (!k || !v)
    return QHPI_ERROR_FATAL;
  auto* cached = packed_cache(
      static_cast<const int32_t*>(qhpi_tensor_raw_data(in[1]))[0], kv_bytes);
  if (!cached || !cached->packed)
    return QHPI_ERROR_FATAL;
  kpack = cached->packed;
  vpack = kpack + kv_bytes;
  const auto* fd_header =
      static_cast<const int32_t*>(qhpi_tensor_raw_data(in[1]));
  const bool same_epoch = cached->epoch == uint32_t(fd_header[2]);
  const bool incremental = same_epoch &&
      (unsigned(fd_header[1]) & exact_append_flag) && cached->last_exact &&
      l[1] == 0 &&
      unsigned(l[2]) == __atomic_load_n(&cached->history, __ATOMIC_ACQUIRE) &&
      l[2] != 0 && !cached->short_dense;
  const unsigned first_block = incremental ? unsigned(l[2]) / 64 : 0,
                 first_tile = incremental ? unsigned(l[2]) / 256 : 0;

  const bool append_one = incremental && qrows == 1 && cached->last_valid == 1;
  const bool exact =
      (unsigned(static_cast<const int32_t*>(qhpi_tensor_raw_data(in[1]))[1]) &
       exact_append_flag) != 0;
  const bool retained = exact && cached->last_exact;
  unsigned recent = incremental ? std::min(unsigned(l[2]), cached->last_valid)
                                : unsigned(l[2]);
  if (!append_one && !(qrows == 1 && keys <= short_key_limit && qheads == 16)) {
    for (unsigned d = slice; d < kvheads * 256; d += slices)
      invalidate(k, d * columns + l[1] + l[2] - recent, recent);
    for (unsigned head = slice; head < kvheads; head += slices)
      invalidate(
          v, (head * columns + l[1] + l[2] - recent) * 256, recent * 256);
  }

  if (qrows == 1 && keys <= short_key_limit && qheads == 16) {
    const bool dense_append = same_epoch && retained && l[1] == 0 && l[2] > 0 &&
        unsigned(l[2]) == __atomic_load_n(&cached->history, __ATOMIC_ACQUIRE) &&
        cached->short_dense;
    const unsigned first = dense_append ? unsigned(l[2]) - 1 : 0;
    for (unsigned task = slice; task < kvheads * 4; task += slices) {
      const unsigned kv = task / 4, col = (task % 4) * 64;
      auto* kd = reinterpret_cast<uint16_t*>(kpack) + kv * kv_rows * 256;
      auto* vd = reinterpret_cast<uint16_t*>(vpack) + kv * kv_rows * 256;
      for (unsigned row = first; row < keys; ++row) {
        const bool old = row < unsigned(l[2]);
        alignas(128) uint16_t kcodes[64];
        if (!old)
          *reinterpret_cast<HVX_Vector*>(kcodes) = vmemu(nk + kv * 256 + col);
        else if (retained && dense_append && row == unsigned(l[2]) - 1)
          *reinterpret_cast<HVX_Vector*>(kcodes) =
              *reinterpret_cast<const HVX_Vector*>(
                  cached->pending_k + kv * 256 + col);
        else
          for (unsigned d = 0; d < 64; ++d) {
            const unsigned at = (kv * 256 + col + d) * columns + l[1] + row;
            invalidate(k, at, 1);
            kcodes[d] = k[at];
          }
        to_half(kd + row * 256 + col, kcodes, 64, old ? kq : nkq);
        const uint16_t* vs;
        if (!old)
          vs = nv + kv * 256 + col;
        else if (retained && dense_append && row == unsigned(l[2]) - 1)
          vs = cached->pending_v + kv * 256 + col;
        else {
          const unsigned at = (kv * columns + l[1] + row) * 256 + col;
          invalidate(v, at, 64);
          vs = v + at;
        }
        to_half(vd + row * 256 + col, vs, 64, old ? vq : nvq);
      }
      if (exact) {
        *reinterpret_cast<HVX_Vector*>(cached->pending_k + task * 64) =
            vmemu(nk + task * 64);
        *reinterpret_cast<HVX_Vector*>(cached->pending_v + task * 64) =
            vmemu(nv + task * 64);
      }
    }
    return QHPI_SUCCESS;
  }

  unsigned blocks = padded / 64;
  if (append_one) {
    for (unsigned task = slice; task < kvheads * 4; task += slices) {
      const unsigned kv = task / 4, d0 = (task % 4) * 64,
                     new_block = unsigned(l[2]) / 64;
      auto* head = reinterpret_cast<uint16_t*>(kpack) + kv * kv_rows * 256;
      if ((unsigned(l[2]) % 64) == 0)
        for (unsigned half = 0; half < 2; ++half) {
          auto* dst =
              head + new_block * 64 * 256 + half * 256 * 32 + (d0 / 2) * 64;
          for (unsigned v = 0; v < 32; ++v)
            reinterpret_cast<HVX_Vector*>(dst)[v] = Q6_V_vzero();
        }
      for (unsigned logical = unsigned(l[2]) - 1; logical <= unsigned(l[2]);
           ++logical) {
        alignas(128) uint16_t codes[64], values[64];
        const bool old = logical < unsigned(l[2]);
        if (old) {
          if (retained)
            *reinterpret_cast<HVX_Vector*>(codes) =
                *reinterpret_cast<const HVX_Vector*>(
                    cached->pending_k + kv * 256 + d0);
          else
            for (unsigned d = 0; d < 64; ++d) {
              const unsigned offset = (kv * 256 + d0 + d) * columns + logical;
              invalidate(k, offset, 1);
              codes[d] = k[offset];
            }
        } else
          *reinterpret_cast<HVX_Vector*>(codes) = vmemu(nk + kv * 256 + d0);
        to_half(values, codes, 64, old ? kq : nkq);
        auto* dst = head + (logical / 64) * 64 * 256 +
            ((logical % 64) / 32) * 256 * 32 + (d0 / 2) * 64 +
            (logical % 32) * 2;
        for (unsigned pair = 0; pair < 32; ++pair)
          *reinterpret_cast<uint32_t*>(dst + pair * 64) =
              reinterpret_cast<const uint32_t*>(values)[pair];
      }
    }
  } else {
    for (unsigned task = slice; task < kvheads * (blocks - first_block);
         task += slices) {
      unsigned kv = task / (blocks - first_block),
               start = (first_block + task % (blocks - first_block)) * 64;
      for (unsigned d = 0; d < 256; ++d) {
        if (start + 64 <= unsigned(l[2]))
          to_half(
              stage + d * 64,
              k + (kv * 256 + d) * columns + l[1] + start,
              64,
              kq);
        else if (start >= unsigned(l[2]) && start + 64 <= keys)
          to_half(
              stage + d * 64,
              nk + (kv * 256 + d) * qrows + start - l[2],
              64,
              nkq);
        else {
          alignas(128) uint16_t hc[64], nc[64], hh[64], nh[64];
          for (unsigned c = 0; c < 64; ++c) {
            unsigned logical = start + c;
            hc[c] = logical < unsigned(l[2])
                ? k[(kv * 256 + d) * columns + l[1] + logical]
                : kq.zero_offset;
            nc[c] = logical >= unsigned(l[2]) && logical < keys
                ? nk[(kv * 256 + d) * qrows + logical - l[2]]
                : nkq.zero_offset;
          }
          to_half(hh, hc, 64, kq);
          to_half(nh, nc, 64, nkq);
          for (unsigned c = 0; c < 64; ++c)
            stage[d * 64 + c] = start + c < unsigned(l[2]) ? hh[c] : nh[c];
        }
      }
      qhblas_hmx_transfer_column_ahf(
          reinterpret_cast<__fp16*>(stage),
          kpack + (kv * kv_rows + start) * 256 * 2,
          256,
          64);
    }
  }

  unsigned tiles = (padded + 255) / 256;
  if (append_one) {
    for (unsigned task = slice; task < kvheads * 4; task += slices) {
      const unsigned kv = task / 4, col = (task % 4) * 64,
                     kt = unsigned(l[2]) / 256,
                     width = std::min(256u, padded - kt * 256);
      auto* head =
          reinterpret_cast<uint16_t*>(vpack) + kv * (kv_rows / 256) * 256 * 256;
      if (unsigned(l[2]) % 64 == 0) {
        for (unsigned row = 0; row < width; ++row) {
          unsigned logical = kt * 256 + row;
          if (logical < unsigned(l[2]))
            to_half(
                stage + row * 64,
                v + (kv * columns + logical) * 256 + col,
                64,
                vq);
          else if (logical < keys)
            to_half(
                stage + row * 64,
                nv + (kv * qrows + logical - l[2]) * 256 + col,
                64,
                nvq);
          else
            *reinterpret_cast<HVX_Vector*>(stage + row * 64) = Q6_V_vzero();
        }
        qhblas_hmx_transfer_column_ahf(
            reinterpret_cast<__fp16*>(stage),
            reinterpret_cast<uint8_t*>(head + kt * 256 * 256 + col * width),
            width,
            64);
      }
      for (unsigned logical = unsigned(l[2]) - 1; logical <= unsigned(l[2]);
           ++logical) {
        const bool old = logical < unsigned(l[2]);
        alignas(128) uint16_t values[64];
        if (old && !retained)
          invalidate(v, (kv * columns + logical) * 256 + col, 64);
        to_half(
            values,
            old ? (retained ? cached->pending_v + kv * 256 + col
                            : v + (kv * columns + logical) * 256 + col)
                : nv + kv * 256 + col,
            64,
            old ? vq : nvq);
        const auto halves = Q6_W_vshuff_VVR(
            Q6_V_vzero(), *reinterpret_cast<const HVX_Vector*>(values), -2);
        const unsigned part = logical / 256, row = logical % 256,
                       kw = part == kt ? width : 256;
        auto mask = Q6_V_vsplat_R((row & 1) ? 0xffff0000 : 0x0000ffff);
        for (unsigned half = 0; half < 2; ++half) {
          auto* dst = reinterpret_cast<HVX_Vector*>(
              head + part * 256 * 256 + col * kw + half * kw * 32 +
              (row / 2) * 64);
          auto value = half ? Q6_V_hi_W(halves) : Q6_V_lo_W(halves);
          if (row & 1)
            value = Q6_Vw_vasl_VwR(value, 16);
          *dst = Q6_V_vor_VV(
              Q6_V_vand_VV(value, mask), Q6_V_vand_VV(*dst, Q6_V_vnot_V(mask)));
        }
      }
    }
  } else {
    for (unsigned task = slice; task < kvheads * (tiles - first_tile) * 4;
         task += slices) {
      unsigned kv = task / ((tiles - first_tile) * 4),
               kt = first_tile + (task / 4) % (tiles - first_tile),
               col = (task % 4) * 64;
      unsigned width = std::min(256u, padded - kt * 256);
      for (unsigned row = 0; row < width; ++row) {
        unsigned logical = kt * 256 + row;
        if (logical < unsigned(l[2]))
          to_half(
              stage + row * 64,
              v + (kv * columns + l[1] + logical) * 256 + col,
              64,
              vq);
        else if (logical < keys)
          to_half(
              stage + row * 64,
              nv + (kv * qrows + logical - l[2]) * 256 + col,
              64,
              nvq);
        else
          *reinterpret_cast<HVX_Vector*>(stage + row * 64) = Q6_V_vzero();
      }
      qhblas_hmx_transfer_column_ahf(
          reinterpret_cast<__fp16*>(stage),
          vpack + (kv * (kv_rows / 256) + kt) * 256 * 256 * 2 + col * width * 2,
          width,
          64);
    }
  }
  if (qrows == 1 && exact)
    for (unsigned task = slice; task < kvheads * 4; task += slices) {
      *reinterpret_cast<HVX_Vector*>(cached->pending_k + task * 64) =
          vmemu(nk + task * 64);
      *reinterpret_cast<HVX_Vector*>(cached->pending_v + task * 64) =
          vmemu(nv + task * 64);
    }

  return QHPI_SUCCESS;
#else
  return QHPI_UNSUPPORTED;
#endif
}
#ifdef __hexagon__
#include "attention/OnlineAttention.h"
#endif
static uint32_t execute(
    QHPI_RuntimeHandle* handle,
    uint32_t no,
    QHPI_Tensor** out,
    uint32_t ni,
    const QHPI_Tensor* const* in) {
#ifdef __hexagon__
  if (no != 1 || ni != 9)
    return QHPI_ERROR_FATAL;
  const auto shape = qhpi_tensor_shape(in[1]);
  const unsigned qheads = shape.dims[1],
                 kvheads = qhpi_tensor_shape(in[3]).dims[1];
  const unsigned kv_bytes = kvheads * kv_rows * 256 * 2;
  const QHPI_Tensor* original[] = {
      in[1], in[2], in[3], in[4], in[5], in[6], in[7]};
  Configuration config{};
  if (!configuration(7, original, config))
    return QHPI_UNSUPPORTED;
  const auto oq_check = qhpi_tensor_quant_parameters(out[0]);
  if (!shape_is(
          qhpi_tensor_shape(out[0]),
          config.geometry.batch,
          config.geometry.heads,
          config.geometry.queries,
          config.geometry.dimension) ||
      !Quantization{oq_check.stepsize, oq_check.zero_offset}.valid())
    return QHPI_UNSUPPORTED;
  if (!config.fast() || !QCOM_HTP_V81)
    return generic_execute(handle, out, in, config);

  auto* work = static_cast<uint8_t*>(qhpi_tensor_raw_data(in[0]));
  auto& queue = *reinterpret_cast<attention_hmx::Queue*>(work);
  const unsigned resource = qhpi_thread_resources(handle);
  if (resource == 0) {
    auto* cached = packed_cache(
        static_cast<const int32_t*>(qhpi_tensor_raw_data(in[2]))[0], kv_bytes);
    const auto* l = static_cast<const int32_t*>(qhpi_tensor_raw_data(in[6]));
    if (qhpi_tensor_shape(in[1]).dims[2] > 1 ||
        (unsigned(l[0] + l[2]) > short_key_limit || qheads != 16)) {
      const unsigned padded = (unsigned(l[0] + l[2]) + 63) & ~63u;
      const auto* descriptor =
          static_cast<const int32_t*>(qhpi_tensor_raw_data(in[2]));
      const bool incremental = cached &&
          cached->epoch == uint32_t(descriptor[2]) &&
          (unsigned(descriptor[1]) & exact_append_flag) && cached->last_exact &&
          l[1] == 0 &&
          unsigned(l[2]) ==
              __atomic_load_n(&cached->history, __ATOMIC_ACQUIRE) &&
          l[2] != 0 && !cached->short_dense &&
          cached->coherent_epoch == uint32_t(descriptor[2]) &&
          cached->coherent_history == unsigned(l[2]);
      const unsigned first_key = incremental ? unsigned(l[2]) & ~63u : 0,
                     first_value = incremental ? unsigned(l[2]) & ~255u : 0;
      if (cached && cached->packed) {
        for (unsigned h = 0; h < kvheads; ++h) {
          qurt_mem_cache_clean(
              uintptr_t(cached->packed + (h * kv_rows + first_key) * 256 * 2),
              (padded - first_key) * 256 * 2,
              QURT_MEM_CACHE_FLUSH,
              QURT_MEM_DCACHE);
          qurt_mem_cache_clean(
              uintptr_t(
                  cached->packed + kv_bytes +
                  (h * kv_rows + first_value) * 256 * 2),
              (padded - first_value) * 256 * 2,
              QURT_MEM_CACHE_FLUSH,
              QURT_MEM_DCACHE);
        }
        cached->coherent_epoch = uint32_t(descriptor[2]);
        cached->coherent_history = unsigned(l[0] + l[2]);
      }
      __atomic_store_n(
          reinterpret_cast<uint32_t*>(work + 1536), 1u, __ATOMIC_RELEASE);
    }
    if (cached) {
      cached->short_dense = qhpi_tensor_shape(in[1]).dims[2] == 1 &&
          unsigned(l[0] + l[2]) <= short_key_limit && qheads == 16;
      cached->last_exact = (unsigned(static_cast<const int32_t*>(
                                qhpi_tensor_raw_data(in[2]))[1]) &
                            exact_append_flag) != 0;
      cached->last_valid = l[0];
      cached->epoch =
          uint32_t(static_cast<const int32_t*>(qhpi_tensor_raw_data(in[2]))[2]);
      __atomic_store_n(
          &cached->history, unsigned(l[0] + l[2]), __ATOMIC_RELEASE);
    }
    const bool hvx_only = qhpi_tensor_shape(in[1]).dims[2] == 1 &&
        unsigned(l[0] + l[2]) <= short_key_limit && qheads == 16;
    unsigned context = qhpi_srm_context(handle);
    bool locked =
        hvx_only || (context && HAP_compute_res_hmx_lock(context) == 0);

    if (!locked)
      __atomic_store_n(&queue.error, 1u, __ATOMIC_RELEASE);
    bool okay = attention_hmx::serve(queue, [&](const attention_hmx::Job& j) {
      for (unsigned c = 0; c < j.n; c += 32)
        qhblas_hmx_matrix_matrix_mpy_hf_sm(
            j.a,
            ((j.k / 32) << 11) | ((j.m - 1) << 2),
            j.b + c * j.k * 2,
            ((j.k / 32) << 11) - 1,
            j.c + c * 64,
            0,
            j.bias);
      return true;
    });

    if (locked && !hvx_only)
      HAP_compute_res_hmx_unlock(context);
    return okay && locked ? QHPI_SUCCESS : QHPI_ERROR_FATAL;
  }
  if (resource != QHPI_RESOURCE_HVX)
    return QHPI_SUCCESS;
  const unsigned slice = qhpi_slice_number(handle),
                 workers = qhpi_num_slices(handle);
  if (!workers || workers > max_workers || slice >= workers) {
    __atomic_store_n(&queue.error, 1u, __ATOMIC_RELEASE);
    return QHPI_UNSUPPORTED;
  }
  const auto* lengths =
      static_cast<const int32_t*>(qhpi_tensor_raw_data(in[6]));
  const auto* meta = static_cast<const float*>(qhpi_tensor_raw_data(in[7]));
  const unsigned qrows = qhpi_tensor_shape(in[1]).dims[2],
                 columns = (unsigned(static_cast<const int32_t*>(
                                qhpi_tensor_raw_data(in[2]))[1]) &
                            logical_size_mask) /
      (kvheads * 256 * 2);
  int valid = lengths[0], first = lengths[1], history = lengths[2],
      capacity = columns;
  if (valid < 1 || unsigned(valid) > qrows || first < 0 || history < 0 ||
      first + history > capacity) {
    __atomic_store_n(&queue.error, 1u, __ATOMIC_RELEASE);
    __atomic_store_n(&queue.jobs[slice].finished, 1u, __ATOMIC_RELEASE);
    return QHPI_UNSUPPORTED;
  }
  const unsigned keys = history + valid, padded = (keys + 63) & ~63u;
  if (qrows > 1 || keys > short_key_limit || qheads != 16) {
    while (!__atomic_load_n(
        reinterpret_cast<uint32_t*>(work + 1536), __ATOMIC_ACQUIRE))
      attention_hmx::pause();
  }
  if (keys >= online_threshold) {
    while (!__atomic_load_n(
        reinterpret_cast<uint32_t*>(work + 1536), __ATOMIC_ACQUIRE))
      attention_hmx::pause();
    return online_prefill(handle, out, in);
  }
  if (qrows == 1 && keys <= short_key_limit && qheads == 16) {
    auto* cached = packed_cache(
        static_cast<const int32_t*>(qhpi_tensor_raw_data(in[2]))[0], kv_bytes);
    for (unsigned shard = slice; shard < max_workers; shard += workers)
      short_attention(
          static_cast<const uint16_t*>(qhpi_tensor_raw_data(in[1])),
          static_cast<uint16_t*>(qhpi_tensor_raw_data(out[0])),
          cached->packed,
          cached->packed + kv_bytes,
          keys,
          shard,
          work,
          qhpi_tensor_quant_parameters(in[1]),
          qhpi_tensor_quant_parameters(out[0]),
          meta);
    __atomic_store_n(&queue.jobs[slice].finished, 1u, __ATOMIC_RELEASE);
    return QHPI_SUCCESS;
  }
  if (qrows == 1 && slice >= kvheads) {
    __atomic_store_n(&queue.jobs[slice].finished, 1u, __ATOMIC_RELEASE);
    return QHPI_SUCCESS;
  }
  bool dma_pending = false;
  auto* base = work + worker_start + slice * per_worker;
  auto* ddr = static_cast<uint8_t*>(qhpi_tensor_raw_data(in[8]));
  auto* cached = packed_cache(
      static_cast<const int32_t*>(qhpi_tensor_raw_data(in[2]))[0], kv_bytes);
  auto* kpack = cached->packed;
  auto* vpack = kpack + kv_bytes;
  auto* qa = reinterpret_cast<uint16_t*>(base);
  auto* packed_a = base + 16384;
  auto* packed_c = reinterpret_cast<uint16_t*>(base + 32768);
  auto* prob = reinterpret_cast<uint16_t*>(base + 49152);
  const unsigned live_stride = qrows == 1 ? padded : stride;
  const unsigned live_score_bytes = kvheads * 8 * live_stride * 2;
  const bool pipeline = qrows == 1 &&
      live_score_bytes <= kvheads * 8 * stride * 2 - kvheads * tile * 256 * 2;
  auto* other_b = ddr + kvheads * 8 * stride * 2 - kvheads * tile * 256 * 2 +
      slice * tile * 256 * 2;
  const bool k_pipeline = pipeline || qrows > 1;
  const unsigned qk_tile = qrows == 1 ? tile : 128;
  auto* key_other_b = qrows == 1 ? other_b : base + 165888;
  auto* scores = reinterpret_cast<uint16_t*>(
      ddr + slice * (qrows == 1 ? 8 * live_stride * 2 : score_bytes));
  auto* acc = reinterpret_cast<float*>(base + 65536);
  auto* bias = reinterpret_cast<uint32_t*>(base + 98304);
  *reinterpret_cast<HVX_Vector*>(bias) = Q6_V_vsplat_R(0x3c00);
  *reinterpret_cast<HVX_Vector*>(bias + 32) = Q6_V_vzero();
  const auto* q = static_cast<const uint16_t*>(qhpi_tensor_raw_data(in[1]));
  auto qquant = qhpi_tensor_quant_parameters(in[1]);
  auto oq = qhpi_tensor_quant_parameters(out[0]);
  auto* y = static_cast<uint16_t*>(qhpi_tensor_raw_data(out[0]));
  for (unsigned head = slice; head < (qrows == 1 ? kvheads : qheads);
       head += workers) {
    unsigned kv = qrows == 1 ? head : head / 8;
    const unsigned queries = qrows == 1 ? 8u : unsigned(valid);
    for (unsigned begin = 0; begin < queries; begin += rows) {
      unsigned actual = std::min(rows, queries - begin);
      if (k_pipeline) {
        dma_start(
            base + 100352,
            kpack + kv * kv_rows * 256 * 2,
            std::min(qk_tile, padded) * 256 * 2,
            reinterpret_cast<DmaDescriptor*>(cached->dma_words[slice]));
        dma_pending = true;
      }
      for (unsigned r = 0; r < (qrows == 1 ? 8u : rows); ++r) {
        if (r < actual)
          to_half(
              qa + r * 256,
              q + (qrows == 1 ? kv * 8 + r : head * qrows + begin + r) * 256,
              256,
              qquant);
        else
          for (unsigned i = 0; i < 4; ++i)
            reinterpret_cast<HVX_Vector*>(qa + r * 256)[i] = Q6_V_vzero();
      }
      if (qrows == 1)
        pack_decode_a(qa, packed_a, 256);
      else
        qhblas_hmx_transfer_row_ahf(
            reinterpret_cast<__fp16*>(qa), packed_a, 256);
      for (unsigned start = 0; start < padded; start += qk_tile) {
        unsigned width = std::min(qk_tile, padded - start);
        auto* packed_b =
            k_pipeline && ((start / qk_tile) & 1) ? key_other_b : base + 100352;
        const auto* source = kpack + (kv * kv_rows + start) * 256 * 2;
        if (k_pipeline) {
          const bool okay = dma_wait();
          dma_pending = false;
          if (!okay) {
            __atomic_store_n(&queue.error, 1u, __ATOMIC_RELEASE);
            goto done;
          }
          if (start + width < padded) {
            const unsigned next = start + width,
                           next_width = std::min(qk_tile, padded - next);
            auto* next_b = ((next / qk_tile) & 1) ? key_other_b : base + 100352;
            dma_start(
                next_b,
                kpack + (kv * kv_rows + next) * 256 * 2,
                next_width * 256 * 2,
                reinterpret_cast<DmaDescriptor*>(cached->dma_words[slice]));
            dma_pending = true;
          }
        } else if (qrows == 1) {
          if (!dma_copy(
                  packed_b,
                  source,
                  width * 256 * 2,
                  reinterpret_cast<DmaDescriptor*>(cached->dma_words[slice]))) {
            __atomic_store_n(&queue.error, 1u, __ATOMIC_RELEASE);
            goto done;
          }
        } else {
#ifdef FA_SYNC_DMA
          if (!dma_copy(
                  packed_b,
                  source,
                  width * 256 * 2,
                  reinterpret_cast<DmaDescriptor*>(cached->dma_words[slice]))) {
            __atomic_store_n(&queue.error, 1u, __ATOMIC_RELEASE);
            goto done;
          }
#else
          for (unsigned i = 0; i < width * 256 * 2; i += 128)
            *reinterpret_cast<HVX_Vector*>(packed_b + i) =
                *reinterpret_cast<const HVX_Vector*>(source + i);
#endif
        }

        if (!attention_hmx::submit(
                queue,
                slice,
                packed_a,
                packed_b,
                reinterpret_cast<uint8_t*>(packed_c),
                reinterpret_cast<uint8_t*>(bias),
                qrows == 1 ? 8 : 32,
                256,
                width))
          goto done;
        if (qrows == 1) {
          auto encode = [&](HVX_Vector z) {
            z = qdq(z, meta[0], int(meta[1]));
            z = qdq(hvxMulF32(z, hvxSplatF32(meta[2])), meta[3], int(meta[4]));
            return Q6_Vw_equals_Vsf(hvxAddF32(
                hvxMulF32(z, hvxSplatF32(1.f / meta[5])),
                hvxSplatF32(meta[6] + .5f)));
          };
          for (unsigned r = 0; r < actual; r += 2)
            for (unsigned c = 0; c < width; c += 64) {
              const auto a =
                  Q6_Wsf_vcvt_Vhf(*reinterpret_cast<const HVX_Vector*>(
                      packed_c + (c / 32) * 1024 + (r / 2) * 64));
              const auto b =
                  Q6_Wsf_vcvt_Vhf(*reinterpret_cast<const HVX_Vector*>(
                      packed_c + (c / 32 + 1) * 1024 + (r / 2) * 64));
              *reinterpret_cast<HVX_Vector*>(
                  scores + r * live_stride + start + c) =
                  Q6_Vuh_vpack_VwVw_sat(
                      encode(Q6_V_lo_W(b)), encode(Q6_V_lo_W(a)));
              *reinterpret_cast<HVX_Vector*>(
                  scores + (r + 1) * live_stride + start + c) =
                  Q6_Vuh_vpack_VwVw_sat(
                      encode(Q6_V_hi_W(b)), encode(Q6_V_hi_W(a)));
            }
        } else {
          for (unsigned r = 0; r < actual; ++r) {
            alignas(128) float values[256];
            unpack_row(values, packed_c, r, width);
            for (unsigned c = 0; c < width; c += 64) {
              HVX_Vector codes[2];
              for (unsigned half = 0; half < 2; ++half) {
                auto z = *reinterpret_cast<HVX_Vector*>(values + c + half * 32);
                z = qdq(z, meta[0], int(meta[1]));
                z = qdq(
                    hvxMulF32(z, hvxSplatF32(meta[2])), meta[3], int(meta[4]));
                z = hvxAddF32(
                    hvxMulF32(z, hvxSplatF32(1.f / meta[5])),
                    hvxSplatF32(meta[6] + .5f));
                codes[half] = Q6_Vw_equals_Vsf(z);
              }
              *reinterpret_cast<HVX_Vector*>(
                  scores + r * live_stride + start + c) =
                  Q6_Vuh_vpack_VwVw_sat(codes[1], codes[0]);
            }
          }
        }
      }

      if (pipeline) {
        dma_start(
            other_b,
            vpack + kv * (kv_rows / 256) * 256 * 256 * 2,
            std::min(tile, padded) * 256 * 2,
            reinterpret_cast<DmaDescriptor*>(cached->dma_words[slice]));
        dma_pending = true;
      }

      if (qrows > 1) {
        dma_start(
            key_other_b,
            vpack + kv * (kv_rows / 256) * 256 * 256 * 2,
            std::min(tile, padded) * 128 * 2,
            reinterpret_cast<DmaDescriptor*>(cached->dma_words[slice]));
        dma_pending = true;
      }
      alignas(128) uint16_t lane_ids[64];
      for (unsigned i = 0; i < 64; ++i)
        lane_ids[i] = i;
      const auto ids = *reinterpret_cast<HVX_Vector*>(lane_ids);
      auto visible = [&](unsigned r, unsigned c) {
        return Q6_Q_vcmp_gt_VuhVuh(
            Q6_Vh_vsplat_R(history + (qrows == 1 ? 0 : begin + r) + 1),
            Q6_Vh_vadd_VhVh(ids, Q6_Vh_vsplat_R(c)));
      };
      for (unsigned r = 0; r < actual; ++r) {
        auto mx = Q6_V_vzero();
        for (unsigned c = 0; c < padded; c += 64) {
          auto raw =
              *reinterpret_cast<HVX_Vector*>(scores + r * live_stride + c);
          raw = Q6_V_vmux_QVV(visible(r, c), raw, Q6_V_vzero());
          mx = Q6_Vuh_vmax_VuhVuh(mx, raw);
        }
        for (unsigned shift = 64; shift >= 2; shift /= 2)
          mx = Q6_Vuh_vmax_VuhVuh(mx, Q6_V_vror_VR(mx, shift));
        unsigned maxcode = Q6_R_vextract_VR(mx, 0) & 65535;
        auto lookup = [&](unsigned c, HVX_Vector& lo, HVX_Vector& hi) {
          auto code =
              *reinterpret_cast<HVX_Vector*>(scores + r * live_stride + c);
          auto delta = Q6_Wuw_vunpack_Vuh(
              Q6_Vh_vsub_VhVh(Q6_Vh_vsplat_R(maxcode), code));
          lo = expv(hvxMulF32(
              Q6_Vsf_equals_Vw(Q6_V_lo_W(delta)), hvxSplatF32(-meta[5])));
          hi = expv(hvxMulF32(
              Q6_Vsf_equals_Vw(Q6_V_hi_W(delta)), hvxSplatF32(-meta[5])));
          auto mask = Q6_Wuw_vunpack_Vuh(
              Q6_V_vmux_QVV(visible(r, c), Q6_Vh_vsplat_R(-1), Q6_V_vzero()));
          lo = Q6_V_vmux_QVV(
              Q6_Q_vcmp_gt_VwVw(Q6_V_lo_W(mask), Q6_V_vzero()),
              lo,
              Q6_V_vzero());
          hi = Q6_V_vmux_QVV(
              Q6_Q_vcmp_gt_VwVw(Q6_V_hi_W(mask), Q6_V_vzero()),
              hi,
              Q6_V_vzero());
        };
        auto* exponent = reinterpret_cast<float*>(base + 100352);
        auto sum = Q6_V_vzero();
        for (unsigned c = 0; c < padded; c += 64) {
          HVX_Vector lo, hi;
          lookup(c, lo, hi);
          *reinterpret_cast<HVX_Vector*>(exponent + c) = lo;
          *reinterpret_cast<HVX_Vector*>(exponent + c + 32) = hi;
          sum = hvxAddF32(sum, hvxAddF32(lo, hi));
        }
        float inv = 1.f / (reduce(sum, false) * meta[7]);
        for (unsigned c = 0; c < padded; c += 64) {
          auto lo = *reinterpret_cast<const HVX_Vector*>(exponent + c),
               hi = *reinterpret_cast<const HVX_Vector*>(exponent + c + 32);
          lo = hvxAddF32(
              hvxMulF32(lo, hvxSplatF32(inv)), hvxSplatF32(meta[8] + .5f));
          hi = hvxAddF32(
              hvxMulF32(hi, hvxSplatF32(inv)), hvxSplatF32(meta[8] + .5f));
          *reinterpret_cast<HVX_Vector*>(scores + r * live_stride + c) =
              Q6_Vuh_vpack_VwVw_sat(Q6_Vw_equals_Vsf(hi), Q6_Vw_equals_Vsf(lo));
        }
      }

      for (unsigned i = 0; i < actual * 256 / 32; ++i)
        reinterpret_cast<HVX_Vector*>(acc)[i] = Q6_V_vzero();
      if (qrows > 1) {
        for (unsigned start = 0; start < padded; start += tile) {
          const unsigned width = std::min(tile, padded - start);

          QHPI_Quant_Parameters probability_quant{int(meta[8]), meta[7]};
          for (unsigned r = 0; r < rows; ++r)
            for (unsigned c = 0; c < width; c += 64) {
              if (r < actual)
                to_half(
                    prob + r * width + c,
                    scores + r * live_stride + start + c,
                    64,
                    probability_quant);
              else
                *reinterpret_cast<HVX_Vector*>(prob + r * width + c) =
                    Q6_V_vzero();
            }
          qhblas_hmx_transfer_row_ahf(
              reinterpret_cast<__fp16*>(prob), packed_a, width);

          const auto* source =
              vpack + (kv * (kv_rows / 256) + start / 256) * 256 * 256 * 2;
          for (unsigned output_col = 0; output_col < 256; output_col += 128) {
            auto* packed_b = output_col ? base + 100352 : key_other_b;

            const bool okay = dma_wait();
            dma_pending = false;
            if (!okay) {
              __atomic_store_n(&queue.error, 1u, __ATOMIC_RELEASE);
              goto done;
            }
            if (!output_col) {
              dma_start(
                  base + 100352,
                  source + 128 * width * 2,
                  width * 128 * 2,
                  reinterpret_cast<DmaDescriptor*>(cached->dma_words[slice]));
              dma_pending = true;
            } else if (start + width < padded) {
              const unsigned next = start + width,
                             next_width = std::min(tile, padded - next);
              dma_start(
                  key_other_b,
                  vpack + (kv * (kv_rows / 256) + next / 256) * 256 * 256 * 2,
                  next_width * 128 * 2,
                  reinterpret_cast<DmaDescriptor*>(cached->dma_words[slice]));
              dma_pending = true;
            }

            if (!attention_hmx::submit(
                    queue,
                    slice,
                    packed_a,
                    packed_b,
                    reinterpret_cast<uint8_t*>(packed_c),
                    reinterpret_cast<uint8_t*>(bias),
                    32,
                    width,
                    128))
              goto done;

            for (unsigned r = 0; r < actual; ++r) {
              alignas(128) float values[128];
              unpack_row(values, packed_c, r, 128);
              for (unsigned c = 0; c < 128; c += 32)
                *reinterpret_cast<HVX_Vector*>(acc + r * 256 + output_col + c) =
                    hvxAddF32(
                        *reinterpret_cast<HVX_Vector*>(
                            acc + r * 256 + output_col + c),
                        *reinterpret_cast<HVX_Vector*>(values + c));
            }
          }
        }
      } else {
        for (unsigned start = 0; start < padded; start += tile) {
          unsigned width = std::min(tile, padded - start);

          QHPI_Quant_Parameters probability_quant{int(meta[8]), meta[7]};
          for (unsigned r = 0; r < (qrows == 1 ? 8u : rows); ++r)
            for (unsigned c = 0; c < width; c += 64) {
              if (r < actual)
                to_half(
                    prob + r * width + c,
                    scores + r * live_stride + start + c,
                    64,
                    probability_quant);
              else
                *reinterpret_cast<HVX_Vector*>(prob + r * width + c) =
                    Q6_V_vzero();
            }
          if (qrows == 1)
            pack_decode_a(prob, packed_a, width);
          else
            qhblas_hmx_transfer_row_ahf(
                reinterpret_cast<__fp16*>(prob), packed_a, width);

          auto* packed_b =
              pipeline && !((start / tile) & 1) ? other_b : base + 100352;
          const auto* source =
              vpack + (kv * (kv_rows / 256) + start / 256) * 256 * 256 * 2;
          if (pipeline) {
            const bool okay = dma_wait();
            dma_pending = false;
            if (!okay) {
              __atomic_store_n(&queue.error, 1u, __ATOMIC_RELEASE);
              goto done;
            }
            if (start + width < padded) {
              const unsigned next = start + width,
                             next_width = std::min(tile, padded - next);
              auto* next_b = ((next / tile) & 1) ? base + 100352 : other_b;
              dma_start(
                  next_b,
                  vpack + (kv * (kv_rows / 256) + next / 256) * 256 * 256 * 2,
                  next_width * 256 * 2,
                  reinterpret_cast<DmaDescriptor*>(cached->dma_words[slice]));
              dma_pending = true;
            }
          } else if (qrows == 1) {
            if (!dma_copy(
                    packed_b,
                    source,
                    width * 256 * 2,
                    reinterpret_cast<DmaDescriptor*>(
                        cached->dma_words[slice]))) {
              __atomic_store_n(&queue.error, 1u, __ATOMIC_RELEASE);
              goto done;
            }
          } else {
#ifdef FA_SYNC_DMA
            if (!dma_copy(
                    packed_b,
                    source,
                    width * 256 * 2,
                    reinterpret_cast<DmaDescriptor*>(
                        cached->dma_words[slice]))) {
              __atomic_store_n(&queue.error, 1u, __ATOMIC_RELEASE);
              goto done;
            }
#else
            for (unsigned i = 0; i < width * 256 * 2; i += 128)
              *reinterpret_cast<HVX_Vector*>(packed_b + i) =
                  *reinterpret_cast<const HVX_Vector*>(source + i);
#endif
          }

          if (!attention_hmx::submit(
                  queue,
                  slice,
                  packed_a,
                  packed_b,
                  reinterpret_cast<uint8_t*>(packed_c),
                  reinterpret_cast<uint8_t*>(bias),
                  qrows == 1 ? 8 : 32,
                  width,
                  256))
            goto done;
          if (qrows == 1) {
            for (unsigned r = 0; r < actual; r += 2)
              for (unsigned c = 0; c < 256; c += 32) {
                const auto values =
                    Q6_Wsf_vcvt_Vhf(*reinterpret_cast<const HVX_Vector*>(
                        packed_c + (c / 32) * 1024 + (r / 2) * 64));
                auto* a = reinterpret_cast<HVX_Vector*>(acc + r * 256 + c);
                auto* b =
                    reinterpret_cast<HVX_Vector*>(acc + (r + 1) * 256 + c);
                *a = hvxAddF32(*a, Q6_V_lo_W(values));
                *b = hvxAddF32(*b, Q6_V_hi_W(values));
              }
          } else {
            for (unsigned r = 0; r < actual; ++r) {
              alignas(128) float values[256];
              unpack_row(values, packed_c, r, 256);
              for (unsigned c = 0; c < 256; c += 32)
                *reinterpret_cast<HVX_Vector*>(acc + r * 256 + c) = hvxAddF32(
                    *reinterpret_cast<HVX_Vector*>(acc + r * 256 + c),
                    *reinterpret_cast<HVX_Vector*>(values + c));
            }
          }
        }
      }

      for (unsigned r = 0; r < actual; ++r)
        for (unsigned c = 0; c < 256; c += 64) {
          auto a = *reinterpret_cast<HVX_Vector*>(acc + r * 256 + c),
               b = *reinterpret_cast<HVX_Vector*>(acc + r * 256 + c + 32);
          a = hvxAddF32(
              hvxMulF32(a, hvxSplatF32(1.f / oq.stepsize)),
              hvxSplatF32(oq.zero_offset + .5f));
          b = hvxAddF32(
              hvxMulF32(b, hvxSplatF32(1.f / oq.stepsize)),
              hvxSplatF32(oq.zero_offset + .5f));
          *reinterpret_cast<HVX_Vector*>(
              y + (qrows == 1 ? kv * 8 + r : head * qrows + begin + r) * 256 +
              c) =
              Q6_Vuh_vpack_VwVw_sat(Q6_Vw_equals_Vsf(b), Q6_Vw_equals_Vsf(a));
        }
    }
    for (unsigned r = valid; qrows != 1 && r < qrows; ++r)
      for (unsigned c = 0; c < 4; ++c)
        reinterpret_cast<HVX_Vector*>(y + (head * qrows + r) * 256)[c] =
            Q6_Vh_vsplat_R(oq.zero_offset);
  }
done:
  if (dma_pending) {
    if (!dma_wait())
      __atomic_store_n(&queue.error, 1u, __ATOMIC_RELEASE);
    dma_pending = false;
  }
  __atomic_store_n(&queue.jobs[slice].finished, 1u, __ATOMIC_RELEASE);
  return QHPI_SUCCESS;
#else
  return QHPI_UNSUPPORTED;
#endif
}

static uint32_t reserve_scratch(
    QHPI_RuntimeHandle*,
    uint32_t,
    QHPI_Tensor**,
    uint32_t,
    const QHPI_Tensor* const*) {
  return QHPI_SUCCESS;
}
static uint32_t inline_execute(
    QHPI_RuntimeHandle* handle,
    uint32_t no,
    QHPI_Tensor** out,
    uint32_t ni,
    const QHPI_Tensor* const* in) {
#ifdef __hexagon__
  if (no != 1 || ni != 9)
    return QHPI_ERROR_FATAL;
  const auto shape = qhpi_tensor_shape(in[1]);
  const unsigned qheads = shape.dims[1],
                 kvheads = qhpi_tensor_shape(in[3]).dims[1];
  const unsigned kv_bytes = kvheads * kv_rows * 256 * 2;
  const QHPI_Tensor* original[] = {
      in[1], in[2], in[3], in[4], in[5], in[6], in[7]};
  Configuration config{};
  if (!configuration(7, original, config))
    return QHPI_UNSUPPORTED;
  const auto oq_check = qhpi_tensor_quant_parameters(out[0]);
  if (!shape_is(
          qhpi_tensor_shape(out[0]),
          config.geometry.batch,
          config.geometry.heads,
          config.geometry.queries,
          config.geometry.dimension) ||
      !Quantization{oq_check.stepsize, oq_check.zero_offset}.valid())
    return QHPI_UNSUPPORTED;
  if (!config.fast() || !QCOM_HTP_V81)
    return generic_execute(handle, out, in, config);

  const auto resource = qhpi_thread_resources(handle);
  if (resource != 0 && resource != QHPI_RESOURCE_HVX)
    return QHPI_SUCCESS;
  const auto* descriptor =
      static_cast<const int32_t*>(qhpi_tensor_raw_data(in[2]));
  if (!cache_pointer(descriptor))
    return QHPI_ERROR_FATAL;
  auto* cached = packed_cache(descriptor[0], kv_bytes);
  if (!cached || !cached->packed)
    return QHPI_ERROR_FATAL;
  auto& sync = cached->inline_sync;
  const auto* l = static_cast<const int32_t*>(qhpi_tensor_raw_data(in[6]));
  const auto* meta = static_cast<const float*>(qhpi_tensor_raw_data(in[7]));
  const auto kq = qhpi_tensor_quant_parameters(in[3]),
             vq = qhpi_tensor_quant_parameters(in[5]);
  if (resource == QHPI_RESOURCE_HVX)
    __atomic_store_n(&sync.expected, qhpi_num_slices(handle), __ATOMIC_RELEASE);
  const bool short_direct = l[0] == 1 && l[1] == 0 && l[2] >= 0 &&
      unsigned(l[2] + 1) <= short_key_limit && qheads == 16 &&
      (unsigned(descriptor[1]) & exact_append_flag) && kq.stepsize == meta[9] &&
      kq.zero_offset == int(meta[10]) && vq.stepsize == meta[11] &&
      vq.zero_offset == int(meta[12]) &&
      (l[2] == 0 ||
       (cached->epoch == uint32_t(descriptor[2]) && cached->short_dense &&
        cached->last_exact &&
        __atomic_load_n(&cached->history, __ATOMIC_ACQUIRE) == unsigned(l[2])));
  if (short_direct) {
    if (resource == QHPI_RESOURCE_HVX) {
      const auto slice = qhpi_slice_number(handle);
      const unsigned slices = qhpi_num_slices(handle);
      if (!slices || slices > workers || slice >= slices) {
        __atomic_store_n(&sync.status, QHPI_UNSUPPORTED, __ATOMIC_RELEASE);
        __atomic_add_fetch(&sync.departed, 1u, __ATOMIC_RELEASE);
        return QHPI_UNSUPPORTED;
      }
      for (unsigned shard = slice; shard < workers; shard += slices)
        short_attention(
            static_cast<const uint16_t*>(qhpi_tensor_raw_data(in[1])),
            static_cast<uint16_t*>(qhpi_tensor_raw_data(out[0])),
            cached->packed,
            cached->packed + kv_bytes,
            unsigned(l[2] + 1),
            shard,
            static_cast<uint8_t*>(qhpi_tensor_raw_data(in[0])),
            qhpi_tensor_quant_parameters(in[1]),
            qhpi_tensor_quant_parameters(out[0]),
            meta,
            static_cast<const uint16_t*>(qhpi_tensor_raw_data(in[3])),
            static_cast<const uint16_t*>(qhpi_tensor_raw_data(in[5])),
            kq,
            vq);
      __atomic_add_fetch(&sync.departed, 1u, __ATOMIC_RELEASE);
      return QHPI_SUCCESS;
    }
    while (!__atomic_load_n(&sync.expected, __ATOMIC_ACQUIRE) ||
           __atomic_load_n(&sync.departed, __ATOMIC_ACQUIRE) !=
               __atomic_load_n(&sync.expected, __ATOMIC_ACQUIRE))
      attention_hmx::pause();
    const auto status = __atomic_load_n(&sync.status, __ATOMIC_ACQUIRE);
    if (status == QHPI_SUCCESS) {
      cached->short_dense = true;
      cached->last_exact = true;
      cached->last_valid = 1;
      cached->epoch = uint32_t(descriptor[2]);
      __atomic_store_n(&cached->history, unsigned(l[2] + 1), __ATOMIC_RELEASE);
    }
    sync.prepared = sync.ready = sync.departed = sync.status = sync.expected =
        0;
    return status;
  }

  QHPI_Tensor* storage[] = {
      const_cast<QHPI_Tensor*>(in[0]), const_cast<QHPI_Tensor*>(in[8])};
  const QHPI_Tensor* args[] = {in[1], in[2], in[3], in[4], in[5], in[6], in[7]};
  if (resource == QHPI_RESOURCE_HVX) {
    auto status = initialize(handle, 2, storage, 7, args);
    if (status != QHPI_SUCCESS)
      __atomic_store_n(&sync.status, status, __ATOMIC_RELEASE);
    __atomic_add_fetch(&sync.prepared, 1u, __ATOMIC_RELEASE);
    while (!__atomic_load_n(&sync.ready, __ATOMIC_ACQUIRE))
      attention_hmx::pause();
    status = __atomic_load_n(&sync.status, __ATOMIC_ACQUIRE);
    if (status == QHPI_SUCCESS)
      status = execute(handle, 1, out, 9, in);
    if (status != QHPI_SUCCESS)
      __atomic_store_n(&sync.status, status, __ATOMIC_RELEASE);
    __atomic_add_fetch(&sync.departed, 1u, __ATOMIC_RELEASE);
    return status;
  }
  while (!__atomic_load_n(&sync.expected, __ATOMIC_ACQUIRE) ||
         __atomic_load_n(&sync.prepared, __ATOMIC_ACQUIRE) !=
             __atomic_load_n(&sync.expected, __ATOMIC_ACQUIRE))
    attention_hmx::pause();
  auto status = __atomic_load_n(&sync.status, __ATOMIC_ACQUIRE);
  __atomic_store_n(&sync.ready, 1u, __ATOMIC_RELEASE);
  if (status == QHPI_SUCCESS)
    status = execute(handle, 1, out, 9, in);
  while (__atomic_load_n(&sync.departed, __ATOMIC_ACQUIRE) !=
         __atomic_load_n(&sync.expected, __ATOMIC_ACQUIRE))
    attention_hmx::pause();
  const auto worker_status = __atomic_load_n(&sync.status, __ATOMIC_ACQUIRE);
  if (status == QHPI_SUCCESS)
    status = worker_status;
  sync.prepared = sync.ready = sync.departed = sync.status = sync.expected = 0;
  return status;
#else
  return QHPI_UNSUPPORTED;
#endif
}

#define S(T, W) \
  { T, QHPI_LAYOUT_FLAT_4, QHPI_STORAGE_DIRECT, W }
#define D(T) S(T, QHPI_MEM_LOC_DDR_ONLY)
#define V(T) S(T, QHPI_MEM_LOC_TCM_ONLY)
static QHPI_Tensor_Signature_v1 init_in[] = {
    D(QHPI_QUINT16),
    D(QHPI_INT32),
    D(QHPI_QUINT16),
    D(QHPI_INT32),
    D(QHPI_QUINT16),
    D(QHPI_INT32),
    D(QHPI_FLOAT32)};
static QHPI_Tensor_Signature_v1 init_out[] = {V(QHPI_FLOAT16), D(QHPI_FLOAT16)};
static QHPI_Tensor_Signature_v1 exec_in[] = {
    V(QHPI_FLOAT16),
    D(QHPI_QUINT16),
    D(QHPI_INT32),
    D(QHPI_QUINT16),
    D(QHPI_INT32),
    D(QHPI_QUINT16),
    D(QHPI_INT32),
    D(QHPI_FLOAT32),
    D(QHPI_FLOAT16)};
static QHPI_Tensor_Signature_v1 exec_out[] = {D(QHPI_QUINT16)};
static QHPI_Tensor_Signature_v1 decode_init_out[] = {
    V(QHPI_FLOAT16),
    V(QHPI_FLOAT16)};
static QHPI_Tensor_Signature_v1 decode_init_in[] = {
    V(QHPI_QUINT16),
    D(QHPI_INT32),
    V(QHPI_QUINT16),
    D(QHPI_INT32),
    V(QHPI_QUINT16),
    D(QHPI_INT32),
    D(QHPI_FLOAT32)};
static QHPI_Tensor_Signature_v1 decode_exec_in[] = {
    V(QHPI_FLOAT16),
    V(QHPI_QUINT16),
    D(QHPI_INT32),
    V(QHPI_QUINT16),
    D(QHPI_INT32),
    V(QHPI_QUINT16),
    D(QHPI_INT32),
    D(QHPI_FLOAT32),
    V(QHPI_FLOAT16)};
static QHPI_Tensor_Signature_v1 decode_exec_out[] = {V(QHPI_QUINT16)};
static uint32_t
init_decode(const QHPI_Op*, uint32_t n, const QHPI_Tensor* const* in) {
  Geometry g{};
  return n == 7 &&
      geometry(
             qhpi_tensor_shape(in[0]),
             qhpi_tensor_shape(in[2]),
             qhpi_tensor_shape(in[4]),
             g) &&
      g.fast() && g.queries == 1;
}
static uint32_t
exec_decode(const QHPI_Op*, uint32_t n, const QHPI_Tensor* const* in) {
  Geometry g{};
  return n == 9 &&
      geometry(
             qhpi_tensor_shape(in[1]),
             qhpi_tensor_shape(in[3]),
             qhpi_tensor_shape(in[5]),
             g) &&
      g.fast() && g.queries == 1;
}
static uint32_t
init_prefill(const QHPI_Op* op, uint32_t n, const QHPI_Tensor* const* in) {
  return !init_decode(op, n, in);
}
static uint32_t
exec_prefill(const QHPI_Op* op, uint32_t n, const QHPI_Tensor* const* in) {
  return !exec_decode(op, n, in);
}
#define K(NAME, FN, RES, MT, NUM, NOUT, IS, OS, PRED)                       \
  {                                                                         \
    .function_name = "DeltaNetOpPackage::" NAME, .function = FN,            \
    .resources = RES, .source_destructive = false, .multithreaded = MT,     \
    .variable_inputs = false, .variable_outputs = false, .min_inputs = NUM, \
    .input_signature = IS, .min_outputs = NOUT, .output_signature = OS,     \
    .sync_block_size = 0, .predicate = PRED                                 \
  }
static QHPI_Kernel_v1 init_k[] = {
    K("AttentionInitialize.decode_scratch",
      reserve_scratch,
      0,
      false,
      7,
      2,
      decode_init_in,
      decode_init_out,
      init_decode),
    K("AttentionInitialize.hvx",
      initialize,
      QHPI_RESOURCE_HVX,
      true,
      7,
      2,
      init_in,
      init_out,
      init_prefill)};
static QHPI_Kernel_v1 exec_k[] = {
    {.function_name = "DeltaNetOpPackage::AttentionExecute.decode_inline",
     .function = inline_execute,
     .resources = QHPI_RESOURCE_EXCLUSIVE,
     .source_destructive = false,
     .multithreaded = true,
     .variable_inputs = false,
     .variable_outputs = false,
     .min_inputs = 9,
     .input_signature = decode_exec_in,
     .min_outputs = 1,
     .output_signature = decode_exec_out,
     .sync_block_size = 0,
     .predicate = exec_decode},
    K("AttentionExecute.exclusive",
      execute,
      QHPI_RESOURCE_EXCLUSIVE,
      true,
      9,
      1,
      exec_in,
      exec_out,
      exec_prefill)};
static QHPI_OpInfo_v1 ops[] = {
    {.name = "DeltaNetOpPackage::AttentionValid",
     .num_kernels = 0,
     .kernels = nullptr,
     .early_rewrite = rewrite},
    {.name = "DeltaNetOpPackage::AttentionInitialize",
     .num_kernels = 2,
     .kernels = init_k},
    {.name = "DeltaNetOpPackage::AttentionExecute",
     .num_kernels = 2,
     .kernels = exec_k}};
} // namespace attention
void register_attention_qhpi_ops() {
  qhpi_register_ops_v1(3, attention::ops, "DeltaNetOpPackage");
}

extern "C" void release_attention_cache_maps() {
#ifdef __hexagon__
  auto& maps = attention::cache_mappings();
  qurt_pimutex_lock(&maps.lock);
  auto* slot = maps.entries;
  __atomic_store_n(
      &maps.entries,
      static_cast<attention::CacheMappings::Slot*>(nullptr),
      __ATOMIC_RELEASE);
  while (slot) {
    auto* next = slot->next;
    if (slot->mapped_by_get)
      HAP_mmap_put(slot->fd);
    else
      HAP_munmap(slot->p, slot->bytes);
    free(slot->packed);
    slot->~Slot();
    free(slot);
    slot = next;
  }
  qurt_pimutex_unlock(&maps.lock);
#endif
}

#else
void register_attention_qhpi_ops() {}
extern "C" void release_attention_cache_maps() {}
#endif
