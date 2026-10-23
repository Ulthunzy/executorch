#include "integer_moe.h"
#include "HTP/core/qhpi.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

// AR1 专家计算：HVX 负责 A8 打包、N2 展开及 SwiGLU，单个 HMX worker
// 执行矩阵乘。 专家结果按原 Top8 rank 存放，读取/完成次序不会改变最终求和顺序。
#ifdef __hexagon__
#include "hexagon_protos.h"
#include "hvx_hexagon_protos.h"
#include "hmx_integer.h"
#include "hvx-sigmoid.h"
#include "HAP_mem.h"
#include "qurt_memory.h"
#include "HAP_perf.h"
#include "qurt_pimutex.h"
#define vmem(p)                    \
  (*reinterpret_cast<HVX_Vector*>( \
      const_cast<void*>(static_cast<const void*>(p))))
#include "../N2Utils.h"
constexpr int kWeightCompression = SM_EXPERT_BITS == 2 ? 2 : 1;

// 每个 HVX worker 向 HMX 提交一个 tile 批次：ready 0=空闲、1=待计算、2=已完成。
// a/w/bias/raw 都指向本次获得的 VTCM 工作区，不能跨资源归还保留其中内容。

uint64_t metrics[12][8];

struct alignas(128) DmaDescriptor {
  void* next;
  volatile uint32_t control;
  uint32_t src, dst;
};

static void dma_start(
    DmaDescriptor& descriptor,
    void* destination,
    const void* source,
    uint32_t bytes) {
  descriptor.next = nullptr;
  descriptor.control = bytes | (1u << 28) | (1u << 29);
  descriptor.src = uintptr_t(source);
  descriptor.dst = uintptr_t(destination);
  asm volatile("release(%0):at" ::"r"(&descriptor) : "memory");
  asm volatile("dmstart(%0)" ::"r"(&descriptor) : "memory");
}

static void dma_wait(DmaDescriptor& descriptor) {
  while (!(descriptor.control >> 31)) {
    uint32_t value;
    asm volatile("%0 = dmwait" : "=r"(value)::"memory");
  }
  asm volatile("" ::: "memory");
}

static void dma_copy(void* destination, const void* source, uint32_t bytes) {
  DmaDescriptor descriptor{};
  dma_start(descriptor, destination, source, bytes);
  dma_wait(descriptor);
}

// 一次调用内的 HVX/HMX 协作屏障；最后由 HMX 在所有 worker 离开后清零。

static void await(const volatile uint32_t* p, uint32_t value) {
  while (read(p) != value)
    asm volatile("pause(#32)");
}

// 按 group 的 absmax 动态生成 A8 scale；存储 code 加 128，bias 中补偿该零点。
static void prepare_activations(
    uint8_t* packed,
    float* scales,
    const float* source,
    int stride,
    const int* tokens,
    int rows,
    int width,
    int grid,
    bool linear = false) {
  for (int m = 0; m < rows; m++)
    for (int g = 0; g < (width + grid - 1) / grid; g++) {
      const float* row = source + tokens[m] * stride + g * grid;
      const int count = std::min(grid, width - g * grid);
      auto max_vector = Q6_V_vzero();
      const auto abs_mask = Q6_V_vsplat_R(0x7fffffff);
      for (int k = 0; k < count; k += 32)
        max_vector = Q6_Vsf_vmax_VsfVsf(
            max_vector, Q6_V_vand_VV(vmem(row + k), abs_mask));
      for (int shift = 64; shift >= 4; shift /= 2)
        max_vector =
            Q6_Vsf_vmax_VsfVsf(max_vector, Q6_V_vror_VR(max_vector, shift));

      union {
        uint32_t i;
        float f;
      } maximum_bits{uint32_t(Q6_R_vextract_VR(max_vector, 0))};

      const float maximum = maximum_bits.f;
      const float scale = maximum > 0 ? maximum / 127.f : 1.f;
      scales[m * 2 + g] = scale;
      const auto inverse = hvx_vec_splat_f32(1.f / scale);
      for (int k = 0; k < count; k += 32) {
        const auto value = Q6_Vsf_vmpy_VsfVsf(vmem(row + k), inverse);
        const auto negative = Q6_Q_vcmp_gt_VsfVsf(Q6_V_vzero(), value);
        const auto half = Q6_V_vmux_QVV(
            negative, hvx_vec_splat_f32(-.5f), hvx_vec_splat_f32(.5f));
        auto code = Q6_Vw_equals_Vsf(Q6_Vsf_vadd_VsfVsf(value, half));
        code = Q6_Vw_vmax_VwVw(code, Q6_V_vsplat_R(-127));
        code = Q6_Vw_vmin_VwVw(code, Q6_V_vsplat_R(127));
        code = Q6_Vw_vadd_VwVw(code, Q6_V_vsplat_R(128));
        if (linear) {
          vmem(packed + (m * width + g * grid + k) * 4) = code;
          continue;
        }
        auto* target = packed + (g * grid + k) / 32 * 2048 + (m / 2) * 128;
        if (m % 2)
          vmem(target) = Q6_V_vor_VV(vmem(target), Q6_Vw_vasl_VwR(code, 24));
        else
          vmem(target) = Q6_Vw_vasl_VwR(code, 8);
      }
    }
}

// GU/down 共用投影流水线：双缓冲 DMA→N2 展开→提交 HMX→恢复 scale。
// GU 后半块到达时就地合成 SiLU(gate)*up；down 再乘对应路由分数。
static void projection(
    Request* request,
    const uint8_t* codes,
    const float* scales,
    const int16_t* sums,
    int n,
    int padded_k,
    int grid,
    int rows,
    const float* activation_scales,
    float* destination,
    int destination_stride,
    bool routed,
    const int* token,
    const int* slot,
    const float* scores) {
  const int groups = (padded_k + grid - 1) / grid;
  const int partials = padded_k / 128;
  const float factor = grid == 1024 ? 32.f : 16.f;
  request->k = std::min(grid, padded_k);
  request->groups = groups;
  auto* weight_buffers = request->w;
  constexpr int transfer_bytes = 65536;
  const int block_bytes = padded_k * 16,
            blocks_per_transfer = transfer_bytes / block_bytes;
  // 每块展开后最多 64 KiB；预读下一块时仍可计算当前块。
  DmaDescriptor transfer[2]{};
  dma_start(
      transfer[0],
      weight_buffers,
      codes,
      std::min(transfer_bytes, n / 32 * block_bytes) / kWeightCompression);
  for (int nb = 0; nb < n / 32;) {
    auto stamp = tick();
    const int batch = nb / blocks_per_transfer;
    if (nb % blocks_per_transfer == 0) {
      dma_wait(transfer[batch % 2]);
      metrics[11][request->worker] += tick() - stamp;
      auto expand_stamp = tick();
#if SM_EXPERT_BITS == 2
      n2_expand_in_place(
          weight_buffers + (batch % 2) * transfer_bytes,
          std::min(transfer_bytes, (n / 32 - nb) * block_bytes) / 2);
#endif
      metrics[request->worker][6] += tick() - expand_stamp;
      if (nb + blocks_per_transfer < n / 32)
        dma_start(
            transfer[(batch + 1) % 2],
            weight_buffers + ((batch + 1) % 2) * transfer_bytes,
            codes +
                size_t(nb + blocks_per_transfer) * block_bytes /
                    kWeightCompression,
            std::min(
                transfer_bytes,
                (n / 32 - nb - blocks_per_transfer) * block_bytes) /
                kWeightCompression);
    }
    request->w = weight_buffers + (batch % 2) * transfer_bytes +
        (nb % blocks_per_transfer) * block_bytes;
    const int blocks = std::min(
        {int(SM_MOE_HMX_BATCH),
         blocks_per_transfer - nb % blocks_per_transfer,
         n / 32 - nb});
    request->nblocks = blocks;
    request->block_bytes = block_bytes;
    metrics[request->worker][1] += tick() - stamp;
    stamp = tick();
    for (int b = 0; b < blocks; b++)
      for (int g = 0; g < groups; g++) {
        uint32_t* bias = request->bias + (b * groups + g) * 64;
        auto sum = Q6_V_vzero();
        for (int p = g * grid / 128;
             p < std::min((g + 1) * grid / 128, partials);
             p++) {
          auto pair = Q6_Ww_vunpack_Vh(
              hvx_vmemu(sums + ((nb + b) * partials + p) * 32));
          sum = Q6_Vw_vadd_VwVw(sum, Q6_V_lo_W(pair));
        }
        vmem(bias) = Q6_V_vsplat_R(grid == 1024 ? 0x2c00 : 0x3000);
        vmem(bias + 32) = Q6_Vw_vsub_VwVw(Q6_V_vzero(), Q6_Vw_vasl_VwR(sum, 7));
      }
    metrics[request->worker][2] += tick() - stamp;
    stamp = tick();
    write(&request->ready, 1);
    await(&request->ready, 2);
    metrics[request->worker][3] += tick() - stamp;
    stamp = tick();
    for (int b = 0; b < blocks; b++)
      for (int m = 0; m < rows; m++) {
        const int block = nb + b;
        HVX_Vector accum = Q6_V_vzero();
        for (int g = 0; g < groups; g++) {
          const auto raw =
              vmem(request->raw + (b * groups + g) * 1024 + (m / 2) * 64);
          const auto integers = Q6_Vw_vdmpy_VhRh_sat(raw, m % 2 ? 0x10000 : 1);
          const auto sw = vmem(scales + (block * groups + g) * 32);
          const auto sa =
              hvx_vec_splat_f32(factor * activation_scales[m * 2 + g]);
          auto v = Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_equals_Vw(integers), sw);
          v = Q6_Vsf_vmpy_VsfVsf(v, sa);
          accum = Q6_Vsf_vadd_VsfVsf(accum, v);
        }
        float* target;
        if (routed) {
#if SM_MOE_ACCUMULATE
          target = destination + size_t(m) * 2048 + block * 32;
#else
          target =
              destination + size_t(token[m] * 8 + slot[m]) * 2048 + block * 32;
#endif
          accum = Q6_Vsf_vmpy_VsfVsf(
              accum, hvx_vec_splat_f32(scores[token[m] * 8 + slot[m]]));
        } else {
          int column = block * 32;
          if (column >= n / 2) {
            column -= n / 2;
            const auto gate =
                vmem(destination + m * destination_stride + column);
            const auto sigmoid = hvx_vec_fast_sigmoid_f32_guard(
                gate,
                hvx_vec_splat_f32(1.f),
                hvx_vec_splat_f32(87.f),
                hvx_vec_splat_f32(-87.f));
            accum =
                Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_vmpy_VsfVsf(gate, sigmoid), accum);
          }
          target = destination + m * destination_stride + column;
        }
        vmem(target) = accum;
      }
    write(&request->ready, 0);
    metrics[request->worker][4] += tick() - stamp;
    nb += blocks;
  }
  request->w = weight_buffers;
}

static BankMapping selected_mappings[2], cached_mapping;

int release_expert_mappings() {
  int error = 0;
  for (auto& mapping : selected_mappings) {
    if (mapping.pointer && HAP_munmap(mapping.pointer, mapping.bytes))
      error = 52;
    mapping = BankMapping{};
  }
  if (cached_mapping.pointer &&
      HAP_munmap(cached_mapping.pointer, cached_mapping.bytes))
    error = 52;
  cached_mapping = BankMapping{};
  return error;
}

uint32_t execute_integer_moe(
    QHPI_RuntimeHandle* handle,
    uint32_t,
    QHPI_Tensor** outputs,
    uint32_t,
    const QHPI_Tensor* const* inputs) {
  auto* sync = static_cast<Sync*>(qhpi_sync_block(handle));
  auto* workspace = static_cast<uint8_t*>(qhpi_tensor_raw_data(outputs[1]));
  auto* route_output = static_cast<float*>(qhpi_tensor_raw_data(outputs[2]));
  const auto* descriptor =
      static_cast<const int32_t*>(qhpi_tensor_raw_data(inputs[4]));
  const auto* x = static_cast<const float*>(qhpi_tensor_raw_data(inputs[0]));
  const auto* ids_ddr =
      static_cast<const int32_t*>(qhpi_tensor_raw_data(inputs[1]));
  const auto* scores_ddr =
      static_cast<const float*>(qhpi_tensor_raw_data(inputs[2]));
  const auto* shared =
      static_cast<const float*>(qhpi_tensor_raw_data(inputs[3]));
  auto* output = static_cast<float*>(qhpi_tensor_raw_data(outputs[0]));
  if (!workspace || !route_output)
    return 21;
  const uint32_t resource = qhpi_thread_resources(handle);
  if (descriptor[2] == -1) {
    if (resource == QHPI_RESOURCE_HMX) {
      output[0] = qhpi_num_slices(handle);
      output[1] = qhpi_slice_number(handle);
      output[2] = resource;
    } else if (resource == QHPI_RESOURCE_HVX)
      output[4 + qhpi_slice_number(handle)] = qhpi_num_slices(handle);
    return 0;
  }
  // A phase mask selects original Top8 positions; retained outputs are summed
  // in rank order.
  const unsigned active_mask =
      descriptor[7] == 1 || descriptor[7] == 2 ? unsigned(descriptor[6]) : 255;
  const auto started = tick();
  if (resource == QHPI_RESOURCE_HMX) {
    while (!read(&sync->workers))
      asm volatile("pause(#32)");
    const uint32_t workers = read(&sync->workers);
    await(&sync->arrived, workers);
    metrics[10][0] = tick() - started;
    auto mapping_stamp = tick();
    BankMapping* mapping = nullptr;
    for (auto& m : selected_mappings)
      if (m.fd == descriptor[0]) {
        mapping = &m;
        break;
      }
    if (!mapping)
      for (auto& m : selected_mappings)
        if (m.fd == -1) {
          mapping = &m;
          break;
        }
    if (!mapping)
      sync->error = 24;
    else {
      if (mapping->fd == -1) {
        mapping->pointer = static_cast<uint8_t*>(HAP_mmap2(
            nullptr,
            descriptor[1],
            HAP_PROT_READ | HAP_PROT_WRITE,
            0,
            descriptor[0],
            0));
        mapping->fd = descriptor[0];
        mapping->bytes = descriptor[1];
      }
      if (!mapping->pointer || mapping->pointer == (void*)-1)
        sync->error = 25;
      else {
        // DMA reads bypass DSP caches. Only the global header/offset table is
        // read by the CPU.
        if (mapping->generation != descriptor[2])
          qurt_mem_cache_clean(
              uintptr_t(mapping->pointer),
              4096,
              QURT_MEM_CACHE_INVALIDATE,
              QURT_MEM_DCACHE);
        mapping->generation = descriptor[2];
        sync->bank = mapping->pointer;
        if (*reinterpret_cast<const uint32_t*>(sync->bank) !=
            (SM_EXPERT_BITS == 2 ? 0x4e32424b : 0x4e34424b))
          sync->error = 26;
      }
    }
    metrics[10][1] = tick() - mapping_stamp;
    mapping_stamp = tick();
    if (descriptor[4] >= 0 && !sync->cached_bank) {
      if (cached_mapping.fd != descriptor[4]) {
        if (cached_mapping.pointer)
          HAP_munmap(cached_mapping.pointer, cached_mapping.bytes);
        cached_mapping = {
            descriptor[4],
            0,
            static_cast<uint8_t*>(HAP_mmap2(
                nullptr,
                descriptor[5],
                HAP_PROT_READ | HAP_PROT_WRITE,
                0,
                descriptor[4],
                0)),
            unsigned(descriptor[5])};
      }
      if (!cached_mapping.pointer || cached_mapping.pointer == (void*)-1)
        sync->error = 25;
      sync->cached_bank = cached_mapping.pointer;
    }
    metrics[10][2] = tick() - mapping_stamp;
    metrics[9][1] = tick() - started;
    write(&sync->go, 1);
    // HMX 串行消费各 HVX worker 已发布的请求，不参与 UFS 读取等待。
    while (read(&sync->done) != workers)
      for (unsigned i = 0; i < workers; i++) {
        Request* r = &sync->requests[i];
        if (read(&r->ready) != 1)
          continue;
        const auto stamp = tick();
        for (unsigned b = 0; b < r->nblocks; b++)
          for (unsigned g = 0; g < r->groups; g++)
            integer_hmx(
                r->a + g * r->k * 64,
                r->w + b * r->block_bytes + g * r->k * 16,
                r->bias + (b * r->groups + g) * 64,
                r->raw + (b * r->groups + g) * 1024,
                r->k);
        metrics[9][0] += tick() - stamp;
        write(&r->ready, 2);
      }
    metrics[9][2] = tick() - started;
    write(&sync->go, 2);
    await(&sync->combined, workers);
    write(&sync->go, 3);
    await(&sync->departed, workers);
    const auto error = sync->error;
    memset(sync, 0, sizeof(*sync));
    return error;
  }
  if (resource != QHPI_RESOURCE_HVX)
    return 22;
  const uint32_t count = qhpi_num_slices(handle);
  write(&sync->workers, count);
  const uint32_t worker = increment(&sync->arrived);
  if (count > 8)
    return 23;
  auto* base = reinterpret_cast<uint8_t*>(
                   (uintptr_t(workspace) + 2047) & ~uintptr_t(2047)) +
      worker * 229376;
  uint8_t* packed = base;
  float* gu = reinterpret_cast<float*>(base + 32768);
  float* middle = gu;
  uint8_t* weights = base + 36864;
  uint32_t* bias = reinterpret_cast<uint32_t*>(base + 225280);
  uint16_t* raw = reinterpret_cast<uint16_t*>(base + 208896);
  float activation_scales[64] __attribute__((aligned(128)));
  Request* request = &sync->requests[worker];
  request->worker = worker;
  request->a = packed;
  request->w = weights;
  request->bias = bias;
  request->raw = raw;
  await(&sync->go, 1);
  auto* ids = reinterpret_cast<int32_t*>(workspace + 2037120);
  auto* scores = reinterpret_cast<float*>(workspace + 2037248);
  if (worker == 0) {
    dma_copy(ids, ids_ddr, 128);
    dma_copy(scores, scores_ddr, 128);
    write(&sync->inputs_ready, 1);
  } else
    await(&sync->inputs_ready, 1);
  const uint8_t* bank = sync->bank;
  const auto* offsets = reinterpret_cast<const uint64_t*>(bank + 32);
  auto* input_codes = workspace + 1966080;
  auto* input_scales = reinterpret_cast<float*>(workspace + 1970176);
  const int first_token = worker * 16,
            local_tokens = std::clamp(kTokens - first_token, 0, 16);
  int local[32];
  for (int i = 0; i < 32; i++)
    local[i] = i;
  // AR1 experts share the GU input tile; only their down-projection tiles are
  // private.
  auto prepared_at = tick();
  if (worker == 0)
    for (int i = 0; i < 8; i++)
      if (unsigned(ids[i]) >= 256)
        write(&sync->error, 31);
  if (local_tokens)
    dma_copy(packed, x + first_token * 2048, local_tokens * 2048 * 4);
  if (local_tokens)
    prepare_activations(
        packed,
        input_scales + first_token * 2,
        reinterpret_cast<float*>(packed),
        2048,
        local,
        local_tokens,
        2048,
        1024,
        true);
  for (int m = 0; m < local_tokens; m++)
    for (int k = 0; k < 2048; k += 64) {
      const auto first = vmem(packed + (m * 2048 + k) * 4),
                 second = vmem(packed + (m * 2048 + k + 32) * 4);
      vmem(input_codes + ((first_token + m) * 2048 + k) * 2) =
          Q6_Vh_vpacke_VwVw(second, first);
      auto* target = workspace + 1835008 + k / 32 * 2048;
      vmem(target) = Q6_Vw_vasl_VwR(first, 8);
      vmem(target + 2048) = Q6_Vw_vasl_VwR(second, 8);
    }
#if SM_MOE_ACCUMULATE
  for (int t = first_token; t < first_token + local_tokens; t++)
    for (int j = 0; j < 2048; j += 32)
      vmem(route_output + t * 2048 + j) = Q6_V_vzero();
#endif
  increment(&sync->prepared);
  await(&sync->prepared, count);
  metrics[worker][0] += tick() - prepared_at;
  auto error = read(&sync->error);
  while (!error) {
    const unsigned rank = increment(&sync->next);
    if (rank >= 8)
      break;
    if (!(active_mask & (1U << rank)))
      continue;
    const unsigned expert = ids[rank];
    int tokens[128], slots[128];
    const int total = 1;
    tokens[0] = 0;
    slots[0] = rank;
    if (sync->ready_words) {
      // 命中项初始已 ready，可立即执行；缺失项只等待自己的权重，不等齐 Top8。
      const auto wait_started = HAP_perf_get_qtimer_count();
      while (!(read(sync->ready_words + 32 * (slots[0] + 1)) & 1U)) {
        if (read(sync->ready_words + 1)) {
          write(&sync->error, 64);
          break;
        }
        if (HAP_perf_get_qtimer_count() - wait_started > 19200000ULL * 10) {
          write(&sync->error, 65);
          break;
        }
        asm volatile("pause(#64)");
      }
#if SM_MOE_PROFILE
      metrics[worker][5] += HAP_perf_get_qtimer_count() - wait_started;
#endif
      error = read(&sync->error);
      if (error)
        break;
    }
    const uint64_t expert_offset = offsets[expert];
    const uint8_t* expert_data = (expert_offset >> 63)
        ? sync->cached_bank + (expert_offset & ~(1ULL << 63))
        : bank + expert_offset;
    auto* header = reinterpret_cast<uint32_t*>(base + 227840);
    // 128 B 专家头：[1] intermediate，[3] payload 大小；
    // [4/5/6] 是 GU codes/scales/sums 偏移，[7/8/9] 是 down 的对应偏移。
    dma_copy(header, expert_data, 128);
    const int width = header[1], down_k = (width + 127) / 128 * 128;
    auto* gu_meta = base + 167936;
    auto* down_meta = gu_meta;
    for (int begin = 0; begin < total; begin += 32) {
      const int rows = std::min(32, total - begin);
      dma_copy(gu_meta, expert_data + header[5], header[7] - header[5]);
      auto stamp = tick();
      for (int m = 0; m < rows; m++) {
        const int token = tokens[begin + m];
        activation_scales[m * 2] = input_scales[token * 2];
        activation_scales[m * 2 + 1] = input_scales[token * 2 + 1];
      }
      metrics[worker][0] += tick() - stamp;
      request->a = workspace + 1835008;
      projection(
          request,
          expert_data + header[4],
          reinterpret_cast<const float*>(gu_meta),
          reinterpret_cast<const int16_t*>(gu_meta + header[6] - header[5]),
          2 * width,
          2048,
          1024,
          rows,
          activation_scales,
          gu,
          512,
          false,
          nullptr,
          nullptr,
          nullptr);
#if SM_MOE_STAGE_READY
      if (sync->ready_words) {
        const auto waited = HAP_perf_get_qtimer_count();
        while (!(read(sync->ready_words + 32 * (slots[0] + 1)) & 2U)) {
          if (read(sync->ready_words + 1)) {
            write(&sync->error, 64);
            break;
          }
          if (HAP_perf_get_qtimer_count() - waited > 19200000ULL * 10) {
            write(&sync->error, 65);
            break;
          }
          asm volatile("pause(#64)");
        }
#if SM_MOE_PROFILE
        metrics[worker][5] += HAP_perf_get_qtimer_count() - waited;
#endif
        error = read(&sync->error);
        if (error)
          break;
      }
#endif
      // GU consumes gate first, then overwrites it with SwiGLU when the up
      // block arrives. GU and down metadata have disjoint lifetimes and share
      // the same 40 KiB region.
      dma_copy(down_meta, expert_data + header[8], header[3] - header[8]);
      stamp = tick();
      for (int m = 0; m < rows; m++)
        memset(middle + m * 512 + width, 0, (512 - width) * 4);
      metrics[worker][0] += tick() - stamp;
      stamp = tick();
      request->a = packed;
      prepare_activations(
          packed, activation_scales, middle, 512, local, rows, down_k, 512);
      metrics[worker][0] += tick() - stamp;
#if SM_MOE_ACCUMULATE
      // AR1 accumulation uses private partials after the shared sum vector.
      auto* down_output = route_output + (worker + 1) * 2048;
#else
      auto* down_output = route_output;
#endif
      projection(
          request,
          expert_data + header[7],
          reinterpret_cast<const float*>(down_meta),
          reinterpret_cast<const int16_t*>(down_meta + header[9] - header[8]),
          2048,
          down_k,
          512,
          rows,
          activation_scales,
          down_output,
          2048,
          true,
          tokens + begin,
          slots + begin,
          scores);
#if SM_MOE_ACCUMULATE
      stamp = tick();
      for (int m = 0; m < rows; m++) {
        const int token = tokens[begin + m];
        qurt_pimutex_lock(sync->token_locks + token);
        for (int j = 0; j < 2048; j += 32)
          vmem(route_output + token * 2048 + j) = Q6_Vsf_vadd_VsfVsf(
              vmem(route_output + token * 2048 + j),
              vmem(down_output + m * 2048 + j));
        qurt_pimutex_unlock(sync->token_locks + token);
      }
      metrics[worker][6] += tick() - stamp;
#endif
    }
  }
  metrics[worker][7] = tick() - started;
  increment(&sync->done);
  await(&sync->go, 2);
  const auto combine_start = tick();
  error = read(&sync->error);
  // 固定 rank 顺序求和，最后只加一次 shared expert 和 residual。
  if (!error && descriptor[7] != 1)
    for (unsigned t = worker; t < kTokens; t += count)
      for (int j = 0; j < 2048; j += 32) {
#if SM_MOE_ACCUMULATE
        auto sum = vmem(route_output + t * 2048 + j);
#else
        auto sum = Q6_V_vzero();
        for (int s = 0; s < 8; s++)
          sum = Q6_Vsf_vadd_VsfVsf(
              sum, vmem(route_output + (t * 8 + s) * 2048 + j));
#endif
        auto result = Q6_Vsf_vadd_VsfVsf(sum, vmem(shared + t * 2048 + j));
        if (sync->residual)
          result =
              Q6_Vsf_vadd_VsfVsf(result, vmem(sync->residual + t * 2048 + j));
        vmem(output + t * 2048 + j) = result;
      }
  metrics[worker][6] += tick() - combine_start;
  increment(&sync->combined);
  await(&sync->go, 3);
  increment(&sync->departed);
  return error;
}
#else
uint32_t execute_integer_moe(
    QHPI_RuntimeHandle*,
    uint32_t,
    QHPI_Tensor** outputs,
    uint32_t,
    const QHPI_Tensor* const* inputs) {
  // Preparation-only host implementation; real expert arithmetic executes on
  // HTP.
  memcpy(
      qhpi_tensor_raw_data(outputs[0]),
      qhpi_tensor_raw_data(inputs[3]),
      1 * 2048 * 4);
  return 0;
}
#endif

#ifndef NATIVE_RPC
static float cost(uint32_t, const QHPI_Tensor* const*) {
  return 1000000.f;
}

static QHPI_Tensor_Signature_v1 input_signature[] = {
    {QHPI_Float32,
     QHPI_Layout_Flat4,
     QHPI_Storage_Direct,
     QHPI_MemLoc_DDR_Only},
    {QHPI_Int32, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_DDR_Only},
    {QHPI_Float32,
     QHPI_Layout_Flat4,
     QHPI_Storage_Direct,
     QHPI_MemLoc_DDR_Only},
    {QHPI_Float32,
     QHPI_Layout_Flat4,
     QHPI_Storage_Direct,
     QHPI_MemLoc_DDR_Only},
    {QHPI_Int32, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_DDR_Only},
};
static QHPI_Tensor_Signature_v1 output_signature[] = {
    {QHPI_Float32,
     QHPI_Layout_Flat4,
     QHPI_Storage_Direct,
     QHPI_MemLoc_DDR_Only},
    {QHPI_QUInt8, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_TCM_Only},
    {QHPI_Float32,
     QHPI_Layout_Flat4,
     QHPI_Storage_Direct,
     QHPI_MemLoc_DDR_Only},
};
static QHPI_Kernel_v1 kernel = {
    .function_name = "IntegerMoEOpPackage::ExpertN4.integer_hmx",
    .function = execute_integer_moe,
    .resources = QHPI_RESOURCE_EXCLUSIVE,
    .source_destructive = false,
    .multithreaded = false,
    .variable_inputs = false,
    .variable_outputs = false,
    .min_inputs = 5,
    .input_signature = input_signature,
    .min_outputs = 3,
    .output_signature = output_signature,
    .cost_function = cost,
    .sync_block_size = 1024,
};
static QHPI_OpInfo_v1 info = {
    .name = "IntegerMoEOpPackage::ExpertN4",
    .num_kernels = 1,
    .kernels = &kernel};

void register_integer_moe() {
  qhpi_register_ops_v1(1, &info, "IntegerMoEOpPackage");
}
#endif
