#include "integer_moe.h"
#include "expert_stream.h"
#include "HTP/core/qhpi.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

// 同层 token 按专家归并，HMX 始终以最多 32 行的 tile 计算。
// HVX 准备输入和后处理，HMX 集中执行矩阵乘；权重读取量由专家去重结果决定。
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

// HVX→HMX 单槽握手：ready 0=空闲、1=待计算、2=已完成，地址均属于当前 VTCM。

uint64_t metrics[10][8];
uint32_t execution_counts[4];

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

// 当前 batch 的协作屏障、路由分桶和计数；HMX 等全部 worker 退出后再复用。

static void await(const volatile uint32_t* p, uint32_t value) {
  while (read(p) != value)
    asm volatile("pause(#32)");
}

// group absmax 的动态 A8 网格；code 加 128 后打包，HMX bias 负责补偿零点。
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

// 双缓冲权重 tile，把下一块 DMA 与当前块 HMX 交叠；N2 仅在 VTCM 展开为 N4。
// 恢复投影实数尺度后，GU 合成 SwiGLU，down 直接写入加权路由结果。
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
  DmaDescriptor transfer[2]{};
  dma_start(
      transfer[0],
      weight_buffers,
      codes,
      std::min(transfer_bytes, n / 32 * block_bytes) / 2);
  for (int nb = 0; nb < n / 32;) {
    auto stamp = tick();
    const int batch = nb / blocks_per_transfer;
    if (nb % blocks_per_transfer == 0) {
      dma_wait(transfer[batch % 2]);
      n2_expand_in_place(
          weight_buffers + (batch % 2) * transfer_bytes,
          std::min(transfer_bytes, (n / 32 - nb) * block_bytes) / 2);
      if (nb + blocks_per_transfer < n / 32)
        dma_start(
            transfer[(batch + 1) % 2],
            weight_buffers + ((batch + 1) % 2) * transfer_bytes,
            codes + size_t(nb + blocks_per_transfer) * block_bytes / 2,
            std::min(
                transfer_bytes,
                (n / 32 - nb - blocks_per_transfer) * block_bytes) /
                2);
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

// Only the current call uses its bank. Two DSP VA windows suffice while the
// 40-layer ARM cache keeps every resident payload.
static BankMapping mappings[2];

int release_banks() {
  int error = 0;
  for (auto& m : mappings) {
    if (m.pointer && HAP_munmap(m.pointer, m.bytes))
      error = 52;
    m = BankMapping{};
  }
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
  const bool streaming = descriptor[4] == expert_stream::kPhase;
  const int phase = streaming ? 0 : descriptor[4];
  const int tokens_count = expert_stream::token_count(descriptor[3]);
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
  const auto started = tick();
  if (resource == QHPI_RESOURCE_HMX) {
    while (!read(&sync->workers))
      asm volatile("pause(#32)");
    const uint32_t workers = read(&sync->workers);
    await(&sync->arrived, workers);
    BankMapping* mapping = nullptr;
    for (auto& m : mappings)
      if (m.fd == descriptor[0]) {
        mapping = &m;
        break;
      }
    if (!mapping)
      for (auto& m : mappings)
        if (m.fd == -1) {
          mapping = &m;
          break;
        }
    if (!mapping) {
      auto& victim = mappings[unsigned(descriptor[3]) % 2];
      if (victim.pointer && HAP_munmap(victim.pointer, victim.bytes))
        sync->error = 52;
      else {
        victim = BankMapping{};
        mapping = &victim;
      }
    }
    if (!mapping)
      sync->error = 24;
    else {
      if (mapping->fd == -1) {
        auto* pointer = static_cast<uint8_t*>(HAP_mmap2(
            nullptr,
            descriptor[1],
            HAP_PROT_READ | HAP_PROT_WRITE,
            0,
            descriptor[0],
            0));
        if (pointer && pointer != reinterpret_cast<void*>(-1))
          *mapping = {descriptor[0], -1, pointer, unsigned(descriptor[1])};
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
        if (*reinterpret_cast<const uint32_t*>(sync->bank) != 0x4e32424b)
          sync->error = 26;
      }
    }
    metrics[9][1] = tick() - started;
    write(&sync->go, 1);
    // HMX 消费 HVX 发布的 tile 请求；每个 worker 只有一个未完成请求。
    bool io_blocked = false;
    uint64_t blocked_since = 0;
    while (read(&sync->done) != workers) {
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
#if SM_MOE_PROFILE
      const bool blocked = (read(&sync->waiting) | read(&sync->done_mask)) ==
          (1u << workers) - 1;
      if (blocked != io_blocked) {
        const auto now = tick();
        if (io_blocked)
          metrics[9][6] += now - blocked_since;
        else
          blocked_since = now;
        io_blocked = blocked;
      }
#endif
    }
#if SM_MOE_PROFILE
    if (io_blocked)
      metrics[9][6] += tick() - blocked_since;
    for (unsigned i = 0; i < workers; i++)
      metrics[9][7] += sync->ready_wait[i];
#endif
    metrics[9][2] = tick() - started;
    write(&sync->go, 2);
    await(&sync->combined, workers);
    write(&sync->go, 3);
    await(&sync->departed, workers);
    if (!sync->error &&
        (sync->executed != unsigned(descriptor[5]) ||
         sync->assignments != unsigned(descriptor[6])))
      sync->error = 53;
    execution_counts[0]++;
    execution_counts[1] += sync->executed;
    execution_counts[2] += sync->assignments;
    execution_counts[3] += (phase == 0 || phase == 3) ? tokens_count / 128 : 0;
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
      worker * 458752;
  uint8_t* packed = base;
  float* gu = reinterpret_cast<float*>(base + 131072);
  float* middle = gu;
  uint8_t* weights = base + 196608;
  uint32_t* bias = reinterpret_cast<uint32_t*>(base + 385024);
  uint16_t* raw = reinterpret_cast<uint16_t*>(base + 368640);
  float activation_scales[64] __attribute__((aligned(128)));
  Request* request = &sync->requests[worker];
  request->worker = worker;
  request->a = packed;
  request->w = weights;
  request->bias = bias;
  request->raw = raw;
  await(&sync->go, 1);
  auto* ids = reinterpret_cast<int32_t*>(base + 389248);
  auto* scores = reinterpret_cast<float*>(base + 393344);
  // AR128 uses worker-local routing copies; larger groups use the mapped
  // read-only tables. Each worker retains the same fixed tile workspace.
  if (tokens_count == 128) {
    dma_copy(ids, ids_ddr, 4096);
    dma_copy(scores, scores_ddr, 4096);
  }
  if (tokens_count > 128)
    qurt_mem_cache_clean(
        uintptr_t(scores_ddr),
        tokens_count * 8 * 4,
        QURT_MEM_CACHE_INVALIDATE,
        QURT_MEM_DCACHE);
  const float* all_scores = tokens_count == 128 ? scores : scores_ddr;
  const uint8_t* bank = sync->bank;
  const auto* offsets = reinterpret_cast<const uint64_t*>(bank + 32);
  auto* input_codes = workspace + 3670016;
  auto* input_scales = reinterpret_cast<float*>(workspace + 447488);
  const int first_token = worker * (tokens_count / count);
  int local[32];
  for (int i = 0; i < 32; i++)
    local[i] = i;
  // Worker 0 padding holds the shared CSR (up to 8192 assignments). Its end
  // at 414976 precedes the per-worker stream scratch at 415104.
  auto* route_offsets = reinterpret_cast<uint32_t*>(workspace + 397440);
  auto* assignments = reinterpret_cast<uint16_t*>(workspace + 398592);
  auto prepared_at = tick();
  if (worker == 0) {
    unsigned counts[256]{}, cursor[256];
    if (tokens_count > 128)
      qurt_mem_cache_clean(
          uintptr_t(ids_ddr),
          tokens_count * 8 * 4,
          QURT_MEM_CACHE_INVALIDATE,
          QURT_MEM_DCACHE);
    const auto* all_ids = tokens_count == 128 ? ids : ids_ddr;
    for (int i = 0; i < tokens_count * 8; i++) {
      if (unsigned(all_ids[i]) >= 256)
        write(&sync->error, 31);
      else
        counts[all_ids[i]]++;
    }
    route_offsets[0] = 0;
    for (int e = 0; e < 256; e++) {
      route_offsets[e + 1] = route_offsets[e] + counts[e];
      cursor[e] = route_offsets[e];
      if (counts[e] > unsigned(tokens_count))
        write(&sync->error, 31);
    }
    for (int i = 0; i < tokens_count * 8; i++)
      if (unsigned(all_ids[i]) < 256)
        assignments[cursor[all_ids[i]]++] = i;
  }
  for (int first = first_token; first < first_token + int(tokens_count / count);
       first += 16) {
    dma_copy(packed, x + first * 2048, 16 * 2048 * 4);
    prepare_activations(
        packed,
        input_scales + first * 2,
        reinterpret_cast<float*>(packed),
        2048,
        local,
        16,
        2048,
        1024,
        true);
    for (int m = 0; m < 16; m++)
      for (int k = 0; k < 2048; k += 64) {
        const auto first_code = vmem(packed + (m * 2048 + k) * 4),
                   second_code = vmem(packed + (m * 2048 + k + 32) * 4);
        vmem(input_codes + ((first + m) * 2048 + k) * 2) =
            Q6_Vh_vpacke_VwVw(second_code, first_code);
      }
  }
#if SM_MOE_ACCUMULATE
  for (int t = first_token; t < first_token + 16; t++)
    for (int j = 0; j < 2048; j += 32)
      vmem(route_output + t * 2048 + j) = Q6_V_vzero();
#endif
  increment(&sync->prepared);
  await(&sync->prepared, count);
  metrics[worker][0] += tick() - prepared_at;
  auto error = read(&sync->error);
  auto* expert_order = reinterpret_cast<uint16_t*>(base + 415104);
  auto* ready = reinterpret_cast<uint32_t*>(base + 415616);
  if (streaming && !error) {
    dma_copy(expert_order, bank + expert_stream::kOrderOffset, 512);
    dma_copy(ready, bank + expert_stream::kReadyOffset, 1024);
  }
  while (!error && !read(&sync->error)) {
    const unsigned index = increment(&sync->next);
    if (index >= (streaming ? unsigned(descriptor[5]) : 256u))
      break;
    const unsigned expert = streaming ? expert_order[index] : index;
    if (expert >= 256) {
      error = 58;
      write(&sync->error, error);
      break;
    }
    if (streaming) {
      // Each reader publishes only its own payload. Ready experts come first;
      // a slow miss holds neither the other HVX workers nor completed results.
      const auto deadline = HAP_perf_get_qtimer_count() + 5 * 19200000ull;
#if SM_MOE_PROFILE
      const auto waiting_at = tick();
      if (!ready[expert])
        __atomic_fetch_or(&sync->waiting, 1u << worker, __ATOMIC_ACQ_REL);
#endif
      while (!ready[expert] && !read(&sync->error)) {
        const unsigned line = expert & ~31u;
        dma_copy(
            ready + line,
            bank + expert_stream::kReadyOffset + line * sizeof(uint32_t),
            128);
        if (HAP_perf_get_qtimer_count() > deadline) {
          error = 59;
          break;
        }
        asm volatile("pause(#32)");
      }
#if SM_MOE_PROFILE
      __atomic_fetch_and(&sync->waiting, ~(1u << worker), __ATOMIC_ACQ_REL);
      sync->ready_wait[worker] += tick() - waiting_at;
#endif
      if (error || ready[expert] != expert_stream::kReady) {
        error = error ? error : 60;
        write(&sync->error, error);
        break;
      }
    }
    int tokens[1024], slots[1024];
    // 同一个专家的 token 连续处理，复用同一权重；offset 为 0 表示不属于当前批。
    const int total = route_offsets[expert + 1] - route_offsets[expert];
    if (!total || !offsets[expert])
      continue;
    increment(&sync->executed);
    __atomic_fetch_add(&sync->assignments, unsigned(total), __ATOMIC_ACQ_REL);
    for (int i = 0; i < total; i++) {
      const int assignment = assignments[route_offsets[expert] + i];
      tokens[i] = assignment >> 3;
      slots[i] = assignment & 7;
    }
    const uint8_t* expert_data = bank + offsets[expert];
    auto* header = reinterpret_cast<uint32_t*>(base + 389120);
    // 专家头与 prepare_experts.py 的 12×UINT32 ABI 一致；
    // GU 的 codes/scales/sums 在 [4/5/6]，down 在
    // [7/8/9]，偏移以专家起点为基准。
    dma_copy(header, expert_data, 128);
    const int width = header[1], down_k = (width + 127) / 128 * 128;
    auto* gu_meta = base + 327680;
    auto* down_meta = gu_meta;
    for (int begin = 0; begin < total; begin += 32) {
      const int rows = std::min(32, total - begin);
      dma_copy(gu_meta, expert_data + header[5], header[7] - header[5]);
      auto stamp = tick();
      for (int m = 0; m < rows; m++) {
        const int token = tokens[begin + m];
        activation_scales[m * 2] = input_scales[token * 2];
        activation_scales[m * 2 + 1] = input_scales[token * 2 + 1];
        for (int k = 0; k < 2048; k += 64) {
          const auto pair =
              Q6_Ww_vunpack_Vh(vmem(input_codes + (token * 2048 + k) * 2));
          auto* target = packed + k / 32 * 2048 + (m / 2) * 128;
          if (m % 2) {
            vmem(target) =
                Q6_V_vor_VV(vmem(target), Q6_Vw_vasl_VwR(Q6_V_lo_W(pair), 24));
            vmem(target + 2048) = Q6_V_vor_VV(
                vmem(target + 2048), Q6_Vw_vasl_VwR(Q6_V_hi_W(pair), 24));
          } else {
            vmem(target) = Q6_Vw_vasl_VwR(Q6_V_lo_W(pair), 8);
            vmem(target + 2048) = Q6_Vw_vasl_VwR(Q6_V_hi_W(pair), 8);
          }
        }
      }
      metrics[worker][0] += tick() - stamp;
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
      // GU consumes gate first, then overwrites it with SwiGLU when the up
      // block arrives. GU and down metadata have disjoint lifetimes and share
      // the same 40 KiB region.
      dma_copy(down_meta, expert_data + header[8], header[3] - header[8]);
      stamp = tick();
      for (int m = 0; m < rows; m++)
        memset(middle + m * 512 + width, 0, (512 - width) * 4);
      metrics[worker][5] += tick() - stamp;
      stamp = tick();
      prepare_activations(
          packed, activation_scales, middle, 512, local, rows, down_k, 512);
      metrics[worker][0] += tick() - stamp;
#if SM_MOE_ACCUMULATE
      // After down A8 packing only the first 32 KiB of packed is live; reuse
      // GU/middle for down.
      auto* down_output = reinterpret_cast<float*>(base + 32768);
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
          all_scores);
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
#if SM_MOE_PROFILE
  __atomic_fetch_or(&sync->done_mask, 1u << worker, __ATOMIC_ACQ_REL);
#endif
  increment(&sync->done);
  await(&sync->go, 2);
  error = read(&sync->error);
  const auto combine_start = tick();
  // 首/中间批保留分路结果，仅单批或末批按 Top8 rank 合并，并加一次
  // shared/residual。
  if (!error && (phase == 0 || phase == 3))
    for (unsigned t = worker; t < unsigned(tokens_count); t += count)
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
      128 * 2048 * 4);
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
