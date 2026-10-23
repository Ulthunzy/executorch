// FastRPC 入口与持久 DSP 线程池；IntegerMoE.cpp 提供实际 HVX/HMX 计算。
#include "integer_moe.h"
#include "../worker_start.h"
#include "HAP_mem.h"
#include "qurt_memory.h"
#include <algorithm>
#include <cstring>
#include <cstdio>
#include "integer_rpc.h"
#include "HAP_compute_res.h"
#include "HAP_power.h"
#include "HAP_dcvs.h"
#include "qurt.h"
#include "qurt_hvx.h"
#include "qurt_sem.h"
#include <cstdlib>
#include <new>
#include <memory>
#include "bank_mapper.h"
#include "../N2Utils.h"

extern "C" void* qhpi_tensor_raw_data(const QHPI_Tensor* t) {
  return t->data;
}

extern "C" uint32_t qhpi_thread_resources(QHPI_RuntimeHandle* h) {
  return h->resource;
}

extern "C" uint32_t qhpi_num_slices(QHPI_RuntimeHandle* h) {
  return h->slices;
}

extern "C" uint32_t qhpi_slice_number(QHPI_RuntimeHandle* h) {
  return h->slice;
}

extern "C" void* qhpi_sync_block(QHPI_RuntimeHandle* h) {
  return h->sync;
}

struct Job {
  QHPI_RuntimeHandle handle;
  QHPI_Tensor** output;
  const QHPI_Tensor* const* input;
  uint32_t resource_context, status;
  WorkerStart* startup;
};

static void run_thread(void* data) {
  Job* job = static_cast<Job*>(data);
  int locked = 0;
  if (job->handle.resource == QHPI_RESOURCE_HMX)
    locked = HAP_compute_res_hmx_lock(job->resource_context);
  else
    locked = qurt_hvx_lock(QURT_HVX_MODE_128B);
  const int error = job->startup->arrive(locked ? 1000 + locked : 0);
  // 已成功获得的锁仍由原 worker 释放，失败 worker 也参与启动屏障和完成通知。
  job->status = error
      ? error
      : execute_integer_moe(&job->handle, 3, job->output, 5, job->input);
  if (!locked) {
    if (job->handle.resource == QHPI_RESOURCE_HMX)
      HAP_compute_res_hmx_unlock(job->resource_context);
    else
      qurt_hvx_unlock();
  }
}

struct PoolEntry {
  Job job{};
  qurt_sem_t ready, done;
  qurt_thread_t thread{};
  volatile uint32_t stop = 0;
  bool started = false;
};

static void pool_thread(void* data) {
  auto* entry = static_cast<PoolEntry*>(data);
  while (true) {
    qurt_sem_down(&entry->ready);
    if (read(&entry->stop))
      break;
    run_thread(&entry->job);
    qurt_sem_up(&entry->done);
  }
}

struct IoMapping {
  int fd = -1;
  void* pointer = nullptr;
  unsigned int bytes = 0;
};

// 生命周期覆盖全部 decode：保留 IO 映射、线程栈、bank mapper 和 CRM 资源描述。
// VTCM 使用权在 QNN 与 Native 间切换，内容只在一次资源占用期间有效。
struct State {
  BankMapper bank_mapper;
  IoMapping io[256];
  const float* residual = nullptr;
  uint32_t resource_context = 0;
  void* vtcm = nullptr;
  uint64_t profile_totals[12][8]{};
  const uint32_t* ready_words = nullptr;
  int pending_fd = -1, pending_generation = 0;
  unsigned pending_mask = 0;
  int pending_ids[8]{};
  uint8_t* route = nullptr;
  void* stacks[kWorkers + 1]{};
  qurt_mutex_t token_locks[128];
  PoolEntry entries[kWorkers + 1];

  State() {
    for (auto& lock : token_locks)
      qurt_pimutex_init(&lock);
    for (auto& entry : entries) {
      qurt_sem_init(&entry.ready);
      qurt_sem_down(&entry.ready);
      qurt_sem_init(&entry.done);
      qurt_sem_down(&entry.done);
    }
  }

  ~State() {
    bank_mapper.close();
    for (auto& entry : entries)
      if (entry.started) {
        write(&entry.stop, 1);
        qurt_sem_up(&entry.ready);
      }
    for (auto& entry : entries) {
      if (entry.started) {
        int status;
        qurt_thread_join(entry.thread, &status);
      }
      qurt_sem_destroy(&entry.ready);
      qurt_sem_destroy(&entry.done);
    }
    for (auto& entry : io)
      if (entry.pointer)
        HAP_munmap(entry.pointer, entry.bytes);
    for (auto& lock : token_locks)
      qurt_pimutex_destroy(&lock);
    if (resource_context)
      HAP_compute_res_release(resource_context);
    HAP_power_destroy(this);
    for (auto stack : stacks)
      free(stack);
    free(route);
  }
};

extern "C" AEEResult integer_rpc_open(const char*, remote_handle64* handle) {
  auto state = std::unique_ptr<State>(new (std::nothrow) State{});
  if (!state)
    return 12;
#if !SM_MOE_ACCUMULATE
  state->route = static_cast<uint8_t*>(memalign(128, 1 * 8 * 2048 * 4));
  if (!state->route)
    return 12;
#endif
  for (auto& stack : state->stacks) {
    stack = memalign(128, 32768);
    if (!stack)
      return 13;
  }
  HAP_power_request_t power{};
  power.type = HAP_power_set_apptype;
  power.apptype = HAP_POWER_COMPUTE_CLIENT_CLASS;
  int error = HAP_power_set(state.get(), &power);
  if (error)
    return error;
  power = {};
  power.type = HAP_power_set_DCVS_v3;
  auto& dcvs = power.dcvs_v3;
  dcvs.set_dcvs_enable = 1;
  dcvs.dcvs_enable = 0;
  dcvs.set_bus_params = 1;
  dcvs.bus_params.min_corner = dcvs.bus_params.target_corner =
      dcvs.bus_params.max_corner = HAP_DCVS_VCORNER_MAX;
  dcvs.set_core_params = 1;
  dcvs.core_params.min_corner = dcvs.core_params.target_corner =
      dcvs.core_params.max_corner = HAP_DCVS_VCORNER_MAX;
  dcvs.set_sleep_disable = 1;
  dcvs.sleep_disable = 1;
  HAP_set_dcvs_v3_protected_bus_corners(&power, 1);
  error = HAP_power_set(state.get(), &power);
  if (error)
    return error;
  power = {};
  power.type = HAP_power_set_HVX;
  power.hvx.power_up = 1;
  error = HAP_power_set(state.get(), &power);
  if (error)
    return error;
  memset(&power, 0, sizeof(power));
  power.type = HAP_power_set_HMX_v2;
  power.hmx_v2.set_power = 1;
  power.hmx_v2.power_up = 1;
  power.hmx_v2.set_clock = 1;
  power.hmx_v2.target_corner = power.hmx_v2.min_corner =
      power.hmx_v2.max_corner = HAP_DCVS_EXP_VCORNER_MAX;
  power.hmx_v2.perf_mode = HAP_CLK_PERF_HIGH;
  error = HAP_power_set(state.get(), &power);
  if (error)
    return error;
#if SM_EXPERT_BITS == 2
  if (qurt_hvx_lock(QURT_HVX_MODE_128B))
    return 75;
  alignas(128) uint8_t packed_check[512];
  for (int i = 0; i < 256; i++)
    packed_check[i] = i;
  n2_expand_in_place(packed_check, 256);
  bool valid = true;
  for (int i = 0; i < 256; i++)
    for (int pair = 0; pair < 2; pair++) {
      const auto code = [&](int lane) {
        constexpr int book[4] = {-7, -2, 2, 7};
        return book[(i >> (lane * 2)) & 3] & 15;
      };
      valid = valid &&
          packed_check[i * 2 + pair] ==
              (code(pair * 2) | (code(pair * 2 + 1) << 4));
    }
  qurt_hvx_unlock();
  if (!valid)
    return 76;
#endif
  for (int i = 0; i < kWorkers + 1; i++) {
    auto& entry = state->entries[i];
    qurt_thread_attr_t attr;
    qurt_thread_attr_init(&attr);
    qurt_thread_attr_set_stack_addr(&attr, state->stacks[i]);
    qurt_thread_attr_set_stack_size(&attr, 32768);
    qurt_thread_attr_set_priority(&attr, 100);
    char name[16];
    snprintf(name, sizeof(name), "n4-%d", i);
    qurt_thread_attr_set_name(&attr, name);
    int error = qurt_thread_create(&entry.thread, &attr, pool_thread, &entry);
    if (error)
      return 100 + error;
    entry.started = true;
  }
  *handle = reinterpret_cast<uintptr_t>(state.release());
  return 0;
}

extern "C" AEEResult integer_rpc_close(remote_handle64 handle) {
  auto* state = reinterpret_cast<State*>(uintptr_t(handle));
  delete state;
  return release_expert_mappings();
}

extern "C" AEEResult integer_rpc_configure_banks(
    remote_handle64 handle,
    const int* descriptors,
    int count) {
  auto* state = reinterpret_cast<State*>(uintptr_t(handle));
  return state->bank_mapper.configure(descriptors, count);
}

extern "C" AEEResult
integer_rpc_profile(remote_handle64 handle, float* output, int count) {
  auto* state = reinterpret_cast<State*>(uintptr_t(handle));
  if (count != 96)
    return 1;
  for (int i = 0; i < 96; i++)
    output[i] = reinterpret_cast<uint64_t*>(state->profile_totals)[i] / 19200.f;
  HAP_power_response_t clock{};
  clock.type = HAP_power_get_clk_Freq;
  if (!HAP_power_get(nullptr, &clock))
    output[8 * 8 + 5] = clock.clkFreqHz / 1.e6f;
  clock.type = HAP_power_get_hmx_core_clk_Freq;
  if (!HAP_power_get(nullptr, &clock))
    output[8 * 8 + 6] = clock.clkFreqHz / 1.e6f;
  output[8 * 8 + 7] = (qurt_hvx_get_units() >> 8) & 255;

  return 0;
}

extern "C" AEEResult integer_rpc_run(
    remote_handle64 handle,
    const float* hidden,
    int hidden_len,
    const int* ids,
    int ids_len,
    const float* scores,
    int scores_len,
    const float* shared,
    int shared_len,
    const int* descriptor,
    int descriptor_len,
    float* result,
    int result_len) {
  auto* state = reinterpret_cast<State*>(uintptr_t(handle));
  const auto start = tick();
  auto stamp = start;
  memset(metrics, 0, sizeof(metrics));
  if (hidden_len != 1 * 2048 || ids_len != 1 * 8 || scores_len != 1 * 8 ||
      shared_len != 1 * 2048 || descriptor_len != 8 || result_len != 1 * 2048)
    return 14;
  // phase 1/2 是互补 Top8 mask 的两段计算；phase 3 用同一 RPC 内的 ready 队列。
  const unsigned phase = unsigned(descriptor[7]),
                 mask = unsigned(descriptor[6]);
  if (phase > 3 || ((phase == 1 || phase == 2) && (mask == 0 || mask >= 255)))
    return 60;
  if (phase == 1 && state->pending_mask)
    return 61;
  if (phase == 2 &&
      (state->pending_fd != descriptor[0] ||
       state->pending_generation != descriptor[2] ||
       (mask ^ state->pending_mask) != 255 ||
       memcmp(state->pending_ids, ids, 8 * sizeof(int))))
    return 62;
  if ((phase == 0 || phase == 3) && state->pending_mask)
    return 63;
  const uint8_t* prepared_bank = nullptr;
  int mapped_layer = -1;
  if (state->bank_mapper.enabled()) {
    const auto mapping_stamp = tick();
    const int error = state->bank_mapper.wait(
        descriptor[4], descriptor[5], prepared_bank, mapped_layer);
    if (error)
      return error;
    metrics[10][3] = tick() - mapping_stamp;
    metrics[10][4] = state->bank_mapper.duration();
  }
  stamp = tick();
  // CRM context/layout 只创建一次，后续 acquire_cached/release_cached 与 QNN
  // 交替用资源。
  if (!state->resource_context) {
    compute_res_attr_t attributes;
    HAP_compute_res_attr_init(&attributes);
    HAP_compute_res_attr_set_vtcm_param(&attributes, 2097152, 1);
    HAP_compute_res_attr_set_hmx_param(&attributes, 1);
    if (HAP_compute_res_attr_set_cache_mode(&attributes, 1))
      return 70;
    state->resource_context = HAP_compute_res_acquire(&attributes, 1000000);
    if (!state->resource_context)
      return 15;
    state->vtcm = HAP_compute_res_attr_get_vtcm_ptr(&attributes);
    if (!state->vtcm) {
      HAP_compute_res_release(state->resource_context);
      state->resource_context = 0;
      return 16;
    }
  }
  const uint32_t context = state->resource_context;
  // Cached release returns resources to QNN while preserving the CRM context
  // and layout.
  if (HAP_compute_res_acquire_cached(context, 1000000))
    return 71;
  void* vtcm = state->vtcm;
  metrics[8][0] = tick() - stamp;
  stamp = tick();
  QHPI_Tensor input[5] = {
      {const_cast<float*>(hidden)},
      {const_cast<int*>(ids)},
      {const_cast<float*>(scores)},
      {const_cast<float*>(shared)},
      {const_cast<int*>(descriptor)}};
  const QHPI_Tensor* in[] = {input, input + 1, input + 2, input + 3, input + 4};
#if SM_MOE_ROUTE_VTCM
  void* route = static_cast<uint8_t*>(vtcm) + 1971584;
#elif SM_MOE_ACCUMULATE
  void* route = static_cast<uint8_t*>(vtcm) + 9437184 + 2048;
#else
  void* route = state->route;
#endif
  // VTCM is released between RPCs; only the 8 x 2048 route results persist in
  // DSP DRAM.
  if (phase == 2)
    memcpy(route, state->route, 8 * 2048 * sizeof(float));
  QHPI_Tensor output[3] = {{result}, {vtcm}, {route}};
  QHPI_Tensor* out[] = {output, output + 1, output + 2};
  Sync sync{};
  sync.cached_bank = prepared_bank;
  sync.token_locks = state->token_locks;
  sync.residual = state->residual;
  sync.ready_words = phase == 3 ? state->ready_words : nullptr;
  WorkerStart startup(kWorkers + 1);
  for (int i = 0; i < kWorkers + 1; i++) {
    auto& entry = state->entries[i];
    entry.job = {
        {uint32_t(i == kWorkers ? QHPI_RESOURCE_HMX : QHPI_RESOURCE_HVX),
         kWorkers,
         uint32_t(i == kWorkers ? 0 : i),
         &sync},
        out,
        in,
        context,
        0,
        &startup};
    qurt_sem_up(&entry.ready);
  }
  metrics[8][1] = tick() - stamp;
  stamp = tick();
  int error = 0;
  for (auto& entry : state->entries) {
    qurt_sem_down(&entry.done);
    if (entry.job.status)
      error = entry.job.status;
  }
  if (!error && phase == 1) {
    memcpy(state->route, route, 8 * 2048 * sizeof(float));
    state->pending_fd = descriptor[0];
    state->pending_generation = descriptor[2];
    state->pending_mask = mask;
    memcpy(state->pending_ids, ids, 8 * sizeof(int));
  } else if (phase == 2 || error)
    state->pending_mask = 0;
  if (!error && mapped_layer >= 0 && phase != 1)
    state->bank_mapper.schedule((mapped_layer + 1) % 40);
  metrics[8][2] = tick() - stamp;
  stamp = tick();
  const int release_status = HAP_compute_res_release_cached(context);
  metrics[8][3] = tick() - stamp;
  metrics[8][4] = tick() - start;
#if SM_MOE_PROFILE
  for (int c = 0; c < 5; c++) {
    const int source[] = {5, 1, 0, 4, 3};
    for (int w = 0; w < 8; w++)
      metrics[9][3 + c] = std::max(metrics[9][3 + c], metrics[w][source[c]]);
  }
  for (int row = 0; row < 12; row++)
    for (int col = 0; col < 8; col++)
      state->profile_totals[row][col] += metrics[row][col];
#endif
  return error ? error : release_status;
}

extern "C" AEEResult integer_rpc_run_mapped(
    remote_handle64 handle,
    const int* descriptors,
    int count) {
  // ARM 提供六个 IO fd 和八个 bank 字段，映射缓存按 fd 复用。
  if (count != 14)
    return 40;
  auto* state = reinterpret_cast<State*>(uintptr_t(handle));
  void* buffers[6]{};
  const unsigned int sizes[] = {
      1 * 2048 * 4,
      1 * 8 * 4,
      1 * 8 * 4,
      1 * 2048 * 4,
      1 * 2048 * 4,
      1 * 2048 * 4};
  for (int i = 0; i < 6; i++) {
    if (i == 5 && descriptors[i] < 0)
      continue;
    IoMapping* found = nullptr;
    for (auto& entry : state->io)
      if (entry.fd == descriptors[i]) {
        found = &entry;
        break;
      }
    if (!found)
      for (auto& entry : state->io)
        if (entry.fd == -1) {
          found = &entry;
          break;
        }
    if (!found)
      return 41;
    if (found->fd == -1) {
      auto* pointer = HAP_mmap2(
          nullptr,
          sizes[i],
          HAP_PROT_READ | HAP_PROT_WRITE,
          0,
          descriptors[i],
          0);
      if (!pointer || pointer == (void*)-1)
        return 42 + i;
      found->fd = descriptors[i];
      found->pointer = pointer;
      found->bytes = sizes[i];
    }
    if (found->bytes != sizes[i])
      return 47;
    buffers[i] = found->pointer;
  }
  if (qurt_mem_cache_clean(
          uintptr_t(buffers[3]),
          sizes[3],
          QURT_MEM_CACHE_INVALIDATE,
          QURT_MEM_DCACHE))
    return 48;
  if (buffers[5] &&
      qurt_mem_cache_clean(
          uintptr_t(buffers[5]),
          sizes[5],
          QURT_MEM_CACHE_INVALIDATE,
          QURT_MEM_DCACHE))
    return 50;
  state->residual = static_cast<const float*>(buffers[5]);
  state->ready_words = nullptr;
  if (descriptors[13] == 3) {
    const int fd = descriptors[12];
    IoMapping* mapping = nullptr;
    for (auto& entry : state->io)
      if (entry.fd == fd) {
        mapping = &entry;
        break;
      }
    if (!mapping)
      for (auto& entry : state->io)
        if (entry.fd == -1) {
          mapping = &entry;
          break;
        }
    if (!mapping || fd < 0)
      return 66;
    if (mapping->fd == -1) {
      auto* pointer = HAP_mmap2(
          nullptr,
          4096,
          HAP_PROT_READ | HAP_PROT_WRITE | HAP_MEM_CACHE_NON_SHARED,
          0,
          fd,
          0);
      if (!pointer || pointer == (void*)-1)
        return 67;
      mapping->fd = fd;
      mapping->pointer = pointer;
      mapping->bytes = 4096;
    }
    if (mapping->bytes != 4096)
      return 68;
    state->ready_words = static_cast<const uint32_t*>(mapping->pointer);
    if (read(state->ready_words) != unsigned(descriptors[8]))
      return 69;
  }
  int error = integer_rpc_run(
      handle,
      static_cast<const float*>(buffers[0]),
      1 * 2048,
      static_cast<const int*>(buffers[1]),
      1 * 8,
      static_cast<const float*>(buffers[2]),
      1 * 8,
      static_cast<const float*>(buffers[3]),
      1 * 2048,
      descriptors + 6,
      8,
      static_cast<float*>(buffers[4]),
      1 * 2048);
  state->residual = nullptr;
  if (error)
    return error;
  if (descriptors[13] != 1 &&
      qurt_mem_cache_clean(
          uintptr_t(buffers[4]),
          sizes[4],
          QURT_MEM_CACHE_FLUSH,
          QURT_MEM_DCACHE))
    return 49;
  return 0;
}

extern "C" AEEResult integer_rpc_release_io(remote_handle64 handle) {
  auto* state = reinterpret_cast<State*>(uintptr_t(handle));
  for (auto& entry : state->io)
    if (entry.pointer) {
      if (HAP_munmap(entry.pointer, entry.bytes))
        return 51;
      entry = IoMapping{};
    }
  return 0;
}

extern "C" AEEResult integer_rpc_release_fds(
    remote_handle64 handle,
    const int* descriptors,
    int count) {
  auto* state = reinterpret_cast<State*>(uintptr_t(handle));
  for (int i = 0; i < count; i++)
    for (auto& entry : state->io)
      if (entry.fd == descriptors[i] && entry.pointer) {
        if (HAP_munmap(entry.pointer, entry.bytes))
          return 51;
        entry = IoMapping{};
        break;
      }
  return 0;
}

// One RPC owns the command loop; each command still returns HMX/VTCM to QNN.
extern "C" AEEResult integer_rpc_serve(remote_handle64 handle, int mailbox_fd) {
  // uncached mailbox 布局与 ARM NativeCore 对齐：0/1 命令、32/33 完成、64
  // 状态、96 描述。
  auto* box = static_cast<uint32_t*>(HAP_mmap2(
      nullptr,
      4096,
      HAP_PROT_READ | HAP_PROT_WRITE | HAP_MEM_CACHE_NON_SHARED,
      0,
      mailbox_fd,
      0));
  if (!box || box == (void*)-1)
    return 84;
  uint32_t generation = 0;
  int result = 0;
  write(box + 64, 1);
  while (true) {
    const auto idle_start = HAP_perf_get_qtimer_count();
    while (read(box) == generation) {
      if (HAP_perf_get_qtimer_count() - idle_start > 19200000ULL * 60) {
        result = 85;
        break;
      }
      asm volatile("pause(#64)");
    }
    if (result)
      break;
    generation = read(box);
    const auto command = read(box + 1);
    if (command == 1) {
      write(box + 33, 0);
      write(box + 32, generation);
      break;
    }
    if (command != 0) {
      result = 86;
      write(box + 33, result);
      write(box + 32, generation);
      break;
    }
    int descriptors[14];
    for (int i = 0; i < 14; i++)
      descriptors[i] = int(read(box + 96 + i));
    result = integer_rpc_run_mapped(handle, descriptors, 14);
    write(box + 33, result);
    write(box + 32, generation);
    if (result)
      break;
  }
  write(box + 64, 2);
  const int unmap = HAP_munmap(box, 4096);
  return result ? result : unmap;
}
