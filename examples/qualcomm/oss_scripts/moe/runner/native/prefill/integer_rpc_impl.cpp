#include "bank_mapping.h"
#include "../power.h"
// Prefill FastRPC 入口；同层专家可归并 token，复用 worker、线程栈和 IO 映射。
#include "../../../../../custom_op/lowbit/prefill/integer_moe.h"
#include "../../../../../custom_op/lowbit/expert_stream.h"
#include "../worker_start.h"
#include "HAP_mem.h"
#include "qurt_memory.h"
#include <algorithm>
#include <cstring>
#include <cstdio>
#include "et_prefill_rpc.h"
#include "HAP_compute_res.h"
#include "HAP_power.h"
#include "HAP_dcvs.h"
#include "qurt.h"
#include "qurt_hvx.h"
#include "qurt_sem.h"
#include <cstdlib>
#include <new>
#include <memory>

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
  // 任一锁失败时，所有 worker 跳过 execute，不会留下等待缺席线程的计算屏障。
  if (!error && job->handle.resource == QHPI_RESOURCE_HMX) {
    const auto* descriptor = static_cast<const int32_t*>(job->input[4]->data);
    if (descriptor[2] != -1)
      prepare_integer_bank(static_cast<Sync*>(job->handle.sync), descriptor);
  }
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

struct ComputeResources {
  uint32_t context = 0;
  void* vtcm = nullptr;
};

static int prepare_resources(ComputeResources& resources, unsigned bytes) {
  if (resources.context)
    return 0;
  compute_res_attr_t attributes;
  HAP_compute_res_attr_init(&attributes);
  HAP_compute_res_attr_set_vtcm_param(
      &attributes, bytes, moe_prefill_vtcm::kCompact ? 0 : 1);
  HAP_compute_res_attr_set_hmx_param(&attributes, 1);
  if (HAP_compute_res_attr_set_cache_mode(&attributes, 1))
    return 70;
  const auto context = HAP_compute_res_acquire(&attributes, 1000000);
  if (!context)
    return 15;
  auto* vtcm = HAP_compute_res_attr_get_vtcm_ptr(&attributes);
  if (!vtcm) {
    HAP_compute_res_release(context);
    return 16;
  }
  // Cacheable creation reserves the address layout, without leasing compute.
  resources = {context, vtcm};
  return 0;
}

// Workers and VTCM address layouts stay with the handle. Compute leases return
// to QNN after each call; VTCM contents cannot survive that handoff.
struct State {
  moe::dsp::PowerVote power{this};
  IoMapping io[256];
  const float* residual = nullptr;
  int pending = -1, assigned = 0;
  uint8_t* route = nullptr;
  int capacity = 128;
  ComputeResources grouped, complete, partitioned;
  uint64_t profile_totals[10][8]{}, profile_calls = 0;
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
    for (auto* resources : {&grouped, &complete, &partitioned})
      if (resources->context)
        HAP_compute_res_release(resources->context);
    for (auto& lock : token_locks)
      qurt_pimutex_destroy(&lock);
    for (auto stack : stacks)
      free(stack);
    free(route);
  }
};

extern "C" AEEResult et_prefill_rpc_open(const char*, remote_handle64* handle) {
  auto state = std::unique_ptr<State>(new (std::nothrow) State{});
  if (!state)
    return 12;
  // Original per-token Top8 slots survive all expert batches and VTCM leases.
  state->route = static_cast<uint8_t*>(memalign(128, 128 * 8 * 2048 * 4));
  if (!state->route)
    return 12;
  for (auto& stack : state->stacks) {
    stack = memalign(128, 32768);
    if (!stack)
      return 13;
  }
  int error = state->power.set(true);
  if (error)
    return error;
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

extern "C" AEEResult et_prefill_rpc_close(remote_handle64 handle) {
  auto* state = reinterpret_cast<State*>(uintptr_t(handle));
  delete state;
  return release_banks();
}

extern "C" AEEResult et_prefill_rpc_configure(remote_handle64 handle, int tokens) {
  auto* state = reinterpret_cast<State*>(uintptr_t(handle));
  if (tokens < 128 || tokens > 1024 || tokens % 128 || state->pending != -1)
    return 14;
  if (tokens != state->capacity) {
    auto* route =
        static_cast<uint8_t*>(memalign(128, size_t(tokens) * 8 * 2048 * 4));
    if (!route)
      return 12;
    free(state->route);
    state->route = route;
    state->capacity = tokens;
  }
  if (tokens > 128)
    return prepare_resources(state->grouped, moe_prefill_vtcm::kGroupedBytes);
  const int error = prepare_resources(state->complete, moe_prefill_vtcm::kAr128Bytes);
  return error ? error : prepare_resources(state->partitioned, moe_prefill_vtcm::kPartitionedBytes);
}

extern "C" AEEResult
et_prefill_rpc_counts(remote_handle64, int* output, int count) {
  if (count != 4)
    return 55;
  for (int i = 0; i < 4; i++)
    output[i] = execution_counts[i];
  return 0;
}

extern "C" AEEResult
et_prefill_rpc_profile(remote_handle64 handle, float* output, int count) {
  auto* state = reinterpret_cast<State*>(uintptr_t(handle));
  if (count != 80 && count != 88)
    return 1;
  const auto* values = count == 80 ? reinterpret_cast<const uint64_t*>(metrics)
                                 : reinterpret_cast<const uint64_t*>(state->profile_totals);
  for (int i = 0; i < 80; i++)
    output[i] = values[i] / 19200.f;
  if (count == 88) {
    std::fill(output + 80, output + 88, 0.f);
    output[80] = SM_MOE_PROFILE;
    output[81] = state->profile_calls;
  }
  HAP_power_response_t clock{};
  clock.type = HAP_power_get_clk_Freq;
  if (!HAP_power_get(nullptr, &clock))
    output[8 * 8 + 5] = clock.clkFreqHz / 1.e6f;
  clock.type = HAP_power_get_hmx_core_clk_Freq;
  if (!HAP_power_get(nullptr, &clock))
    output[8 * 8 + 6] = clock.clkFreqHz / 1.e6f;
  output[8 * 8 + 7] = (qurt_hvx_get_units() >> 8) & 255;
  qurt_sysenv_max_hthreads_t hw{};
  qurt_sysenv_get_max_hw_threads(&hw);
  output[9 * 8 + 3] = hw.max_hthreads;
  unsigned int total = 0, available = 0;
  compute_res_vtcm_page_t total_pages{}, available_pages{};
  if (!HAP_compute_res_query_VTCM(
          0, &total, &total_pages, &available, &available_pages)) {
    output[9 * 8 + 4] = total / 1048576.f;
    output[9 * 8 + 5] = available / 1048576.f;
  }
  return 0;
}

extern "C" AEEResult et_prefill_rpc_run(
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
  // bank 描述符决定本批专家和累加阶段，跨批的路由结果保存在 DSP DRAM。
  const auto start = tick();
  auto stamp = start;
  memset(metrics, 0, sizeof(metrics));
  if (descriptor_len != 8)
    return 14;
  const int tokens = expert_stream::token_count(descriptor[3]);
  const int valid = expert_stream::valid_token_count(descriptor[3]);
  if (tokens < 128 || tokens > state->capacity || tokens % 128 || valid < 1 ||
      valid > tokens || hidden_len != tokens * 2048 || ids_len != tokens * 8 ||
      scores_len != tokens * 8 || shared_len != tokens * 2048 ||
      result_len != tokens * 2048)
    return 14;
  const int phase = descriptor[4] == expert_stream::kPhase ? 0 : descriptor[4];
  if (phase < 0 || phase > 3 || descriptor[5] < 1 || descriptor[5] > 256 ||
      descriptor[6] < 1 || descriptor[6] > valid * 8)
    return 54;
  if ((phase == 0 || phase == 1) && state->pending != -1)
    return 56;
  if ((phase == 2 || phase == 3) && state->pending != descriptor[7])
    return 56;
  if ((phase == 0 && descriptor[6] != valid * 8) ||
      (phase == 3 && state->assigned + descriptor[6] != valid * 8))
    return 56;
#if SM_MOE_ACCUMULATE
  if (phase || tokens != 128)
    return 57;
#endif
  auto& resources = tokens > 128 ? state->grouped
      : phase                    ? state->partitioned
                                 : state->complete;
  const int prepared = prepare_resources(
      resources, tokens > 128 ? moe_prefill_vtcm::kGroupedBytes
          : phase ? moe_prefill_vtcm::kPartitionedBytes
                  : moe_prefill_vtcm::kAr128Bytes);
  if (prepared)
    return prepared;
  const uint32_t context = resources.context;
  const int acquired = HAP_compute_res_acquire_cached(context, 1000000);
  if (acquired)
    return 71;
  void* vtcm = resources.vtcm;
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
  void* route = moe_prefill_vtcm::kRouteInVtcm
      ? static_cast<uint8_t*>(vtcm) + 4194304
      : state->route;
#elif SM_MOE_ACCUMULATE
  void* route = static_cast<uint8_t*>(vtcm) + 9437184 + 2048;
#else
  void* route = state->route;
#endif
  if (phase || tokens > 128)
    route = state->route;
  QHPI_Tensor output[3] = {{result}, {vtcm}, {route}};
  QHPI_Tensor* out[] = {output, output + 1, output + 2};
  Sync sync{};
  sync.token_locks = state->token_locks;
  sync.residual = state->residual;
  const int workers = moe_prefill_vtcm::workersForTokens(tokens);
  WorkerStart startup(workers + 1);
  for (int i = 0; i < kWorkers + 1; i++) {
    if (i >= workers && i != kWorkers)
      continue;
    auto& entry = state->entries[i];
    entry.job = {
        {uint32_t(i == kWorkers ? QHPI_RESOURCE_HMX : QHPI_RESOURCE_HVX),
         unsigned(workers),
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
  for (int i = 0; i < kWorkers + 1; i++) {
    if (i >= workers && i != kWorkers)
      continue;
    auto& entry = state->entries[i];
    qurt_sem_down(&entry.done);
    if (entry.job.status)
      error = entry.job.status;
  }
  metrics[8][2] = tick() - stamp;
  stamp = tick();
  const int released = HAP_compute_res_release_cached(context);
  metrics[8][3] = tick() - stamp;
  metrics[8][4] = tick() - start;
  if (!error && (phase == 1 || phase == 2)) {
    if (phase == 1)
      state->assigned = 0;
    state->pending = descriptor[7];
    state->assigned += descriptor[6];
  } else {
    state->pending = -1;
    state->assigned = 0;
  }
#if SM_MOE_PROFILE
  for (int row = 0; row < 10; ++row)
    for (int column = 0; column < 8; ++column)
      state->profile_totals[row][column] += metrics[row][column];
  ++state->profile_calls;
#endif
  return error ? error : released;
}

extern "C" AEEResult et_prefill_rpc_run_mapped(
    remote_handle64 handle,
    const int* descriptors,
    int count) {
  if (count != 14 && count != 20)
    return 40;
  auto* state = reinterpret_cast<State*>(uintptr_t(handle));
  const int tokens = expert_stream::token_count(descriptors[9]);
  if (tokens < 128 || tokens > state->capacity || tokens % 128)
    return 14;
  void* buffers[6]{};
  const unsigned int sizes[] = {
      unsigned(tokens * 2048 * 4),
      unsigned(tokens * 8 * 4),
      unsigned(tokens * 8 * 4),
      unsigned(tokens * 2048 * 4),
      unsigned(tokens * 2048 * 4),
      unsigned(tokens * 2048 * 4)};
  for (int i = 0; i < 6; i++) {
    if (i == 5 && descriptors[i] < 0)
      continue;
    const unsigned int mapped_bytes = count == 20
        ? static_cast<unsigned int>(descriptors[14 + i])
        : sizes[i];
    const unsigned int maximum_bytes =
        unsigned(state->capacity * (i == 1 || i == 2 ? 8 : 2048) * 4);
    if (mapped_bytes < sizes[i] || mapped_bytes > maximum_bytes)
      return 47;
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
          mapped_bytes,
          HAP_PROT_READ | HAP_PROT_WRITE,
          0,
          descriptors[i],
          0);
      if (!pointer || pointer == (void*)-1)
        return 42 + i;
      found->fd = descriptors[i];
      found->pointer = pointer;
      found->bytes = mapped_bytes;
    }
    if (found->bytes < sizes[i])
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
  int error = et_prefill_rpc_run(
      handle,
      static_cast<const float*>(buffers[0]),
      tokens * 2048,
      static_cast<const int*>(buffers[1]),
      tokens * 8,
      static_cast<const float*>(buffers[2]),
      tokens * 8,
      static_cast<const float*>(buffers[3]),
      tokens * 2048,
      descriptors + 6,
      8,
      static_cast<float*>(buffers[4]),
      tokens * 2048);
  state->residual = nullptr;
  if (error)
    return error;
  if (qurt_mem_cache_clean(
          uintptr_t(buffers[4]),
          sizes[4],
          QURT_MEM_CACHE_FLUSH,
          QURT_MEM_DCACHE))
    return 49;
  return 0;
}

extern "C" AEEResult et_prefill_rpc_release_io(remote_handle64 handle) {
  auto* state = reinterpret_cast<State*>(uintptr_t(handle));
  for (auto& entry : state->io)
    if (entry.pointer) {
      if (HAP_munmap(entry.pointer, entry.bytes))
        return 51;
      entry = IoMapping{};
    }
  return release_banks();
}

extern "C" AEEResult et_prefill_rpc_set_active(remote_handle64 handle, int active) {
  if (!handle)
    return 12;
  return reinterpret_cast<State*>(uintptr_t(handle))->power.set(active != 0);
}
