#include <executorch/runtime/backend/interface.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/runtime/platform/runtime.h>
#include "decoder_state.h"
#include "sampling.h"
#include "../decode_api.h"
#include <set>
#include <sstream>

using namespace executorch::runtime;
std::string model_root;
std::unique_ptr<DecodeSession> session;
static std::vector<std::unique_ptr<MappedBinary>> front_overrides;

static void memory_snapshot(const char* stage) {
  std::ifstream status("/proc/self/status");
  std::string line;
  while (std::getline(status, line))
    for (const char* key : {"VmRSS:", "VmHWM:", "VmSwap:"})
      if (line.rfind(key, 0) == 0)
        printf("PROCESS_MEMORY stage=%s %s\n", stage, line.c_str());
  std::ifstream rollup("/proc/self/smaps_rollup");
  while (std::getline(rollup, line))
    for (const char* key : {"Pss:", "Private_Dirty:", "SwapPss:"})
      if (line.rfind(key, 0) == 0)
        printf("PROCESS_MEMORY stage=%s %s\n", stage, line.c_str());
}

struct Stage {
  uint32_t kind, layer;
};

// 解析自定义 PTE 的 122 阶段计划；融合 Attention/Router binary 可以覆盖基础图。
// 所有层共用一个 DecodeSession，QNN context/state 和专家缓存跨 token 保留。
class DecodeBackend final : public BackendInterface {
 public:
  bool is_available() const override {
    return true;
  }

  Result<DelegateHandle*> init(
      BackendInitContext&,
      FreeableBuffer* processed,
      ArrayRef<CompileSpec>) const override {
    try {
      if (processed->size() < 128)
        return Error::DelegateInvalidCompatibility;
      auto* p = static_cast<const uint8_t*>(processed->data());
      uint32_t version, kind, layer;
      uint64_t size, aux_offset, aux_size;
      if (memcmp(p, "Q36DECO1", 8))
        return Error::DelegateInvalidCompatibility;
      memcpy(&version, p + 8, 4);
      memcpy(&kind, p + 12, 4);
      memcpy(&layer, p + 16, 4);
      memcpy(&size, p + 24, 8);
      memcpy(&aux_offset, p + 32, 8);
      memcpy(&aux_size, p + 40, 8);
      if ((version != 1 && !(version == 3 && getenv("SHARED_MODEL_ROOT"))) ||
          kind > 4 || layer >= 40 || size > processed->size() - 128 ||
          (aux_size &&
           (aux_offset > processed->size() ||
            aux_size > processed->size() - aux_offset)))
        return Error::DelegateInvalidCompatibility;
      if (kind == 0) {
        if (aux_size)
          return Error::DelegateInvalidCompatibility;
        auto current = bytes(model_root + "/cache-encoding.bin");
        try {
          validate_cache_encoding_payload(current, p + 128, size);
        } catch (const std::exception& error) {
          fprintf(stderr, "backend cache compatibility: %s\n", error.what());
          return Error::DelegateInvalidCompatibility;
        }
        cache_encoding = std::move(current);
        return static_cast<DelegateHandle*>(new Stage{kind, layer});
      }
      if (version == 3) {
        if (kind == 1) {
          if (size || aux_size < 8 || aux_size % 4)
            return Error::DelegateInvalidCompatibility;
          layer_io[layer].resize(aux_size / 4);
          memcpy(layer_io[layer].data(), p + aux_offset, aux_size);
        } else if (size || aux_size)
          return Error::DelegateInvalidCompatibility;
        return static_cast<DelegateHandle*>(new Stage{kind, layer});
      }
      const auto base = model_root + "/layer" + std::to_string(layer);
      if (kind == 1) {
        if (!size || aux_size < 8 || aux_size % 4)
          return Error::DelegateInvalidCompatibility;
        if (layer % 4 == 3) {
          auto io = bytes(base + "/io.bin");
          layer_io[layer].resize(io.size() / 4);
          memcpy(layer_io[layer].data(), io.data(), io.size());
        } else {
          embedded_binaries[base + "/attention.bin"] = {p + 128, size};
          layer_io[layer].resize(aux_size / 4);
          memcpy(layer_io[layer].data(), p + aux_offset, aux_size);
        }
      } else if (kind == 2) {
        if (!size)
          return Error::DelegateInvalidCompatibility;
        embedded_binaries[base + "/moe-front.bin"] = {p + 128, size};
      } else if (kind == 4) {
        if (!size)
          return Error::DelegateInvalidCompatibility;
        embedded_binaries[model_root + "/head.bin"] = {p + 128, size};
      }
      if (kind == 1 && layer % 4 != 3 && getenv("FRONT_OVERRIDE")) {
        const std::string directory = std::string(getenv("FRONT_OVERRIDE")) +
            "/layer" + std::to_string(layer);
        auto file =
            std::make_unique<MappedBinary>(directory + "/attention.bin");
        embedded_binaries[base + "/attention.bin"] = {
            file->data(), file->size()};
        front_overrides.push_back(std::move(file));
        auto io = bytes(directory + "/io.bin");
        if (io.size() % 4)
          throw std::runtime_error("override IO size");
        layer_io[layer].resize(io.size() / 4);
        memcpy(layer_io[layer].data(), io.data(), io.size());
      }
      return static_cast<DelegateHandle*>(new Stage{kind, layer});
    } catch (const std::exception& e) {
      fprintf(stderr, "backend init: %s\n", e.what());
      return Error::Internal;
    }
  }

  Error execute(
      BackendExecutionContext&,
      DelegateHandle* handle,
      Span<EValue*> args) const override {
    try {
      if (!session)
        throw std::runtime_error("decode session not initialized");
      auto stage = *static_cast<Stage*>(handle);
      const size_t counts[] = {3, 3, 4, 6, 2};
      if (args.size() != counts[stage.kind])
        throw std::runtime_error("stage arguments");
      auto input = [&](int i) {
        return args[i]->toTensor().const_data_ptr<float>();
      };
      auto output = [&](int i) {
        return args[i]->toTensor().mutable_data_ptr<float>();
      };
      switch (stage.kind) {
        case 0:
          session->embed(
              args[0]->toTensor().const_data_ptr<int64_t>(),
              args[1]->toTensor().const_data_ptr<int64_t>()[0],
              output(2));
          break;
        case 1:
          session->attend(stage.layer, input(0), output(1), output(2));
          break;
        case 2:
          session->route(
              stage.layer,
              input(0),
              args[1]->toTensor().mutable_data_ptr<int32_t>(),
              output(2),
              output(3));
          break;
        case 3:
          session->experts(
              stage.layer,
              input(0),
              args[1]->toTensor().const_data_ptr<int32_t>(),
              input(2),
              input(3),
              input(4),
              output(5));
          break;
        case 4:
          session->project(input(0), output(1));
          break;
      }
      return Error::Ok;
    } catch (const std::exception& e) {
      fprintf(stderr, "stage execute: %s\n", e.what());
      return Error::Internal;
    }
  }

  void destroy(DelegateHandle* handle) const override {
    delete static_cast<Stage*>(handle);
  }
};

static DecodeBackend backend_instance;
static Backend backend{"Qwen36DecodeStage", &backend_instance};
static auto registered = register_backend(backend);

static int decode_main(int argc, const char** argv) try {
  runtime_init();
  if (argc != 5)
    throw std::runtime_error(
        "stream_runner ROOT MAX_NEW_TOKENS CACHE_MIB EOS_IDS");
  model_root = argv[1];
  const int max_tokens = std::stoi(argv[2]), cache_mib = std::stoi(argv[3]);
  const int prefix_tokens = getenv("PREFILL_VALID_TOKENS")
      ? atoi(getenv("PREFILL_VALID_TOKENS"))
      : getenv("PREFILL_TOKENS") ? atoi(getenv("PREFILL_TOKENS"))
                                 : 128;
  if (max_tokens < 1 ||
      max_tokens > (getenv("DECODE_CONTEXT_CAPACITY")
                        ? atoi(getenv("DECODE_CONTEXT_CAPACITY"))
                        : 1152) +
              1 - prefix_tokens ||
      cache_mib < 0 || cache_mib > 4096)
    throw std::runtime_error("invalid generation limits");
  std::set<int64_t> eos_ids;
  std::stringstream eos_parser(argv[4]);
  std::string part;
  while (std::getline(eos_parser, part, ',')) {
    size_t used;
    const int64_t id = std::stoll(part, &used);
    if (used != part.size() || id < 0 || id >= 248320)
      throw std::runtime_error("invalid EOS token");
    eos_ids.insert(id);
  }
  if (eos_ids.empty())
    throw std::runtime_error("empty EOS list");

  const auto setup = Clock::now();
  executorch::extension::Module module(
      model_root + "/decode.pte",
      executorch::extension::Module::LoadMode::Mmap);
  const auto status = module.load_method("decode_forward");
  if (status != Error::Ok)
    throw std::runtime_error("PTE load " + std::to_string(int(status)));
  if (embedded_binaries.size() != 71 && !getenv("SHARED_MODEL_ROOT"))
    throw std::runtime_error("missing embedded QNN graphs");
  session = std::make_unique<DecodeSession>(model_root, cache_mib);

  struct Lifetime {
    ~Lifetime() {
      session.reset();
      embedded_binaries.clear();
      front_overrides.clear();
    }
  } lifetime;

  const auto prefill_logits = bytes(model_root + "/prefill-logits.f32");
  if (prefill_logits.size() != 248320 * 4)
    throw std::runtime_error("prefill logits shape");
  const auto* first_logits =
      reinterpret_cast<const float*>(prefill_logits.data());
  int64_t current = checked_argmax(first_logits, 248320);
  auto& rt = session->runtime;
  printf(
      "EXPERT_SCHEDULER hit_first=%d ready_queue=%d partial_result_bytes=65536 sum_order=top8_rank\n",
      session->hit_first,
      session->ready_queue);
  printf(
      "DECODE_READY ar_len=1 layers=40 prefix=%d max_new_tokens=%d setup_ms=%.6f\n",
      prefix_tokens,
      max_tokens,
      elapsed(setup));
  session->reset();
  const char* warm = getenv("SEED_PREFILL_LRU");
  if (warm && std::string(warm) == "1")
    session->warm_from_prefill();
  if (session->cache->imported) {
    std::vector<int32_t> selected;
    for (int layer = 0; layer < 40; layer++)
      for (int slot = 0; slot < session->cache->slots_per_layer(); slot++)
        selected.push_back(session->cache->expert_at(layer, slot));
    save(
        model_root + "/cache-seed-ids.i32",
        selected.data(),
        selected.size() * 4);
    printf(
        "EXPERT_SEED layers=40 slots_per_layer=%d experts=%d read_bytes=0 preload_ms=0 scope=reused_from_prefill\n",
        session->cache->slots_per_layer(),
        int(selected.size()));
  }
  printf(
      "EXPERT_CACHE_READY slots_per_layer=%d expert_cache_bytes=%zu scratch_bytes=%zu rpc_buffer_bytes=%llu peak_rpc_buffer_bytes=%llu seeded=%d\n",
      session->cache->slots_per_layer(),
      session->cache->bytes,
      2 * session->bank_slots[0]->size,
      (unsigned long long)rt.buffer_bytes,
      (unsigned long long)rt.peak_buffer_bytes,
      (warm && std::string(warm) == "1") || session->cache->imported);
  memory_snapshot("ready");
  const auto creates = rt.context_creates, frees = rt.context_frees;
  const auto registers = rt.memory_registers,
             deregisters = rt.memory_deregisters;
  const auto allocations = rt.buffer_allocations,
             deallocations = rt.buffer_frees;
  std::vector<float> hidden, logits;
  std::vector<int64_t> input_ids, output_ids;
  hidden.reserve((max_tokens - 1) * 2048);
  const bool save_all_logits = getenv("SAVE_ALL_LOGITS") &&
      std::string(getenv("SAVE_ALL_LOGITS")) == "1";
  logits.reserve((save_all_logits ? max_tokens - 1 : 1) * 248320);
  input_ids.reserve(max_tokens - 1);
  output_ids.reserve(max_tokens - 1);
  double total = 0, read = 0, native = 0, attention = 0, router = 0,
         feedback = 0, head = 0;
  std::array<std::array<double, 3>, 40> layer_totals{};
  uint64_t direct = 0, hits = 0;
  uint64_t hot_hits = 0, cold_hits = 0;
  double wait_ms = 0, sync_ms = 0, pread_worker_ms = 0, copy_worker_ms = 0;
  printf("DECODE_DIAGNOSTICS save_all_logits=%d\n", save_all_logits);
  const auto generation_begin = Clock::now();
  auto previous = generation_begin;
  const bool ignore_eos = getenv("BENCH_IGNORE_EOS") &&
      std::string(getenv("BENCH_IGNORE_EOS")) == "1";
  std::vector<int64_t> teacher;
  if (const char* path = getenv("BENCH_TOKENS_PATH")) {
    const auto data = bytes(path);
    if (data.size() % 8 || data.size() / 8 < size_t(max_tokens - 1))
      throw std::runtime_error("teacher token count");
    teacher.resize(data.size() / 8);
    memcpy(teacher.data(), data.data(), data.size());
  }
  bool stopped = eos_ids.count(current);
  printf(
      "TOKEN {\"index\":0,\"id\":%lld,\"position\":%d,\"device_ms\":0,\"device_gap_ms\":0,\"model_ms\":0,\"eos\":%s}\n",
      (long long)current,
      prefix_tokens,
      stopped ? "true" : "false");

  const bool trace_layers = getenv("DECODE_TRACE_LAYERS") &&
      std::string(getenv("DECODE_TRACE_LAYERS")) == "1";
  while ((!stopped || ignore_eos) && int(output_ids.size()) + 1 < max_tokens) {
    const int step = output_ids.size();
    int64_t token = teacher.empty() ? current : teacher[step];
    input_ids.push_back(token);
    auto input = executorch::extension::make_tensor_ptr(
        {1, 1}, &token, executorch::aten::ScalarType::Long);
    int64_t position = prefix_tokens + step;
    auto position_input = executorch::extension::make_tensor_ptr(
        {1}, &position, executorch::aten::ScalarType::Long);
    const auto start = Clock::now();
    auto result = module.execute(
        "decode_forward", std::vector<EValue>{*input, *position_input});
    const double model_ms = elapsed(start);
    if (!result.ok())
      throw std::runtime_error(
          "PTE execute " + std::to_string(int(result.error())));
    if (result->size() != 2)
      throw std::runtime_error("decode output count");
    const auto* scores = result->at(0).toTensor().const_data_ptr<float>();
    const auto* state = result->at(1).toTensor().const_data_ptr<float>();
    for (int i = 0; i < 2048; i++)
      if (!std::isfinite(state[i]))
        throw std::runtime_error("nonfinite hidden");
    current = checked_argmax(scores, 248320);
    stopped = eos_ids.count(current);
    const auto now = Clock::now();
    const double device_ms =
        std::chrono::duration<double, std::milli>(now - generation_begin)
            .count();
    const double gap_ms =
        std::chrono::duration<double, std::milli>(now - previous).count();
    previous = now;
    printf(
        "TOKEN {\"index\":%d,\"id\":%lld,\"position\":%lld,\"device_ms\":%.6f,\"device_gap_ms\":%.6f,\"model_ms\":%.6f,\"eos\":%s}\n",
        step + 1,
        (long long)current,
        (long long)(position + 1),
        device_ms,
        gap_ms,
        model_ms,
        stopped ? "true" : "false");
    output_ids.push_back(current);
    if (save_all_logits)
      logits.insert(logits.end(), scores, scores + 248320);
    else
      logits.assign(scores, scores + 248320);
    hidden.insert(hidden.end(), state, state + 2048);
    const auto& m = session->metrics;
    for (int layer = 0; layer < 40; layer++)
      for (int stage = 0; stage < 3; stage++)
        layer_totals[layer][stage] += m.execution[layer][stage];
    if (trace_layers || session->async_io)
      printf(
          "PREFETCH_STEP step=%d async=%d requests=%llu read_bytes=%llu used=%llu remaining_wait_ms=%.6f io_union_ms=%.6f overlap_exec_ms=%.6f\n",
          step,
          session->async_io,
          (unsigned long long)m.prediction_reads,
          (unsigned long long)m.prediction_bytes,
          (unsigned long long)m.prediction_used,
          m.prediction_wait,
          m.prediction_union,
          m.prediction_overlap);
    if (trace_layers)
      for (int layer = 0; layer < 40; layer++)
        printf(
            "LAYER_EXEC step=%d layer=%d attention_ms=%.6f router_ms=%.6f native_ms=%.6f\n",
            step,
            layer,
            m.execution[layer][0],
            m.execution[layer][1],
            m.execution[layer][2]);
    if (m.calls[0] != 1 || m.calls[1] != 40 || m.calls[2] != 40 ||
        m.calls[3] != 40 || m.calls[4] != 1 || m.requests != 320 ||
        session->position != position + 1)
      throw std::runtime_error("decode task coverage");
    total += model_ms;
    read += m.read;
    native += m.native;
    attention += m.attention;
    router += m.router;
    feedback += m.feedback;
    head += m.head;
    direct += m.direct;
    hits += m.hits;
    if (trace_layers)
      for (int layer = 0; layer < 40; layer++) {
        printf("ROUTE step=%d layer=%d ids=", step, layer);
        for (int id : m.routes[layer])
          printf("%d,", id);
        printf("\n");
      }
    if (trace_layers)
      printf(
          "CACHE_STEP step=%d hits=%llu requests=%llu direct_bytes=%llu\n",
          step,
          (unsigned long long)m.hits,
          (unsigned long long)m.requests,
          (unsigned long long)m.direct);
    for (int layer = 0; layer < 40; layer++) {
      const auto& io = m.io[layer];
      hot_hits += io.hot;
      cold_hits += io.cold;
      wait_ms += io.wait;
      sync_ms += io.sync;
      pread_worker_ms += io.pread_worker;
      copy_worker_ms += io.copy_worker;
      if (trace_layers)
        printf(
            "IO_LAYER step=%d layer=%d hot_hits=%d cold_hits=%d misses=%d direct_bytes=%llu wait_ms=%.6f sync_ms=%.6f pread_worker_ms=%.6f copy_worker_ms=%.6f feedback_overlap_ms=%.6f read_calls=%d\n",
            step,
            layer,
            io.hot,
            io.cold,
            8 - io.hot - io.cold,
            (unsigned long long)io.bytes,
            io.wait,
            io.sync,
            io.pread_worker,
            io.copy_worker,
            io.feedback_overlap,
            io.read_calls);
      if (trace_layers)
        printf(
            "IO_OVERLAP step=%d layer=%d hit_mask=%d split=%d hit_native_ms=%.6f hit_overlap_ms=%.6f tail_wait_ms=%.6f\n",
            step,
            layer,
            io.hit_mask,
            io.split,
            io.hit_native,
            io.hit_overlap,
            io.tail_wait);
    }
    if (trace_layers)
      printf(
          "DECODE_STEP step=%d input_id=%lld output_id=%lld total_ms=%.6f attention_ms=%.6f router_ms=%.6f read_ms=%.6f native_ms=%.6f feedback_ms=%.6f head_ms=%.6f\n",
          step,
          (long long)token,
          (long long)current,
          model_ms,
          m.attention,
          m.router,
          m.read,
          m.native,
          m.feedback,
          m.head);
  }
  const double loop_ms = elapsed(generation_begin);
  check(session->core.stop_service(), "stop native service after decode");
  printf(
      "NATIVE_DISPATCH direct_rpcs=%llu service_rpcs=%llu mailbox_commands=%llu arm_service_threads=%llu\n",
      (unsigned long long)session->core.direct_invocations,
      (unsigned long long)session->core.service_invocations,
      (unsigned long long)session->core.mailbox_commands,
      (unsigned long long)session->core.service_threads);
  if (getenv("MOE_PROFILE_TOTAL")) {
    float profile[96]{};
    check(
        integer_rpc_profile(session->core.handle, profile, 96),
        "native profile totals");
    for (int row = 0; row < 12; row++) {
      printf("PROFILE_TOTAL %d", row);
      for (int col = 0; col < 8; col++)
        printf(" %.6f", profile[row * 8 + col]);
      printf("\n");
    }
  }
  const int steps = output_ids.size();
  for (int layer = 0; layer < 40; layer++)
    printf(
        "LAYER_TOTAL layer=%d steps=%d attention_ms=%.6f router_ms=%.6f native_ms=%.6f\n",
        layer,
        steps,
        layer_totals[layer][0],
        layer_totals[layer][1],
        layer_totals[layer][2]);
  printf(
      "DECODE_STOP reason=%s generated_tokens=%d decode_steps=%d\n",
      stopped ? "eos" : "max_new_tokens",
      steps + 1,
      steps);
  printf(
      "DECODE_LOOP steps=%d loop_ms=%.6f ms_per_token=%.6f tokens_per_second=%.6f\n",
      steps,
      loop_ms,
      steps ? loop_ms / steps : 0,
      loop_ms ? 1000 * steps / loop_ms : 0);
  printf(
      "EXPERT_IO_TOTAL hot_hits=%llu cold_hits=%llu misses=%llu direct_bytes=%llu wait_ms=%.6f sync_ms=%.6f pread_worker_ms=%.6f copy_worker_ms=%.6f\n",
      (unsigned long long)hot_hits,
      (unsigned long long)cold_hits,
      (unsigned long long)(steps * 320 - hot_hits - cold_hits),
      (unsigned long long)direct,
      wait_ms,
      sync_ms,
      pread_worker_ms,
      copy_worker_ms);
  printf(
      "DECODE_RESULT steps=%d total_ms=%.6f attention_ms=%.6f router_ms=%.6f read_ms=%.6f native_ms=%.6f feedback_ms=%.6f head_ms=%.6f direct_bytes=%llu cache_hits=%llu cache_requests=%d context_creates_in_loop=%llu context_frees_in_loop=%llu memory_registers_in_loop=%llu memory_deregisters_in_loop=%llu rpc_allocations_in_loop=%llu rpc_frees_in_loop=%llu\n",
      steps,
      total,
      attention,
      router,
      read,
      native,
      feedback,
      head,
      (unsigned long long)direct,
      (unsigned long long)hits,
      steps * 320,
      (unsigned long long)(rt.context_creates - creates),
      (unsigned long long)(rt.context_frees - frees),
      (unsigned long long)(rt.memory_registers - registers),
      (unsigned long long)(rt.memory_deregisters - deregisters),
      (unsigned long long)(rt.buffer_allocations - allocations),
      (unsigned long long)(rt.buffer_frees - deallocations));
  if (rt.context_creates != creates || rt.context_frees != frees ||
      rt.memory_registers != registers ||
      rt.memory_deregisters != deregisters ||
      rt.buffer_allocations != allocations || rt.buffer_frees != deallocations)
    throw std::runtime_error("changed context/buffer lifetime");
  memory_snapshot("after_decode");
  session->cache->verify_cache();
  std::vector<int32_t> final_cache;
  for (int layer = 0; layer < 40; layer++)
    for (int slot = 0; slot < session->cache->slots_per_layer(); slot++)
      final_cache.push_back(session->cache->expert_at(layer, slot));
  save(
      model_root + "/cache-final-ids.i32",
      final_cache.data(),
      final_cache.size() * 4);
  const auto prefix = model_root + "/decode-pte-0-";
  save(prefix + "hidden.f32", hidden.data(), hidden.size() * 4);
  save(prefix + "logits.f32", logits.data(), logits.size() * 4);
  save(prefix + "input.i64", input_ids.data(), input_ids.size() * 8);
  save(prefix + "output.i64", output_ids.data(), output_ids.size() * 8);
  for (int layer = 0; layer < 40; layer++)
    for (int c = 0; c < 2; c++)
      session->attention[layer]->save_cache(
          prefix + "layer" + std::to_string(layer) + "-cache" +
              std::to_string(c) + ".u16",
          c);
  session.reset();
  embedded_binaries.clear();
  return 0;
} catch (const std::exception& e) {
  fprintf(stderr, "FATAL %s\n", e.what());
  session.reset();
  return 1;
}

extern "C" int qwen35_decode_run(int argc, const char** argv) {
  return decode_main(argc, argv);
}

#ifndef MOE_DECODE_LIBRARY
int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  std::vector<const char*> arguments(argv, argv + argc);
  return decode_main(argc, arguments.data());
}
#endif
