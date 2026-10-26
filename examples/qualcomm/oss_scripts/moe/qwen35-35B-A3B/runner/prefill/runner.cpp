#include <executorch/runtime/backend/interface.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/runtime/platform/runtime.h>
#include <cmath>
#include "prefill_session.h"
#include "vision_encoder.h"
#include "../handoff.h"
#include "../decode_api.h"
#include "../prefill_api.h"
#include "../shared/graph_binding.h"
#ifdef MOE_CHAT_RUNNER
#include "../chat/chat.h"
#ifdef MOE_JNI_RUNNER
#include "../chat/conversation.h"
#endif
#endif

using namespace executorch::runtime;
static std::string model_root;
static std::unique_ptr<PrefillSession> session;

void qwen35_prefill_append(
    const int64_t* tokens,
    size_t count,
    const float* features,
    size_t feature_count,
    const float* cos,
    const float* sin,
    size_t table_count,
    Qwen35SessionState& state,
    float* logits,
    const std::function<bool()>& stopped) {
  if (!session)
    throw std::runtime_error("prefill session is not resident");
  session->append(
      tokens,
      count,
      features,
      feature_count,
      cos,
      sin,
      table_count,
      state,
      logits,
      stopped);
}

struct Stage {
  uint32_t kind, layer, plan_tokens;
};

// PTE 只描述 122 个阶段：embedding、40×(Attention/Router/Native)、head。
// delegate binary 在 init 阶段借用 PTE 内存，执行阶段由同一个 PrefillSession
// 调度。
class PrefillBackend final : public BackendInterface {
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
      uint32_t version, kind, layer, plan_tokens;
      uint64_t size, aux_offset, aux_size;
      if (memcmp(p, "Q36PREF1", 8))
        return Error::DelegateInvalidCompatibility;
      memcpy(&version, p + 8, 4);
      memcpy(&kind, p + 12, 4);
      memcpy(&layer, p + 16, 4);
      memcpy(&plan_tokens, p + 20, 4);
      memcpy(&size, p + 24, 8);
      memcpy(&aux_offset, p + 32, 8);
      memcpy(&aux_size, p + 40, 8);
      if (!((version == 1 && plan_tokens == 0) ||
            (version == 3 && getenv("SHARED_MODEL_ROOT") &&
             plan_tokens >= 128 && plan_tokens <= 1024 &&
             plan_tokens % 128 == 0) ||
            (version == 2 && plan_tokens > 128 && plan_tokens <= 1024 &&
             plan_tokens % 128 == 0)) ||
          kind > 4 || layer >= 40 || size > processed->size() - 128 ||
          (aux_size &&
           (aux_offset > processed->size() ||
            aux_size > processed->size() - aux_offset)))
        return Error::DelegateInvalidCompatibility;
      if (version == 3) {
        if (size || aux_size)
          return Error::DelegateInvalidCompatibility;
        return static_cast<DelegateHandle*>(
            new Stage{kind, layer, plan_tokens});
      }
      std::string base = model_root + "/layer" + std::to_string(layer);
      if (kind == 1) {
        if (!size)
          return Error::DelegateInvalidCompatibility;
        auto path = layer % 4 == 3
            ? model_root + "/cold/layer" + std::to_string(layer)
            : base;
        embedded_binaries[path + "/forward_0.bin"] = {p + 128, size};
        if (layer % 4 == 3) {
          if (!aux_size)
            return Error::DelegateInvalidCompatibility;
          embedded_binaries[base + "/forward_0.bin"] = {
              p + aux_offset, aux_size};
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
      return static_cast<DelegateHandle*>(
          new Stage{kind, layer, version == 1 ? 128 : plan_tokens});
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
        throw std::runtime_error("prefill session not initialized");
      auto stage = *static_cast<Stage*>(handle);
      if (stage.plan_tokens != uint32_t(session->sequence_tokens))
        throw std::runtime_error("prefill stage sequence length mismatch");
      const size_t counts[] = {2, 3, 4, 6, 2};
      if (args.size() != counts[stage.kind])
        throw std::runtime_error("stage arguments");
      auto input = [&](int i) {
        return args[i]->toTensor().const_data_ptr<float>();
      };
      auto output = [&](int i) {
        return args[i]->toTensor().mutable_data_ptr<float>();
      };
      const int chunks = session->sequence_tokens / 128;
      // PTE 的 workspace 覆盖整段输入，QNN/Native 每次仍处理 128 个 token。
      // Attention 的 state 在每块之后立即反馈，MoE 权重则复用到本层结束。
      if (stage.kind == 3 && session->layer_memo && chunks >= 4) {
        session->chunk_start = 0;
        session->experts(
            stage.layer,
            input(0),
            args[1]->toTensor().const_data_ptr<int32_t>(),
            input(2),
            input(3),
            input(4),
            output(5),
            session->sequence_tokens);
        return Error::Ok;
      }
      for (int chunk = 0; chunk < chunks; chunk++) {
        if (chunks > 1)
          session->chunk_start = chunk * 128;
        session->valid_tokens = std::min(
            128, session->sequence_valid_tokens - session->chunk_start);
        const size_t hidden_offset = size_t(chunk) * 128 * 2048;
        const size_t route_offset = size_t(chunk) * 128 * 8;
        switch (stage.kind) {
          case 0:
            session->embed(
                args[0]->toTensor().const_data_ptr<int64_t>() + chunk * 128,
                output(1) + hidden_offset);
            break;
          case 1:
            session->attend(
                stage.layer,
                input(0) + hidden_offset,
                output(1) + hidden_offset,
                output(2) + hidden_offset);
            if (chunks > 1) {
              auto started = Clock::now();
              session->attention[stage.layer]->feedback();
              session->metrics.feedback += elapsed(started);
            }
            break;
          case 2:
            session->route(
                stage.layer,
                input(0) + hidden_offset,
                args[1]->toTensor().mutable_data_ptr<int32_t>() + route_offset,
                output(2) + route_offset,
                output(3) + hidden_offset);
            break;
          case 3:
            session->experts(
                stage.layer,
                input(0) + hidden_offset,
                args[1]->toTensor().const_data_ptr<int32_t>() + route_offset,
                input(2) + route_offset,
                input(3) + hidden_offset,
                input(4) + hidden_offset,
                output(5) + hidden_offset);
            break;
          case 4:
            if (chunk + 1 == chunks)
              session->project(input(0) + hidden_offset, output(1));
            break;
        }
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

static PrefillBackend backend_instance;
static Backend backend{"Qwen36PrefillStage", &backend_instance};
static auto registered = register_backend(backend);

static int prefill_main(int argc, char** argv) try {
#ifndef MOE_JNI_RUNNER
  setvbuf(stdout, nullptr, _IONBF, 0);
#endif
  runtime_init();
  if (argc != 4)
    throw std::runtime_error("usage: pte_runner ROOT SAMPLES WARMUPS");
  model_root = argv[1];
  int samples = atoi(argv[2]), warmups = atoi(argv[3]);
  if (samples < 1 || warmups < 0)
    throw std::runtime_error("sample count");
  const auto setup = Clock::now();
  executorch::extension::Module module(
      getenv("CHAT_PREFILL_PLAN") ? std::string(getenv("CHAT_PREFILL_PLAN"))
                                  : model_root + "/prefill.pte",
      executorch::extension::Module::LoadMode::Mmap);
  auto error = module.load_method("prefill_forward");
  if (error != Error::Ok)
    throw std::runtime_error("PTE load_method " + std::to_string(int(error)));
  if (embedded_binaries.size() != 91 && !getenv("SHARED_MODEL_ROOT"))
    throw std::runtime_error("missing embedded context binaries");
  auto meta = module.method_meta("prefill_forward");
  if (!meta.ok())
    throw std::runtime_error("prefill method metadata");
  auto input_meta = meta->input_tensor_meta(0);
  if (!input_meta.ok() || input_meta->sizes().size() != 2 ||
      input_meta->sizes()[0] != 1)
    throw std::runtime_error("prefill input dimensions");
  const int plan_tokens = input_meta->sizes()[1];
  session = std::make_unique<PrefillSession>(model_root, plan_tokens);

  struct SessionLifetime {
    ~SessionLifetime() {
      session.reset();
      embedded_binaries.clear();
    }
  } session_lifetime;

  const double setup_ms = elapsed(setup);

  const int prompt_tokens =
      getenv("PREFILL_TOKENS") ? atoi(getenv("PREFILL_TOKENS")) : 128;
  const int valid_tokens = getenv("PREFILL_VALID_TOKENS")
      ? atoi(getenv("PREFILL_VALID_TOKENS"))
      : prompt_tokens;
  if (valid_tokens < 1 || valid_tokens > session->logical_capacity ||
      valid_tokens > prompt_tokens || prompt_tokens - valid_tokens >= 128)
    throw std::runtime_error("invalid prefill padding length");
  const int max_prefill_tokens = session->attention[3]->history + 128;
  if (samples != 1 || warmups || prompt_tokens < 128 ||
      prompt_tokens > max_prefill_tokens || prompt_tokens % 128 ||
      (plan_tokens != 128 && plan_tokens != prompt_tokens))
    throw std::runtime_error(
        "prefill sequence exceeds the compiled KV history or plan alignment");
  auto data = bytes(model_root + "/tokens.i64");
  if (data.size() != size_t(prompt_tokens) * 8)
    throw std::runtime_error("token input size");
  printf(
      "PTE_READY layers=40 ar_len=128 prompt_tokens=%d chunks=%d plan_calls=122 embedded_qnn_contexts=%d expert_slots=%zu expert_buffer_bytes=%zu rpc_buffer_bytes=%llu peak_rpc_buffer_bytes=%llu setup_ms=%.6f context_creates=%llu memory_registers=%llu hot=0\n",
      prompt_tokens,
      prompt_tokens / 128,
      getenv("SHARED_MODEL_ROOT") ? 0 : 81,
      session->expert_slots.size() * session->capacity,
      session->expert_slots.size() * session->expert_slots[0]->size,
      (unsigned long long)session->runtime.buffer_bytes,
      (unsigned long long)session->runtime.peak_buffer_bytes,
      setup_ms,
      (unsigned long long)session->runtime.context_creates,
      (unsigned long long)session->runtime.memory_registers);
  session->reset();
#ifdef MOE_CHAT_RUNNER
  session->image_features = moe_chat::initial_image(session->vision.get());
#else
  if (session->vision)
    session->image_features = session->vision->encode();
#endif
  std::vector<int32_t> all_routes, seeds;
  std::vector<uint64_t> seed_stamps;
  double graph_total = 0;
  uint64_t direct_total = 0;
  int native_total = 0;
  const auto sequence_begin = Clock::now();
  const int chunks_per_plan = plan_tokens / 128;
  for (int chunk = 0; chunk < prompt_tokens / 128; chunk += chunks_per_plan) {
    const int active_tokens = std::min(plan_tokens, valid_tokens - chunk * 128);
    session->sequence_valid_tokens =
        active_tokens + (plan_tokens == 128 ? chunk * 128 : 0);
    for (const auto& cache : session->caches)
      for (const auto& entry : cache.entries) {
        seeds.push_back(entry.expert);
        seed_stamps.push_back(entry.stamp);
      }
    auto tokens = executorch::extension::make_tensor_ptr(
        {1, plan_tokens},
        data.data() + chunk * 128 * 8,
        executorch::aten::ScalarType::Long);
    auto& rt = session->runtime;
    auto creates = rt.context_creates, frees = rt.context_frees,
         regs = rt.memory_registers, deregs = rt.memory_deregisters,
         allocs = rt.buffer_allocations, deallocs = rt.buffer_frees;
    int before[4];
    check(integer_rpc_counts(session->core.handle, before, 4), "counts before");
    uint64_t physical_before = physical_reads();
    auto t = Clock::now();
    auto result =
        module.execute("prefill_forward", std::vector<EValue>{*tokens});
    const double total_ms = elapsed(t);
    uint64_t physical = physical_reads() - physical_before;
    if (!result.ok() || result->size() != 2)
      throw std::runtime_error("PTE execute/output failure");
    auto& m = session->metrics;
    int counts[4];
    check(integer_rpc_counts(session->core.handle, counts, 4), "counts after");
    for (int i = 0; i < 4; i++)
      counts[i] -= before[i];
    if (m.calls[0] != chunks_per_plan || m.calls[1] != 40 * chunks_per_plan ||
        m.calls[2] != 40 * chunks_per_plan ||
        m.calls[3] != 40 * chunks_per_plan || m.calls[4] != 1 ||
        m.assignments != active_tokens * 8 * 40 ||
        counts[0] != m.native_calls || counts[1] != int(m.requests) ||
        counts[2] != active_tokens * 8 * 40 ||
        counts[3] != 40 * chunks_per_plan ||
        session->chunk_start != chunk * 128 + active_tokens)
      throw std::runtime_error("chunk task coverage");
    if (rt.context_creates != creates || rt.context_frees != frees ||
        rt.memory_registers != regs || rt.memory_deregisters != deregs ||
        rt.buffer_allocations != allocs || rt.buffer_frees != deallocs)
      throw std::runtime_error("changed runtime buffer/context lifetime");
    for (const auto& v : *result) {
      auto& tensor = v.toTensor();
      auto* values = tensor.const_data_ptr<float>();
      for (int i = 0; i < tensor.numel(); i++)
        if (!std::isfinite(values[i]))
          throw std::runtime_error("nonfinite PTE output");
    }
    graph_total += total_ms;
    direct_total += m.direct;
    native_total += m.native_calls;
    if (chunks_per_plan > 1) {
      all_routes = session->sequence_routes;
      seeds = session->sequence_seeds;
      seed_stamps = session->sequence_stamps;
    } else
      for (const auto& routes : session->route_history)
        all_routes.insert(all_routes.end(), routes.begin(), routes.end());
    printf(
        "PTE_RESULT chunk=%d start=%d plan_tokens=%d total_ms=%.6f tokens_per_second=%.6f embedding_ms=%.6f attention_ms=%.6f router_ms=%.6f moe_ms=%.6f head_ms=%.6f read_ms=%.6f native_ms=%.6f feedback_ms=%.6f direct_bytes=%llu physical_read_bytes=%llu hits=%llu requests=%llu seed_hits=%llu native_calls=%d assignments=%d context_creates_in_loop=%llu context_frees_in_loop=%llu memory_registers_in_loop=%llu memory_deregisters_in_loop=%llu rpc_allocations_in_loop=%llu rpc_frees_in_loop=%llu\n",
        chunk,
        chunk * 128,
        plan_tokens,
        total_ms,
        active_tokens * 1000. / total_ms,
        m.embedding,
        m.attention,
        m.router,
        m.moe,
        m.head,
        m.read,
        m.native,
        m.feedback,
        (unsigned long long)m.direct,
        (unsigned long long)physical,
        (unsigned long long)m.hits,
        (unsigned long long)m.requests,
        (unsigned long long)m.seed_hits,
        m.native_calls,
        m.assignments,
        (unsigned long long)(rt.context_creates - creates),
        (unsigned long long)(rt.context_frees - frees),
        (unsigned long long)(rt.memory_registers - regs),
        (unsigned long long)(rt.memory_deregisters - deregs),
        (unsigned long long)(rt.buffer_allocations - allocs),
        (unsigned long long)(rt.buffer_frees - deallocs));
    printf(
        "PREFILL_MEMO reused_bytes=%llu seed_copy_ms=%.6f retained_copy_ms=%.6f\n",
        (unsigned long long)m.layer_reuse_bytes,
        m.layer_seed_copy,
        m.staged_copy);
    printf(
        "PREFILL_PROFILE batch_setup_ms=%.6f batch_wait_ms=%.6f io_wall_ms=%.6f io_overlap_ms=%.6f seed_copy_ms=%.6f retained_copy_ms=%.6f input_copy_sync_ms=%.6f pread_worker_ms=%.6f payload_copy_worker_ms=%.6f ready_native_ms=%.6f missing_native_ms=%.6f ready_calls=%d missing_calls=%d\n",
        m.batch_setup,
        m.io_wait,
        m.io_wall,
        m.io_overlap,
        m.layer_seed_copy,
        m.staged_copy,
        m.input_copy_sync,
        m.pread_worker,
        m.payload_copy_worker,
        m.ready_native,
        m.missing_native,
        m.ready_calls,
        m.missing_calls);
    printf(
        "PREFILL_COPY seed_wait_ms=%.6f seed_overlap_ms=%.6f retained_wait_ms=%.6f retained_overlap_ms=%.6f retained_prepare_ms=%.6f input_io_overlap_ms=%.6f router_io_overlap_ms=%.6f attention_io_overlap_ms=%.6f\n",
        m.seed_wait,
        m.seed_overlap,
        m.retained_wait,
        m.retained_overlap,
        m.retained_prepare,
        m.input_io_overlap,
        m.router_io_overlap,
        m.attention_io_overlap);
    if (chunk + chunks_per_plan == prompt_tokens / 128) {
      if (session->image_cursor != session->image_features.size())
        throw std::runtime_error(
            "encoded image features were not fully consumed");
      const double loop_ms = elapsed(sequence_begin);
      printf(
          "PREFILL_TOTAL tokens=%d chunks=%d graph_ms=%.6f loop_ms=%.6f tokens_per_second=%.6f direct_bytes=%llu native_calls=%d padded_tokens=%d\n",
          valid_tokens,
          prompt_tokens / 128,
          graph_total,
          loop_ms,
          1000 * valid_tokens / loop_ms,
          (unsigned long long)direct_total,
          native_total,
          prompt_tokens);
      auto* scores = result->at(0).toTensor().const_data_ptr<float>();
      const int64_t first = std::max_element(scores, scores + 248320) - scores;
      printf(
          "PREFILL_TOKEN {\"index\":0,\"id\":%lld,\"model_ms\":%.6f}\n",
          (long long)first,
          loop_ms);
      const std::string prefix = model_root + "/pte-0-";
      save(prefix + "logits.f32", scores, 248320 * 4);
      save(
          prefix + "hidden.f32",
          result->at(1).toTensor().const_data_ptr<float>() +
              (plan_tokens - 128) * 2048,
          128 * 2048 * 4);
      for (int layer = 0; layer < 40; layer++)
        for (int c = 0; c < 2; c++)
          save(
              prefix + "layer" + std::to_string(layer) + "-cache" +
                  std::to_string(c) + ".u16",
              session->attention[layer]->export_data(c),
              session->attention[layer]->export_bytes(c));
    }
  }
  save(
      model_root + "/prefill-all-routes.i32",
      all_routes.data(),
      all_routes.size() * 4);
  save(model_root + "/prefill-chunk-seeds.i32", seeds.data(), seeds.size() * 4);
  save(
      model_root + "/prefill-chunk-stamps.u64",
      seed_stamps.data(),
      seed_stamps.size() * 8);
  const auto history_root = model_root;
  if (session->vision)
    save(
        model_root + "/image-features.f32",
        session->image_features.data(),
        session->image_features.size() * 4);
  std::vector<int32_t> ids;
  std::vector<uint64_t> stamps;
  for (int layer = 0; layer < 40; layer++)
    for (int slot = 0; slot < 64; slot++) {
      ids.push_back(session->layer_history[layer][slot].expert);
      stamps.push_back(session->layer_history[layer][slot].stamp);
    }
  save(history_root + "/prefill-history-ids.i32", ids.data(), ids.size() * 4);
  save(
      history_root + "/prefill-history-stamps.u64",
      stamps.data(),
      stamps.size() * 8);
  save(
      history_root + "/prefill-routes.i32",
      session->route_history.data(),
      sizeof(session->route_history));
  printf(
      "PREFILL_REUSE_TOTAL enabled=%d expert_buffers=%zu expert_slots=%d buffer_bytes=%zu weight_copy_bytes=0 extra_read_bytes=0\n",
      session->reuse,
      session->expert_slots.size(),
      session->capacity,
      session->expert_slots.size() * session->expert_slots[0]->size);
  const char* decode_root = getenv("HANDOFF_DECODE_ROOT");
  const char* max_tokens = getenv("HANDOFF_MAX_TOKENS");
  const char* eos_ids = getenv("HANDOFF_EOS_IDS");
  if (decode_root) {
    if (!session->reuse || samples != 1 || warmups || !max_tokens || !eos_ids)
      throw std::runtime_error("invalid direct reuse handoff");
    auto started = Clock::now();
    // decode 导入同一组 DMA buffer；dup 的 fd 由导入端关闭，原 fd 仍由
    // prefill session 持有，不重新读取或复制整套常驻专家。
    ResidentExperts transfer{};
    memcpy(transfer.magic, "Q36DATA2", 8);
    transfer.stride = session->stride;
    transfer.data_offset = 4096;
    transfer.capacity = session->capacity;
    transfer.layers = 40;
    std::vector<int32_t> selected;
    std::vector<uint64_t> selected_stamps;
    uint64_t total_bytes = 0;
    for (int layer = 0; layer < 40; layer++) {
      auto& buffer = *session->expert_slots[layer];
      auto& record = transfer.layer[layer];
      record.fd = fcntl(buffer.fd, F_DUPFD, 128);
      if (record.fd < 0)
        throw std::runtime_error("duplicate reuse fd");
      record.bytes = buffer.size;
      total_bytes += buffer.size;
      for (int slot = 0; slot < session->capacity; slot++) {
        const auto entry = session->caches[layer].entries[slot];
        if (entry.expert < -1 || (entry.expert == -1 && entry.stamp))
          throw std::runtime_error("invalid reuse seed");
        record.ids[slot] = entry.expert;
        record.stamps[slot] = entry.stamp;
        selected.push_back(entry.expert);
        selected_stamps.push_back(entry.stamp);
      }
      record.fingerprint = residentFingerprint(
          static_cast<const uint8_t*>(buffer.data),
          transfer.stride,
          transfer.data_offset,
          transfer.capacity,
          record.ids);
      if (record.hot_mask)
        throw std::runtime_error("all-LRU cache forbids hot slots");
      printf(
          "PREFILL_CACHE_LAYER layer=%d hot_mask=%u hot_slots=%d\n",
          layer,
          record.hot_mask,
          __builtin_popcount(record.hot_mask));
    }
    save(
        std::string(decode_root) + "/resident-handoff.bin",
        &transfer,
        sizeof(transfer));
    save(
        std::string(decode_root) + "/reused-ids.i32",
        selected.data(),
        selected.size() * 4);
    save(
        std::string(decode_root) + "/reused-stamps.u64",
        selected_stamps.data(),
        selected_stamps.size() * 8);
    std::vector<uint32_t> masks;
    for (const auto& layer : transfer.layer)
      masks.push_back(layer.hot_mask);
    save(
        std::string(decode_root) + "/hot-masks.u32",
        masks.data(),
        masks.size() * 4);
    printf(
        "EXPERT_RESIDENT_EXPORT layers=40 experts=%d buffer_bytes=%llu metadata_bytes=%zu read_bytes=0 copy_bytes=0 export_ms=%.6f\n",
        40 * session->capacity,
        (unsigned long long)total_bytes,
        sizeof(transfer),
        elapsed(started));
  }
  if (decode_root) {
    const auto adsp = std::string(decode_root) + ";" + model_root +
        ";/odm/lib64/rfsa;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/dsp";
    setenv("ADSP_LIBRARY_PATH", adsp.c_str(), 1);
    const char* decode_async = getenv("ASYNC_DECODE_IO");
    if (decode_async)
      setenv("ASYNC_EXPERT_IO", decode_async, 1);
    session->prepare_decode();
    const auto creates = session->runtime.context_creates;
    const auto frees = session->runtime.context_frees;
    const auto registrations = session->runtime.memory_registers;
    const auto vision_context =
        session->vision ? session->vision->graph.context : nullptr;
    const auto library = std::string(decode_root) + "/libmoe_decode.so";
    void* decoder = dlopen(library.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!decoder)
      throw std::runtime_error(dlerror());
    auto run =
        reinterpret_cast<DecodeMain>(dlsym(decoder, "qwen35_decode_run"));
    if (!run) {
      const std::string message = dlerror();
      dlclose(decoder);
      throw std::runtime_error(message);
    }
    const char* arguments[] = {
        "decode_session", decode_root, max_tokens, "4096", eos_ids};
#ifdef MOE_CHAT_RUNNER
    const int result = moe_chat::converse(decoder);
#else
    const int result = run(5, arguments);
#endif
    dlclose(decoder);
    if (session->runtime.context_creates != creates ||
        session->runtime.context_frees != frees ||
        session->runtime.memory_registers != registrations ||
        (session->vision && session->vision->graph.context != vision_context))
      throw std::runtime_error("prefill/vision lifetime changed during decode");
    printf(
        "MODEL_RESIDENT retained_contexts=%llu vision_alive=%d context_frees_during_decode=0 decode_exit=%d\n",
        (unsigned long long)(session->runtime.live_contexts +
                             shared_context_count()),
        session->vision != nullptr,
        result);
    return result;
  }
  session.reset();
  embedded_binaries.clear();
  return 0;
} catch (const std::exception& e) {
  fprintf(stderr, "FATAL %s\n", e.what());
  session.reset();
  return 1;
}

#ifdef MOE_JNI_RUNNER
int qwen35_moe_run_client(moe_chat::Options options, moe_chat::Client& client) {
  return moe_chat::run_client(std::move(options), client, prefill_main);
}
#else
int main(int argc, char** argv) {
#ifdef MOE_CHAT_RUNNER
  return moe_chat::main(argc, argv, prefill_main);
#else
  return prefill_main(argc, argv);
#endif
}

#endif
