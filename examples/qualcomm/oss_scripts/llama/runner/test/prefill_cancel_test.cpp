/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/client_mem.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/prompt_processor.h>
#include <executorch/runtime/platform/runtime.h>
#include <algorithm>
#include <functional>
#include <iostream>
#include <stdexcept>

using example::ClientMem;
using example::DecoderRunner;
using example::KVManager;
using example::PromptProcessor;
using executorch::extension::Module;
using executorch::runtime::EValue;
using executorch::runtime::MethodMeta;
using executorch::runtime::Result;

static void check(bool value, const char* message) {
  if (!value) {
    throw std::runtime_error(message);
  }
}

class StopModule : public Module {
 public:
  using Module::Module;
  std::function<void()> before_step;
  std::function<void()> after_step;

  Result<std::vector<EValue>> execute(
      const std::string& method,
      const std::vector<EValue>& inputs) override {
    if (before_step) {
      before_step();
    }
    auto result = Module::execute(method, inputs);
    if (after_step) {
      after_step();
    }
    return result;
  }
};

class InspectableDecoder : public DecoderRunner {
 public:
  using DecoderRunner::DecoderRunner;

  float penalized(uint64_t token) const {
    std::vector<float> logits(16, 1.f);
    repetition_penalty_.apply(logits);
    return logits[token];
  }
};

static void test_cancel(const char* path, int32_t ar) {
  constexpr int32_t window = 512;
  constexpr int32_t cache_len = window - 1;
  constexpr int32_t context = 2048;
  const std::string method = "ar" + std::to_string(ar);
  StopModule module(path);
  InspectableDecoder decoder(&module, 16, 0.f, 2.f);
  check(
      decoder.load({method}) == executorch::runtime::Error::Ok, "load failed");
  auto meta = module.method_meta(method);
  check(meta.ok(), "missing fixture metadata");
  KVManager manager(
      {context, ar, context - 1, 1, 1, {}},
      std::make_unique<MethodMeta>(meta.get()));
  ClientMem memory;
  manager.init_cache_swa(&memory, ar, {cache_len}, {true});
  check(!manager.has_inplace_cache_updates(), "expected separate AR output");
  PromptProcessor processor(
      &decoder,
      &manager,
      method,
      {context, 1, 1, ar, 16, true, window, CacheMode::HybridCache},
      std::make_unique<MethodMeta>(meta.get()));
  processor.init_io(&memory, module.method_meta(method));
  std::atomic<bool> stop{false};
  decoder.set_stop_flag(&stop);

  // A sampled pending token is already in the penalty history. Its later KV
  // prefill must not evict one extra token from the bounded penalty window.
  std::vector<uint64_t> previous(256, 2);
  previous[ar - 1] = 5;
  previous.back() = 7;
  decoder.begin_turn(previous, true);
  std::vector<uint64_t> next_prompt(ar, 9);
  next_prompt.front() = 7;
  auto continuation = processor.prefill(next_prompt, 0, false, nullptr, true);
  check(continuation.ok(), "pending token prefill failed");
  check(
      decoder.penalized(5) == 0.5f,
      "pending token counted twice in penalty window");
  decoder.record_token(9);
  check(
      decoder.penalized(5) == 1.f,
      "penalty window did not advance on new token");

  for (int32_t history : {0, window + 3}) {
    for (int32_t stop_at : {1, 3}) {
      manager.reset_state(ar, true);
      decoder.begin_turn({}, true);
      stop = false;
      module.before_step = {};
      module.after_step = {};
      if (history) {
        auto seeded = processor.prefill(
            std::vector<uint64_t>(history, 3), 0, false, nullptr);
        check(
            seeded.ok() && seeded->num_committed_tokens == history,
            "seed failed");
      }
      int calls = 0;
      std::vector<float> before_k, before_v;
      module.before_step = [&]() {
        if (++calls == stop_at) {
          const auto* k =
              reinterpret_cast<float*>(manager.get_k_cache_()[0].buffer);
          const auto* v =
              reinterpret_cast<float*>(manager.get_v_cache_()[0].buffer);
          before_k.assign(k, k + 2 * cache_len);
          before_v.assign(v, v + 2 * cache_len);
        }
      };
      module.after_step = [&]() {
        if (calls == stop_at) {
          stop = true;
        }
      };
      std::vector<uint64_t> prompt(3 * ar, 1);
      std::fill(prompt.begin() + (stop_at - 1) * ar, prompt.end(), 7);
      auto partial = processor.prefill(prompt, history, false, nullptr);
      const int32_t committed = (stop_at - 1) * ar;
      check(
          partial.ok() && !partial->next_token,
          "cancel must return progress without a token");
      check(
          partial->num_committed_tokens == committed,
          "wrong committed AR count");
      check(calls == stop_at, "executed another AR after stop");
      check(
          std::equal(
              before_k.begin(),
              before_k.end(),
              reinterpret_cast<float*>(manager.get_k_cache_()[0].buffer)),
          "cancelled AR overwrote K history");
      check(
          std::equal(
              before_v.begin(),
              before_v.end(),
              reinterpret_cast<float*>(manager.get_v_cache_()[0].buffer)),
          "cancelled AR overwrote V history");
      check(
          decoder.penalized(7) == 1.f,
          "uncommitted token entered repetition history");
      check(
          decoder.penalized(1) == (committed ? 0.5f : 1.f),
          "committed penalty missing");
      check(
          decoder.penalized(3) == (history && committed < 256 ? 0.5f : 1.f),
          "cancelled AR advanced the penalty window or old history was retained");

      stop = false;
      module.before_step = {};
      module.after_step = {};
      const int32_t resumed_count = ar + 1;
      auto resumed = processor.prefill(
          std::vector<uint64_t>(resumed_count, 9),
          history + committed,
          false,
          nullptr);
      check(
          resumed.ok() && resumed->next_token &&
              resumed->num_committed_tokens == resumed_count,
          "continuation or final partial AR failed");
      check(decoder.penalized(9) == 0.5f, "resumed penalty missing");
      check(
          decoder.penalized(1) ==
              (committed && resumed_count < 256 ? 0.5f : 1.f),
          "resumed penalty window did not evict old tokens");
      std::vector<float> expected(cache_len, 0);
      for (int t = 0; t < history + committed + resumed_count; ++t) {
        expected[t % cache_len] = t < history ? 3.f
            : t < history + committed         ? 1.f
                                              : 9.f;
      }
      const auto* k =
          reinterpret_cast<float*>(manager.get_k_cache_()[0].buffer);
      const auto* v =
          reinterpret_cast<float*>(manager.get_v_cache_()[0].buffer);
      for (int t = 0; t < cache_len; ++t) {
        for (int d = 0; d < 2; ++d) {
          check(
              k[d * cache_len + t] == expected[t],
              "wrong resumed K ring position");
          check(v[t * 2 + d] == expected[t], "wrong resumed V ring position");
        }
      }
    }
  }
}

int main(int argc, char** argv) {
  executorch::runtime::runtime_init();
  check(argc == 2, "expected fixture PTE path");
  for (int32_t ar : {4, 128}) {
    test_cancel(argv[1], ar);
  }
  std::cout
      << "AR cancellation, committed progress, penalty and SWA continuation passed\n";
}
