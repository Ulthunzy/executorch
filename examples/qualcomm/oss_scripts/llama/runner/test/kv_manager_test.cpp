/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/client_mem.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/kv_manager.h>
#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/runtime.h>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <stdexcept>

using example::ClientMem;
using example::KVManager;
using executorch::runtime::MethodMeta;
using executorch::runtime::Program;

void check(bool value, const char* message) {
  if (!value) {
    throw std::runtime_error(message);
  }
}

template <typename T>
void check_masks_and_cache(const MethodMeta& meta, T visible) {
  constexpr int32_t cache_len = 7;
  constexpr int32_t window = 8;
  constexpr int32_t head_dim = 4;
  ClientMem memory;
  KVManager manager({32, 16, 31, 1, 1, {}}, std::make_unique<MethodMeta>(meta));
  manager.init_cache_swa(&memory, 16, {cache_len}, {true});
  for (int32_t prefill_ar : {1, 2, 8, 16}) {
    manager.reset_state(prefill_ar, true);
    int32_t past = 0;
    std::vector<int32_t> slot_tokens(cache_len, -1);
    for (int32_t ar :
         {prefill_ar, prefill_ar, prefill_ar, prefill_ar, 1, prefill_ar}) {
      manager.rearrange_cache_swa(ar);
      std::vector<int32_t> parents(ar);
      std::iota(parents.begin(), parents.end(), -1);
      const int32_t context = cache_len + ar;
      std::vector<T> mask(ar * context);
      auto verify = [&]() {
        for (int32_t i = 0; i < ar; ++i) {
          for (int32_t j = 0; j < cache_len; ++j) {
            bool expected =
                slot_tokens[j] >= 0 && past + i - slot_tokens[j] < window;
            check(
                (mask[i * context + j] == visible) == expected,
                "cache mask mismatch");
          }
          for (int32_t j = 0; j < ar; ++j) {
            bool expected = j <= i && i - j < window;
            check(
                (mask[i * context + cache_len + j] == visible) == expected,
                "chunk mask mismatch");
          }
        }
      };
      manager.init_attention_mask_swa(
          reinterpret_cast<std::byte*>(mask.data()),
          parents,
          ar,
          past,
          context,
          window);
      verify();
      for (int32_t updates : {ar, std::max(1, ar / 2), ar}) {
        auto* k_out =
            reinterpret_cast<float*>(manager.get_k_cache_()[0].output_buffer);
        auto* v_out =
            reinterpret_cast<float*>(manager.get_v_cache_()[0].output_buffer);
        for (int32_t d = 0; d < head_dim; ++d) {
          for (int32_t t = 0; t < ar; ++t) {
            k_out[d * ar + t] = v_out[t * head_dim + d] =
                past + t + 1 + 1000 * d;
          }
        }
        manager.update_cache_swa(ar, past, updates, {});
        for (int32_t t = 0; t < updates; ++t) {
          slot_tokens[(past + t) % cache_len] = past + t;
        }
        const auto* k_in =
            reinterpret_cast<const float*>(manager.get_k_cache_()[0].buffer);
        const auto* v_in =
            reinterpret_cast<const float*>(manager.get_v_cache_()[0].buffer);
        for (int32_t d = 0; d < head_dim; ++d) {
          for (int32_t t = 0; t < cache_len; ++t) {
            if (slot_tokens[t] < 0) {
              continue;
            }
            float expected = slot_tokens[t] + 1 + 1000 * d;
            check(k_in[d * cache_len + t] == expected, "K ring mismatch");
            check(v_in[t * head_dim + d] == expected, "V ring mismatch");
          }
        }
        manager.update_attention_mask_swa(
            reinterpret_cast<std::byte*>(mask.data()),
            parents,
            ar,
            past,
            updates,
            context,
            window);
        past += updates;
        verify();
      }
    }
  }

  manager.reset_state(8, true);
  const std::vector<int32_t> parents{-1, 0, 1, 2, 1, 4, 5, 6};
  const std::vector<int32_t> offsets{0, 1, 2, 3, 2, 3, 4, 5};
  std::vector<T> mask(8 * 15);
  manager.init_attention_mask_swa(
      reinterpret_cast<std::byte*>(mask.data()), parents, 8, 0, 15, 3, offsets);
  for (int32_t i = 0; i < 8; ++i) {
    std::vector<bool> expected(8, false);
    for (int32_t j = i; j >= 0; j = parents[j]) {
      expected[j] = offsets[i] - offsets[j] < 3;
    }
    for (int32_t j = 0; j < 8; ++j) {
      check(
          (mask[i * 15 + cache_len + j] == visible) == expected[j],
          "lookahead window mismatch");
    }
  }
}

int main(int argc, char** argv) {
  executorch::runtime::runtime_init();
  check(argc == 2, "expected fixture PTE path");
  auto loader = executorch::extension::FileDataLoader::from(argv[1]);
  check(loader.ok(), "fixture file load failed");
  auto program = Program::load(&loader.get());
  check(program.ok(), "fixture program load failed");

  for (const char* name :
       {"prefill",
        "decode",
        "recurrent",
        "circular_float",
        "circular_uint16"}) {
    auto meta = program->method_meta(name);
    check(meta.ok(), "missing fixture method");
    bool circular = std::string(name).find("circular") == 0;
    auto mode = KVManager::infer_cache_update_mode(
        meta->input_tensor_meta(4).get(),
        meta->input_tensor_meta(5).get(),
        meta->output_tensor_meta(1).get(),
        meta->output_tensor_meta(2).get(),
        circular);
    auto expected = circular               ? KVManager::Metadata::kCircular
        : std::string(name) == "recurrent" ? KVManager::Metadata::kFullOverwrite
                                           : KVManager::Metadata::kAppend;
    check(mode == expected, "cache mode misclassified");
  }
  check_masks_and_cache<float>(
      program->method_meta("circular_float").get(), 0.f);
  check_masks_and_cache<uint16_t>(
      program->method_meta("circular_uint16").get(), 65535);
  {
    const auto meta = program->method_meta("mixed").get();
    std::vector<KVManager::Metadata::CacheSpec> specs;
    for (int layer = 0; layer < 2; ++layer) {
      specs.push_back(
          {meta.input_tensor_meta(4 + layer)->nbytes(),
           meta.output_tensor_meta(1 + layer)->nbytes(),
           meta.input_tensor_meta(6 + layer)->nbytes(),
           meta.output_tensor_meta(3 + layer)->nbytes(),
           layer == 0 ? KVManager::Metadata::kFullOverwrite
                      : KVManager::Metadata::kAppend,
           layer == 0 ? 30000 : 128,
           layer == 0 ? 31000 : 129});
    }
    ClientMem memory;
    KVManager manager(
        {8, 1, 7, 1, 2, specs}, std::make_unique<MethodMeta>(meta));
    manager.init_cache(&memory, 1);
    const auto& keys = manager.get_k_cache_();
    const auto& values = manager.get_v_cache_();
    check(
        reinterpret_cast<const uint16_t*>(keys[0].buffer)[0] == 30000,
        "GDN state must keep UINT16 zero point");
    check(
        reinterpret_cast<const uint8_t*>(keys[1].buffer)[0] == 128,
        "FA cache must use UINT8 zero point");
    for (int step = 0; step < 4; ++step) {
      for (int dim = 0; dim < 4; ++dim) {
        reinterpret_cast<uint8_t*>(keys[1].output_buffer)[dim] =
            step + dim * 10;
        reinterpret_cast<uint8_t*>(values[1].output_buffer)[dim] =
            step + dim * 20;
      }
      manager.update_cache(1, step, 1, {});
    }
    for (int step = 0; step < 4; ++step) {
      for (int dim = 0; dim < 4; ++dim) {
        check(
            reinterpret_cast<const uint8_t*>(keys[1].buffer)[dim * 7 + step] ==
                step + dim * 10,
            "mixed cache K append used the wrong byte stride");
        check(
            reinterpret_cast<const uint8_t*>(values[1].buffer)[step * 4 + dim] ==
                step + dim * 20,
            "mixed cache V append used the wrong byte stride");
      }
    }
  }
  std::cout << "KV modes, ring cache, AR transitions and SWA masks passed"
            << std::endl;
}
