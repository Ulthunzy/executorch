/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/client_mem.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/kv_manager.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/multimodal_prompt_processor.h>
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

template <typename Code>
void check_mapped_cache(const MethodMeta& meta) {
  class MappedMemory final : public example::IMemAlloc {
   public:
    std::vector<std::vector<std::byte>> buffers;
    std::vector<std::vector<std::byte>> mapped;
    std::byte* allocate(size_t size) override {
      buffers.emplace_back(size);
      return buffers.back().data();
    }
    MappedBuffer allocate_mapped(size_t size) override {
      mapped.emplace_back(size);
      return {mapped.back().data(), static_cast<int32_t>(16 + mapped.size())};
    }
    void add_memory_info(void*, size_t, executorch::runtime::TensorInfo)
        override {}
  } memory;
  const auto mode = KVManager::infer_cache_update_mode(
      meta.input_tensor_meta(3).get(),
      meta.input_tensor_meta(4).get(),
      meta.output_tensor_meta(1).get(),
      meta.output_tensor_meta(2).get(),
      false);
  check(
      mode == KVManager::Metadata::kMappedAppend, "mapped cache misclassified");
  const int head_dim = meta.output_tensor_meta(1)->sizes()[2];
  const size_t raw_bytes = 256 * head_dim * sizeof(Code);
  KVManager::Metadata::CacheSpec spec{
      12,
      128 * head_dim * sizeof(Code),
      12,
      128 * head_dim * sizeof(Code),
      mode,
      120,
      121,
      256};
  KVManager manager(
      {256, 128, 256, 1, 1, {spec}}, std::make_unique<MethodMeta>(meta));
  manager.init_cache(&memory, 128);
  const auto& key = manager.get_k_cache_()[0];
  const auto& value = manager.get_v_cache_()[0];
  int32_t lengths[3]{};
  const auto* kd = reinterpret_cast<const int32_t*>(key.buffer);
  const auto* vd = reinterpret_cast<const int32_t*>(value.buffer);
  manager.prepare_attention_lengths(
      reinterpret_cast<std::byte*>(lengths), 3, 0);
  check(
      lengths[0] == 3 && lengths[1] == 0 && lengths[2] == 0,
      "tail lengths mismatch");
  check(
      kd[0] == 17 && vd[0] == 18 &&
          uint32_t(kd[1]) ==
              uint32_t(
                  (1u << 30) | raw_bytes | (head_dim == 256 ? (1u << 31) : 0u)),
      "FD descriptor mismatch");
  check(
      uint32_t(vd[1]) == ((1u << 30) | raw_bytes),
      "value descriptor flags mismatch");
  const size_t expected_key_bytes = head_dim == 256
      ? ((raw_bytes + 4095) & ~size_t(4095)) + 1 * 256 * 512 * 4
      : raw_bytes;
  check(
      memory.mapped[0].size() == expected_key_bytes,
      "packed allocation size mismatch");
  check(memory.mapped[1].size() == raw_bytes, "value allocation size mismatch");
  const int32_t epoch = kd[2];
  for (int dim = 0; dim < head_dim; ++dim) {
    for (int token = 0; token < 128; ++token) {
      reinterpret_cast<Code*>(key.output_buffer)[dim * 128 + token] =
          20 * dim + token;
      reinterpret_cast<Code*>(value.output_buffer)[token * head_dim + dim] =
          20 * dim + token;
    }
  }
  manager.update_cache(128, 0, 3, {});
  manager.rearrange_cache(1);
  manager.prepare_attention_lengths(
      reinterpret_cast<std::byte*>(lengths), 1, 3);
  for (int dim = 0; dim < head_dim; ++dim) {
    reinterpret_cast<Code*>(key.output_buffer)[dim] = 20 * dim + 3;
    reinterpret_cast<Code*>(value.output_buffer)[dim] = 20 * dim + 3;
  }
  manager.update_cache(1, 3, 1, {});
  const auto* raw_k = reinterpret_cast<const Code*>(memory.mapped[0].data());
  const auto* raw_v = reinterpret_cast<const Code*>(memory.mapped[1].data());
  for (int dim = 0; dim < head_dim; ++dim) {
    for (int token = 0; token < 256; ++token) {
      check(
          raw_k[dim * 256 + token] ==
              (token < 4 ? Code(20 * dim + token) : Code(120)),
          "mapped K stride or tail mismatch");
      check(
          raw_v[token * head_dim + dim] ==
              (token < 4 ? Code(20 * dim + token) : Code(121)),
          "mapped V stride or tail mismatch");
    }
  }
  manager.reset_state(128, true);
  manager.prepare_attention_lengths(
      reinterpret_cast<std::byte*>(lengths), 7, 0);
  check(
      kd[2] == epoch + 1 && vd[2] == kd[2],
      "reset did not invalidate packed KV");
}

int main(int argc, char** argv) {
  executorch::runtime::runtime_init();
  example::MultimodalPromptProcessor::Metadata bounds{};
  bounds.context_len = 1000;
  bounds.ar_len = 128;
  bounds.max_prefill_past = 1000;
  bounds.position_capacity = 1128;
  for (int length : {1, 896, 897, 999, 1000}) {
    check(bounds.can_prefill(0, length, 1000), "valid tail block rejected");
  }
  check(bounds.can_prefill(950, 50, 1000), "valid unaligned turn rejected");
  check(!bounds.can_prefill(950, 51, 1000), "turn exceeds mapped KV capacity");
  check(
      !bounds.can_prefill(0, 1001, 1000), "prompt exceeds mapped KV capacity");
  check(!bounds.can_prefill(0, 0, 1000), "empty prompt accepted");
  check(!bounds.can_prefill(-1, 10, 1000), "negative start accepted");
  bounds.position_capacity = 1024;
  check(bounds.can_prefill(0, 999, 1000), "rounded position capacity rejected");
  check(!bounds.can_prefill(950, 50, 1000), "position table overflow accepted");
  bounds.position_capacity = 0;
  check(!bounds.can_prefill(0, 1, 1000), "missing position capacity accepted");
  bounds.context_len = 1128;
  bounds.position_capacity = 1128;
  bounds.max_prefill_past = 896;
  check(bounds.can_prefill(0, 999, 1127), "SDK prefill capacity regressed");
  check(!bounds.can_prefill(900, 50, 1127), "SDK past limit was bypassed");
  check(argc == 2, "expected fixture PTE path");
  auto loader = executorch::extension::FileDataLoader::from(argv[1]);
  check(loader.ok(), "fixture file load failed");
  auto program = Program::load(&loader.get());
  check(program.ok(), "fixture program load failed");
  check_mapped_cache<uint16_t>(program->method_meta("mapped").get());
  check_mapped_cache<uint8_t>(program->method_meta("mapped_uint8").get());
  check_mapped_cache<uint8_t>(program->method_meta("mapped_hmx8").get());
  check_mapped_cache<uint16_t>(program->method_meta("mapped_hmx16").get());

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
            reinterpret_cast<const uint8_t*>(
                values[1].buffer)[step * 4 + dim] == step + dim * 20,
            "mixed cache V append used the wrong byte stride");
      }
    }
  }
  std::cout << "KV modes, ring cache, AR transitions and SWA masks passed"
            << std::endl;
}
