/* Copyright (c) Meta Platforms, Inc. and affiliates.
 * Licensed under the BSD-style license in the root LICENSE file.
 */

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/runner.h>
#include <nlohmann/json.hpp>
#include "driver_contract.h"
#include <regex>
#include <set>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
#include <sys/syscall.h>
#include <unistd.h>

using json = nlohmann::json;
using executorch::runtime::Error;

#ifndef ET_QNN_PPL_EVALUATION
#error Build this entry point with ET_QNN_PPL_EVALUATION
#endif

namespace example {
const uint64_t* forced_tokens = nullptr;
size_t forced_count = 0;
size_t forced_position = 0;

int32_t next_teacher_forcing_token() {
  if (forced_position >= forced_count) {
    throw std::runtime_error("Teacher-forcing tokens exhausted");
  }
  return static_cast<int32_t>(forced_tokens[forced_position++]);
}
} // namespace example

void save(const std::string& path, const json& data) {
  std::ofstream output(path + ".tmp");
  output << data.dump() << '\n';
  output.close();
  if (!output || std::rename((path + ".tmp").c_str(), path.c_str())) {
    throw std::runtime_error("Cannot save " + path);
  }
}

int main(int argc, char** argv) {
  try {
    if (argc != 4) {
      std::cerr << "Usage: ppl_driver CONFIG_JSON TOKENS_U64 OUTPUT_DIR\n";
      return 2;
    }
    std::ifstream configuration(argv[1]);
    const auto settings = json::parse(configuration);
    if (settings.at("protocol") != "executorch.ppl.driver.v1") {
      throw std::runtime_error("Unsupported PPL driver protocol");
    }
    const auto seq_len = settings.at("seq_len").get<int64_t>();
    const std::string mode = settings.at("mode");
    const std::string decoder = settings.at("decoder");
    const std::set<std::string> decoders = {"lincal", "qwen3", "qwen2_5", "llama3", "llama2"};
    if (seq_len < 2 || seq_len > std::numeric_limits<int32_t>::max() ||
        (mode != "hybrid" && mode != "kv") || !decoders.count(decoder)) {
      throw std::runtime_error("Unsupported seq_len, mode or decoder");
    }
    const std::string directory = argv[3];
    const std::string input_path = directory + "/window_tokens.raw";
    const int logits_fd = syscall(__NR_memfd_create, "ppl_logits", 0);
    if (logits_fd < 0) {
      throw std::runtime_error("Cannot allocate anonymous logits file");
    }
    const std::string logits_path = "/proc/self/fd/" + std::to_string(logits_fd);
    std::ifstream tokens_file(argv[2], std::ios::binary | std::ios::ate);
    const auto bytes = tokens_file.tellg();
    if (bytes <= 0 || bytes % sizeof(uint64_t)) {
      throw std::runtime_error("Invalid token file");
    }
    std::vector<uint64_t> tokens(bytes / sizeof(uint64_t));
    tokens_file.seekg(0);
    tokens_file.read(reinterpret_cast<char*>(tokens.data()), bytes);
    if (!tokens_file) {
      throw std::runtime_error("Cannot read token file");
    }
    for (uint64_t token : tokens) {
      if (token > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
        throw std::runtime_error("Token ID exceeds teacher-forcing int32 ABI");
      }
    }
    auto owned = std::make_unique<executorch::extension::Module>(
        settings.at("model").get<std::string>(),
        executorch::extension::Module::LoadMode::MmapUseMlockIgnoreErrors);
    auto* module = owned.get();
    const std::string prefill_method = mode == "hybrid" ? "prefill_forward" : "kv_forward";
    auto method = module->method_meta(prefill_method);

    if (!method.ok()) {
      throw std::runtime_error("Missing selected prompt method");
    }
    auto output_meta = method.get().output_tensor_meta(0);
    if (!output_meta.ok()) {
      throw std::runtime_error("Missing logits metadata");
    }
    const auto shape = output_meta.get().sizes();
    if (shape.size() != 3 || shape[0] != 1 || shape[1] < 1 || shape[2] < 2) {
      throw std::runtime_error("Expected prompt logits [1, AR, vocab]");
    }
    const size_t ar = shape[1];
    const size_t vocab = shape[2];
    const auto tokenizer_max_id = settings.at("tokenizer_max_id").get<int64_t>();
    if (tokenizer_max_id < 0 || static_cast<uint64_t>(tokenizer_max_id) >= vocab) {
      throw std::runtime_error("Tokenizer vocabulary exceeds exported logits vocabulary");
    }
    const auto dtype = output_meta.get().scalar_type();
    auto decode_method = module->method_meta("kv_forward");
    if (!decode_method.ok()) {
      throw std::runtime_error("Missing AR1 decode method");
    }
    auto decode_meta = decode_method.get().output_tensor_meta(0);
    if (!decode_meta.ok() || decode_meta.get().sizes().size() != 3 ||
        decode_meta.get().sizes()[0] != 1 || decode_meta.get().sizes()[1] != 1 ||
        decode_meta.get().sizes()[2] != static_cast<int64_t>(vocab) ||
        decode_meta.get().scalar_type() != dtype) {
      throw std::runtime_error("Expected matching prompt logits and AR1 decode logits");
    }
    auto prompt_mask = method.get().input_tensor_meta(1);
    auto decode_mask = decode_method.get().input_tensor_meta(1);
    auto prompt_tokens = method.get().input_tensor_meta(0);
    auto decode_tokens = decode_method.get().input_tensor_meta(0);
    if (!prompt_mask.ok() || !decode_mask.ok() ||
        !prompt_tokens.ok() || !decode_tokens.ok()) {
      throw std::runtime_error("Missing token or attention-mask metadata");
    }
    const auto pm = prompt_mask.get().sizes();
    const auto dm = decode_mask.get().sizes();
    const auto pt = prompt_tokens.get().sizes();
    const auto dt = decode_tokens.get().sizes();
    if (pm.size() != 4 || dm.size() != 4 || pm[0] != 1 || pm[1] != 1 ||
        dm[0] != 1 || dm[1] != 1 || pm[2] != static_cast<int64_t>(ar) ||
        dm[2] != 1 || pm[3] != dm[3] || dm[3] < 2 ||
        seq_len > dm[3] || ar > static_cast<size_t>(dm[3]) ||
        pt.size() != 2 || dt.size() != 2 || pt[0] != 1 || dt[0] != 1 ||
        pt[1] != static_cast<int64_t>(ar) || dt[1] != 1) {
      throw std::runtime_error("Unsupported prompt/decode shapes or seq_len exceeds compiled context");
    }
    const auto token_type = prompt_tokens.get().scalar_type();
    if ((token_type != executorch::aten::ScalarType::Int &&
         token_type != executorch::aten::ScalarType::Long) ||
        decode_tokens.get().scalar_type() != token_type) {
      throw std::runtime_error("Expected matching int32/int64 token inputs");
    }
    const size_t context = dm[3];
    for (uint64_t token : tokens) {
      if (token >= vocab) {
        throw std::runtime_error("Corpus token ID outside exported vocabulary");
      }
    }
    example::Runner runner(
        std::move(owned), decoder, settings.at("model"),
        settings.at("tokenizer"), logits_path, "", 0.f,
        mode == "hybrid" ? example::Runner::kHybrid : example::Runner::kKVCached,
        false, 0, 0, 0, nullptr, nullptr, 1.f, 128);
    if (runner.load() != Error::Ok) {
      throw std::runtime_error("PTE loading failed");
    }
    const bool quantized = dtype == executorch::aten::ScalarType::UInt16;
    const bool half = dtype == executorch::aten::ScalarType::Half;
    if (!quantized && !half && dtype != executorch::aten::ScalarType::Float) {
      throw std::runtime_error("Expected UInt16, Half or Float logits");
    }
    double scale = 1;
    int64_t zero_point = 0;
    if (quantized) {
      auto s = module->get("get_logits_scale");
      auto z = module->get("get_logits_zero_point");
      if (!s.ok() || !z.ok() || !s.get().isDouble() || !z.get().isInt()) {
        throw std::runtime_error("Missing quantized logits scale/zero point");
      }
      scale = s.get().toDouble();
      zero_point = z.get().toInt();
    }
    if (!std::isfinite(scale) || scale <= 0 || ar < 1) {
      throw std::runtime_error("Invalid logits metadata");
    }
    std::vector<double> exponential(65536);
    for (size_t i = 0; i < exponential.size(); ++i) {
      exponential[i] = std::exp(-static_cast<double>(i) * scale);
    }
    const json identity = { {"protocol", "executorch.ppl.driver.v1"},
        {"config", settings}, {"vocab", vocab}, {"context_len", context},
        {"ar", ar}, {"logits_dtype", quantized ? "uint16" : (half ? "float16" : "float32")},
        {"scale", scale}, {"zero_point", zero_point},
        {"corpus_tokens", tokens.size()}, {"logits_storage", "anonymous_memfd"} };
    std::ifstream previous_metadata(directory + "/metadata.json");
    if (previous_metadata.good() && json::parse(previous_metadata) != identity) {
      throw std::runtime_error("Resume metadata mismatch");
    }
    save(directory + "/metadata.json", identity);
    double total_nll = 0;
    size_t total_scored = 0;
    std::vector<uint16_t> q(vocab);
    std::vector<float> f(vocab);
    std::vector<executorch::aten::Half> h(vocab);
    for (const auto& window : settings.at("windows")) {
      const size_t begin = window.at("begin");
      const size_t end = window.at("end");
      const size_t target_begin = window.at("target_begin");
      const std::string id = window.at("id");
      const std::string output_path = directory + "/" + id + ".json";
      if (!std::regex_match(id, std::regex("W[0-9]+")) ||
          begin >= end || end > tokens.size() || begin + 1 > target_begin ||
          target_begin >= end || end - begin > static_cast<size_t>(seq_len)) {
        throw std::runtime_error("Invalid evaluation window");
      }
      std::ifstream previous(output_path);
      if (previous.good()) {
        const auto result = json::parse(previous);
        if (result.at("window") != window || result.at("count").get<size_t>() != end - target_begin) {
          throw std::runtime_error("Invalid completed window");
        }
        const auto values = result.at("token_nll").get<std::vector<double>>();
        const auto top1 =
            result.at("top1_token_ids").get<std::vector<size_t>>();
        if (values.size() != end - target_begin) {
          throw std::runtime_error("Completed window NLL length mismatch");
        }
        if (top1.size() != values.size() ||
            std::any_of(top1.begin(), top1.end(), [vocab](size_t token) {
              return token >= vocab;
            })) {
          throw std::runtime_error("Completed window top-1 mismatch");
        }
        double checked_nll = 0;
        for (double value : values) {
          if (!std::isfinite(value) || value < 0) {
            throw std::runtime_error("Invalid cached token NLL");
          }
          checked_nll += value;
        }
        const double saved_nll = result.at("nll_sum");
        if (!std::isfinite(saved_nll) ||
            std::abs(checked_nll - saved_nll) > 1e-10 * std::max(1.0, checked_nll)) {
          throw std::runtime_error("Completed window NLL checksum mismatch");
        }
        total_nll += checked_nll;
        total_scored += values.size();
        continue;
      }
      const size_t input_count = end - begin - 1;
      const size_t prefill_count = ppl::prefill_count(context, ar, input_count);
      if (prefill_count != input_count && prefill_count % ar) {
        throw std::runtime_error("Tail decoding requires an AR-aligned prefill");
      }
      example::forced_tokens = tokens.data() + begin + prefill_count;
      example::forced_count = end - begin - prefill_count;
      example::forced_position = 0;
      std::ofstream input(input_path, std::ios::binary);
      input.write(reinterpret_cast<const char*>(tokens.data() + begin),
                  prefill_count * sizeof(uint64_t));
      input.close();
      if (!input) {
        throw std::runtime_error("Cannot write window input");
      }
      runner.reset();
      executorch::extension::llm::GenerationConfig generation;
      generation.seq_len = seq_len;
      generation.max_new_tokens = example::forced_count;
      generation.echo = false;
      generation.ignore_eos = true;
      generation.temperature = 0;
      generation.num_bos = 0;
      generation.num_eos = 0;
      if (runner.generate_from_prompt_or_file(input_path, true, generation) != Error::Ok) {
        throw std::runtime_error("Teacher-forced inference failed");
      }
      if (example::forced_position != example::forced_count) {
        throw std::runtime_error("Teacher-forcing cursor mismatch");
      }
      std::ifstream logits(logits_path, std::ios::binary | std::ios::ate);
      const size_t element_size = quantized ? sizeof(uint16_t) : (half ? sizeof(executorch::aten::Half) : sizeof(float));
      const size_t rows = ((prefill_count + ar - 1) / ar) * ar + input_count - prefill_count;
      const size_t expected_bytes = rows * vocab * element_size;
      if (!logits || logits.tellg() != static_cast<std::streamoff>(expected_bytes)) {
        throw std::runtime_error("Wrong logits row count; refusing token misalignment");
      }
      logits.seekg((target_begin - begin - 1) * vocab * element_size);
      std::vector<double> nlls;
      std::vector<size_t> top1_token_ids;
      double nll_sum = 0;
      for (size_t position = target_begin; position < end; ++position) {
        const auto target = tokens[position];
        if (target >= vocab) {
          throw std::runtime_error("Token ID outside vocabulary");
        }
        double nll = 0;
        if (quantized) {
          logits.read(reinterpret_cast<char*>(q.data()), vocab * sizeof(uint16_t));
          nll = ppl::quantized_nll(q, target, exponential, scale);
          top1_token_ids.push_back(ppl::top1_token(q));
        } else if (half) {
          logits.read(reinterpret_cast<char*>(h.data()), vocab * sizeof(executorch::aten::Half));
          for (size_t i = 0; i < vocab; ++i) {
            f[i] = static_cast<float>(h[i]);
          }
          nll = ppl::float_nll(f, target);
          top1_token_ids.push_back(ppl::top1_token(f));
        } else {
          logits.read(reinterpret_cast<char*>(f.data()), vocab * sizeof(float));
          nll = ppl::float_nll(f, target);
          top1_token_ids.push_back(ppl::top1_token(f));
        }
        if (!logits || !std::isfinite(nll) || nll < 0) {
          throw std::runtime_error("Nonfinite NLL or incomplete logits");
        }
        nlls.push_back(nll);
        nll_sum += nll;
      }
      save(output_path, {{"window", window}, {"nll_sum", nll_sum},
                         {"count", nlls.size()}, {"token_nll", nlls},
                         {"top1_token_ids", top1_token_ids},
                         {"prefill_tokens", prefill_count}, {"decode_inputs", input_count - prefill_count},
                         {"forced_token_count", example::forced_position}});
      total_nll += nll_sum;
      total_scored += nlls.size();
      logits.close();
      if (ftruncate(logits_fd, 0)) {
        throw std::runtime_error("Cannot release anonymous scratch logits");
      }
      std::cout << "PPL_WINDOW " << window.at("id") << " count=" << nlls.size()
                << " running_ppl=" << std::exp(total_nll / total_scored) << std::endl;
    }
    size_t expected_scored = 0;
    for (const auto& window : settings.at("windows")) {
      expected_scored += window.at("end").get<size_t>() - window.at("target_begin").get<size_t>();
    }
    if (!total_scored || total_scored != expected_scored) {
      throw std::runtime_error("Incomplete target coverage");
    }
    save(directory + "/summary.json", {{"nll_sum", total_nll}, {"count", total_scored},
         {"ppl", std::exp(total_nll / total_scored)}, {"complete", true}});
    close(logits_fd);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "PPL_ERROR " << error.what() << std::endl;
    return 2;
  }
}
