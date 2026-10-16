/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// A llama 3.2 runner that includes preprocessing and post processing
// logic. The text_decoder takes in a string as input and emits a string as
// output.

#include <executorch/examples/models/llama/tokenizer/llama_tiktoken.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/client_mem.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/multimodal_embedding_merger.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/multimodal_lhd_token_generator.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/multimodal_runner.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/rpc_mem.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/utils.h>
#include <executorch/extension/llm/runner/text_llm_runner.h>
#include <executorch/extension/llm/runner/util.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/platform/log.h>
#include <pytorch/tokenizers/hf_tokenizer.h>
#include <pytorch/tokenizers/llama2c_tokenizer.h>
#include <algorithm>
#include <cmath>
#include <fstream>

#if defined(__aarch64__)
#include "arm_neon.h"
#endif

using executorch::extension::Module;
using executorch::extension::llm::Audio;
using executorch::extension::llm::get_rss_bytes;
using executorch::extension::llm::Image;
using executorch::extension::llm::MultimodalInput;
using executorch::extension::llm::print_report;
using executorch::extension::llm::Stats;
using executorch::extension::llm::time_in_ms;
using executorch::runtime::Error;
using executorch::runtime::MethodMeta;
using executorch::runtime::Result;
using executorch::runtime::TensorInfo;
namespace llm = ::executorch::extension::llm;

namespace example {
namespace {
void print_performance_report(
    const Stats& stats,
    const std::string& performance_output_path) {
  // For now, we just print the total inference time for CI, can save more info
  // in future if needed.
  std::ofstream outfile(performance_output_path.c_str());
  if (outfile.is_open()) {
    double num_tok = 0;
    if (stats.num_generated_tokens == 0) {
      // For cases like evaluate perplexity where prompt_len == cache_len
      num_tok = ((stats.num_prompt_tokens)) /
          (double)(stats.prompt_eval_end_ms - stats.inference_start_ms) *
          stats.SCALING_FACTOR_UNITS_PER_SECOND;
    } else {
      num_tok = (stats.num_generated_tokens) /
          (double)(stats.inference_end_ms - stats.inference_start_ms) *
          stats.SCALING_FACTOR_UNITS_PER_SECOND;
    }

    outfile << num_tok;
    outfile.close();
  } else {
    ET_LOG(Error, "Error saving the inference speed file");
  }
}

void save_logits(
    const std::string& dump_logits_path,
    const std::vector<std::byte>& prefill_logits,
    const std::vector<std::byte>& decode_logits) {
  std::ofstream outFile(dump_logits_path.c_str(), std::ios::binary);
  if (outFile.is_open()) {
    outFile.write(
        reinterpret_cast<const char*>(prefill_logits.data()),
        prefill_logits.size());

    outFile.write(
        reinterpret_cast<const char*>(decode_logits.data()),
        decode_logits.size());
    outFile.close();
  } else {
    ET_CHECK_MSG(false, "Error saving the dump logits file");
  }
}

int32_t get_position_id_dim(
    const Result<MethodMeta>& method_meta,
    size_t input_pos_idx,
    int32_t ar_len) {
  if (!method_meta.ok() || ar_len <= 0) {
    return 1;
  }
  Result<TensorInfo> input_pos = method_meta->input_tensor_meta(input_pos_idx);
  if (!input_pos.ok()) {
    return 1;
  }
  const size_t base_bytes = static_cast<size_t>(ar_len) * sizeof(int32_t);
  if (base_bytes == 0 || input_pos->nbytes() % base_bytes != 0) {
    return 1;
  }
  return static_cast<int32_t>(input_pos->nbytes() / base_bytes);
}

size_t aligned_input_nbytes(
    const Result<MethodMeta>& method_meta,
    size_t index) {
  ET_CHECK_MSG(method_meta.ok(), "Failed to get method metadata");
  auto tensor_meta = method_meta->input_tensor_meta(index);
  ET_CHECK_MSG(tensor_meta.ok(), "Failed to get input tensor metadata");
  return align_shared_buffer_allocation_size(tensor_meta->nbytes());
}

size_t aligned_output_nbytes(
    const Result<MethodMeta>& method_meta,
    size_t index) {
  ET_CHECK_MSG(method_meta.ok(), "Failed to get method metadata");
  auto tensor_meta = method_meta->output_tensor_meta(index);
  ET_CHECK_MSG(tensor_meta.ok(), "Failed to get output tensor metadata");
  return align_shared_buffer_allocation_size(tensor_meta->nbytes());
}

size_t multimodal_prompt_processor_io_size_from_meta(
    const Result<MethodMeta>& method_meta,
    bool is_bert,
    CacheMode cache_mode) {
  size_t total_bytes = 0;
  size_t idx = 0;
  total_bytes += aligned_input_nbytes(method_meta, idx++);
  total_bytes += aligned_input_nbytes(method_meta, idx++);
  if (cache_mode == CacheMode::HybridCache) {
    total_bytes += aligned_input_nbytes(method_meta, idx++);
  }
  if (!is_bert) {
    total_bytes += aligned_input_nbytes(method_meta, idx++);
  }
  if (has_last_token_index_input(method_meta.get())) {
    total_bytes +=
        aligned_input_nbytes(method_meta, method_meta->num_inputs() - 1);
  }
  total_bytes += aligned_output_nbytes(method_meta, 0);
  return total_bytes;
}

size_t multimodal_token_generator_io_size_from_meta(
    const Result<MethodMeta>& method_meta,
    CacheMode cache_mode) {
  size_t total_bytes = 0;
  size_t idx = 0;
  total_bytes += aligned_input_nbytes(method_meta, idx++);
  total_bytes += aligned_input_nbytes(method_meta, idx++);
  if (cache_mode == CacheMode::HybridCache) {
    total_bytes += aligned_input_nbytes(method_meta, idx++);
  }
  total_bytes += aligned_input_nbytes(method_meta, idx++);
  total_bytes += aligned_output_nbytes(method_meta, 0);
  return total_bytes;
}

bool is_qwen35_family_vlm(const ModelVersion& model_version) {
  const auto* vlm = std::get_if<VisionLanguageModel>(&model_version);
  return vlm != nullptr &&
      (*vlm == VisionLanguageModel::kQwen3_5 ||
       *vlm == VisionLanguageModel::kLincal3_5);
}

int32_t read_required_int_metadata(
    Module* module,
    const std::string& method_name) {
  ET_CHECK_MSG(
      module->method_names()->count(method_name) != 0,
      "Required metadata %s is missing from Qwen3.5-family PTE",
      method_name.c_str());
  auto value = module->get(method_name.c_str());
  ET_CHECK_MSG(
      value.ok(),
      "Failed to read required metadata %s from Qwen3.5-family PTE",
      method_name.c_str());
  return static_cast<int32_t>(value.get().toScalar().to<int64_t>());
}

void append_text_position_ids(
    std::vector<int32_t>& position_ids,
    int32_t position_id_dim,
    int32_t& rope_pos,
    int32_t num_tokens) {
  for (int32_t i = 0; i < num_tokens; ++i) {
    for (int32_t dim = 0; dim < position_id_dim; ++dim) {
      position_ids.push_back(rope_pos + i);
    }
  }
  rope_pos += num_tokens;
}

void append_qwen35_image_position_ids(
    std::vector<int32_t>& position_ids,
    int32_t position_id_dim,
    int32_t& rope_pos,
    int32_t num_tokens) {
  const int32_t grid = static_cast<int32_t>(std::sqrt(num_tokens));
  ET_CHECK_MSG(
      grid * grid == num_tokens,
      "Qwen3.5 image token count (%d) must form a square grid.",
      num_tokens);
  for (int32_t h = 0; h < grid; ++h) {
    for (int32_t w = 0; w < grid; ++w) {
      position_ids.push_back(rope_pos);
      if (position_id_dim > 1) {
        position_ids.push_back(rope_pos + h);
      }
      if (position_id_dim > 2) {
        position_ids.push_back(rope_pos + w);
      }
    }
  }
  rope_pos += grid;
}

} // namespace

QNNMultimodalRunner::QNNMultimodalRunner(
    std::unique_ptr<executorch::extension::Module> encoder,
    std::unique_ptr<executorch::extension::Module> tok_embedding,
    std::unique_ptr<executorch::extension::Module> text_decoder,
    const std::string& model_version,
    const std::string& tokenizer_path,
    const std::string& dump_logits_path,
    const std::string& performance_output_path,
    const float temperature,
    const int eval_mode,
    const bool shared_buffer,
    const int ngram,
    const int window,
    const int gcap,
    const float repetition_penalty)
    // TODO: The arguments for the MultimodalRunner base class are currently
    // unused in this runner implementation. QNNMultimodalRunner will be
    // refactored in the future to align with MultimodalRunner's usage.
    : MultimodalRunner(
          {},
          nullptr,
          nullptr,
          nullptr,
          nullptr,
          nullptr,
          nullptr,
          std::make_unique<Stats>()),
      encoder_(std::move(encoder)),
      tok_embedding_(std::move(tok_embedding)),
      text_decoder_(std::move(text_decoder)),
      ngram_(ngram),
      window_(window),
      gcap_(gcap),
      tokenizer_path_(tokenizer_path),
      performance_output_path_(performance_output_path),
      dump_logits_path_(dump_logits_path),
      temperature_(temperature),
      repetition_penalty_(repetition_penalty),
      eval_mode_(static_cast<EvalMode>(eval_mode)),
      shared_buffer_(shared_buffer) {
  stats_.reset();

  if (model_version == "smolvlm") {
    model_version_ = VisionLanguageModel::kSmolvlm;
  } else if (model_version == "internvl3") {
    model_version_ = VisionLanguageModel::kInternvl3;
  } else if (model_version == "granite_speech") {
    model_version_ = AudioLanguageModel::kGraniteSpeech;
  } else if (model_version == "qwen2_5_vl") {
    model_version_ = VisionLanguageModel::kQwen2_5_VL;
  } else if (model_version == "lincal_vl") {
    model_version_ = VisionLanguageModel::kQwen2_5_VL;
    cache_mode_ = CacheMode::HybridCache;
  } else if (model_version == "lincal3_vl") {
    model_version_ = VisionLanguageModel::kLincal3_VL;
    cache_mode_ = CacheMode::HybridCache;
  } else if (model_version == "qwen3_5") {
    model_version_ = VisionLanguageModel::kQwen3_5;
  } else if (model_version == "lincal3_5") {
    model_version_ = VisionLanguageModel::kLincal3_5;
    cache_mode_ = CacheMode::HybridCache;
  } else {
    ET_CHECK_MSG(false, "Unsupported Decoder Model");
  }

  ET_LOG(Info, "creating runner: tokenizer_path=%s", tokenizer_path_.c_str());
  ET_LOG(Info, "eval mode=%d", eval_mode_);
}

bool QNNMultimodalRunner::is_loaded() const {
  return encoder_->is_loaded() && tok_embedding_->is_loaded() &&
      text_decoder_->is_loaded() && embedding_merger_ && tokenizer_ &&
      decoder_runner_ && prompt_processor_ && token_generator_ && kv_manager_ &&
      buffer_manager_;
}

Error QNNMultimodalRunner::load() {
  if (is_loaded()) {
    return Error::Ok;
  }
#if ET_QNN_RUNNER_DEV_MEM_LOG
  const dev_mem::RssSample load_before = dev_mem::sample_rss();
  dev_mem::log_proc_status_mem("multimodal load stage: entry");
#endif
  std::string prompt_embedding_method_name, tok_embedding_method_name;
  std::string token_generator_method_name, prompt_processor_method_name;
  std::vector<std::string> method_names;
  switch (eval_mode_) {
    case EvalMode::kKVCached:
      prompt_embedding_method_name = "tok_embedding_kv_forward";
      tok_embedding_method_name = "tok_embedding_kv_forward";
      prompt_processor_method_name = "kv_forward";
      token_generator_method_name = "kv_forward";
      method_names.emplace_back(prompt_processor_method_name);
      method_names.emplace_back(token_generator_method_name);
      break;
    case EvalMode::kHybrid:
    case EvalMode::kLookaheadDecoding:
      prompt_embedding_method_name = "tok_embedding_prefill_forward";
      tok_embedding_method_name = "tok_embedding_kv_forward";
      prompt_processor_method_name = "prefill_forward";
      token_generator_method_name = "kv_forward";
      method_names.emplace_back(prompt_processor_method_name);
      method_names.emplace_back(token_generator_method_name);
      break;
    case EvalMode::kUnsupported:
      ET_CHECK_MSG(false, "Unsupported llama evaluation mode");
      break;
  }
  auto eos_ids = std::make_unique<std::unordered_set<uint64_t>>();
  if (tokenizer_ != nullptr) {
    eos_ids->insert(tokenizer_->encode("<|eot_id|>", 0, 0).get()[0]);
    eos_ids->insert(tokenizer_->encode("<|eot|>", 0, 0).get()[0]);
    eos_ids->insert(tokenizer_->encode("<|end_of_text|>", 0, 0).get()[0]);
  } else {
    tokenizer_ = llm::load_tokenizer(tokenizer_path_);
    if (tokenizer_ == nullptr) {
      ET_LOG(
          Error, "Failed to load tokenizer with %s", tokenizer_path_.c_str());
      return Error::Internal;
    }
    eos_ids->insert(tokenizer_->eos_tok());
  }
  if (const auto* vlm = std::get_if<VisionLanguageModel>(&model_version_)) {
    if (*vlm == VisionLanguageModel::kSmolvlm) {
      eos_ids->insert(tokenizer_->encode("<end_of_utterance>", 0, 0).get()[0]);
    } else if (
        *vlm == VisionLanguageModel::kInternvl3 ||
        *vlm == VisionLanguageModel::kQwen2_5_VL ||
        *vlm == VisionLanguageModel::kLincal3_VL ||
        *vlm == VisionLanguageModel::kQwen3_5 ||
        *vlm == VisionLanguageModel::kLincal3_5) {
      eos_ids->insert(tokenizer_->encode("<|im_end|>", 0, 0).get()[0]);
    }
  } else if (
      const auto* alm = std::get_if<AudioLanguageModel>(&model_version_)) {
    if (*alm == AudioLanguageModel::kGraniteSpeech) {
      eos_ids->insert(tokenizer_->encode("<|end_of_text|>", 0, 0).get()[0]);
    }
  }

  Result<MethodMeta> method_meta =
      text_decoder_->method_meta(token_generator_method_name);
  Result<MethodMeta> prompt_method_meta =
      text_decoder_->method_meta(prompt_processor_method_name);

  // For some tokenizer.json, runtime vocab_size might be different, use output
  // shape to get vocab size.
  int32_t vocab_size = method_meta->output_tensor_meta(0)->sizes()[2];
  decoder_runner_ = std::make_unique<DecoderRunner>(
      text_decoder_.get(), vocab_size, temperature_, repetition_penalty_);

  ET_CHECK_OK_OR_RETURN_ERROR(decoder_runner_->load(method_names));

  ET_LOG(Info, "Reading metadata from model");
#if ET_QNN_RUNNER_DEV_MEM_LOG
  const dev_mem::RssSample metadata_stage_before = dev_mem::sample_rss();
  dev_mem::log_proc_status_mem(
      "multimodal load stage: metadata extraction (before)");
#endif
  // retrieve any method meta, can be either prefill or kv
  ET_ASSIGN_OR_RETURN(num_layers_evalue__, text_decoder_->get("get_n_layers"));
  int64_t num_layers = num_layers_evalue__.toScalar().to<int64_t>();

  ET_CHECK_MSG(num_layers != -1, "Could not retrieve num layers");
  int64_t num_heads = 0;
  if (is_qwen35_family_vlm(model_version_) &&
      text_decoder_->method_names()->count("get_n_kv_heads") > 0 &&
      text_decoder_->method_names()->count("get_head_dim") > 0) {
    num_heads = ET_UNWRAP(text_decoder_->get("get_n_kv_heads"))
                    .toScalar()
                    .to<int64_t>();
  } else {
    // k_cache: [1, n_heads, head_dim, seq_len]
    auto k_cache_shape = method_meta->output_tensor_meta(1)->sizes();
    num_heads = k_cache_shape[1];
  }

  // hidden_state: [1, ar_len, dim]
  int64_t dim = tok_embedding_->method_meta(tok_embedding_method_name)
                    ->output_tensor_meta(0)
                    ->sizes()[2];
  bool use_int64_token =
      tok_embedding_->method_meta(tok_embedding_method_name)
          ->input_tensor_meta(0)
          ->scalar_type() == executorch::aten::ScalarType::Long;

  // Use attention mask length to retrieve AR length and context length
  // Cache len equals to context_len - ar_len
  int32_t prompt_processor_ar_len = 0;
  int32_t token_generator_ar_len = 0;
  int32_t max_cache_len = 0;
  int32_t max_ar_len = 0;
  // atten mask: [1, 1, AR-N, CL]
  auto atten_mask_meta_token = method_meta->input_tensor_meta(1);
  token_generator_ar_len = atten_mask_meta_token->sizes()[2];
  context_len_ = atten_mask_meta_token->sizes()[3];
  if (eval_mode_ == EvalMode::kKVCached) {
    prompt_processor_ar_len = token_generator_ar_len;
  } else if (
      eval_mode_ == EvalMode::kHybrid ||
      eval_mode_ == EvalMode::kLookaheadDecoding) {
    auto atten_mask_meta_prompt = prompt_method_meta->input_tensor_meta(1);
    prompt_processor_ar_len = atten_mask_meta_prompt->sizes()[2];
  }
  if (prompt_processor_ar_len == context_len_)
    max_cache_len = context_len_;
  else
    max_cache_len = context_len_ -
        std::min(token_generator_ar_len, prompt_processor_ar_len);
  max_ar_len = std::max(token_generator_ar_len, prompt_processor_ar_len);

  const size_t input_pos_idx = cache_mode_ == CacheMode::HybridCache ? 3 : 2;
  const int32_t token_position_id_dim =
      get_position_id_dim(method_meta, input_pos_idx, token_generator_ar_len);
  const int32_t prompt_position_id_dim = get_position_id_dim(
      prompt_method_meta, input_pos_idx, prompt_processor_ar_len);
  ET_CHECK_MSG(
      token_position_id_dim == prompt_position_id_dim,
      "Prompt/token position id dim mismatch: %d vs %d",
      prompt_position_id_dim,
      token_position_id_dim);
  position_id_dim_ = prompt_position_id_dim;

  bool enable_true_swa = false;
  std::vector<int32_t> layer_cache_len;
  std::vector<bool> layer_is_swa;
  int32_t window_context_len = context_len_;
  int32_t prompt_window_context_len = context_len_;
  if (cache_mode_ == CacheMode::HybridCache) {
    const auto window_mask_sizes = method_meta->input_tensor_meta(2)->sizes();
    window_context_len = window_mask_sizes[window_mask_sizes.size() - 1];
    if (window_context_len != context_len_) {
      enable_true_swa = true;
    }

    const size_t input_cache_start =
        kv_cache_input_start(method_meta.get(), num_layers);
    const int32_t full_cache_len = context_len_ - token_generator_ar_len;
    layer_cache_len.resize(num_layers, 0);
    layer_is_swa.assign(num_layers, false);
    for (int64_t layer = 0; layer < num_layers; ++layer) {
      Result<TensorInfo> k_in = method_meta->input_tensor_meta(
          input_cache_start + static_cast<size_t>(layer));
      if (!k_in.ok()) {
        ET_LOG(
            Error,
            "Failed to fetch kv_forward key cache metadata for layer %lld",
            static_cast<long long>(layer));
        return Error::Internal;
      }
      if (k_in->sizes().size() == 4) {
        const int32_t cache_len = k_in->sizes()[3];
        layer_cache_len[layer] = cache_len;
        if (cache_len != full_cache_len) {
          layer_is_swa[layer] = true;
          enable_true_swa = true;
        }
      }
    }

    const auto prompt_window_mask_sizes =
        prompt_method_meta->input_tensor_meta(2)->sizes();
    prompt_window_context_len =
        prompt_window_mask_sizes[prompt_window_mask_sizes.size() - 1];
    if (prompt_window_context_len != context_len_) {
      enable_true_swa = true;
    }

    const size_t prompt_input_cache_start =
        kv_cache_input_start(prompt_method_meta.get(), num_layers);
    const int32_t prefill_full_cache_len =
        context_len_ - prompt_processor_ar_len;
    for (int64_t layer = 0; layer < num_layers; ++layer) {
      Result<TensorInfo> k_in = prompt_method_meta->input_tensor_meta(
          prompt_input_cache_start + static_cast<size_t>(layer));
      if (!k_in.ok()) {
        ET_LOG(
            Error,
            "Failed to fetch prefill key cache metadata for layer %lld",
            static_cast<long long>(layer));
        return Error::Internal;
      }
      if (k_in->sizes().size() == 4 &&
          k_in->sizes()[3] != prefill_full_cache_len) {
        layer_is_swa[layer] = true;
        enable_true_swa = true;
      }
    }
  }
  if (enable_true_swa) {
    int64_t swa_layers = 0;
    for (bool is_swa : layer_is_swa) {
      if (is_swa) {
        ++swa_layers;
      }
    }
    ET_LOG(
        Info,
        "True SWA enabled for multimodal runner: decode_window_context_len=%d, prompt_window_context_len=%d, full_context_len=%d, swa_layers=%lld/%lld",
        window_context_len,
        prompt_window_context_len,
        context_len_,
        static_cast<long long>(swa_layers),
        static_cast<long long>(num_layers));
  }

  std::vector<KVManager::Metadata::CacheSpec> cache_specs;
  if (is_qwen35_family_vlm(model_version_)) {
    using CacheSpec = KVManager::Metadata::CacheSpec;
    cache_specs.resize(
        num_layers, CacheSpec{0, 0, 0, 0, KVManager::Metadata::kAppend});

    for (const std::string& method_name : method_names) {
      Result<MethodMeta> cache_method_meta =
          text_decoder_->method_meta(method_name);
      if (!cache_method_meta.ok()) {
        ET_LOG(
            Error,
            "Failed to fetch method metadata for %s",
            method_name.c_str());
        return Error::Internal;
      }

      const size_t input_cache_start =
          kv_cache_input_start(cache_method_meta.get(), num_layers);
      for (int64_t layer = 0; layer < num_layers; ++layer) {
        Result<TensorInfo> k_in = cache_method_meta->input_tensor_meta(
            input_cache_start + static_cast<size_t>(layer));
        Result<TensorInfo> v_in = cache_method_meta->input_tensor_meta(
            input_cache_start + static_cast<size_t>(num_layers + layer));
        Result<TensorInfo> k_out = cache_method_meta->output_tensor_meta(
            1 + static_cast<size_t>(layer));
        Result<TensorInfo> v_out = cache_method_meta->output_tensor_meta(
            1 + static_cast<size_t>(num_layers + layer));
        if (!k_in.ok() || !v_in.ok() || !k_out.ok() || !v_out.ok()) {
          ET_LOG(
              Error,
              "Failed to fetch cache tensor metadata for %s layer %lld",
              method_name.c_str(),
              static_cast<long long>(layer));
          return Error::Internal;
        }

        auto& spec = cache_specs[layer];
        spec.k_input_bytes = std::max(spec.k_input_bytes, k_in->nbytes());
        spec.v_input_bytes = std::max(spec.v_input_bytes, v_in->nbytes());
        spec.k_output_bytes = std::max(spec.k_output_bytes, k_out->nbytes());
        spec.v_output_bytes = std::max(spec.v_output_bytes, v_out->nbytes());

        spec.update_mode = KVManager::infer_cache_update_mode(
            k_in.get(),
            v_in.get(),
            k_out.get(),
            v_out.get(),
            !layer_is_swa.empty() && layer_is_swa[layer]);
        spec.k_zero_point =
            executorch::runtime::isFloatingType(k_in->scalar_type())
            ? 0
            : read_required_int_metadata(
                  text_decoder_.get(),
                  "get_kv_input_" + std::to_string(layer) + "_zero_point");
        spec.v_zero_point =
            executorch::runtime::isFloatingType(v_in->scalar_type())
            ? 0
            : read_required_int_metadata(
                  text_decoder_.get(),
                  "get_kv_input_" + std::to_string(num_layers + layer) +
                      "_zero_point");
      }
    }
  }

  if (enable_true_swa &&
      (eval_mode_ == EvalMode::kHybrid ||
       eval_mode_ == EvalMode::kLookaheadDecoding)) {
    const size_t input_cache_start =
        kv_cache_input_start(prompt_method_meta.get(), num_layers);
    for (int64_t layer = 0; layer < num_layers; ++layer) {
      if (!layer_is_swa.empty() && layer_is_swa[layer]) {
        Result<TensorInfo> k_in = prompt_method_meta->input_tensor_meta(
            input_cache_start + static_cast<size_t>(layer));
        if (!k_in.ok()) {
          ET_LOG(
              Error,
              "Failed to fetch prefill key cache metadata for layer %lld",
              static_cast<long long>(layer));
          return Error::Internal;
        }
        const int32_t prefill_cache_len =
            k_in->sizes().size() == 4 ? k_in->sizes()[3] : 0;
        if (prefill_cache_len != layer_cache_len[layer]) {
          ET_LOG(
              Error,
              "Unsupported true SWA export: prefill_forward and kv_forward use different cache layouts "
              "(layer=%lld prefill_cache_len=%d decode_cache_len=%d).",
              static_cast<long long>(layer),
              prefill_cache_len,
              layer_cache_len[layer]);
          return Error::NotSupported;
        }
      }
    }
  }

  // Initialize Encoder
  encoder_runner_ = std::make_unique<EncoderRunner>(encoder_.get());
  ET_CHECK_OK_OR_RETURN_ERROR(encoder_runner_->load());

  // Initialize TokenEmbeddingProcessor
  tok_embedding_runner_ =
      std::make_unique<TokenEmbeddingRunner>(tok_embedding_.get());
  ET_CHECK_OK_OR_RETURN_ERROR(tok_embedding_runner_->load(
      {prompt_embedding_method_name, tok_embedding_method_name}));
  tok_embedding_processor_ = std::make_unique<TokenEmbeddingProcessor>(
      tok_embedding_runner_.get(),
      prompt_embedding_method_name,
      TokenEmbeddingProcessor::Metadata{
          context_len_,
          prompt_processor_ar_len,
          vocab_size,
          use_int64_token,
          static_cast<int32_t>(dim)});

  // Load the sliding window size if the model supports it.
  // This is used to configure the attention mask for models with window
  // attention
  int32_t sliding_window = context_len_;
  if (text_decoder_->method_names()->count("get_sliding_window") > 0) {
    ET_ASSIGN_OR_RETURN(
        sliding_window_evalue__, text_decoder_->get("get_sliding_window"));
    sliding_window = sliding_window_evalue__.toInt();
  }
#if ET_QNN_RUNNER_DEV_MEM_LOG
  const dev_mem::RssSample metadata_stage_after = dev_mem::sample_rss();
  dev_mem::log_rss_delta(
      "multimodal load stage: metadata extraction",
      metadata_stage_before,
      metadata_stage_after);
  dev_mem::log_proc_status_mem(
      "multimodal load stage: metadata extraction (after)");
  const dev_mem::RssSample load_io_before = dev_mem::sample_rss();
  dev_mem::log_proc_status_mem(
      "multimodal load stage: io/cache allocation (before)");
#endif
  kv_manager_ = std::make_unique<KVManager>(
      KVManager::Metadata{
          context_len_,
          max_ar_len,
          max_cache_len,
          num_heads,
          num_layers,
          std::move(cache_specs)},
      std::make_unique<MethodMeta>(std::move(
          text_decoder_->method_meta(token_generator_method_name).get())));

  int32_t max_prefill_past = INT32_MAX;
  if (prompt_processor_method_name == "prefill_forward" &&
      text_decoder_->method_names()->count("get_prefill_max_past") != 0) {
    max_prefill_past =
        read_required_int_metadata(text_decoder_.get(), "get_prefill_max_past");
  }
  prompt_processor_ = std::make_unique<MultimodalPromptProcessor>(
      decoder_runner_.get(),
      kv_manager_.get(),
      prompt_processor_method_name,
      MultimodalPromptProcessor::Metadata{
          context_len_,
          num_heads,
          num_layers,
          prompt_processor_ar_len,
          vocab_size,
          use_int64_token,
          sliding_window,
          cache_mode_,
          static_cast<int32_t>(dim),
          position_id_dim_,
          max_prefill_past},
      std::make_unique<MethodMeta>(std::move(
          text_decoder_->method_meta(prompt_processor_method_name).get())));

  // Initialize EmbeddingGenerator
  tok_embedding_generator_ = std::make_unique<TokenEmbeddingProcessor>(
      tok_embedding_runner_.get(),
      tok_embedding_method_name,
      TokenEmbeddingProcessor::Metadata{
          context_len_,
          token_generator_ar_len,
          vocab_size,
          use_int64_token,
          static_cast<int32_t>(dim)});
  if (eval_mode_ == EvalMode::kLookaheadDecoding) {
    // Initialize TokenGenerator
    token_generator_ = std::make_unique<MultimodalLhdTokenGenerator>(
        tokenizer_.get(),
        tok_embedding_generator_.get(),
        decoder_runner_.get(),
        kv_manager_.get(),
        token_generator_method_name,
        std::move(eos_ids),
        MultimodalLhdTokenGenerator::Metadata{
            context_len_,
            num_heads,
            num_layers,
            token_generator_ar_len,
            vocab_size,
            use_int64_token,
            ngram_,
            window_,
            gcap_,
            sliding_window,
            cache_mode_,
            static_cast<int32_t>(dim),
            position_id_dim_},
        &stats_,
        std::make_unique<MethodMeta>(std::move(
            text_decoder_->method_meta(token_generator_method_name).get())));
  } else {
    token_generator_ = std::make_unique<MultimodalTokenGenerator>(
        tokenizer_.get(),
        tok_embedding_generator_.get(),
        decoder_runner_.get(),
        kv_manager_.get(),
        token_generator_method_name,
        std::move(eos_ids),
        MultimodalTokenGenerator::Metadata{
            context_len_,
            num_heads,
            num_layers,
            token_generator_ar_len,
            vocab_size,
            use_int64_token,
            sliding_window,
            cache_mode_,
            static_cast<int32_t>(dim),
            position_id_dim_},
        &stats_,
        std::make_unique<MethodMeta>(std::move(
            text_decoder_->method_meta(token_generator_method_name).get())));
  }
  if (enable_true_swa) {
    kv_manager_->configure_cache_size_swa(layer_cache_len, layer_is_swa);
  }

  buffer_manager_ = std::make_unique<ClientMem>();
  if (shared_buffer_) {
    buffer_manager_ = std::make_unique<RpcMem>(
        kv_manager_->total_cache_size_in_bytes(),
        multimodal_prompt_processor_io_size_from_meta(
            prompt_method_meta,
            prompt_processor_ar_len == context_len_,
            cache_mode_),
        multimodal_token_generator_io_size_from_meta(method_meta, cache_mode_),
        tok_embedding_processor_->total_embedding_processor_io_size_in_bytes(),
        tok_embedding_generator_->total_embedding_processor_io_size_in_bytes());
  }

  ET_LOG(Info, "creating io_memory");
  // prepare io
  if (enable_true_swa) {
    kv_manager_->init_cache_swa(
        buffer_manager_.get(),
        prompt_processor_ar_len,
        layer_cache_len,
        layer_is_swa);
  } else {
    kv_manager_->init_cache(buffer_manager_.get(), prompt_processor_ar_len);
  }
  prompt_processor_->init_io(
      buffer_manager_.get(),
      text_decoder_->method_meta(prompt_processor_method_name));
  token_generator_->init_io(
      buffer_manager_.get(),
      text_decoder_->method_meta(token_generator_method_name));
  // Prepare io for embedding
  tok_embedding_processor_->init_io(
      buffer_manager_.get(),
      tok_embedding_->method_meta(prompt_embedding_method_name));
  tok_embedding_generator_->init_io(
      buffer_manager_.get(),
      tok_embedding_->method_meta(tok_embedding_method_name));

  if (const auto* vlm = std::get_if<VisionLanguageModel>(&model_version_)) {
    if (text_decoder_->method_names()->count("image_token_id") > 0) {
      image_token_id_ = ET_UNWRAP(text_decoder_->get("image_token_id")).toInt();
      ET_LOG(
          Info,
          "Image placeholder token ID for vision modality loaded: %zu",
          image_token_id_);
    } else if (
        *vlm == VisionLanguageModel::kQwen2_5_VL ||
        *vlm == VisionLanguageModel::kLincal3_VL) {
      ET_CHECK_MSG(
          false,
          "Qwen2.5-VL compatible models require image_token_id metadata.");
    }
  }

  // Initialize embedding merger
  embedding_merger_ =
      std::make_unique<MultimodalEmbeddingMerger>(static_cast<int32_t>(dim));

#if ET_QNN_RUNNER_DEV_MEM_LOG
  const dev_mem::RssSample load_io_after = dev_mem::sample_rss();
  dev_mem::log_rss_delta(
      "multimodal load stage: io/cache allocation",
      load_io_before,
      load_io_after);
  dev_mem::log_proc_status_mem(
      "multimodal load stage: io/cache allocation (after)");

  const size_t prompt_io_bytes =
      prompt_processor_->total_prompt_processor_io_size_in_bytes();
  const size_t token_io_bytes =
      token_generator_->total_token_generator_io_size_in_bytes();
  const size_t embedding_prefill_io_bytes =
      tok_embedding_processor_->total_embedding_processor_io_size_in_bytes();
  const size_t embedding_decode_io_bytes =
      tok_embedding_generator_->total_embedding_processor_io_size_in_bytes();
  const size_t total_static_buffers = kv_manager_->total_cache_size_in_bytes() +
      prompt_io_bytes + token_io_bytes + embedding_prefill_io_bytes +
      embedding_decode_io_bytes;
  const dev_mem::RssSample load_after = dev_mem::sample_rss();
  ET_LOG(
      Info,
      "[DEV_MEM] multimodal load current RSS: before=%.2f MiB after=%.2f MiB delta=%.2f MiB",
      dev_mem::bytes_to_mib(load_before.current_bytes),
      dev_mem::bytes_to_mib(load_after.current_bytes),
      dev_mem::bytes_to_mib(dev_mem::saturating_sub(
          load_after.current_bytes, load_before.current_bytes)));
  ET_LOG(
      Info,
      "[DEV_MEM] multimodal load peak RSS: before=%.2f MiB after=%.2f MiB delta=%.2f MiB",
      dev_mem::bytes_to_mib(load_before.peak_bytes),
      dev_mem::bytes_to_mib(load_after.peak_bytes),
      dev_mem::bytes_to_mib(dev_mem::saturating_sub(
          load_after.peak_bytes, load_before.peak_bytes)));
  ET_LOG(
      Info,
      "[DEV_MEM] multimodal static buffers: kv_cache=%.2f MiB prompt_io=%.2f MiB token_io=%.2f MiB embedding_prefill_io=%.2f MiB embedding_decode_io=%.2f MiB total=%.2f MiB",
      dev_mem::bytes_to_mib(kv_manager_->total_cache_size_in_bytes()),
      dev_mem::bytes_to_mib(prompt_io_bytes),
      dev_mem::bytes_to_mib(token_io_bytes),
      dev_mem::bytes_to_mib(embedding_prefill_io_bytes),
      dev_mem::bytes_to_mib(embedding_decode_io_bytes),
      dev_mem::bytes_to_mib(total_static_buffers));
#endif

  return Error::Ok;
}

executorch::runtime::Error QNNMultimodalRunner::generate(
    const std::vector<MultimodalInput>& inputs,
    const llm::GenerationConfig& config,
    std::function<void(const std::string&)> token_callback,
    std::function<void(const Stats&)> stats_callback) {
  ET_CHECK_MSG(!inputs.empty(), "inputs cannot be empty");
  if (!is_loaded()) {
    stats_.model_load_start_ms = time_in_ms();
    ET_CHECK_OK_OR_RETURN_ERROR(load());
    stats_.model_load_end_ms = time_in_ms();
  }
  stats_.inference_start_ms = time_in_ms();
  ET_CHECK_OR_RETURN_ERROR(
      !config.repair_think_eos && !config.enable_think_budget,
      NotSupported,
      "Thinking controls currently support the text KV/hybrid runner only");
  int32_t seq_len = config.seq_len;
  if (seq_len > context_len_) {
    ET_LOG(
        Info,
        "Warning: Requested seq_len (%d) exceeds compiled max_seq_len (%d). Clamping to %d.",
        seq_len,
        context_len_,
        context_len_);
    seq_len = context_len_;
  } else if (seq_len <= 0) {
    ET_LOG(
        Info,
        "Warning: Invalid seq_len (%d). Using compiled max_seq_len (%d).",
        seq_len,
        context_len_);
    seq_len = context_len_;
  }
  // For multimodal, we will disable n_bos
  int32_t n_bos = 0;

  std::string prompt;
  std::vector<uint64_t> prompt_tokens;
  bool dump_logits = !dump_logits_path_.empty();

  // Reset merger for new generation
  embedding_merger_->reset();
  const bool use_qwen35_mrope =
      is_qwen35_family_vlm(model_version_) && position_id_dim_ == 3;
  const bool placeholder_merge =
      modality_of(model_version_) == Modality::kVision &&
      image_token_id_ != 0 && !is_qwen35_family_vlm(model_version_);
  std::vector<int32_t> prompt_position_ids;
  int32_t mrope_pos = 0;

  // Process each input and add embeddings to merger
  for (const MultimodalInput& input : inputs) {
    if (input.is_text() || input.is_tokens()) {
      std::vector<uint64_t> tokens;
      if (input.is_tokens()) {
        tokens = input.get_tokens();
      } else {
        std::string text = input.get_text();
        prompt += text;
        auto encode_res = tokenizer_->encode(text, n_bos, 0);
        ET_CHECK_TK_OK_OR_RETURN_ERROR(
            encode_res.error(), "failed to encode prompt %s", text.c_str());
        tokens = encode_res.get();
      }
      tok_embedding_processor_->prefill(tokens);
      const TensorStruct<float>& text_embeddings =
          tok_embedding_processor_->get_prompt_embeddings();

      // Add text embeddings to merger
      if (placeholder_merge) {
        embedding_merger_->add_text_embeddings(text_embeddings);
      } else {
        embedding_merger_->add_embeddings(text_embeddings);
      }
      if (use_qwen35_mrope) {
        append_text_position_ids(
            prompt_position_ids,
            position_id_dim_,
            mrope_pos,
            static_cast<int32_t>(tokens.size()));
      }

      prompt_tokens.insert(prompt_tokens.end(), tokens.begin(), tokens.end());

    } else if (input.is_audio()) {
      const Audio& audio = input.get_audio();
      auto audio_tensor_res = audio.toTensor();
      executorch::extension::TensorPtr audio_tensor_ptr =
          audio_tensor_res.get();

      auto encode_res = encoder_runner_->encode(audio_tensor_ptr);
      executorch::aten::Tensor audio_embeddings_tensor = encode_res.get();

      // Add audio embeddings to merger
      embedding_merger_->add_embeddings(audio_embeddings_tensor);
    } else if (input.is_image()) {
      const Image& image = input.get_image();
      auto encode_res = encoder_runner_->encode(image);
      executorch::aten::Tensor image_embeddings_tensor = encode_res.get();

      // Add image embeddings to merger
      if (placeholder_merge) {
        embedding_merger_->add_image_embeddings(image_embeddings_tensor);
      } else {
        embedding_merger_->add_embeddings(image_embeddings_tensor);
      }
      if (use_qwen35_mrope) {
        append_qwen35_image_position_ids(
            prompt_position_ids,
            position_id_dim_,
            mrope_pos,
            static_cast<int32_t>(image_embeddings_tensor.size(1)));
      }

    } else {
      ET_CHECK_MSG(false, "Unsupported input data type");
    }
  }

  TensorStruct<float> merged_embeddings = placeholder_merge
      ? embedding_merger_->merge(prompt_tokens, image_token_id_)
      : embedding_merger_->get_merged_embeddings();
  int num_prompt_tokens = embedding_merger_->get_total_tokens();
  if (use_qwen35_mrope) {
    ET_CHECK_MSG(
        prompt_position_ids.size() ==
            static_cast<size_t>(num_prompt_tokens * position_id_dim_),
        "Qwen3.5 M-RoPE position ids mismatch: got %zu, expected %d",
        prompt_position_ids.size(),
        num_prompt_tokens * position_id_dim_);
  }
  token_generator_->set_position_delta(
      use_qwen35_mrope ? mrope_pos - num_prompt_tokens : 0);

  ET_CHECK_MSG(num_prompt_tokens >= 1, "Expected at least 1 prompt token");
  ET_CHECK_MSG(
      cur_pos_ + num_prompt_tokens < seq_len,
      "sequence length exceeded - please increase the seq_len value");

  if (token_callback && config.echo) {
    token_callback(prompt);
  }

#if ET_QNN_RUNNER_DEV_MEM_LOG
  const dev_mem::RssSample prefill_before = dev_mem::sample_rss();
#endif
  decoder_runner_->begin_turn(prompt_tokens, cur_pos_ == 0);
  auto prefill_res = prompt_processor_->prefill(
      merged_embeddings,
      cur_pos_,
      dump_logits,
      nullptr,
      use_qwen35_mrope ? &prompt_position_ids : nullptr);
  ET_CHECK_OK_OR_RETURN_ERROR(prefill_res.error());
  uint64_t cur_token = prefill_res.get();
  decoder_runner_->record_token(cur_token);
  cur_pos_ += num_prompt_tokens;
  stats_.first_token_ms = time_in_ms();
  stats_.prompt_eval_end_ms = time_in_ms();
#if ET_QNN_RUNNER_DEV_MEM_LOG
  const dev_mem::RssSample prefill_after = dev_mem::sample_rss();
  dev_mem::log_rss_delta("multimodal prefill", prefill_before, prefill_after);
  dev_mem::log_proc_status_mem("multimodal prefill (after)");
#endif

  // print the first token from prefill. No prev_token so use cur_token for
  // it.
  if (token_callback) {
    ET_ASSIGN_OR_RETURN_TOKENIZER(
        decoded_token__, tokenizer_->decode(cur_token, cur_token));
    token_callback(decoded_token__);
  }
  ET_LOG(
      Info,
      "RSS after prompt prefill: %f MiB (0 if unsupported)",
      get_rss_bytes() / 1024.0 / 1024.0);

  // start the main loop
  prompt_tokens.push_back(cur_token);

#if ET_QNN_RUNNER_DEV_MEM_LOG
  const dev_mem::RssSample decode_before = dev_mem::sample_rss();
#endif
  ET_ASSIGN_OR_RETURN(
      num_generated_tokens,
      token_generator_->generate(
          prompt_tokens,
          cur_pos_,
          seq_len,
          token_callback,
          dump_logits,
          nullptr,
          config.ignore_eos));
  stats_.inference_end_ms = time_in_ms();
#if ET_QNN_RUNNER_DEV_MEM_LOG
  const dev_mem::RssSample decode_after = dev_mem::sample_rss();
  dev_mem::log_rss_delta("multimodal decode", decode_before, decode_after);
  dev_mem::log_proc_status_mem("multimodal decode (after)");
#endif
  ET_LOG(
      Info,
      "RSS after finishing text generation: %f MiB (0 if unsupported)",
      get_rss_bytes() / 1024.0 / 1024.0);
  cur_pos_ += num_generated_tokens;
  if (cur_pos_ == seq_len) {
    ET_LOG(Info, "Sequence length (%i tokens) reached!", seq_len);
  }

  stats_.num_prompt_tokens = num_prompt_tokens;
  stats_.num_generated_tokens = num_generated_tokens;
  print_report(stats_);
  print_performance_report(stats_, performance_output_path_);
  if (dump_logits) {
    save_logits(
        dump_logits_path_,
        prompt_processor_->get_all_logits(),
        token_generator_->get_all_logits());
  }
  if (stats_callback) {
    stats_callback(stats_);
  }
  return Error::Ok;
}

Result<ModelVersion> QNNMultimodalRunner::get_model_version() {
  if (!is_loaded()) {
    stats_.model_load_start_ms = time_in_ms();
    ET_CHECK_OK_OR_RETURN_ERROR(load());
    stats_.model_load_end_ms = time_in_ms();
  }
  return model_version_;
}

Result<MethodMeta> QNNMultimodalRunner::get_encoder_method_meta() {
  if (!is_loaded()) {
    ET_CHECK_OK_OR_RETURN_ERROR(load());
  }
  return encoder_->method_meta(kEncoderForwardName);
}

} // namespace example
