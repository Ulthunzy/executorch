/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <executorch/extension/llm/runner/image.h>
#include <executorch/extension/llm/runner/irunner.h>
#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/llm/runner/multimodal_input.h>
#include <executorch/extension/llm/runner/multimodal_runner.h>
#include <executorch/extension/llm/runner/text_llm_runner.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/platform.h>
#include <executorch/runtime/platform/runtime.h>

#include <executorch/extension/android/jni/jni_helper.h>

#if defined(ET_USE_THREADPOOL)
#include <executorch/extension/threadpool/cpuinfo_utils.h>
#include <executorch/extension/threadpool/threadpool.h>
#endif

#include <fbjni/ByteBuffer.h>
#include <fbjni/fbjni.h>

#if defined(EXECUTORCH_BUILD_QNN)
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/runner.h>
#include <executorch/extension/android/jni/encrypted_model_loader.h>
#include <executorch/extension/android/jni/dense/session.h>
#endif

#if defined(EXECUTORCH_BUILD_MEDIATEK)
#include <executorch/examples/mediatek/executor_runner/mtk_llama_runner.h>
#endif

namespace llm = ::executorch::extension::llm;
using ::executorch::runtime::Error;

using executorch::jni_helper::utf8_check_validity;

namespace executorch_jni {

class ExecuTorchLlmCallbackJni
    : public facebook::jni::JavaClass<ExecuTorchLlmCallbackJni> {
 public:
  constexpr static const char* kJavaDescriptor =
      "Lorg/pytorch/executorch/extension/llm/LlmCallback;";

  void onResult(const std::string& result) const {
    static auto cls = ExecuTorchLlmCallbackJni::javaClassStatic();
    static const auto method =
        cls->getMethod<void(facebook::jni::local_ref<jstring>)>("onResult");
    method(self(), facebook::jni::make_jstring(result));
  }

  void onStats(const llm::Stats& result) const {
    static auto cls = ExecuTorchLlmCallbackJni::javaClassStatic();
    static const auto on_stats_method =
        cls->getMethod<void(facebook::jni::local_ref<jstring>)>("onStats");
    on_stats_method(
        self(),
        facebook::jni::make_jstring(
            executorch::extension::llm::stats_to_json_string(result)));
  }

  void onError(int errorCode, const std::string& message) const {
    static auto cls = ExecuTorchLlmCallbackJni::javaClassStatic();
    static const auto on_error_method =
        cls->getMethod<void(jint, facebook::jni::local_ref<jstring>)>(
            "onError");
    on_error_method(self(), errorCode, facebook::jni::make_jstring(message));
  }
};

class ExecuTorchLlmJni : public facebook::jni::HybridClass<ExecuTorchLlmJni> {
 private:
  friend HybridBase;
  float temperature_ = 0.0f;
  int num_bos_ = 0;
  int num_eos_ = 0;
  int model_type_category_;
  std::unique_ptr<llm::IRunner> runner_;
  bool needs_bos_ = true;
  bool default_thinking_protection_ = false;

 public:
  constexpr static auto kJavaDescriptor =
      "Lorg/pytorch/executorch/extension/llm/LlmModule;";

  constexpr static int MODEL_TYPE_CATEGORY_LLM = 1;
  constexpr static int MODEL_TYPE_CATEGORY_MULTIMODAL = 2;
  constexpr static int MODEL_TYPE_MEDIATEK_LLAMA = 3;
  constexpr static int MODEL_TYPE_QNN_LLAMA = 4;

  static facebook::jni::local_ref<jhybriddata> initHybrid(
      facebook::jni::alias_ref<jclass> cls,
      jint model_type_category,
      facebook::jni::alias_ref<jstring> model_path,
      facebook::jni::alias_ref<jstring> tokenizer_path,
      jfloat temperature,
      facebook::jni::alias_ref<facebook::jni::JList<jstring>::javaobject>
          data_files,
      jint num_bos,
      jint num_eos,
      jint load_mode,
      facebook::jni::alias_ref<jstring> qnn_decoder_model,
      jint qnn_eval_mode,
      jboolean qnn_shared_buffer,
      jfloat qnn_repetition_penalty,
      facebook::jni::alias_ref<jstring> qnn_model_key_hex) {
    return initHybridWithOptions(
        cls,
        model_type_category,
        model_path,
        tokenizer_path,
        temperature,
        data_files,
        num_bos,
        num_eos,
        load_mode,
        qnn_decoder_model,
        qnn_eval_mode,
        qnn_shared_buffer,
        qnn_repetition_penalty,
        qnn_model_key_hex,
        128);
  }

  static facebook::jni::local_ref<jhybriddata> initHybridWithOptions(
      facebook::jni::alias_ref<jclass>,
      jint model_type_category,
      facebook::jni::alias_ref<jstring> model_path,
      facebook::jni::alias_ref<jstring> tokenizer_path,
      jfloat temperature,
      facebook::jni::alias_ref<facebook::jni::JList<jstring>::javaobject>
          data_files,
      jint num_bos,
      jint num_eos,
      jint load_mode,
      facebook::jni::alias_ref<jstring> qnn_decoder_model,
      jint qnn_eval_mode,
      jboolean qnn_shared_buffer,
      jfloat qnn_repetition_penalty,
      facebook::jni::alias_ref<jstring> qnn_model_key_hex,
      jint qnn_repeat_last_n) {
    return makeCxxInstance(
        model_type_category,
        model_path,
        tokenizer_path,
        temperature,
        data_files,
        num_bos,
        num_eos,
        load_mode,
        qnn_decoder_model,
        qnn_eval_mode,
        qnn_shared_buffer,
        qnn_repetition_penalty,
        qnn_model_key_hex,
        qnn_repeat_last_n);
  }

  static executorch::extension::Module::LoadMode load_mode_from_int(
      jint load_mode) {
    switch (load_mode) {
      case 0:
        return executorch::extension::Module::LoadMode::File;
      case 1:
        return executorch::extension::Module::LoadMode::Mmap;
      case 2:
        return executorch::extension::Module::LoadMode::MmapUseMlock;
      case 3:
        return executorch::extension::Module::LoadMode::
            MmapUseMlockIgnoreErrors;
      default:
        return executorch::extension::Module::LoadMode::Mmap;
    }
  }

  ExecuTorchLlmJni(
      jint model_type_category,
      facebook::jni::alias_ref<jstring> model_path,
      facebook::jni::alias_ref<jstring> tokenizer_path,
      jfloat temperature,
      facebook::jni::alias_ref<jobject> data_files,
      jint num_bos,
      jint num_eos,
      jint load_mode,
      facebook::jni::alias_ref<jstring> qnn_decoder_model,
      jint qnn_eval_mode,
      jboolean qnn_shared_buffer,
      jfloat qnn_repetition_penalty,
      facebook::jni::alias_ref<jstring> qnn_model_key_hex,
      jint qnn_repeat_last_n) {
    if (!model_path || !tokenizer_path || !std::isfinite(temperature) ||
        temperature < 0 || num_bos < -1 || num_bos > 127 || num_eos < -1 ||
        num_eos > 127) {
      executorch::jni_helper::throwExecutorchException(
          static_cast<uint32_t>(Error::InvalidArgument),
          "Invalid LlmModule paths, temperature, or BOS/EOS counts");
      return;
    }
    try {
      temperature_ = temperature;
      num_bos_ = std::max(0, num_bos);
      num_eos_ = std::max(0, num_eos);
#if defined(ET_USE_THREADPOOL)
      // Reserve 1 thread for the main thread.
      int32_t num_performant_cores =
          ::executorch::extension::cpuinfo::get_num_performant_cores() - 1;
      if (num_performant_cores > 0) {
        ET_LOG(
            Info, "Resetting threadpool to %d threads", num_performant_cores);
        ::executorch::extension::threadpool::get_threadpool()
            ->_unsafe_reset_threadpool(num_performant_cores);
      }
#endif

      model_type_category_ = model_type_category;
      auto cpp_load_mode = load_mode_from_int(load_mode);
      std::vector<std::string> data_files_vector;
      if (data_files != nullptr) {
        // Convert Java List<String> to C++ std::vector<string>
        auto list_class = facebook::jni::findClassStatic("java/util/List");
        auto size_method = list_class->getMethod<jint()>("size");
        auto get_method =
            list_class->getMethod<facebook::jni::local_ref<jobject>(jint)>(
                "get");

        jint size = size_method(data_files);
        for (jint i = 0; i < size; ++i) {
          auto str_obj = get_method(data_files, i);
          auto jstr = facebook::jni::static_ref_cast<jstring>(str_obj);
          data_files_vector.push_back(jstr->toStdString());
        }
      }
      if (model_type_category == MODEL_TYPE_CATEGORY_MULTIMODAL) {
        runner_ = llm::create_multimodal_runner(
            model_path->toStdString().c_str(),
            llm::load_tokenizer(tokenizer_path->toStdString()),
            std::nullopt,
            cpp_load_mode);
      } else if (model_type_category == MODEL_TYPE_CATEGORY_LLM) {
        runner_ = executorch::extension::llm::create_text_llm_runner(
            model_path->toStdString(),
            llm::load_tokenizer(tokenizer_path->toStdString()),
            data_files_vector,
            /*temperature=*/-1.0f,
            /*event_tracer=*/nullptr,
            /*method_name=*/"forward",
            cpp_load_mode);
#if defined(EXECUTORCH_BUILD_QNN)
      } else if (model_type_category == MODEL_TYPE_QNN_LLAMA) {
        if (!qnn_decoder_model ||
            (qnn_eval_mode != -1 &&
             qnn_eval_mode != example::Runner::kKVCached &&
             qnn_eval_mode != example::Runner::kHybrid) ||
            !std::isfinite(qnn_repetition_penalty) ||
            qnn_repetition_penalty < 1.0f || qnn_repeat_last_n <= 0) {
          throw std::invalid_argument("Invalid QNN runner configuration");
        }
        const auto decoder_model = qnn_decoder_model->toStdString();
        default_thinking_protection_ =
            decoder_model == "lincal" || decoder_model == "qwen3";
        const std::vector<std::string> supported_models = {
            "llama2",
            "llama3",
            "gemma",
            "gemma2",
            "gemma3",
            "gemma4",
            "granite",
            "phi_4_mini",
            "qwen2_5",
            "qwen3",
            "lincal",
            "smollm2_135m",
            "smollm3",
            "codegen",
            "glm"};
        if (std::find(
                supported_models.begin(),
                supported_models.end(),
                decoder_model) == supported_models.end()) {
          throw std::invalid_argument(
              "Unsupported QNN decoder: " + decoder_model);
        }
        auto encrypted = load_encrypted_model(
            model_path->toStdString(),
            qnn_model_key_hex ? qnn_model_key_hex->toStdString() : "");
        if (!encrypted.ok()) {
          executorch::jni_helper::throwExecutorchException(
              static_cast<uint32_t>(encrypted.error()),
              "Cannot open or authenticate QNN model");
          return;
        }
        std::unique_ptr<executorch::extension::Module> module;
        if (*encrypted) {
          if (!data_files_vector.empty()) {
            executorch::jni_helper::throwExecutorchException(
                static_cast<uint32_t>(Error::NotSupported),
                "Encrypted QNN containers do not support external data files");
            return;
          }
          module = std::make_unique<executorch::extension::Module>(
              std::move(*encrypted));
        } else {
          module = std::make_unique<executorch::extension::Module>(
              model_path->toStdString(), data_files_vector, cpp_load_mode);
        }
        if (qnn_eval_mode == -1) {
          const auto methods = module->method_names();
          if (!methods.ok()) {
            executorch::jni_helper::throwExecutorchException(
                static_cast<uint32_t>(methods.error()),
                "Cannot inspect QNN model methods");
            return;
          }
          if (!methods->count("kv_forward")) {
            throw std::invalid_argument("QNN model has no kv_forward method");
          }
          qnn_eval_mode = methods->count("prefill_forward")
              ? example::Runner::kHybrid
              : example::Runner::kKVCached;
        }
        ET_LOG(
            Info,
            "Creating QNN runner: decoder=%s, eval_mode=%d, temperature=%g, repetition_penalty=%g, repeat_last_n=%zu",
            decoder_model.c_str(),
            static_cast<int>(qnn_eval_mode),
            static_cast<double>(temperature_),
            static_cast<double>(qnn_repetition_penalty),
            static_cast<size_t>(qnn_repeat_last_n));
        runner_ = std::make_unique<example::Runner>(
            std::move(module),
            decoder_model.c_str(),
            model_path->toStdString().c_str(),
            tokenizer_path->toStdString().c_str(),
            "",
            "",
            temperature_,
            qnn_eval_mode,
            static_cast<bool>(qnn_shared_buffer),
            0,
            0,
            0,
            nullptr,
            nullptr,
            qnn_repetition_penalty,
            static_cast<size_t>(qnn_repeat_last_n));
#endif
#if defined(EXECUTORCH_BUILD_MEDIATEK)
      } else if (model_type_category == MODEL_TYPE_MEDIATEK_LLAMA) {
        runner_ = std::make_unique<MTKLlamaRunner>(
            model_path->toStdString().c_str(),
            tokenizer_path->toStdString().c_str());
        // Interpret the model type as LLM
        model_type_category_ = MODEL_TYPE_CATEGORY_LLM;
#endif
      }
      if (!runner_) {
        throw std::invalid_argument(
            "Unsupported model type or backend not compiled: " +
            std::to_string(model_type_category));
      }
    } catch (const facebook::jni::JniException&) {
      throw;
    } catch (const std::invalid_argument& e) {
      executorch::jni_helper::throwExecutorchException(
          static_cast<uint32_t>(Error::InvalidArgument), e.what());
    } catch (const std::exception& e) {
      executorch::jni_helper::throwExecutorchException(
          static_cast<uint32_t>(Error::Internal),
          std::string("Failed to create LlmModule: ") + e.what());
    } catch (...) {
      executorch::jni_helper::throwExecutorchException(
          static_cast<uint32_t>(Error::Internal),
          "Failed to create LlmModule: unknown native error");
    }
  }

  jint generate(
      facebook::jni::alias_ref<jstring> prompt,
      jint seq_len,
      facebook::jni::alias_ref<ExecuTorchLlmCallbackJni> callback,
      jboolean echo,
      jfloat temperature,
      jint num_bos,
      jint num_eos,
      jint max_new_tokens,
      jboolean warming,
      jboolean ignore_eos) {
    return generate_with_thinking(
        prompt,
        seq_len,
        callback,
        echo,
        temperature,
        num_bos,
        num_eos,
        max_new_tokens,
        warming,
        ignore_eos,
        default_thinking_protection_ && !ignore_eos,
        default_thinking_protection_ && !ignore_eos,
        512);
  }

  jint generate_with_thinking(
      facebook::jni::alias_ref<jstring> prompt,
      jint seq_len,
      facebook::jni::alias_ref<ExecuTorchLlmCallbackJni> callback,
      jboolean echo,
      jfloat temperature,
      jint num_bos,
      jint num_eos,
      jint max_new_tokens,
      jboolean warming,
      jboolean ignore_eos,
      jboolean repair_think_eos,
      jboolean enable_think_budget,
      jint think_budget) {
    Error err = Error::Ok;
    if (!prompt || (seq_len != -1 && seq_len <= 0) || max_new_tokens < -1 ||
        !std::isfinite(temperature) || num_bos < -1 || num_bos > 127 ||
        num_eos < -1 || num_eos > 127 || think_budget <= 0 ||
        ((repair_think_eos || enable_think_budget) && ignore_eos)) {
      err = Error::InvalidArgument;
      if (callback) {
        callback->onError(
            static_cast<int>(err),
            "generate() failed: invalid prompt or generation configuration");
      }
      return static_cast<jint>(err);
    }
    if ((repair_think_eos || enable_think_budget) &&
        model_type_category_ != MODEL_TYPE_QNN_LLAMA) {
      if (callback) {
        callback->onError(
            static_cast<int>(Error::NotSupported),
            "Thinking controls currently require a QNN text KV/hybrid runner");
      }
      return static_cast<jint>(Error::NotSupported);
    }

    float effective_temperature = temperature >= 0 ? temperature : temperature_;
    std::string token_buffer;
    auto token_callback = [callback, &token_buffer](const std::string& token) {
      token_buffer += token;
      if (!utf8_check_validity(token_buffer.c_str(), token_buffer.size())) {
        ET_LOG(
            Info, "Current token buffer is not valid UTF-8. Waiting for more.");
        return;
      }
      std::string result = token_buffer;
      token_buffer.clear();
      if (callback) {
        callback->onResult(result);
      }
    };

    if (!runner_) {
      err = Error::InvalidState;
      if (callback) {
        callback->onError(
            static_cast<int>(err), "generate() failed: runner not initialized");
      }
      return static_cast<jint>(err);
    }

    try {
      executorch::extension::llm::GenerationConfig config{
          .echo = static_cast<bool>(echo),
          .ignore_eos = static_cast<bool>(ignore_eos),
          .max_new_tokens = max_new_tokens,
          .warming = static_cast<bool>(warming),
          .seq_len = seq_len,
          .temperature = effective_temperature,
          .num_bos =
              (needs_bos_ || model_type_category_ == MODEL_TYPE_QNN_LLAMA)
              ? (num_bos >= 0 ? num_bos : num_bos_)
              : 0,
          .num_eos = num_eos >= 0 ? num_eos : num_eos_,
          .repair_think_eos = static_cast<bool>(repair_think_eos),
          .enable_think_budget = static_cast<bool>(enable_think_budget),
          .think_budget = think_budget,
      };
      err = runner_->generate(
          prompt->toStdString(),
          config,
          token_callback,
          [callback](const llm::Stats& result) {
            if (callback) {
              callback->onStats(result);
            }
          });
      if (err == Error::Ok && max_new_tokens != 0) {
        needs_bos_ = false;
      }
      if (err != Error::Ok && callback) {
        const bool invalid_qnn_context =
            model_type_category_ == MODEL_TYPE_QNN_LLAMA &&
            err == Error::InvalidState;
        callback->onError(
            static_cast<int>(err),
            invalid_qnn_context
                ? "QNN context is incomplete after an execution error; "
                  "call resetContext() and resend the conversation history"
                : "generate() failed with error code " +
                    std::to_string(static_cast<int>(err)));
      }
    } catch (const std::exception& e) {
      if (callback) {
        callback->onError(
            static_cast<int>(Error::Internal),
            std::string("generate() threw: ") + e.what());
      }
      return static_cast<jint>(Error::Internal);
    }
    return static_cast<jint>(err);
  }

  jint generate_with_chat(
      facebook::jni::alias_ref<jstring> prompt,
      jint seq_len,
      facebook::jni::alias_ref<ExecuTorchLlmCallbackJni> callback,
      jboolean echo,
      facebook::jni::alias_ref<jstring> system_prompt) {
    if (!prompt || seq_len < -1) {
      if (callback) {
        callback->onError(
            static_cast<int>(Error::InvalidArgument), "Invalid chat input");
      }
      return static_cast<jint>(Error::InvalidArgument);
    }
    const auto input = prompt->toStdString();
    if (input.empty()) {
      return 0;
    }
#if defined(EXECUTORCH_BUILD_QNN)
    if (model_type_category_ == MODEL_TYPE_QNN_LLAMA && runner_) {
      auto* qnn = static_cast<example::Runner*>(runner_.get());
      const auto version = qnn->get_decoder_model_version();
      if (!version.ok()) {
        if (callback) {
          callback->onError(
              static_cast<int>(version.error()), "Cannot load QNN chat model");
        }
        return static_cast<jint>(version.error());
      }
      const auto system = system_prompt ? system_prompt->toStdString() : "";
      const bool has_history = qnn->has_history();
      std::string formatted;
      switch (*version) {
        case example::kQwen3:
        case example::kQwen2_5:
        case example::kLincal:
          if (has_history) {
            formatted = "\n";
          } else if (!system.empty()) {
            formatted = "<|im_start|>system\n" + system + "<|im_end|>\n";
          }
          formatted += "<|im_start|>user\n" + input +
              "<|im_end|>\n<|im_start|>assistant\n";
          break;
        case example::kGemma3:
          formatted = "<start_of_turn>user\n";
          if (!has_history && !system.empty()) {
            formatted += system + "\n";
          }
          formatted += input + "<end_of_turn>\n<start_of_turn>model\n";
          break;
        default:
          formatted = input;
          break;
      }
      return generate(
          facebook::jni::make_jstring(formatted),
          -1,
          callback,
          echo,
          temperature_,
          num_bos_,
          num_eos_,
          seq_len,
          false,
          false);
    }
#endif
    return generate(
        prompt,
        seq_len,
        callback,
        echo,
        temperature_,
        num_bos_,
        num_eos_,
        -1,
        false,
        false);
  }

  // Returns status_code
  // Contract is valid within an AAR (JNI + corresponding Java code)
  jint prefill_text_input(facebook::jni::alias_ref<jstring> prompt) {
    if (!runner_) {
      return static_cast<jint>(Error::InvalidState);
    }
    if (!prompt) {
      return static_cast<jint>(Error::InvalidArgument);
    }
    std::vector<llm::MultimodalInput> inputs;
    inputs.emplace_back(llm::MultimodalInput{prompt->toStdString()});
    int32_t bos = needs_bos_ ? num_bos_ : 0;
    auto result = runner_->prefill(inputs, bos, /*num_eos=*/0);
    if (!result.ok()) {
      return static_cast<jint>(result.error());
    }
    needs_bos_ = false;
    return 0;
  }

  // Returns status_code
  jint prefill_images_input(
      facebook::jni::alias_ref<jintArray> image,
      jint width,
      jint height,
      jint channels) {
    if (!runner_) {
      return static_cast<jint>(Error::InvalidState);
    }
    if (image == nullptr) {
      return static_cast<jint>(Error::InvalidArgument);
    }
    auto image_size = image->size();
    if (image_size == 0) {
      return 0;
    }
    std::vector<jint> image_data_jint(image_size);
    std::vector<uint8_t> image_data(image_size);
    image->getRegion(0, image_size, image_data_jint.data());
    for (int i = 0; i < image_size; i++) {
      image_data[i] = image_data_jint[i];
    }
    llm::Image image_runner{std::move(image_data), width, height, channels};
    std::vector<llm::MultimodalInput> inputs;
    inputs.emplace_back(llm::MultimodalInput{std::move(image_runner)});
    int32_t bos = needs_bos_ ? num_bos_ : 0;
    auto result = runner_->prefill(inputs, bos, /*num_eos=*/0);
    if (!result.ok()) {
      return static_cast<jint>(result.error());
    }
    needs_bos_ = false;
    return 0;
  }

  jint prefill_images_input_buffer(
      facebook::jni::alias_ref<facebook::jni::JByteBuffer> image,
      jint width,
      jint height,
      jint channels) {
    if (!runner_) {
      return static_cast<jint>(Error::InvalidState);
    }
    if (image == nullptr || width <= 0 || height <= 0 || channels <= 0) {
      return static_cast<jint>(Error::InvalidArgument);
    }
    auto* data = image->getDirectBytes();
    auto size = image->getDirectSize();
    size_t expected = static_cast<size_t>(width) * height * channels;
    if (data == nullptr || size < expected) {
      return static_cast<jint>(Error::InvalidArgument);
    }
    std::vector<uint8_t> image_data(data, data + expected);
    llm::Image image_runner{std::move(image_data), width, height, channels};
    std::vector<llm::MultimodalInput> inputs;
    inputs.emplace_back(llm::MultimodalInput{std::move(image_runner)});
    int32_t bos = needs_bos_ ? num_bos_ : 0;
    auto result = runner_->prefill(inputs, bos, /*num_eos=*/0);
    if (!result.ok()) {
      return static_cast<jint>(result.error());
    }
    needs_bos_ = false;
    return 0;
  }

  jint prefill_normalized_images_input_buffer(
      facebook::jni::alias_ref<facebook::jni::JByteBuffer> image,
      jint width,
      jint height,
      jint channels) {
    if (!runner_) {
      return static_cast<jint>(Error::InvalidState);
    }
    if (image == nullptr || width <= 0 || height <= 0 || channels <= 0) {
      return static_cast<jint>(Error::InvalidArgument);
    }
    auto* data = image->getDirectBytes();
    auto size = image->getDirectSize();
    size_t expected_bytes =
        static_cast<size_t>(width) * height * channels * sizeof(float);
    if (data == nullptr || size < expected_bytes || size % sizeof(float) != 0) {
      return static_cast<jint>(Error::InvalidArgument);
    }
    size_t num_floats = static_cast<size_t>(width) * height * channels;
    std::vector<float> image_data(num_floats);
    std::memcpy(image_data.data(), data, expected_bytes);
    llm::Image image_runner{std::move(image_data), width, height, channels};
    std::vector<llm::MultimodalInput> inputs;
    inputs.emplace_back(llm::MultimodalInput{std::move(image_runner)});
    int32_t bos = needs_bos_ ? num_bos_ : 0;
    auto result = runner_->prefill(inputs, bos, /*num_eos=*/0);
    if (!result.ok()) {
      return static_cast<jint>(result.error());
    }
    needs_bos_ = false;
    return 0;
  }

  // Returns status_code
  jint prefill_normalized_images_input(
      facebook::jni::alias_ref<jfloatArray> image,
      jint width,
      jint height,
      jint channels) {
    if (!runner_) {
      return static_cast<jint>(Error::InvalidState);
    }
    if (image == nullptr) {
      return static_cast<jint>(Error::InvalidArgument);
    }
    auto image_size = image->size();
    if (image_size == 0) {
      return 0;
    }
    std::vector<jfloat> image_data_jfloat(image_size);
    std::vector<float> image_data(image_size);
    image->getRegion(0, image_size, image_data_jfloat.data());
    for (int i = 0; i < image_size; i++) {
      image_data[i] = image_data_jfloat[i];
    }
    llm::Image image_runner{std::move(image_data), width, height, channels};
    std::vector<llm::MultimodalInput> inputs;
    inputs.emplace_back(llm::MultimodalInput{std::move(image_runner)});
    int32_t bos = needs_bos_ ? num_bos_ : 0;
    auto result = runner_->prefill(inputs, bos, /*num_eos=*/0);
    if (!result.ok()) {
      return static_cast<jint>(result.error());
    }
    needs_bos_ = false;
    return 0;
  }

  // Returns status_code
  jint prefill_audio_input(
      facebook::jni::alias_ref<jbyteArray> data,
      jint batch_size,
      jint n_bins,
      jint n_frames) {
    if (!runner_) {
      return static_cast<jint>(Error::InvalidState);
    }
    if (data == nullptr) {
      return static_cast<jint>(Error::InvalidArgument);
    }
    auto data_size = data->size();
    if (data_size == 0) {
      return 0;
    }
    std::vector<jbyte> data_jbyte(data_size);
    std::vector<uint8_t> data_u8(data_size);
    data->getRegion(0, data_size, data_jbyte.data());
    for (int i = 0; i < data_size; i++) {
      data_u8[i] = data_jbyte[i];
    }
    llm::Audio audio{std::move(data_u8), batch_size, n_bins, n_frames};
    std::vector<llm::MultimodalInput> inputs;
    inputs.emplace_back(llm::MultimodalInput{std::move(audio)});
    int32_t bos = needs_bos_ ? num_bos_ : 0;
    auto result = runner_->prefill(inputs, bos, /*num_eos=*/0);
    if (!result.ok()) {
      return static_cast<jint>(result.error());
    }
    needs_bos_ = false;
    return 0;
  }

  // Returns status_code
  jint prefill_audio_input_float(
      facebook::jni::alias_ref<jfloatArray> data,
      jint batch_size,
      jint n_bins,
      jint n_frames) {
    if (!runner_) {
      return static_cast<jint>(Error::InvalidState);
    }
    if (data == nullptr) {
      return static_cast<jint>(Error::InvalidArgument);
    }
    auto data_size = data->size();
    if (data_size == 0) {
      return 0;
    }
    std::vector<jfloat> data_jfloat(data_size);
    std::vector<float> data_f(data_size);
    data->getRegion(0, data_size, data_jfloat.data());
    for (int i = 0; i < data_size; i++) {
      data_f[i] = data_jfloat[i];
    }
    llm::Audio audio{std::move(data_f), batch_size, n_bins, n_frames};
    std::vector<llm::MultimodalInput> inputs;
    inputs.emplace_back(llm::MultimodalInput{std::move(audio)});
    int32_t bos = needs_bos_ ? num_bos_ : 0;
    auto result = runner_->prefill(inputs, bos, /*num_eos=*/0);
    if (!result.ok()) {
      return static_cast<jint>(result.error());
    }
    needs_bos_ = false;
    return 0;
  }

  // Returns status_code
  jint prefill_raw_audio_input(
      facebook::jni::alias_ref<jbyteArray> data,
      jint batch_size,
      jint n_channels,
      jint n_samples) {
    if (!runner_) {
      return static_cast<jint>(Error::InvalidState);
    }
    if (data == nullptr) {
      return static_cast<jint>(Error::InvalidArgument);
    }
    auto data_size = data->size();
    if (data_size == 0) {
      return 0;
    }
    std::vector<jbyte> data_jbyte(data_size);
    std::vector<uint8_t> data_u8(data_size);
    data->getRegion(0, data_size, data_jbyte.data());
    for (int i = 0; i < data_size; i++) {
      data_u8[i] = data_jbyte[i];
    }
    llm::RawAudio audio{std::move(data_u8), batch_size, n_channels, n_samples};
    std::vector<llm::MultimodalInput> inputs;
    inputs.emplace_back(llm::MultimodalInput{std::move(audio)});
    int32_t bos = needs_bos_ ? num_bos_ : 0;
    auto result = runner_->prefill(inputs, bos, /*num_eos=*/0);
    if (!result.ok()) {
      return static_cast<jint>(result.error());
    }
    needs_bos_ = false;
    return 0;
  }

  void stop() {
    if (runner_) {
      runner_->stop();
    }
  }

  void reset_context() {
    if (runner_) {
      runner_->reset();
    }
    needs_bos_ = true;
  }

  jint load() {
    if (!runner_) {
      std::stringstream ss;
      ss << "Model runner was not created. model_type_category="
         << model_type_category_
         << ". Valid values: " << MODEL_TYPE_CATEGORY_LLM << " (LLM), "
         << MODEL_TYPE_CATEGORY_MULTIMODAL << " (Multimodal)";
      executorch::jni_helper::throwExecutorchException(
          static_cast<uint32_t>(Error::InvalidState), ss.str().c_str());
      return static_cast<jint>(Error::InvalidState);
    }
    const auto load_result = static_cast<jint>(runner_->load());
    if (load_result != static_cast<jint>(Error::Ok)) {
      executorch::jni_helper::throwExecutorchException(
          static_cast<uint32_t>(load_result), "Failed to load model runner");
    }
    return load_result;
  }

  static void registerNatives() {
    registerHybrid({
        makeNativeMethod("initHybrid", ExecuTorchLlmJni::initHybrid),
        makeNativeMethod(
            "initHybridWithOptions", ExecuTorchLlmJni::initHybridWithOptions),
        makeNativeMethod("generateNative", ExecuTorchLlmJni::generate),
        makeNativeMethod(
            "generateWithThinkingNative",
            ExecuTorchLlmJni::generate_with_thinking),
        makeNativeMethod(
            "generateWithChatNative", ExecuTorchLlmJni::generate_with_chat),
        makeNativeMethod("stopNative", ExecuTorchLlmJni::stop),
        makeNativeMethod("loadNative", ExecuTorchLlmJni::load),
        makeNativeMethod(
            "prefillImagesInput", ExecuTorchLlmJni::prefill_images_input),
        makeNativeMethod(
            "prefillImagesInputBuffer",
            ExecuTorchLlmJni::prefill_images_input_buffer),
        makeNativeMethod(
            "prefillNormalizedImagesInput",
            ExecuTorchLlmJni::prefill_normalized_images_input),
        makeNativeMethod(
            "prefillNormalizedImagesInputBuffer",
            ExecuTorchLlmJni::prefill_normalized_images_input_buffer),
        makeNativeMethod(
            "prefillAudioInput", ExecuTorchLlmJni::prefill_audio_input),
        makeNativeMethod(
            "prefillAudioInputFloat",
            ExecuTorchLlmJni::prefill_audio_input_float),
        makeNativeMethod(
            "prefillRawAudioInput", ExecuTorchLlmJni::prefill_raw_audio_input),
        makeNativeMethod(
            "prefillTextInput", ExecuTorchLlmJni::prefill_text_input),
        makeNativeMethod("resetContextNative", ExecuTorchLlmJni::reset_context),
    });
  }
};

#if defined(EXECUTORCH_BUILD_QNN)
class Qwen35DenseStream : public facebook::jni::JavaClass<Qwen35DenseStream> {
 public:
  static constexpr const char* kJavaDescriptor =
      "Lorg/pytorch/executorch/extension/llm/Qwen35DenseModule$Stream;";
  void onBytes(const std::string& piece) const {
    static const auto method =
        javaClassStatic()
            ->getMethod<void(
                facebook::jni::local_ref<facebook::jni::JArrayByte>)>(
                "onBytes");
    auto bytes = facebook::jni::JArrayByte::newArray(piece.size());
    bytes->setRegion(
        0, piece.size(), reinterpret_cast<const jbyte*>(piece.data()));
    method(self(), bytes);
  }
};
#endif

class Qwen35DenseJni : public facebook::jni::HybridClass<Qwen35DenseJni> {
 public:
  static constexpr const char* kJavaDescriptor =
      "Lorg/pytorch/executorch/extension/llm/Qwen35DenseModule;";

  static bool isAvailable(facebook::jni::alias_ref<jclass>) {
#if defined(EXECUTORCH_BUILD_QNN)
    return true;
#else
    return false;
#endif
  }

#if defined(EXECUTORCH_BUILD_QNN)
  explicit Qwen35DenseJni(executorch::android::dense::Config config)
      : session_(std::move(config)) {}

  static facebook::jni::local_ref<jhybriddata> initHybrid(
      facebook::jni::alias_ref<jclass>,
      std::string decoder,
      std::string embedding,
      std::string encoder,
      std::string tokenizer,
      std::string libraries,
      std::string system,
      jint capacity,
      jint height,
      jint width) {
    return makeCxxInstance(executorch::android::dense::Config{
        std::move(decoder),
        std::move(embedding),
        std::move(encoder),
        std::move(tokenizer),
        std::move(libraries),
        std::move(system),
        capacity,
        height,
        width});
  }

  facebook::jni::local_ref<jstring> generate(
      std::string prompt,
      std::string image,
      jint count,
      jfloat temperature,
      jboolean thinking,
      jboolean ignore_eos,
      facebook::jni::alias_ref<Qwen35DenseStream> callback) {
    try {
      auto stats = session_.generate(
          prompt,
          image,
          count,
          temperature,
          thinking,
          ignore_eos,
          [&](const std::string& piece) { callback->onBytes(piece); });
      return facebook::jni::make_jstring(stats);
    } catch (const std::invalid_argument& error) {
      facebook::jni::throwNewJavaException(
          "java/lang/IllegalArgumentException", error.what());
    }
  }
  void stop() {
    session_.stop();
  }
  void prepare() {
    session_.prepare();
  }
  void reset() {
    session_.reset();
  }
#endif

  static void registerNatives() {
    registerHybrid({
        makeNativeMethod("isAvailableNative", Qwen35DenseJni::isAvailable),
#if defined(EXECUTORCH_BUILD_QNN)
        makeNativeMethod("initHybrid", Qwen35DenseJni::initHybrid),
        makeNativeMethod("generateNative", Qwen35DenseJni::generate),
        makeNativeMethod("stopNative", Qwen35DenseJni::stop),
        makeNativeMethod("prepareNative", Qwen35DenseJni::prepare),
        makeNativeMethod("resetNative", Qwen35DenseJni::reset),
#endif
    });
  }

 private:
#if defined(EXECUTORCH_BUILD_QNN)
  executorch::android::dense::Session session_;
#endif
};

} // namespace executorch_jni

void register_natives_for_llm() {
  executorch_jni::ExecuTorchLlmJni::registerNatives();
  // Older AARs do not contain this optional API; keep their registration
  // intact.
  auto env = facebook::jni::Environment::current();
  auto dense_class =
      env->FindClass("org/pytorch/executorch/extension/llm/Qwen35DenseModule");
  if (dense_class) {
    env->DeleteLocalRef(dense_class);
    executorch_jni::Qwen35DenseJni::registerNatives();
  } else {
    env->ExceptionClear();
  }
}
