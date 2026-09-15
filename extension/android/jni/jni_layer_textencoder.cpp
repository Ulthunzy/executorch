/*
 * JNI bindings for the legacy sentence-embedding encoder.
 */

#include <executorch/extension/android/jni/jni_helper.h>
#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

#include <fbjni/fbjni.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace executorch_jni {

using executorch::runtime::Error;

class ExecuTorchTextEncoderJni
    : public facebook::jni::HybridClass<ExecuTorchTextEncoderJni> {
 private:
  friend HybridBase;
  executorch::extension::Module module_;
  int max_seq_len_;
  std::unique_ptr<tokenizers::Tokenizer> tokenizer_;
  std::string tokenizer_path_;

  facebook::jni::local_ref<jfloatArray> execute(
      std::vector<int32_t>& ids,
      std::vector<float>& mask) {
    using namespace executorch::extension;
    using executorch::aten::ScalarType;
    const std::vector<executorch::aten::SizesType> sizes = {
        1, static_cast<executorch::aten::SizesType>(ids.size())};
    auto ids_tensor = from_blob(ids.data(), sizes, ScalarType::Int);
    auto mask_tensor = from_blob(mask.data(), sizes, ScalarType::Float);
    auto result = module_.forward({*ids_tensor, *mask_tensor});
    if (!result.ok()) {
      executorch::jni_helper::throwExecutorchException(
          static_cast<uint32_t>(result.error()),
          "TextEncoder execution failed");
      return {};
    }
    if (result->empty() || !result->at(0).isTensor() ||
        result->at(0).toTensor().scalar_type() != ScalarType::Float) {
      executorch::jni_helper::throwExecutorchException(
          static_cast<uint32_t>(Error::InvalidArgument),
          "TextEncoder expects a float32 tensor as its first output");
      return {};
    }
    const auto tensor = result->at(0).toTensor();
    if (tensor.numel() > std::numeric_limits<jsize>::max()) {
      executorch::jni_helper::throwExecutorchException(
          static_cast<uint32_t>(Error::InvalidArgument),
          "TextEncoder output is too large for a Java array");
      return {};
    }
    auto output = facebook::jni::make_float_array(tensor.numel());
    output->setRegion(0, tensor.numel(), tensor.const_data_ptr<float>());
    return output;
  }

 public:
  constexpr static auto kJavaDescriptor =
      "Lorg/pytorch/executorch/extension/textencoder/TextEncoderModule;";

  static facebook::jni::local_ref<jhybriddata> initHybrid(
      facebook::jni::alias_ref<jclass>,
      facebook::jni::alias_ref<jstring> model_path,
      jint hidden_size,
      jint max_seq_len) {
    return makeCxxInstance(model_path, hidden_size, max_seq_len);
  }

  ExecuTorchTextEncoderJni(
      facebook::jni::alias_ref<jstring> model_path,
      jint /*hidden_size*/,
      jint max_seq_len)
      : module_(
            model_path->toStdString(),
            executorch::extension::Module::LoadMode::MmapUseMlockIgnoreErrors),
        max_seq_len_(max_seq_len) {}

  facebook::jni::local_ref<jfloatArray> encode(
      facebook::jni::alias_ref<jintArray> input_ids,
      facebook::jni::alias_ref<jfloatArray> attention_mask) {
    std::vector<int32_t> ids(input_ids->size());
    std::vector<float> mask(attention_mask->size());
    input_ids->getRegion(0, ids.size(), ids.data());
    attention_mask->getRegion(0, mask.size(), mask.data());
    return execute(ids, mask);
  }

  facebook::jni::local_ref<jfloatArray> encode_text(
      facebook::jni::alias_ref<jstring> tokenizer_path,
      facebook::jni::alias_ref<jstring> text) {
    if (!tokenizer_path || !text || text->toStdString().empty()) {
      executorch::jni_helper::throwExecutorchException(
          static_cast<uint32_t>(Error::InvalidArgument),
          "TextEncoder requires a tokenizer path and nonempty text");
      return {};
    }
    const auto path = tokenizer_path->toStdString();
    if (!tokenizer_ || path != tokenizer_path_) {
      tokenizer_ = executorch::extension::llm::load_tokenizer(path);
      tokenizer_path_ = path;
    }
    if (!tokenizer_ || !tokenizer_->is_loaded()) {
      executorch::jni_helper::throwExecutorchException(
          static_cast<uint32_t>(Error::InvalidArgument),
          "TextEncoder could not load tokenizer");
      return {};
    }
    auto tokens = tokenizer_->encode(text->toStdString(), 0, 0);
    if (!tokens.ok() || tokens->empty()) {
      executorch::jni_helper::throwExecutorchException(
          static_cast<uint32_t>(Error::InvalidArgument),
          "TextEncoder tokenization failed");
      return {};
    }
    const size_t length = max_seq_len_ > 0 ? max_seq_len_ : tokens->size();
    if (length > std::numeric_limits<executorch::aten::SizesType>::max()) {
      executorch::jni_helper::throwExecutorchException(
          static_cast<uint32_t>(Error::InvalidArgument),
          "TextEncoder input is too large");
      return {};
    }
    std::vector<int32_t> ids(length, 0);
    std::vector<float> mask(length, 0.0f);
    for (size_t i = 0; i < std::min(length, tokens->size()); ++i) {
      if (tokens->at(i) > std::numeric_limits<int32_t>::max()) {
        executorch::jni_helper::throwExecutorchException(
            static_cast<uint32_t>(Error::InvalidArgument),
            "TextEncoder token ID exceeds int32 range");
        return {};
      }
      ids[i] = tokens->at(i);
      mask[i] = 1.0f;
    }
    return execute(ids, mask);
  }

  static void registerNatives() {
    registerHybrid({
        makeNativeMethod("initHybrid", ExecuTorchTextEncoderJni::initHybrid),
        makeNativeMethod("encodeNative", ExecuTorchTextEncoderJni::encode),
        makeNativeMethod(
            "encodeTextNative", ExecuTorchTextEncoderJni::encode_text),
    });
  }
};
} // namespace executorch_jni

void register_natives_for_textencoder() {
  executorch_jni::ExecuTorchTextEncoderJni::registerNatives();
}
