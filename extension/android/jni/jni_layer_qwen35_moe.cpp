#include "moe/session.h"

#include <jni.h>
#include <cmath>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>

namespace {
using qwen35_jni::Session;
std::mutex registry_mutex;
std::map<jlong, std::shared_ptr<Session>> sessions;
jlong next_id = 1;
bool releasing = false;

std::string utf8(JNIEnv* env, jstring value) {
  if (!value)
    throw std::invalid_argument("Null MoE string argument");
  const jsize size = env->GetStringLength(value);
  struct ReleaseChars {
    JNIEnv* env;
    jstring value;
    void operator()(const jchar* chars) const {
      env->ReleaseStringChars(value, chars);
    }
  };
  std::unique_ptr<const jchar, ReleaseChars> chars(
      env->GetStringChars(value, nullptr), ReleaseChars{env, value});
  const jchar* data = chars.get();
  if (!data)
    throw std::runtime_error("Cannot read Java string");
  std::string result;
  bool valid = true;
  for (jsize i = 0; i < size; ++i) {
    uint32_t c = data[i];
    if (c >= 0xd800 && c <= 0xdbff && i + 1 < size && data[i + 1] >= 0xdc00 &&
        data[i + 1] <= 0xdfff) {
      c = 0x10000 + ((c - 0xd800) << 10) + (data[++i] - 0xdc00);
    } else if ((c >= 0xd800 && c <= 0xdfff) || c == 0) {
      valid = false;
      break;
    }
    if (c < 0x80)
      result += char(c);
    else if (c < 0x800) {
      result += char(0xc0 | (c >> 6));
      result += char(0x80 | (c & 63));
    } else if (c < 0x10000) {
      result += char(0xe0 | (c >> 12));
      result += char(0x80 | ((c >> 6) & 63));
      result += char(0x80 | (c & 63));
    } else {
      result += char(0xf0 | (c >> 18));
      result += char(0x80 | ((c >> 12) & 63));
      result += char(0x80 | ((c >> 6) & 63));
      result += char(0x80 | (c & 63));
    }
  }
  if (!valid)
    throw std::invalid_argument("NUL or unpaired surrogate in MoE string");
  return result;
}

void raise(JNIEnv* env, const std::exception& error) {
  if (env->ExceptionCheck())
    return;
  const char* type = dynamic_cast<const std::invalid_argument*>(&error)
      ? "java/lang/IllegalArgumentException"
      : "java/lang/IllegalStateException";
  auto cls = env->FindClass(type);
  if (cls) {
    env->ThrowNew(cls, error.what());
    env->DeleteLocalRef(cls);
  }
}

std::shared_ptr<Session> find(jlong handle) {
  std::lock_guard<std::mutex> lock(registry_mutex);
  const auto found = sessions.find(handle);
  if (found == sessions.end())
    throw std::runtime_error("MoE module is closed");
  return found->second;
}

struct CallbackException : std::exception {};
} // namespace

extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM*, void*) {
  return JNI_VERSION_1_6;
}

extern "C" JNIEXPORT jlong JNICALL
Java_org_pytorch_executorch_extension_llm_Qwen35MoeModule_createNative(
    JNIEnv* env,
    jclass,
    jstring model,
    jstring shared,
    jstring tokenizer,
    jstring workspace,
    jstring libraries,
    jstring system,
    jint capacity,
    jint experts,
    jboolean vision) {
  try {
    std::lock_guard<std::mutex> lock(registry_mutex);
    // The existing MoE backend and shared context provider are process
    // singletons.
    if (!sessions.empty() || releasing)
      throw std::runtime_error(
          "Only one Qwen35 MoE module may be open per process");
    auto root = utf8(env, model), cache = utf8(env, workspace);
    auto files = std::make_unique<qwen35_jni::Workspace>(
        root, cache, utf8(env, libraries));
    std::vector<std::string> args = {
        "qwen35_moe_jni",
        "--model-root",
        files->root(),
        "--shared-model",
        utf8(env, shared),
        "--tokenizer-path",
        utf8(env, tokenizer),
        "--system-prompt",
        utf8(env, system),
        "--max-seq-len",
        std::to_string(capacity),
        "--cache-experts",
        std::to_string(experts),
        "--prompt",
        "JNI initialization"};
    if (vision)
      args.push_back("--interactive");
    std::vector<char*> argv;
    for (auto& arg : args)
      argv.push_back(arg.data());
    auto options = moe_chat::parse_options(argv.size(), argv.data());
    if (!std::filesystem::is_regular_file(options.tokenizer_path))
      throw std::invalid_argument("Tokenizer file does not exist");
    if (vision &&
        !std::filesystem::is_regular_file(
            options.model_root + "/prefill/vision.bin"))
      throw std::invalid_argument("Vision graph is missing");
    const jlong handle = next_id++;
    sessions.emplace(
        handle,
        std::make_shared<Session>(std::move(options), std::move(files)));
    return handle;
  } catch (const std::exception& error) {
    raise(env, error);
    return 0;
  }
}

extern "C" JNIEXPORT jstring JNICALL
Java_org_pytorch_executorch_extension_llm_Qwen35MoeModule_generateNative(
    JNIEnv* env,
    jclass,
    jlong handle,
    jstring prompt,
    jstring image,
    jint count,
    jfloat temperature,
    jboolean thinking,
    jobject callback) {
  try {
    auto session = find(handle);
    if (!callback || count < 1 || count > session->capacity() ||
        !std::isfinite(temperature) ||
        temperature < 0)
      throw std::invalid_argument("Invalid generation settings or callback");
    moe_chat::Request request;
    request.prompt = utf8(env, prompt);
    request.image_path = utf8(env, image);
    request.max_new_tokens = count;
    request.temperature = temperature;
    request.thinking = thinking;
    if (request.prompt.empty())
      throw std::invalid_argument("Prompt is empty");
    auto cls = env->GetObjectClass(callback);
    if (!cls)
      throw CallbackException{};
    auto method = env->GetMethodID(cls, "onBytes", "([B)V");
    env->DeleteLocalRef(cls);
    if (!method)
      return nullptr;
    const auto stats =
        session->invoke(std::move(request), [&](const std::string& text) {
          auto bytes = env->NewByteArray(text.size());
          if (!bytes)
            throw CallbackException{};
          env->SetByteArrayRegion(
              bytes,
              0,
              text.size(),
              reinterpret_cast<const jbyte*>(text.data()));
          if (!env->ExceptionCheck())
            env->CallVoidMethod(callback, method, bytes);
          env->DeleteLocalRef(bytes);
          if (env->ExceptionCheck())
            throw CallbackException{};
        });
    return env->NewStringUTF(stats.c_str());
  } catch (const std::exception& error) {
    raise(env, error);
    return nullptr;
  }
}

extern "C" JNIEXPORT void JNICALL
Java_org_pytorch_executorch_extension_llm_Qwen35MoeModule_stopNative(
    JNIEnv* env,
    jclass,
    jlong handle) {
  try {
    find(handle)->stop();
  } catch (const std::exception& error) {
    raise(env, error);
  }
}

extern "C" JNIEXPORT void JNICALL
Java_org_pytorch_executorch_extension_llm_Qwen35MoeModule_resetNative(
    JNIEnv* env,
    jclass,
    jlong handle) {
  try {
    moe_chat::Request request;
    request.kind = moe_chat::Request::Kind::Reset;
    find(handle)->invoke(request, [](const std::string&) {});
  } catch (const std::exception& error) {
    raise(env, error);
  }
}

extern "C" JNIEXPORT void JNICALL
Java_org_pytorch_executorch_extension_llm_Qwen35MoeModule_closeNative(
    JNIEnv* env,
    jclass,
    jlong handle) {
  try {
    std::shared_ptr<Session> session;
    {
      std::lock_guard<std::mutex> lock(registry_mutex);
      const auto found = sessions.find(handle);
      if (found == sessions.end())
        return;
      session = found->second;
      sessions.erase(found);
      releasing = true;
    }
    session->shutdown();
    session.reset();
    std::lock_guard<std::mutex> lock(registry_mutex);
    releasing = false;
  } catch (const std::exception& error) {
    raise(env, error);
  }
}
