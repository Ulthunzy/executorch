#include <jni.h>

#include <executorch/examples/qualcomm/oss_scripts/yolo12/runner/inference.h>
#include <executorch/extension/module/module.h>
#include <executorch/runtime/platform/runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

using ::executorch::extension::Module;
using ::executorch::runtime::EValue;
using ::executorch::runtime::Error;

namespace {

struct Yolo12Handle {
  std::unique_ptr<Module> module;
  std::vector<executorch::aten::SizesType> input_shape;
  std::vector<float> input_buffer;
  float score_threshold{0.5f};
  float nms_threshold{0.5f};
  bool loaded{false};
  std::mutex mutex;
};

struct JavaClassCache {
  jclass detection_class{nullptr};
  jmethodID detection_ctor{nullptr};
  jclass result_class{nullptr};
  jmethodID result_ctor{nullptr};
};

struct JavaFrameClassCache {
  jclass frame_result_class{nullptr};
  jmethodID frame_result_ctor{nullptr};
};

struct RgbaFrameInput {
  const uint8_t* rgba{nullptr};
  int width{0};
  int height{0};
  int row_stride{0};
  int rotation_degrees{0};
  bool mirror_horizontally{false};
};

struct DetectionRunOutput {
  example::InputMeta input_meta;
  std::vector<example::Detection> detections;
  float latency_ms{0.0f};
};

JavaClassCache g_java_cache;
JavaFrameClassCache g_frame_java_cache;
std::once_flag g_java_cache_once;
std::once_flag g_frame_java_cache_once;
std::once_flag g_runtime_init_once;

void throwJavaException(
    JNIEnv* env,
    const char* class_name,
    const std::string& message) {
  jclass exception_class = env->FindClass(class_name);
  if (exception_class != nullptr) {
    env->ThrowNew(exception_class, message.c_str());
    env->DeleteLocalRef(exception_class);
  }
}

std::string jstringToString(JNIEnv* env, jstring value) {
  if (value == nullptr) {
    return "";
  }
  const char* chars = env->GetStringUTFChars(value, nullptr);
  if (chars == nullptr) {
    return "";
  }
  std::string result(chars);
  env->ReleaseStringUTFChars(value, chars);
  return result;
}

void ensureRuntimeInitialized() {
  std::call_once(g_runtime_init_once, []() {
    executorch::runtime::runtime_init();
  });
}

bool initializeJavaCache(JNIEnv* env) {
  bool success = true;
  std::call_once(g_java_cache_once, [&]() {
    jclass detection_local = env->FindClass(
        "org/pytorch/executorch/extension/yolo/Yolo12Detection");
    if (detection_local == nullptr) {
      success = false;
      return;
    }
    g_java_cache.detection_class =
        reinterpret_cast<jclass>(env->NewGlobalRef(detection_local));
    env->DeleteLocalRef(detection_local);
    if (g_java_cache.detection_class == nullptr) {
      success = false;
      return;
    }
    g_java_cache.detection_ctor = env->GetMethodID(
        g_java_cache.detection_class, "<init>", "(IFIIII)V");
    if (g_java_cache.detection_ctor == nullptr) {
      success = false;
      return;
    }

    jclass result_local =
        env->FindClass("org/pytorch/executorch/extension/yolo/Yolo12Result");
    if (result_local == nullptr) {
      success = false;
      return;
    }
    g_java_cache.result_class =
        reinterpret_cast<jclass>(env->NewGlobalRef(result_local));
    env->DeleteLocalRef(result_local);
    if (g_java_cache.result_class == nullptr) {
      success = false;
      return;
    }
    g_java_cache.result_ctor = env->GetMethodID(
        g_java_cache.result_class,
        "<init>",
        "([Lorg/pytorch/executorch/extension/yolo/Yolo12Detection;IIF)V");
    if (g_java_cache.result_ctor == nullptr) {
      success = false;
    }
  });
  return success && g_java_cache.detection_class != nullptr &&
      g_java_cache.result_class != nullptr &&
      g_java_cache.detection_ctor != nullptr && g_java_cache.result_ctor != nullptr;
}

jobjectArray makeDetectionArray(
    JNIEnv* env,
    const std::vector<example::Detection>& detections) {
  if (!initializeJavaCache(env)) {
    throwJavaException(
        env,
        "java/lang/RuntimeException",
        "Failed to initialize YOLO12 JNI class cache.");
    return nullptr;
  }

  jobjectArray detection_array = env->NewObjectArray(
      static_cast<jsize>(detections.size()), g_java_cache.detection_class, nullptr);
  if (detection_array == nullptr) {
    return nullptr;
  }

  for (jsize index = 0; index < static_cast<jsize>(detections.size()); ++index) {
    const auto& detection = detections[static_cast<size_t>(index)];
    jobject detection_object = env->NewObject(
        g_java_cache.detection_class,
        g_java_cache.detection_ctor,
        static_cast<jint>(detection.class_id),
        static_cast<jfloat>(detection.confidence),
        static_cast<jint>(detection.left),
        static_cast<jint>(detection.top),
        static_cast<jint>(detection.right),
        static_cast<jint>(detection.bottom));
    if (detection_object == nullptr) {
      env->DeleteLocalRef(detection_array);
      return nullptr;
    }
    env->SetObjectArrayElement(detection_array, index, detection_object);
    env->DeleteLocalRef(detection_object);
  }

  return detection_array;
}

bool initializeFrameJavaCache(JNIEnv* env) {
  if (!initializeJavaCache(env)) {
    return false;
  }

  bool success = true;
  std::call_once(g_frame_java_cache_once, [&]() {
    jclass frame_result_local =
        env->FindClass("org/pytorch/executorch/extension/yolo/Yolo12FrameResult");
    if (frame_result_local == nullptr) {
      success = false;
      return;
    }
    g_frame_java_cache.frame_result_class = reinterpret_cast<jclass>(
        env->NewGlobalRef(frame_result_local));
    env->DeleteLocalRef(frame_result_local);
    if (g_frame_java_cache.frame_result_class == nullptr) {
      success = false;
      return;
    }
    g_frame_java_cache.frame_result_ctor = env->GetMethodID(
        g_frame_java_cache.frame_result_class,
        "<init>",
        "([BIIIJ[Lorg/pytorch/executorch/extension/yolo/Yolo12Detection;F)V");
    if (g_frame_java_cache.frame_result_ctor == nullptr) {
      success = false;
    }
  });

  return success && g_frame_java_cache.frame_result_class != nullptr &&
      g_frame_java_cache.frame_result_ctor != nullptr;
}

jobject makeResultObject(
    JNIEnv* env,
    const std::vector<example::Detection>& detections,
    jint image_width,
    jint image_height,
    jfloat latency_ms) {
  if (!initializeJavaCache(env)) {
    throwJavaException(
        env,
        "java/lang/RuntimeException",
        "Failed to initialize YOLO12 JNI class cache.");
    return nullptr;
  }

  jobjectArray detection_array = makeDetectionArray(env, detections);
  if (detection_array == nullptr) {
    return nullptr;
  }

  jobject result = env->NewObject(
      g_java_cache.result_class,
      g_java_cache.result_ctor,
      detection_array,
      image_width,
      image_height,
      latency_ms);
  env->DeleteLocalRef(detection_array);
  return result;
}

jobject makeFrameResultObject(
    JNIEnv* env,
    const std::vector<uint8_t>& rgba_bytes,
    jint image_width,
    jint image_height,
    jint row_stride,
    jlong timestamp_ns,
    const std::vector<example::Detection>& detections,
    jfloat latency_ms) {
  if (!initializeFrameJavaCache(env)) {
    throwJavaException(
        env,
        "java/lang/RuntimeException",
        "Failed to initialize YOLO12 JNI frame class cache.");
    return nullptr;
  }

  jobjectArray detection_array = makeDetectionArray(env, detections);
  if (detection_array == nullptr) {
    return nullptr;
  }

  jbyteArray rgba_array =
      env->NewByteArray(static_cast<jsize>(rgba_bytes.size()));
  if (rgba_array == nullptr) {
    env->DeleteLocalRef(detection_array);
    return nullptr;
  }
  env->SetByteArrayRegion(
      rgba_array,
      0,
      static_cast<jsize>(rgba_bytes.size()),
      reinterpret_cast<const jbyte*>(rgba_bytes.data()));

  jobject result = env->NewObject(
      g_frame_java_cache.frame_result_class,
      g_frame_java_cache.frame_result_ctor,
      rgba_array,
      image_width,
      image_height,
      row_stride,
      timestamp_ns,
      detection_array,
      latency_ms);
  env->DeleteLocalRef(rgba_array);
  env->DeleteLocalRef(detection_array);
  return result;
}

int normalizedRotation(int rotation_degrees) {
  int normalized = rotation_degrees % 360;
  if (normalized < 0) {
    normalized += 360;
  }
  return normalized;
}

bool isSupportedRotation(int rotation_degrees) {
  return rotation_degrees == 0 || rotation_degrees == 90 ||
      rotation_degrees == 180 || rotation_degrees == 270;
}

struct SampledPixel {
  float red{0.0f};
  float green{0.0f};
  float blue{0.0f};
};

struct OrientedRgbaFrame {
  int width{0};
  int height{0};
  int row_stride{0};
  std::vector<uint8_t> rgba;
};

SampledPixel samplePixel(
    const uint8_t* rgba,
    int source_width,
    int source_height,
    int row_stride,
    float source_x,
    float source_y) {
  const float clamped_x = std::clamp(source_x, 0.0f, source_width - 1.0f);
  const float clamped_y = std::clamp(source_y, 0.0f, source_height - 1.0f);
  const int x0 = static_cast<int>(std::floor(clamped_x));
  const int y0 = static_cast<int>(std::floor(clamped_y));
  const int x1 = std::min(x0 + 1, source_width - 1);
  const int y1 = std::min(y0 + 1, source_height - 1);
  const float wx = clamped_x - static_cast<float>(x0);
  const float wy = clamped_y - static_cast<float>(y0);

  auto load_pixel = [&](int x, int y) -> SampledPixel {
    const size_t offset =
        static_cast<size_t>(y) * static_cast<size_t>(row_stride) +
        static_cast<size_t>(x) * 4U;
    return SampledPixel{
        rgba[offset + 0] / 255.0f,
        rgba[offset + 1] / 255.0f,
        rgba[offset + 2] / 255.0f,
    };
  };

  const SampledPixel top_left = load_pixel(x0, y0);
  const SampledPixel top_right = load_pixel(x1, y0);
  const SampledPixel bottom_left = load_pixel(x0, y1);
  const SampledPixel bottom_right = load_pixel(x1, y1);

  auto bilinear = [&](float tl, float tr, float bl, float br) -> float {
    const float top = tl + (tr - tl) * wx;
    const float bottom = bl + (br - bl) * wx;
    return top + (bottom - top) * wy;
  };

  return SampledPixel{
      bilinear(
          top_left.red, top_right.red, bottom_left.red, bottom_right.red),
      bilinear(
          top_left.green,
          top_right.green,
          bottom_left.green,
          bottom_right.green),
      bilinear(
          top_left.blue, top_right.blue, bottom_left.blue, bottom_right.blue),
  };
}

example::InputMeta preprocessRgbaFrame(
    const uint8_t* rgba,
    int source_width,
    int source_height,
    int row_stride,
    int rotation_degrees,
    bool mirror_horizontally,
    int target_width,
    int target_height,
    std::vector<float>* output) {
  const int rotated_width =
      (rotation_degrees == 90 || rotation_degrees == 270) ? source_height
                                                          : source_width;
  const int rotated_height =
      (rotation_degrees == 90 || rotation_degrees == 270) ? source_width
                                                          : source_height;
  const float scale = std::min(
      static_cast<float>(target_width) / static_cast<float>(rotated_width),
      static_cast<float>(target_height) / static_cast<float>(rotated_height));
  const int resized_width =
      std::max(1, static_cast<int>(std::lround(rotated_width * scale)));
  const int resized_height =
      std::max(1, static_cast<int>(std::lround(rotated_height * scale)));
  const int pad_x = (target_width - resized_width) / 2;
  const int pad_y = (target_height - resized_height) / 2;

  output->assign(
      static_cast<size_t>(3) * static_cast<size_t>(target_width) *
          static_cast<size_t>(target_height),
      0.0f);

  const size_t channel_stride =
      static_cast<size_t>(target_width) * static_cast<size_t>(target_height);

  for (int target_y = 0; target_y < target_height; ++target_y) {
    const float rotated_y =
        (static_cast<float>(target_y - pad_y) + 0.5f) / scale - 0.5f;
    if (rotated_y < 0.0f || rotated_y >= static_cast<float>(rotated_height)) {
      continue;
    }
    for (int target_x = 0; target_x < target_width; ++target_x) {
      float rotated_x =
          (static_cast<float>(target_x - pad_x) + 0.5f) / scale - 0.5f;
      if (rotated_x < 0.0f ||
          rotated_x >= static_cast<float>(rotated_width)) {
        continue;
      }

      if (mirror_horizontally) {
        rotated_x = static_cast<float>(rotated_width - 1) - rotated_x;
      }

      float source_x = 0.0f;
      float source_y = 0.0f;
      switch (rotation_degrees) {
        case 0:
          source_x = rotated_x;
          source_y = rotated_y;
          break;
        case 90:
          source_x = rotated_y;
          source_y = static_cast<float>(source_height - 1) - rotated_x;
          break;
        case 180:
          source_x = static_cast<float>(source_width - 1) - rotated_x;
          source_y = static_cast<float>(source_height - 1) - rotated_y;
          break;
        case 270:
          source_x = static_cast<float>(source_width - 1) - rotated_y;
          source_y = rotated_x;
          break;
      }

      const SampledPixel pixel = samplePixel(
          rgba,
          source_width,
          source_height,
          row_stride,
          source_x,
          source_y);
      const size_t index = static_cast<size_t>(target_y) *
              static_cast<size_t>(target_width) +
          static_cast<size_t>(target_x);
      (*output)[index] = pixel.red;
      (*output)[channel_stride + index] = pixel.green;
      (*output)[channel_stride * 2U + index] = pixel.blue;
    }
  }

  return example::InputMeta{
      .original_width = rotated_width,
      .original_height = rotated_height,
      .scale = scale,
      .pad_x = pad_x,
      .pad_y = pad_y,
  };
}

OrientedRgbaFrame orientRgbaFrame(
    const uint8_t* rgba,
    int source_width,
    int source_height,
    int row_stride,
    int rotation_degrees,
    bool mirror_horizontally) {
  const int output_width =
      (rotation_degrees == 90 || rotation_degrees == 270) ? source_height
                                                          : source_width;
  const int output_height =
      (rotation_degrees == 90 || rotation_degrees == 270) ? source_width
                                                          : source_height;
  const int output_row_stride = output_width * 4;

  OrientedRgbaFrame output;
  output.width = output_width;
  output.height = output_height;
  output.row_stride = output_row_stride;
  output.rgba.resize(
      static_cast<size_t>(output_row_stride) * static_cast<size_t>(output_height));

  for (int target_y = 0; target_y < output_height; ++target_y) {
    for (int target_x = 0; target_x < output_width; ++target_x) {
      int rotated_x = target_x;
      if (mirror_horizontally) {
        rotated_x = output_width - 1 - rotated_x;
      }

      int source_x = 0;
      int source_y = 0;
      switch (rotation_degrees) {
        case 0:
          source_x = rotated_x;
          source_y = target_y;
          break;
        case 90:
          source_x = target_y;
          source_y = source_height - 1 - rotated_x;
          break;
        case 180:
          source_x = source_width - 1 - rotated_x;
          source_y = source_height - 1 - target_y;
          break;
        case 270:
          source_x = source_width - 1 - target_y;
          source_y = rotated_x;
          break;
      }

      const size_t source_offset =
          static_cast<size_t>(source_y) * static_cast<size_t>(row_stride) +
          static_cast<size_t>(source_x) * 4U;
      const size_t target_offset =
          static_cast<size_t>(target_y) * static_cast<size_t>(output_row_stride) +
          static_cast<size_t>(target_x) * 4U;

      output.rgba[target_offset + 0] = rgba[source_offset + 0];
      output.rgba[target_offset + 1] = rgba[source_offset + 1];
      output.rgba[target_offset + 2] = rgba[source_offset + 2];
      output.rgba[target_offset + 3] = rgba[source_offset + 3];
    }
  }

  return output;
}

bool validateRgbaFrameInput(
    JNIEnv* env,
    jobject rgbaBuffer,
    jint width,
    jint height,
    jint rowStride,
    jint rotationDegrees,
    jboolean mirrorHorizontally,
    RgbaFrameInput* output) {
  if (rgbaBuffer == nullptr) {
    throwJavaException(
        env,
        "java/lang/IllegalArgumentException",
        "RGBA buffer must not be null.");
    return false;
  }
  if (width <= 0 || height <= 0) {
    throwJavaException(
        env,
        "java/lang/IllegalArgumentException",
        "Frame dimensions must be positive.");
    return false;
  }

  const int normalized_rotation = normalizedRotation(rotationDegrees);
  if (!isSupportedRotation(normalized_rotation)) {
    throwJavaException(
        env,
        "java/lang/IllegalArgumentException",
        "Only 0/90/180/270 degree rotations are supported.");
    return false;
  }

  auto* rgba = static_cast<uint8_t*>(env->GetDirectBufferAddress(rgbaBuffer));
  const jlong capacity = env->GetDirectBufferCapacity(rgbaBuffer);
  const jint effective_row_stride =
      rowStride > 0 ? rowStride : width * static_cast<jint>(4);
  if (rgba == nullptr || capacity < static_cast<jlong>(effective_row_stride) *
              static_cast<jlong>(height)) {
    throwJavaException(
        env,
        "java/lang/IllegalArgumentException",
        "RGBA buffer must be a direct ByteBuffer large enough for the frame.");
    return false;
  }

  *output = RgbaFrameInput{
      .rgba = rgba,
      .width = width,
      .height = height,
      .row_stride = effective_row_stride,
      .rotation_degrees = normalized_rotation,
      .mirror_horizontally = mirrorHorizontally == JNI_TRUE,
  };
  return true;
}

bool runDetectionPipeline(
    JNIEnv* env,
    Yolo12Handle* handle,
    const RgbaFrameInput& frame_input,
    DetectionRunOutput* output) {
  std::lock_guard<std::mutex> lock(handle->mutex);
  const int input_height = static_cast<int>(handle->input_shape[2]);
  const int input_width = static_cast<int>(handle->input_shape[3]);

  const auto start = std::chrono::steady_clock::now();

  output->input_meta = preprocessRgbaFrame(
      frame_input.rgba,
      frame_input.width,
      frame_input.height,
      frame_input.row_stride,
      frame_input.rotation_degrees,
      frame_input.mirror_horizontally,
      input_width,
      input_height,
      &handle->input_buffer);

  std::vector<EValue> model_outputs;
  Error error = example::run_yolo_inference(
      handle->module.get(),
      handle->input_shape,
      handle->input_buffer,
      &model_outputs);
  if (error != Error::Ok) {
    throwJavaException(
        env,
        "java/lang/RuntimeException",
        "YOLO12 forward failed with ExecuTorch error code " +
            std::to_string(static_cast<uint32_t>(error)) + ".");
    return false;
  }

  error = example::postprocess_yolo_outputs(
      model_outputs,
      output->input_meta,
      0,
      handle->score_threshold,
      handle->nms_threshold,
      &output->detections);
  if (error != Error::Ok) {
    throwJavaException(
        env,
        "java/lang/RuntimeException",
        "YOLO12 inference failed with ExecuTorch error code " +
            std::to_string(static_cast<uint32_t>(error)) + ".");
    return false;
  }

  const auto elapsed = std::chrono::steady_clock::now() - start;
  output->latency_ms = static_cast<float>(
      std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count()) /
      1000.0f;
  return true;
}

bool ensureLoaded(JNIEnv* env, Yolo12Handle* handle) {
  if (handle == nullptr || handle->module == nullptr) {
    throwJavaException(
        env, "java/lang/IllegalStateException", "YOLO12 module was destroyed.");
    return false;
  }
  if (!handle->loaded) {
    throwJavaException(
        env, "java/lang/IllegalStateException", "YOLO12 module is not loaded.");
    return false;
  }
  return true;
}

} // namespace

extern "C" {

JNIEXPORT jlong JNICALL
Java_org_pytorch_executorch_extension_yolo_Yolo12Module_nativeCreate(
    JNIEnv* env,
    jobject,
    jstring modelPath,
    jfloat scoreThreshold,
    jfloat nmsThreshold) {
  const std::string model_path = jstringToString(env, modelPath);
  if (model_path.empty()) {
    throwJavaException(
        env,
        "java/lang/IllegalArgumentException",
        "Model path must not be empty.");
    return 0;
  }

  try {
    ensureRuntimeInitialized();
    auto handle = std::make_unique<Yolo12Handle>();
    handle->score_threshold = scoreThreshold;
    handle->nms_threshold = nmsThreshold;
    handle->module = std::make_unique<Module>(
        model_path.c_str(), Module::LoadMode::MmapUseMlockIgnoreErrors);
    return reinterpret_cast<jlong>(handle.release());
  } catch (const std::exception& exception) {
    throwJavaException(
        env,
        "java/lang/RuntimeException",
        std::string("Failed to create YOLO12 module: ") + exception.what());
    return 0;
  }
}

JNIEXPORT void JNICALL
Java_org_pytorch_executorch_extension_yolo_Yolo12Module_nativeDestroy(
    JNIEnv*,
    jobject,
    jlong nativeHandle) {
  if (nativeHandle != 0) {
    delete reinterpret_cast<Yolo12Handle*>(nativeHandle);
  }
}

JNIEXPORT jint JNICALL
Java_org_pytorch_executorch_extension_yolo_Yolo12Module_nativeLoad(
    JNIEnv* env,
    jobject,
    jlong nativeHandle) {
  auto* handle = reinterpret_cast<Yolo12Handle*>(nativeHandle);
  if (handle == nullptr || handle->module == nullptr) {
    throwJavaException(
        env, "java/lang/IllegalStateException", "YOLO12 module was destroyed.");
    return static_cast<jint>(Error::InvalidState);
  }

  std::lock_guard<std::mutex> lock(handle->mutex);
  Error error = handle->module->load();
  if (error != Error::Ok) {
    return static_cast<jint>(error);
  }

  error = handle->module->load_forward();
  if (error != Error::Ok) {
    return static_cast<jint>(error);
  }

  auto method_meta = handle->module->method_meta("forward");
  if (!method_meta.ok()) {
    return static_cast<jint>(method_meta.error());
  }
  auto input_meta = method_meta->input_tensor_meta(0);
  if (!input_meta.ok()) {
    return static_cast<jint>(input_meta.error());
  }

  const auto& input_sizes = input_meta->sizes();
  handle->input_shape.assign(input_sizes.begin(), input_sizes.end());
  if (handle->input_shape.size() != 4U) {
    throwJavaException(
        env,
        "java/lang/RuntimeException",
        "Expected YOLO12 input rank 4.");
    return static_cast<jint>(Error::InvalidArgument);
  }

  handle->loaded = true;
  return static_cast<jint>(Error::Ok);
}

JNIEXPORT jobject JNICALL
Java_org_pytorch_executorch_extension_yolo_Yolo12Module_nativeDetectFromRgba(
    JNIEnv* env,
    jobject,
    jlong nativeHandle,
    jobject rgbaBuffer,
    jint width,
    jint height,
    jint rowStride,
    jint rotationDegrees,
    jboolean mirrorHorizontally) {
  auto* handle = reinterpret_cast<Yolo12Handle*>(nativeHandle);
  if (!ensureLoaded(env, handle)) {
    return nullptr;
  }
  RgbaFrameInput frame_input;
  if (!validateRgbaFrameInput(
          env,
          rgbaBuffer,
          width,
          height,
          rowStride,
          rotationDegrees,
          mirrorHorizontally,
          &frame_input)) {
    return nullptr;
  }

  DetectionRunOutput detection_output;
  if (!runDetectionPipeline(env, handle, frame_input, &detection_output)) {
    return nullptr;
  }

  return makeResultObject(
      env,
      detection_output.detections,
      static_cast<jint>(detection_output.input_meta.original_width),
      static_cast<jint>(detection_output.input_meta.original_height),
      detection_output.latency_ms);
}

JNIEXPORT jobject JNICALL
Java_org_pytorch_executorch_extension_yolo_Yolo12Module_nativeDetectFrameFromRgba(
    JNIEnv* env,
    jobject,
    jlong nativeHandle,
    jobject rgbaBuffer,
    jint width,
    jint height,
    jint rowStride,
    jint rotationDegrees,
    jboolean mirrorHorizontally,
    jlong timestampNs) {
  auto* handle = reinterpret_cast<Yolo12Handle*>(nativeHandle);
  if (!ensureLoaded(env, handle)) {
    return nullptr;
  }

  RgbaFrameInput frame_input;
  if (!validateRgbaFrameInput(
          env,
          rgbaBuffer,
          width,
          height,
          rowStride,
          rotationDegrees,
          mirrorHorizontally,
          &frame_input)) {
    return nullptr;
  }

  DetectionRunOutput detection_output;
  if (!runDetectionPipeline(env, handle, frame_input, &detection_output)) {
    return nullptr;
  }

  const OrientedRgbaFrame oriented_frame = orientRgbaFrame(
      frame_input.rgba,
      frame_input.width,
      frame_input.height,
      frame_input.row_stride,
      frame_input.rotation_degrees,
      frame_input.mirror_horizontally);

  return makeFrameResultObject(
      env,
      oriented_frame.rgba,
      static_cast<jint>(oriented_frame.width),
      static_cast<jint>(oriented_frame.height),
      static_cast<jint>(oriented_frame.row_stride),
      timestampNs,
      detection_output.detections,
      detection_output.latency_ms);
}

} // extern "C"
