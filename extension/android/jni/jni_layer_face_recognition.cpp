#include <jni.h>

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/platform/runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

using ::executorch::extension::from_blob;
using ::executorch::extension::Module;
using ::executorch::runtime::EValue;
using ::executorch::runtime::Error;

namespace {

constexpr float kDetectMeanBgr[3] = {104.0f, 117.0f, 123.0f};
constexpr float kRecognizeMeanRgb[3] = {127.5f, 127.5f, 127.5f};
constexpr float kRecognizeStdRgb[3] = {128.0f, 128.0f, 128.0f};
constexpr std::array<std::array<int, 2>, 3> kMinSizes = {
    std::array<int, 2>{16, 32},
    std::array<int, 2>{64, 128},
    std::array<int, 2>{256, 512},
};
constexpr std::array<int, 3> kSteps = {8, 16, 32};
constexpr std::array<float, 2> kVariance = {0.1f, 0.2f};
constexpr std::array<std::array<float, 2>, 5> kArcFaceTemplate = {
    std::array<float, 2>{38.2946f, 51.6963f},
    std::array<float, 2>{73.5318f, 51.5014f},
    std::array<float, 2>{56.0252f, 71.7366f},
    std::array<float, 2>{41.5493f, 92.3655f},
    std::array<float, 2>{70.7299f, 92.2041f},
};

struct Pixel {
  float red{0.0f};
  float green{0.0f};
  float blue{0.0f};
};

struct ImageInfo {
  float scale_x{1.0f};
  float scale_y{1.0f};
  int original_width{0};
  int original_height{0};
};

struct FaceDetection {
  float score{0.0f};
  float left{0.0f};
  float top{0.0f};
  float right{0.0f};
  float bottom{0.0f};
  std::array<float, 10> landmarks{};
  bool has_landmarks{false};
  bool registration_eligible{false};
  std::string quality_message;

  float area() const {
    return std::max(0.0f, right - left) * std::max(0.0f, bottom - top);
  }
};

struct FaceRecognitionHandle {
  std::unique_ptr<Module> module;
  std::vector<executorch::aten::SizesType> detect_input_shape;
  std::vector<executorch::aten::SizesType> recognize_input_shape;
  std::vector<float> priors;
  std::vector<float> detect_input_buffer;
  std::vector<float> recognize_input_buffer;
  std::vector<uint8_t> upright_rgba_buffer;
  bool loaded{false};
  float score_threshold{0.6f};
  float nms_threshold{0.4f};
  std::mutex mutex;
};

struct JavaClassCache {
  jclass detection_class{nullptr};
  jmethodID detection_ctor{nullptr};
  jclass result_class{nullptr};
  jmethodID result_ctor{nullptr};
};

JavaClassCache g_java_cache;
std::once_flag g_java_cache_once;
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
    jclass detection_local =
        env->FindClass("org/pytorch/executorch/extension/face/FaceDetection");
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
        g_java_cache.detection_class,
        "<init>",
        "(FIIIIZZLjava/lang/String;)V");
    if (g_java_cache.detection_ctor == nullptr) {
      success = false;
      return;
    }

    jclass result_local =
        env->FindClass("org/pytorch/executorch/extension/face/FaceRecognitionResult");
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
        "([Lorg/pytorch/executorch/extension/face/FaceDetection;IIF[FZLjava/lang/String;)V");
    if (g_java_cache.result_ctor == nullptr) {
      success = false;
    }
  });
  return success && g_java_cache.detection_class != nullptr &&
      g_java_cache.result_class != nullptr &&
      g_java_cache.detection_ctor != nullptr && g_java_cache.result_ctor != nullptr;
}

jobject makeResultObject(
    JNIEnv* env,
    const std::vector<FaceDetection>& detections,
    int primary_index,
    jint image_width,
    jint image_height,
    jfloat latency_ms,
    const std::vector<float>& embedding) {
  if (!initializeJavaCache(env)) {
    throwJavaException(
        env,
        "java/lang/RuntimeException",
        "Failed to initialize face-recognition JNI class cache.");
    return nullptr;
  }

  jobjectArray detection_array = env->NewObjectArray(
      static_cast<jsize>(detections.size()), g_java_cache.detection_class, nullptr);
  if (detection_array == nullptr) {
    return nullptr;
  }

  for (jsize index = 0; index < static_cast<jsize>(detections.size()); ++index) {
    const auto& detection = detections[static_cast<size_t>(index)];
    jstring quality_message = env->NewStringUTF(detection.quality_message.c_str());
    jobject detection_object = env->NewObject(
        g_java_cache.detection_class,
        g_java_cache.detection_ctor,
        static_cast<jfloat>(detection.score),
        static_cast<jint>(std::lround(detection.left)),
        static_cast<jint>(std::lround(detection.top)),
        static_cast<jint>(std::lround(detection.right)),
        static_cast<jint>(std::lround(detection.bottom)),
        static_cast<jboolean>(index == primary_index),
        static_cast<jboolean>(detection.registration_eligible),
        quality_message);
    if (detection_object == nullptr) {
      return nullptr;
    }
    env->SetObjectArrayElement(detection_array, index, detection_object);
    env->DeleteLocalRef(detection_object);
    env->DeleteLocalRef(quality_message);
  }

  jfloatArray embedding_array = nullptr;
  if (!embedding.empty()) {
    embedding_array = env->NewFloatArray(static_cast<jsize>(embedding.size()));
    if (embedding_array == nullptr) {
      return nullptr;
    }
    env->SetFloatArrayRegion(
        embedding_array,
        0,
        static_cast<jsize>(embedding.size()),
        embedding.data());
  }

  jboolean primary_registration_eligible = JNI_FALSE;
  jstring primary_quality_message = nullptr;
  if (primary_index >= 0 &&
      static_cast<size_t>(primary_index) < detections.size()) {
    primary_registration_eligible =
        detections[static_cast<size_t>(primary_index)].registration_eligible
        ? JNI_TRUE
        : JNI_FALSE;
    primary_quality_message = env->NewStringUTF(
        detections[static_cast<size_t>(primary_index)].quality_message.c_str());
  }

  return env->NewObject(
      g_java_cache.result_class,
      g_java_cache.result_ctor,
      detection_array,
      image_width,
      image_height,
      latency_ms,
      embedding_array,
      primary_registration_eligible,
      primary_quality_message);
}

std::string evaluateFaceQuality(FaceDetection* detection) {
  if (detection == nullptr) {
    return "未检测到人脸";
  }
  const float width = detection->right - detection->left;
  const float height = detection->bottom - detection->top;
  if (width < 80.0f || height < 80.0f) {
    detection->registration_eligible = false;
    return "人脸过小，请靠近镜头后重试";
  }
  if (!detection->has_landmarks) {
    detection->registration_eligible = false;
    return "人脸关键点不足，请正对镜头重新采集";
  }

  const float left_eye_x = detection->landmarks[0];
  const float left_eye_y = detection->landmarks[1];
  const float right_eye_x = detection->landmarks[2];
  const float right_eye_y = detection->landmarks[3];
  const float nose_x = detection->landmarks[4];
  const float nose_y = detection->landmarks[5];
  const float mouth_left_x = detection->landmarks[6];
  const float mouth_left_y = detection->landmarks[7];
  const float mouth_right_x = detection->landmarks[8];
  const float mouth_right_y = detection->landmarks[9];
  const std::array<std::pair<float, float>, 5> keypoints = {{
      {left_eye_x, left_eye_y},
      {right_eye_x, right_eye_y},
      {nose_x, nose_y},
      {mouth_left_x, mouth_left_y},
      {mouth_right_x, mouth_right_y},
  }};

  for (const auto& keypoint : keypoints) {
    const float x = keypoint.first;
    const float y = keypoint.second;
    if (!std::isfinite(x) || !std::isfinite(y) || x <= 1.0f || y <= 1.0f) {
      detection->registration_eligible = false;
      return "人脸关键点不足，请重新采集";
    }
    if (x < detection->left - width * 0.15f ||
        x > detection->right + width * 0.15f ||
        y < detection->top - height * 0.15f ||
        y > detection->bottom + height * 0.15f) {
      detection->registration_eligible = false;
      return "人脸关键点异常，请重新采集";
    }
  }

  if (left_eye_x >= right_eye_x || mouth_left_x >= mouth_right_x) {
    detection->registration_eligible = false;
    return "人脸关键点异常，请重新采集";
  }

  const float eye_distance = std::max(1.0f, right_eye_x - left_eye_x);
  const float mouth_distance = std::max(1.0f, mouth_right_x - mouth_left_x);
  const float eye_center_y = (left_eye_y + right_eye_y) * 0.5f;
  const float mouth_center_x = (mouth_left_x + mouth_right_x) * 0.5f;
  const float mouth_center_y = (mouth_left_y + mouth_right_y) * 0.5f;
  const float face_width = std::max(1.0f, width);
  const float face_height = std::max(1.0f, height);

  if (eye_distance / face_width < 0.12f || mouth_distance / face_width < 0.10f) {
    detection->registration_eligible = false;
    return "人脸关键点不足，请靠近镜头后重试";
  }

  const float eye_line_tilt = std::abs(right_eye_y - left_eye_y) / eye_distance;
  if (eye_line_tilt > 0.32f) {
    detection->registration_eligible = false;
    return "人脸倾斜过大，请保持头部基本端正";
  }

  const float eye_center_x = (left_eye_x + right_eye_x) * 0.5f;
  const float nose_offset = std::abs(nose_x - eye_center_x) / eye_distance;
  if (nose_offset > 0.35f) {
    detection->registration_eligible = false;
    return "人脸左右偏转过大，请稍微正对镜头";
  }

  const float mouth_offset = std::abs(mouth_center_x - eye_center_x) / eye_distance;
  if (mouth_offset > 0.35f) {
    detection->registration_eligible = false;
    return "人脸关键点偏移过大，请重新采集";
  }

  if (!(eye_center_y < nose_y && nose_y < mouth_center_y)) {
    detection->registration_eligible = false;
    return "人脸关键点异常，请重新采集";
  }

  const float nose_vertical_ratio =
      (nose_y - eye_center_y) / face_height;
  if (nose_vertical_ratio < 0.10f || nose_vertical_ratio > 0.68f) {
    detection->registration_eligible = false;
    return "人脸上下角度过大，请调整后重试";
  }

  const float mouth_vertical_ratio =
      (mouth_center_y - nose_y) / face_height;
  if (mouth_vertical_ratio < 0.04f || mouth_vertical_ratio > 0.34f) {
    detection->registration_eligible = false;
    return "人脸上下角度过大，请调整后重试";
  }

  detection->registration_eligible = true;
  return "人脸关键点正常，可用于注册";
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

Pixel samplePixel(
    const std::vector<uint8_t>& rgba,
    int width,
    int height,
    float source_x,
    float source_y) {
  if (width <= 0 || height <= 0 || rgba.empty()) {
    return {};
  }

  const float clamped_x =
      std::clamp(source_x, 0.0f, static_cast<float>(width - 1));
  const float clamped_y =
      std::clamp(source_y, 0.0f, static_cast<float>(height - 1));
  const int x0 = static_cast<int>(std::floor(clamped_x));
  const int y0 = static_cast<int>(std::floor(clamped_y));
  const int x1 = std::min(x0 + 1, width - 1);
  const int y1 = std::min(y0 + 1, height - 1);
  const float wx = clamped_x - static_cast<float>(x0);
  const float wy = clamped_y - static_cast<float>(y0);

  auto load_pixel = [&](int x, int y) -> Pixel {
    const size_t offset =
        (static_cast<size_t>(y) * static_cast<size_t>(width) +
         static_cast<size_t>(x)) *
        4U;
    return Pixel{
        static_cast<float>(rgba[offset + 0]),
        static_cast<float>(rgba[offset + 1]),
        static_cast<float>(rgba[offset + 2]),
    };
  };

  const Pixel top_left = load_pixel(x0, y0);
  const Pixel top_right = load_pixel(x1, y0);
  const Pixel bottom_left = load_pixel(x0, y1);
  const Pixel bottom_right = load_pixel(x1, y1);

  auto bilinear = [&](float tl, float tr, float bl, float br) -> float {
    const float top = tl + (tr - tl) * wx;
    const float bottom = bl + (br - bl) * wx;
    return top + (bottom - top) * wy;
  };

  return Pixel{
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

void rotateRgbaFrame(
    const uint8_t* rgba,
    int source_width,
    int source_height,
    int row_stride,
    int rotation_degrees,
    bool mirror_horizontally,
    std::vector<uint8_t>* upright_rgba,
    int* upright_width,
    int* upright_height) {
  const int rotated_width =
      (rotation_degrees == 90 || rotation_degrees == 270) ? source_height
                                                          : source_width;
  const int rotated_height =
      (rotation_degrees == 90 || rotation_degrees == 270) ? source_width
                                                          : source_height;
  upright_rgba->assign(
      static_cast<size_t>(rotated_width) * static_cast<size_t>(rotated_height) * 4U,
      0);

  for (int y = 0; y < rotated_height; ++y) {
    for (int x = 0; x < rotated_width; ++x) {
      const int rotated_x = mirror_horizontally ? (rotated_width - 1 - x) : x;
      int source_x = 0;
      int source_y = 0;
      switch (rotation_degrees) {
        case 0:
          source_x = rotated_x;
          source_y = y;
          break;
        case 90:
          source_x = y;
          source_y = source_height - 1 - rotated_x;
          break;
        case 180:
          source_x = source_width - 1 - rotated_x;
          source_y = source_height - 1 - y;
          break;
        case 270:
          source_x = source_width - 1 - y;
          source_y = rotated_x;
          break;
      }

      const uint8_t* source_pixel =
          rgba + static_cast<size_t>(source_y) * static_cast<size_t>(row_stride) +
          static_cast<size_t>(source_x) * 4U;
      uint8_t* destination_pixel =
          upright_rgba->data() +
          (static_cast<size_t>(y) * static_cast<size_t>(rotated_width) +
           static_cast<size_t>(x)) *
              4U;
      destination_pixel[0] = source_pixel[0];
      destination_pixel[1] = source_pixel[1];
      destination_pixel[2] = source_pixel[2];
      destination_pixel[3] = source_pixel[3];
    }
  }

  *upright_width = rotated_width;
  *upright_height = rotated_height;
}

ImageInfo preprocessDetectorImage(
    const std::vector<uint8_t>& upright_rgba,
    int upright_width,
    int upright_height,
    int input_width,
    int input_height,
    std::vector<float>* output) {
  output->assign(
      static_cast<size_t>(3) * static_cast<size_t>(input_width) *
          static_cast<size_t>(input_height),
      0.0f);

  const float scale_x =
      static_cast<float>(upright_width) / static_cast<float>(input_width);
  const float scale_y =
      static_cast<float>(upright_height) / static_cast<float>(input_height);
  const size_t channel_stride =
      static_cast<size_t>(input_width) * static_cast<size_t>(input_height);

  for (int y = 0; y < input_height; ++y) {
    const float source_y = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;
    for (int x = 0; x < input_width; ++x) {
      const float source_x = (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
      const Pixel pixel =
          samplePixel(upright_rgba, upright_width, upright_height, source_x, source_y);
      const size_t index =
          static_cast<size_t>(y) * static_cast<size_t>(input_width) +
          static_cast<size_t>(x);
      (*output)[index] = pixel.blue - kDetectMeanBgr[0];
      (*output)[channel_stride + index] = pixel.green - kDetectMeanBgr[1];
      (*output)[channel_stride * 2U + index] = pixel.red - kDetectMeanBgr[2];
    }
  }

  ImageInfo info;
  info.scale_x = scale_x;
  info.scale_y = scale_y;
  info.original_width = upright_width;
  info.original_height = upright_height;
  return info;
}

void generatePriors(
    int input_height,
    int input_width,
    std::vector<float>* priors) {
  priors->clear();
  for (size_t level = 0; level < kSteps.size(); ++level) {
    const int step = kSteps[level];
    const int feature_height = (input_height + step - 1) / step;
    const int feature_width = (input_width + step - 1) / step;
    for (int y = 0; y < feature_height; ++y) {
      for (int x = 0; x < feature_width; ++x) {
        for (int min_size : kMinSizes[level]) {
          priors->push_back((static_cast<float>(x) + 0.5f) * step / input_width);
          priors->push_back((static_cast<float>(y) + 0.5f) * step / input_height);
          priors->push_back(static_cast<float>(min_size) / input_width);
          priors->push_back(static_cast<float>(min_size) / input_height);
        }
      }
    }
  }
}

float computeIou(const FaceDetection& a, const FaceDetection& b) {
  const float left = std::max(a.left, b.left);
  const float top = std::max(a.top, b.top);
  const float right = std::min(a.right, b.right);
  const float bottom = std::min(a.bottom, b.bottom);
  const float width = std::max(0.0f, right - left);
  const float height = std::max(0.0f, bottom - top);
  const float intersection = width * height;
  const float union_area = a.area() + b.area() - intersection;
  if (union_area <= 1e-6f) {
    return 0.0f;
  }
  return intersection / union_area;
}

std::vector<int> nmsDetections(
    const std::vector<FaceDetection>& detections,
    float threshold) {
  std::vector<int> order(detections.size());
  for (size_t i = 0; i < detections.size(); ++i) {
    order[i] = static_cast<int>(i);
  }
  std::sort(order.begin(), order.end(), [&](int lhs, int rhs) {
    return detections[static_cast<size_t>(lhs)].score >
        detections[static_cast<size_t>(rhs)].score;
  });

  std::vector<int> keep;
  std::vector<bool> suppressed(detections.size(), false);
  for (size_t i = 0; i < order.size(); ++i) {
    const int candidate = order[i];
    if (suppressed[static_cast<size_t>(candidate)]) {
      continue;
    }
    keep.push_back(candidate);
    for (size_t j = i + 1; j < order.size(); ++j) {
      const int other = order[j];
      if (suppressed[static_cast<size_t>(other)]) {
        continue;
      }
      if (computeIou(
              detections[static_cast<size_t>(candidate)],
              detections[static_cast<size_t>(other)]) > threshold) {
        suppressed[static_cast<size_t>(other)] = true;
      }
    }
  }
  return keep;
}

bool decodeDetectorOutputs(
    const std::vector<EValue>& outputs,
    const std::vector<float>& priors,
    const ImageInfo& image_info,
    float score_threshold,
    float nms_threshold,
    std::vector<FaceDetection>* detections) {
  if (outputs.size() < 3 || !outputs[0].isTensor() || !outputs[1].isTensor() ||
      !outputs[2].isTensor()) {
    return false;
  }

  const auto loc_tensor = outputs[0].toTensor();
  const auto conf_tensor = outputs[1].toTensor();
  const auto landm_tensor = outputs[2].toTensor();
  if (loc_tensor.scalar_type() != executorch::aten::ScalarType::Float ||
      conf_tensor.scalar_type() != executorch::aten::ScalarType::Float ||
      landm_tensor.scalar_type() != executorch::aten::ScalarType::Float) {
    return false;
  }

  const size_t loc_numel = static_cast<size_t>(loc_tensor.numel());
  const size_t conf_numel = static_cast<size_t>(conf_tensor.numel());
  const size_t landm_numel = static_cast<size_t>(landm_tensor.numel());
  if (loc_numel % 4U != 0U || landm_numel % 10U != 0U) {
    return false;
  }

  const size_t num_priors = loc_numel / 4U;
  if (priors.size() != num_priors * 4U) {
    return false;
  }

  const float* loc_data = loc_tensor.const_data_ptr<float>();
  const float* conf_data = conf_tensor.const_data_ptr<float>();
  const float* landm_data = landm_tensor.const_data_ptr<float>();
  if (loc_data == nullptr || conf_data == nullptr || landm_data == nullptr) {
    return false;
  }

  const size_t conf_stride =
      conf_numel == num_priors ? 1U
                               : (conf_numel % num_priors == 0U ? conf_numel / num_priors
                                                                : 0U);
  if (conf_stride == 0U) {
    return false;
  }

  std::vector<FaceDetection> filtered;
  filtered.reserve(num_priors);
  const float input_width =
      static_cast<float>(image_info.original_width) / image_info.scale_x;
  const float input_height =
      static_cast<float>(image_info.original_height) / image_info.scale_y;

  for (size_t index = 0; index < num_priors; ++index) {
    const float score = conf_stride > 1U ? conf_data[index * conf_stride + 1U]
                                         : conf_data[index];
    if (score < score_threshold) {
      continue;
    }

    const float prior_cx = priors[index * 4U + 0U];
    const float prior_cy = priors[index * 4U + 1U];
    const float prior_w = priors[index * 4U + 2U];
    const float prior_h = priors[index * 4U + 3U];

    const float center_x =
        prior_cx + loc_data[index * 4U + 0U] * kVariance[0] * prior_w;
    const float center_y =
        prior_cy + loc_data[index * 4U + 1U] * kVariance[0] * prior_h;
    const float width =
        prior_w * std::exp(loc_data[index * 4U + 2U] * kVariance[1]);
    const float height =
        prior_h * std::exp(loc_data[index * 4U + 3U] * kVariance[1]);

    FaceDetection detection;
    detection.score = score;
    detection.left =
        (center_x - width / 2.0f) * input_width * image_info.scale_x;
    detection.top =
        (center_y - height / 2.0f) * input_height * image_info.scale_y;
    detection.right =
        (center_x + width / 2.0f) * input_width * image_info.scale_x;
    detection.bottom =
        (center_y + height / 2.0f) * input_height * image_info.scale_y;

    for (size_t point = 0; point < 5U; ++point) {
      const float landmark_x =
          prior_cx + landm_data[index * 10U + point * 2U + 0U] * kVariance[0] *
              prior_w;
      const float landmark_y =
          prior_cy + landm_data[index * 10U + point * 2U + 1U] * kVariance[0] *
              prior_h;
      detection.landmarks[point * 2U + 0U] =
          landmark_x * input_width * image_info.scale_x;
      detection.landmarks[point * 2U + 1U] =
          landmark_y * input_height * image_info.scale_y;
    }
    detection.has_landmarks = true;

    detection.left = std::clamp(
        detection.left, 0.0f, static_cast<float>(image_info.original_width));
    detection.right = std::clamp(
        detection.right, 0.0f, static_cast<float>(image_info.original_width));
    detection.top = std::clamp(
        detection.top, 0.0f, static_cast<float>(image_info.original_height));
    detection.bottom = std::clamp(
        detection.bottom, 0.0f, static_cast<float>(image_info.original_height));
    filtered.push_back(detection);
  }

  const std::vector<int> kept_indices = nmsDetections(filtered, nms_threshold);
  detections->clear();
  detections->reserve(kept_indices.size());
  for (int kept_index : kept_indices) {
    detections->push_back(filtered[static_cast<size_t>(kept_index)]);
  }
  return true;
}

Error executeMethod(
    Module* module,
    const std::string& method_name,
    const std::vector<executorch::aten::SizesType>& input_shape,
    const std::vector<float>& input_buffer,
    std::vector<EValue>* outputs) {
  auto input_tensor = from_blob(
      (void*)input_buffer.data(),
      std::vector<executorch::aten::SizesType>(input_shape.begin(), input_shape.end()),
      executorch::aten::ScalarType::Float);
  const std::vector<EValue> inputs{input_tensor};
  auto outputs_res = module->execute(method_name, inputs);
  if (outputs_res.error() != Error::Ok) {
    return outputs_res.error();
  }
  *outputs = outputs_res.get();
  return Error::Ok;
}

bool estimateSimilarityTransform(
    const FaceDetection& detection,
    int target_width,
    int target_height,
    float* scale,
    float* cos_theta,
    float* sin_theta,
    float* translation_x,
    float* translation_y) {
  if (!detection.has_landmarks) {
    return false;
  }

  float source_mean_x = 0.0f;
  float source_mean_y = 0.0f;
  float destination_mean_x = 0.0f;
  float destination_mean_y = 0.0f;
  for (size_t index = 0; index < 5U; ++index) {
    source_mean_x += detection.landmarks[index * 2U + 0U];
    source_mean_y += detection.landmarks[index * 2U + 1U];
    destination_mean_x +=
        kArcFaceTemplate[index][0] * target_width / 112.0f;
    destination_mean_y +=
        kArcFaceTemplate[index][1] * target_height / 112.0f;
  }
  source_mean_x /= 5.0f;
  source_mean_y /= 5.0f;
  destination_mean_x /= 5.0f;
  destination_mean_y /= 5.0f;

  double a = 0.0;
  double b = 0.0;
  double source_variance = 0.0;
  for (size_t index = 0; index < 5U; ++index) {
    const double src_x =
        static_cast<double>(detection.landmarks[index * 2U + 0U] - source_mean_x);
    const double src_y =
        static_cast<double>(detection.landmarks[index * 2U + 1U] - source_mean_y);
    const double dst_x =
        static_cast<double>(
            kArcFaceTemplate[index][0] * target_width / 112.0f -
            destination_mean_x);
    const double dst_y =
        static_cast<double>(
            kArcFaceTemplate[index][1] * target_height / 112.0f -
            destination_mean_y);
    a += src_x * dst_x + src_y * dst_y;
    b += src_x * dst_y - src_y * dst_x;
    source_variance += src_x * src_x + src_y * src_y;
  }

  const double norm = std::sqrt(a * a + b * b);
  if (norm <= 1e-6 || source_variance <= 1e-6) {
    return false;
  }

  *scale = static_cast<float>(norm / source_variance);
  *cos_theta = static_cast<float>(a / norm);
  *sin_theta = static_cast<float>(b / norm);
  *translation_x = destination_mean_x -
      *scale * (*cos_theta * source_mean_x - *sin_theta * source_mean_y);
  *translation_y = destination_mean_y -
      *scale * (*sin_theta * source_mean_x + *cos_theta * source_mean_y);
  return true;
}

void preprocessRecognizerImage(
    const std::vector<uint8_t>& upright_rgba,
    int upright_width,
    int upright_height,
    const FaceDetection& detection,
    int target_width,
    int target_height,
    std::vector<float>* output) {
  output->assign(
      static_cast<size_t>(3) * static_cast<size_t>(target_width) *
          static_cast<size_t>(target_height),
      0.0f);
  const size_t channel_stride =
      static_cast<size_t>(target_width) * static_cast<size_t>(target_height);

  float scale = 0.0f;
  float cos_theta = 1.0f;
  float sin_theta = 0.0f;
  float translation_x = 0.0f;
  float translation_y = 0.0f;
  const bool has_transform = estimateSimilarityTransform(
      detection,
      target_width,
      target_height,
      &scale,
      &cos_theta,
      &sin_theta,
      &translation_x,
      &translation_y);

  float crop_left = detection.left;
  float crop_top = detection.top;
  float crop_size = std::max(
      std::max(1.0f, detection.right - detection.left),
      std::max(1.0f, detection.bottom - detection.top));
  crop_size *= 1.2f;
  if (!has_transform) {
    const float center_x = (detection.left + detection.right) * 0.5f;
    const float center_y = (detection.top + detection.bottom) * 0.5f;
    crop_left = center_x - crop_size * 0.5f;
    crop_top = center_y - crop_size * 0.5f;
  }

  for (int y = 0; y < target_height; ++y) {
    for (int x = 0; x < target_width; ++x) {
      float source_x = 0.0f;
      float source_y = 0.0f;
      if (has_transform && scale > 1e-6f) {
        const float dst_x = static_cast<float>(x);
        const float dst_y = static_cast<float>(y);
        const float delta_x = dst_x - translation_x;
        const float delta_y = dst_y - translation_y;
        source_x = (cos_theta * delta_x + sin_theta * delta_y) / scale;
        source_y = (-sin_theta * delta_x + cos_theta * delta_y) / scale;
      } else {
        source_x =
            crop_left + (static_cast<float>(x) + 0.5f) * crop_size / target_width;
        source_y =
            crop_top + (static_cast<float>(y) + 0.5f) * crop_size / target_height;
      }

      const Pixel pixel =
          samplePixel(upright_rgba, upright_width, upright_height, source_x, source_y);
      const size_t index =
          static_cast<size_t>(y) * static_cast<size_t>(target_width) +
          static_cast<size_t>(x);
      (*output)[index] =
          (pixel.red - kRecognizeMeanRgb[0]) / kRecognizeStdRgb[0];
      (*output)[channel_stride + index] =
          (pixel.green - kRecognizeMeanRgb[1]) / kRecognizeStdRgb[1];
      (*output)[channel_stride * 2U + index] =
          (pixel.blue - kRecognizeMeanRgb[2]) / kRecognizeStdRgb[2];
    }
  }
}

bool decodeEmbedding(
    const std::vector<EValue>& outputs,
    std::vector<float>* embedding) {
  if (outputs.empty() || !outputs[0].isTensor()) {
    return false;
  }
  const auto tensor = outputs[0].toTensor();
  if (tensor.scalar_type() != executorch::aten::ScalarType::Float) {
    return false;
  }
  const float* data = tensor.const_data_ptr<float>();
  if (data == nullptr) {
    return false;
  }
  const size_t numel = static_cast<size_t>(tensor.numel());
  if (numel == 0U) {
    return false;
  }

  embedding->assign(data, data + numel);
  double norm = 0.0;
  for (float value : *embedding) {
    norm += static_cast<double>(value) * static_cast<double>(value);
  }
  norm = std::sqrt(norm);
  if (norm > 1e-12) {
    for (float& value : *embedding) {
      value = static_cast<float>(value / norm);
    }
  }
  return true;
}

bool ensureLoaded(JNIEnv* env, FaceRecognitionHandle* handle) {
  if (handle == nullptr || handle->module == nullptr) {
    throwJavaException(
        env,
        "java/lang/IllegalStateException",
        "FaceRecognition module was destroyed.");
    return false;
  }
  if (!handle->loaded) {
    throwJavaException(
        env,
        "java/lang/IllegalStateException",
        "FaceRecognition module is not loaded.");
    return false;
  }
  return true;
}

} // namespace

extern "C" {

JNIEXPORT jlong JNICALL
Java_org_pytorch_executorch_extension_face_FaceRecognitionModule_nativeCreate(
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
    auto handle = std::make_unique<FaceRecognitionHandle>();
    handle->score_threshold = scoreThreshold;
    handle->nms_threshold = nmsThreshold;
    handle->module = std::make_unique<Module>(
        model_path.c_str(), Module::LoadMode::MmapUseMlockIgnoreErrors);
    return reinterpret_cast<jlong>(handle.release());
  } catch (const std::exception& exception) {
    throwJavaException(
        env,
        "java/lang/RuntimeException",
        std::string("Failed to create FaceRecognition module: ") +
            exception.what());
    return 0;
  }
}

JNIEXPORT void JNICALL
Java_org_pytorch_executorch_extension_face_FaceRecognitionModule_nativeDestroy(
    JNIEnv*,
    jobject,
    jlong nativeHandle) {
  if (nativeHandle != 0L) {
    delete reinterpret_cast<FaceRecognitionHandle*>(nativeHandle);
  }
}

JNIEXPORT jint JNICALL
Java_org_pytorch_executorch_extension_face_FaceRecognitionModule_nativeLoad(
    JNIEnv* env,
    jobject,
    jlong nativeHandle) {
  auto* handle = reinterpret_cast<FaceRecognitionHandle*>(nativeHandle);
  if (handle == nullptr || handle->module == nullptr) {
    throwJavaException(
        env,
        "java/lang/IllegalStateException",
        "FaceRecognition module was destroyed.");
    return static_cast<jint>(Error::InvalidState);
  }

  std::lock_guard<std::mutex> lock(handle->mutex);
  Error error = handle->module->load();
  if (error != Error::Ok) {
    return static_cast<jint>(error);
  }

  error = handle->module->load_method("detect");
  if (error != Error::Ok) {
    return static_cast<jint>(error);
  }
  error = handle->module->load_method("recognize");
  if (error != Error::Ok) {
    return static_cast<jint>(error);
  }

  auto detect_meta = handle->module->method_meta("detect");
  if (!detect_meta.ok()) {
    return static_cast<jint>(detect_meta.error());
  }
  auto detect_input_meta = detect_meta->input_tensor_meta(0);
  if (!detect_input_meta.ok()) {
    return static_cast<jint>(detect_input_meta.error());
  }
  const auto& detect_sizes = detect_input_meta->sizes();
  handle->detect_input_shape.assign(detect_sizes.begin(), detect_sizes.end());

  auto recognize_meta = handle->module->method_meta("recognize");
  if (!recognize_meta.ok()) {
    return static_cast<jint>(recognize_meta.error());
  }
  auto recognize_input_meta = recognize_meta->input_tensor_meta(0);
  if (!recognize_input_meta.ok()) {
    return static_cast<jint>(recognize_input_meta.error());
  }
  const auto& recognize_sizes = recognize_input_meta->sizes();
  handle->recognize_input_shape.assign(
      recognize_sizes.begin(), recognize_sizes.end());

  if (handle->detect_input_shape.size() != 4U ||
      handle->recognize_input_shape.size() != 4U) {
    throwJavaException(
        env,
        "java/lang/RuntimeException",
        "Expected detect and recognize methods to use rank-4 inputs.");
    return static_cast<jint>(Error::InvalidArgument);
  }

  generatePriors(
      static_cast<int>(handle->detect_input_shape[2]),
      static_cast<int>(handle->detect_input_shape[3]),
      &handle->priors);
  handle->loaded = true;
  return static_cast<jint>(Error::Ok);
}

JNIEXPORT jobject JNICALL
Java_org_pytorch_executorch_extension_face_FaceRecognitionModule_nativeAnalyzeFromRgba(
    JNIEnv* env,
    jobject,
    jlong nativeHandle,
    jobject rgbaBuffer,
    jint width,
    jint height,
    jint rowStride,
    jint rotationDegrees,
    jboolean mirrorHorizontally) {
  auto* handle = reinterpret_cast<FaceRecognitionHandle*>(nativeHandle);
  if (!ensureLoaded(env, handle)) {
    return nullptr;
  }
  if (rgbaBuffer == nullptr) {
    throwJavaException(
        env,
        "java/lang/IllegalArgumentException",
        "RGBA buffer must not be null.");
    return nullptr;
  }
  if (width <= 0 || height <= 0) {
    throwJavaException(
        env,
        "java/lang/IllegalArgumentException",
        "Frame dimensions must be positive.");
    return nullptr;
  }

  const int normalized_rotation = normalizedRotation(rotationDegrees);
  if (!isSupportedRotation(normalized_rotation)) {
    throwJavaException(
        env,
        "java/lang/IllegalArgumentException",
        "Only 0/90/180/270 degree rotations are supported.");
    return nullptr;
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
    return nullptr;
  }

  std::lock_guard<std::mutex> lock(handle->mutex);
  const auto start = std::chrono::steady_clock::now();

  int upright_width = 0;
  int upright_height = 0;
  rotateRgbaFrame(
      rgba,
      width,
      height,
      effective_row_stride,
      normalized_rotation,
      mirrorHorizontally == JNI_TRUE,
      &handle->upright_rgba_buffer,
      &upright_width,
      &upright_height);

  const ImageInfo image_info = preprocessDetectorImage(
      handle->upright_rgba_buffer,
      upright_width,
      upright_height,
      static_cast<int>(handle->detect_input_shape[3]),
      static_cast<int>(handle->detect_input_shape[2]),
      &handle->detect_input_buffer);

  std::vector<EValue> detect_outputs;
  Error error = executeMethod(
      handle->module.get(),
      "detect",
      handle->detect_input_shape,
      handle->detect_input_buffer,
      &detect_outputs);
  if (error != Error::Ok) {
    throwJavaException(
        env,
        "java/lang/RuntimeException",
        "Face detector inference failed with ExecuTorch error code " +
            std::to_string(static_cast<uint32_t>(error)) + ".");
    return nullptr;
  }

  std::vector<FaceDetection> detections;
  if (!decodeDetectorOutputs(
          detect_outputs,
          handle->priors,
          image_info,
          handle->score_threshold,
          handle->nms_threshold,
          &detections)) {
    throwJavaException(
        env,
        "java/lang/RuntimeException",
        "Failed to decode detector outputs.");
    return nullptr;
  }

  int primary_index = -1;
  float largest_area = -1.0f;
  for (size_t index = 0; index < detections.size(); ++index) {
    const float area = detections[index].area();
    if (area > largest_area) {
      largest_area = area;
      primary_index = static_cast<int>(index);
    }
  }

  if (primary_index >= 0) {
    detections[static_cast<size_t>(primary_index)].quality_message =
        evaluateFaceQuality(&detections[static_cast<size_t>(primary_index)]);
  }

  std::vector<float> embedding;
  if (primary_index >= 0) {
    preprocessRecognizerImage(
        handle->upright_rgba_buffer,
        upright_width,
        upright_height,
        detections[static_cast<size_t>(primary_index)],
        static_cast<int>(handle->recognize_input_shape[3]),
        static_cast<int>(handle->recognize_input_shape[2]),
        &handle->recognize_input_buffer);

    std::vector<EValue> recognize_outputs;
    error = executeMethod(
        handle->module.get(),
        "recognize",
        handle->recognize_input_shape,
        handle->recognize_input_buffer,
        &recognize_outputs);
    if (error != Error::Ok) {
      throwJavaException(
          env,
          "java/lang/RuntimeException",
          "Face recognizer inference failed with ExecuTorch error code " +
              std::to_string(static_cast<uint32_t>(error)) + ".");
      return nullptr;
    }
    if (!decodeEmbedding(recognize_outputs, &embedding)) {
      throwJavaException(
          env,
          "java/lang/RuntimeException",
          "Failed to decode face embedding.");
      return nullptr;
    }
  }

  const auto elapsed = std::chrono::steady_clock::now() - start;
  const float latency_ms =
      static_cast<float>(
          std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count()) /
      1000.0f;

  return makeResultObject(
      env,
      detections,
      primary_index,
      upright_width,
      upright_height,
      latency_ms,
      embedding);
}

} // extern "C"
