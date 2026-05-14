/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef EXECUTORCH_EXAMPLES_QUALCOMM_OSS_SCRIPTS_YOLO12_RUNNER_INFERENCE_H
#define EXECUTORCH_EXAMPLES_QUALCOMM_OSS_SCRIPTS_YOLO12_RUNNER_INFERENCE_H

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/platform/log.h>

#include <algorithm>
#include <cmath>
#include <cinttypes>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <ios>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace example {

using executorch::aten::ScalarType;
using executorch::extension::from_blob;
using executorch::extension::Module;
using executorch::runtime::EValue;
using executorch::runtime::Error;

struct InputMeta {
  int32_t original_width{0};
  int32_t original_height{0};
  float scale{1.0f};
  int32_t pad_x{0};
  int32_t pad_y{0};
};

struct Detection {
  int32_t class_id{0};
  float confidence{0.0f};
  int32_t left{0};
  int32_t top{0};
  int32_t right{0};
  int32_t bottom{0};
};

inline std::vector<std::string> split_line(const std::string& line) {
  std::vector<std::string> tokens;
  std::istringstream stream(line);
  std::string token;
  while (stream >> token) {
    tokens.push_back(token);
  }
  return tokens;
}

inline bool read_binary_file(
    const std::string& file_path,
    std::vector<float>* output_data) {
  std::ifstream input(file_path, std::ios::binary | std::ios::ate);
  if (!input.is_open()) {
    ET_LOG(Error, "Unable to open input file: %s", file_path.c_str());
    return false;
  }

  const std::streamsize size = input.tellg();
  if (size < 0 || size % static_cast<std::streamsize>(sizeof(float)) != 0) {
    ET_LOG(Error, "Unexpected file size for %s", file_path.c_str());
    return false;
  }

  input.seekg(0, std::ios::beg);
  output_data->resize(size / static_cast<std::streamsize>(sizeof(float)));
  if (!input.read(
          reinterpret_cast<char*>(output_data->data()),
          static_cast<std::streamsize>(output_data->size() * sizeof(float)))) {
    ET_LOG(Error, "Failed to read input file: %s", file_path.c_str());
    return false;
  }
  return true;
}

inline std::vector<std::string> parse_input_list_file(
    const std::string& input_list_path) {
  std::vector<std::string> input_paths;
  std::ifstream input_list(input_list_path);
  if (!input_list.is_open()) {
    ET_LOG(Error, "Unable to open input list: %s", input_list_path.c_str());
    return input_paths;
  }

  std::string line;
  while (std::getline(input_list, line)) {
    if (line.empty()) {
      continue;
    }
    const auto tokens = split_line(line);
    if (tokens.empty()) {
      continue;
    }
    input_paths.push_back(tokens.front());
  }
  return input_paths;
}

inline std::vector<std::string> load_labels(const std::string& labels_path) {
  std::vector<std::string> labels;
  std::ifstream input(labels_path);
  if (!input.is_open()) {
    ET_LOG(
        Info,
        "Unable to open labels file: %s. Falling back to class ids.",
        labels_path.c_str());
    return labels;
  }

  std::string line;
  while (std::getline(input, line)) {
    if (!line.empty()) {
      labels.push_back(line);
    }
  }
  return labels;
}

inline std::vector<InputMeta> load_input_meta(const std::string& meta_path) {
  std::vector<InputMeta> metas;
  std::ifstream input(meta_path);
  if (!input.is_open()) {
    ET_LOG(Error, "Unable to open input meta file: %s", meta_path.c_str());
    return metas;
  }

  std::string line;
  while (std::getline(input, line)) {
    if (line.empty()) {
      continue;
    }
    std::istringstream stream(line);
    InputMeta meta;
    stream >> meta.original_width >> meta.original_height >> meta.scale >>
        meta.pad_x >> meta.pad_y;
    if (stream.fail()) {
      ET_LOG(Error, "Malformed input meta line: %s", line.c_str());
      metas.clear();
      return metas;
    }
    metas.push_back(meta);
  }
  return metas;
}

inline float intersection_over_union(
    const Detection& lhs,
    const Detection& rhs) {
  const int32_t inter_left = std::max(lhs.left, rhs.left);
  const int32_t inter_top = std::max(lhs.top, rhs.top);
  const int32_t inter_right = std::min(lhs.right, rhs.right);
  const int32_t inter_bottom = std::min(lhs.bottom, rhs.bottom);

  const int32_t inter_width = std::max(0, inter_right - inter_left);
  const int32_t inter_height = std::max(0, inter_bottom - inter_top);
  const float inter_area =
      static_cast<float>(inter_width) * static_cast<float>(inter_height);

  const float lhs_area = static_cast<float>(std::max(0, lhs.right - lhs.left)) *
      static_cast<float>(std::max(0, lhs.bottom - lhs.top));
  const float rhs_area = static_cast<float>(std::max(0, rhs.right - rhs.left)) *
      static_cast<float>(std::max(0, rhs.bottom - rhs.top));
  const float denom = lhs_area + rhs_area - inter_area;
  if (denom <= std::numeric_limits<float>::epsilon()) {
    return 0.0f;
  }
  return inter_area / denom;
}

inline std::vector<Detection> apply_nms(
    const std::vector<Detection>& detections,
    float nms_threshold) {
  std::vector<int32_t> order(detections.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int32_t lhs, int32_t rhs) {
    return detections[lhs].confidence > detections[rhs].confidence;
  });

  std::vector<Detection> filtered;
  std::vector<bool> suppressed(detections.size(), false);
  for (size_t i = 0; i < order.size(); ++i) {
    const int32_t idx = order[i];
    if (suppressed[idx]) {
      continue;
    }
    filtered.push_back(detections[idx]);
    for (size_t j = i + 1; j < order.size(); ++j) {
      const int32_t next_idx = order[j];
      if (suppressed[next_idx]) {
        continue;
      }
      if (detections[idx].class_id != detections[next_idx].class_id) {
        continue;
      }
      if (intersection_over_union(detections[idx], detections[next_idx]) >
          nms_threshold) {
        suppressed[next_idx] = true;
      }
    }
  }
  return filtered;
}

inline std::vector<Detection> decode_detections(
    const executorch::aten::Tensor& output_tensor,
    const InputMeta& input_meta,
    size_t num_classes,
    float score_threshold,
    float nms_threshold) {
  ET_CHECK_MSG(
      output_tensor.dim() == 3,
      "Expected YOLO output of rank 3, got rank %zd",
      output_tensor.dim());

  const auto* sizes = output_tensor.sizes().data();
  const int64_t dim1 = sizes[1];
  const int64_t dim2 = sizes[2];
  const bool channels_first = dim1 <= dim2;
  const int64_t attributes = channels_first ? dim1 : dim2;
  const int64_t candidates = channels_first ? dim2 : dim1;

  ET_CHECK_MSG(
      attributes >= 5,
      "Unexpected YOLO output shape [%lld, %lld, %lld]",
      static_cast<long long>(sizes[0]),
      static_cast<long long>(dim1),
      static_cast<long long>(dim2));

  const float* data = static_cast<const float*>(output_tensor.const_data_ptr());
  auto value_at = [&](int64_t attr, int64_t candidate) -> float {
    if (channels_first) {
      return data[attr * candidates + candidate];
    }
    return data[candidate * attributes + attr];
  };

  const size_t class_count =
      num_classes > 0 ? num_classes : static_cast<size_t>(attributes - 4);

  std::vector<Detection> detections;
  detections.reserve(candidates);
  for (int64_t candidate = 0; candidate < candidates; ++candidate) {
    float best_score = 0.0f;
    int32_t best_class = -1;
    for (size_t class_index = 0; class_index < class_count; ++class_index) {
      const float score = value_at(4 + static_cast<int64_t>(class_index), candidate);
      if (score > best_score) {
        best_score = score;
        best_class = static_cast<int32_t>(class_index);
      }
    }
    if (best_class < 0 || best_score < score_threshold) {
      continue;
    }

    const float center_x = value_at(0, candidate);
    const float center_y = value_at(1, candidate);
    const float width = value_at(2, candidate);
    const float height = value_at(3, candidate);

    const float left_f =
        (center_x - 0.5f * width - static_cast<float>(input_meta.pad_x)) /
        input_meta.scale;
    const float top_f =
        (center_y - 0.5f * height - static_cast<float>(input_meta.pad_y)) /
        input_meta.scale;
    const float right_f =
        (center_x + 0.5f * width - static_cast<float>(input_meta.pad_x)) /
        input_meta.scale;
    const float bottom_f =
        (center_y + 0.5f * height - static_cast<float>(input_meta.pad_y)) /
        input_meta.scale;

    Detection detection;
    detection.class_id = best_class;
    detection.confidence = best_score;
    detection.left = std::max(0, static_cast<int32_t>(std::floor(left_f)));
    detection.top = std::max(0, static_cast<int32_t>(std::floor(top_f)));
    detection.right = std::min(
        input_meta.original_width,
        static_cast<int32_t>(std::ceil(right_f)));
    detection.bottom = std::min(
        input_meta.original_height,
        static_cast<int32_t>(std::ceil(bottom_f)));

    if (detection.right > detection.left && detection.bottom > detection.top) {
      detections.push_back(detection);
    }
  }

  return apply_nms(detections, nms_threshold);
}

inline Error run_yolo_inference(
    Module* module,
    const std::vector<executorch::aten::SizesType>& input_shape,
    const std::vector<float>& input_buffer,
    std::vector<EValue>* model_outputs) {
  ET_CHECK_MSG(module != nullptr, "YOLO module must not be null");
  ET_CHECK_MSG(model_outputs != nullptr, "YOLO model outputs must not be null");

  const auto expected_elements = std::accumulate(
      input_shape.begin(),
      input_shape.end(),
      static_cast<int64_t>(1),
      std::multiplies<int64_t>());
  ET_CHECK_MSG(
      expected_elements == static_cast<int64_t>(input_buffer.size()),
      "Input size mismatch. Expected %lld float elements, got %zu",
      static_cast<long long>(expected_elements),
      input_buffer.size());

  std::vector<executorch::aten::SizesType> input_sizes(
      input_shape.begin(), input_shape.end());
  const auto input_tensor =
      from_blob((void*)input_buffer.data(), std::move(input_sizes), ScalarType::Float);
  auto result = module->forward(input_tensor);
  if (!result.ok()) {
    ET_LOG(
        Error,
        "Execution of forward failed with status 0x%" PRIx32,
        static_cast<uint32_t>(result.error()));
    return result.error();
  }

  *model_outputs = std::move(result.get());
  return Error::Ok;
}

inline Error postprocess_yolo_outputs(
    const std::vector<EValue>& model_outputs,
    const InputMeta& input_meta,
    size_t num_classes,
    float score_threshold,
    float nms_threshold,
    std::vector<Detection>* output_detections) {
  ET_CHECK_MSG(
      output_detections != nullptr,
      "YOLO output detections destination must not be null");
  ET_CHECK_MSG(
      !model_outputs.empty(),
      "YOLO forward must return at least one output tensor");

  const auto output_tensor = model_outputs[0].toTensor();
  *output_detections = decode_detections(
      output_tensor,
      input_meta,
      num_classes,
      score_threshold,
      nms_threshold);
  return Error::Ok;
}

inline Error run_yolo_once(
    Module* module,
    const std::vector<executorch::aten::SizesType>& input_shape,
    const std::vector<float>& input_buffer,
    const InputMeta& input_meta,
    size_t num_classes,
    float score_threshold,
    float nms_threshold,
    std::vector<Detection>* output_detections) {
  std::vector<EValue> model_outputs;
  Error error = run_yolo_inference(module, input_shape, input_buffer, &model_outputs);
  if (error != Error::Ok) {
    return error;
  }
  return postprocess_yolo_outputs(
      model_outputs,
      input_meta,
      num_classes,
      score_threshold,
      nms_threshold,
      output_detections);
}

inline bool write_detections(
    const std::string& output_path,
    const std::vector<Detection>& detections) {
  std::ofstream output(output_path);
  if (!output.is_open()) {
    ET_LOG(Error, "Unable to create output file: %s", output_path.c_str());
    return false;
  }
  output << std::fixed << std::setprecision(6);
  for (const auto& detection : detections) {
    output << detection.class_id << " " << detection.confidence << " "
           << detection.left << " " << detection.top << " "
           << detection.right << " " << detection.bottom << "\n";
  }
  return true;
}

inline bool ensure_directory(const std::string& directory_path) {
  std::error_code error_code;
  std::filesystem::create_directories(directory_path, error_code);
  if (error_code) {
    ET_LOG(
        Error,
        "Unable to create directory %s: %s",
        directory_path.c_str(),
        error_code.message().c_str());
    return false;
  }
  return true;
}

} // namespace example

#endif
