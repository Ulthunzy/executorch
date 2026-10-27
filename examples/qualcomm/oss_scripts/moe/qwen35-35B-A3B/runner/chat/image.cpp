#include "image.h"
#include "../prefill/vision_encoder.h"
#include "../runtime_utils.h"

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/qwen_image.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cstring>
#include <fstream>

namespace moe_chat {
namespace {
std::vector<float> table(const std::string& path) {
  const auto data = bytes(path);
  if (data.size() < 2048 * 32 * sizeof(float) ||
      data.size() % (32 * sizeof(float)))
    throw std::runtime_error("Invalid base RoPE table: " + path);
  std::vector<float> result(data.size() / sizeof(float));
  memcpy(result.data(), data.data(), data.size());
  return result;
}
} // namespace

ImageLayout::ImageLayout(const std::string& root) {
  nlohmann::json metadata, config;
  std::ifstream(root + "/VISION.json") >> metadata;
  std::ifstream(root + "/model-config.json") >> config;
  const auto& v = config.at("vision_config");
  height = metadata.at("image_height");
  width = metadata.at("image_width");
  patch = v.at("patch_size");
  merge = v.at("spatial_merge_size");
  temporal = v.at("temporal_patch_size");
  sections = config.at("text_config")
                 .at("rope_parameters")
                 .at("mrope_section")
                 .get<std::array<int, 3>>();
  if (patch <= 0 || merge <= 0 || temporal <= 0 || height <= 0 || width <= 0 ||
      height > 4096 || width > 4096 || patch > 128 || merge > 8 ||
      temporal > 8 || height % (patch * merge) || width % (patch * merge) ||
      int64_t(height) * width * 3 * temporal * sizeof(float) >
          64 * 1024 * 1024 ||
      v.at("in_channels") != 3 || config.at("image_token_id") != 248056 ||
      metadata.at("image_token_id") != 248056 || metadata.at("version") != 1 ||
      metadata.at("precision") != "W8A16" ||
      metadata.at("input_shape") !=
          nlohmann::json::array(
              {1,
               height / patch * (width / patch),
               3 * temporal * patch * patch}) ||
      metadata.at("output_shape") !=
          nlohmann::json::array({1, tokens(), 2048}) ||
      std::any_of(sections.begin(), sections.end(), [](int n) {
        return n <= 0 || n > 32;
      }))
    throw std::runtime_error("Vision metadata does not match the model");
}

int ImageLayout::tokens() const {
  return height / (patch * merge) * (width / (patch * merge));
}

std::vector<float> image_patches(
    const std::string& path,
    const ImageLayout& layout) {
  const auto begin = Clock::now();
  auto result = example::qwen_image_patches(
      path, layout.height, layout.width, layout.patch, layout.merge, layout.temporal);
  printf("IMAGE_PREPROCESS target=%dx%d tokens=%d decode_resize_patch_ms=%.6f device=1\n",
      layout.width, layout.height, layout.tokens(), elapsed(begin));
  return result;
}

std::vector<float> encode_image(
    VisionEncoder* vision,
    const std::vector<float>& pixels) {
  if (!vision ||
      pixels.size() * sizeof(float) != Graph::size(vision->graph.inputs[0]))
    throw std::runtime_error(
        "Image patches differ from the resident vision graph");
  memcpy(vision->graph.input(0), pixels.data(), pixels.size() * sizeof(float));
  return vision->encode();
}

RotaryPositions::RotaryPositions(const std::string& root)
    : base_cos(table(root + "/base-cos.f32")),
      base_sin(table(root + "/base-sin.f32")) {
  if (base_cos.size() != base_sin.size())
    throw std::runtime_error("RoPE table sizes differ");
  reset();
}

void RotaryPositions::reset() {
  cos = base_cos;
  sin = base_sin;
  delta = 0;
}

void RotaryPositions::append(
    const std::vector<uint64_t>& ids,
    size_t start,
    const ImageLayout* image) {
  if (start + ids.size() > cos.size() / 32)
    throw std::runtime_error("Input exceeds RoPE table size");
  int logical = int(start) + delta;
  size_t cursor = 0;
  auto row = [&](size_t index, const std::array<int, 3>& positions) {
    for (int j = 0; j < 32; ++j) {
      int axis = 0;
      if (image && j % 3 == 1 && j < image->sections[1] * 3)
        axis = 1;
      if (image && j % 3 == 2 && j < image->sections[2] * 3)
        axis = 2;
      if (positions[axis] < 0 || size_t(positions[axis]) >= cos.size() / 32)
        throw std::runtime_error("Image position exceeds RoPE table size");
      cos[index * 32 + j] = base_cos[positions[axis] * 32 + j];
      sin[index * 32 + j] = base_sin[positions[axis] * 32 + j];
    }
  };
  while (cursor < ids.size()) {
    if (ids[cursor] != 248056) {
      row(start + cursor++, {logical, logical, logical});
      ++logical;
      continue;
    }
    if (!image || cursor + image->tokens() > ids.size() ||
        !std::all_of(
            ids.begin() + cursor,
            ids.begin() + cursor + image->tokens(),
            [](auto id) {
              return id == 248056;
            }))
      throw std::runtime_error("Invalid image token span");
    const int h = image->height / (image->patch * image->merge);
    const int w = image->width / (image->patch * image->merge);
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x)
        row(start + cursor++, {logical, logical + y, logical + x});
    logical += std::max(h, w);
  }
  delta = logical - int(start + ids.size());
  for (size_t i = start + ids.size(); i < cos.size() / 32; ++i) {
    const int pos = int(i) + delta;
    row(i, {pos, pos, pos});
  }
}

void RotaryPositions::save(const std::string& root) const {
  ::save(root + "/cos.f32", cos.data(), cos.size() * sizeof(float));
  ::save(root + "/sin.f32", sin.data(), sin.size() * sizeof(float));
}
} // namespace moe_chat
