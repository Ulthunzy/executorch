#include "image.h"
#include "../prefill/vision_encoder.h"
#include "../runtime_utils.h"

#include <android/bitmap.h>
#include <android/imagedecoder.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <dlfcn.h>
#include <fcntl.h>
#include <fstream>
#include <unistd.h>

namespace moe_chat {
namespace {
struct Bitmap {
  int width, height;
  std::vector<uint8_t> pixels;
};

Bitmap decode_image(const std::string& path) {
  struct Decoder {
    void* library = dlopen("libjnigraphics.so", RTLD_NOW | RTLD_LOCAL);
    int fd = -1;
    AImageDecoder* decoder = nullptr;
    void (*destroy)(AImageDecoder*) = nullptr;

    ~Decoder() {
      if (decoder && destroy)
        destroy(decoder);
      if (fd >= 0)
        close(fd);
      if (library)
        dlclose(library);
    }
  } owner;

  if (!owner.library)
    throw std::runtime_error(
        "Android ImageDecoder requires Android 11 or newer");
  const auto create = reinterpret_cast<int (*)(int, AImageDecoder**)>(
      dlsym(owner.library, "AImageDecoder_createFromFd"));
  owner.destroy = reinterpret_cast<decltype(owner.destroy)>(
      dlsym(owner.library, "AImageDecoder_delete"));
  const auto header = reinterpret_cast<
      const AImageDecoderHeaderInfo* (*)(const AImageDecoder*)>(
      dlsym(owner.library, "AImageDecoder_getHeaderInfo"));
  const auto width =
      reinterpret_cast<int32_t (*)(const AImageDecoderHeaderInfo*)>(
          dlsym(owner.library, "AImageDecoderHeaderInfo_getWidth"));
  const auto height =
      reinterpret_cast<int32_t (*)(const AImageDecoderHeaderInfo*)>(
          dlsym(owner.library, "AImageDecoderHeaderInfo_getHeight"));
  const auto format = reinterpret_cast<int (*)(AImageDecoder*, int32_t)>(
      dlsym(owner.library, "AImageDecoder_setAndroidBitmapFormat"));
  const auto alpha = reinterpret_cast<int (*)(AImageDecoder*, bool)>(
      dlsym(owner.library, "AImageDecoder_setUnpremultipliedRequired"));
  const auto decode =
      reinterpret_cast<int (*)(AImageDecoder*, void*, size_t, size_t)>(
          dlsym(owner.library, "AImageDecoder_decodeImage"));
  if (!create || !owner.destroy || !header || !width || !height || !format ||
      !alpha || !decode)
    throw std::runtime_error("Android ImageDecoder API is unavailable");
  owner.fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (owner.fd < 0 ||
      create(owner.fd, &owner.decoder) != ANDROID_IMAGE_DECODER_SUCCESS)
    throw std::runtime_error("Cannot decode image: " + path);
  const auto* info = header(owner.decoder);
  Bitmap result{width(info), height(info), {}};
  if (result.width <= 0 || result.height <= 0 ||
      int64_t(result.width) * result.height > 32 * 1024 * 1024)
    throw std::runtime_error("Image exceeds the 32 megapixel decode limit");
  if (format(owner.decoder, ANDROID_BITMAP_FORMAT_RGBA_8888) ||
      alpha(owner.decoder, true))
    throw std::runtime_error("Cannot configure RGB image decoding");
  result.pixels.resize(size_t(result.width) * result.height * 4);
  const int status = decode(
      owner.decoder,
      result.pixels.data(),
      result.width * 4,
      result.pixels.size());
  if (status != ANDROID_IMAGE_DECODER_SUCCESS)
    throw std::runtime_error("Incomplete or invalid image: " + path);
  return result;
}

double cubic(double value) {
  const double x = std::abs(value);
  if (x < 1)
    return ((1.5 * x - 2.5) * x) * x + 1;
  if (x < 2)
    return ((-0.5 * x + 2.5) * x - 4) * x + 2;
  return 0;
}

struct Filter {
  int begin;
  std::vector<int32_t> weights;
};

std::vector<Filter> filters(int input, int output) {
  std::vector<Filter> result;
  const double scale = double(input) / output, stretch = std::max(scale, 1.0);
  for (int i = 0; i < output; ++i) {
    const double center = (i + 0.5) * scale;
    const int begin = std::max(int(center - 2 * stretch + 0.5), 0);
    const int end = std::min(int(center + 2 * stretch + 0.5), input);
    std::vector<double> weights;
    double sum = 0;
    for (int j = begin; j < end; ++j) {
      weights.push_back(cubic((j - center + 0.5) / stretch));
      sum += weights.back();
    }
    Filter filter{begin, {}};
    for (double weight : weights)
      filter.weights.push_back(int32_t(std::round(weight / sum * (1 << 22))));
    result.push_back(std::move(filter));
  }
  return result;
}

Bitmap resize(const Bitmap& input, int width, int height) {
  // Match Pillow's antialiased BICUBIC: normalize each filter, use 22-bit
  // coefficients and round/clip after each separable pass on RGB bytes.
  const auto horizontal = filters(input.width, width);
  const auto vertical = filters(input.height, height);
  Bitmap intermediate{
      width,
      input.height,
      std::vector<uint8_t>(size_t(width) * input.height * 4)};
  Bitmap result{
      width, height, std::vector<uint8_t>(size_t(width) * height * 4)};
  for (int y = 0; y < input.height; ++y)
    for (int x = 0; x < width; ++x)
      for (int c = 0; c < 3; ++c) {
        int64_t sum = 1 << 21;
        const auto& filter = horizontal[x];
        for (size_t k = 0; k < filter.weights.size(); ++k)
          sum +=
              int64_t(
                  input.pixels
                      [(size_t(y) * input.width + filter.begin + k) * 4 + c]) *
              filter.weights[k];
        intermediate.pixels[(size_t(y) * width + x) * 4 + c] =
            std::clamp<int64_t>(sum >> 22, 0, 255);
      }
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x)
      for (int c = 0; c < 3; ++c) {
        int64_t sum = 1 << 21;
        const auto& filter = vertical[y];
        for (size_t k = 0; k < filter.weights.size(); ++k)
          sum +=
              int64_t(intermediate
                          .pixels[((filter.begin + k) * width + x) * 4 + c]) *
              filter.weights[k];
        result.pixels[(size_t(y) * width + x) * 4 + c] =
            std::clamp<int64_t>(sum >> 22, 0, 255);
      }
  return result;
}

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
  const auto source = decode_image(path);
  const auto image = resize(source, layout.width, layout.height);
  std::vector<float> result;
  result.reserve(size_t(layout.height) * layout.width * 3 * layout.temporal);
  for (int block_y = 0; block_y < layout.height / (layout.patch * layout.merge);
       ++block_y)
    for (int block_x = 0;
         block_x < layout.width / (layout.patch * layout.merge);
         ++block_x)
      for (int my = 0; my < layout.merge; ++my)
        for (int mx = 0; mx < layout.merge; ++mx)
          for (int c = 0; c < 3; ++c)
            for (int t = 0; t < layout.temporal; ++t)
              for (int py = 0; py < layout.patch; ++py)
                for (int px = 0; px < layout.patch; ++px) {
                  const int y =
                      (block_y * layout.merge + my) * layout.patch + py;
                  const int x =
                      (block_x * layout.merge + mx) * layout.patch + px;
                  result.push_back(
                      (float(
                           image
                               .pixels[(size_t(y) * image.width + x) * 4 + c]) /
                           255.0f -
                       0.5f) /
                      0.5f);
                }
  printf(
      "IMAGE_PREPROCESS source=%dx%d target=%dx%d patches=%zu tokens=%d decode_resize_patch_ms=%.6f device=1\n",
      source.width,
      source.height,
      layout.width,
      layout.height,
      result.size() / (3 * layout.temporal * layout.patch * layout.patch),
      layout.tokens(),
      elapsed(begin));
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
