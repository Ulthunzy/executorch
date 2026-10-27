#include "qwen_image.h"

#include <android/bitmap.h>
#include <android/imagedecoder.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace example {
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

} // namespace

std::vector<float> qwen_image_patches(
    const std::string& path,
    int height,
    int width,
    int patch,
    int merge,
    int temporal) {
  if (height <= 0 || width <= 0 || height > 4096 || width > 4096 ||
      patch <= 0 || patch > 128 || merge <= 0 || merge > 8 || temporal <= 0 ||
      temporal > 8 || height % (patch * merge) || width % (patch * merge) ||
      int64_t(height) * width * 3 * temporal > 16 * 1024 * 1024)
    throw std::invalid_argument("Unsupported Qwen image geometry");
  const auto source = decode_image(path);
  const auto image = resize(source, width, height);
  std::vector<float> result;
  result.reserve(size_t(height) * width * 3 * temporal);
  for (int block_y = 0; block_y < height / (patch * merge); ++block_y)
    for (int block_x = 0; block_x < width / (patch * merge); ++block_x)
      for (int my = 0; my < merge; ++my)
        for (int mx = 0; mx < merge; ++mx)
          for (int c = 0; c < 3; ++c)
            for (int t = 0; t < temporal; ++t)
              for (int py = 0; py < patch; ++py)
                for (int px = 0; px < patch; ++px) {
                  const int y = (block_y * merge + my) * patch + py;
                  const int x = (block_x * merge + mx) * patch + px;
                  result.push_back(
                      (float(
                           image
                               .pixels[(size_t(y) * image.width + x) * 4 + c]) /
                           255.0f -
                       0.5f) /
                      0.5f);
                }
  return result;
}

} // namespace example
