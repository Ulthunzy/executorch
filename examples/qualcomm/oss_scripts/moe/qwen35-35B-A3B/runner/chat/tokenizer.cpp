#include "tokenizer.h"

#include <dlfcn.h>
#include <fstream>
#include <limits>
#include <unicode/unorm2.h>
#include <unicode/ustring.h>
#include <unistd.h>

namespace moe_chat {
namespace {
// The bundled NFC implementation calls a lossy NFD helper that drops accents.
// Use Android's public ICU API; retain HFTokenizer's BPE and streaming decoder.
struct Unicode {
  using GetNfc = const UNormalizer2* (*)(UErrorCode*);
  using Normalize = int32_t (*)(
      const UNormalizer2*,
      const UChar*,
      int32_t,
      UChar*,
      int32_t,
      UErrorCode*);
  using FromUtf8 =
      UChar* (*)(UChar*, int32_t, int32_t*, const char*, int32_t, UErrorCode*);
  using ToUtf8 =
      char* (*)(char*, int32_t, int32_t*, const UChar*, int32_t, UErrorCode*);
  void* library = dlopen("libicu.so", RTLD_NOW | RTLD_LOCAL);
  GetNfc get_nfc = library ? reinterpret_cast<decltype(get_nfc)>(
                                 dlsym(library, "unorm2_getNFCInstance"))
                           : nullptr;
  Normalize normalize = library ? reinterpret_cast<decltype(normalize)>(
                                      dlsym(library, "unorm2_normalize"))
                                : nullptr;
  FromUtf8 from_utf8 = library
      ? reinterpret_cast<decltype(from_utf8)>(dlsym(library, "u_strFromUTF8"))
      : nullptr;
  ToUtf8 to_utf8 = library
      ? reinterpret_cast<decltype(to_utf8)>(dlsym(library, "u_strToUTF8"))
      : nullptr;

  std::string nfc(const std::string& text) {
    if (!get_nfc || !normalize || !from_utf8 || !to_utf8)
      throw std::runtime_error(
          "Android public ICU APIs are required for NFC tokenization");
    if (text.size() >= size_t(std::numeric_limits<int32_t>::max()))
      throw std::runtime_error("Prompt is too large");
    UErrorCode status = U_ZERO_ERROR;
    std::vector<UChar> input(text.size() + 1);
    int32_t input_size = 0;
    from_utf8(
        input.data(),
        input.size(),
        &input_size,
        text.data(),
        text.size(),
        &status);
    if (U_FAILURE(status))
      throw std::runtime_error("Invalid UTF-8 input");
    const auto* normalizer = get_nfc(&status);
    const int32_t length =
        normalize(normalizer, input.data(), input_size, nullptr, 0, &status);
    if (status != U_BUFFER_OVERFLOW_ERROR && U_FAILURE(status))
      throw std::runtime_error("NFC sizing failed");
    status = U_ZERO_ERROR;
    std::vector<UChar> output(length + 1);
    normalize(
        normalizer,
        input.data(),
        input_size,
        output.data(),
        output.size(),
        &status);
    if (U_FAILURE(status))
      throw std::runtime_error("NFC normalization failed");
    std::string result(size_t(length) * 3 + 1, '\0');
    int32_t bytes = 0;
    to_utf8(
        result.data(), result.size(), &bytes, output.data(), length, &status);
    if (U_FAILURE(status))
      throw std::runtime_error("UTF-8 conversion failed");
    result.resize(bytes);
    return result;
  }
};
} // namespace

tokenizers::Error DeviceTokenizer::load(const std::string& path) {
  std::ifstream source(path);
  if (!source)
    throw std::runtime_error("Cannot open tokenizer: " + path);
  nlohmann::json config;
  source >> config;
  if (config.at("normalizer") != nlohmann::json{{"type", "NFC"}})
    throw std::runtime_error(
        "Expected the Qwen3.5 NFC tokenizer configuration");
  config["normalizer"] = nullptr;
  char temporary[] = "/data/local/tmp/qwen35-tokenizer-XXXXXX";
  const int fd = mkstemp(temporary);
  if (fd < 0)
    throw std::runtime_error("Cannot create tokenizer configuration");
  ::close(fd);

  struct Remove {
    const char* path;

    ~Remove() {
      unlink(path);
    }
  } remove{temporary};

  {
    std::ofstream file(temporary);
    file << config;
    if (!file)
      throw std::runtime_error("Cannot write tokenizer configuration");
  }
  return tokenizers::HFTokenizer::load(temporary);
}

tokenizers::Result<std::vector<uint64_t>>
DeviceTokenizer::encode(const std::string& text, int8_t bos, int8_t eos) const {
  static Unicode unicode;
  return tokenizers::HFTokenizer::encode(unicode.nfc(text), bos, eos);
}
} // namespace moe_chat
