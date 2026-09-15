/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/android/jni/encrypted_model_loader.h>
#include <executorch/extension/data_loader/shared_ptr_data_loader.h>

#include <openssl/crypto.h>
#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <fstream>
#include <limits>
#include <vector>

namespace executorch_jni {
namespace {

constexpr std::array<uint8_t, 8> kEtenc1 = {'E', 'T', 'E', 'N', 'C', '1', 0, 0};
constexpr std::array<uint8_t, 8> kLegacy =
    {0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
constexpr size_t kNonceSize = 12;
constexpr size_t kTagSize = 16;
constexpr size_t kChunkSize = 1024 * 1024;

struct Key {
  std::array<uint8_t, 32> bytes{};
  size_t size = 0;
  ~Key() {
    OPENSSL_cleanse(bytes.data(), bytes.size());
  }
};

bool parse_key(const std::string& text, Key& key) {
  if (text.size() != 32 && text.size() != 64) {
    return false;
  }
  key.size = text.size() / 2;
  for (size_t i = 0; i < text.size(); ++i) {
    const char c = text[i];
    int value = c >= '0' && c <= '9' ? c - '0'
        : c >= 'a' && c <= 'f'       ? c - 'a' + 10
        : c >= 'A' && c <= 'F'       ? c - 'A' + 10
                                     : -1;
    if (value < 0) {
      return false;
    }
    key.bytes[i / 2] = (key.bytes[i / 2] << 4) | value;
  }
  return true;
}
} // namespace

using executorch::runtime::DataLoader;
using executorch::runtime::Error;
using executorch::runtime::Result;

Result<std::unique_ptr<DataLoader>> load_encrypted_model(
    const std::string& path,
    const std::string& key_hex) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  ET_CHECK_OR_RETURN_ERROR(
      input.good(), AccessFailed, "Cannot open model file");
  const auto file_end = input.tellg();
  ET_CHECK_OR_RETURN_ERROR(
      file_end >= 8, InvalidProgram, "Model header is truncated");
  ET_CHECK_OR_RETURN_ERROR(
      static_cast<uint64_t>(file_end) <= std::numeric_limits<size_t>::max(),
      InvalidArgument,
      "Model file is too large");
  const auto file_size = static_cast<size_t>(file_end);
  input.seekg(0);
  std::array<uint8_t, 8> magic{};
  input.read(reinterpret_cast<char*>(magic.data()), magic.size());
  ET_CHECK_OR_RETURN_ERROR(
      input.good(), AccessFailed, "Cannot read model header");
  if (magic != kEtenc1 && magic != kLegacy) {
    return std::unique_ptr<DataLoader>{};
  }

  const bool legacy = magic == kLegacy;
  const size_t header_size = 8 + kNonceSize + (legacy ? 32 : kTagSize);
  const size_t overhead = header_size + (legacy ? kTagSize : 0);
  ET_CHECK_OR_RETURN_ERROR(
      file_size > overhead, InvalidProgram, "Encrypted model is truncated");
  Key key;
  if (legacy) {
    input.read(reinterpret_cast<char*>(key.bytes.data()), key.bytes.size());
    key.size = key.bytes.size();
  }
  if (!key_hex.empty()) {
    ET_CHECK_OR_RETURN_ERROR(
        parse_key(key_hex, key), InvalidArgument, "Invalid AES model key");
  } else {
    ET_CHECK_OR_RETURN_ERROR(
        legacy, InvalidArgument, "ETENC1 requires qnnModelKeyHex");
  }
  std::array<uint8_t, kNonceSize> nonce{};
  std::array<uint8_t, kTagSize> tag{};
  input.read(reinterpret_cast<char*>(nonce.data()), nonce.size());
  ET_CHECK_OR_RETURN_ERROR(
      input.good(), AccessFailed, "Cannot read encrypted model nonce");
  if (legacy) {
    input.seekg(file_size - kTagSize);
  }
  input.read(reinterpret_cast<char*>(tag.data()), tag.size());
  ET_CHECK_OR_RETURN_ERROR(
      input.good(), AccessFailed, "Cannot read encrypted model tag");
  input.seekg(header_size);
  ET_CHECK_OR_RETURN_ERROR(
      input.good(), AccessFailed, "Cannot seek encrypted model payload");

  std::unique_ptr<EVP_CIPHER_CTX, decltype(&EVP_CIPHER_CTX_free)> cipher(
      EVP_CIPHER_CTX_new(), EVP_CIPHER_CTX_free);
  ET_CHECK_OR_RETURN_ERROR(
      cipher != nullptr, MemoryAllocationFailed, "Cannot allocate AES context");
  ET_CHECK_OR_RETURN_ERROR(
      EVP_DecryptInit_ex(
          cipher.get(),
          key.size == 16 ? EVP_aes_128_gcm() : EVP_aes_256_gcm(),
          nullptr,
          key.bytes.data(),
          nonce.data()) == 1 &&
          EVP_CIPHER_CTX_ctrl(
              cipher.get(), EVP_CTRL_GCM_SET_TAG, tag.size(), tag.data()) == 1,
      Internal,
      "Cannot initialize AES-GCM decryption");

  const size_t payload_size = file_size - overhead;
  auto* data = new (std::nothrow) uint8_t[payload_size];
  ET_CHECK_OR_RETURN_ERROR(
      data != nullptr,
      MemoryAllocationFailed,
      "Cannot allocate decrypted model");
  std::shared_ptr<void> owner(data, [payload_size](void* ptr) {
    OPENSSL_cleanse(ptr, payload_size);
    delete[] static_cast<uint8_t*>(ptr);
  });
  std::vector<uint8_t> chunk(std::min(payload_size, kChunkSize));
  for (size_t offset = 0; offset < payload_size;) {
    const auto size = std::min(chunk.size(), payload_size - offset);
    input.read(reinterpret_cast<char*>(chunk.data()), size);
    ET_CHECK_OR_RETURN_ERROR(
        input.gcount() == static_cast<std::streamsize>(size),
        AccessFailed,
        "Cannot read encrypted model payload");
    int written = 0;
    ET_CHECK_OR_RETURN_ERROR(
        EVP_DecryptUpdate(
            cipher.get(), data + offset, &written, chunk.data(), size) == 1 &&
            written == static_cast<int>(size),
        InvalidProgram,
        "AES-GCM decryption failed");
    offset += size;
  }
  std::array<uint8_t, EVP_MAX_BLOCK_LENGTH> tail{};
  int written = 0;
  const int authenticated =
      EVP_DecryptFinal_ex(cipher.get(), tail.data(), &written);
  OPENSSL_cleanse(tail.data(), tail.size());
  ET_CHECK_OR_RETURN_ERROR(
      authenticated == 1 && written == 0,
      InvalidProgram,
      "Encrypted model authentication failed");
  return std::unique_ptr<DataLoader>(
      new executorch::extension::SharedPtrDataLoader(
          std::move(owner), payload_size));
}

} // namespace executorch_jni
