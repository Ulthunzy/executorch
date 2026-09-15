/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/android/jni/encrypted_model_loader.h>
#include <executorch/runtime/platform/runtime.h>

#include <cassert>
#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>

int main(int argc, char** argv) {
  assert(argc == 5);
  executorch::runtime::runtime_init();
  auto result = executorch_jni::load_encrypted_model(argv[1], argv[2]);
  assert(static_cast<int>(result.error()) == std::stoi(argv[3]));
  if (!result.ok()) {
    return 0;
  }
  if (std::string(argv[4]) == "plain") {
    assert(*result == nullptr);
    return 0;
  }
  assert(*result != nullptr);
  std::ifstream input(argv[4], std::ios::binary);
  assert(input.good());
  std::vector<char> expected(
      (std::istreambuf_iterator<char>(input)),
      std::istreambuf_iterator<char>());
  auto& loader = **result;
  assert(loader.size().ok() && *loader.size() == expected.size());
  using Info = executorch::runtime::DataLoader::SegmentInfo;
  auto buffer = loader.load(0, expected.size(), Info(Info::Type::Program));
  assert(buffer.ok());
  assert(std::memcmp(buffer->data(), expected.data(), expected.size()) == 0);
  auto segment = loader.load(7, 9, Info(Info::Type::Constant));
  assert(segment.ok());
  assert(std::memcmp(segment->data(), expected.data() + 7, 9) == 0);
  assert(!loader.load(expected.size(), 1, Info(Info::Type::Program)).ok());
}
