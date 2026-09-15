/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/data_loader.h>
#include <executorch/runtime/core/result.h>
#include <memory>
#include <string>

namespace executorch_jni {

// Returns a null loader for plain files. Encrypted buffers are exposed only
// after AES-GCM authentication succeeds and are erased when the loader is
// freed.
executorch::runtime::Result<std::unique_ptr<executorch::runtime::DataLoader>>
load_encrypted_model(const std::string& path, const std::string& key_hex);

} // namespace executorch_jni
