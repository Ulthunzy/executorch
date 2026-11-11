// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace htp_op_package {
struct Operation {
  const char* name;
  uint32_t inputs;
  uint32_t outputs;
};

inline constexpr Operation operations[] = {
    {"Chunk", 7, 2},
    {"ChunkReserve", 7, 1},
    {"ChunkExecute", 8, 2},
    {"ChunkStateUpdateBatch", 5, 1},
    {"ChunkStateExecuteBatch", 6, 1},
    {"DeltaNetCoreGatedVtcm", 8, 2},
    {"ChunkStateUpdate", 5, 1},
    {"ChunkStateReserve", 5, 1},
    {"ChunkStateExecute", 6, 1},
    {"DeltaNetCore", 6, 2},
    {"DeltaNetCoreStream", 6, 2},
    {"DeltaNetCoreVtcm", 6, 2},
    {"DeltaNetCoreFp16Accum", 6, 2},
    {"DeltaNetCoreFp16State", 6, 2},
    {"DeltaNetCoreValid", 7, 2},
    {"DeltaNetCoreValidVtcm", 7, 2},
    {"AttentionValid", 7, 1},
    {"AttentionInitialize", 7, 2},
    {"AttentionExecute", 9, 1},
};

constexpr auto operationNames() {
  std::array<const char*, sizeof(operations) / sizeof(operations[0])> names{};
  for (size_t i = 0; i < names.size(); ++i)
    names[i] = operations[i].name;
  return names;
}
} // namespace htp_op_package
