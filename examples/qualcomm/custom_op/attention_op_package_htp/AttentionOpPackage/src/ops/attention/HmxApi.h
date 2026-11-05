// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <cstdint>
#ifdef __hexagon__
extern "C" {
int32_t qhblas_hmx_transfer_row_ahf(__fp16*, uint8_t*, uint32_t);
int32_t qhblas_hmx_transfer_column_ahf(__fp16*, uint8_t*, uint32_t, uint32_t);
int32_t qhblas_hmx_matrix_matrix_mpy_hf_sm(
    uint8_t*,
    uint32_t,
    uint8_t*,
    uint32_t,
    uint8_t*,
    uint32_t,
    uint8_t*);
}
#endif
