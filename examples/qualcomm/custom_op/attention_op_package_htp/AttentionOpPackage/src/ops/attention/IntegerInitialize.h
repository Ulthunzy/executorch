// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
static HVX_Vector
pack_integer_weight(const uint16_t* source, uint8_t* target, unsigned k) {
  auto sum = Q6_V_vzero();
  for (unsigned i = 0; i < k; i += 4) {
    auto v0 = *reinterpret_cast<const HVX_Vector*>(source + i * 64);
    auto v1 = *reinterpret_cast<const HVX_Vector*>(source + (i + 1) * 64);
    auto v2 = *reinterpret_cast<const HVX_Vector*>(source + (i + 2) * 64);
    auto v3 = *reinterpret_cast<const HVX_Vector*>(source + (i + 3) * 64);
    sum = Q6_Vh_vadd_VhVh(
        sum, Q6_Vh_vadd_VhVh(Q6_Vh_vadd_VhVh(v0, v1), Q6_Vh_vadd_VhVh(v2, v3)));
    auto p01 = Q6_W_vshuff_VVR(v1, v0, -2), p23 = Q6_W_vshuff_VVR(v3, v2, -2);
    auto b01 = Q6_Vb_vpacke_VhVh(Q6_V_hi_W(p01), Q6_V_lo_W(p01));
    auto b23 = Q6_Vb_vpacke_VhVh(Q6_V_hi_W(p23), Q6_V_lo_W(p23));
    auto result = Q6_W_vshuff_VVR(b23, b01, -2);
    const unsigned offset = (i / 32) * 1024 + (i % 32 / 4) * 128;
    *reinterpret_cast<HVX_Vector*>(target + offset) =
        Q6_V_vxor_VV(Q6_V_lo_W(result), Q6_V_vsplat_R(0x80808080));
    *reinterpret_cast<HVX_Vector*>(target + k * 32 + offset) =
        Q6_V_vxor_VV(Q6_V_hi_W(result), Q6_V_vsplat_R(0x80808080));
  }
  return sum;
}
static uint32_t initialize_integer(
    QHPI_RuntimeHandle* handle,
    uint32_t no,
    QHPI_Tensor** out,
    uint32_t ni,
    const QHPI_Tensor* const* in,
    const Configuration* supplied_config = nullptr,
    CacheMappings::Slot* supplied_cache = nullptr) {
  if (no != 2 || ni != 7)
    return QHPI_ERROR_FATAL;
  Configuration parsed{};
  const auto& config = supplied_config ? *supplied_config : parsed;
  if ((!supplied_config && !configuration(ni, in, parsed)) || !config.fast() ||
      !QCOM_HTP_V81)
    return QHPI_UNSUPPORTED;
  const unsigned slice = qhpi_slice_number(handle),
                 slices = qhpi_num_slices(handle);
  if (!slices || slices > max_workers || slice >= slices)
    return QHPI_UNSUPPORTED;
  const auto shape = qhpi_tensor_shape(in[0]);
  const unsigned qrows = shape.dims[2],
                 kvheads = qhpi_tensor_shape(in[2]).dims[1],
                 capacity = config.capacity;
  if (shape.dims[1] != kvheads * 4)
    return QHPI_UNSUPPORTED;
  const unsigned kv_rows = attention_cache::packed_rows(capacity),
                 kv_bytes = kvheads * kv_rows * 256 * 2;
  auto* work = static_cast<uint8_t*>(qhpi_tensor_raw_data(out[0]));
  if (!slice) {
    for (unsigned i = 0; i < header / 128; ++i)
      reinterpret_cast<HVX_Vector*>(work)[i] = Q6_V_vzero();
    __atomic_store_n(
        reinterpret_cast<uint32_t*>(work + integer_mode_offset),
        1u,
        __ATOMIC_RELEASE);
    __atomic_store_n(
        &reinterpret_cast<attention_hmx::Queue*>(work)->workers,
        slices,
        __ATOMIC_RELEASE);
  }
  const auto* lengths =
      static_cast<const int32_t*>(qhpi_tensor_raw_data(in[5]));
  const unsigned history = lengths[2], first = lengths[1], valid = lengths[0],
                 keys = history + valid, padded = (keys + 63) & ~63u;
  const auto* meta = static_cast<const float*>(qhpi_tensor_raw_data(in[6]));
  const auto* fd = static_cast<const int32_t*>(qhpi_tensor_raw_data(in[1]));
  auto* key = supplied_cache
      ? reinterpret_cast<uint8_t*>(supplied_cache->p)
      : reinterpret_cast<uint8_t*>(cache_pointer(fd, 2 * kv_bytes));
  auto* cached =
      supplied_cache ? supplied_cache : packed_cache(fd[0], kv_bytes);
  if (!key || !cached)
    return QHPI_ERROR_FATAL;
  const auto* nk = static_cast<const uint8_t*>(qhpi_tensor_raw_data(in[2]));
  const auto* nv = static_cast<const uint8_t*>(qhpi_tensor_raw_data(in[4]));
  auto kq = qhpi_tensor_quant_parameters(in[2]),
       vq = qhpi_tensor_quant_parameters(in[4]);
  if (kq.stepsize != meta[9] || kq.zero_offset != int(meta[10]) ||
      vq.stepsize != meta[11] || vq.zero_offset != int(meta[12]))
    return QHPI_UNSUPPORTED;
  const bool incremental = cached->can_append(fd, first, history, true);
  const PackedKv8 packed(cached->packed, kvheads, kv_rows);
  auto* sums = packed.key_sums;
  if (incremental && qrows == 1 && valid == 1 && history % 64) {
    for (unsigned kv = slice; kv < kvheads; kv += slices) {
      const unsigned column = history % 64;
      auto* panel = packed.key_panel(kv, history & ~63u);
      alignas(128) uint32_t words[64];
      const auto lo = vmemu(nk + kv * 256);
      const auto hi = vmemu(nk + kv * 256 + 128);
      vmemu(words) = Q6_V_vxor_VV(lo, Q6_V_vsplat_R(0x80808080));
      vmemu(words + 32) = Q6_V_vxor_VV(hi, Q6_V_vsplat_R(0x80808080));
      auto sum = Q6_Vw_vadd_VwVw(
          Q6_Vuw_vrmpy_VubRub(lo, 0x01010101),
          Q6_Vuw_vrmpy_VubRub(hi, 0x01010101));
      for (unsigned shift = 64; shift >= 4; shift /= 2)
        sum = Q6_Vw_vadd_VwVw(sum, Q6_V_vror_VR(sum, shift));
      for (unsigned d = 0; d < 256; d += 4)
        *reinterpret_cast<uint32_t*>(
            panel + (column / 32) * 8192 + (d / 32) * 1024 +
            (d % 32 / 4) * 128 + (column % 32) * 4) = words[d / 4];
      sums[kv * kv_rows + history] = Q6_R_vextract_VR(sum, 0);
    }
    const unsigned row = history % 256;
    const unsigned width = std::min(256u, padded - history / 256 * 256);
    const auto byte_mask =
        Q6_Q_vcmp_eq_VbVb(Q6_V_vsplat_R(0x03020100), Q6_Vb_vsplat_R(row % 4));
    for (unsigned task = slice; task < kvheads * 4; task += slices) {
      const unsigned kv = task / 4, col = task % 4 * 64;
      const auto bytes = attention_fixed::load_codes64(nv + kv * 256, col);
      const auto pair = Q6_Wuw_vunpack_Vuh(bytes);
      auto* destination =
          packed.value_column(kv, history & ~255u, col, padded) +
          row / 32 * 1024 + row % 32 / 4 * 128;
      const auto lo = Q6_Vw_vasl_VwR(
          Q6_V_vxor_VV(Q6_V_lo_W(pair), Q6_V_vsplat_R(128)), row % 4 * 8);
      const auto hi = Q6_Vw_vasl_VwR(
          Q6_V_vxor_VV(Q6_V_hi_W(pair), Q6_V_vsplat_R(128)), row % 4 * 8);
      Q6_vmem_QRIV(byte_mask, reinterpret_cast<HVX_Vector*>(destination), lo);
      Q6_vmem_QRIV(
          byte_mask,
          reinterpret_cast<HVX_Vector*>(destination + width * 32),
          hi);
    }
    return QHPI_SUCCESS;
  }
  auto* value = reinterpret_cast<uint8_t*>(
      cache_pointer(static_cast<const int32_t*>(qhpi_tensor_raw_data(in[3]))));
  if (!value)
    return QHPI_ERROR_FATAL;
  const unsigned first_block = incremental ? history / 64 : 0,
                 first_tile = incremental ? history / 256 : 0,
                 recent = incremental ? std::min(history, cached->last_valid)
                                      : history;
  // AR1 appends bypass raw history; invalidate the complete tail before
  // repacking.
  const bool direct_tail =
      incremental && (qrows == 1 || cached->last_valid == 1);
  const unsigned recent_keys =
      direct_tail ? history - first_block * 64 : recent;
  const unsigned recent_values =
      direct_tail ? history - first_tile * 256 : recent;
  for (unsigned d = slice; d < kvheads * 256; d += slices)
    invalidate(key, d * capacity + first + history - recent_keys, recent_keys);
  for (unsigned h = slice; h < kvheads; h += slices)
    invalidate(
        value,
        (h * capacity + first + history - recent_values) * 256,
        recent_values * 256);
  // Other workers read the rows invalidated by this worker.
  const auto pack_epoch =
      __atomic_load_n(&cached->pack_epoch, __ATOMIC_ACQUIRE);
  if (__atomic_add_fetch(&cached->pack_arrived, 1u, __ATOMIC_ACQ_REL) ==
      slices) {
    __atomic_store_n(&cached->pack_arrived, 0u, __ATOMIC_RELAXED);
    __atomic_add_fetch(&cached->pack_epoch, 1u, __ATOMIC_RELEASE);
  } else {
    while (__atomic_load_n(&cached->pack_epoch, __ATOMIC_ACQUIRE) == pack_epoch)
      attention_hmx::pause();
  }
  const unsigned stage_stride =
      qrows == 1 ? kvheads * per_worker / slices : per_worker;
  auto* stage =
      reinterpret_cast<uint16_t*>(work + worker_start + slice * stage_stride);
  const unsigned blocks = padded / 64;
  for (unsigned task = slice; task < kvheads * (blocks - first_block);
       task += slices) {
    unsigned kv = task / (blocks - first_block),
             start = (first_block + task % (blocks - first_block)) * 64;
    for (unsigned d = 0; d < 256; ++d) {
      HVX_Vector raw;
      if (start + 64 <= history)
        raw = attention_fixed::load_codes64(
            key + (kv * 256 + d) * capacity, first + start);
      else if (start >= history && start + 64 <= keys)
        raw = attention_fixed::load_codes64(
            nk + (kv * 256 + d) * qrows, start - history);
      else {
        alignas(128) uint16_t codes[64];
        for (unsigned c = 0; c < 64; ++c) {
          unsigned at = start + c;
          codes[c] = at < history ? key[(kv * 256 + d) * capacity + first + at]
              : at < keys         ? nk[(kv * 256 + d) * qrows + at - history]
                                  : uint16_t(meta[10]);
        }
        raw = *reinterpret_cast<HVX_Vector*>(codes);
      }
      *reinterpret_cast<HVX_Vector*>(stage + d * 64) = raw;
    }
    auto sum = pack_integer_weight(stage, packed.key_panel(kv, start), 256);
    *reinterpret_cast<HVX_Vector*>(sums + kv * kv_rows + start) = sum;
  }
  const unsigned tiles = (padded + 255) / 256;
  for (unsigned task = slice; task < kvheads * (tiles - first_tile) * 4;
       task += slices) {
    unsigned kv = task / ((tiles - first_tile) * 4),
             kt = first_tile + (task / 4) % (tiles - first_tile),
             col = task % 4 * 64, width = std::min(256u, padded - kt * 256);
    for (unsigned row = 0; row < width; ++row) {
      unsigned at = kt * 256 + row;
      auto raw = at < history
          ? attention_fixed::load_codes64(
                value + (kv * capacity + first + at) * 256, col)
          : at < keys ? attention_fixed::load_codes64(
                            nv + (kv * qrows + at - history) * 256, col)
                      : Q6_Vh_vsplat_R(int(meta[12]));
      *reinterpret_cast<HVX_Vector*>(stage + row * 64) = raw;
    }
    pack_integer_weight(
        stage, packed.value_column(kv, kt * 256, col, padded), width);
  }
  return QHPI_SUCCESS;
}
