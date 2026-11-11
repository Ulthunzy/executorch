// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

static uint32_t grouped_prefill(
    QHPI_RuntimeHandle* handle,
    QHPI_Tensor** out,
    const QHPI_Tensor* const* in,
    unsigned kv_rows) {
  constexpr unsigned batches = 2, batch_stride = rows * 256 * 2,
                     probability_stride = rows * 512 * 2;
  const unsigned slice = qhpi_slice_number(handle),
                 active = unsigned(qhpi_num_slices(handle));
  auto* work = static_cast<uint8_t*>(qhpi_tensor_raw_data(in[0]));
  auto& queue = *reinterpret_cast<attention_hmx::Queue*>(work);
  if (slice >= active) {
    __atomic_store_n(&queue.jobs[slice].finished, 1u, __ATOMIC_RELEASE);
    return QHPI_SUCCESS;
  }
  const unsigned qrows = qhpi_tensor_shape(in[1]).dims[2],
                 kvheads = qhpi_tensor_shape(in[3]).dims[1],
                 kv_bytes = kvheads * kv_rows * 256 * 2;
  const auto* lengths =
      static_cast<const int32_t*>(qhpi_tensor_raw_data(in[6]));
  const auto* meta = static_cast<const float*>(qhpi_tensor_raw_data(in[7]));
  const unsigned valid = lengths[0], history = lengths[2],
                 padded = (history + valid + 63) & ~63u,
                 query_tiles = (valid + rows - 1) / rows;
  auto* cached = packed_cache(
      static_cast<const int32_t*>(qhpi_tensor_raw_data(in[2]))[0], kv_bytes);
  const auto* kpack = cached->packed;
  const auto* vpack = kpack + kv_bytes;
  auto& buffer = *reinterpret_cast<GroupedScratch*>(
      work + worker_start + slice * sizeof(GroupedScratch));
  // Query packing finishes before HMX reuses the product buffer.
  auto* qa = buffer.product;
  auto* packed_q = buffer.query;
  auto* packed_a = buffer.probability;
  auto* packed_c = buffer.product;
  auto* codes = buffer.score;
  auto* acc = buffer.accumulator;
  auto* key_buffer = buffer.kv;
  auto* bias = buffer.bias;
  auto* maximum = buffer.maximum;
  auto* normalizer = buffer.normalizer;
  auto* rescale = buffer.rescale;
  auto* descriptors =
      reinterpret_cast<DmaDescriptor*>(cached->dma_words[slice]);
  const auto* q = static_cast<const uint16_t*>(qhpi_tensor_raw_data(in[1]));
  auto* y = static_cast<uint16_t*>(qhpi_tensor_raw_data(out[0]));
  const auto qq = qhpi_tensor_quant_parameters(in[1]),
             oq = qhpi_tensor_quant_parameters(out[0]);
  *reinterpret_cast<HVX_Vector*>(bias) = Q6_V_vsplat_R(0x3c00);
  *reinterpret_cast<HVX_Vector*>(bias + 32) = Q6_V_vzero();
  alignas(128) uint32_t lanes[32];
  for (unsigned c = 0; c < 32; ++c)
    lanes[c] = c / 2;
  const auto lane = *reinterpret_cast<HVX_Vector*>(lanes);
  bool pending = false;
  for (unsigned task = slice; task < kvheads * 2 * query_tiles;
       task += active) {
    const unsigned group = task / query_tiles, kv = group / 2,
                   first_head = kv * 4 + group % 2 * batches,
                   begin = task % query_tiles * rows,
                   actual = std::min(rows, valid - begin);
    for (unsigned r = 0; r < batches * rows; ++r) {
      maximum[r] = 0;
      normalizer[r] = 0;
    }
    for (unsigned c = 0; c < batches * rows * 256 / 32; ++c)
      reinterpret_cast<HVX_Vector*>(acc)[c] = Q6_V_vzero();
    for (unsigned batch = 0; batch < batches; ++batch) {
      const unsigned head = first_head + batch;
      for (unsigned r = 0; r < rows; ++r) {
        if (r < actual)
          to_half(qa + r * 256, q + (head * qrows + begin + r) * 256, 256, qq);
        else
          for (unsigned c = 0; c < 4; ++c)
            reinterpret_cast<HVX_Vector*>(qa + r * 256)[c] = Q6_V_vzero();
      }
      qhblas_hmx_transfer_row_ahf(
          reinterpret_cast<__fp16*>(qa), packed_q + batch * batch_stride, 256);
    }
    dma_start(
        key_buffer,
        kpack + kv * kv_rows * 256 * 2,
        std::min(256u, padded) * 256 * 2,
        descriptors);
    pending = true;
    for (unsigned start = 0; start < padded; start += 512) {
      const unsigned width = std::min(512u, padded - start);
      if (!dma_wait()) {
        __atomic_store_n(&queue.error, 1u, __ATOMIC_RELEASE);
        goto finished;
      }
      pending = false;
      for (unsigned part = 0; part < width; part += 256) {
        const unsigned piece = std::min(256u, width - part);
        if (part) {
          if (!dma_wait()) {
            __atomic_store_n(&queue.error, 1u, __ATOMIC_RELEASE);
            goto finished;
          }
          pending = false;
        }
        if (!attention_hmx::submit(
                queue,
                slice,
                packed_q,
                key_buffer,
                reinterpret_cast<uint8_t*>(packed_c),
                reinterpret_cast<uint8_t*>(bias),
                32,
                256,
                piece,
                batches,
                batch_stride,
                batch_stride))
          goto finished;
        if (part + piece < width) {
          dma_start(
              key_buffer,
              kpack + (kv * kv_rows + start + part + piece) * 256 * 2,
              std::min(256u, width - part - piece) * 256 * 2,
              descriptors);
          pending = true;
        }
        auto encode = [&](HVX_Vector z) {
          z = qdq(z, meta[0], int(meta[1]));
          z = qdq(hvxMulF32(z, hvxSplatF32(meta[2])), meta[3], int(meta[4]));
          z = hvxAddF32(
              hvxMulF32(z, hvxSplatF32(1.f / meta[5])),
              hvxSplatF32(meta[6] + .5f));
          return Q6_Vw_vmax_VwVw(
              Q6_V_vzero(),
              Q6_Vw_vmin_VwVw(Q6_Vw_equals_Vsf(z), Q6_V_vsplat_R(65535)));
        };
        for (unsigned batch = 0; batch < batches; ++batch) {
          auto* product = packed_c + batch * batch_stride / 2;
          auto* score = codes + batch * rows * 512;
          for (unsigned r = 0; r < actual; r += 2)
            for (unsigned c = 0; c < piece; c += 64) {
              const auto a = Q6_Wsf_vcvt_Vhf(*reinterpret_cast<HVX_Vector*>(
                  product + (c / 32) * 1024 + (r / 2) * 64));
              const auto b = Q6_Wsf_vcvt_Vhf(*reinterpret_cast<HVX_Vector*>(
                  product + (c / 32 + 1) * 1024 + (r / 2) * 64));
              *reinterpret_cast<HVX_Vector*>(score + r * 512 + part + c) =
                  Q6_Vuh_vpack_VwVw_sat(
                      encode(Q6_V_lo_W(b)), encode(Q6_V_lo_W(a)));
              if (r + 1 < actual)
                *reinterpret_cast<HVX_Vector*>(
                    score + (r + 1) * 512 + part + c) =
                    Q6_Vuh_vpack_VwVw_sat(
                        encode(Q6_V_hi_W(b)), encode(Q6_V_hi_W(a)));
            }
        }
      }
      auto load_value = [&](unsigned output_col) {
        unsigned count = 0;
        for (unsigned col = 0; col < 4; ++col)
          for (unsigned part = 0; part < width; part += 256) {
            const unsigned at = start + part,
                           source_width =
                               std::min(256u, padded - (at / 256) * 256);
            const unsigned piece = std::min(256u, width - part);
            auto& d = descriptors[count];
            d.next = descriptors + count + 1;
            d.control = piece * 32 * 2 | (1u << 28) | (1u << 29);
            d.source = vpack +
                (kv * (kv_rows / 256) + at / 256) * 256 * 256 * 2 +
                (output_col / 32 + col) * source_width * 32 * 2;
            d.destination = key_buffer + col * width * 32 * 2 + part * 32 * 2;
            ++count;
          }
        descriptors[count - 1].next = nullptr;
        asm volatile("release(%0):at" ::"r"(descriptors) : "memory");
        asm volatile("dmstart(%0)" ::"r"(descriptors) : "memory");
        pending = true;
      };
      load_value(0);
      for (unsigned batch = 0; batch < batches; ++batch) {
        auto* score = codes + batch * rows * 512;
        for (unsigned r = 0; r < rows; r += 2) {
          const unsigned index = batch * rows + r;
          if (r >= actual) {
            for (unsigned c = 0; c < width; c += 32)
              *reinterpret_cast<HVX_Vector*>(
                  packed_a + batch * probability_stride + (c / 32) * 2048 +
                  (r / 2) * 128) = Q6_V_vzero();
            continue;
          }
          const auto limits = Q6_V_lo_W(Q6_W_vshuff_VVR(
              Q6_V_vsplat_R(r + 1 < actual ? history + begin + r + 2 : 0),
              Q6_V_vsplat_R(history + begin + r + 1),
              -4));
          auto mx = Q6_V_lo_W(Q6_W_vshuff_VVR(
              Q6_V_vsplat_R(maximum[index + 1]),
              Q6_V_vsplat_R(maximum[index]),
              -4));
          auto load_row = [&](unsigned row, unsigned c) {
            if (row >= actual)
              return Q6_V_vzero();
            const auto raw = *reinterpret_cast<const HVX_Vector*>(
                score + row * 512 + (c & ~63u));
            const auto words = Q6_Wuw_vunpack_Vuh(raw);
            return (c & 32) ? Q6_V_hi_W(words) : Q6_V_lo_W(words);
          };
          for (unsigned c = 0; c < width; c += 32) {
            const auto pair =
                Q6_W_vshuff_VVR(load_row(r + 1, c), load_row(r, c), -4);
            auto lo = Q6_V_lo_W(pair), hi = Q6_V_hi_W(pair);
            lo = Q6_V_vmux_QVV(
                Q6_Q_vcmp_gt_VwVw(
                    limits, Q6_Vw_vadd_VwVw(lane, Q6_V_vsplat_R(start + c))),
                lo,
                Q6_V_vzero());
            hi = Q6_V_vmux_QVV(
                Q6_Q_vcmp_gt_VwVw(
                    limits,
                    Q6_Vw_vadd_VwVw(lane, Q6_V_vsplat_R(start + c + 16))),
                hi,
                Q6_V_vzero());
            mx = Q6_Vw_vmax_VwVw(mx, Q6_Vw_vmax_VwVw(lo, hi));
          }
          for (unsigned shift = 64; shift >= 8; shift /= 2)
            mx = Q6_Vw_vmax_VwVw(mx, Q6_V_vror_VR(mx, shift));
          const auto old = Q6_V_lo_W(Q6_W_vshuff_VVR(
              Q6_V_vsplat_R(maximum[index + 1]),
              Q6_V_vsplat_R(maximum[index]),
              -4));
          const auto alpha = expv(hvxMulF32(
              Q6_Vsf_equals_Vw(Q6_Vw_vsub_VwVw(old, mx)),
              hvxSplatF32(meta[5])));
          union {
            uint32_t u;
            float f;
          } a0{uint32_t(Q6_R_vextract_VR(alpha, 0))},
              a1{uint32_t(Q6_R_vextract_VR(alpha, 4))};
          rescale[index] = normalizer[index] > 0 ? a0.f : 0.f;
          rescale[index + 1] = normalizer[index + 1] > 0 ? a1.f : 0.f;
          auto sum = Q6_V_vzero();
          for (unsigned c = 0; c < width; c += 32) {
            const auto pair =
                Q6_W_vshuff_VVR(load_row(r + 1, c), load_row(r, c), -4);
            auto exponential = [&](HVX_Vector code, unsigned offset) {
              const auto visible = Q6_Q_vcmp_gt_VwVw(
                  limits,
                  Q6_Vw_vadd_VwVw(lane, Q6_V_vsplat_R(start + c + offset)));
              code = Q6_V_vmux_QVV(visible, code, mx);
              const auto e = expv(hvxMulF32(
                  Q6_Vsf_equals_Vw(Q6_Vw_vsub_VwVw(code, mx)),
                  hvxSplatF32(meta[5])));
              return Q6_V_vmux_QVV(visible, e, Q6_V_vzero());
            };
            const auto lo = exponential(Q6_V_lo_W(pair), 0),
                       hi = exponential(Q6_V_hi_W(pair), 16);
            sum = hvxAddF32(sum, hvxAddF32(lo, hi));
            *reinterpret_cast<HVX_Vector*>(
                packed_a + batch * probability_stride + (c / 32) * 2048 +
                (r / 2) * 128) = Q6_Vh_vdeal_Vh(Q6_Vhf_vcvt_VsfVsf(lo, hi));
          }
          for (unsigned shift = 64; shift >= 8; shift /= 2)
            sum = hvxAddF32(sum, Q6_V_vror_VR(sum, shift));
          union {
            uint32_t u;
            float f;
          } s0{uint32_t(Q6_R_vextract_VR(sum, 0))},
              s1{uint32_t(Q6_R_vextract_VR(sum, 4))};
          normalizer[index] = normalizer[index] * rescale[index] + s0.f;
          normalizer[index + 1] =
              normalizer[index + 1] * rescale[index + 1] + s1.f;
          maximum[index] = Q6_R_vextract_VR(mx, 0);
          maximum[index + 1] = Q6_R_vextract_VR(mx, 4);
        }
      }
      for (unsigned output_col = 0; output_col < 256; output_col += 128) {
        if (!dma_wait()) {
          __atomic_store_n(&queue.error, 1u, __ATOMIC_RELEASE);
          goto finished;
        }
        pending = false;
        if (!attention_hmx::submit(
                queue,
                slice,
                packed_a,
                key_buffer,
                reinterpret_cast<uint8_t*>(packed_c),
                reinterpret_cast<uint8_t*>(bias),
                32,
                width,
                128,
                batches,
                probability_stride,
                batch_stride))
          goto finished;
        if (!output_col)
          load_value(128);
        else if (start + width < padded) {
          const unsigned next_width = std::min(256u, padded - start - width);
          dma_start(
              key_buffer,
              kpack + (kv * kv_rows + start + width) * 256 * 2,
              next_width * 256 * 2,
              descriptors);
          pending = true;
        }
        for (unsigned batch = 0; batch < batches; ++batch)
          for (unsigned r = 0; r < actual; r += 2) {
            const unsigned index = batch * rows + r;
            const float f0 = rescale[index], f1 = rescale[index + 1];
            const auto a0 = hvxSplatF32(f0), a1 = hvxSplatF32(f1);
            for (unsigned c = 0; c < 128; c += 32) {
              const auto values =
                  Q6_Wsf_vcvt_Vhf(*reinterpret_cast<HVX_Vector*>(
                      packed_c + batch * batch_stride / 2 + (c / 32) * 1024 +
                      (r / 2) * 64));
              auto* p0 = reinterpret_cast<HVX_Vector*>(
                  acc + index * 256 + output_col + c);
              auto* p1 = reinterpret_cast<HVX_Vector*>(
                  acc + (index + 1) * 256 + output_col + c);
              *p0 = f0 == 0.f ? Q6_V_lo_W(values)
                  : f0 == 1.f
                  ? hvxAddF32(*p0, Q6_V_lo_W(values))
                  : hvxAddF32(hvxMulF32(*p0, a0), Q6_V_lo_W(values));
              if (r + 1 < actual)
                *p1 = f1 == 0.f ? Q6_V_hi_W(values)
                    : f1 == 1.f
                    ? hvxAddF32(*p1, Q6_V_hi_W(values))
                    : hvxAddF32(hvxMulF32(*p1, a1), Q6_V_hi_W(values));
            }
          }
      }
    }
    for (unsigned batch = 0; batch < batches; ++batch)
      for (unsigned r = 0; r < actual; ++r)
        for (unsigned c = 0; c < 256; c += 64) {
          const unsigned index = batch * rows + r, head = first_head + batch;
          const auto inverse =
              hvxSplatF32(1.f / (normalizer[index] * oq.stepsize));
          const auto a = hvxAddF32(
              hvxMulF32(
                  *reinterpret_cast<HVX_Vector*>(acc + index * 256 + c),
                  inverse),
              hvxSplatF32(oq.zero_offset + .5f));
          const auto b = hvxAddF32(
              hvxMulF32(
                  *reinterpret_cast<HVX_Vector*>(acc + index * 256 + c + 32),
                  inverse),
              hvxSplatF32(oq.zero_offset + .5f));
          *reinterpret_cast<HVX_Vector*>(
              y + (head * qrows + begin + r) * 256 + c) =
              Q6_Vuh_vpack_VwVw_sat(Q6_Vw_equals_Vsf(b), Q6_Vw_equals_Vsf(a));
        }
  }
  for (unsigned head = slice; head < kvheads * 4; head += active)
    for (unsigned r = valid; r < qrows; ++r)
      for (unsigned c = 0; c < 4; ++c)
        reinterpret_cast<HVX_Vector*>(y + (head * qrows + r) * 256)[c] =
            Q6_Vh_vsplat_R(oq.zero_offset);
finished:
  if (pending && !dma_wait())
    __atomic_store_n(&queue.error, 1u, __ATOMIC_RELEASE);
  __atomic_store_n(&queue.jobs[slice].finished, 1u, __ATOMIC_RELEASE);
  return QHPI_SUCCESS;
}
