// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

static uint32_t online_prefill(
    QHPI_RuntimeHandle* handle,
    QHPI_Tensor** out,
    const QHPI_Tensor* const* in) {
  const unsigned slice = qhpi_slice_number(handle),
                 workers = qhpi_num_slices(handle);
  const unsigned qheads = qhpi_tensor_shape(in[1]).dims[1],
                 kvheads = qhpi_tensor_shape(in[3]).dims[1],
                 kv_bytes = kvheads * kv_rows * 256 * 2;
  auto* work = static_cast<uint8_t*>(qhpi_tensor_raw_data(in[0]));
  auto& queue = *reinterpret_cast<attention_hmx::Queue*>(work);

  const auto* l = static_cast<const int32_t*>(qhpi_tensor_raw_data(in[6]));
  const auto* meta = static_cast<const float*>(qhpi_tensor_raw_data(in[7]));
  const unsigned valid = l[0], history = l[2], keys = history + valid,
                 padded = (keys + 63) & ~63u;
  const unsigned qrows = qhpi_tensor_shape(in[1]).dims[2];
  if (qrows == 1 && slice >= kvheads) {
    __atomic_store_n(&queue.jobs[slice].finished, 1u, __ATOMIC_RELEASE);
    return QHPI_SUCCESS;
  }
  auto* base = work + worker_start + slice * per_worker;
  auto* cached = packed_cache(
      static_cast<const int32_t*>(qhpi_tensor_raw_data(in[2]))[0], kv_bytes);
  const auto* kpack = cached->packed;
  const auto* vpack = kpack + kv_bytes;
  auto* qa = reinterpret_cast<uint16_t*>(base);
  auto* packed_a = base + 16384;
  auto* packed_c = reinterpret_cast<uint16_t*>(base + 32768);
  auto* acc = reinterpret_cast<float*>(base + 65536);
  auto* bias = reinterpret_cast<uint32_t*>(base + 98304);
  auto* maximum = reinterpret_cast<uint32_t*>(base + 98560);
  auto* normalizer = reinterpret_cast<float*>(base + 98688);
  auto* rescale = reinterpret_cast<float*>(base + 98816);
  auto* key_buffer = base + 100352;
  auto* descriptors =
      reinterpret_cast<DmaDescriptor*>(cached->dma_words[slice]);
  const auto* q = static_cast<const uint16_t*>(qhpi_tensor_raw_data(in[1]));
  auto* y = static_cast<uint16_t*>(qhpi_tensor_raw_data(out[0]));
  const auto qq = qhpi_tensor_quant_parameters(in[1]),
             oq = qhpi_tensor_quant_parameters(out[0]);
  *reinterpret_cast<HVX_Vector*>(bias) = Q6_V_vsplat_R(0x3c00);
  *reinterpret_cast<HVX_Vector*>(bias + 32) = Q6_V_vzero();
  bool pending = false;

  for (unsigned head = slice; head < (qrows == 1 ? kvheads : qheads);
       head += workers) {
    const unsigned kv = qrows == 1 ? head : head / 8;
    const unsigned queries = qrows == 1 ? 8u : valid;
    for (unsigned begin = 0; begin < queries; begin += rows) {
      const unsigned actual = std::min(rows, queries - begin);
      for (unsigned r = 0; r < (qrows == 1 ? 8u : rows); ++r) {
        maximum[r] = 0;
        normalizer[r] = 0;
        if (r < actual)
          to_half(
              qa + r * 256,
              q + (qrows == 1 ? head * 8 + r : head * qrows + begin + r) * 256,
              256,
              qq);
        else
          for (unsigned c = 0; c < 4; ++c)
            reinterpret_cast<HVX_Vector*>(qa + r * 256)[c] = Q6_V_vzero();
      }
      for (unsigned c = 0; c < actual * 256 / 32; ++c)
        reinterpret_cast<HVX_Vector*>(acc)[c] = Q6_V_vzero();
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

        for (unsigned r = 0; r < (qrows == 1 ? 8u : rows); ++r) {
          if (r < actual)
            to_half(
                qa + r * 256,
                q +
                    (qrows == 1 ? head * 8 + r : head * qrows + begin + r) *
                        256,
                256,
                qq);
          else
            for (unsigned c = 0; c < 4; ++c)
              reinterpret_cast<HVX_Vector*>(qa + r * 256)[c] = Q6_V_vzero();
        }
        if (qrows == 1)
          pack_decode_a(qa, packed_a, 256);
        else
          qhblas_hmx_transfer_row_ahf(
              reinterpret_cast<__fp16*>(qa), packed_a, 256);
        auto encode = [&](HVX_Vector z) {
          z = qdq(z, meta[0], int(meta[1]));
          z = qdq(hvxMulF32(z, hvxSplatF32(meta[2])), meta[3], int(meta[4]));
          z = hvxAddF32(
              hvxMulF32(z, hvxSplatF32(1.f / meta[5])),
              hvxSplatF32(meta[6] + .5f));
          auto code = Q6_Vw_equals_Vsf(z);
          return Q6_Vw_vmax_VwVw(
              Q6_V_vzero(), Q6_Vw_vmin_VwVw(code, Q6_V_vsplat_R(65535)));
        };
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
                  packed_a,
                  key_buffer,
                  reinterpret_cast<uint8_t*>(packed_c),
                  reinterpret_cast<uint8_t*>(bias),
                  qrows == 1 ? 8 : 32,
                  256,
                  piece))
            goto finished;

          if (part + piece < width) {
            const unsigned next = start + part + piece,
                           next_width = std::min(256u, width - part - piece);
            dma_start(
                key_buffer,
                kpack + (kv * kv_rows + next) * 256 * 2,
                next_width * 256 * 2,
                descriptors);
            pending = true;
          }

          auto* codes = reinterpret_cast<uint16_t*>(base + (part ? 49152 : 0));
          for (unsigned r = 0; r < actual; r += 2)
            for (unsigned c = 0; c < piece; c += 64) {
              auto a = Q6_Wsf_vcvt_Vhf(*reinterpret_cast<const HVX_Vector*>(
                  packed_c + (c / 32) * 1024 + (r / 2) * 64));
              auto b = Q6_Wsf_vcvt_Vhf(*reinterpret_cast<const HVX_Vector*>(
                  packed_c + (c / 32 + 1) * 1024 + (r / 2) * 64));
              *reinterpret_cast<HVX_Vector*>(codes + r * 256 + c) =
                  Q6_Vuh_vpack_VwVw_sat(
                      encode(Q6_V_lo_W(b)), encode(Q6_V_lo_W(a)));
              if (r + 1 < actual)
                *reinterpret_cast<HVX_Vector*>(codes + (r + 1) * 256 + c) =
                    Q6_Vuh_vpack_VwVw_sat(
                        encode(Q6_V_hi_W(b)), encode(Q6_V_hi_W(a)));
            }
        }
        auto load_value = [&](unsigned output_col) {
          unsigned ndesc = 0;
          for (unsigned n = 0; n < 4; ++n)
            for (unsigned part = 0; part < width; part += 256) {
              const unsigned logical = start + part,
                             source_tile = logical / 256,
                             source_width =
                                 std::min(256u, padded - source_tile * 256),
                             piece = std::min(256u, width - part);
              auto& d = descriptors[ndesc];
              d.next = descriptors + ndesc + 1;
              d.control = piece * 32 * 2 | (1u << 28) | (1u << 29);
              d.source = vpack +
                  (kv * (kv_rows / 256) + source_tile) * 256 * 256 * 2 +
                  (output_col / 32 + n) * source_width * 32 * 2;
              d.destination = key_buffer + n * width * 32 * 2 + part * 32 * 2;
              ++ndesc;
            }
          descriptors[ndesc - 1].next = nullptr;
          asm volatile("release(%0):at" ::"r"(descriptors) : "memory");
          asm volatile("dmstart(%0)" ::"r"(descriptors) : "memory");
          pending = true;
        };
        load_value(0);

        alignas(128) uint32_t lanes[32];
        for (unsigned c = 0; c < 32; ++c)
          lanes[c] = c / 2;
        const auto lane = *reinterpret_cast<const HVX_Vector*>(lanes);
        for (unsigned r = 0; r < (qrows == 1 ? 8u : rows); r += 2) {
          if (r >= actual) {
            for (unsigned c = 0; c < width; c += 32)
              *reinterpret_cast<HVX_Vector*>(
                  packed_a + (c / 32) * 2048 + (r / 2) * 128) = Q6_V_vzero();
            continue;
          }
          const auto limit0 = Q6_V_vsplat_R(
                         history + (qrows == 1 ? 0 : begin + r) + 1),
                     limit1 = Q6_V_vsplat_R(
                         r + 1 < actual
                             ? history + (qrows == 1 ? 0 : begin + r + 1) + 1
                             : 0);
          const auto limits = Q6_V_lo_W(Q6_W_vshuff_VVR(limit1, limit0, -4));
          auto mx = Q6_V_lo_W(Q6_W_vshuff_VVR(
              Q6_V_vsplat_R(maximum[r + 1]), Q6_V_vsplat_R(maximum[r]), -4));
          auto load_row = [&](unsigned row, unsigned c) {
            if (row >= actual)
              return Q6_V_vzero();
            const auto* codes = reinterpret_cast<const uint16_t*>(
                base + (c >= 256 ? 49152 : 0));
            const auto raw = *reinterpret_cast<const HVX_Vector*>(
                codes + row * 256 + ((c % 256) & ~63u));
            const auto words = Q6_Wuw_vunpack_Vuh(raw);
            return (c & 32) ? Q6_V_hi_W(words) : Q6_V_lo_W(words);
          };
          auto load_pair = [&](unsigned c) {
            return Q6_W_vshuff_VVR(load_row(r + 1, c), load_row(r, c), -4);
          };
          for (unsigned c = 0; c < width; c += 32) {
            const auto pair = load_pair(c);
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
              Q6_V_vsplat_R(maximum[r + 1]), Q6_V_vsplat_R(maximum[r]), -4));
          auto alpha = expv(hvxMulF32(
              Q6_Vsf_equals_Vw(Q6_Vw_vsub_VwVw(old, mx)),
              hvxSplatF32(meta[5])));
          union {
            uint32_t u;
            float f;
          } a0{uint32_t(Q6_R_vextract_VR(alpha, 0))},
              a1{uint32_t(Q6_R_vextract_VR(alpha, 4))};
          rescale[r] = normalizer[r] > 0 ? a0.f : 0.f;
          rescale[r + 1] = normalizer[r + 1] > 0 ? a1.f : 0.f;
          auto sum = Q6_V_vzero();
          for (unsigned c = 0; c < width; c += 32) {
            const auto pair = load_pair(c);
            auto exponential = [&](HVX_Vector code, unsigned offset) {
              auto visible = Q6_Q_vcmp_gt_VwVw(
                  limits,
                  Q6_Vw_vadd_VwVw(lane, Q6_V_vsplat_R(start + c + offset)));
              code = Q6_V_vmux_QVV(visible, code, mx);
              auto e = expv(hvxMulF32(
                  Q6_Vsf_equals_Vw(Q6_Vw_vsub_VwVw(code, mx)),
                  hvxSplatF32(meta[5])));
              return Q6_V_vmux_QVV(visible, e, Q6_V_vzero());
            };
            const auto lo = exponential(Q6_V_lo_W(pair), 0),
                       hi = exponential(Q6_V_hi_W(pair), 16);
            sum = hvxAddF32(sum, hvxAddF32(lo, hi));
            *reinterpret_cast<HVX_Vector*>(
                packed_a + (c / 32) * 2048 + (r / 2) * 128) =
                Q6_Vh_vdeal_Vh(Q6_Vhf_vcvt_VsfVsf(lo, hi));
          }
          for (unsigned shift = 64; shift >= 8; shift /= 2)
            sum = hvxAddF32(sum, Q6_V_vror_VR(sum, shift));
          union {
            uint32_t u;
            float f;
          } s0{uint32_t(Q6_R_vextract_VR(sum, 0))},
              s1{uint32_t(Q6_R_vextract_VR(sum, 4))};
          normalizer[r] = normalizer[r] * rescale[r] + s0.f;
          normalizer[r + 1] = normalizer[r + 1] * rescale[r + 1] + s1.f;
          maximum[r] = Q6_R_vextract_VR(mx, 0);
          maximum[r + 1] = Q6_R_vextract_VR(mx, 4);
        }

        for (unsigned output_col = 0; output_col < 256; output_col += 128) {
          if (!dma_wait()) {
            __atomic_store_n(&queue.error, 1u, __ATOMIC_RELEASE);
            goto finished;
          }
          pending = false;

          auto* current_buffer = key_buffer;

          if (!attention_hmx::submit(
                  queue,
                  slice,
                  packed_a,
                  current_buffer,
                  base,
                  reinterpret_cast<uint8_t*>(bias),
                  qrows == 1 ? 8 : 32,
                  width,
                  128))
            goto finished;

          if (!output_col)
            load_value(128);
          else if (start + width < padded) {
            const unsigned next = start + width,
                           next_width = std::min(256u, padded - next);
            dma_start(
                key_buffer,
                kpack + (kv * kv_rows + next) * 256 * 2,
                next_width * 256 * 2,
                descriptors);
            pending = true;
          }

          if (qrows == 1) {
            for (unsigned r = 0; r < actual; r += 2) {
              const float f0 = rescale[r], f1 = rescale[r + 1];
              const auto a0 = hvxSplatF32(f0), a1 = hvxSplatF32(f1);
              for (unsigned c = 0; c < 128; c += 32) {
                const auto values =
                    Q6_Wsf_vcvt_Vhf(*reinterpret_cast<const HVX_Vector*>(
                        base + ((c / 32) * 1024 + (r / 2) * 64) * 2));
                auto* p0 = reinterpret_cast<HVX_Vector*>(
                    acc + r * 256 + output_col + c);
                auto* p1 = reinterpret_cast<HVX_Vector*>(
                    acc + (r + 1) * 256 + output_col + c);
                *p0 = f0 == 0.f ? Q6_V_lo_W(values)
                    : f0 == 1.f
                    ? hvxAddF32(*p0, Q6_V_lo_W(values))
                    : hvxAddF32(hvxMulF32(*p0, a0), Q6_V_lo_W(values));
                *p1 = f1 == 0.f ? Q6_V_hi_W(values)
                    : f1 == 1.f
                    ? hvxAddF32(*p1, Q6_V_hi_W(values))
                    : hvxAddF32(hvxMulF32(*p1, a1), Q6_V_hi_W(values));
              }
            }
          } else {
            for (unsigned r = 0; r < actual; ++r) {
              alignas(128) float values[128];
              unpack_row(values, reinterpret_cast<uint16_t*>(base), r, 128);
              const float factor = rescale[r];
              const auto alpha = hvxSplatF32(factor);
              if (factor == 0.f) {
                for (unsigned c = 0; c < 128; c += 32)
                  *reinterpret_cast<HVX_Vector*>(
                      acc + r * 256 + output_col + c) =
                      *reinterpret_cast<const HVX_Vector*>(values + c);
              } else if (factor == 1.f) {
                for (unsigned c = 0; c < 128; c += 32) {
                  auto* o = reinterpret_cast<HVX_Vector*>(
                      acc + r * 256 + output_col + c);
                  *o = hvxAddF32(
                      *o, *reinterpret_cast<const HVX_Vector*>(values + c));
                }
              } else {
                for (unsigned c = 0; c < 128; c += 32) {
                  auto* o = reinterpret_cast<HVX_Vector*>(
                      acc + r * 256 + output_col + c);
                  *o = hvxAddF32(
                      hvxMulF32(*o, alpha),
                      *reinterpret_cast<const HVX_Vector*>(values + c));
                }
              }
            }
          }
        }
      }

      for (unsigned r = 0; r < actual; ++r)
        for (unsigned c = 0; c < 256; c += 64) {
          const auto inverse = hvxSplatF32(1.f / (normalizer[r] * oq.stepsize));
          auto a = hvxAddF32(
              hvxMulF32(
                  *reinterpret_cast<HVX_Vector*>(acc + r * 256 + c), inverse),
              hvxSplatF32(oq.zero_offset + .5f));
          auto b = hvxAddF32(
              hvxMulF32(
                  *reinterpret_cast<HVX_Vector*>(acc + r * 256 + c + 32),
                  inverse),
              hvxSplatF32(oq.zero_offset + .5f));
          *reinterpret_cast<HVX_Vector*>(
              y + (qrows == 1 ? head * 8 + r : head * qrows + begin + r) * 256 +
              c) =
              Q6_Vuh_vpack_VwVw_sat(Q6_Vw_equals_Vsf(b), Q6_Vw_equals_Vsf(a));
        }
    }
    for (unsigned r = valid; r < qrows; ++r)
      for (unsigned c = 0; c < 4; ++c)
        reinterpret_cast<HVX_Vector*>(y + (head * qrows + r) * 256)[c] =
            Q6_Vh_vsplat_R(oq.zero_offset);
  }
finished:
  if (pending && !dma_wait())
    __atomic_store_n(&queue.error, 1u, __ATOMIC_RELEASE);
  __atomic_store_n(&queue.jobs[slice].finished, 1u, __ATOMIC_RELEASE);
  return QHPI_SUCCESS;
}
