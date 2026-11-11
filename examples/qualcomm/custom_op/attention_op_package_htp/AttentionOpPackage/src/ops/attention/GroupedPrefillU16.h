// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

static uint32_t grouped_prefill_u16(
    QHPI_RuntimeHandle* handle,
    QHPI_Tensor** out,
    const QHPI_Tensor* const* in,
    unsigned kv_rows) {
  constexpr unsigned batches = 2, batch_stride = rows * 256 * 2,
                     probability_stride = rows * 1024 * 2;
  const unsigned slice = qhpi_slice_number(handle),
                 active = std::min(8u, unsigned(qhpi_num_slices(handle)));
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
  const int raw_zero = int(meta[1]), logits_zero = int(meta[6]),
            key_zero = int(meta[10]), value_zero = int(meta[12]);
  const unsigned valid = lengths[0], history = lengths[2],
                 padded = (history + valid + 31) & ~31u,
                 query_tiles = (valid + rows - 1) / rows;
  auto* cached = packed_cache(
      static_cast<const int32_t*>(qhpi_tensor_raw_data(in[2]))[0], kv_bytes);
  const PackedKv8 packed(cached->packed, kvheads, kv_rows);
  auto& buffer = *reinterpret_cast<GroupedScratchU16*>(
      work + worker_start + slice * sizeof(GroupedScratchU16));
  // Query packing finishes before HMX reuses the product buffer.
  auto* qa = buffer.product;
  auto* packed_q = buffer.query;
  auto* packed_a = buffer.probability;
  auto* packed_c = buffer.product;

  auto* acc = buffer.accumulator;
  // Reuse each KV panel across every query tile and head group for one KV head.
  const unsigned tasks_per_kv = 2 * query_tiles;
  const bool cooperative = active == 8 && tasks_per_kv <= active;
  const unsigned readers = cooperative ? tasks_per_kv : active;
  const unsigned first_task =
      cooperative && slice >= readers ? kvheads * tasks_per_kv : slice;
  auto* key_buffer = buffer.kv;
  unsigned issued = 0;
  auto acquire = [&]() {
    if (cooperative)
      key_buffer = attention_stream::buffer(work, ++issued);
    return true;
  };
  auto start_dma = [&](uint8_t* destination,
                       const uint8_t* source,
                       unsigned bytes,
                       DmaDescriptor* descriptors) {
    if (cooperative) {
      acquire();
      return;
    }
    dma_start(destination, source, bytes, descriptors);
  };
  auto wait_ready = [&]() { return cooperative || dma_wait(); };
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
  const auto raw_scale = attention_fixed::Scale::ratio(
      uint64_t(
          std::llround(double(qq.stepsize) * meta[9] / meta[0] * (1ULL << 40))),
      1ULL << 40);
  const auto logits_scale = attention_fixed::Scale::ratio(
      uint64_t(
          std::llround(double(meta[0]) * meta[2] / meta[5] * (1ULL << 40))),
      1ULL << 40);
  const double output_ratio_value = double(meta[11]) / oq.stepsize * 256 * 4096;
  const bool integer_output_ratio =
      output_ratio_value >= .5 && output_ratio_value <= (1ULL << 48);
  const uint64_t output_ratio =
      integer_output_ratio ? uint64_t(std::llround(output_ratio_value)) : 0;
  *reinterpret_cast<HVX_Vector*>(bias) = Q6_V_vsplat_R(0x4400);
  *reinterpret_cast<HVX_Vector*>(bias + 32) = Q6_V_vzero();
  *reinterpret_cast<HVX_Vector*>(bias + 64) = Q6_V_vsplat_R(0x0400);
  *reinterpret_cast<HVX_Vector*>(bias + 96) = Q6_V_vzero();
  const auto* key_sums = packed.key_sums;
  alignas(128) uint32_t lanes[32];
  for (unsigned c = 0; c < 32; ++c)
    lanes[c] = c / 2;
  const auto lane = *reinterpret_cast<HVX_Vector*>(lanes);
  const auto exp_parameters = attention_exp::Parameters::make(meta[5]);
  const unsigned exp_coefficient = exp_parameters.coefficient,
                 exp_bits = exp_parameters.bits;
  const auto exp_table =
      *reinterpret_cast<const HVX_Vector*>(attention_exp::powers);
  const auto exp_slope_table =
      *reinterpret_cast<const HVX_Vector*>(attention_exp::slopes);
  bool pending = false;
  for (unsigned task = first_task; task < kvheads * 2 * query_tiles;
       task += cooperative ? tasks_per_kv : active) {
    const unsigned group = task / query_tiles, kv = group / 2,
                   first_head = kv * 4 + group % 2 * batches,
                   begin = task % query_tiles * rows,
                   actual = std::min(rows, valid - begin);
    // Skip fully invisible N32 tiles; masking remains inside the diagonal tile.
    const unsigned query_end = (history + begin + actual + 31) & ~31u;
    for (unsigned r = 0; r < batches * rows; ++r) {
      maximum[r] = 0;
      buffer.query_sum[r] = 0;
      buffer.probability_sum[r] = 0;
      normalizer[r] = 0;
    }
    for (unsigned batch = 0; batch < batches; ++batch) {
      const unsigned head = first_head + batch;
      for (unsigned r = 0; r < rows; ++r) {
        if (r < actual) {
          auto sum = Q6_V_vzero();
          for (unsigned c = 0; c < 256; c += 64) {
            auto raw = *reinterpret_cast<const HVX_Vector*>(
                q + (head * qrows + begin + r) * 256 + c);
            *reinterpret_cast<HVX_Vector*>(qa + r * 256 + c) = raw;
            auto unpack = Q6_Wuw_vunpack_Vuh(raw);
            sum = Q6_Vw_vadd_VwVw(
                sum, Q6_Vw_vadd_VwVw(Q6_V_lo_W(unpack), Q6_V_hi_W(unpack)));
          }
          for (unsigned shift = 64; shift >= 4; shift /= 2)
            sum = Q6_Vw_vadd_VwVw(sum, Q6_V_vror_VR(sum, shift));
          buffer.query_sum[batch * rows + r] =
              Q6_R_vextract_VR(sum, 0) - qq.zero_offset * 256;
        } else
          for (unsigned c = 0; c < 4; ++c)
            reinterpret_cast<HVX_Vector*>(qa + r * 256)[c] = Q6_V_vzero();
      }
      attention_fixed::pack_query(qa, packed_q + batch * batch_stride);
    }
    start_dma(
        key_buffer,
        packed.key_panel(kv, 0),
        std::min(256u, padded) * 256,
        descriptors);
    pending = !cooperative;
    for (unsigned start = 0; start < padded; start += 1024) {
      const unsigned width = std::min(1024u, padded - start);
      const unsigned local_width =
          start < query_end ? std::min(width, query_end - start) : 0;
      if (!wait_ready()) {
        __atomic_store_n(&queue.error, 1u, __ATOMIC_RELEASE);
        goto finished;
      }
      pending = false;
      for (unsigned part = 0; part < width; part += 256) {
        const unsigned piece = std::min(256u, width - part);
        const unsigned local_piece =
            part < local_width ? std::min(piece, local_width - part) : 0;
        if (part) {
          if (!wait_ready()) {
            __atomic_store_n(&queue.error, 1u, __ATOMIC_RELEASE);
            goto finished;
          }
          pending = false;
        }
        auto* high_qk = reinterpret_cast<uint16_t*>(buffer.kv + 65536);
        auto* qk_bias = reinterpret_cast<uint32_t*>(buffer.kv + 98304);
        for (unsigned c = 0; c < local_piece; c += 64) {
          auto sums = Q6_Wuw_vunpack_Vuh(*reinterpret_cast<const HVX_Vector*>(
              key_sums + kv * kv_rows + start + part + c));
          for (unsigned half = 0; half < 2; ++half) {
            auto wsum = Q6_Vw_vsub_VwVw(
                half ? Q6_V_hi_W(sums) : Q6_V_lo_W(sums),
                Q6_V_vsplat_R(128 * 256));
            auto correction = Q6_Vw_vmpyi_VwRh(
                wsum,
                Q6_R_combine_RlRl(
                    -(qq.zero_offset >> 8), -(qq.zero_offset >> 8)));
            auto* p = qk_bias + (c / 32 + half) * 128;
            *reinterpret_cast<HVX_Vector*>(p) = Q6_V_vsplat_R(0x4400);
            *reinterpret_cast<HVX_Vector*>(p + 32) = correction;
            *reinterpret_cast<HVX_Vector*>(p + 64) = Q6_V_vsplat_R(0x0400);
            *reinterpret_cast<HVX_Vector*>(p + 96) = correction;
          }
        }
        if (local_piece &&
            !attention_hmx::submit(
                queue,
                slice,
                packed_q,
                key_buffer,
                reinterpret_cast<uint8_t*>(packed_c),
                reinterpret_cast<uint8_t*>(qk_bias),
                32,
                256,
                local_piece,
                batches,
                batch_stride,
                batch_stride,
                reinterpret_cast<uint8_t*>(high_qk),
                512,
                0,
                cooperative ? issued : 0))
          goto finished;
        if (part + piece < width) {
          start_dma(
              key_buffer,
              packed.key_panel(kv, start + part + piece),
              std::min(256u, width - part - piece) * 256,
              descriptors);
          pending = !cooperative;
        }
        // Keep the full-vector loop free of per-vector tail branches.
        if (!(local_piece & 63u)) {
          for (unsigned batch = 0; batch < batches; ++batch) {
            for (unsigned r = 0; r < actual; r += 2) {
              const int row_correction[2] = {
                  (128 - key_zero) * buffer.query_sum[batch * rows + r],
                  (128 - key_zero) * buffer.query_sum[batch * rows + r + 1]};
              for (unsigned c = 0; c < local_piece; c += 64) {
                auto sums =
                    Q6_Wuw_vunpack_Vuh(*reinterpret_cast<const HVX_Vector*>(
                        key_sums + kv * kv_rows + start + part + c));
                auto convert = [&](unsigned col, bool second, unsigned half) {
                  const unsigned at = batch * batch_stride / 2 +
                      (col / 32) * 1024 + (r / 2) * 64;
                  auto l = *reinterpret_cast<HVX_Vector*>(packed_c + at);
                  auto h = *reinterpret_cast<HVX_Vector*>(high_qk + at);
                  auto t = attention_fixed::unpack_result(l, h, second);
                  auto ws = Q6_Vw_vsub_VwVw(
                      half ? Q6_V_hi_W(sums) : Q6_V_lo_W(sums),
                      Q6_V_vsplat_R(128 * 256));
                  auto correction = Q6_Vw_vmpyi_VwRh(
                      ws,
                      Q6_R_combine_RlRl(
                          -(qq.zero_offset & 255), -(qq.zero_offset & 255)));
                  auto dot = Q6_Vw_vadd_VwVw(
                      Q6_Vw_vadd_VwVw(Q6_Vw_vasl_VwR(t, 8), correction),
                      Q6_V_vsplat_R(row_correction[unsigned(second)]));
                  auto raw_code =
                      attention_fixed::affine_u16(dot, raw_scale, raw_zero);
                  return attention_fixed::affine_u16(
                      Q6_Vw_vsub_VwVw(raw_code, Q6_V_vsplat_R(raw_zero)),
                      logits_scale,
                      logits_zero);
                };
                auto a0 = convert(c, false, 0), a1 = convert(c, true, 0),
                     b0 = convert(c + 32, false, 1),
                     b1 = convert(c + 32, true, 1);
                auto first = Q6_W_vshuff_VVR(a1, a0, -4),
                     second = Q6_W_vshuff_VVR(b1, b0, -4);
                *reinterpret_cast<HVX_Vector*>(
                    packed_a + batch * probability_stride +
                    ((part + c) / 32) * 2048 + (r / 2) * 128) =
                    Q6_Vuh_vpack_VwVw_sat(Q6_V_hi_W(first), Q6_V_lo_W(first));
                *reinterpret_cast<HVX_Vector*>(
                    packed_a + batch * probability_stride +
                    ((part + c + 32) / 32) * 2048 + (r / 2) * 128) =
                    Q6_Vuh_vpack_VwVw_sat(Q6_V_hi_W(second), Q6_V_lo_W(second));
              }
            }
          }
        } else {
          for (unsigned batch = 0; batch < batches; ++batch) {
            for (unsigned r = 0; r < actual; r += 2) {
              const int row_correction[2] = {
                  (128 - key_zero) * buffer.query_sum[batch * rows + r],
                  (128 - key_zero) * buffer.query_sum[batch * rows + r + 1]};
              for (unsigned c = 0; c < local_piece; c += 64) {
                auto sums =
                    Q6_Wuw_vunpack_Vuh(*reinterpret_cast<const HVX_Vector*>(
                        key_sums + kv * kv_rows + start + part + c));
                auto convert = [&](unsigned col, bool second, unsigned half) {
                  const unsigned at = batch * batch_stride / 2 +
                      (col / 32) * 1024 + (r / 2) * 64;
                  auto l = *reinterpret_cast<HVX_Vector*>(packed_c + at);
                  auto h = *reinterpret_cast<HVX_Vector*>(high_qk + at);
                  auto t = attention_fixed::unpack_result(l, h, second);
                  auto ws = Q6_Vw_vsub_VwVw(
                      half ? Q6_V_hi_W(sums) : Q6_V_lo_W(sums),
                      Q6_V_vsplat_R(128 * 256));
                  auto correction = Q6_Vw_vmpyi_VwRh(
                      ws,
                      Q6_R_combine_RlRl(
                          -(qq.zero_offset & 255), -(qq.zero_offset & 255)));
                  auto dot = Q6_Vw_vadd_VwVw(
                      Q6_Vw_vadd_VwVw(Q6_Vw_vasl_VwR(t, 8), correction),
                      Q6_V_vsplat_R(row_correction[unsigned(second)]));
                  auto raw_code =
                      attention_fixed::affine_u16(dot, raw_scale, raw_zero);
                  return attention_fixed::affine_u16(
                      Q6_Vw_vsub_VwVw(raw_code, Q6_V_vsplat_R(raw_zero)),
                      logits_scale,
                      logits_zero);
                };
                auto a0 = convert(c, false, 0), a1 = convert(c, true, 0);
                auto pair = Q6_W_vshuff_VVR(a1, a0, -4);
                *reinterpret_cast<HVX_Vector*>(
                    packed_a + batch * probability_stride +
                    ((part + c) / 32) * 2048 + (r / 2) * 128) =
                    Q6_Vuh_vpack_VwVw_sat(Q6_V_hi_W(pair), Q6_V_lo_W(pair));
                if (c + 32 < local_piece) {
                  auto b0 = convert(c + 32, false, 1),
                       b1 = convert(c + 32, true, 1);
                  auto second = Q6_W_vshuff_VVR(b1, b0, -4);
                  *reinterpret_cast<HVX_Vector*>(
                      packed_a + batch * probability_stride +
                      ((part + c + 32) / 32) * 2048 + (r / 2) * 128) =
                      Q6_Vuh_vpack_VwVw_sat(
                          Q6_V_hi_W(second), Q6_V_lo_W(second));
                }
              }
            }
          }
        }
      }
      auto load_value = [&](unsigned output_col) {
        if (!acquire()) {
          __atomic_store_n(&queue.error, 1u, __ATOMIC_RELEASE);
          return;
        }
        if (cooperative) {
          pending = false;
          return;
        }
        unsigned count = 0;
        for (unsigned col = 0; col < 2; ++col)
          for (unsigned part = 0; part < width; part += 256) {
            const unsigned at = start + part;
            const unsigned piece = std::min(256u, width - part);
            auto& d = descriptors[count];
            d.next = descriptors + count + 1;
            d.control = piece * 32 | (1u << 28) | (1u << 29);
            d.source = packed.value_column(
                kv, at, output_col + col * 32, (history + valid + 63) & ~63u);
            d.destination = key_buffer + col * width * 32 + part * 32;
            ++count;
          }
        descriptors[count - 1].next = nullptr;
        asm volatile("release(%0):at" ::"r"(descriptors) : "memory");
        asm volatile("dmstart(%0)" ::"r"(descriptors) : "memory");
        pending = !cooperative;
      };
      load_value(0);
      if (local_width)
        for (unsigned batch = 0; batch < batches; ++batch) {
          for (unsigned r = 0; r < rows; r += 2) {
            const unsigned index = batch * rows + r;
            if (r >= actual) {
              for (unsigned c = 0; c < local_width; c += 32)
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
            auto load_pair = [&](unsigned c) {
              return Q6_Wuw_vunpack_Vuh(*reinterpret_cast<const HVX_Vector*>(
                  packed_a + batch * probability_stride + (c / 32) * 2048 +
                  (r / 2) * 128));
            };
            for (unsigned c = 0; c < local_width; c += 32) {
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
            if (start) {
              const auto old = Q6_V_lo_W(Q6_W_vshuff_VVR(
                  Q6_V_vsplat_R(maximum[index + 1]),
                  Q6_V_vsplat_R(maximum[index]),
                  -4));
              const auto alpha = Q6_Vuw_vlsr_VuwR(
                  Q6_Vw_vadd_VwVw(
                      attention_exp::codes(
                          Q6_Vw_vsub_VwVw(mx, old),
                          exp_coefficient,
                          exp_table,
                          exp_slope_table,
                          exp_bits),
                      Q6_V_vsplat_R(1)),
                  1);
              rescale[index] =
                  normalizer[index] ? unsigned(Q6_R_vextract_VR(alpha, 0)) : 0;
              rescale[index + 1] = normalizer[index + 1]
                  ? unsigned(Q6_R_vextract_VR(alpha, 4))
                  : 0;
            } else {
              rescale[index] = rescale[index + 1] = 0;
            }
            auto code_sum = Q6_V_vzero();
            for (unsigned c = 0; c < local_width; c += 32) {
              const auto pair = load_pair(c);
              auto exponential = [&](HVX_Vector code, unsigned offset) {
                const auto visible = Q6_Q_vcmp_gt_VwVw(
                    limits,
                    Q6_Vw_vadd_VwVw(lane, Q6_V_vsplat_R(start + c + offset)));
                code = Q6_V_vmux_QVV(visible, code, mx);
                const auto e = attention_exp::codes(
                    Q6_Vw_vsub_VwVw(mx, code),
                    exp_coefficient,
                    exp_table,
                    exp_slope_table,
                    exp_bits);
                return Q6_V_vmux_QVV(visible, e, Q6_V_vzero());
              };
              const auto lo = exponential(Q6_V_lo_W(pair), 0),
                         hi = exponential(Q6_V_hi_W(pair), 16);

              auto lcode = lo, hcode = hi;
              code_sum =
                  Q6_Vw_vadd_VwVw(code_sum, Q6_Vw_vadd_VwVw(lcode, hcode));
              *reinterpret_cast<HVX_Vector*>(
                  packed_a + batch * probability_stride + (c / 32) * 2048 +
                  (r / 2) * 128) = Q6_Vuh_vpack_VwVw_sat(hcode, lcode);
            }
            for (unsigned shift = 64; shift >= 8; shift /= 2) {
              code_sum =
                  Q6_Vw_vadd_VwVw(code_sum, Q6_V_vror_VR(code_sum, shift));
            }
            buffer.probability_sum[index] = Q6_R_vextract_VR(code_sum, 0);
            buffer.probability_sum[index + 1] = Q6_R_vextract_VR(code_sum, 4);
            normalizer[index] = attention_fixed::multiply_q15(
                                    normalizer[index], rescale[index]) +
                buffer.probability_sum[index];
            normalizer[index + 1] =
                attention_fixed::multiply_q15(
                    normalizer[index + 1], rescale[index + 1]) +
                buffer.probability_sum[index + 1];
            maximum[index] = Q6_R_vextract_VR(mx, 0);
            maximum[index + 1] = Q6_R_vextract_VR(mx, 4);
          }
        }
      int32_t value_correction[batches * rows];
      uint32_t row_rescale[batches * rows];
      if (local_width)
        for (unsigned batch = 0; batch < batches; ++batch)
          for (unsigned r = 0; r < actual; ++r) {
            const unsigned index = batch * rows + r;
            value_correction[index] =
                int((int64_t(128 - value_zero) * buffer.probability_sum[index] +
                     128) >>
                    8);
            row_rescale[index] = rescale[index];
          }
      for (unsigned output_col = 0; output_col < 256; output_col += 64) {
        if (!wait_ready()) {
          __atomic_store_n(&queue.error, 1u, __ATOMIC_RELEASE);
          goto finished;
        }
        pending = false;
        if (local_width &&
            !attention_hmx::submit(
                queue,
                slice,
                packed_a,
                key_buffer,
                reinterpret_cast<uint8_t*>(packed_c),
                reinterpret_cast<uint8_t*>(bias),
                32,
                local_width,
                64,
                batches,
                probability_stride,
                batch_stride,
                buffer.kv + 65536,
                0,
                width,
                cooperative ? issued : 0))
          goto finished;
        if (output_col + 64 < 256)
          load_value(output_col + 64);
        else if (start + width < padded) {
          const unsigned next_width = std::min(256u, padded - start - width);
          start_dma(
              key_buffer,
              packed.key_panel(kv, start + width),
              next_width * 256,
              descriptors);
          pending = !cooperative;
        }
        if (local_width)
          for (unsigned batch = 0; batch < batches; ++batch)
            for (unsigned r = 0; r < actual; r += 2) {
              const unsigned index = batch * rows + r;
              const unsigned f0 = row_rescale[index],
                             f1 = r + 1 < actual ? row_rescale[index + 1] : 0;
              const int correction0 = value_correction[index],
                        correction1 =
                            r + 1 < actual ? value_correction[index + 1] : 0;
              for (unsigned c = 0; c < 64; c += 32) {
                const unsigned at =
                    batch * batch_stride / 2 + (c / 32) * 1024 + (r / 2) * 64;
                auto l = *reinterpret_cast<HVX_Vector*>(packed_c + at),
                     h = *reinterpret_cast<HVX_Vector*>(
                         reinterpret_cast<uint16_t*>(buffer.kv + 65536) + at);
                auto value0 = Q6_Vw_vadd_VwVw(
                    attention_fixed::unpack_result(l, h, false),
                    Q6_V_vsplat_R(correction0));
                auto value1 = Q6_Vw_vadd_VwVw(
                    attention_fixed::unpack_result(l, h, true),
                    Q6_V_vsplat_R(correction1));
                auto* p0 = reinterpret_cast<HVX_Vector*>(
                    acc + index * 256 + output_col + c);
                auto* p1 = reinterpret_cast<HVX_Vector*>(
                    acc + (index + 1) * 256 + output_col + c);
                *p0 = start
                    ? Q6_Vw_vadd_VwVw(
                          attention_fixed::multiply_q15(*p0, f0), value0)
                    : value0;
                if (r + 1 < actual)
                  *p1 = start
                      ? Q6_Vw_vadd_VwVw(
                            attention_fixed::multiply_q15(*p1, f1), value1)
                      : value1;
              }
            }
      }
    }
    for (unsigned batch = 0; batch < batches; ++batch)
      for (unsigned r = 0; r < actual; ++r) {
        const unsigned index = batch * rows + r, head = first_head + batch;
        const auto output_scale = integer_output_ratio
            ? attention_fixed::Scale::ratio(
                  output_ratio, uint64_t(normalizer[index]) * 4096)
            : attention_fixed::Scale::from_double(
                  output_ratio_value / (double(normalizer[index]) * 4096));
        for (unsigned c = 0; c < 256; c += 64) {
          auto a = attention_fixed::affine_u16(
              *reinterpret_cast<HVX_Vector*>(acc + index * 256 + c),
              output_scale,
              oq.zero_offset);
          auto b = attention_fixed::affine_u16(
              *reinterpret_cast<HVX_Vector*>(acc + index * 256 + c + 32),
              output_scale,
              oq.zero_offset);
          *reinterpret_cast<HVX_Vector*>(
              y + (head * qrows + begin + r) * 256 + c) =
              Q6_Vuh_vpack_VwVw_sat(b, a);
        }
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
