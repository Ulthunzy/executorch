# DeltaNet HTP Custom Op Benchmark

Date: 2026-05-29

## Environment

- Repo: `/hunter/cpp_code/ET/npu/executorch`
- Python/env: `conda activate et_npu` and `source /hunter/cpp_code/ET/npu/executorch/env.sh`
- QNN SDK: `/hunter/qnn_sdk/src/backends/qnn/2.45.0.260326`
- Device: `3B15AK00GLW00000`
- SoC: `SM8850`, HTP v81
- Runner args: `-b build-android -s 3B15AK00GLW00000 -m SM8850 --backend htp`
- Op package build:

```bash
source /home/hunter/anaconda3/etc/profile.d/conda.sh
conda activate et_npu
source /hunter/cpp_code/ET/npu/executorch/env.sh
make clean && make htp_x86 htp_aarch64 htp_v81
cp build/hexagon-v81/libQnnDeltaNetOpPackage.so build/hexagon-v81/libQnnDeltaNetOpPackage_HTP.so
```

## Implementation Status

- Implemented `qnn_custom_ops::deltanet` mapped to QNN op type `DeltaNet`.
- Implemented `qnn_custom_ops::deltanet_core` mapped to QNN op type `DeltaNetCore`.
- The op takes `query`, `key`, `value`, `g`, `beta`, `recurrent_state`, scalar `scaling`, and scalar `epsilon`.
- The op returns two outputs: `core_attn_out` and `new_recurrent_state`.
- Also kept debug/probe ops in the package:
  - `DeltaNetIdentity`
  - `DeltaNetDecayState`
  - `DeltaNetStatePassthrough`
  - `DeltaNetPacked`
- Python benchmark/export script: `examples/qualcomm/custom_op/deltanet.py`
- simple_reg HTP package source: `examples/qualcomm/custom_op/deltanet_op_package_htp/DeltaNetOpPackage/src/ops/DeltaNet.cpp`
- QHPI HTP package source: `examples/qualcomm/custom_op/deltanet_op_package_htp/DeltaNetOpPackage/src/ops/DeltaNetQhpi.cpp`

Current `DeltaNet` implementation is correct and has:

- Raw-data fast paths for fp32, fp16, QUInt16, and QUInt8 tensors. These avoid `Tensor::operator()` in the hot recurrent loops and walk recurrent state as `[key_dim, value_dim]` rows so the inner loop is contiguous over `value_dim`.
- Hexagon-only HVX fast paths for fp32/fp16 `DeltaNet` and `DeltaNetPacked`. The fp16 path converts contiguous half rows to fp32 vectors, accumulates `kv_mem` and output pre-sums with HVX, and writes updated state/output back as fp16.
- `DeltaNetCore` takes pre-normalized `q/k` and precomputed `decay = exp(g)`, so native QNN ops can handle L2 norm and exp. It is registered through QHPI with fp32, fp16, and QUInt16 kernels, requests `QHPI_RESOURCE_HVX`, and self-slices over `(batch, head)` with `qhpi_num_slices()` / `qhpi_slice_number()`.
- The HVX paths now fuse `delta = (value - kv_mem) * beta` and `output = output_pre + delta * dot(k, q)` into one vector pass, reducing one scratch read/write pass over `delta`.

The fp16 and QUInt16 `DeltaNetCore` paths are now close to the decomposed QNN
baseline on the target generation shape. The older full `DeltaNet` simple_reg
quantized path is still scalar/raw and remains useful mainly as a correctness
reference.

## Qwen3.5-0.8B Quant Recipe

Current code only registers `Qwen3_5_0_8BQuantRecipe`; there is no separate active
`Qwen3_5_0_8BQuantRecipeSpeed` class in `static_llm_quant_recipe.py`. The older
`qwen35_adaptation_status.txt` notes mention a speed recipe, but that is stale
relative to the current source.

The active recipe is mixed precision inside one class:

- Default: `QuantDtype.use_16a4w`.
- Q/K/V-style projection convs such as `in_proj_qkv_z`, `in_proj_b`, `wq/wk/wv`,
  and selected FFN convs: `QuantDtype.use_16a4w_block`.
- Sensitive projection/depthwise/output convs such as `in_proj_a`, `out_proj`,
  `causal_conv2d`, `wo`, `w2_conv`, and `output.conv`:
  `QuantDtype.use_16a8w`.

For the DeltaNet recurrent core itself there are no weights, so the important
format to match is the activation/state format. All of the `16a*` choices use
16-bit activations, which arrive in the op package as QUInt16/uint16 tensors.
Therefore the primary same-quant comparison for Qwen3.5-0.8B DeltaNet should be
`use_16a4w` as the active recipe default, with `use_16a8w` only kept as an
optional sensitivity check for the recipe's explicit 16a8w override regions.

## Qwen3.5-0.8B Same-Quant DeltaNet Test

Shape comes from `examples/models/qwen3_5/config/0_8b_config.json`:
`batch=1`, `ar_len=1`, `heads=16`, `key_dim=128`, `value_dim=128`.

Command:

```bash
python examples/qualcomm/custom_op/deltanet.py \
  -b build-android -s 3B15AK00GLW00000 -m SM8850 --backend htp \
  --variants baseline,custom \
  --quant_dtypes use_16a4w \
  --batch 1 --ar_len 1 --heads 16 --head_k_dim 128 --head_v_dim 128 \
  --calibration_samples 4 --iteration 200 \
  -a deltanet-qnn-qwen35-0_8b-use16a4w-sm8850 --keep_artifact
```

Artifact: `deltanet-qnn-qwen35-0_8b-use16a4w-sm8850/summary.json`.

| Quant | Variant | Avg ms | Output SQNR dB | State SQNR dB | Output max diff | State max diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| use_16a4w | baseline | 0.73504 | 16.9362 | 15.6590 | 7.334e-03 | 1.342e-01 |
| use_16a4w | custom `DeltaNet` | 24.92029 | 66.0339 | 81.7284 | 9.476e-06 | 1.565e-03 |

The same shape was also tested with `use_16a8w` as a sensitivity check for
the recipe's explicit 16a8w override regions.

Artifact: `deltanet-qnn-qwen35-0_8b-use16a8w-sm8850/summary.json`.

| Quant | Variant | Avg ms | Output SQNR dB | State SQNR dB | Output max diff | State max diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| use_16a8w | baseline | 0.73000 | 16.9362 | 15.6590 | 7.334e-03 | 1.342e-01 |
| use_16a8w | custom `DeltaNet` | 24.90423 | 66.0339 | 81.7284 | 9.476e-06 | 1.565e-03 |

Conclusion:

- The current quantized full `DeltaNet` custom op is much more accurate than the
  decomposed same-quant baseline, but it is roughly `34x` slower on this shape.
- The speed loss is expected: full `DeltaNet` QUInt16 uses the simple_reg raw
  quant path, dequantizes to float, computes in scalar loops, and requantizes.
  It does not use the QHPI/HVX `DeltaNetCore` kernel.

### QUInt16 DeltaNetCore Kernel

Added a QHPI `DeltaNetCore.quint16.fixed` kernel for `QUInt16, Flat4, DDR`
inputs/outputs. This kernel:

- reads QHPI quant params through `qhpi_tensor_quant_parameters`;
- uses centered integer values for the hot `key_dim * value_dim` recurrent
  loops;
- accumulates `kv_mem` and `output_pre` as integer dot products;
- uses fixed-point state-update coefficients instead of per-element float
  dequant/requant in the update pass;
- requantizes `core_attn_out` and `new_recurrent_state` back to QUInt16;
- self-slices over `(batch, head)` using the existing QHPI slice path.

Build:

```bash
make htp_x86 htp_aarch64 htp_v81
cp build/hexagon-v81/libQnnDeltaNetOpPackage.so \
  build/hexagon-v81/libQnnDeltaNetOpPackage_HTP.so
```

Compile-only smoke:

```bash
python examples/qualcomm/custom_op/deltanet.py \
  -b build-android -s 3B15AK00GLW00000 -m SM8850 --backend htp \
  --variants core \
  --quant_dtypes use_16a4w \
  --batch 1 --ar_len 1 --heads 16 --head_k_dim 128 --head_v_dim 128 \
  --calibration_samples 4 --iteration 20 \
  -a deltanet-qnn-qwen35-0_8b-core-use16a4w-qhpi-u16-compile-sm8850 \
  --compile_only --keep_artifact
```

Compile-only result: graph prepare succeeds and the PTE is generated at
`deltanet-qnn-qwen35-0_8b-core-use16a4w-qhpi-u16-compile-sm8850/deltanet_core_use_16a4w_b1_t1_h16_k128_v128/deltanet_core_use_16a4w_b1_t1_h16_k128_v128.pte`.

Runtime artifacts:

- `deltanet-qnn-qwen35-0_8b-core-use16a4w-qhpi-u16-sm8850/summary.json`
- `deltanet-qnn-qwen35-0_8b-core-use16a8w-qhpi-u16-sm8850/summary.json`
- `deltanet-qnn-qwen35-0_8b-core-use16a4w-qhpi-u16-fixed-sm8850/summary.json`
- `deltanet-qnn-qwen35-0_8b-core-use16a8w-qhpi-u16-fixed-sm8850/summary.json`

| Quant | Variant | Avg ms | Output SQNR dB | State SQNR dB | Output max diff | State max diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| use_16a4w | core `DeltaNetCore.quint16.scalar` | 5.17072 | 64.0626 | 76.4361 | 3.452e-05 | 1.579e-03 |
| use_16a8w | core `DeltaNetCore.quint16.scalar` | 5.12464 | 64.0626 | 76.4361 | 3.452e-05 | 1.579e-03 |
| use_16a4w | core `DeltaNetCore.quint16.fixed` | 1.56429 | 64.0613 | 76.4331 | 3.452e-05 | 1.579e-03 |
| use_16a8w | core `DeltaNetCore.quint16.fixed` | 1.55565 | 64.0613 | 76.4331 | 3.452e-05 | 1.579e-03 |

The fixed-point QUInt16 core was about `3.3x` faster than the first QUInt16
QHPI core (`5.17 ms`) and about `16x` faster than the full quantized simple_reg
`DeltaNet` path (`24.92 ms`). It was still about `2.1x` slower than the
decomposed QNN baseline (`0.735 ms`) because the fixed-point path was still
scalar inside each QHPI worker; the follow-up sections record the HVX
accumulator, state update, and output/delta passes that closed most of this
gap.

### QUInt16 HVX Accumulator

Added a first HVX path for the QUInt16 recurrent accumulator:

- affine QUInt16 values are shifted down by `4` bits before centering, so the
  arbitrary activation zero point from the Qwen3.5 recipe can safely fit into
  signed halfword lanes;
- `state * key` and `state * query` are accumulated with HVX halfword-to-word
  products;
- the halfword-to-word product pair is reordered with
  `Q6_W_vshuff_VVR(hi, lo, -4)` before writing contiguous `value_dim` lanes;
- the scalar `int64` path is kept as the exact fallback/reference path.

Important failed intermediate checks:

- Directly subtracting arbitrary QUInt16 zero point in halfword lanes produced
  bad SQNR because non-32768 zero points can wrap signed halfword values.
- Switching the scalar fallback scratch from `int64` to `int32` also broke SQNR
  because `key_dim=128` centered products can overflow 32-bit accumulators.
- Shift `3` improved output SQNR but was slower on SM8850 in this test
  (`1.81166 ms`), so the current checked-in path uses shift `4`.

Runtime artifacts:

- `deltanet-qnn-qwen35-0_8b-core-use16a4w-qhpi-u16-hvxshuffle-sm8850/summary.json`
- `deltanet-qnn-qwen35-0_8b-core-use16a8w-qhpi-u16-hvxshuffle-sm8850/summary.json`

| Quant | Variant | Avg ms | Output SQNR dB | State SQNR dB | Output max diff | State max diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| use_16a4w | core `DeltaNetCore.quint16.hvx_accum_s4` | 1.32380 | 56.9734 | 75.0139 | 6.383e-05 | 1.579e-03 |
| use_16a8w | core `DeltaNetCore.quint16.hvx_accum_s4` | 1.76148 | 56.9734 | 75.0139 | 6.383e-05 | 1.579e-03 |

For this recurrent core, `use_16a4w` and `use_16a8w` both arrive as 16-bit
activation/state tensors and there are no weights inside the op, so SQNR is the
same. The `use_16a8w` latency above is treated as a noisy sensitivity run; the
more representative latency for this kernel revision is the `use_16a4w` run.

Current bottleneck after HVX accumulator:

- QUInt16 I/O is still 16-bit, so it does not reduce state traffic relative to
  fp16. The main win must come from integer HVX compute, not from bytes moved.
- At this point the first `kv_mem/output_pre` pass was vectorized, but state
  update and output requant were still scalar fixed-point loops. The follow-up
  work below moves those two passes onto HVX F32.

### QUInt16 Follow-up Experiments

The current correct QUInt16 path keeps the HVX accumulator and now uses the
integer HVX state-update path first:

- `kQUint16HvxStateUpdateInputShift = 0`
- `kQUint16HvxStateUpdateCoeffShift = 7`
- `kEnableQUint16HvxStateUpdate = true`
- `kEnableQUint16HvxStateUpdateF32 = true` remains as the fallback if integer
  guards fail

The fused F32 HVX pass still computes:

```text
delta = (value - kv_mem) * beta
output = output_pre + delta * dot(key, query)
```

When the integer update path is active, that same fused pass also writes the
shifted fixed-point `delta_update_fixed32` values for state update and skips the
float `delta_value_scratch` store on full HVX-vector tiles. This removes the
F32 conversion from the QUInt16 state-update main path while keeping output
requantization in the fused F32 vector pass.

Runtime artifacts:

- `deltanet-qnn-qwen35-0_8b-core-use16a4w-qhpi-u16-hvxupdate-deal-sm8850/summary.json`
- `deltanet-qnn-qwen35-0_8b-core-use16a4w-qhpi-u16-kunroll2-sm8850/summary.json`
- `deltanet-qnn-qwen35-0_8b-core-use16a4w-qhpi-u16-hvxacc-scalarassign-sm8850/summary.json`
- `deltanet-qnn-qwen35-0_8b-core-use16a4w-qhpi-u16-floatrequant-sm8850/summary.json`
- `deltanet-qnn-qwen35-0_8b-core-use16a4w-qhpi-u16-stateinline-sm8850/summary.json`
- `deltanet-qnn-qwen35-0_8b-core-use16a4w-qhpi-u16-hvxf32update-swapdeal-sm8850/summary.json`
- `deltanet-qnn-qwen35-0_8b-core-use16a4w-qhpi-u16-hvxdeltaout-sm8850/summary.json`
- `deltanet-qnn-qwen35-0_8b-core-use16a4w-qhpi-u16-hvxassign-hvxdeltaout-sm8850/summary.json`
- `deltanet-qnn-qwen35-0_8b-core-use16a4w-qhpi-u16-alignedscratch-sm8850/summary.json`
- `deltanet-qnn-qwen35-0_8b-core-use16a4w-qhpi-u16-keycache-sm8850/summary.json`
- `deltanet-qnn-qwen35-0_8b-core-use16a4w-qhpi-u16-final-iter1000-sm8850/summary.json`
- `deltanet-qnn-qwen35-0_8b-core-use16a4w-qhpi-u16-intupdate-s8-sm8850/summary.json`
- `deltanet-qnn-qwen35-0_8b-core-use16a4w-qhpi-u16-intupdate-s8-final-sm8850/summary.json`
- `deltanet-qnn-qwen35-0_8b-core-use16a4w-qhpi-u16-intupdate-s8-nofloatdelta-final-sm8850/summary.json`
- `deltanet-qnn-qwen35-0_8b-core-use16a4w-qhpi-u16-intupdate-s8-clean-smoke-sm8850/summary.json`
- `deltanet-qnn-qwen35-0_8b-core-use16a4w-qhpi-u16-intupdate-s6-sm8850/summary.json`
- `deltanet-qnn-qwen35-0_8b-core-use16a4w-qhpi-u16-intupdate-s7-sm8850/summary.json`
- `deltanet-qnn-qwen35-0_8b-core-use16a4w-qhpi-u16-intupdate-s7-final-sm8850/summary.json`
- `deltanet-qnn-qwen35-0_8b-core-use16a4w-qhpi-u16-intupdate-s9-sm8850/summary.json`

| Experiment | Avg ms | Output SQNR dB | State SQNR dB | Decision |
| --- | ---: | ---: | ---: | --- |
| HVX approximate state update | 0.82886 | 56.9734 | -3.0068 | Rejected: state output is numerically wrong. |
| 2-way k unroll for HVX accumulator | 1.72069 | 56.9734 | 75.0139 | Rejected: slower than the single-k HVX path. |
| Scalar k=0 assign + HVX k>=1 accumulator | 1.75923 | 56.9734 | 75.0139 | Kept as the conservative stable path on this build; it preserves SQNR. |
| Float output requant scales | 1.80416 | 56.9737 | 75.0173 | Rejected: no speed gain. |
| Inline exact scalar state update | 1.71974 | 56.9734 | 75.0139 | Kept: small speedup, exact same SQNR. |
| F32 HVX state update, fixed lane order | 0.84367 | 56.9734 | 75.0182 | Kept: correct state tensor and large speedup. |
| F32 HVX delta/output fused pass | 0.81748 | 56.9807 | 75.0182 | Kept: removes the scalar delta/output loop. |
| HVX k=0 assign with fused delta/output | 0.83602 | 56.9807 | 75.0182 | Rejected: slower than scalar k=0 initialization. |
| Aligned scratch load/store experiment | 0.83448 | 56.9807 | 75.0182 | Rejected: slower than `vmemu` on this build. |
| Cached shifted key/query centers | 0.82761 | 56.9807 | 75.0181 | Rejected: extra scratch traffic outweighed scalar savings. |
| F32 HVX update final, 1000 iterations | 0.81521 | 56.9807 | 75.0182 | Previous stable source before integer update. |
| Integer HVX state update, coeff shift 8, 200 iterations | 0.73538 | 56.9807 | 73.2792 | Promising smoke; close to decomposed baseline but not stable enough alone. |
| Integer HVX state update, coeff shift 8, 1000 iterations | 0.80134 | 56.9807 | 73.2792 | Kept: removes F32 conversion from state update main path. |
| Integer HVX update + skip full-tile float delta scratch, coeff shift 8, 1000 iterations | 0.79541 | 56.9807 | 73.2792 | Kept at the time; later superseded by coeff shift 7. |
| Clean source smoke after removing failed cache experiment, 200 iterations | 0.80043 | 56.9807 | 73.2792 | Confirms the cleaned source still runs correctly. |
| Integer HVX update, coeff shift 6, 200 iterations | 0.81559 | 56.9807 | 74.8133 | Rejected: state SQNR improves, but latency regresses to the old F32-update range. |
| Integer HVX update + skip full-tile float delta scratch, coeff shift 7, 200 iterations | 0.80396 | 56.9807 | 74.4192 | Better state SQNR than shift 8; speed is within run noise. |
| Integer HVX update + skip full-tile float delta scratch, coeff shift 7, 1000 iterations | 0.79231 | 56.9807 | 74.4192 | Kept: current best stable QUInt16 core. |
| Integer HVX update, coeff shift 9, 200 iterations | 0.80823 | 56.9807 | 70.3562 | Rejected: lower precision and no speed win. |

Notes:

- The first fast state-update attempt was wrong because the final word-to-u16
  pack stored lanes in the wrong order. The correct store for this path is
  `Q6_Vuh_vasr_VwVwR_sat(hi, lo, 0)` followed by `Q6_Vh_vdeal_Vh`.
- 2-way k unroll increased register pressure/instruction overhead enough to
  regress latency on SM8850 for `heads=16, key_dim=128, value_dim=128`.
- The scalar k=0 assign looks odd, but that pass only initializes the scratch
  once per token/head. The repeated `k>=1` accumulator remains HVX-vectorized.
- The first corrected integer update attempt needed the fused delta/output pass
  to produce `delta_update_fixed32`; without that, the vector main segment used
  uninitialized update data and state SQNR collapsed.
- A local `int32` state-cache experiment was attempted to avoid the second state
  read, but it failed on device with QNN DSP execution error `1003`; it is not
  kept in source. This likely needs VTCM/QHPI-managed scratch rather than a
  large per-slice stack buffer.
- The current custom core is now within about `8%` of the decomposed same-quant
  QNN baseline (`0.735 ms`) on the stable 1000-iteration comparison, while
  preserving much higher recurrent-state SQNR than that baseline (`74.4 dB` vs
  `15.7 dB` in this synthetic test).
- Remaining likely bottlenecks are the two full state reads per token
  (accumulate pass and update pass), F32 conversion in the fused output/delta
  pass, and QHPI custom-op scheduling overhead. Beating the decomposed QNN
  baseline likely needs VTCM/local-state retention for `seq_len > 1`,
  safer state-cache plumbing through QHPI, or graph-level fusion around the core
  op.

## Debug Notes

- QNN/HTP presents the PyTorch 3D gate tensors `g` and `beta` as rank-4 tensors in the custom op path. Strict `rank == 3` validation made state outputs zero. The op now reads gate tensors through `readTimeHeadValue`, which accepts the observed rank-3/rank-4 layouts.
- `scaling` and `epsilon` must be scalar parameters, not `float[]`. With `float[]`, ExecuTorch's custom-op builder created a TensorParam from `np.array([python_float])`, which defaulted to float64 while the QNN datatype was float32. Reading the low 4 bytes of the double made `scaling=0.088388...` appear as about `3e23`, causing output overflow.
- `scaling` and `epsilon` are now declared as `float` in the Torch custom op schema and as `SCALAR` in the XML.
- Reading scalar params with `param(0, 0, 0, 0)` works after switching to scalar params.
- The first implementation wrote the decayed recurrent state to the output tensor and then read it back for the update. The raw fast path now computes `kv_mem` from the decayed state, recomputes the decayed value from the previous state in the update pass, and writes only the final state. This avoids one full intermediate state write/read and avoids an extra quantize/dequantize round in the quantized path.
- The HVX path additionally computes the output as `sum(decayed_state * query) + delta * dot(key, query)`, which avoids accumulating output during the update writeback pass.
- The full `DeltaNet` op is still registered through `simple_reg`, so it does not expose llama.cpp's `htp_ops_context`, worker pool, DMA queue, or VTCM scratch pointer.
- `DeltaNetCore` now uses QHPI self-slicing. This confirms the QNN SDK path for llama.cpp-style outer parallelism: mark the kernel `multithreaded = true`, request `QHPI_RESOURCE_HVX`, and split `(batch, head)` work with `qhpi_num_slices()` / `qhpi_slice_number()`. VTCM/DMA state staging is still not implemented.
- QHPI output signatures must use a concrete layout. `QHPI_Layout_Any` worked for input matching, but using it for the two rank-4 outputs failed during graph preparation with `tensor_generator.h:90::Lookup in 1 tensor types failed`. Changing both outputs to `QHPI_Layout_Flat4` fixed QNN prepare.
- The benchmark label `fp16` currently means "no PTQ quantizer" in `deltanet.py`; the pulled output files for these cases are `float32`. The QHPI core therefore registers both fp32 and fp16 kernels, but the current default `fp16` benchmark below is matching `DeltaNetCore.fp32.hvx`.

## Llama.cpp HTP Notes

Reference inspected: `/hunter/cpp_code/llama.cpp/ggml/src/ggml-hexagon/htp/gated-delta-net-ops.c`.

The most useful design points from llama.cpp are:

- It keeps L2 norm as a separate op. `GGML_OP_L2_NORM` and `GGML_OP_GATED_DELTA_NET` are separate HTP ops, so DeltaNet itself assumes normalized `q/k` plus the final scale.
- It has separate paths for prefill and token generation. `n_tokens == 1` uses an 8-row block kernel and VTCM scratch for the state; prefill uses a 4-row block kernel.
- It partitions work by `(sequence, head)` rows with a worker pool.
- The inner kernels combine decay, dot, update, and output dot while walking contiguous state rows with HVX reductions.
- It supports scalar gate and KDA vector gate. For scalar gate it uses `expf(g[0])`; for vector gate it computes `exp(g)` with the HVX exp helper.

Our ExecuTorch/QNN state layout is `[batch, heads, key_dim, value_dim]`, so directly copying llama.cpp's `[value row][key]` kernel would make fixed-value access strided. Instead, the current raw/HVX fast path uses the same idea transposed for our layout: iterate `key_dim` rows and keep the inner loop contiguous over `value_dim`.

## Commands

Small quant smoke:

```bash
python examples/qualcomm/custom_op/deltanet.py \
  -b build-android -s 3B15AK00GLW00000 -m SM8850 --backend htp \
  --variants baseline,custom \
  --quant_dtypes fp16,use_16a8w,use_16a4w,use_8a8w \
  --batch 1 --ar_len 1 --heads 2 --head_k_dim 8 --head_v_dim 8 \
  --calibration_samples 4 --iteration 20 \
  -a deltanet-qnn-quant-smoke-sm8850 --keep_artifact
```

Main benchmark:

```bash
python examples/qualcomm/custom_op/deltanet.py \
  -b build-android -s 3B15AK00GLW00000 -m SM8850 --backend htp \
  --variants baseline,custom \
  --quant_dtypes fp16,use_16a8w,use_16a4w,use_8a8w \
  --batch 1 --ar_len 1 --heads 16 --head_k_dim 128 --head_v_dim 128 \
  --calibration_samples 4 --iteration 100 \
  -a deltanet-qnn-benchmark-sm8850 --keep_artifact
```

HVX benchmark:

```bash
python examples/qualcomm/custom_op/deltanet.py \
  -b build-android -s 3B15AK00GLW00000 -m SM8850 --backend htp \
  --variants baseline,custom \
  --quant_dtypes fp16,use_16a8w,use_16a4w,use_8a8w \
  --batch 1 --ar_len 1 --heads 16 --head_k_dim 128 --head_v_dim 128 \
  --calibration_samples 4 --iteration 100 \
  -a deltanet-qnn-hvx-final-sm8850 --keep_artifact
```

## Small Smoke Results

Shape: `batch=1, ar_len=1, heads=2, key_dim=8, value_dim=8`, `iteration=20`.

| Quant | Variant | Avg ms | Output SQNR dB | State SQNR dB | Output max diff | State max diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| fp16 | baseline | 0.22395 | 61.7836 | 66.8878 | 2.254e-05 | 1.568e-04 |
| fp16 | custom | 0.25635 | 141.387 | 153.693 | 1.630e-09 | 7.451e-09 |
| use_16a8w | baseline | 0.25680 | 7.1181 | 2.8098 | 6.957e-03 | 1.020e-01 |
| use_16a8w | custom | 0.26100 | 40.9870 | 38.2040 | 2.949e-04 | 1.159e-02 |
| use_16a4w | baseline | 0.21135 | 7.1181 | 2.8098 | 6.957e-03 | 1.020e-01 |
| use_16a4w | custom | 0.26125 | 40.9870 | 38.2040 | 2.949e-04 | 1.159e-02 |
| use_8a8w | baseline | 0.23815 | 7.1824 | 2.7809 | 6.739e-03 | 1.016e-01 |
| use_8a8w | custom | 0.30765 | 24.2608 | 35.9750 | 8.352e-04 | 1.172e-02 |

## Initial Scalar Main Benchmark Results

Shape: `batch=1, ar_len=1, heads=16, key_dim=128, value_dim=128`, `iteration=100`.

| Quant | Variant | Avg ms | Output SQNR dB | State SQNR dB | Output max diff | State max diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| fp16 | baseline | 0.72533 | 63.9606 | 67.7232 | 2.942e-05 | 3.937e-04 |
| fp16 | custom | 72.18952 | 131.797 | 149.467 | 1.490e-08 | 4.843e-08 |
| use_16a8w | baseline | 0.74135 | 16.9362 | 15.6590 | 7.334e-03 | 1.342e-01 |
| use_16a8w | custom | 75.39902 | 66.0056 | 78.1162 | 9.476e-06 | 4.305e-03 |
| use_16a4w | baseline | 0.92245 | 16.9362 | 15.6590 | 7.334e-03 | 1.342e-01 |
| use_16a4w | custom | 75.39819 | 66.0056 | 78.1162 | 9.476e-06 | 4.305e-03 |
| use_8a8w | baseline | 0.75362 | 15.6881 | 13.6146 | 8.350e-03 | 1.645e-01 |
| use_8a8w | custom | 71.48144 | 17.9291 | 34.5503 | 2.157e-03 | 5.767e-03 |

## Raw Row Main Benchmark Results

Shape: `batch=1, ar_len=1, heads=16, key_dim=128, value_dim=128`, `iteration=100`.

Artifact: `deltanet-qnn-raw-final-sm8850/summary.json`.

| Quant | Variant | Avg ms | Output SQNR dB | State SQNR dB | Output max diff | State max diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| fp16 | baseline | 0.70133 | 63.9606 | 67.7232 | 2.942e-05 | 3.937e-04 |
| fp16 | custom | 2.82055 | 131.645 | 145.931 | 1.490e-08 | 4.470e-08 |
| use_16a8w | baseline | 0.76217 | 16.9362 | 15.6590 | 7.334e-03 | 1.342e-01 |
| use_16a8w | custom | 24.96503 | 66.0339 | 81.7284 | 9.476e-06 | 1.565e-03 |
| use_16a4w | baseline | 0.73501 | 16.9362 | 15.6590 | 7.334e-03 | 1.342e-01 |
| use_16a4w | custom | 24.75185 | 66.0339 | 81.7284 | 9.476e-06 | 1.565e-03 |
| use_8a8w | baseline | 0.74125 | 15.6881 | 13.6146 | 8.350e-03 | 1.645e-01 |
| use_8a8w | custom | 27.00889 | 17.9608 | 35.7955 | 2.163e-03 | 4.266e-03 |

The raw row path improves fp16 custom latency from `72.18952 ms` to `2.82055 ms`. Quantized custom latency also improves from roughly `71-75 ms` to `25-27 ms`, while keeping the higher SQNR advantage over the decomposed baseline.

## HVX Main Benchmark Results

Shape: `batch=1, ar_len=1, heads=16, key_dim=128, value_dim=128`.

Main artifact: `deltanet-qnn-hvx-final-sm8850/summary.json`, `iteration=100`.

| Quant | Variant | Avg ms | Output SQNR dB | State SQNR dB | Output max diff | State max diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| fp16 | baseline | 0.70583 | 63.9606 | 67.7232 | 2.942e-05 | 3.937e-04 |
| fp16 | custom | 1.18253 | 132.367 | 149.395 | 1.304e-08 | 4.470e-08 |
| use_16a8w | baseline | 0.82997 | 16.9362 | 15.6590 | 7.334e-03 | 1.342e-01 |
| use_16a8w | custom | 24.75061 | 66.0339 | 81.7284 | 9.476e-06 | 1.565e-03 |
| use_16a4w | baseline | 0.73774 | 16.9362 | 15.6590 | 7.334e-03 | 1.342e-01 |
| use_16a4w | custom | 24.73462 | 66.0339 | 81.7284 | 9.476e-06 | 1.565e-03 |
| use_8a8w | baseline | 0.86575 | 15.6881 | 13.6146 | 8.350e-03 | 1.645e-01 |
| use_8a8w | custom | 27.00720 | 17.9608 | 35.7955 | 2.163e-03 | 4.266e-03 |

An isolated fp16 confirmation run with `iteration=200` is in `deltanet-qnn-hvx-fp16-confirm-sm8850/summary.json`:

| Quant | Variant | Avg ms | Output SQNR dB | State SQNR dB |
| --- | --- | ---: | ---: | ---: |
| fp16 | baseline | 0.69376 | 63.9606 | 67.7232 |
| fp16 | custom | 0.77438 | 132.367 | 149.395 |

So the fp16 HVX path improves custom latency from `2.82055 ms` to `0.77-1.18 ms`, depending on run conditions. It is now close to baseline latency while retaining much higher SQNR. Later QUInt16 work moved the `DeltaNetCore` quantized path from scalar raw loops to HVX-assisted accumulator, state update, and output/delta passes; the full simple_reg `DeltaNet` quantized path remains scalar.

`DeltaNetPacked` was also tested in `deltanet-qnn-hvx-packed-fp16-main-sm8850/summary.json`. It keeps L2 norm in native QNN ops and sends only the recurrent core to the custom op, but the current pack/unpack graph adds enough slice/view overhead that fp16 packed latency is `1.20059 ms`, so it is not the best route yet.

## QHPI DeltaNetCore Results

Shape: `batch=1, ar_len=1, heads=16, key_dim=128, value_dim=128`, `iteration=200`.

Artifact: `deltanet-qnn-fusedpass-compare-fp32-sm8850/summary.json`.

| Quant label | Variant | Runtime path | Avg ms | Output SQNR dB | State SQNR dB | Output max diff | State max diff |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| fp16 | baseline | decomposed QNN graph | 0.84097 | 63.9606 | 67.7232 | 2.942e-05 | 3.937e-04 |
| fp16 | custom | simple_reg `DeltaNet` full op | 1.04750 | 132.367 | 149.395 | 1.304e-08 | 4.470e-08 |
| fp16 | core | QHPI `DeltaNetCore.fp32.hvx` + native QNN norm/exp | 0.75148 | 66.9835 | 74.1249 | 2.225e-05 | 1.200e-04 |

An isolated core run in `deltanet-qnn-core-qhpi-fused-fp32-flat4-sm8850/summary.json` measured `0.75588 ms` with the same SQNR. This makes `DeltaNetCore` the fastest current custom-op route on this shape, and it removes scalar `std::exp`/`std::sqrt` from the custom op. It still trails the best older decomposed-baseline run (`0.69376 ms` in `deltanet-qnn-hvx-fp16-confirm-sm8850`), so more kernel-side optimization is needed before calling it a clear win.

## Next QHPI Optimization Plan

Current `DeltaNetCore` is a good correctness and first-performance baseline, but the remaining bottlenecks are mostly scheduling, memory traffic, and small-tile HVX occupancy rather than algebraic fusion.

Main risks and optimization points:

1. Slicing only over `batch * heads` can leave QHPI slices idle when the number of rows is smaller than the number of slices. The next parallel fallback should split `value_dim` tiles because `kv_mem[v]`, `delta[v]`, `output[v]`, and `state[k, v]` are independent across `v`.
2. F16 only gets a full HVX vector when `value_dim >= 64`; F32 only gets one when `value_dim >= 32`. Smaller value dimensions currently fall mostly into scalar tails, so small-head models need cross-head packing or a packed state layout.
3. The F16 path reads each state row twice and converts F16 to F32 twice: once for `kv_mem/output_pre`, then again for state update. Possible follow-ups are F32 state or local F32 state scratch for small `key_dim * value_dim`.
4. `key_query_dot` is still scalar. It is not the first bottleneck for large `value_dim`, but should be vectorized or precomputed if profiling shows it hot.
5. Scratch zeroing with `std::fill` can be folded into the first `k == 0` accumulation pass.
6. Old split helpers for `delta` and output writeback should be removed now that `deltanetQhpiHvxDeltaAndWriteOutput*` is used.
7. The kernel assumes flat contiguous `[B, T, H, D]` inputs and `[B, H, K, V]` state, so QHPI signatures should not advertise fully arbitrary layout for the rank-4 tensors.
8. All input and output shapes should be checked before raw pointer indexing: query/key `[B,T,H,K]`, value/output `[B,T,H,V]`, decay/beta `[B,T,H]` or `[B,T,H,1]`, state `[B,H,K,V]`.
9. The QHPI kernel now has a coarse cost function, but still has no rewrite or
   custom tiling hook beyond self-slicing.
10. All tensors are `DDR_Only`. For `seq_len > 1`, a local state scratch path can avoid rereading and rewriting intermediate state through DDR for every token.
11. Strided row assignment (`row += num_slices`) balances work, but contiguous block assignment can improve locality for large rows. Both should be benchmarked.
12. The `k` loop can be unrolled by 2 or 4 to reduce helper calls, splats, and repeated scratch load/store.
13. FMA-form HVX intrinsics should replace explicit multiply plus add where available.
14. `vmemu` is safe but pessimistic. If tensor rows and scratch pointers are 128-byte aligned, an aligned-load/store fast path may help, especially for `value_dim=64` F16.

Phased execution:

- Phase 0: make the current assumptions explicit without destabilizing execution. Keep output signatures on `Flat4`, keep input signatures as `Any` until QHPI layout matching is better understood, and move full shape validation to a future QHPI shape/prepare hook rather than the execute hot path.
- Phase 1: apply low-risk kernel cleanup. Replace scratch `std::fill` with `k == 0` assign accumulation. Contiguous row-block slicing was tested but is not kept because it caused intermittent DSP DMA 1100 failures on SM8850.
- Phase 2: add the `value_dim` tile fallback for small-row cases, then keep it behind a conservative runtime heuristic if benchmarks show regressions.
- Phase 3: handle small `value_dim` HVX occupancy with cross-head packing or a packed state layout.
- Phase 4: reduce state traffic with a local-state/VTCM path for `seq_len > 1`.
- Phase 5: tune the inner loop with `k` unroll, FMA forms, and aligned memory fast paths.

Phase 0/1 execution notes:

- Full execute-time shape guards over all inputs/outputs were tested first. They made QNN prepare succeed but caused intermittent phone-side failures (`QnnDsp` DMA error 1100), so they were removed from the execute path.
- Requiring `QHPI_Layout_Flat4` for rank-4 inputs was also tested and then reverted. The safe current signature keeps inputs as `QHPI_Layout_Any` and outputs as `QHPI_Layout_Flat4`.
- Contiguous row-block slicing was tested with an extra `slice < num_slices` guard, but it still produced intermittent DMA 1100 failures. The stable path remains strided self-slicing: `row = slice; row < batch * heads; row += num_slices`.
- The landed kernel change is the `k == 0` assign accumulation. It removes two scratch zero-fill passes for `kv_mem` and `output_pre` while keeping the previous math and strided QHPI slicing.

Phase 0/1 benchmark:

Shape: `batch=1, ar_len=1, heads=16, key_dim=128, value_dim=128`, `iteration=200`, SM8850.

Artifact: `deltanet-qnn-qhpi-phase01-compare-sm8850/summary.json`.

| Quant label | Variant | Runtime path | Avg ms | Output SQNR dB | State SQNR dB | Output max diff | State max diff |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| fp16 | baseline | decomposed QNN graph | 0.40193 | 63.9606 | 67.7232 | 2.942e-05 | 3.937e-04 |
| fp16 | custom | simple_reg `DeltaNet` full op | 0.65669 | 132.367 | 149.395 | 1.304e-08 | 4.470e-08 |
| fp16 | core | QHPI `DeltaNetCore.fp32.hvx` + native QNN norm/exp | 0.40224 | 66.9835 | 74.1249 | 2.225e-05 | 1.200e-04 |

Final core-only confirmation after rebuilding the cleaned source is in `deltanet-qnn-qhpi-phase01-core-final-sm8850/summary.json`: `0.40716 ms`, output SQNR `66.9835 dB`, state SQNR `74.1249 dB`.

Phase 2 execution notes:

- `DeltaNetCore` now has a value-tile execution path. When enabled, work is split as `(batch * heads) * value_tile_count`, and each slice owns a disjoint `[v_begin, v_end)` range, so `kv_mem`, `delta`, output, and state updates do not race.
- A QHPI cost function was added so the scheduler can see the approximate `batch * seq_len * heads * key_dim * value_dim` cost instead of treating the op as a zero-cost custom kernel.
- Forced value tiling was not a universal win on SM8850. It helped one `heads=1, value_dim=256` run, but it regressed `heads=2` cases and did not reproduce consistently after adding the conservative heuristic.
- The current default therefore only attempts value tiling for `batch * heads == 1` and `value_dim >= 256`; all other cases keep the stable strided row slicing from Phase 0/1.

Phase 2 benchmark:

All runs below use `batch=1`, `ar_len=1`, `key_dim=128`, `iteration=200`, SM8850, quant label `fp16`. The pulled tensors are still `float32` for these core runs.

| Shape | Variant / policy | Artifact | Avg ms | Output SQNR dB | State SQNR dB |
| --- | --- | --- | ---: | ---: | ---: |
| `heads=1, value_dim=256` | baseline | `deltanet-qnn-qhpi-phase02-h1-v256-sm8850` | 0.217145 | 65.8318 | 68.5039 |
| `heads=1, value_dim=256` | forced value tile | `deltanet-qnn-qhpi-phase02-h1-v256-sm8850` | 0.17241 | 69.8078 | 76.1159 |
| `heads=1, value_dim=256` | forced no tile | `deltanet-qnn-qhpi-phase02-h1-v256-notile-sm8850` | 0.22614 | 69.8077 | 76.1159 |
| `heads=1, value_dim=256` | final heuristic + cost | `deltanet-qnn-qhpi-phase02-cost-h1-v256-sm8850` | 0.24116 | 69.8078 | 76.1159 |
| `heads=2, value_dim=128` | baseline | `deltanet-qnn-qhpi-phase02-h2-sm8850` | 0.162725 | 64.5969 | 68.0624 |
| `heads=2, value_dim=128` | forced value tile | `deltanet-qnn-qhpi-phase02-h2-sm8850` | 0.236215 | 67.2893 | 75.9368 |
| `heads=2, value_dim=128` | forced no tile | `deltanet-qnn-qhpi-phase02-h2-notile-sm8850` | 0.15922 | 67.2893 | 75.9368 |
| `heads=2, value_dim=256` | baseline | `deltanet-qnn-qhpi-phase02-h2-v256-sm8850` | 0.20659 | 64.7553 | 68.0933 |
| `heads=2, value_dim=256` | forced value tile | `deltanet-qnn-qhpi-phase02-h2-v256-sm8850` | 0.22161 | 69.8530 | 75.8743 |
| `heads=2, value_dim=256` | forced no tile | `deltanet-qnn-qhpi-phase02-h2-v256-notile-sm8850` | 0.19121 | 69.8530 | 75.8743 |
| `heads=2, value_dim=256` | final heuristic + cost | `deltanet-qnn-qhpi-phase02-final-h2-v256-sm8850` | 0.216015 | 69.8530 | 75.8743 |
| `heads=16, value_dim=128` | final sanity | `deltanet-qnn-qhpi-phase02-final-h16-v128-sm8850` | 0.38504 | 66.9835 | 74.1249 |

Phase 2 conclusion: the value-tile code is correct and useful as an experiment path, but simple value tiling duplicates per-tile `key_dim` work and is too shape-sensitive to enable broadly. The next useful step is to make slice allocation observable or controllable, then try cross-head packing or a local-state/VTCM path instead of splitting value tiles blindly.

True FP16 input experiment:

`deltanet.py` now has `--input_dtype fp16` so the no-PTQ `fp16` quant label can be tested with actual FP16 input tensors instead of float32 tensors. Non-default input dtype is encoded in the case name as `infp16`.

Shape: `batch=1, ar_len=1, heads=16, key_dim=128, value_dim=128`, `iteration=200`, SM8850.

Artifact: `deltanet-qnn-qhpi-truefp16-h16-v128-v2-sm8850/summary.json`.

| Quant label | Input dtype | Variant | Runtime path | Avg ms | Output dtype | State dtype | Output SQNR dB | State SQNR dB |
| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: |
| fp16 | fp16 | baseline | decomposed QNN graph, FP16-preserving | 0.474675 | float16 | float16 | 24.0434 | 21.0673 |
| fp16 | fp16 | core | QHPI `DeltaNetCore.fp16.hvx` + native QNN norm/exp | 0.55826 | float16 | float16 | 64.2675 | 70.9061 |

This confirms the custom core can run with FP16 I/O. A fully FP16-preserving decomposed baseline is faster than the current FP16 custom core, but its recurrent-state SQNR drops sharply. The custom core keeps much better SQNR because it uses FP16 storage with FP32 scratch/accumulate internally. It is still slower than the fp32-input core run (`0.38504 ms` in `deltanet-qnn-qhpi-phase02-final-h16-v128-sm8850`) because `DeltaNetCore.fp16.hvx` still converts state/value to FP32 scratch and reads each state row twice. The next FP16-specific optimization should reduce those conversions, or introduce a controlled native-FP16 accumulate/update mode and measure the SQNR loss explicitly.

FP16 scratch/accumulate experiment:

Added `qnn_custom_ops::deltanet_core_fp16_accum`, mapped to QNN op type `DeltaNetCoreFp16Accum`. This variant uses FP16 I/O, FP16 scratch buffers for `kv_mem`, `delta`, and `output_pre`, and HVX `qf16` multiply/add/sub helpers. The Python benchmark variant is `core_fp16accum`. Its SQNR is still measured against the FP32-accumulate custom-op reference, so the numbers show the accuracy loss from using FP16 accumulation.

Shape: `batch=1, ar_len=1, heads=16, key_dim=128, value_dim=128`, `iteration=200`, SM8850.

Artifact: `deltanet-qnn-qhpi-fp16accum-h16-v128-sm8850/summary.json`.

| Variant | Accumulate / scratch | Avg ms | Output dtype | State dtype | Output SQNR dB | State SQNR dB |
| --- | --- | ---: | --- | --- | ---: | ---: |
| baseline | decomposed FP16 graph | 0.551075 | float16 | float16 | 24.0434 | 21.0673 |
| core | FP32 scratch/accumulate | 0.5568 | float16 | float16 | 64.2675 | 70.9061 |
| core_fp16accum | FP16 scratch/accumulate | 0.682855 | float16 | float16 | 42.5854 | 67.0689 |

For `heads=1, key_dim=128, value_dim=256`, the current FP32-scratch core measured `0.258145 ms` in `deltanet-qnn-qhpi-fp16accum-h1-v256-sm8850`, while `core_fp16accum` measured `0.247565 ms` in `deltanet-qnn-qhpi-fp16accum-h1-v256-only-sm8850` with output/state SQNR `42.4711 / 69.6830 dB`. This suggests FP16 accumulation can help a small single-row case, but it is not a general speed win yet and it costs output accuracy.

## Conclusion

The fused custom op is numerically correct. For fp16, the Hexagon HVX path brings the custom op close to the decomposed QNN baseline while keeping much higher SQNR.

For quantized cases, the fused op still improves SQNR because most recurrent DeltaNet math stays inside one op instead of being quantized between many decomposed ops, but it is not performance-competitive yet. Quantized custom latency remains about `25-27 ms` because the quant path still uses scalar dequantize/update/quantize loops.

The next performance step should continue following the llama.cpp HTP design:

1. Keep QNN native support for standard pieces such as L2 norm/rsqrt/sum where possible.
2. Continue with QHPI `DeltaNetCore` as the main performance path, because it already splits `(batch, head)` rows across HVX slices and keeps norm/exp in native QNN ops.
3. Add VTCM state scratch and DMA-style state movement for `ar_len == 1`, similar to llama.cpp's token-generation path.
4. Tune the QHPI inner kernel: reduce the second full state read/convert, use larger `k` tiles only when it helps, and try FMA-form instructions where available.
5. Add quantized HVX/QHPI kernels instead of scalar dequantize/update/quantize loops.
6. Compare SQNR and latency after each step before adding more fusion.
