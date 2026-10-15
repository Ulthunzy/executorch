# DeltaNet HTP Custom Op Benchmark

> 2026-10-15 清理说明：本文件归档原 `DeltaNet_benchmark.md` 的算子测试记录。历史独立 benchmark 脚本、实验源码/库副本、日志和 PTE 已清理；下方历史命令仅记录当时参数，实验路径不代表当前可用产物。当前集成导出和验证结果见[llama README](../../../oss_scripts/llama/README.md#validation-record-2026-10-15)，完整模型测试历史见[测试记录归档](../../../oss_scripts/llama/README_QWEN35_TEST_RECORDS.md)。


Date: 2026-05-29

## Current package: QHPI kernels only (2026-10-10)

The legacy `src/ops/DeltaNet.cpp` implementation has been removed. The package
now registers only `DeltaNetCore`, `DeltaNetCoreVtcm`, `DeltaNetCoreStream`, and
`DeltaNetCoreFp16Accum` through `src/ops/DeltaNetQhpi.cpp`. Package metadata and
Python export mappings match those four operators. The standalone benchmark
defaults to `--variants baseline,core`; the full linear-attention benchmark
continues to use `--fused_recurrence`, with optional `--state_vtcm`.

The older full `DeltaNet`, packed, and probe results below are historical
measurements from the removed implementation.

Validation after removal: clean x86, Android arm64, and Hexagon v79 builds
passed, and a fresh `use_16a4w` core graph exported successfully. The cached
128-token full-layer DDR and VTCM graphs each ran five warmups and three
measured iterations with the rebuilt package; all three outputs remained
byte-for-byte identical to the saved baseline. These were functional checks,
not a new performance comparison. Results are preserved in
`qwen35-linear-attention-core-fast/experiments/qhpi_only_ddr` and
`qwen35-linear-attention-core-fast/experiments/qhpi_only_vtcm`.

## SM8750 128-token prefill: vectorized setup and state update (2026-10-09)

Continued from the 5.55 ms register-reuse implementation of the original
`DeltaNetCore`. The state remains `[B,H,K,V]`, and token recurrence remains
sequential. Target: Qwen3.5-0.8B layer 0, batch 1, 16 heads, K=V=128,
`use_16a4w`, SM8750/v79, QNN `2.49.40.260810`, Hexagon SDK `6.7.0.0` /
Tools `21.0.04`, device `3B1F57E9B8L74X5D`.

Changes in `DeltaNetQhpi.cpp`:

- Vectorize Q/K centering, key range checks, and the exact 64-bit Q/K dot.
  Unsigned products are split into low/high 16-bit halves before reduction,
  then the affine zero-point correction is applied in 64 bits.
- Use non-saturating HVX multiply-accumulate instructions for state
  reductions. Aligned state pointers use aligned loads and stores; other
  pointers retain the unaligned path.
- Update state with unsigned halfword products for nonnegative decay
  coefficients. Centering becomes a per-token offset, delta lane rearrangement
  occurs outside the K loop, and the output pack directly restores channel
  order. Negative coefficients retain the original row update path.
- Reuse delta vectors and coefficients in the F32 fallback. The key/state
  quantization-scale ratio is calculated once outside the token loop.
- Process two 64-value vectors together where alignment and tile size permit,
  sharing K loads and coefficient preparation. Assign contiguous head ranges
  to workers; this improves the measured VTCM version's worker balance for
  this layer and input.
- Remove shifts by zero and combine repeated range checks with Q/K preparation.

Full linear-attention timing, profiling disabled, five warmup iterations and
100 measured iterations per run. Each round runs all four variants in order.
DDR variants use the same PTE with only the HTP library swapped; the VTCM
variant uses the corresponding PTE with VTCM state placement. All twelve
runs produce three outputs byte-for-byte identical to the original baseline.

| Round | Original DDR (ms) | Previous register DDR (ms) | Optimized DDR (ms) | Optimized VTCM (ms) |
| --- | ---: | ---: | ---: | ---: |
| 1 | 7.69482 | 5.56083 | 3.28848 | 2.79272 |
| 2 | 7.71292 | 5.52387 | 3.32306 | 2.78274 |
| 3 | 7.72505 | 5.54263 | 3.28081 | 2.80123 |
| Mean | 7.71093 | 5.54244 | 3.29745 | 2.79223 |

Optimized DDR takes 40.51% less time than the previous register version.
Optimized VTCM takes 63.79% less time than the original approximately 7.7 ms
full layer. VTCM is beneficial with these optimized loops; the older VTCM
experiment below predates them.

Separate timer instrumentation around each worker's head loop uses the
19.2 MHz hardware counter. A single measured invocation after five warmups
gave a slowest-worker interval of 1.48490 ms for DDR and 1.33380 ms for VTCM.
These are internal GDN worker intervals, excluding QNN scheduling and output
transfers, rather than complete operator or layer latency. The instrumented
build places timing markers in final-state output slots; it is retained only
as a diagnostic artifact and is absent from the production source/library.
The complete linear-attention layer has not reached 2 ms.

The final DDR and VTCM libraries also pass the separate 128-token graph with
random nonzero initial state: attention output and final state match the
previous core byte-for-byte. Its timing includes Q/K normalization and graph
overhead, and uses different inputs from the real-weight full layer.

Reproduce the fastest full-layer version with the rebuilt v79 package:

```bash
source /home/hunter/anaconda3/etc/profile.d/conda.sh
conda activate et_npu
source env.sh
python examples/qualcomm/oss_scripts/linear_attention.py \
  --build_folder build-android --device 3B1F57E9B8L74X5D \
  --soc_model SM8750 --backend htp --ar_len 128 \
  --iteration 100 --calibration_samples 2 --quant_dtypes use_16a4w \
  --fused_recurrence --state_vtcm --profile_level 0 \
  --workspace /data/local/tmp/executorch/qwen35_core_fast_vtcm \
  -a qwen35-linear-attention-core-fast-vtcm
```

Omit `--state_vtcm` to use optimized DDR. The three-round timings, output
checks, library hashes, standalone checks and instrumented worker intervals
are in `qwen35-linear-attention-core-fast/final_comparison.json`.
Source and the final library are preserved in that directory's
`libraries/optimized` subdirectory; `bench_compare.py` reproduces the
same-PTE library-swap measurements against the prepared device workspaces.

Rejected experiments in this iteration: independent 64-value tasks across all
tokens, zero-point correction hoisting in the state reduction, K-loop unroll
by two, and completing each 64-value block before the next block within a
token. They preserved the tested outputs but did not improve full-layer time.

## SM8750 128-token prefill: original-core register reuse (2026-10-09)

Optimized the original approximately 7.7 ms `DeltaNetCore` implementation.
The state layout remains `[B,H,K,V]`, with the same integer arithmetic,
rounding, and quantization parameters. Target: Qwen3.5-0.8B layer 0,
batch 1, 16 heads, K=V=128, `use_16a4w`, SM8750/v79, device
`3B1F57E9B8L74X5D`, QNN `2.49.40.260810`, Hexagon SDK `6.7.0.0` /
Tools `21.0.04`.

Two changes in the QUInt16 HVX path:

- For each 64-value tile, keep both K/state and Q/state reduction
  accumulators in four HVX registers across the complete K dimension.
  Shuffle lanes and write the scratch results once after the reduction.
- During state update, load the tile's delta vectors once and reuse them
  across the complete K dimension. Per-token update coefficients are also
  prepared outside that loop.

The original fallback remains for unsupported tiles or quantization ranges.
Each token still traverses state twice, for reductions and then updates.
The optimization reduces temporary-array traffic; these tests do not measure
actual DDR or VTCM bandwidth.

Full linear-attention timing, profiling disabled, the same PTE with only the
HTP library swapped, five warmup iterations and 100 measured iterations per
run:

| Round | Original core (ms) | Register reuse (ms) |
| --- | ---: | ---: |
| 1 | 7.71997 | 5.56112 |
| 2 | 7.71601 | 5.51517 |
| 3 | 7.73317 | 5.57364 |
| Mean | 7.72305 | 5.54998 |

The full layer takes **28.14% less time**, a **1.392x speedup**. All three
full-layer outputs are byte-for-byte identical to the original core.
A separate 128-token test with random nonzero initial state also produces
identical attention output and final state. That isolated graph, including
Q/K normalization, measured 4.23949 ms; it is a separate workload and cannot
be subtracted from the full-layer timing to infer projection time.

Reproduce with the rebuilt v79 package:

```bash
source /home/hunter/anaconda3/etc/profile.d/conda.sh
conda activate et_npu
source env.sh
python examples/qualcomm/oss_scripts/linear_attention.py \
  --build_folder build-android --device 3B1F57E9B8L74X5D \
  --soc_model SM8750 --backend htp --ar_len 128 \
  --iteration 100 --calibration_samples 2 --quant_dtypes use_16a4w \
  --fused_recurrence --profile_level 0 \
  --workspace /data/local/tmp/executorch/qwen35_core_register \
  -a qwen35-linear-attention-core-register
```

The register path is selected automatically by `--fused_recurrence` when
the tile and quantization checks pass. Paired results, library hashes and
correctness checks are in
`qwen35-linear-attention-core-register/register_final_comparison.json`.
The original and optimized HTP libraries are preserved under that artifact
directory's `libraries/baseline` and `libraries/register` subdirectories.

## SM8750 128-token prefill: VTCM state experiment (2026-10-09)

Target: Qwen3.5-0.8B layer 0, batch 1, 16 heads, K=V=128,
`use_16a4w`, SM8750/v79, device `3B1F57E9B8L74X5D`, QNN
`2.49.40.260810`, Hexagon SDK `6.7.0.0` / Tools `21.0.04`.
The full block uses real layer weights and deterministic random inputs,
seed 666, with two calibration samples. QKV/Z fusion is enabled in both versions.

Added the opt-in `DeltaNetCoreVtcm.quint16.hvx` kernel in `DeltaNetQhpi.cpp`:

- State output has `QHPI_MemLoc_TCM_Only`; other inputs and attention output
  retain their DDR placement. QNN manages the final state transfer to the
  graph output.
- Each worker processes a complete head or a V-axis tile with the full K
  dimension. It copies initial state codes into VTCM once, then runs all 128
  tokens against the same resident state region.
- The first step retains the input state's quantization parameters; subsequent
  steps use the output state's parameters. Arithmetic, rounding, and HVX
  fallback rules are shared with the DDR kernel.
- Each head is 32 KiB. QNN reserves the complete 16-head state output, 512 KiB,
  in VTCM; this version does not recycle a smaller per-worker scratch buffer.
- A token still needs two local state traversals: combined K/Q reductions,
  followed by state update. Intermediate states remain in VTCM.

Full-layer timing, profiling disabled, three paired rounds of 100 iterations:

| Round | DDR fused core (ms) | VTCM fused core (ms) |
| --- | ---: | ---: |
| 1 | 7.89603 | 8.00964 |
| 2 | 7.91088 | 8.01856 |
| 3 | 7.75871 | 7.94162 |
| Mean | 7.85521 | 7.98994 |

The VTCM version was 1.72% slower in the paired measurements. It is retained
as an experiment; the default remains the DDR fused core.

Separate single optrace captures gave GDN critical-path contributions of
6.10805 ms (DDR) and 6.97796 ms (VTCM), with measured graph clocks of
2.11200 GHz and 2.11201 GHz. These are profile contributions, not the
100-iteration averages above. QHAS confirms six QUInt16/HVX slices and VTCM
state output placement. Its tensor-byte accounting does not count repeated
state accesses inside the custom C++ loop, so it cannot establish the actual
DDR bandwidth of this recurrence.

Correctness checks:

- All three full-layer outputs are byte-for-byte identical between versions.
- A separate 128-token core test with random nonzero initial state also has
  identical attention output and final state. That isolated test, including
  Q/K normalization, measured 6.47191 ms (DDR) and 7.10066 ms (VTCM) over
  100 iterations; it uses different inputs from the real-weight full layer.

Reproduce the VTCM full-layer run with the built v79 package:

```bash
source /home/hunter/anaconda3/etc/profile.d/conda.sh
conda activate et_npu
source env.sh
python examples/qualcomm/oss_scripts/linear_attention.py \
  --build_folder build-android --device 3B1F57E9B8L74X5D \
  --soc_model SM8750 --backend htp --ar_len 128 \
  --iteration 100 --calibration_samples 2 --quant_dtypes use_16a4w \
  --fused_recurrence --state_vtcm --profile_level 0 \
  --workspace /data/local/tmp/executorch/qwen35_linear_attention_vtcm \
  -a qwen35-linear-attention-vtcm-v79-t128-use16a4w-sm8750
```

Omit `--state_vtcm` for the DDR implementation. For a new optrace export,
use `--profile_level 3 --optrace` and a separate artifact folder.

Results: `qwen35-linear-attention-vtcm-v79-t128-use16a4w-sm8750/vtcm_comparison.json`.

## SM8750 128-token prefill: token-major state-row experiment (2026-10-09)

`DeltaNetCoreStream` uses a physical state layout of `[B,H,V,K]`. For each
token and each value channel, the logical state column `S[:,v]` is one
contiguous K-element row: it is loaded once, used for the `k^T S[:,v]` and
`q^T S[:,v]` reductions, updated with `decay*S[:,v] + k*delta_v`, and stored
once. The dot products and K-element state update use HVX for dimensions at
least 64; smaller dimensions retain the scalar path.

The stream kernel was checked with a non-square standalone case (`K=16,
V=32`) and with the real Qwen3.5 layer (`K=V=128`). It produces the expected
transposed state shape and similar quantization error. After fusing the two
K-reductions into one state-row scan and moving the hot scalar path to float,
the real layer measured 54.27910 ms for the isolated recurrence versus
6.47191 ms for the existing V-major DDR kernel. The full linear-attention block
measured 54.87905 ms versus 7.71547 ms for the existing fused core. This is a
data-flow validation, not a performance win: processing one state row at a time
causes many more K-vector loads than the existing kernel's V-vector traversal.

Run it with:

```bash
python examples/qualcomm/oss_scripts/linear_attention.py \
  --build_folder build-android --device 3B1F57E9B8L74X5D \
  --soc_model SM8750 --backend htp --ar_len 128 \
  --iteration 20 --calibration_samples 2 --quant_dtypes use_16a4w \
  --fused_recurrence --stream_recurrence --profile_level 0 \
  --workspace /data/local/tmp/executorch/qwen35_linear_attention_stream \
  -a qwen35-linear-attention-stream-v79-t128-hvx
```

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
# 2026-10-13：完整精度归约与门控精度修复

单层 128-token prefill 的设备误差测试定位到两个来源：旧快速归约在
Q/K/state 乘法前丢掉 4 个低位，以及 W4 V 投影误差被 RMSNorm/z 加权放大。
该实验的 `deltanetQhpiHvxReduceStateU16` 保留完整中心化值并以 HVX FP32
累加，state 用 UINT16 存储。随后按性能要求将默认实现恢复为快速缩位归约；
完整精度实验可通过 `DELTANET_U16_FULL_PRECISION_REDUCTION=1` 编译启用。

K/z W8、Q/V/A/B W4，相同校准参数与评估文本：

| 设备 SQNR | 旧缩位归约 | 完整精度归约 | 完整精度 + V W4 GPTQ |
|---|---:|---:|---:|
| RMSNorm/z 门控 | 23.77 dB | 28.17 dB | 29.61 dB |
| out_proj 后 attention | 20.03 dB | 24.66 dB | 25.76 dB |
| FFN 后层 hidden state | 17.59 dB | 20.50 dB | 21.00 dB |

同一带 22 个检查点的完整层 PTE，包含 FFN，warmup 10、计时 100 次，
交替两轮平均耗时为 4.731 ms → 6.331 ms。历史 linear-attention 性能表
使用不同输出和图范围，需分别阅读。GPTQ 结果为单层实验，完整模型 PPL
尚未复测；全模型导出入口尚未自动加载该实验的 V weight patch。

完整诊断、逐输入替换、量化范围和原始结果见
[门控精度报告](../../../oss_scripts/llama/README_QWEN35_TEST_RECORDS.md#record-19)。

## 2026-10-13：完整模型 PPL 与整数归约的精度修复

第 7 层 Q/K 的 U16 ReduceMean 执行路径存在明显数值差异；改为二维
native RMSNorm 后，同八个 128-token 窗口的完整设备 PPL 从 102.343 降至
87.398。第 8 层的 GDN readout 则受归约前丢弃低四位影响，完整 FP32
归约进一步把 PPL 降到 50.100，state 仍为 UINT16。

`DELTANET_U16_SPLIT_REDUCTION=1` 实现三项整数 dot：高位积、交叉项和
低位积，最后以 FP32 合成；K≤128 时这些 int32 项不溢出。一次处理
64 个 V 的版本，PPL 为 50.076；同轮三输出纯 attention 为原缩位版
2.947 ms、FP32 归约 4.527 ms、新整数版 3.550 ms。新整数版增加约
20.5% 耗时，保留为明确启用的选项。Makefile 支持通过
`HEXAGON_CPPFLAGS=-DDELTANET_U16_SPLIT_REDUCTION=1` 编译该路径。

完整配置、同输入算子证据、量化范围及运行库/PTE 产物见
[PPL 修复报告](../../../oss_scripts/llama/README_QWEN35_TEST_RECORDS.md#record-12)。

## 2026-10-13：所有 Linear 回退 W4，恢复快速 VTCM

默认恢复 int32 缩位归约。单层实验取消 K/z 拆分，qkv_z、a、b、out_proj、
FFN w1/w2/w3 全为 block-32 W4；depthwise conv W8、激活和 cache UINT16。
校准使用 CUDA，8 个 WikiText 128-token 窗口；本轮不使用 GPTQ patch。

性能图只包含完整 linear-attention，输入为前置 RMSNorm 后的 hidden 和两个
cache，输出为 attention、conv cache、recurrent state。FFN 和精度检查点
从计时图中删除。相同 8 个输入，warmup 5、计时 100，交替三轮：

| 版本 | 平均耗时 / ms |
|---|---:|
| 所有 Linear W4，快速 VTCM | 3.07860 |
| K/z W8，V W4 GPTQ，快速 VTCM | 3.10889 |
| K/z W8，V W4 GPTQ，完整精度归约 | 4.66706 |

原历史 VTCM PTE 与原输入复测为 2.68531 ms，三个输出均与当时基线逐字节
一致。当前新 W4 图采用不同输入、校准、L2 和 block-32 量化配置，实测
3.07860 ms，尚未达到 2.73 ms；两者分别记录。

当前全 Linear W4 检查点图的设备 SQNR：GDN 输出 29.924 dB、state 输出
20.896 dB、norm/z 20.968 dB、attention/out_proj 14.454 dB、完整层 hidden
state 12.735 dB。真实三输出 PTE 复核得到 attention 14.459 dB、state
20.897 dB。

精度配置、量化范围、统一口径计时和复现命令见
[全 Linear W4 报告](../../../oss_scripts/llama/README_QWEN35_TEST_RECORDS.md#record-06)。
