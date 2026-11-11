# DeltaNet HTP Custom Op Benchmark

This directory owns GDN kernels only. The shared HTP build and package
registration live in [htp_op_package](../../htp_op_package/README.md).
Attention is maintained separately in
[attention_op_package_htp](../../attention_op_package_htp/AttentionOpPackage/README.md).

## Source boundaries

`src/ops` owns QHPI tensor adapters, validation, rewrites and registration.
`src/kernels` owns recurrence and chunk computation.
`src/utils` owns common definitions, configuration, profiling and HMX tile
layout helpers. The unused decode-fusion bridge and its `src/interfaces`
header have been removed.

| Operator source | Responsibility |
|---|---|
| `DeltaNetQhpi.cpp` | FP16/FP32/U16 recurrence adapters and the fused AR1 entry |
| `Chunk.cpp` | AR128 chunk validation, VTCM reservation and HVX/HMX dispatch |
| `ChunkStateUpdate.cpp` | Standalone state-update operator |

The shared package's `include/OpDefinitions.h` is the operation-name and
input/output-count catalog used by both package discovery and validation.
`include/OpRegistration.h` declares registration entry points. The package
identity and retained kernel names stay compatible with serialized contexts.
`Chunk` replaces the experimental `ChunkRound2` interface with different
inputs. Contexts containing `ChunkRound2` require their original package or
re-export; the new package does not alias the old interface.

The unused `ChunkReadout` and `QuantizedGemv` operators have been removed.
Chunk retains their former shared packing code as `utils/HmxTileLayout.h`;
`make test-hmx-tile-layout` checks it. Historical measurements below remain
as records, including experiments that are no longer build targets.

The 2026-11-11 cleanup also removes `GdnPrepared*`, standalone `GdnTriangle*`,
`GdnGatedNorm`, `GdnWindowLookup` and `GdnLosslessLookup`, together with their
exclusive helpers and tests. The shared bundle now registers 19 operations,
down from 33. These interfaces are absent from the current llama export.
Historical `GdnPreparedCached` **did** improve one full-model decode test by
about 2%; it is removed because the current entry does not use it, not because
that measurement was invalid. GatedNorm/Post and compressed/window lookup
experiments did not establish a stable gain. Current `Chunk` performs its own
triangular solve; it does not call the removed standalone operator.

Evidence remains in the historical sections below and in
[`qwen3_5-2b-mnn-fusion`](../../../../../llama-output/qwen3_5-2b-mnn-fusion/).
The source snapshot before this cleanup and its validation results are in
[`qwen3_5-interface-cleanup-20261111`](../../../../../llama-output/qwen3_5-interface-cleanup-20261111/).
Contexts containing removed operations require their original package or a
new export. Retained operations keep their serialized names.

The historical Round2 Qwen3.5-2B contexts below used these build definitions:

```text
-DDELTANET_U16_FULL_PRECISION_REDUCTION=0
-DDELTANET_U16_SPLIT_REDUCTION=1
-DDELTANET_U16_KARATSUBA_REDUCTION=0
-DDELTANET_U16_FIXED_DECODE=1
-DCHUNK_ROUND2_VECTOR_WEIGHT=1
-DCHUNK_ROUND2_ACTIVE_WORKERS=4
```

Use the matching definitions when loading those contexts; worker/layout
variants have distinct serialized kernel names. General SDK export defaults
and optional operators are retained for other graphs.

## Fused AR128 Chunk

`src/kernels/Chunk.h` contains the validated AR128 computation, including
Gram/QK matmuls, fused decay/mask and tile packing, FP16 W recurrence and
paired triangular solves, correction, readout and the state-update epilogue.
The prefix product retains FP32 accumulation. Correction stays in HMX tile
layout until its consumers finish. Six HVX workers share the HMX queue and
reserve 3,172,352 bytes of VTCM per invocation.

`src/ops/Chunk.cpp` contains only the QHPI adapter. Its public `Chunk` operation
is rewritten into `ChunkReserve` and `ChunkExecute`; these are workspace and
execution entries, not two token chunks. Each AR128 GDN layer calls `Chunk`
once. Internal 32-row triangular leaves and dependent 64-row correction
panels remain part of that invocation.

All seven inputs are FP16. Query, key, value, state and transposed key have
shape `[B,H,128,128]`; decay and beta have shape `[B,H,1,128]`. The two outputs
are attention and updated state, both `[B,H,128,128]`. The transposed-key
input must be the transpose of key. Supported execution is HTP v81 with
`0 < B*H <= 256` and K=V=128.

`ChunkGDN(fused_chunk=True)` and `chunk_package(host, device)` expose this
interface to Python and the QNN exporter. The llama SM8850 prefill export
selects it automatically for AR128/K=V=128; other shapes and targets keep
the portable chunk path. On v81, that path also selects fused state update
automatically; `Chunk` already includes it. Rebuild the package and re-export
to use it.

`make test-chunk` checks HMX layout, weighting across all 128 rows and FP16
epilogue rounding. `custom_op/tests/test_chunk.py` checks the single-call
export, unsupported shapes, padding and recurrent numerical behavior.
The SM8850 migration comparison uses the same frozen complete layer, weights
and quantization: three rounds of 384 calls, excluding the first 128 per
round, give 1.645 ms for the experimental implementation and 1.643 ms for
`Chunk`. Zero/nonzero-state outputs and all eight feedback blocks are
byte-identical. This is a single-layer synthetic-input check, not full-model
throughput or PPL. Commands, raw samples, package hashes and final-build
checks are in the
[migration report](../../../../../llama-output/qwen3_5-2b-gdn-chunk128-20261111/w_triangle_200/promote_chunk/report.json).

## Fused AR1 decode gate

`DeltaNetCoreGatedVtcm` replaces the calibrated decay subgraph and the usual
U16 recurrence with one operator. The recurrence indexes an SDK-generated
table using the original activation code; it retains the existing projection,
delta/output and state-update arithmetic. The table contains every U16 input
code for each head and must be generated with the actual AR1 SDK graph.
Generating it with a larger tensor can change SDK rounding at some codes.

The additional inputs are a flat DDR table stored as `[1,H,256,128]` INT32
words containing consecutive little-endian U16 codes, and FP32 grids
`[activation_scale, activation_zero, decay_scale, decay_zero]`. A 16-head
table occupies 2 MiB per layer; 18 layers add 36 MiB of constants. The kernel
indexes each head's table directly rather than copying the table into VTCM.

The llama Qwen3.5 entry registers and selects this interface automatically on
SM8850/v81. `prepare_gate_tables` generates tables on the target with the current
SDK and caches them in `<artifact>/decode_gate_tables`; cache misses require
ADB access and `ANDROID_NDK_ROOT`. `fuse_decode_gates` validates arithmetic and
constant fingerprints, input/output encodings, table size and SHA256 before
embedding the tables in the graph. Generation happens during export, and adds
no inference-time table generation or warmup. Other supported SoCs retain
`DeltaNetCoreVtcm` with SDK gates.

## Fused ChunkGDN state update

`ChunkStateUpdate` computes
`decay * state + key @ (weight * correction)` from five FP16 tensors. Their
shapes are `[B,H,K,V]`, `[B,H,K,C]`, `[B,H,C,V]`, `[B,H,C,1]` and `[B,H,1,1]`.
K must be a positive multiple of 32 up to 256, V a multiple of 64 up to 256,
and C a multiple of 32 up to 128.

On v81, HVX weights correction while packing the HMX operand. One request per
head batches its 32-row matrix tiles. The matrix result stays in VTCM; HVX
applies decay to the old state and adds the product before writing the new
state. Weighted correction, the matrix result and decayed state are not
separate graph tensors. Weight multiplication, matrix output, decay
multiplication and final addition retain their individual FP16 rounding;
the epilogue does not use FMA. Chunk size, triangular inverse and recurrent
state quantization remain unchanged.

For K=V=128 and C=64, the shared queue and eight worker buffers use 530 KiB
VTCM. The queue implementation is shared with Attention through
`htp_op_package/include/HmxDispatch.h`. Other architectures retain a reference
implementation; the optimized llama export is selected on v81.

Use `ChunkGDN(fused_state_update=True)` and register `state_update_package`
when lowering a standalone graph. The llama entry point selects fusion on v81
automatically. Fusion runs after calibration
when generation replaces the prefill recurrence cores. It does not alter the
decode recurrence. Rebuild the shared package and re-export the PTE to use
the new operator.

`make test-chunk-state-update` checks packed row-pair layout and separate
FP16 decay/add feedback with SDK libnative. Python export and feedback tests
are in `custom_op/tests/test_chunk_state_update.py`. Device numerical and
complete-model measurements are required in addition to these host checks.

## Optional typed FP16 recurrent state

`DeltaNetCoreFp16State` accepts affine QUInt16 Q/K/value/decay/beta and a
genuinely typed Float16 recurrent state. It returns QUInt16 attention and
Float16 state, with FP32 scratch arithmetic and Half feedback after each
token. The Python alias is `qnn_custom_ops.deltanet_core_fp16_state`.
This opt-in alias does not change existing kernels or model defaults. It
requires matching Float16 cache metadata and correct output-name binding;
an existing QUInt16 state cache cannot be reinterpreted as Float16.

On SM8850/v81, aligned K=V=128 projections use a fixed-dimension inline
call to the existing FP32 reduction helper. Constant dimensions let the
compiler specialize addressing and register allocation without changing
the accumulation order or per-token Half feedback. Other shapes and
v79/v85 retain the generic path. The selected implementation was verified
with profiling disabled; the outlined variant performed better only with
instrumentation and was rejected. See the
[focused reduction results](../../../../../llama-output/qwen35-sm8850--1028/gdn-tradeoff-20261028/gdn-reduce-optimized/REPORT.md).

## SM8850 / SDK 2.51.200.261021 (2026-10-27)

The production v81 build retains the radix-16 FP32 reconstruction. On
2026-10-28, the experimental radix-256 reduction reproduced a multi-turn
generation regression: after `你好,你是谁`, `你是qwen3.6吗`, and
`深圳到北京的距离是多少`, greedy decoding repeated `1` 206 times. Only
replacing the GDN library removed the loop while retaining optimized N2.
The radix-256 reduction is now disabled by default on all architectures.
Integer decomposition alone does not preserve the original FP32 rounding.
`make test-reference-reduction` checks 153600 projection lanes bit-for-bit
against the original radix-16 combine, including the endpoint/random inputs
that exposed this difference. The remaining FP16 register, N2, and IO changes
are retained. The isolated-core figures below describe the frozen experimental
libraries and have not been remeasured for this correction. The corrected
complete-model package subsequently completed 16 rounds: 23/24 rates exceed
the supplied SM8975 screenshot, with image256/64 prefill below its reference.
[Corrected full-model results](../../../../../llama-output/qwen35-sm8850--1028/performance-gdn-corrected-20261028/REPORT.md)
retain every measured round and the changed-export comparison limits.

The experimental radix-256 state reduction applies to `K <= 128`.
It writes `S = 256*Sh + Sl` and centers each Q/K coefficient as
`(raw - 32768) + (32768 - zero)`. The shared state sums supply the bias
correction after the K loop. This retains every calibrated integer bit while
reducing the Q/K projection from six to four multiply-accumulates per state
vector. The bounds `128*255 = 32640` and `128*255*65535 = 2139062400` keep
the state sums in signed int16 and the projection terms in signed int32.
The final FP32 reconstruction can round differently from radix-16.

The v79/SM8750 and v85/SM8975 builds retain the previous reduction. Other K
sizes and unsupported zero points use the previous fallback. Define
`DELTANET_U16_RADIX256_REDUCTION=1` in `HEXAGON_CPPFLAGS` only to reproduce
the rejected experiment; it changes FP32 reconstruction and is unsuitable
for production generation. QHPI enum spellings now match SDK 2.48 through 2.51; FP16/FP32
signatures specify contiguous Flat4, Direct, DDR tensors, as required by the
raw-pointer implementation and the SDK 2.51 tensor generator.

Paired SM8850 measurements use the same serialized graph and shared recurrent
state quantization, swapping only the DSP library. Both libraries use the
existing MoE `-O3` and `DELTANET_U16_KARATSUBA_REDUCTION=1` flags. For
B=1, H=32, K=V=128, five DDR rounds of 1000 iterations and three VTCM
rounds of 400 iterations, each after five warmups, gave:

| State placement | Tokens | Original median ms | Radix-256 median ms | Speedup |
| --- | ---: | ---: | ---: | ---: |
| DDR | 1 | 0.450665 | 0.447267 | 1.008x |
| DDR | 128 | 3.739388 | 3.481400 | 1.074x |
| VTCM | 1 | 0.461938 | 0.462728 | 0.998x |
| VTCM | 128 | 3.084865 | 2.695968 | 1.144x |

The one-token difference is within dispatch variability. Internal phase
timings are needed to distinguish kernel changes from launch overhead.
Final-state tensors were byte-identical; maximum attention differences were 8.62e-7 and 1.06e-6.
The 128-token attention SQNR against the FP32 reference remained 64.44 dB.
These isolated graph times include graph dispatch and boundary conversions;
they are not complete linear-attention or MoE model throughput. The MoE
prefill path currently uses ChunkGDN, so this table does not establish a
prefill model speedup.

Reproducible paired measurements are provided by
[gdn_benchmark.py](../../../oss_scripts/moe/qwen35-35B-A3B/tools/gdn_benchmark.py).
Artifacts and rejected experiment source snapshots are under
`llama-output/gdn-sm8850-sdk251-20261027/` in the repository workspace.
With the QNN and Hexagon SDK environment variables set, run
`make test-reduction` for the native HVX numerical checks. They cover aligned and
unaligned state, K=1/16/64/128, K=256 fallback, V=64/128/192/256, endpoint
codes, random codes, and asymmetric Q/K/state zero points against int64 dots.

`DELTANET_PROFILE_PHASES=1` enables per-worker `GDN_PHASE` logs for the
QUInt16 core and typed FP16-state core: setup,
state reduction, delta/output, state update and state copy using the 19.2 MHz
hardware counter. It does not modify tensor outputs. Leave this flag disabled
for performance comparisons; logical traffic estimates alone cannot show DDR
or compute saturation.

The v81 FP16 core also retains the Q/K projections in FP32 registers across
K, sharing the FP16 state load and conversion, and retains delta vectors
across state updates. Multiplication and addition keep their previous order;
state still rounds to FP16 after every token. This is enabled only on v81 by
`DELTANET_FP16_REGISTER_REUSE=1`; v79/v85 keep the original implementation.
Non-vector tails retain the original path. Define the macro as zero to rebuild
the v81 FP16 baseline.

On the same SM8850 layer-0 context, inputs and ExecuTorch driver, three rounds
of 200 executions gave median 8.591355 ms for the original FP16 core and
6.924430 ms with register reuse (1.241x); all four layer outputs were
byte-identical and repeatable. ChunkGDN remained faster at 2.880585 ms, so the
MoE prefill default remains ChunkGDN. These are layer timings, not complete
model throughput. Evidence is under
`llama-output/qwen35-moe-sm8850-sdk251-20261027/pilot-layer0/executor-unified-ab/`.

Run `make test-fp16-register` for native HVX bit checks: FP32 projection scratch
and FP16 updates match the original helpers, and complete recurrence outputs
and final states match builds with reuse disabled over 240 cases. They cover
K=1/16/128/256, V=1/63/64/65/127/128/129/192/193/256, different initial/output
alignments, nonzero initial state, three recurrent tokens and V partitions.

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

## 2026-11-08：GDN 融合与 prefill 调度实验

`GdnFusion.cpp` 提供显式接图的实验算子，现有导出默认路径不变。
`GdnPreparedCore` 合并四 tap INT8 depthwise Conv、SiLU、Q/K 归一化、
beta 与 U16 gated recurrence，限定单 batch、单 token、K=V=128。
Q/K/V 的中间量化参数必须分别传入，不能复用 SiLU 的量化参数解释 V。

`GdnPreparedHistory` 支持 NCHW/NHWC 的五列历史，只计算末尾四列，并复用
已经装填的向量写出下一步 Conv state，避免独立的输入和输出历史切片。
实验接图也支持直接接收 QVAB/KZ 投影及原四列历史；该路径要求相关量化
网格一致，省去 Q/V 拆分、QKV 拼接和五列历史临时张量。
`GdnGatedNorm` 与 `GdnPreparedPost` 分别用于独立和整体融合的门控归一化
对照；本轮实验中它们没有稳定性能收益，未放入整模候选。

`GdnTriangle` 对 32 或 64 阶严格下三角矩阵计算 `(I-L)^-1`，内部保留
FP32 累加，接口为 FP16。选用的 prefill 变体只替换两个 32 阶对角块，
保留跨块 HMX 乘法与既有 FP16 subnormal 规避逻辑，GDN chunk 仍为 64。
`ChunkStateUpdateBatch` 将两个 head 合入一次 HMX 请求，保留为调度对照。

完整层计时必须包含输入/输出投影、Conv history 与门控归一化；单独核心的
收益不能直接推算为整模收益。实验仍有非逐位一致的量化/归约误差，设备
数值对照与短检索通过不能替代 PPL 验证。

本轮 prefill 与 decode 分开计时，未运行完整模型。最终构建
`linear-final2` 在 SM8850 上采用五轮交替测试，每轮 128 次执行，排除前
8 次，记录各轮中位数的中位数。冻结原权重与量化参数，不重新校准。

| Prefill 范围 | AR128 原版 → 候选 | AR256 原版 → 候选 |
| --- | --- | --- |
| GDN 核心 | 579.453 → 450.989 μs | 1047.655 → 833.542 μs |
| 完整 linear-attention 单层 | 2011.620 → 1886.540 μs | 3431.695 → 3231.540 μs |

核心优化将三角逆的系数按行加载到 HVX，再以 `vdelta` 广播，减少内层
循环的标量 VTCM 读取。对角线与其余结果合并为向量写出。AR256 候选
单 token 层耗时为 12.623 μs，相对 AR128 原版的吞吐提升 24.5%，其中
同时包含批量增大与核心优化的收益。表中完整层包含投影、Conv history、
GDN 和 GatedRMSNorm，不包含输入 RMSNorm、残差与 MLP。

Decode 固定 AR1：原版单层 345.027 μs，直接接投影的融合版 335.443 μs，
仍只有约 2.8% 耗时收益。归约时复制 state 的实验无稳定收益，默认关闭。
设备内部分阶段计时显示 Conv/SiLU/QK 准备约 17 μs，首次 state 拷贝
约 0.625 μs；这些时间不包含 QNN 调度和图边界搬运。

`GdnPreparedLossless` 为显式启用的查表实验：SiLU 表以分段线性预测和
4-bit 残差从 128 KiB 压缩到 36 KiB，使用 VTCM/HVX gather 查找。
18 层各 65536 个码离线还原一致，layer 0 设备全码及相对未压缩融合版的
三个输出逐值一致；
但单层 340.104 μs 比直接投影融合版慢，未选用。自定义 U16 GEMV 的
W4/W8 原型也慢于 SDK，保留独立对照，不接入选定单层。

原始日志、构建、复现脚本与精度对照位于
[`linear-block-report.json`](../../../../../llama-output/qwen3_5-2b-mnn-fusion/linear-block-report.json)。

### Decode 前处理继续优化

`GDN_PREP_HOIST_METADATA=1` 将 history 量化参数、数据指针与 A 的输入
量化参数移到 head/channel 循环外。对于 16 heads、8 workers 的 history
融合路径，history 的 QHPI 查询由每算子 384 次降到 16 次。
`GDN_PREP_CACHE_QUANTIZERS=1` 在每个回调中预先计算量化比例倒数，保持
原来的 FP32 运算顺序、量化网格与 U16 输出。两项默认启用。

同一源码分别关闭和启用两项优化，AR1 五轮交替测试：

| 范围 | 关闭 → 启用 |
| --- | --- |
| GDN 回调，包含初始化 | 30.521 → 22.969 μs |
| Conv/SiLU/QK 前处理 | 17.135 → 8.646 μs |
| 独立 GdnPreparedCore 图 | 128.880 → 122.526 μs |
| 直接接投影的完整单层 | 336.276 → 327.552 μs |

前两行使用设备 Qtimer。回调总时间从入口计至最后一个 head 完成，再取
最慢 worker；前处理按每个 worker 两个 head 合计取最大。计时开销包含
在内；回调初始化从 3.594 增加到 4.896 μs，已经计入回调总时间。
后两行关闭内部计时，包含 QNN 执行、调度与图边界成本，不能混为同一范围。
SDK 单层对照为 344.297 → 344.740 μs，未随该优化获得收益。

正常性能对照、四组不同幅度输入及 U16 state 端点测试、分阶段计时对照的
正常输出均逐值一致。未运行整模型。剩余前处理里 SiLU 标量查表约
4.323 μs；512/1024/4096 码 TCM 缓存窗口及两种打包查表原型虽通过数值
检查，但没有稳定的大幅收益，保留显式实验路径，未默认启用。

复现构建为 `prep-control`、`prep-selected`，分阶段构建为
`prep-control-profile-final`、`prep-selected-profile`。原始数据、开关参数、
源码与设备库哈希见
[`decode-prep-report.json`](../../../../../llama-output/qwen3_5-2b-mnn-fusion/decode-prep-report.json)。

### Decode 图加载预计算实验

`GdnPreparedCached` 将形状检查、量化参数、11 个比例倒数及常量 DDR 地址
移到 QHPI 图加载预计算阶段。kernel predicate 要求输入 4–8 是常量，
signature 强制这些输入放在 DDR；动态输入与输出地址仍在每次执行时查询。
实验导出脚本用 `--precomputed` 显式启用，不接入整模默认路径。
当前序列化函数名带 `.v2.hvx`，v1 context 必须配用当时的库，不能混用。

`DELTANET_U16_FIXED_DECODE=1` 仅特化融合适配器的 AR1、K=V=128 recurrence，
保留 radix-16 累加、FP32 重组和量化顺序；通用 recurrence 路径不变。
本轮仍冻结 layer 0 的原权重与量化参数，prefill 沿用上一轮结果。

SM8850、SDK 2.51，五轮交替，每轮 128 次执行，排除前 8 次：

| 范围 | 前一版融合 → 图加载预计算候选 |
| --- | --- |
| GDN 回调，含初始化与两个 head/worker | 21.875 → 16.458 μs |
| 回调初始化 | 4.688 → 0.573 μs |
| Conv/SiLU/QK 前处理 | 8.542 → 7.187 μs |
| 含投影的 linear-attention 单层 | 327.630 → 321.198 μs |

前三行使用带插桩构建的 Qtimer 最慢 worker 时间，各阶段最大值不可相加。
最后一行关闭内部计时，包含 QNN 调度及图 IO；基线是前一版融合实现，
不是原 SDK。单层只节省 2.0%，仍不足以作为大幅提速的整模候选。
本轮没有重导出整模，也不据此推算 tok/s。

三个正常输出在五轮对照、四组不同幅度输入及 U16 state 端点测试中逐值
一致。同一 context 五次重绑定全部动态 IO 到新缓冲区，保留并覆盖旧缓冲
区后，原始输出也全部一致。

额外实验保留显式开关，不选用：

- 1024/2048 码 SiLU 寄存器表用整数线性预测和 1-bit 精确修正，分别占
  264/392 字节；异常 block 与窗口外输入回退到原表。18 层全 65536 码的
  host 检查及 layer 0 设备全码检查通过，但大幅度输入时无稳定收益。
- `GDN_PREP_ACTIVE_SLICES=4/2` 比 8 workers 更慢，默认仍为 8。
- `DELTANET_U16_PREPACK_COEFFICIENTS=1` 和
  `DELTANET_U16_SPLIT_VECTORS=2` 分别尝试复用标量装填和扩大 V 分块；
  单层为 321.355/322.318 μs，对照 320.026 μs，均不启用。
- Karatsuba 的一个或两个 V 分块没有稳定收益，候选构建显式使用
  `DELTANET_U16_KARATSUBA_REDUCTION=0`。

候选的 Q/K × state 归约约 5.000 μs、SiLU 查表约 4.271 μs、state 更新
约 1.771 μs。SDK detailed profiler 另行记录投影和 GDN 节点，原始 cycle
可能包含 worker/fusion 归属，不能直接换算成 wall time 占比。
构建开关、逐轮数据、分阶段数据、校验与源码/库哈希见
[`decode-deep-report.json`](../../../../../llama-output/qwen3_5-2b-mnn-fusion/decode-deep-report.json)。

### 完整 Transformer 单层复测

从同一冻结量化图提取第 0 层，计时范围扩展到输入 RMSNorm、linear attention、
两次残差、post-attention RMSNorm 与完整 MLP，不包含 embedding 与其他层。
三个图均只有一个 QNN delegate。上一版直接使用保留的 `prep-selected` 库，
新版使用 `prep-deep-final` 与 `GdnPreparedCached`，不重新校准。

SM8850、decode AR1，五轮交替，每轮 256 次调用，排除前 8 次，关闭所有
内部计时插桩。固定相同 hidden、history 和非零 state 重复执行：

| 完整单层版本 | 各轮中位数的中位数 |
| --- | --- |
| 未融合前后处理的对照，保留已有 GDN gated core | 625.860 μs |
| 上一版融合，`prep-selected` | 605.260 μs |
| 本轮候选，`prep-deep-final` | 600.417 μs |

新版比上一版少 4.843 μs，即 0.80%；五轮配对均更快，分别少
2.396、3.672、4.088、6.249、5.025 μs。该收益小于只测 linear-attention
部分时的约 2%，未据此接入整模或推算模型 tok/s。
新版与上一版的层 hidden、history、state 三个输出在五轮中全部逐值一致。
未融合对照与融合版存在已有的数值差异，其 hidden/history/state SQNR
分别为 57.04/66.94/70.34 dB，不能称三版全部逐值一致。

提取与导出脚本为 `prepare_transformer_layer.py`、`export.py --scope transformer-layer`；
设备复现为 `benchmark_transformer_layer.py`，逐轮数据、输出校验与库/context
哈希见
[`complete-transformer-layer-report.json`](../../../../../llama-output/qwen3_5-2b-mnn-fusion/complete-transformer-layer-report.json)。

### MNN 思路适配实验：U16 GEMV 与 SiLU

参考 MNN 固定版本 `5b74a931887ba5b4d9f65b222cc18dd1a16599a8` 的
Hexagon GEMV 激活复用、成对 INT4 解包和向量激活函数。其 GEMV 快路径
会动态量化激活到 INT8；本实验保留当前 U16 激活与原量化网格，未采用
A8。源码对照及固定版本文件清单分别见
[`mnn-hexagon-audit.json`](../../../../../llama-output/qwen3_5-2b-mnn-fusion/mnn-hexagon-audit.json)
和 `mnn-reference/sources.json`。

改动均为默认关闭的实验开关，当前所选单层仍用 SDK 投影：

- `GEMV_PAIR_UNPACK=1`：一次 128 字节权重装载的两半均用于计算。
- `GEMV_PREPARED_ACTIVATION=1`：在已有 reserve 节点内预广播低/高字节，
  每次执行刷新；单独开启未带来额外收益。
- `GEMV_CENTERED_ACTIVATION=1`：仅支持 W4、group32 的 metadata kind2，
  打包权重与 scale，省去每组 `sum(weight)`。用 low8、signed high8 与
  可选的第 17 位 carry 精确表示 `U16-zero`；先按组在 int32 中合并，
  再沿用原 FP32 scale 累加顺序。界限为 `32*65535*8 < 2^24`。
- `GEMV_CENTERED_STREAM=1`：每 worker 用紧凑字节流配合寄存器广播，
  避免重复读取展开后的激活向量。需同时开启上述三个开关与
  `GEMV_DMA=1`，仅接受 kind2。

decode AR1、冻结的 Qwen3.5-2B 第 0 层投影，SM8850/SDK2.51。五轮
交替，每轮 128 次，排除前 8 次，取各轮中位数的中位数：

| 投影范围 | 原自定义 GEMV | 紧凑 U16 版本 | 同轮 SDK |
| --- | --- | --- | --- |
| QVAB | 207.032 μs | 165.078 μs | 139.167 μs |
| 输出投影 | 141.355 μs | 121.068 μs | 106.302 μs |

相对原自定义版本分别少 20.26% / 14.35%，五轮均更快；仍比 SDK 慢
18.62% / 13.89%，不接入当前所选单层或整模。QVAB 的权重及元数据
DMA 描述符合计从 6.047 降到 5.039 MiB；这是逻辑 payload，未统计
激活、workspace、其他 IO，也不是硬件 DDR 流量测量。

`GEMV_DMA_TILES=2/4` 的合批实验反而更慢，保持默认 1。
prepared v1、centered v2、batch2/4 和 centered.stream.v1 使用不同 QHPI
函数名；context 与库必须配套，不能复用旧导出结果。导出原型为实验目录
`prepare_projections.py --centered-meta` 与 `export.py` 的 projection scope。
这些测试使用提取的 QNN context，不代表原型 PTE 已具备完整 ET IO 契约。

SiLU 的两项实验也未选用：

- `GDN_PREP_VECTOR_SILU=1`：负指数多项式加 reciprocal Newton 迭代。
  18 层全 U16 码范围测试对数学参考最多差 1 码，对 SDK 表最多差
  2–3 码；decode core 122.500 → 124.349 μs，未获得收益，也未测 PPL。
- `GDN_PREP_FULL_GATHER=1`：128 KiB 原始表放入 TCM，用两个 64 KiB
  bank 做 gather，混合 bank 输入也精确。查表子阶段 4.427 → 3.542 μs，
  但回调总时长 16.458 → 16.406 μs，关闭内部计时的整个
  linear-attention block 为 321.042 → 323.568 μs，未改善。
  该版本使用 cached v3.gather 函数名，动态查询 TCM 表地址，不缓存
  staging 指针；需要 state 输入输出分离。

66 组正常/压力/批量及 profile 输出对照全部精确，其中 profile 只比较
三个模型输出；全表 gather 另检查乱序的全部 65536 码。centered 与
stream 各投影通过同一 context 内 A→零输入→A 的五轮刷新检查；五个
context 均通过保留并覆盖旧缓冲区后的五轮 IO 重绑定。默认及实验构建
通过 host GEMV 测试，默认库通过 x86/Android/v81 编译。

prefill 本轮未改动或复测。MNN 更值得继续借鉴的是整个 chunk 的 phase
融合与 HMX 中间 tile 复用。当前 `ChunkStateUpdate` 已经每次 op 只锁一次
HMX；后续目标应是跨 Gram、readout、三角乘积和 state update 保留 tile
与调度，而不是重复修改该锁。外层 AR128/256 分开测试，不等同于增加
内部 chunk 长度。

复现、逐轮数据、数值检查及源码/库/context 哈希见
[`mnn-reuse-report.json`](../../../../../llama-output/qwen3_5-2b-mnn-fusion/mnn-reuse-report.json)。
本轮没有新的完整 Transformer 单层、整模 tok/s 或 PPL 结论。

### MNN prefill 适配：Gram 合并与共享 state readout

本实验仅测 prefill。采用 Qwen3.5-2B 的 H16、K=V128 维度和合成的归一化
Q/K 输入，保留现有 chunk64、leaf32 三角求解和 FP16 计算边界。
参考固定版本 MNN 的 `la_hmx_round1/round2/la_hmx_prefill`；MNN 本身使用
chunk32、FP32 持久 state，没有直接移植其精度与 chunk 设置。

实验图变换位于 `llama-output/qwen3_5-2b-mnn-fusion/prefill_pair.py`：

- `--prefill-pair gram` 将 `(K*beta) @ Kᵀ` 与 `Q @ Kᵀ` 的左矩阵按行合并，
  共用右矩阵，拆分结果供原图消费。
- `--prefill-pair readout/both` 用相同方式合并 K/Q 的 state readout，分别
  做消融；仅图级 concat/slice 没有获得稳定的大收益。
- `--prefill-pair kernel` 显式替换为 `ChunkReadout`，在算子内部复用一次
  state 打包，两个结果仍保持原来的 FP16 边界。

SM8850、SDK2.51，五轮交替、每轮 128 次、排除前 8 次，取各轮中位数的
中位数。以下是 GDN 核心，不含投影、Conv、门控归一化或 Transformer MLP：

| 核心范围 | 现有 leaf32 | Gram 图级合并 | 共享 state 的 readout 原型 |
| --- | --- | --- | --- |
| AR128 | 452.084 μs | 446.380 μs | 546.693 μs |
| AR256 | 821.667 μs | 797.110 μs | 1019.375 μs |

Gram 合并减少 1.26% / 2.99%，五轮配对均更快，但收益不足以选用，未测
新的完整 linear-attention 层或整模。readout 原型则明确更慢。

独立 readout 对照包含两个矩阵乘法、结果拼接及 QNN/RPC/图 IO：SDK
146.251 μs，共享 state 原型 205.443 μs。另用
`CHUNK_READOUT_DIRECT_HMX=1` 借鉴 MNN 每次调用只装一次输出 scale 的方式，
内联 HMX 指令；三轮对照仍为 200.912 μs，SDK 为 145.521 μs，未选用。
默认该开关为 0，direct 版本使用单独的 QHPI 函数名与 context。

`ChunkReadoutProfile` 提供独立的诊断输出，正常算子不插入计时。三个进程
末次执行的 Qtimer 快照显示：两次 readout、16 heads 共发起 256 个
32×128×32 HMX 调用；HMX 服务合计 47.396 μs，最慢 worker 回调约
54.635 μs，pack 的每 head 中位数 1.667 μs。部分 HVX 输出读回阶段有
较大的停顿。state 输入和两个输出的运行时位置均为 TCM_ONLY。
这些阶段最大值不可相加，也不能直接用来分摊 SDK 与原型的全部图时间
差异；本实验没有测量硬件 DDR 流量或 SDK 内部的 tile 策略。

67 组正常、消融、压力和 profile 模型输出的 first/final dump 均与相应
对照逐字节一致。压力输入覆盖零初始 state、相关 key、不同 decay/beta、
放大的 value 与非零 state。四个 context 各通过五轮动态 IO 重绑定。
host 检查 2,359,296 个原始 FP16 布局 lane 及回转排列；最终源码通过
x86/Android/v81 编译、Python 编译检查和 diff 空白检查。

当前模型与 `ChunkGDN.forward` 的默认路径未接入这些实验。进一步可验证的
方向仍是跨阶段保留 correction tile，供输出与 state update 的 HMX 乘法
直接消费；本轮只验证了 Gram/readout，未实现整个 chunk 的阶段融合。
复现参数、原始逐轮数据、诊断字段及源码/库/context 哈希见
[`prefill-mnn-report.json`](../../../../../llama-output/qwen3_5-2b-mnn-fusion/prefill-mnn-report.json)。

### MNN round2：correction tile 复用验证

`ChunkRound2.cpp` 提供显式接图的实验算子，将 correction 生成、输出计算和
state update 合在一次 HMX 资源持有期间执行：

```text
C = FP16(inverse @ source)
Y = FP16(readout + FP16(attention @ C))
S' = FP16(FP16(decay * S) + FP16(Kt @ FP16(weight * C)))
```

限定 chunk64、K=V128，输入和计算边界保持原 FP16 定义。`uint16_t` scratch
保存的是 FP16 位模式。correction 写出时直接选择 B 操作数的 tile 顺序，
解决 chunk64 中两个 32-row block 与 N-tile 排列不同的问题，不先转成行式
张量。保留 `weight*C` 的 FP16 舍入，HVX 在 tile 中计算加权副本并复用已经
死亡的 packed Z 空间。后续两次 HMX 乘法分别消费原 C 与加权 C。

MNN 固定版本仍为 `5b74a931887ba5b4d9f65b222cc18dd1a16599a8`，参考
`la_hmx_round2`。MNN 本身使用 chunk32、FP32 持久 state，并在 K 上提前
乘权重；本实验没有直接替换这两个精度与 chunk 设置，也未运行 MNN 整模。

显式实验开关：

- `CHUNK_ROUND2_VECTOR_WEIGHT=1`：一次 HVX 读取 64 个 FP16 权重，用
  `vdelta` 广播 row pair，代替 64 次标量读取。唯一权重 payload 仍为
  128 字节，不将标量指令减少解释成相同比例的 IO 流量减少。
- `CHUNK_ROUND2_ACTIVE_WORKERS=4`：本次 AR256 最好的数值一致候选。
  仍处理全部 16 个 head，只减少并发 head slot。该 op 的 TCM workspace
  从 1,067,008 降至 534,528 字节，包含为消融保留的行式临时区。
- `CHUNK_ROUND2_REPACK=1`：强制 C tile→行式→tile；其算术、调度和
  workspace 与相同 worker 配置的 tile 版一致，用于拆包消融。
- `CHUNK_ROUND2_WEIGHTED_KEY=1`：单独的精度实验，提前给 K 加权，从而
  连续执行三个 HMX 乘法；会改变 FP16 舍入，不作为数值一致候选。

默认新路径仍需显式 FX 替换。构建最佳候选需同时传入
`-DCHUNK_ROUND2_VECTOR_WEIGHT=1 -DCHUNK_ROUND2_ACTIVE_WORKERS=4`，
保留 REPACK、WEIGHTED_KEY、BATCH_PHASES 为 0。不同变体使用不同 QHPI
函数名，context 与库必须配套。

SM8850/v81、SDK2.51，Qwen3.5-2B 的 H16、K=V128 维度与合成输入，
三角求解仍为现有 leaf32。以下计时范围是整个 GDN recurrence 核心，
不含投影、Conv、GatedRMSNorm、MLP 或其他 Transformer 层。

| AR | 现有核心，独立进程 | 新版 4-worker | 耗时下降 |
| --- | --- | --- | --- |
| 128 | 450.443 μs | 438.203 μs | 2.72% |
| 256 | 820.052 μs | 715.365 μs | 12.77% |

上表为五轮交替，每个进程 128 次执行、排除前 8 次，取各轮中位数的
中位数。另增加同一进程内两个 context 交替执行，每个图先 warmup 16 次，
256 对调用逐次反转先后顺序，排除前 8 对，重复三个进程：

| AR | 同进程现有核心 | 新版 4-worker | 耗时下降 |
| --- | --- | --- | --- |
| 128 | 432.266 μs | 424.427 μs | 1.81% |
| 256 | 810.990 μs | 721.614 μs | 11.02% |

两个口径有不同的 context/cache 行为，绝对时间不能混用。AR256 在三个
进程的全部 744 对有效调用中都更快；AR128 仍只有小收益。

AR256 的同进程消融：8-worker 强制重打包版为 760.652 μs，同组对照
810.833 μs；8-worker 直接复用 tile 为 747.891 μs，同组对照 809.766 μs。
4-worker 进一步达到上表的 721.614 μs。不能把全部收益都归给消除一次
拆包；跨节点融合、tile 生命周期和 workspace/head 并发选择也在变化。
独立 round2 微图包含八个外部输入的转换、QNN/RPC 和 IO，4-worker
与当前 state-update 融合版约为 194.166/194.271 μs，基本持平；未声称
HMX 算术本身提速 11%。

提前给 K 加权的实验在 AR256 同进程测试为 728.672 μs，同组对照
811.537 μs，快约 10.21%，但会改变反馈 state。正常输入 state SQNR
约 68 dB，压力输入最差约 61 dB，最大 state 差异 0.01171875；未测 PPL，
未选用。它沿用我们的 FP16 数据，不能等同于 MNN 的 FP32 master state。
2-worker 版更慢，未选用。

验证包含 90 个压力 context，覆盖零 state、相关 key、大小数值、混合
decay/beta，以及有效 token 数 0/1/63/65/AR-1 的 padding。数值一致路径
未发现非零值差异；极小输入的 Y 存在正负零符号位差异，因此不称全部
输出逐字节一致。state 数值一致，各 case 的 first/final 输出确定。
10 个 context 各通过五轮 IO 重绑定及同一 context 内原输入→全零→原输入
检查。host 测试覆盖 4,718,592 个布局、权重乘法和 epilogue lane；默认及
候选的最终源码通过 x86/Android/v81 编译。

等待所有切片就绪的调度实验首轮触发 CDSP SSR，未保留在当前源码中，
归档为 `round2-batch-wait-all.cpp`；新进程的正常路径随后恢复并完成上述
测试。当前可选 BATCH_PHASES 路径只处理已就绪的 wave，不等待未就绪
切片，但没有性能优势，也不推荐启用。不能据该失败推断具体硬件线程数。

提取/导出使用实验目录的 `fuse_round2.py` 和 `export.py --round2`；
`--round2-profile` 另输出每个 chunk 的阶段数据。`benchmark_round2.py`、
`run_pair.py`、`validate_round2.py` 与 `validate_round2_workers4.py` 提供复现。
阶段 Qtimer 记录为各进程末次执行的快照，包含插桩，跨 worker 的最大值
不可相加。未测量硬件 DDR 流量。

默认 `ChunkGDN.forward`、`llama.py` 与模型部署路径未接入该候选。
原始逐轮数据、两种计时口径、精度检查及源码/库/context 哈希见
[`round2-report.json`](../../../../../llama-output/qwen3_5-2b-mnn-fusion/round2-report.json)。

### round2 完整图 1K 验证

将上述数值一致的 4-worker/vector-weight 版本显式接入 24 层
Qwen3.5-2B 完整图。18 层 GDN 的三角求解保持 leaf32；六层 FA 保持
KV8；decode 保持 GdnPreparedHistory。AR128 替换 36 处 round2，
AR256 替换 72 处，不启用提前给 K 加权的舍入变体。

SM8850/v81、SDK2.51，冻结同一份 8K 校准权重和量化参数，不重新校准。
名义上下文 1K，物理 cache 为 1152，输入 1024 token、强制生成 64 token。
四个版本使用同一 runner 和同一算子库，分别预热后交替测试五轮，以下
为中位数；模型加载时间不计入 prefill。

| AR | 现有完整图 prefill | round2 完整图 prefill | 同 AR 吞吐提升 |
| --- | --- | --- | --- |
| 128 | 400 ms / 2560 tok/s | 391 ms / 2619 tok/s | 2.30% |
| 256 | 550 ms / 1862 tok/s | 534 ms / 1918 tok/s | 3.00% |

AR128 现有图范围 399–403 ms，新图 388–397 ms；AR256 分别为
548–552 / 531–535 ms。每个 AR 的五轮配对都是 round2 更快。另一次
仅 AR128 的五轮交替为 402→390 ms，支持收益方向，但不混合两轮数据。
核心测试的 AR256 11% 收益在完整图中只有约 3% 的吞吐提升；AR256
完整图本身还慢于 AR128，因此本轮最快候选为 AR128 + round2。
AR256 整图的额外耗时尚未做逐算子定位，不能归因于 round2 核心。

Decode 中位数：AR128 为 47.94→47.83 tok/s，AR256 为
47.69→47.87 tok/s，没有稳定提升。四个版本的检索全部通过；同一 AR
新旧版本强制生成 64 token 的文本一致，五轮输出确定，未做 PPL 测试。
两种方法均为一个 QNN delegate、零 CPU kernel fallback。

AR256 从冻结图重新展开四个 chunk64，检查了 880 个共享张量、
2,319,499,896 字节，原有权重和量化 buffer 一致，仅
gdn_prefill_positions 从 128 扩展为 256。配套 CPU 4bit embedding PTE
只调整调用方分配的输入输出形状，packed weight/scales 哈希保持一致。

实验目录为 `llama-output/qwen3_5-2b-mnn-fusion`，完整 PTE 分别位于
`full-history-1k`、`full-round2-ar128-1k`、`full-control-ar256-1k`、
`full-round2-ar256-1k`；匹配算子库为 `round2-workers4`。导出入口是
`export_full.py --variant fa --history-state`，用 `--round2` 显式接入，
`--prefill-ar 256` 选择 AR256。AR256 导出前需运行
`retarget_prefill.py` 和 `prepare_embedding_ar256.py`。

```bash
source env.sh
task_python=/home/hunter/anaconda3/envs/et_npu/bin/python
FUSION_BUILD=round2-workers4 "$task_python" \
  llama-output/qwen3_5-2b-mnn-fusion/export_full.py \
  --variant fa --history-state --round2 --label full-round2-ar128-1k
"$task_python" llama-output/qwen3_5-2b-mnn-fusion/benchmark_round2_full.py \
  --repeats 5 --lengths 1024 --tag round2-full-reproduce
```

测试脚本在独立目录部署四份 PTE，核对设备哈希，结束后恢复实验算子库。
默认 llama.py/ChunkGDN 路径未切换；直接运行候选 PTE 需要匹配的
round2-workers4 库，AR256 另需匹配的 embedding PTE。

完整结果、PTE 委派检查、模型与库哈希、逐轮结果、生成文本和恢复状态见
[`round2-full-report.json`](../../../../../llama-output/qwen3_5-2b-mnn-fusion/round2-full-report.json)。

### 完整图 decode 与质量复核

在相同 24 层 Qwen3.5-2B、冻结量化参数和 AR128 round2 prefill 上，
显式接入 18 处直接投影输入的 GdnPreparedCached，并启用
DELTANET_U16_FIXED_DECODE。构建为 `round2-decode-final`，仍保留
4-worker/vector-weight round2。图加载预计算只缓存常量参数；动态 IO
地址每次执行查询。`fuse_projected.py` 支持完整图的 channels-last
clone/permute 包装，并继续检查所有相关量化网格。

SM8850/v81、SDK2.51，名义 1K、物理 cache 1152；896-token 输入后
强制生成 256 token，五轮交替取中位数。使用普通性能 runner，质量评分
驱动的额外计算不参与计时。

| Decode 版本 | 吞吐 | QNN execute/step |
| --- | --- | --- |
| 上一版图与库，GdnPreparedHistory | 47.4513 tok/s | 20.9542 ms |
| 上一版图 + AR1 特化新库 | 47.3986 tok/s | 20.9818 ms |
| 新库 + 直接投影输入/GdnPreparedCached | 48.4115 tok/s | 20.5379 ms |

新版吞吐提高约 2.02%，五轮配对均更快，仍未达到 55 tok/s。
单独的 AR1 特化没有稳定性能收益。本轮 PTE 为
`full-round2-cached-ar128-1k/hybrid_llama_qnn.pte`，两个方法均为一个
QNN delegate、零 CPU kernel fallback。默认导出未启用该候选。

质量使用固定 WikiText-2 test 前 4093 个 token，四个 1024-token 窗口、
stride 1023。无 chat template、无额外 special token。AR1 从每个窗口的
第一个 token 开始 teacher forcing，恰好评分 4092 个 next-token 目标。
另一组固定前 128 token 为 prefill，只评分其后的 3584 个条件目标。
两种 PPL 的目标集合不同，不能横向比较。

| 指标 | 上一版量化 PTE | 仅 round2 prefill | 再接入新 decode |
| --- | --- | --- | --- |
| AR1 PPL，4092 个目标 | 15.497531 | 未重复测量 | 15.470219 |
| 固定 128-token 前缀条件 PPL，3584 个目标 | 13.958712 | 13.958712 | 13.950013 |

round2 prefill 的全部 3584 个逐步 logits 校验值、Top1、NLL 与上一版
一致。新 decode 并非无损：AR1 Top1 一致率 97.4096%，最大逐 token
NLL 差异 0.97965；固定前缀 Top1 一致率 97.4609%。四个窗口的首行
完整 logits 均一致，后续轨迹存在差异。旧图配新库控制组在全部 4092
个目标上 logits 校验值及 NLL 一致，将变化范围缩小到新增图融合路径，
没有进一步区分直接投影输入与图加载预计算各自的数值影响。
该小样本未观察到 PPL 退化，不能据此声称质量提升或代表完整测试集。

独立的 `quality-driver` 在私有 decoder header overlay 中读取采样前的
全词表 UInt16 logits，使用 FP64 logsumexp 等价计算并返回参考 token。
每个后续 decode 的实际输入均核对为上一步参考 token；每个 case 的首行
完整 logits 由 Python 独立复算 NLL。原始 runner/header 未修改。

八组固定提示词的生成文本，上一版、仅 round2、新 decode 三版均一致。
五个可严格判定的任务均为 3/5：三版都将 17×23 回答为 411，正确值
为 391；三版都在要求纯 JSON 时添加代码围栏。翻译、回文 Python 函数
和英文摘要样例已保存，但不视作通用能力 benchmark。已有错误没有被
本次优化修复。

复现入口为 `export_full.py --variant fa --history-state --round2
--decode-cached --label full-round2-cached-ar128-1k`，需设置
FUSION_BUILD=round2-decode-final。`benchmark_decode_full.py` 包含旧图旧库、
旧图新库和新图新库的五轮对照。`build_quality_driver.py` 构建隔离评分
runner，`run_quality.py` 执行固定 token 评分，`generation_quality.py`
执行固定提示词；`decode_quality_report.py` 汇总结果。所有设备阶段均
核对模型/库哈希，结束后恢复原实验库。

逐轮性能、PPL 目标与 NLL、logits 对照、生成文本、PTE/库/驱动哈希见
[`decode-quality-report.json`](../../../../../llama-output/qwen3_5-2b-mnn-fusion/decode-quality-report.json)。
