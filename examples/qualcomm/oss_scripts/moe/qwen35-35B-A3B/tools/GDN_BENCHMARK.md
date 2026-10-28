# SM8850 GDN measurements

`gdn_benchmark.py` exports a single recurrent core with Q/K normalization and
`exp(g)` computed on the host. Defaults match the MoE core after key-head
replication: batch 1, 32 value heads, K=V=128, AR1 and AR128, QUInt16
activations. No model weights are required. This is a core microbenchmark,
not a full linear-attention layer or end-to-end MoE throughput measurement.
`gdn_benchmark_export.py` shares the initial/final state quantization encoding,
matching the recurrent IO constraint. `--independent-state-qparams` is only
for reproducing older generic-benchmark results; separate observers can make
the intermediate state saturate during AR128 recurrence.

Build both libraries with the same SDK, optimization flags, and Karatsuba
setting; preserve the baseline before rebuilding the optimized library.
The MoE build uses `native/gdn_arch.mk` (`-O3` and Karatsuba reduction).
Use an ExecuTorch Python binding built against the selected SDK.

For example, from the repository root, build a baseline into a separate
directory before building the default v81 optimization:

```bash
gdn_pkg="$PWD/examples/qualcomm/custom_op/deltanet_op_package_htp/DeltaNetOpPackage"
moe_root="$PWD/examples/qualcomm/oss_scripts/moe/qwen35-35B-A3B"
make -C "$gdn_pkg" -f Makefile -f "$moe_root/native/gdn_arch.mk" \
  WORK=build-sm8850-baseline HEXAGON_CPPFLAGS=-DDELTANET_U16_RADIX256_REDUCTION=0 \
  htp_x86 htp_aarch64 htp_v81
make -C "$gdn_pkg" -f Makefile -f "$moe_root/native/gdn_arch.mk" \
  htp_x86 htp_aarch64 htp_v81
```

Use `build-sm8850-baseline/hexagon-v81/libQnnDeltaNetOpPackage_HTP.so` as
the baseline and `build/hexagon-v81/libQnnDeltaNetOpPackage_HTP.so` as the
optimized library, both relative to the package directory.

```bash
source env.sh
export QNN_SDK_ROOT=/hunter/qnn_sdk/src/backends/qnn/2.51.200.261021
export LD_LIBRARY_PATH="$QNN_SDK_ROOT/lib/x86_64-linux-clang:$HEXAGON_TOOLS_ROOT/Tools/lib"

python examples/qualcomm/oss_scripts/moe/qwen35-35B-A3B/tools/gdn_benchmark.py run \
  --device 3B15AK00GLW00000 \
  --qnn-binding build/qnn251-host/backends/qualcomm/PyQnnManagerAdaptor.cpython-311-x86_64-linux-gnu.so \
  --op-package-dir /path/to/built/DeltaNetOpPackage \
  --baseline-library /path/to/baseline/libQnnDeltaNetOpPackage_HTP.so \
  --optimized-library /path/to/optimized/libQnnDeltaNetOpPackage_HTP.so \
  --artifact llama-output/gdn-sm8850-paired \
  --capture-profile
```

The package directory must contain `build/x86_64-linux-clang`,
`build/aarch64-android`, and the matching DSP build. To test 8750, use
`--soc SM8750` and v79 libraries. Select `--quant-dtype fp16` for the FP16
path. Keep other inference processes off the device during measurements.

Each shape uses one PTE and identical seeded inputs. Library snapshots and
SHA256 hashes are recorded. The tool alternates baseline/optimized order over
three rounds, with five warmups and 100 measured iterations per round. Both
attention and final state must be byte-identical across all runs by default.
For experimental kernels with different rounding, explicit `--atol` and
`--rtol` enable a numerical comparison while still reporting byte differences.
Inspect each variant's error against the same floating-point reference before
accepting such a kernel. The comparison
reports medians of round averages, range, standard deviation, and speedup.
These averages do not provide per-invocation tail-latency percentiles.
Before and after each measured variant, it records the device SoC,
`dumpsys thermalservice`, and battery status/temperature. Inspect these
snapshots and iteration counts before comparing absolute times across runs.

`--capture-profile` performs a separate detailed-profile export and run using
the optimized library. `profile.etdp` and `profile-cycles.tsv` retain the QNN
events for inspecting core and conversion costs. Values retain native units
shown in event names (cycles, microseconds, or counts); the table's generic
cycle header does not convert these values. Profiling runs are excluded from
latency medians. The regular latency includes graph dispatch and graph
boundary conversions/copies, so it is not an internal HVX worker interval.

The traffic report distinguishes three quantities:

* `minimum_external_bytes`: one initial-state read, one final-state write,
  Q/K/V/decay/beta reads, and attention-output writes.
* `estimated_loop_bytes`: two logical state reads plus one state write per
  token in the fused core, plus the vector IO above. This excludes scratch,
  cache-line effects and repeated Q/K loads from tiling.
* `mathematical_operations`: the recurrence's operation count with an FMA
  counted as two. It is not a count of Hexagon instructions; integer scaling,
  quantization, reductions, and fallback paths have additional costs.

Loop traffic can hit cache or VTCM. Dividing it by elapsed time gives an
effective logical access rate, **not measured DDR bandwidth**. Thus neither
the estimated bandwidth nor mathematical operation rate establishes whether
IO or compute is saturated. To compare against separately calibrated peaks
at the same clocks, dtype, worker count, and memory placement:

```bash
python examples/qualcomm/oss_scripts/moe/qwen35-35B-A3B/tools/gdn_benchmark.py analyze \
  llama-output/gdn-sm8850-paired/comparison.json \
  --peak-gbps 100 --peak-gops 200 --output /tmp/gdn-roofline.json
```

The numbers `100` and `200` above are placeholders, not SM8850 specifications
or measurements. To claim saturation, collect hardware stall, memory, and
vector utilization counters in addition to elapsed time. The tool never
inserts assumed hardware peak values.

## Measurements on 2026-10-27

Device `3B15AK00GLW00000`, SM8850/v81, SDK `2.51.200.261021`. Both libraries
use the MoE `-O3` / Karatsuba settings. The baseline is the original recurrence
algorithm rebuilt with SDK enum and direct/Flat4 signature compatibility
fixes. Shared initial/final state encoding is enabled. Shapes are B=1, H=32,
K=V=128, with deterministic nonzero initial state and calibration seeds
666/667. Times below include graph boundary QDQ, transfers and dispatch.

| State placement | AR | Original median ms | Radix256-full median ms | Speedup | Measured rounds |
| --- | ---: | ---: | ---: | ---: | --- |
| DDR | 1 | 0.450665 | 0.447267 | 1.0076x | 5 x 1000 |
| DDR | 128 | 3.739388 | 3.481400 | 1.0741x | 5 x 1000 |
| VTCM | 1 | 0.461938 | 0.462728 | 0.9983x | 3 x 400 |
| VTCM | 128 | 3.084865 | 2.695968 | 1.1443x | 3 x 400 |

Each row is a same-PTE, alternating-library comparison with five warmups.
DDR and VTCM use separately exported graphs. Combining radix256-full and
VTCM reduces AR128 time by 27.90% relative to the original DDR graph in these
runs; that is a cross-layout comparison, not a same-PTE library comparison.
Decode changes are within the observed timing variation and do not establish
a meaningful decode speedup. Thermal status remained 0; battery temperature
was 33.7–33.9 C during these confirmation runs.

Both final-state outputs are byte-identical to the baseline for all measured
rounds. Attention is not byte-identical: maximum absolute differences are
8.62e-7 (AR1) and 1.06e-6 (AR128). AR128 attention reference SQNR changes from
64.43809 to 64.43813 dB; final-state SQNR remains 58.69443 dB. This validates
these microbenchmark inputs, not complete-model accuracy or long decode
trajectories.

Artifacts are under `llama-output/gdn-sm8850-sdk251-20261027/`:

* `radix256full-confirm/comparison.json`: DDR confirmation, output hashes,
  floating reference error and per-round environment snapshots.
* `radix256full-vtcm/comparison.json`: VTCM confirmation with the same checks.
* `vtcm-selfcheck/`: original-library placement comparison, separate detailed
  profiles, and byte-identical outputs versus the DDR graph.
* `phase/`: diagnostic per-worker phase logs and output hash checks. These
  instrumented single executions are excluded from the timings above.
* `phase-baseline-confirm/` and `phase-optimized/`: final single-invocation
  phase comparisons on the shared-state graphs; outputs match their
  corresponding uninstrumented libraries byte-for-byte.

The final AR128 diagnostic slowest worker accumulates 3.08370 ms in the
baseline and 2.72760 ms with radix256-full. Reduction changes from 1.33573 to
1.00589 ms; state update changes from 1.11161 to 1.09115 ms. Setup is about
0.34 ms and delta/output about 0.29 ms. These are instrumented per-worker
intervals from one invocation without warmups. Workers run in parallel;
their intervals must not be summed, and these numbers do not replace the
production measurements in the table.

The other measured candidates were not selected: one-vector Karatsuba
reduction tiles (AR128 slower by about 6.7%), range-scanning radix256 (about
2.2% slower), next-head prefetch (AR1 about 6.1% slower), plain F32 reduction
(AR128 about 8.2% slower), QF32 accumulation and invariant hoisting (changes
near timing variation). Earlier `tile1`, `radix256`, and `prefetch` artifacts
used separate state observers; their within-run A/B comparisons are preserved
but their absolute times and AR128 accuracy are not representative of the
shared-state configuration.

AR128 traverses approximately 406.9 MB of logical state/vector traffic but
needs only approximately 6.31 MB of minimum external IO. The observed benefit
from VTCM placement confirms that state-access placement matters; the
additional radix256-full benefit confirms that reduction work also matters.
Neither result measures DDR transactions or establishes IO/compute saturation.
These core measurements are not a comparison against the SM8975 end-to-end
MoE tokens/s table.

An external hardware-counter capability check used the SDK's sysMonApp
10.0.29 in default mode (`--q6 cdsp --debugLevel 1 --dcvsOption 1 --duration 3`).
The device returned `Sysmon registering failed; returnval = 256` before any
GDN execution. No AXI, HVX or frequency counter samples were collected, and
no custom PMU mode, DCVS override, system configuration change or APK
installation was performed. The tool's automatic debugfs mount attempt
failed. Raw help, registration output and configuration are retained in
`sysmon-capability/`; saturation therefore remains unmeasured.
