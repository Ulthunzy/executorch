# SM8850 GDN / MoE measurements

Target: SM8850/V81, device `3B15AK00GLW00000`, QNN
`/hunter/qnn_sdk/src/backends/qnn/2.51.200.261021`, Hexagon SDK 6.7.0.0,
Tools 21.0.04. Initial results were collected on 2026-10-27; follow-up
experiments continued on 2026-10-28.

## Current integrated version

当前整合版本为 `sm8850-u16-native512-20261028`，对应刚完成 PPL 和原始对话回放的
U16 GDN state 模型。[版本记录](sm8850-current.json) 固定芯片、SDK、调度、库指纹和验证范围。
FP16 state 保留为另一套可选导出，不替换这次已验证的模型。

| 项目 | 当前值 |
| --- | --- |
| 运行入口 | `run_chat.py --config configs/sm8850.json` |
| 模型配置 | `configs/model_sm8850.json`，KV/GDN/conv state 均为 uint16 |
| 导出配置 | `configs/export_sm8850.json`，支持 `--output` 自定义目录 |
| SDK / DSP | QNN 2.51.200.261021 / V81 |
| 上下文 / AR | 2048 / prefill128 / decode1 |
| 专家 | W2、Top8、每层64槽 |
| IO | prefill16、decode8、512 KiB |
| 构建产物 | `build/moe/qwen35-35B-A3B-sm8850` |

主源码的职责如下，构建和推理不加载实验目录里的代码。

| 路径 | 职责 |
| --- | --- |
| `workflow/config.py`、`configs/sm8850.json` | 入口、路径及运行参数；命令行覆盖配置，配置中的 `cache_experts` 覆盖旧包内的槽数默认值 |
| `runner/chat/` | 分词、对话历史、reset、EOS与流式输出 |
| `runner/prefill/` | 每层 Attention/GDN → Router/shared → Native MoE；缓存反馈与专家 IO 交叠，最后执行 head |
| `runner/decode/` | AR1、命中专家优先、缺失专家异步读取、复用共享 context |
| `native/prefill/IntegerMoE.cpp` | V81 的512维 GU/down 投影专用代码、下一块权重 DMA 与 N2 展开交叠；其它宽度保留通用路径 |
| `native/N2Utils.cpp` | 原地、逆序 N2→N4 展开；V81 unroll4，其它架构保留原路径 |
| `native/prefill/vtcm_layout.h` | V81 的 VTCM 布局、AR128四worker与大组八worker |
| `examples/qualcomm/custom_op/deltanet_op_package_htp/DeltaNetOpPackage/src/ops/DeltaNetQhpi.cpp` | 保持 radix-16 的 FP32 重组及累加顺序，避免原 radix-256 连续数字问题 |

8750/V79 和8975/V85保留各自分支。主配置没有接入图片预处理、异步图片加载或实验 IO 线程池改动。

最近一次精度验证使用固定 WikiText-test 片段：1024 token上下文、后续256个目标、64槽。
PPL为 **5.5612606857700495 → 5.5612606857700495**；FP32参考为4.667211698004229，
logits SQNR为9.50029120768596 dB。七份输入及输出数据逐字节一致。这是固定片段结果，
不是全量语料PPL，也不是另一套FP16 state模型的5.4836。

重放初始介绍、reset及原始三问后，深圳问题生成200 tokens并自然EOS，未复现连续数字1；
两次身份问题仍各输出相同的51 tokens。所有生成ID与修正后基线相同，当前保留这项回答复用。

最近一次单层 Native 微基准中，1024行调用（含raw RPC）为14.374→11.107 ms，
8个HVX worker累计后处理为33.696→19.773 ms。它与整模吞吐是不同口径。
本机主构建目录保存已完成精度、对话和性能验证的快库，`CURRENT_VERSION.json`记录其指纹及源码快照。
完整重新链接库尚未复现同等性能，原因未定位；重新构建的库应单独验证，不能沿用冻结库的性能结论。

从仓库根目录运行当前模型：

```bash
PYTHONPATH="$PWD/.." /home/hunter/anaconda3/envs/et_npu/bin/python \
  examples/qualcomm/oss_scripts/moe/qwen35-35B-A3B/run_chat.py \
  --config examples/qualcomm/oss_scripts/moe/qwen35-35B-A3B/configs/sm8850.json \
  --weights-path llama-output/qwen35-sm8850--1028/model.pte \
  --prompt "你好，请用中文介绍一下你自己。" \
  --max-new-tokens 256 --temperature 0 \
  --no-enable-thinking --no-ignore-eos --no-with-vision --interactive --stream
```

先追加 `--check` 可进行本机资源预检。旧 `device.json` 和构建路径仍可用；旧构建目录
链接到主构建目录。更换模型产物时修改 `--weights-path`，更换本机构建目录时传 `--build-dir`。

历史实验和原始日志保留作证据；下文各矩阵保留各自的模型、输入和版本范围。

## GDN numerical correction on 2026-10-28

Radix-256 is disabled by default. Its integer decomposition preserves all bits,
but its FP32 reconstruction changes the reference rounding order. The user's
exact conversation produced 206 consecutive `1` tokens with radix-256. Replacing
only GDN, while retaining optimized N2, restored the old GDN trajectory.

The corrected library retains Karatsuba/O3, FP16 register reuse, grouped8, N2
and IO scheduling. At temperature zero, the exact replay now generates
141/51/51/200 tokens across warmup and three user turns; the last turn stops at
EOS. All token IDs and 82 hidden/logits/cache files match the old GDN baseline
byte for byte. The model, QNN graphs and quantized weights were not re-exported.
[Fix and installation record](../../../../../../llama-output/qwen35-sm8850--1028/diagnostics/repetition-20261028/FIX_REPORT.md).

The corrected package subsequently completed the full 16-round protocol using
the user's current export. All 8192 generated IDs, 8176 AR1 forwards and 327040
mailbox commands passed independent log checks. Rates are token/s, 32 / 64 slots:

| Input | Input/output tokens | Prefill, 32 / 64 | Decode, 32 / 64 |
| --- | ---: | ---: | ---: |
| Text | 256 / 256 | 164.157 / 184.024 | 11.024 / 13.740 |
| Text | 512 / 512 | 293.777 / 326.131 | 11.385 / 14.168 |
| Text | 1024 / 1024 | 491.477 / 503.393 | 11.150 / 13.884 |
| Image + text | 256 / 256 | 94.629 / 115.552 | 12.257 / 15.543 |
| Image + text | 512 / 512 | 182.922 / 212.184 | 12.594 / 15.738 |
| Image + text | 1024 / 1024 | 334.969 / 380.952 | 12.448 / 16.916 |

23/24 median rates exceed the screenshot, including every decode condition.
Image256/64 prefill remains below 119.20 at 115.5515 token/s. Its two rounds are
116.047 and 115.056; both are retained. Compared with historical R4, average
complete prefill increases by 71.51 ms: CPU preprocessing and vision become
20.69 ms faster combined, while the LLM loop adds 91.48 ms. Native, including
expert readiness/IO waits, accounts for 90.24 ms of that increase. The logical
read volume, expert hits, requests and assignments are unchanged. These
overlapping intervals do not establish storage saturation or a GDN-only cause.

The R1-R4 results below remain frozen historical evidence. The new export
recompiled shared contexts and changed some runtime binaries; input files match
R4 byte for byte, while generated IDs are compared for consistency within the
corrected version. This is an entire-version throughput comparison. Neither
the fixed-workload matrix nor the exact-dialogue replay establishes general
free-generation equivalence. Peak memory was not measured.

[Corrected report and all rounds](../../../../../../llama-output/qwen35-sm8850--1028/performance-gdn-corrected-20261028/REPORT.md),
[independent audit](../../../../../../llama-output/qwen35-sm8850--1028/performance-gdn-corrected-20261028/INDEPENDENT_AUDIT.json),
[phase comparison](../../../../../../llama-output/qwen35-sm8850--1028/performance-gdn-corrected-20261028/PHASE_COMPARISON.json).

## Mainline runtime defaults

The accepted GDN and Native kernels are in the production source/build paths.
The V81 native CLI and JNI now also apply the measured runtime defaults when
loading a bundled configuration: prefill IO 16, decode IO 8 and 512 KiB read
chunks. This covers older PTEs that embed the original 8/256 schedule.
An explicit native `--model-config` preserves its schedule; explicit
`--prefill-io-workers`, `--decode-io-workers` and `--read-chunk-kib` values take
precedence. V79/V85 retain their previous defaults. Host tests cover all three
architectures, explicit overrides and invalid values, including the resulting
environment used by the backend.

The earlier CLI/JNI startup update after R4 left its measured schedule and
Native/GDN libraries unchanged. The subsequent GDN numerical correction is
documented above. The R4 results retain the original measured runner/JNI
identities; the startup update did not claim another device benchmark or require
a model re-export/repack.

[Integration/build record](../../../../../../llama-output/qwen35-moe-sm8850-sdk251-20261027/experiments/mainline-defaults-20261028/BUILD_RESULT.json).

## R4 final optimized Native matrix

The frozen R4 V81 version combines grouped8 dispatch with the backward unroll4
N2 helper in both prefill and decode DSP. The PTE, QNN graphs, GDN op package,
quantization, inputs and cold tables remain unchanged. Runtime remains prefill
IO 16, decode IO 8, 512 KiB chunks, context 2048, AR128, W2/Top8. The fixed
16-run protocol matches R3: two opposite-order rounds at 256, one round at
512/1024.

| Input | Input/output tokens | Prefill, 32 / 64 slots | Decode, 32 / 64 slots |
| --- | ---: | ---: | ---: |
| Text | 256 / 256 | 182.649 / 190.044 | 11.718 / 14.935 |
| Text | 512 / 512 | 299.024 / 337.963 | 11.650 / 14.386 |
| Text | 1024 / 1024 | 514.037 / 516.661 | 11.525 / 14.179 |
| Image + text | 256 / 256 | 100.732 / 119.406 | 12.895 / 15.910 |
| Image + text | 512 / 512 | 189.279 / 218.151 | 13.315 / 16.151 |
| Image + text | 1024 / 1024 | 349.369 / 370.344 | 13.121 / 17.262 |

24/24 median throughput metrics exceed the screenshot thresholds. All 8192
generated IDs match the fixed R2/R3 trajectory; all 8176 decode forwards and
327040 mailbox commands pass. All 16 precommitted measurements are retained.
15/16 individual measurements exceed both rates. Image256/64 records
118.794 and 120.019 token/s; its median 119.4065 exceeds 119.20 by only 0.173%.
The first below-threshold round is retained; a stable margin is not established.
Image prefill includes CPU preprocessing, the vision encoder and LLM. Matching
these historical numbers with new inputs does not isolate a cross-SoC speedup.
Natural cooling follows the registered gate; no thermal policy or speed-based
replacement is used.

The same-input 64-slot comparison against R3 is:

| Input | Tokens | R3 prefill | R4 prefill | Change |
| --- | ---: | ---: | ---: | ---: |
| Text | 256 | 195.174 | 190.044 | -2.63% |
| Text | 512 | 333.568 | 337.963 | +1.32% |
| Text | 1024 | 428.145 | 516.661 | +20.67% |
| Image + text | 256 | 117.930 | 119.406 | +1.25% |
| Image + text | 512 | 214.855 | 218.151 | +1.53% |
| Image + text | 1024 | 382.090 | 370.344 | -3.07% |

The 64-slot, 1024-token text PTE execution wall changes 2383.402→1978.270 ms,
including Native 745.314→501.052 ms, Attention 1103.924→1070.253 ms and Router
212.401→184.675 ms.
The arithmetic graphs are unchanged; the entire measured difference is not
attributed to Native alone. See the separate interval diagnostics below.

Evidence: [all R4 measurements](../../../../../../llama-output/qwen35-moe-sm8850-sdk251-20261027/matrix-r4-fixed/RESULTS.json),
[independent final audit](../../../../../../llama-output/qwen35-moe-sm8850-sdk251-20261027/matrix-r4-fixed/ROOT_AUDIT.json),
and [R3/R4 time comparison](../../../../../../llama-output/qwen35-moe-sm8850-sdk251-20261027/experiments/prefill64-criticalpath-r4/README.md).

## 64-slot critical path and accepted kernel changes

The R3 logs were first partitioned into disjoint PTE stages for all six 64-slot
conditions. Two 256-token rounds are averaged here; longer cases retain their
single original measurement. These are milliseconds, excluding image CPU/vision:

| Input | Tokens | PTE wall | Attention | Router | Native, including waits | Seed join | Other PTE work |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Text | 256 | 1310.219 | 295.851 | 53.803 | 853.919 | 49.890 | 56.756 |
| Text | 512 | 1532.597 | 563.022 | 99.245 | 717.467 | 52.385 | 100.478 |
| Text | 1024 | 2383.402 | 1103.924 | 212.401 | 745.314 | 75.159 | 246.604 |
| Image + text | 256 | 1938.646 | 302.339 | 55.738 | 1469.248 | 52.926 | 58.395 |
| Image + text | 512 | 2124.030 | 580.068 | 108.332 | 1273.557 | 54.548 | 107.525 |
| Image + text | 1024 | 2408.556 | 1086.326 | 196.373 | 879.087 | 55.060 | 191.710 |

Native occupies about 65%/76% of the text/image 256 PTE loop. At 1024, Attention
occupies about46%/45%. Native is not pure computation: it includes expert
readiness and DSP/RPC waits. Reader wall time, parallel worker sums, feedback
and IO overlap must not be added again to these closed wall partitions.

[The six-condition figure](../../../../../../llama-output/qwen35-moe-sm8850-sdk251-20261027/experiments/prefill64-criticalpath/TIME_BREAKDOWN.png),
[raw phase CSV](../../../../../../llama-output/qwen35-moe-sm8850-sdk251-20261027/experiments/prefill64-criticalpath/ROUNDS.csv)
and [definitions/source evidence](../../../../../../llama-output/qwen35-moe-sm8850-sdk251-20261027/experiments/prefill64-criticalpath/README.md)
preserve all eight R3 records and480 Native calls.

A separate diagnostic build then recorded each Native call, the eight worker
slots and separate DMA-wait/N2 intervals. Its single end-of-prefill trace RPC
keeps the original80/88-field profile ABI and adds an opt-in112-float record;
there are no per-layer profile RPCs. Each case below is one diagnostic sample,
not a throughput replicate. The last column is included in Native:

| Input | Tokens | Host Native ms | GDN graph ms | Full attention ms | HMX common expert-ready wait ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Text | 256 | 855.098 | 201.855 | 100.009 | 335.591 |
| Image + text | 256 | 1468.163 | 200.077 | 96.732 | 824.383 |
| Text | 1024 | 756.634 | 746.267 | 365.345 | 1.148 |
| Image + text | 1024 | 970.896 | 722.757 | 353.489 | 160.860 |

Text 1024 has almost no common expert-ready wait. Its four parallel HVX workers
accumulate1233.880 ms in dequantization/SwiGLU/store,598.280 ms in N2 expansion
and444.487 ms in combine. These are worker sums, not additive critical-path
contributions. Image 256 instead spends824.383 ms with all workers waiting for
expert readiness. The two workloads need different optimizations.

V81 grouped calls now use eight HVX workers and an8 MiB multipage workspace;
AR128 keeps four workers and3 MiB. The source changes active dispatch/join and
the input-code offset together. Maximum1024-token scratch, including worst
alignment slack, is7,866,367 bytes. V79/V85 keep their previous paths. The real
prepare/pack routines and layout boundaries pass on all three architectures;
V79/V85 RPC skel and DSP libraries also compile/link with SDK2.51.

In an unprofiled, predeclared text 1024/64 ABBA screen, Native medians fall
745.191→491.781 ms and LLM prefill rises435.835→528.883 token/s(+21.35%).
All712 numerical/history snapshots and all output prefixes match R3. The
image 1024 complete-prefill gain is only0.19%, not a stable image improvement.
Two fresh grouped8 diagnostic captures explain the different outcome:

| Input1024 | Native4→8 ms | HMX loop4→8 ms | Common ready wait4→8 ms | GDN graph after ms | Full attention after ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Text | 756.634→492.676 | 604.151→362.238 | 1.148→1.881 | 705.750 | 349.169 |
| Image + text | 970.896→951.493 | 823.767→824.109 | 160.860→353.320 | 718.445 | 355.697 |

The parallel N2/dequant work sums stay similar or grow, while text wall time
falls. The image input instead exposes more readiness waiting after faster
processing. This is an inference from the interval pattern, not evidence from PMU counters
of IO or arithmetic saturation. HMX busy time is only23.428/26.094 ms in these
text/image captures; service duty is not percent of peak arithmetic throughput.

An additional V81-only N2 helper change loads four128-byte vectors before
performing backward in-place expansion. It retains the original ABI, LUT and
remainder loop; V79/V85 preprocess to the original function. The actual V81
replacement passes3329 scalar-reference/byte/canary cases under ASAN/UBSAN,
covering every128-byte length through32768 and all 256 byte codes.
A fixed six-phase4-worker VTCM microbenchmark saves all 48,000 timer samples:
original/unroll2/unroll4 medians are1.771/1.406/1.198 µs. The32.35% function-time
reduction is not a whole-model speedup. Its first attempt mistakenly deployed
the SDK link-only FastRPC stub and stopped before opening DSP with zero
samples; the corrected device-transport run is retained separately.

The resulting grouped8+N2 candidate passes a second fixed eight-run screen,
text 256/64 ABBA then image 256/64 ABBA, with2 output tokens. Text LLM prefill
medians improve210.911→228.843 token/s(+8.50%); complete image prefill changes
120.251→120.657(+0.34%), with no stable image gain. All712 snapshots, IDs and
expert/cache work match. This comparison uses original5c versus the combined
candidate, not an isolated whole-model N2 control. The updated shared helper
is built into both final prefill and decode DSP libraries; complete-output
validation must therefore cover both stages. No GPTQ, QNN re-export, PTE
repack or cold-table change is involved.

The attempted lazy-seed host optimization is rejected. It removes the seed
join but moves copies into the Native/route queue; image Native rises3.34%
and LLM prefill falls0.94%. Its apparent complete-image median improvement is
confounded by the first baseline CPU-image time223.777 ms versus45–46 ms in the
other three runs. The text results overlap strongly. All original rounds are
retained. Chunk32 GDN is also rejected after all correctness gates pass: its
real-layer timing is about15.5% slower than Chunk64. Chunk128 fails the fixed
relative numerical gate on24/32 checks and is not sent to device.

Evidence: [grouped8 screen](../../../../../../llama-output/qwen35-moe-sm8850-sdk251-20261027/experiments/prefill64-grouped8/screen/abba-r0/REPORT.md),
[original detailed Native traces](../../../../../../llama-output/qwen35-moe-sm8850-sdk251-20261027/experiments/prefill64-diagnostic/capture-r0/RESULTS.json),
[grouped8 detailed traces](../../../../../../llama-output/qwen35-moe-sm8850-sdk251-20261027/experiments/prefill64-grouped8/diagnostic/capture-r0/RESULTS.json),
[N2 microbenchmark](../../../../../../llama-output/qwen35-moe-sm8850-sdk251-20261027/experiments/n2-unroll-microbench/MICRO_RESULT_SUMMARY.json), and
[combined candidate screen](../../../../../../llama-output/qwen35-moe-sm8850-sdk251-20261027/experiments/prefill64-n2-unroll4/screen/abba-r0/RESULTS.json).

## R3 baseline before this follow-up


The previous validated runner integrates both independent cold-start profiles. The fixed
protocol has twelve conditions and sixteen full-output measurements: two
predeclared opposite-order rounds at 256 tokens, one round at 512/1024. Values
are token/s, ordered **32 slots / 64 slots**. Image prefill includes CPU image
processing, the vision encoder and LLM prefill.

| Input | Input/output tokens | Prefill, 32 / 64 slots | Decode, 32 / 64 slots |
| --- | ---: | ---: | ---: |
| Text | 256 / 256 | 169.708 / 195.174 | 11.8102 / 14.7863 |
| Text | 512 / 512 | 295.027 / 333.568 | 11.7230 / 14.0532 |
| Text | 1024 / 1024 | 433.440 / 428.145 | 11.5144 / 14.1275 |
| Image + text | 256 / 256 | 101.381 / **117.930†** | 13.0266 / 15.6921 |
| Image + text | 512 / 512 | 190.902 / 214.855 | 13.2102 / 16.3135 |
| Image + text | 1024 / 1024 | 343.049 / 382.090 | 12.9451 / 17.1404 |

Twenty-three of twenty-four median metrics exceed the screenshot thresholds.
The remaining image256/64 prefill rate is 1.065% below 119.20. Its two rounds
are 116.735 and 119.125; both below-threshold measurements are retained. All
decode thresholds pass, and 14/16 individual measurements exceed both metrics.
The text256/64 issue in R2 is resolved with an 11.27% prefill margin. R3 still
does **not** establish all-target superiority.

The protocol extends short ChatML system messages with real space tokens to
reach the requested template-inclusive length. This is a synthetic fixed-length
workload, separately from masked AR128 padding and the zero-extra-space natural
heldout tests below. The historical SM8975 inputs and temperatures are not
matched; this comparison does not isolate a cross-SoC speedup. Peak memory and
broad model quality are not measured.

IO counters have distinct scopes. `direct_bytes` sums aligned expert read
requests; it is neither payload-only traffic nor a hardware DDR counter.
`physical_read_bytes` is the difference in `/proc/self/io` `read_bytes` around
the entire `module.execute` call, so it also covers any other process storage
reads in that interval. Equal expert requests do not require this process-wide
counter to be identical. Startup `read_bytes` counts the aligned preload
requests. Reports preserve these quantities separately.

All 8192 generated IDs match R2. All 8176 decode forwards and 327040 mailbox
commands pass the native service checks. Natural cooling waits total 22.14
minutes; no thermal/DCVS policy changes or speed-based extra rounds are used.
The first attempted R3 is separately preserved: an incorrect text/image log
marker stopped its strict validator before any accepted measurement. The marker
was corrected, the runner rebuilt, and the entire sixteen-run matrix restarted.

The R3 runner SHA-256 is
`457f1e26267ffc7e9c7f8de0f7eebe396d4b009d88c4ce07dc680cd4760fd1d5`.
The PTE SHA-256 is
`e6a4c24238067f7ebaa95b7f12a6c9d65c8955f4d98660f56719409ef6a1c9d4`.
At R3, GDN and Native DSP fingerprints remained unchanged from their earlier validated versions. The follow-up above updates Native DSP and the host workspace diagnostic.

Evidence: [R3 report](../../../../../../llama-output/qwen35-moe-sm8850-sdk251-20261027/matrix-r3-fixed/REPORT.md),
[independent raw-log and median audit](../../../../../../llama-output/qwen35-moe-sm8850-sdk251-20261027/matrix-r3-fixed/ROOT_AUDIT.json), and
[superseded marker run](../../../../../../llama-output/qwen35-moe-sm8850-sdk251-20261027/matrix-r3/SUPERSEDED.json).

## Follow-up IO experiment

The isolated Linux AIO prototype preserves expert bytes and publishes readiness
after completing all copies. A twelve-run raw-bank test compares the original
sixteen blocking readers with queue depths 16/32/64, using the same expert order
and verifying every payload byte. Queue depth 16 improves median total time by
only 1.16%; deeper queues regress. Earlier readiness in this test does not
establish an end-to-end benefit.

The predeclared five-run image256/64 ABABA screen rejects the single-coordinator
AIO16 implementation. Complete prefill medians are 122.959 versus 115.3445
token/s, a 6.19% regression. All 89 snapshots, output IDs, cache work and physical
read counts match. Native median time falls from 1456.863 to 1407.533 ms, but
serializing cached-only copies increases seed-copy time from 80.680 to 203.945 ms
and retained-copy time from 71.627 to 169.427 ms. These counters overlap and must
not be added as independent costs. The production reader remains unchanged.

Evidence: [raw-bank comparison](../../../../../../llama-output/qwen35-moe-sm8850-sdk251-20261027/experiments/linux-aio-host/pure-io/REPORT.md),
[full-model screen](../../../../../../llama-output/qwen35-moe-sm8850-sdk251-20261027/experiments/linux-aio-host/image25664-ababa/REPORT.md), and
[per-round phase counters](../../../../../../llama-output/qwen35-moe-sm8850-sdk251-20261027/experiments/linux-aio-host/image25664-ababa/ababa-r0/PHASE_CONTEXT.json).

A separately reviewed hybrid restores the original sixteen-worker path for
non-growable all-cached batches, sharing the existing staging pool. Growable or
disk-containing batches use one AIO dispatcher. Its fixed ABABA screen also
regresses: complete prefill medians are 122.547 versus 117.8095 token/s
(-3.87%), with identical snapshots, IDs and read counts. Native time improves,
but copy and outer waiting measurements increase. Cached copies are parallel
in this version; the measurements do not establish why they become slower.
Neither AIO candidate is promoted.

Evidence: [hybrid full-model screen](../../../../../../llama-output/qwen35-moe-sm8850-sdk251-20261027/experiments/linux-aio-hybrid/image25664-ababa/ababa-r0/REPORT.md) and
[complete phase measurements](../../../../../../llama-output/qwen35-moe-sm8850-sdk251-20261027/experiments/linux-aio-hybrid/image25664-ababa/ababa-r0/PHASES.json).

Five fresh-process image diagnostics attribute 112.858 ms of the 126.719 ms
median preprocessing cost to the first decoder-library `dlopen`. Temporarily
using CPU 6 reduces total median time to 104.488 ms, including affinity setup
and restoration. RGBA and all image-patch floats remain byte-identical. Actual
CPU clocks were not sampled, so this does not isolate a frequency effect or
establish full-model benefit. No production affinity change is applied.

Evidence: [first-image phase report](../../../../../../llama-output/qwen35-moe-sm8850-sdk251-20261027/experiments/image-cold-phases/device-r0/REPORT.md).

A separate fixed ABABA experiment moves the first CPU preprocessing call after
QNN construction while retaining CPU, vision and LLM time in the complete
prefill metric. It is also rejected: baseline rates are 119.682, 121.847 and
122.723 token/s; candidate rates are 119.181 and 119.682. Median throughput
falls by 1.98%. CPU preprocessing takes 44.914–50.113 ms in these baseline
processes and 78.418–91.646 ms in the candidate processes. All snapshots,
generated IDs and read counts match. The standalone diagnostic above must not
be used to infer a gain in this model experiment. Neither execution order nor
the production timing formula changes.

Evidence: [deferred-preprocessing experiment](../../../../../../llama-output/qwen35-moe-sm8850-sdk251-20261027/experiments/defer-initial-image/image25664-ababa/ababa-r0/RESULTS.json).

A fixed five-run screen also rejects temporary caller affinity during initial
image preprocessing. The candidate selects CPU 6 from the allowed set and
restores the exact original mask before model setup. Complete-prefill medians
are 121.327 versus 120.077 token/s (-1.03%). Its two inclusive CPU intervals
are 132.898646 and 47.139635 ms, against baseline intervals of 47.520990,
44.726823 and 44.447395 ms. Selection, binding, restoration and logging remain
in complete prefill time; neither the slow first candidate nor its overhead
is discarded. All 89 files, IDs and expert/cache work match. Actual clocks
were not sampled, so these data do not identify the cause of the variation.
The prototype stays isolated and is not enabled in the runner or JNI.

Evidence: [caller-affinity screen](../../../../../../llama-output/qwen35-moe-sm8850-sdk251-20261027/experiments/first-image-affinity/screen/ababa-r0/RESULTS.json).

An eight-run original-reader screen retains sixteen prefill readers. Twelve
readers improve image256/64 complete-prefill medians by 0.89%, but LLM-only
medians differ by just 0.11% and image CPU times vary. Text256/32 improves by
0.37%. This does not establish a useful IO improvement. Both variants preserve
all 89 files and expert requests. Process storage-read surplus is recorded
without filtering: the first image baseline has 593,920 extra bytes and the
first text baseline has 2,609,152; the remaining six rounds have zero surplus.

The first attempted screen was stopped by an incorrect equality requirement on
the process-wide storage counter. Its 471,040-byte surplus and original sample
remain separate; a read-only check confirms all 89 files match R3. The complete
new eight-run protocol separates expert requests from process storage reads and
does not reuse that sample or select runs by either speed or storage surplus.

Evidence: [twelve/sixteen-reader screen](../../../../../../llama-output/qwen35-moe-sm8850-sdk251-20261027/experiments/prefill-workers16-vs12/abba-r2/REPORT.md),
[independent result audit](../../../../../../llama-output/qwen35-moe-sm8850-sdk251-20261027/experiments/prefill-workers16-vs12/INDEPENDENT_RESULT_REVIEW_MATRIX.json), and
[original stopped-run diagnosis](../../../../../../llama-output/qwen35-moe-sm8850-sdk251-20261027/experiments/prefill-workers16-vs12/abba-r0-diagnostic/RESULTS.json).

## Follow-up HVX worker screen

A fixed ten-run screen compares the four-worker U16 DSP with an isolated
eight-worker AR128 candidate, using the current image/text cold tables. Image
complete-prefill medians change from 122.079 to 122.871 token/s (+0.65%); its
LLM-only change is +0.26%. Text256/32 medians change from 170.699 to 188.046,
but baseline rates span 168.209–200.209. Both candidate results lie inside the
baseline range in each case. These measurements do not establish stable gains,
so the four-worker production DSP is retained.

The candidate requests 4 MiB VTCM for AR128 instead of 3 MiB and adds 128 KiB
of worker stacks. It retains four workers/6 MiB for grouped inputs, which this
screen does not exercise. U16 encoding and Top8 accumulation order are unchanged;
all 89 files, generated IDs and expert/cache work match. Private DSP files and
SDK libraries are fingerprinted before every run; worker counts are reviewed
code settings, not hardware utilization measurements. All ten samples are kept,
with no speed-based retry. The R3 result above remains unchanged.

Evidence: [ten-run worker screen](../../../../../../llama-output/qwen35-moe-sm8850-sdk251-20261027/experiments/u16-adaptive-revisit-current/screen/ababa-r0/REPORT.md)
and [all phase measurements](../../../../../../llama-output/qwen35-moe-sm8850-sdk251-20261027/experiments/u16-adaptive-revisit-current/screen/ababa-r0/PHASES.json).

## Independent natural-text startup profile

The follow-up text profile uses three complete natural texts about a kitchen,
gardening and geometry. Each formatted input has exactly 256 valid tokens and
zero added spaces. Three-fold leave-one-out coverage selects the predefined
sample-presence/token-frequency ranking. Neither the matrix prompt nor a fourth,
previously frozen natural traffic/history text contributes routes to training.
The original required cold table is retained; V81 selects the optional text or
image table from the actual initial-turn modality. V79/V85 keep their old path.

| Fixed comparison | Baseline prefill (token/s) | New prefill (token/s) | Change |
| --- | ---: | ---: | ---: |
| Matrix text256/64, ABABA, 2 output tokens | 179.249 median | 208.286 median | +16.20% |
| Independent natural heldout, ABA, 2 output tokens | 156.359 median | 165.060 | +5.56% |
| Matrix text256/64, BA, full 256 output tokens | 189.136 | 223.321 | +18.07% |

Both short-screen candidate rounds exceed all three baseline rounds. The
independent heldout candidate also exceeds both bracketed baselines. Full-length
text prefill records 3,064,709,120 → 2,629,828,608 process storage-read bytes,
a 14.19% reduction, at the same 2,191,523,840-byte aligned startup preload.
All 84 mathematical snapshots and
all 256 generated IDs match; all 255 decode cache-work records match. Two chunk
seed/stamp history files change as expected. Full-length decode is 14.4355 →
14.0220 token/s in this pair; this does not establish a decode improvement.
Short two-output screens do not establish full-length decode performance.

Initial-chunk seed hits and full-input expert coverage have different meanings:
the matrix prompt's first AR128 chunk has 526 → 975 seed hits; its full 256-token
unique-expert coverage is 833 → 1341. The 508 additional covered experts explain
508 × 856,064 = 434,880,512 fewer aligned expert-request bytes. The separate
process storage-read counter can include other files and page faults; its
additional bytes are not attributed to a particular cause by these logs.

The new package adds two text assets and two aliases while retaining all 308
previous resource hashes and 102 aliases. Its size is 10,922,216,147 bytes and
SHA-256 is `e6a4c24238067f7ebaa95b7f12a6c9d65c8955f4d98660f56719409ef6a1c9d4`.
Android runner/JNI and host builds pass; the expanded Python suite passes 156
tests. The full final-runner results are above; historical R2 measurements must
not be substituted with these short screens.

Evidence: [fixed ABABA](../../../../../../llama-output/qwen35-moe-sm8850-sdk251-20261027/experiments/text-cold-start/ababa-natural-r2/REPORT.md),
[natural heldout](../../../../../../llama-output/qwen35-moe-sm8850-sdk251-20261027/experiments/text-cold-start/heldout-natural-aba-r2/REPORT.md),
[full BA](../../../../../../llama-output/qwen35-moe-sm8850-sdk251-20261027/experiments/text-cold-start/full-natural-ba-r2/REPORT.md), and
[package audit](../../../../../../llama-output/qwen35-moe-sm8850-sdk251-20261027/experiments/text-cold-start/production-integration/PACKAGE.json).

## R2 integrated model matrix (historical)

R2 uses the integrated image cold-start profile, final SDK 2.51 model package,
and the same four-worker production DSP kernels. Its sixteen measurements
complete twelve cases: 256-token cases use the median of two predeclared
opposite-order rounds; 512/1024-token cases use one round. Values below are
token/s, ordered **32 slots / 64 slots**. Image prefill includes preprocessing
and the encoder. All actual input IDs and generated IDs match R1.

The fixed-length matrix extends the short ChatML system message with real
space tokens to reach the requested template-inclusive length. These are valid
model inputs, separately from masked AR128 padding. This is a synthetic
fixed-length workload; it is not a natural 256/512/1024-token corpus. The same
input IDs are retained across R1, R2 and the final-runner follow-up. The natural
text heldout above is a separate test with zero added spaces.

| Input | Input/output tokens | Prefill, 32 / 64 slots | Decode, 32 / 64 slots |
| --- | ---: | ---: | ---: |
| Text | 256 / 256 | 169.019 / **174.526†** | 11.6633 / 14.6465 |
| Text | 512 / 512 | 307.318 / 325.638 | 11.7932 / 14.5096 |
| Text | 1024 / 1024 | 443.576 / 438.467 | 11.5627 / 14.1760 |
| Image + text | 256 / 256 | 96.366 / 119.835 | 12.9389 / 15.6680 |
| Image + text | 512 / 512 | 191.545 / 221.837 | 12.9877 / 16.1759 |
| Image + text | 1024 / 1024 | 354.448 / 379.259 | 13.0616 / 17.1258 |

Twenty-three of twenty-four median throughput metrics exceed the screenshot.
Text256/64 prefill remains 0.50% below 175.40; R2 does **not** pass all targets.
Image256/64 has only 0.53% median headroom over 119.20: its two complete
prefill rates are 121.097 and 118.573. Text256/64 has 178.919594 and
170.133381. Fourteen of sixteen individual measurements exceed both targets.
The two below-threshold runs are preserved. No speed-based retry or sample
replacement is used. Image64's second-round IO wall time increases by about
56 ms at the same physical read count; image preprocessing does not slow down.
Small median headroom must not be presented as stable per-run superiority.

Every 256-token round starts at thermal status 0; long rounds start at status
0 or 1 and can finish at 2. Natural cooling waits are recorded; phone thermal
and DCVS policies remain unchanged. The historical SM8975 inputs and thermal
conditions are not matched, so this is a comparison with screenshot thresholds.
Peak memory and broad model quality are not measured by this matrix.

Final runner SHA-256 is
`4b9d1c15ffdf9435efc615b9bebb01b576bea69b810c2c42a8dd4fa73ff5a710`;
the PTE is 10,922,195,242 bytes, SHA-256
`4ec4ef05ced766b5daf14151c797479811a3037a140d968b3dc0912223f11c5c`.
Repackaging adds two optional assets and two links. All 306 original model
resources and 100 original links retain their hashes/targets. The original
PTE is preserved. GDN, Native DSP and decode/shared libraries retain their
validated fingerprints. Android runner/JNI builds, 147 Python regressions
and the host checks pass. V79 compatibility is compile-tested; no new SM8750
device performance is claimed.

Evidence: [R2 report](../../../../../../llama-output/qwen35-moe-sm8850-sdk251-20261027/matrix-r2/RESULTS.json),
[independent median audit](../../../../../../llama-output/qwen35-moe-sm8850-sdk251-20261027/matrix-r2/ROOT_AUDIT.json),
[package resource audit](../../../../../../llama-output/qwen35-moe-sm8850-sdk251-20261027/experiments/image-cold-start/production-integration/PACKAGE.json), and
[dispatcher audit](../../../../../../llama-output/qwen35-moe-sm8850-sdk251-20261027/experiments/matrix-r2-dispatcher-audit/DISPATCHER_AUDIT.json).
The dispatcher hash change after plan generation is independently reconstructed
as a pre-launch PTE integrity check; commands, counts, order, thermal gates and
statistics are unchanged. The original plan is preserved.

## Baseline and timing scope

The standalone recurrence uses B=1, H=32, K=V=128, U16 activations/state,
and host-precomputed Q/K normalization and decay. Both libraries use the
existing MoE `-O3` and `DELTANET_U16_KARATSUBA_REDUCTION=1` flags. The same
PTE and inputs are reused when swapping DSP libraries. Graph boundaries in
this microbenchmark include conversions; these are not identical to the
MoE graphs' UINT16 state boundaries.

The final benchmark shares the initial/final state encoding, as the MoE path
requires. Five warmups, profiling disabled, median of round averages:

| Layout | Sequence | Rounds × executions | Original (ms) | Optimized (ms) | Speedup |
| --- | --- | --- | ---: | ---: | ---: |
| DDR | AR1 | 5 × 1000 | 0.450665 | 0.447267 | 1.0076× |
| DDR | AR128 | 5 × 1000 | 3.739388 | 3.481400 | 1.0741× |
| VTCM | AR1 | 3 × 400 | 0.461938 | 0.462728 | 0.9983× |
| VTCM | AR128 | 3 × 400 | 3.084865 | 2.695968 | 1.1443× |

Decode's difference is within the observed noise range; no significant decode
speedup is established. AR128 improves by 6.9% on DDR and 12.6% on VTCM. The
radix-256 reduction was enabled for these V81 measurements. It is now disabled
by default after the free-generation regression described above. V79 retains the
existing reduction. Initial/final U16 storage and the state layout are unchanged.

Final state files are byte-identical in all paired runs. AR128 attention's
maximum difference is approximately 1.06e-6 and its SQNR to the floating
reference remains approximately 64.44 dB. Floating reconstruction changes the
rounding order; this does not establish end-to-end generation/quality equivalence.
All final thermal snapshots report status zero, with battery temperature
approximately 33.7–33.9°C.

Evidence: `llama-output/gdn-sm8850-sdk251-20261027/` under
`radix256full-confirm/comparison.json` and `radix256full-vtcm/comparison.json`.
Each paired comparison uses identical graphs and inputs. Comparing original DDR
to optimized VTCM gives 27.9% lower AR128 latency, but changes graph placement
and is explicitly a cross-layout comparison.

## Internal phases

Build the diagnostic DSP library with
`HEXAGON_CPPFLAGS=-DDELTANET_PROFILE_PHASES=1`, keeping the normal MoE flags.
The production default is zero. Run one execution without warmup and capture
`adb logcat` lines containing `GDN_PHASE`. No timings are written into model
outputs. The instrumented and baseline attention/state files were byte-identical.

```bash
python examples/qualcomm/oss_scripts/moe/qwen35-35B-A3B/tools/gdn_phase_report.py \
  llama-output/gdn-sm8850-sdk251-20261027/phase/phase-logcat.log \
  --output llama-output/gdn-sm8850-sdk251-20261027/phase/PHASES.json
```

Eight workers process four heads each. The table uses the worker with the
longest accumulated interval in each invocation, at 19.2 MHz qtimer frequency.
Worker intervals run in parallel and must not be added as graph wall time.

The confirmation captures compare each library on the same diagnostic graph:

| Sequence / library | Q/K and coefficient setup | K/Q × state reduction | Delta/output | State update | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| AR128 original | 0.34479 | 1.33573 | 0.29156 | 1.11161 | 3.08370 ms |
| AR128 optimized | 0.33750 | 1.00589 | 0.29307 | 1.09115 | 2.72760 ms |

The reduction interval falls by approximately 24.7%. A separate AR1 diagnostic
capture gives 0.10245 → 0.08583 ms for the slowest worker's internal intervals;
this does not establish an equal improvement in complete graph latency.
Diagnostics include timer overhead and exclude graph scheduling, boundary
transfers, and FARF logging; use uninstrumented paired runs for throughput.

[Phase comparison figure](../../../../../../llama-output/gdn-sm8850-sdk251-20261027/GDN_PHASES.png)
and [SVG](../../../../../../llama-output/gdn-sm8850-sdk251-20261027/GDN_PHASES.svg)
are generated from the two phase reports by the adjacent `plot_gdn_phases.py`.

Evidence: `llama-output/gdn-sm8850-sdk251-20261027/` under
`phase-baseline-confirm/` and `phase-optimized/`, each containing
`phase-logcat.log`, `phase-report.json`, and `output-checks.json`. Every
instrumented output is byte-identical to its corresponding uninstrumented
library's output. The earlier `phase/` directory retains the initial capture.

## Memory and compute interpretation

The AR128 logical recurrence traffic estimate is about 406.9 MB: state is
read for reduction, read for update, and written for every token. Dividing
this by the initial 3.83155 ms measurement gives about 106.2 GB/s. The minimum graph-input/output
traffic estimate is only about 6.31 MB, or 1.65 GB/s at that latency. Neither
number measures DDR traffic: state accesses can hit cache or VTCM, and boundary
conversions add traffic. Mathematical operation counts also differ from the
actual radix-reduction instruction count.

Consequently these results do not establish DDR, HVX, or HMX saturation.
Calibrated bandwidth/compute rates must match memory placement, dtype and
clocks. The SDK's `QNN_HTP_PROFILE_EVENTTYPE_GRAPH_FINALIZE_PERF_ESTIMATE_*`
bandwidth fields are compiler estimates, not runtime memory-controller counters.
The benchmark tool labels its derived rates accordingly.

## Real-layer comparison

Layer 0 uses the same native-IO executor, four identical nonzero held-out
inputs, explicit Burst mode, profiling disabled, ten warmups and three rounds
of 200 executions. Values are medians of round averages.

| Prefill implementation | Layer latency (ms) |
| --- | ---: |
| ChunkGDN mixed FP16 | 2.880585 |
| Custom FP16, original | 8.591355 |
| Custom FP16, register reuse | 6.924430 |
| Custom U16, original DDR | 6.066880 |
| Custom U16, optimized DDR | 5.856165 |
| Custom U16, optimized VTCM | 4.597170 |

FP16 register reuse improves the real layer by 1.2407× with four byte-identical
outputs. U16's convolution/GDN state codes are identical; the two float outputs
have SQNR 101.45/109.55 dB against the original. Optimized DDR and VTCM outputs
are byte-identical. ChunkGDN remains the fastest measured prefill implementation.
Decode uses the optimized U16 custom op.

V81 enables `DELTANET_FP16_REGISTER_REUSE` by default. V79/V85 retain the
previous FP16 path. SDK 2.51 x86/ARM64/V79/V81/V85 and SDK 2.48 V79 builds
pass; 240 native full-recurrence cases and FP32 projection scratch comparisons
are byte-identical with reuse disabled. The final library also passes device
FP16/U16 output-hash regression against the measured references.

Evidence: `pilot-layer0/executor-unified-ab/RESULTS.json` under the MoE artifact
directory. Earlier `qnn-net-run` pilot timings used a different driver and must
not be compared directly with these executor measurements.

## SM8850 native workspace and IO

The original native MoE prefill requested a 12 MiB AR128 workspace; this device
has 8 MiB VTCM. V81 uses four HVX workers for AR128 and eight for grouped
calls, 448 KiB per worker, and DDR route output. AR128/partitioned requests are
3 MiB and grouped requests are8MiB, with multipage allocation and2KiB aligned
HMX buffers. V79/V85
retain their original worker count and resource requests. These worker counts
refer to MoE experts, separately from the GDN recurrence's eight slices.
Real 128/256/1024-token prefill and decode handoff all pass; the 256-token
image path also runs successfully. Layout tests cover all three architectures.

Long-output validation exposed a host mailbox completion race: the final ACK
could arrive between the first ACK read and the service-done read. The host
now rechecks ACK after observing service completion; a missing ACK still
reports failure. This keeps each native RPC bounded to one forward.

Nine legal settings sweep decode IO workers 4/8/16 and chunks 64/256/512 KiB
using fixed 256-token input and 64 generated tokens. Every successful run has
identical token IDs and per-step cache requests/hits/bytes. 1024 KiB is invalid:
these N2 banks have 852096-byte expert payloads and an 860160-byte reader limit.

The 4-worker/512 KiB decode setting reaches 12.2888 token/s against
11.1459 for 8/256. Reverse-order confirmation gives 12.3077 versus 11.5067,
or 6.96% higher throughput. Internal profiling shows native time falling
3193.87 → 2685.78 ms and HVX weight-IO waiting 2772.34 → 2265.79 ms;
HMX busy time remains approximately 30.4 ms. Read calls halve, 22676 → 11338.
HMX service duty is approximately 1.0–1.2%, not percent of peak arithmetic
throughput. This evidence identifies weight readiness as the main native
decode delay for this workload, without measuring memory-controller saturation.

Decode 8/512 reaches 12.2605 token/s, within 0.23% of 4/512. The runtime keeps
eight decode readers and selects 512 KiB chunks. Independently, prefill IO
8/16 ABBA gives medians 153.1745/163.8198 token/s, a 6.95% gain with 16 readers;
84 output/state/input/route files remain byte-identical in every run.
`configs/sm8850.json` selects these runtime defaults. The existing native
`--prefill-io-workers` option is also available through Python.

R3 image256/64 round 0 requests 4,263,198,720 aligned expert bytes over
1634.389060 ms of reader wall time, an observed request rate of 2.61 GB/s.
The isolated sixteen-reader bank test reaches 2.85 GB/s for 219,152,384 aligned
bytes over its 76.848125 ms median. The latter writes ordinary heap memory;
the model writes DSP memo buffers and interleaves cached copies and compute.
These are different workloads, so their ratio is not an IO utilization
percentage or proof of peak UFS/DDR bandwidth. The raw bank probe's historical
`physical_bytes` label counts aligned requests, not hardware transactions.

The attempted SDK SysMon registration fails with return value 256 and provides
no PMU measurements. The cause is unknown; the probe reports tool 10.0.29 and
v75 despite the configured V81 target. This does not establish an architecture
or permission diagnosis. Phase timing and HMX service duty do not establish
GDN HVX peak utilization or DDR saturation. No PMU arbitration override is used.
The bounded [local capability review](../../../../../../llama-output/qwen35-moe-sm8850-sdk251-20261027/experiments/pmu-capability-review/REVIEW.json)
records these limits.

Evidence: `io-sweep/REPORT.md` and its sibling experiment directories
`unified-screening`, `unified-confirm`, `unified-profile` and `prefill-io-abba`
under the MoE artifact directory.

## MoE acceptance protocol

`configs/export_sm8850.json` selects SDK 2.51 and an isolated build/export
directory. `configs/model_sm8850.json` selects context 2048, AR128, W2, Top8,
and 32 expert slots; runtime accepts 32 or 64 slots.

MoE prefill uses `ChunkGDN` FP16 matrix operations, selected by the real-layer
comparison above. Decode uses the recurrent U16 custom op.

Generate or execute the screenshot's twelve throughput cases with new fixed
inputs, as requested. `--run` executes; omitting it only saves the commands.

```bash
python examples/qualcomm/oss_scripts/moe/qwen35-35B-A3B/tools/moe_benchmark.py \
  --weights-path llama-output/qwen35-moe-sm8850-sdk251-20261027/pte \
  --io-workers 8 --prefill-io-workers 16 --read-chunk-kib 512 \
  --image test.jpg --repeats 3 \
  --output llama-output/qwen35-moe-sm8850-sdk251-20261027/matrix --run
```

Text prefill uses `PREFILL_TOTAL` (LLM only); image prefill uses
`CHAT_STATS.prefill_token_per_sec`, including image preprocessing and encoder,
matching the screenshot's image column. The LLM-only image timing is retained
in `PREFILL_TOTAL` for diagnostics. Decode uses native `CHAT_STATS`.
Inputs and actual outputs must both have
256/512/1024 tokens; context-truncated or early-EOS runs fail validation.
All twelve cases and all requested rounds must finish before the report can
mark all throughput targets passed. This matrix does not measure peak memory
and does not establish equal quality to the historical model.

## R1 full-model matrix

R1 completed all twelve cases with one measurement per case, context 2048,
AR128, W2/Top8, greedy generation, thinking disabled, EOS ignored and
input/output lengths equal. Runtime uses 16 prefill IO readers, eight decode
readers and 512 KiB read chunks. The frozen host includes the mailbox ACK
completion fix; the GDN library includes the V81 radix-256 reduction and FP16
register reuse described above.

Values below are token/s, ordered **32 slots / 64 slots**. Text prefill is
LLM-only `PREFILL_TOTAL`; image prefill includes image decoding, preprocessing
and the encoder, using `CHAT_STATS.prefill_token_per_sec`. A dagger marks a
prefill result below the corresponding SM8975 screenshot threshold.

| Input | Input/output tokens | Prefill, 32 / 64 slots | Decode, 32 / 64 slots |
| --- | ---: | ---: | ---: |
| Text | 256 / 256 | **155.509† / 168.073†** | 11.5759 / 14.4813 |
| Text | 512 / 512 | 298.266 / 308.381 | 11.6732 / 14.4225 |
| Text | 1024 / 1024 | 463.929 / 442.544 | 11.4895 / 14.0913 |
| Image + text | 256 / 256 | 96.7864 / **104.832†** | 12.7224 / 15.3046 |
| Image + text | 512 / 512 | 186.657 / 198.835 | 13.4418 / 16.3961 |
| Image + text | 1024 / 1024 | 350.085 / 375.504 | 13.1540 / 17.4437 |

All twelve decode rates exceed the screenshot thresholds. Three prefill
thresholds remain unmet: text 256/32 is 2.60% below 159.66; text 256/64 is
4.18% below 175.40; image 256/64 is 12.05% below 119.20. R1 therefore does
not satisfy all throughput targets. All six pairs of 32/64-slot cases have
byte-identical actual input IDs and generated IDs; no runtime case failed.

The first seven cases ran continuously, beginning with text 1024/32 at
Thermal Status 0 and reaching status 3 during the sequence. The eighth case,
image 256/64, completed before the dispatcher was interrupted; its complete
native log and successful `RUN.json` were recovered, but immediate historical
before/after thermal snapshots are unavailable. Each of the last four image
cases started after natural cooling to status 0; their after-status values
were 1, 1, 2 and 1. No phone thermal/DCVS policy was changed. These are not
uniform cold-device measurements. The user approved new inputs; the SM8975
screenshot's workload, clocks and temperatures were not matched, so threshold
comparisons do not isolate a cross-SoC speedup. Peak memory was not measured.

The fixed text prompt is `请介绍注意力机制的原理和应用。`; the image prompt is
`请详细描述这张图片。`. The original image is
[`test.jpg`](../../../../../../test.jpg), SHA-256
`7d466b5e482da8ad6a2c5ac65d01a9d7758204f9715af5859e131d9016cc5c8d`.
Actual padded/template-inclusive inputs and generated IDs are retained as
`input-tokens.i64` and `generated.i64` in each case directory. Commands,
absolute native-log paths and hashes are included in the report.

R1 host `build/qwen35_moe_runner` SHA-256 is
`4cecce87e7951292c6732fd9aaab150125da6f4e610d0e294cee855daf06ffa9`;
V81 `libQnnDeltaNetOpPackage_HTP.so` SHA-256 is
`24f59cdaca5f0ca148d8441897a84622b195187cf02ae63004190d406c7b5350`.
The complete host/DSP/export fingerprint set remained unchanged throughout R1.

The original `RESULTS.json` incorrectly compared LLM-only image prefill with
the screenshot's image-inclusive column. Only image prefill rates and their
summary comparisons were affected; the original native logs remain valid.
The independent corrected report preserves both LLM-only and full prompt
rates, explains the correction, and records the actual thermal conditions.
The raw report retains its original calculation scope and now includes all
completed cases; the first-seven-measurement backup remains unchanged.

Evidence: [corrected R1 report](../../../../../../llama-output/qwen35-moe-sm8850-sdk251-20261027/matrix-r1/RESULTS.scope-corrected.json),
[original summary](../../../../../../llama-output/qwen35-moe-sm8850-sdk251-20261027/matrix-r1/RESULTS.json), and
[all raw logs and case evidence](../../../../../../llama-output/qwen35-moe-sm8850-sdk251-20261027/matrix-r1/). The three unmet prefill cases have
original native logs for [text 256/32](../../../../../../llama-output/qwen35-moe-sm8850-sdk251-20261027/matrix-r1/text-256-slots32-r0/20261027-224357-972420-chat/runner.log),
[text 256/64](../../../../../../llama-output/qwen35-moe-sm8850-sdk251-20261027/matrix-r1/text-256-slots64-r0/20261027-224432-349906-chat/runner.log) and
[image 256/64](../../../../../../llama-output/qwen35-moe-sm8850-sdk251-20261027/matrix-r1/image-256-slots64-r0/20261027-224856-755154-chat/runner.log).

## Follow-up diagnosis and rejected candidates

An isolated image256/64 diagnostic build reports 2146.58 ms of LLM prefill,
including 1682.60 ms Native, 201.20 ms GDN, 96.88 ms full attention and
54.01 ms router/shared. HMX busy is 18.896 ms within a 1608.732 ms service
loop; 941.208 ms of that loop has every active HVX worker waiting for expert
payloads. The parallel sum of HVX IO waiting is 4913.867 ms. Host IO wall
time is 1839.952 ms, with 1649.290 ms overlapping other work. Its explicit
join wait is only 2.257 ms. Consequently a small outer `read_ms` does not
mean expert IO is cheap. These instrumented measurements locate waiting;
they are not formal throughput or DDR-controller utilization measurements.

The following candidates remain isolated experiments:

- Lowering the Native grouped threshold from 512 to 256 halves calls from
  80 to 40, but text256/32 changes 167.006 to 165.188 token/s and image256/64
  changes 108.659 to 107.835 token/s including image processing. Physical
  reads, 84 mathematical files, five handoff files and generated IDs match.
  The original threshold is retained.
- Increasing image256/64 prefill readers from 16 to 24 or 32 gives
  107.203 / 105.698 / 105.873 token/s including image processing. Native
  time remains 1697 / 1701 / 1702 ms. All mathematical files match; the
  extra 13.125 MiB of 32-reader staging gives no benefit. Sixteen remains
  the default.
- Eight HVX workers with a compact U8 input-code cache give an image256/64
  ABA short-output LLM speedup of 1.0433x over the median original latency.
  All mathematical/handoff files match. Complete prefill is 113.677 token/s
  in the candidate run, still below 119.20. Earlier text128 tests regress;
  the four-worker production layout is retained pending stronger evidence.
- Direct `O_DIRECT` reads into both cached and uncached rpcmem return
  `EFAULT(14)` on SM8850/SDK 2.51, while a normal aligned allocation succeeds.
  The probe checks 4 KiB alignment and byte equality without opening DSP
  graphs. Removing the miss copy through `raw_staging` is therefore rejected.

Evidence is under the MoE artifact directory's `experiments/grouped256`,
`prefill-workers`, `u8-worker8-image64/aba-r2`, and `rpcmem-direct`.
An ELF inspection without an explicit output ELF rewrote two local files;
the full original SHA was restored from a complete backup. Every allocated
section was unchanged, but the affected `u8-worker8-image64/full-ba-r1`
confirmation is excluded from frozen-binary comparisons. Earlier R1 and
`aba-r2` results are unaffected. See
[`RECOVERY.json`](../../../../../../llama-output/qwen35-moe-sm8850-sdk251-20261027/experiments/build-fingerprint-mutation/RECOVERY.json).

Ready-first expert dispatch, eight-worker U16 AR128, earlier next-buffer DMA,
and longer idle polling were also tested in isolated builds. Individual short
screens sometimes improved LLM prefill by roughly 3–4%, but combinations did
not reproduce a reliable improvement or reach the complete image256/64 target.
Ordering host reads by file offset reduced IO wall time while delaying the
experts needed by DSP, and made LLM prefill slower. None of these changes is
enabled in the production layout. Their raw results remain under
`experiments/ready-first-image64`, `u16-adaptive-image64`,
`prefill-combinations-image64`, `prefill-wait-image64`, and
`prefill-offset-image64`.

## Independent image cold-start profile

The original text-calibrated startup table also served image inputs. The new
optional SM8850 image profile changes the experts initially placed in the same
32/64 cache slots. It changes neither routing nor model arithmetic.

Training uses three repository images: `examples/models/dinov2/dog.jpg`,
`third-party/ao/examples/sam2_amg_server/dog.jpg`, and
`extension/llm/apple/ExecuTorchLLM/__tests__/resources/IMG_0005.jpg`, with the
fixed prompt `请说明图像中的主要内容。`. Each collection has 256 input tokens.
The benchmark's `test.jpg` and its routes are excluded from table construction
and policy selection. Two predefined rankings were compared by three-fold
leave-one-image-out unique expert coverage: token-frequency ranking scores
7366; image-presence then token-frequency ranking scores 7409. The selected
profile is fitted on all three training images, with expert ID breaking ties.
Its 40 × 64 little-endian int32 table SHA-256 is
`cd884a2a8198198654bb8fae35126508a37b66071d0ffb0d64834d21021e639a`.

On held-out image256/64, a short-output ABA comparison gives complete prefill
110.440 → 123.018 → 108.613 token/s, and LLM-only prefill
118.796 → 133.243 → 116.464 token/s. All three runs start and finish at thermal
status 0. All 84 mathematical snapshots and generated IDs match; two cache
history files change as expected. Both profiles preload exactly 2560 experts,
requesting 2,191,523,840 aligned bytes during startup. Process storage reads decrease
from 4,945,481,728 to 4,263,198,720 bytes (13.80%), and Native time falls from
1682.80/1719.24 ms to 1453.33 ms. Evaluation after freezing the table finds
797 additional required experts present at startup; 797 × 856064 bytes equals
the measured saving. This held-out evaluation is not fed back into training.

`seed_hits` counts the first AR128 block only because the seeded flags are
cleared after that block. It must not be interpreted as startup coverage for
the full 256-token sequence. In these runs, the process storage-read counts
equal the aligned expert-request totals; the two counters have distinct scopes.

The short-output screen is separate from the formal input=output matrix.
Its frozen evidence is in
[`cold-aba-loo-r1`](../../../../../../llama-output/qwen35-moe-sm8850-sdk251-20261027/experiments/image-cold-start/cold-aba-loo-r1/).

Reversed-order confirmation with 256 generated tokens gives new/old complete
prefill 122.664/111.986 token/s and decode 15.3321/15.5633 token/s. Both runs
start and finish at thermal status 0. All 84 mathematical files, all 256
generated IDs and all 255 decode-step cache-work records match. Prefill
process storage reads again fall by 13.80%; aligned startup requests remain identical.
Thus the prefill gain does not move additional IO into model initialization.
Decode throughput varies despite identical work; no decode speedup is
attributed to this table. Evidence:
[`cold-full-ba-loo-r1/RESULTS.json`](../../../../../../llama-output/qwen35-moe-sm8850-sdk251-20261027/experiments/image-cold-start/cold-full-ba-loo-r1/RESULTS.json).

The repository profile is `configs/sm8850_image_cold.json`. Materialize the
optional assets beside an existing export or PTE with:

```bash
python examples/qualcomm/oss_scripts/moe/qwen35-35B-A3B/tools/image_cold_profile.py \
  --output-dir llama-output/qwen35-moe-sm8850-sdk251-20261027/pte
```

`--collection .../calibration-r1/COLLECTION.json` recomputes the two rankings,
three-fold selection and final table from the three preserved route files;
source image and route hashes are checked. No benchmark routes are read.
`--overwrite` explicitly replaces only the two optional image assets.
SM8850 exports with vision materialize and package the default profile.
The single PTE contains both assets and their prefill links; old packages
without them remain valid. Updating the profile invalidates only the package
checkpoint, without changing the numerical model or calibration recipe.

The common CLI/JNI conversation path sets `PREFILL_INITIAL_IMAGE` from the
actual initial image path before constructing the prefill session. V81 uses
the optional table only for that case. Vision-resident text sessions, V79/V85
and old models without the optional assets use the original text table.
Later image turns reuse an already warm cache. The diagnostic runner sets
the same flag from actual image/pixel input. `STARTUP_COLD_PROFILE` records
the selected kind and path. The JNI CMake target reads the Native build's
`TARGET.json` so the host architecture and default GDN package agree.

## Corrected GDN: memory and text accuracy, 2026-10-28

The current user export `qwen35-sm8850--1028/model.pte`, SDK 2.51.200.261021,
V81, 2048 context and AR128 was evaluated without changing quantization.
Eight non-overlapping 1024-token WikiText-2 raw test windows use a 128-token
prefix and 896 teacher-forced targets each: 7168 targets across six articles.
These accuracy runs use 64 expert slots per layer.
All original windows are retained; four additional articles were selected
from fixed 20/40/60/80% stream positions before observing their scores.
This is sampled test PPL, not a complete WikiText benchmark or vision evaluation.

| Metric | Measured |
| --- | ---: |
| Original checkpoint FP32 reference PPL | 5.747884736 |
| Current NPU PPL | 7.904025099 |
| PPL increase | 37.5119% |
| Final residual hidden SQNR | 7.027713 dB |
| Unweighted RMS-normalized hidden SQNR | 6.678919 dB |
| Raw logits SQNR | 8.170083 dB |
| NPU/FP32 top1 agreement | 75.1814% |
| Exact next-token label top1, FP32 → NPU | 60.1004% → 54.6317% |
| Label top1 change | −5.4688 percentage points |

The reference promotes original BF16 checkpoint values to FP32 arithmetic,
disables TF32 and uses the repository's attention, GDN, MoE and original head.
Static review against the installed Transformers 5.8.1 architecture finds
no obvious formula mismatch, but an independent HF end-to-end run has not
been performed. The gap therefore describes the whole current quantized
execution chain relative to that repository reference, not a measured
contribution of any individual quantizer or GDN kernel. A PPL increase is
not an accuracy-drop percentage. Top1 agreement compares two models'
argmax, not their accuracy against corpus labels. The separate label top1
row uses the same 7168 teacher-forced targets; it is not downstream task accuracy.

Separate normal CLI runs disable all-logits retention, use prefill IO16,
decode IO8 and 512KiB reads, and generate the full requested output.
All 7168 generated IDs match the corresponding unmonitored performance runs.

| Input | Input → output | 32-slot sampled peak GiB | 64-slot sampled peak GiB |
| --- | --- | ---: | ---: |
| Text | 256 → 256 | 3.592493 | 4.634196 |
| Text | 512 → 512 | 3.634155 | 4.660088 |
| Text | 1024 → 1024 | 3.686910 | 4.712533 |
| Image + text | 256 → 256 | 4.160695 | 5.209474 |
| Image + text | 512 → 512 | 4.221574 | 5.243351 |
| Image + text | 1024 → 1024 | 4.251867 | 5.277187 |

Each frame accounts `PSS - matched DMA mapped PSS + unique DMA capacity`
using `(dev,inode)` identities. Independent raw FDinfo/smaps reconstruction
passes for every accepted peak. Of 3334 frames, 3006 have complete ownership;
all 124 frames with unresolved DMA mappings also have unstable FD snapshots
and are excluded from accepted peaks. Ordinary ELF paths such as
`libdmabufheap.so` are counted as file PSS, not DMA. SwapPSS is zero throughout.
The target interval is 200ms; actual per-case medians range from 203.8 to
230.3ms and the maximum is 237.4ms. Unvalidated racy accounting candidates
can be 18.82MiB above a case's accepted peak; these are neither validated
peaks nor physical upper bounds. Periodic sampling can miss transients.
JNI/JVM, other processes and unexposed driver allocations are outside this scope.
Quality runs retain approximately 0.89GB of logits and are not used as
normal-inference memory measurements.

A separate replay uses exactly the same quantized model and original window0
with the old GDN and old prefill/decode N2 DSP libraries. All 167 raw token,
hidden, logits and state/KV files match the current operators byte for byte
through 896 predicted positions; both versions have PPL 6.0104101357282325.
This control finds no added numerical loss from the operator optimization
on that window, while the whole quantized model still differs from FP32.
All private diagnostic libraries and inputs were restored and SHA-verified;
the user model, build and performance runtime were not modified.

Full window scores, sufficient-statistic aggregation, source and dataset
fingerprints, raw tensors, private reference-op replay and independent audits:
[quality and memory report](../../../../../../llama-output/qwen35-sm8850--1028/quality-memory-20261028/REPORT.md).
