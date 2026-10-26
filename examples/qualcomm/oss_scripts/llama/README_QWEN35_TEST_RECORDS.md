# Qwen3.5 / Lincal3.5 测试记录归档

最初归档：2026-10-15；最近更新：2026-10-19。

当前导出配置、最终性能/PPL和验证范围见[主 README](README.md#validation-record-2026-10-15)。本文保留清理前52份报告中的测试数据、过程和结论；各记录使用当时的配置与测试口径，较早的实验配置和问题描述不代表当前实现。

2026-10-15归档的历史实验脚本、新增测试代码、日志、PTE/PT2、输入输出张量及临时库副本已清理。文中的历史命令与路径仅用于说明当时的参数和证据来源；导出使用主 README 的普通 `llama.py` 命令。已清理产物的链接改为文字，报告间链接指向本文对应记录。后续新增的1K–8K矩阵PTE与测量结果仍保留；本轮仅清理临时测试代码，见[正式入口接入记录](#qwen35-cli-integration-20261017)。算子测量历史见[DeltaNet README](../../custom_op/deltanet_op_package_htp/DeltaNetOpPackage/README.md)。

## 目录

- [2026-10-19：第 1 层 GDN 极小权重 scale 饱和修复](#qwen36-layer1-tiny-scales-20261019)
- [2026-10-17：Qwen3.6-35B-A3B 真实单层 MoE 可行性测试](#qwen36-moe-layer0-20261017)
- [2026-10-17：正式入口接入、review与测试代码清理](#qwen35-cli-integration-20261017)
- [01. llama-output/qwen3_5-mainline-integration/cross-model-fixes/REPORT.md](#record-01)
- [02. llama-output/qwen3_5-prefill-optimization/FINAL_SUMMARY.md](#record-02)
- [03. llama-output/qwen3_5-mainline-integration/review-fixes/REPORT.md](#record-03)
- [04. llama-output/qwen3_5-0_8b-layer-analysis/HIDDEN_STATE_ERROR_REPORT.md](#record-04)
- [05. llama-output/qwen3_5-0_8b-layer-analysis/QUANTIZATION_DIAGNOSIS_REPORT.md](#record-05)
- [06. llama-output/qwen3_5-0_8b-layer-analysis/all_linear_w4_128/REPORT.md](#record-06)
- [07. llama-output/qwen3_5-0_8b-layer-analysis/all_linear_w4_128/gptq_out_proj/REPORT.md](#record-07)
- [08. llama-output/qwen3_5-0_8b-layer-analysis/all_linear_w4_128/out_proj_analysis/REPORT.md](#record-08)
- [09. llama-output/qwen3_5-0_8b-layer-analysis/full_gptq_prefill128/REPORT.md](#record-09)
- [10. llama-output/qwen3_5-0_8b-layer-analysis/full_kz_w8_gptq_prefill128/REPORT.md](#record-10)
- [11. llama-output/qwen3_5-0_8b-layer-analysis/full_kz_w8_gptq_prefill128/complete_prefill128/REPORT.md](#record-11)
- [12. llama-output/qwen3_5-0_8b-layer-analysis/full_kz_w8_gptq_prefill128/layer7_8_replay/REPORT.md](#record-12)
- [13. llama-output/qwen3_5-0_8b-layer-analysis/full_kz_w8_gptq_prefill128/ppl_prefill128/REPORT.md](#record-13)
- [14. llama-output/qwen3_5-0_8b-layer-analysis/full_kz_w8_gptq_prefill128/qk_native_rms/complete_hybrid/README.md](#record-14)
- [15. llama-output/qwen3_5-0_8b-layer-analysis/full_kz_w8_gptq_prefill128/qk_native_rms/context_performance_20261014/REPORT.md](#record-15)
- [16. llama-output/qwen3_5-0_8b-layer-analysis/full_kz_w8_gptq_prefill128/qk_native_rms/performance_regression_20261014/REPORT.md](#record-16)
- [17. llama-output/qwen3_5-0_8b-layer-analysis/gdn_128_after_l2fix/REPORT.md](#record-17)
- [18. llama-output/qwen3_5-0_8b-layer-analysis/gdn_128_key_w8_rest_w4/REPORT.md](#record-18)
- [19. llama-output/qwen3_5-0_8b-layer-analysis/kz_gate_drop/REPORT.md](#record-19)
- [20. llama-output/qwen3_5-0_8b-layer-analysis/kz_gate_drop/full_precision_reduction/REPORT.md](#record-20)
- [21. llama-output/qwen3_5-0_8b-layer-analysis/kz_gate_drop/gptq_value/REPORT.md](#record-21)
- [22. llama-output/qwen3_5-0_8b-layer-analysis/l2norm_fix/REPORT.md](#record-22)
- [23. llama-output/qwen3_5-0_8b-layer-analysis/merged_kz_layer128/COMPARISON.md](#record-23)
- [24. llama-output/qwen3_5-0_8b-layer-analysis/merged_kz_layer128/REPORT.md](#record-24)
- [25. llama-output/qwen3_5-0_8b-layer-analysis/merged_qvabz_layer128/REPORT.md](#record-25)
- [26. llama-output/qwen3_5-0_8b-layer-analysis/merged_qvabz_layer128_baseline/REPORT.md](#record-26)
- [27. llama-output/qwen3_5-0_8b-layer-analysis/one_layer_128/REPORT.md](#record-27)
- [28. llama-output/qwen3_5-0_8b-layer-analysis/one_layer_128_fused_w4/REPORT.md](#record-28)
- [29. llama-output/qwen3_5-0_8b-layer-analysis/one_layer_128_fused_w4_ffnw8/REPORT.md](#record-29)
- [30. llama-output/qwen3_5-0_8b-layer-analysis/one_layer_128_fused_w4_l2fixed/REPORT.md](#record-30)
- [31. llama-output/qwen3_5-0_8b-layer-analysis/one_layer_128_fused_w4_l2float/REPORT.md](#record-31)
- [32. llama-output/qwen3_5-0_8b-layer-analysis/one_layer_128_fused_w4_native_l2/REPORT.md](#record-32)
- [33. llama-output/qwen3_5-0_8b-layer-analysis/one_layer_128_fused_w8/REPORT.md](#record-33)
- [34. llama-output/qwen3_5-0_8b-layer-analysis/one_layer_128_w4/REPORT.md](#record-34)
- [35. llama-output/qwen3_5-0_8b-layer-analysis/rms_z_gate_analysis/REPORT.md](#record-35)
- [36. llama-output/qwen3_5-0_8b-ppl/REPORT.md](#record-36)
- [37. llama-output/qwen3_5-0_8b-split-z-sqnr/comparison.md](#record-37)
- [38. llama-output/qwen3_5-0_8b-split-z-sqnr/w8a16_fp16state.md](#record-38)
- [39. llama-output/qwen3_5-mainline-integration/cross-model-review/REPORT.md](#record-39)
- [40. llama-output/qwen3_5-prefill-optimization/REPORT.md](#record-40)
- [41. llama-output/qwen3_5-prefill-optimization/baseline/REPORT.md](#record-41)
- [42. llama-output/qwen3_5-prefill-optimization/context4096_ar128/REPORT.md](#record-42)
- [43. llama-output/qwen3_5-prefill-optimization/decode_diagnosis/PREFILL_REPORT.md](#record-43)
- [44. llama-output/qwen3_5-prefill-optimization/decode_diagnosis/REPORT.md](#record-44)
- [45. llama-output/qwen3_5-prefill-optimization/image_benchmark/REPORT.md](#record-45)
- [46. llama-output/qwen3_5-prefill-optimization/image_benchmark/encoder_accuracy/REPORT.md](#record-46)
- [47. llama-output/qwen3_5-prefill-optimization/image_benchmark/encoder_accuracy/repair/REPORT.md](#record-47)
- [48. llama-output/qwen3_5-prefill-optimization/multimodal_calibration/REPORT.md](#record-48)
- [49. llama-output/qwen3_5-prefill-optimization/multimodal_calibration/performance/REPORT.md](#record-49)
- [50. llama-output/qwen3_5-prefill-optimization/release_ar128/README.md](#record-50)
- [51. llama-output/qwen3_5-prefill-optimization/text_accuracy/REPORT.md](#record-51)
- [52. llama-output/qwen3_5-size-comparison/REPORT.md](#record-52)

---

<a id="record-01"></a>

## 记录 01：cross-model-fixes / REPORT.md

原始记录位置：`llama-output/qwen3_5-mainline-integration/cross-model-fixes/REPORT.md`。

# Lincal3.5 共享优化流程与兼容性修复

日期：2026-10-15。

按用户最新要求，`lincal3_5-0_8b`、`lincal3_5-2b` 与 Qwen3.5 共用优化好的 GDN、量化和校准流程。Lincal 的 SWA 语义保留。此前[跨模型 review](README_QWEN35_TEST_RECORDS.md#record-39)中的两项 P1 已修复；共享 W4 recipe 缺少对应 GPTQ 的 P2 通过接入同一校准/GPTQ 流程解决。完整 Lincal 的精度和性能仍需用真实权重验收。

## 最终配置

| 项目 | Qwen3.5 / Lincal3.5 共同配置 |
|---|---|
| 输入投影 | K/z 合并 W8 per-channel；q/v/a/b 合并 W4 group32 |
| 其他 Linear | W4 group32；GDN out_proj 使用 fixed-grid GPTQ |
| Norm | 原生 Q/K norm、RMSNorm 与 gated RMSNorm |
| 外部 cache | conv、recurrent state、attention K/V 为校准得到的 UINT16 编码；输入输出及 AR128/AR1 对齐 |
| Prefill | AR128，FP16 chunk64 / inverse leaf8；只输出选定位置的 logits；padding 不改变有效 conv/recurrent state |
| Decode | AR1、DeltaNetCoreVtcm；精确 radix-16 归约构建配置不变 |
| 校准 | 单个 AR128 图串行校准，派生 warm-cache AR1 并合并激活范围；采用 `--calibration_device` 指定的设备 |
| MHA/SHA | 两个家族均关闭 MHA→SHA，使用同一 GQA 图变换 |
| 视觉 | 共享 W8A16 视觉配置及配对图文校准 |
| 入口 | `llama.py`，hybrid / kv，batch=1；默认物理 context 为逻辑长度加128，shared buffer 开启 |

## 具体修复

### 1. Lincal 默认校准的负 mask 维度

旧流程给 Lincal 添加了物理 padding，却仍使用完整物理长度作为校准 AR，并以较短逻辑长度建立 causal mask。现在两家族均使用 AR128 校准及物理 `max_seq_len`，通过 `Qwen35TextDecoder` 执行共享 GPTQ 和范围合并。

回归覆盖两个 Lincal 注册版本、逻辑长度128和1024，以及 kv/hybrid 入口。逻辑1024的默认物理容量为1152，校准 AR 为128，避免原来的 `[1152,-128]`。

### 2. SWA 双 mask 与固定 cache

共享流程根据 positions 输入的位置定位 cache，不再假设前缀总是 `hidden/mask/positions`。Lincal 的前缀为 `hidden/causal_mask/window_mask/positions`。

校准中的 cache 保持时间顺序，window mask 为每个 query 限定最近的 sliding window，包括当前 AR128 分块内部。更新后只保留固定长度的窗口历史；派生 AR1 时仅改变 window mask 的当前 token 区域，SWA cache 不随 AR 扩容。单独识别 GDN recurrent state，防止 state 的128维恰好等于 history 长度时被误扩容。

运行时继续使用现有 SWA 环形 cache 与相应 mask。模型 metadata 在存在 SWA 时发布 `get_sliding_window`，混合 full/SWA 模型也如此。

纯 SWA 模型没有全历史 FA 计算，跳过 FA256 桶和内部 FA 裁剪。混合 full/SWA 模型仍可创建 FA256 桶，但只裁剪 full-attention cache 和 causal mask。

### 3. 生成图 reshape 的重复维度

混合模型小图测试发现：若分组 query 行数 `groups × AR` 恰好等于 context 长度，原来的 `shape.index(context)` 会误改 query 行数。例如 AR128、4组、context512时，reshape中同时存在两个512。

现在从实际发生变化的输入轴和其后连续元素数量确定输出轴，保留 query 行数，只修改 history 维度。混合模型的短桶和完整图均执行了回归。

### 4. 通用 runner 的最后一个 AR 分块越界

`MultimodalPromptProcessor::Metadata::can_prefill()` 在修改 cache 之前检查：

- 起点、token 数和总 cache 容量；
- 最后一块起点不超过编译图的 `max_prefill_past`；
- 最后一块的完整 AR 位置范围不超过物理 context。

检查避免普通模型在非 AR 对齐的 context 下生成越界的 padding 位置，也保留有物理 padding 的优化模型的有效输入范围。原生测试覆盖非对齐1000、对齐1024、Qwen物理4224/逻辑4096、续写起点、小容量和非法输入。

## 验证结果

- 最终回归日志（历史产物已清理）：**103项测试、12项子测试通过**。包括两个模型家族的入口/投影/recipe、SWA、GQA、cache encoding、norm、GPTQ、custom op输出encoding、量化/浮点Convert、observer、视觉GELU，以及原生 C++ ASAN/UBSAN cache测试。
- AR128校准与逐 token FP32 decode 比较覆盖纯 SWA、混合 FA/SWA，样本300 token、warm起点256、window8，验证 AR大于window、滑窗历史更新和连续状态；允许的绝对误差为1e-5、相对误差为2e-5。
- 真实执行小模型的 W4/W8校准、out_proj GPTQ、AR1派生、共享范围、GQA改写、padding处理及FP16 ChunkGDN生成图；混合模型同时验证 FA256 桶。
- 小型 PTE（历史产物已清理）、编译记录（历史产物已清理）、输入类型与GPTQ记录（历史产物已清理）、复现脚本（历史产物已清理）：SM8750，1层GDN+1层SWA、dim64，AR128/AR1。两方法各有一个 QNN delegate，无 CPU fallback；两个公开mask和全部公开cache为UINT16，PTE为708,648字节。生成的QNN二进制包含 `DeltaNetOpPackage::DeltaNetCoreVtcm.quint16.hvx`。
- Android构建日志（历史产物已清理）：现有 build-android 配置增量编译 `qnn_multimodal_runner`、`qnn_llama_runner` 成功。
- `git diff --check` 通过；修改过的 Python 文件已用 ufmt 格式化。修复后再次核查 MHA/SHA、SWA更新、OP映射、mask裁剪和模型配置分支，没有发现本次范围内尚未处理的阻断项。

## 验证范围

本轮没有导出完整 Lincal 0.8B/2B，没有重跑整模 PPL、手机生成或性能，没有更新手机上的既有模型/runner。小模型验证证明流程、输入类型与状态布局可用，不能据此给出完整 Lincal 的精度或性能结论。此前 Qwen3.5 的完整PTE与实机数据仍是历史验收结果。

---

<a id="record-02"></a>

## 记录 02：qwen3_5-prefill-optimization / FINAL_SUMMARY.md

原始记录位置：`llama-output/qwen3_5-prefill-optimization/FINAL_SUMMARY.md`。

# Qwen3.5-0.8B：保留 AR128 / chunk64 版本

**普通 `llama.py` 已集成优化和图文校准修复，当前导出及设备验证见第19节；此前修复前后的配对性能复测见第18节。** 下方保留各阶段的历史记录。

2026-10-14 按用户要求停止性能优化，保留已完成端到端验证的版本：

**`qwen35-prefill-ar128-chunk64-20261014`**

- **对外 prefill AR = 128**：每次模型调用处理128个token。
- GDN内部 chunk = 64：128个token分为两个块，第一块的末尾state交给第二块。
- 每个GDN层仍产生128个位置的attention output；生成用的LM head只投影选定的最后一个位置。
- KV decode AR = 1。`get_ar_len=1`是decode元数据，不代表prefill改成了1。

## 1. 最终实测

设备SM8750，Qwen3.5-0.8B，纯文本输入，128-token prompt，温度0；不计模型加载时间。不同阶段分别做配对测试，表中范围来自各自两轮实测。

| 阶段 | 128→1端到端prefill | 说明 |
|---|---:|---|
| 本轮起点 | 848–877 token/s | 4K FA计算长度、原整数GDN |
| FA桶裁剪 + 精确整数归约 | 1208–1231 token/s | 保持与起点逐元素一致 |
| **选定版本：FA裁剪 + FP16 chunk64 GDN** | **1684–1707 token/s** | 数值有变化，PPL和生成已验证 |

选定版本的128→1端到端耗时约75–76 ms；完整24层prefill graph配对测试平均73.044 ms，即约1752 token/s（graph-only）。两种计时口径不能混用。128→64真实生成测试中的prefill为1580–1641 token/s，decode为37.28–37.75 token/s；同轮原版decode为39.55–40.03 token/s，没有证据说明decode也获得提升。

同一组8个独立128-token留出窗口，共1016个预测目标：

| 版本 | 抽样PPL |
|---|---:|
| 原版 / 精确整数桶版本 | 50.0248078862 |
| **选定chunk64版本** | **49.8616637907** |

这组PPL用于回归比较，不是完整标准WikiText评测。真实生成未出现乱码，措辞与原版不同；测试使用ignore_eos以固定输出预算，实际使用可以正常在EOS处停止。单层真实输入相对FP32 recurrence的SQNR：attention output约65.21 dB，state约52.63 dB；不能把单层结果当成整模型hidden state的SQNR。

证据：端到端生成（历史产物已清理）、完整图性能及49输出比较（历史产物已清理）、PPL（历史产物已清理）、单层GDN（历史产物已清理）。

## 2. 到底做了哪些优化

| 改动 | 原来的开销 / 问题 | 保留版本的做法 |
|---|---|---|
| **裁剪FA中被mask的历史padding** | 6个FA层对静态4K长度做计算，即使prompt很短 | 这份短prompt PTE内部FA长度为256：128个历史位置 + 128个当前位置；保留4K外部cache接口 |
| **GDN由逐token整数递推改为分块矩阵计算** | 18个GDN层各执行128次顺序state更新 | 两个64-token块；块内用矩阵计算处理token间校正，块之间传递state |
| **GDN内部使用FP16** | 原core内部每一步按UINT16 state网格舍入，且有整数拆分归约 | UINT16输入先按校准参数转换成FP16；块内state用FP16；128-token调用结束再按原校准参数写UINT16输出 |
| **稳定的三角求逆** | 直接对64×64下三角矩阵做Neumann矩阵幂，部分后续层中间值会溢出 | 先求8×8小块的逆，再逐级合并成16/32/64块；避免形成大矩阵幂 |
| **限制指数的数值范围** | 设备FP16 exp遇到很大的负数产生非有限值 | decay在取log前限制下界为2^-14；exp输入限制到[-16,0]；影响包含在实测误差和PPL中 |
| **修复量化/浮点边界转换** | 原builder在只有一侧带量化参数时选择Cast，不能正确处理这条混合精度路径 | 选择QNN Convert，让校准的scale/zero point参与UINT16↔FP16转换 |
| **处理Gather接Concat的设备重组问题** | odd gather用于MatMul时正确，直接接Concat时读到错误的分块结果 | 加入最小FP16 subnormal（2^-24）的加法，使结果物化；微小扰动已计入数值验证 |
| **Runner执行检查与计时** | 执行失败曾被误记为异常高吞吐；端到端开销不清楚 | 检查每次执行结果，记录prepare/execute/cache时间，并检查该prefill图支持的历史范围 |

数学主线（省略batch/head维度）：

```text
D[i,j] = exp(sum(log(decay[j+1:i+1])))，i >= j
T[i,j] = -beta[i] * dot(k[i], k[j]) * D[i,j]，i > j
R = (I - T)^(-1)
U = R @ [beta * (V - prefix_decay * (K @ S_in))]
Y = prefix_decay * (Q @ S_in) + [(Q @ K.T) * D] @ U
S_out = last_prefix_decay * S_in + K.T @ (last_decay_row * U)
```

8×8对角块分别求逆；合并两个块时使用下三角块矩阵的逆公式。这里的8和64都是内部算法分块，**对外AR始终为128**。

## 3. 哪些设置是继承的

本轮没有重新校准或重新量化投影权重。选定版本继承原来的权重codes、scale/zero point、原生Q/K RMSNorm和GPTQ结果：

- K/z合并投影使用W8；q/v/a/b合并投影使用W4；其他大Linear按原W4 group32配方；causal conv保持原W8。
- GDN out_proj保持原来的fixed-grid GPTQ结果。
- state、conv、FA cache的外部UINT16量化网格保持原样；这些值来自已有校准图，不是新写死的任意范围。
- GQA和只计算最后选定位置LM head的优化，在本轮起点已经存在，不重复计作本轮新增收益。
- **原KV decode图保留原精确整数自定义core**，本轮chunk改动作用于prefill。

注意：UINT16外部接口不变，并不意味着内部state还是逐token UINT16运算；prefill内部已是FP16。这正是选定版本与旧core的重要差别。

## 4. 代码位置

| 内容 | 文件 |
|---|---|
| **选定版本的GDN算法** | chunk_gdn.py（历史产物已清理）：`ChunkGDN.forward`、`triangular_inverse` |
| 将18个prefill core替换成chunk图 | export_chunk_prefill.py（历史产物已清理）：`replace_cores` |
| UINT16 / FP16转换修复 | [op_to.py](../../../../backends/qualcomm/builders/op_to.py)：`To.is_cast_node` |
| 与原KV图配对、保留QNN图名 | merge_chunk_hybrid.py（历史产物已清理） |
| bucket选择、执行检查、计时、历史范围检查 | [multimodal_prompt_processor.cpp](runner/multimodal_runner/multimodal_prompt_processor.cpp)、[multimodal_runner.cpp](runner/multimodal_runner/multimodal_runner.cpp) |
| 后端转换回归测试 | test_quantized_float_conversion.py（历史测试代码已清理） |

选定版本的prefill GDN由普通PyTorch图降低成原生QNN算子，**不调用DeltaNetQhpi.cpp的自定义recurrence实现**。KV decode仍使用原自定义op。因此，不能把后面的FP16 chunk提速归因于DSP自定义op的Karatsuba或O3。

## 5. 中间尝试与未采用项

- 精确radix16 Karatsuba归约把4组整数乘积降为3组，保持原FP32合成顺序；O2独立收益约2%，O3与FA桶组合得到约1200 token/s。它属于前一阶段整数core版本，未替换选定PTE的原KV core。
- radix32归约改变舍入；原生模型RMSNorm融合/DLBC收益小；自定义core内部FP16 state反而慢；AR512没有达到预期，均未采用。
- 首版chunk图约68 ms，但后续层矩阵幂溢出，结果无效，未采用。
- SLC在本机当前SDK/固件下报`Request feature slc allocator enable with value 1 unsupported`，没有可用性能结果，未采用。
- 后续`chunk_preconditioned.py`单核约0.93 ms，但未完成整模和PPL验证；`chunk_fused.py`只完成参考检查。**按用户要求停止，不纳入保留版本。**

详细历史及失败记录见[REPORT.md](README_QWEN35_TEST_RECORDS.md#record-40)。

## 6. 保存和适用范围

- 起点快照：baseline/manifest.json（历史产物已清理），标记`qwen35-prefill-baseline-20261014`。
- 选定快照：release_ar128/manifest.json（历史产物已清理），保存准确PTE/SO/runner校验值、源码及测试证据索引；未创建Git commit/tag。
- 选定PTE：hybrid_chunk_qnn.pte（历史产物已清理），941852424 bytes。KV和prefill目前为两个context，未跨context共享权重。
- **1684–1707仅代表128-token prompt。这份冻结PTE的hybrid prefill最多支持256-token prompt，KV容量为4096。** Runner会阻止越过当前prefill历史范围，不能把4K cache容量理解成这份短prompt图已支持4K prefill。后续另导出的4096输入测试图见[长上下文实测](README_QWEN35_TEST_RECORDS.md#record-42)，冻结PTE和原始结果保留。
- 前一阶段支持更长prompt的四方法PTE仍保留在optimized/（历史产物已清理），其已验证速度约1200–1250 token/s。

## 7. 使用与复现

使用已部署的选定版本，默认采用实测的同一份128-token prompt，正常遵循EOS：

```bash
python llama-output/qwen3_5-prefill-optimization/release_ar128/run_selected.py
# 仅生成首token，复现prefill口径：
python llama-output/qwen3_5-prefill-optimization/release_ar128/run_selected.py --output-tokens 1
```

模型AR已经编译进PTE，无需在runner再设置AR。以上命令用于复现这份冻结快照。普通 `llama.py` 的集成入口、默认配置和重新校准后的成绩见第19节。

## 8. 后续4096输入测试

按用户新指定的口径，另导出长上下文PTE，保持prefill AR=128、GDN chunk64 / leaf8 / FP16和原KV整数VTCM核，测试4096输入→1输出、16输入→4080输出。物理容量设为4224（128对齐并留出首token位置），prefill内部FA计算长度扩展为4096；与冻结短prompt PTE的内部FA长度256不同。

每项两轮，强制忽略EOS。完整结果及统计口径见[context4096_ar128/REPORT.md](README_QWEN35_TEST_RECORDS.md#record-42)，导出和测试脚本分别为export_context4096.py（历史产物已清理）、benchmark_context4096.py（历史产物已清理）。本次只扩展上下文并测量性能，没有继续更改GDN算法或量化配方。

## 9. 后续decode采样开销修复

4096总长度测试的33.06 token/s包含约4.15 ms/token的CPU采样开销。temperature=0且repetition penalty关闭时，原runner仍先将整个词表UINT16 logits转成float再扫描取argmax；现改为直接对UINT16做ARM NEON归约，并保持并列最大值选第一个位置的规则。

同一个PTE、16输入→4080输出，两轮新结果为37.34和37.30 token/s，平均**37.32 token/s**，decode阶段平均109.2945秒，较原33.06提高**12.90%**。新采样约0.173 ms/token；两轮长程生成文本均与原版逐字节一致。主机和设备上的标量/NEON边界测试通过。prefill AR仍为128；权重、校准、PTE和GDN实现未改，冻结快照保留。

官方同代Galaxy S25 / 8 Elite的4K记录中，NPU为37.72 token/s，CPU为52.20 token/s，详见[官方perf.yaml](https://github.com/qualcomm/ai-hub-models/blob/main/src/qai_hub_models/models/qwen3_5_0_8b/perf.yaml)。具体测试口径和量化格式与本机不同。诊断、分阶段计时、代码位置、部署的新版runner及复现命令见[decode_diagnosis/REPORT.md](README_QWEN35_TEST_RECORDS.md#record-44)。

## 10. 采样修复后的prefill复测

AR保持128。新旧runner交替测试，128输入使用原冻结短prompt PTE，4096输入使用长上下文PTE。此次短prompt存在较大模型执行波动，保留初次两轮和追加四轮的全部结果：新版128→1平均80.83 ms / 1583.51 token/s，中位79 ms / 1620.25 token/s，单次77–92 ms；新版4096→1两轮平均4092.50 ms / 1000.86 token/s。

采样只在prompt末尾发生一次，所以没有长decode那样的收益。本轮4K新旧runner的执行时间差约95 ms，掩盖了执行之外约3.47 ms的减少，未测到prefill整体提速。4K模型图执行约占97.8%，未修改模型或AR。早先1684–1707与1050.53 token/s的原始记录仍保留；当前复测统计、波动说明、官方同代4K NPU prefill参考889.51 token/s及全部原始数据见[decode_diagnosis/PREFILL_REPORT.md](README_QWEN35_TEST_RECORDS.md#record-43)。

## 11. 图片输入影响

此前所有性能记录都是纯文本。补测一张448×448的猫照片，原视觉编码器输出196个图像token，编码约26 ms。使用同一个4K PTE、AR128、输出64 token，两轮平均：同一句短文字无图19 token时首token125 ms，加图217 token时279 ms；固定合并输入4096 token时，纯文本首token4035 ms，包含图片为4041.5 ms。各项decode均约37 token/s，未显示明显变化。

这份PTE的带图内容有错误：hybrid短图文输出异常，4K图文描述了背景文字，补测KV模式也未正确描述图片。因此，目前只有性能对照完成，图文精度尚未通过；不能把它视为多模态模型已完整验证。图片编码、计时范围、输出问题及全部原始证据见[image_benchmark/REPORT.md](README_QWEN35_TEST_RECORDS.md#record-45)。

## 12. 纯文本与Transformers FP32的精度对照

使用相同token IDs、位置和因果历史做teacher-forced计分，设备输入来自实际embedding PTE，原生产PTE负责PPL，追加已有hidden输出的诊断PTE负责LM head之前的最终hidden SQNR。44个抽查位置的全词表logits codes与原PTE逐元素完全一致。

| 输入 | Transformers FP32 PPL | Prefill AR128 PPL | KV AR1 PPL | Prefill最终hidden SQNR | KV最终hidden SQNR |
|---|---:|---:|---:|---:|---:|
| WikiText连续1024 token | 10.87851 | 11.72747 | 13.70082 | 11.22 dB | 7.31 dB |
| 8个冷启动128-token窗口 | 41.32729 | 49.85514 | 67.40884 | 11.88 dB | 6.99 dB |

纯文本存在精度损失，连续1K的prefill PPL较FP32高7.80%，KV高25.94%。末端RMSNorm之前已分别只有11.91和7.96 dB；零历史首token也已明显偏离，因此需要优先逐层定位KV首次forward，不能只解释为长上下文cache累计。实际embedding本身SQNR为84.24 dB，误差远小于最终hidden。

连续1K仅是WikiText-2 raw test前1024 token，不是完整测试集标准PPL；冷窗口来自此前校准文章的保留后缀。此前49.86对旧量化版本50.02的验证没有包含FP32基准，不能据此认为浮点精度已恢复。原PTE和AR128保留，本轮没有更改量化配方。逐128-token变化、量化网格、端点比例、原始输出及复现脚本见[text_accuracy/REPORT.md](README_QWEN35_TEST_RECORDS.md#record-51)。

## 13. 视觉encoder输出与输入边界精度

对设备实际视觉PTE（SHA256 `2fa0f90e...f3930`）补测相同猫照片，比较encoder和merger之后的196×1024图像embedding。当前静态PyTorch FP32相对HF FP32为**98.81 dB**，设备视觉PTE为**−3.46 dB**、cosine **0.541126**。大误差在进入语言模型之前已存在；具体是视觉PTQ还是后端执行导致，需要CPU QDQ与设备逐层对照后确定。

当前decoder输入网格为scale `4.440613793121884e-06`、zero point `31008`，实数范围仅[-0.13769,0.15332]。设备图像embedding有**55.90%**越界；独立使用HF正确视觉输出也有**24.09%**越界，经过现有输入QDQ后SQNR只有**3.78 dB**。因此，视觉encoder恢复后还需要按图文输入重新校准decoder输入及后续激活。本轮没有替换模型。预处理一致性、量化参数来源、设备原始输出及复现见[视觉精度报告](README_QWEN35_TEST_RECORDS.md#record-46)。

## 14. 视觉encoder修复与部署

单独重建并部署视觉PTE：内部改用FP16，保留FP32输入/输出；Qwen3.5视觉block显式展开GELU(tanh)，使用数学等价的sigmoid公式，避免native QNN Gelu的erf语义及半精度误差。没有新增自定义DSP op，整个视觉模型仍为一个QNN delegate。

同一猫图raw的新旧配对，SQNR **−3.46→34.68 dB**，cosine **0.541→0.999830**；另两张图为33.03、33.79 dB。warm-up 3次、计时10次，encoder耗时 **25.62→41.30 ms**；PTE文件 **107→206 MB**。显式tanh的95 ms版本未采用，sigmoid版本三图平均约40 ms。3个数学/导出回归测试通过。

当前4K workspace的encoder链接已切换到新版本，原PTE与冻结快照保留。以后标准导出根据 `Qwen35VLEncoder.use_fp16=True` 选择FP16视觉塔；4K部署脚本优先使用保存的新视觉PTE。语言模型PTE、量化配方和prefill AR128均未改变。

同一个Transformers FP32语言模型使用新视觉特征时，已能正确描述两只猫与遥控器。**完整量化PTE的带图输出仍错误**：现有decoder输入网格继续截断新特征的24.08%，图文输入校准和此前量化语言路径误差仍需继续处理。详细演化、代码、代价、实际整模输出及部署校验见[视觉修复报告](README_QWEN35_TEST_RECORDS.md#record-47)。

## 15. 修正 GELU 后的视觉 W8 对照

单独导出并上设备测试两版视觉 W8，保留 GELU 修正。51 个 Linear 权重全部为 INT8 per-channel；W8A16 使用 GPU 激活校准及 patch bias 范围保护，另一版保留 FP16 激活。语言 PTE 和 AR128 未改。

本轮相同猫图 raw、HF FP32 参考、warm-up 3 次、执行 10 次的配对结果：

| 视觉配置 | 输出 SQNR | Encoder 耗时 | PTE 大小 |
|---|---:|---:|---:|
| FP16 基线 | 34.68 dB | 40.17 ms | 205.55 MB |
| W8A16 | 23.73 dB | 28.18 ms | 108.48 MB |
| W8＋FP16 激活 | 23.52 dB | 45.84 ms | 107.17 MB |

W8A16 耗时减少 29.85%、体积减少 47.23%，但精度明显降低，未参与校准的狗图只有17.88 dB。去掉激活量化后狗图改善至20.27 dB，仍未恢复 FP16 精度；CPU 权重量化图本身也已偏离。两版 W8 特征接同一个 HF FP32 decoder 都能正确描述两只猫和遥控器，这仅是视觉隔离验证。

W8 未解决现有 decoder 输入网格问题，猫图特征仍约24.6%越界。该阶段尚未重测 W8 的完整量化整模生成，后续手机结果见第16节；默认部署继续保留 FP16 视觉塔，W8 PTE 和全部逐层结果留在独立实验目录。详细三图误差、逐层边界、产物、SHA256及复现见[视觉修复报告第7节](README_QWEN35_TEST_RECORDS.md#record-47)。

## 16. W8A16 视觉 PTE 接完整量化语言 PTE 的手机结果

使用实际 ExecuTorch 图文 runner，显式指定已编译的 W8A16 视觉 PTE，接现有 token embedding 和混合 W4/W8 语言 PTE；不是 HF decoder。三个 PTE 的完整路径和哈希见产物清单（历史产物已清理）。

同一猫图，217 输入 token、最多64输出、temperature=0、遵循 EOS：hybrid AR128 的视觉编码28.13 ms、首token271 ms、decode39.08 token/s，但生成 `** 1. 1 1 ...`；KV AR1 的视觉编码30.10 ms、首token5763 ms、decode36.93 token/s，生成 `(1) 3 3 3 ...`。两项均正常运行，但达到长度上限前一直重复数字，图文质量未通过。KV的长首token时间来自逐token处理217个prompt输入，不能用来比较AR128 prefill。

另做纯文本 `2+3` 对照，输出 `5` 并正常 EOS。每项仅一次功能诊断，速度不当作重复 benchmark 结论。W8 特征接 HF FP32 decoder 正常、接量化语言 PTE 异常；现有 decoder 输入网格仍截断约24.55%的该图特征，后续需要修复图文输入校准并复测。原始输出、日志、计时和复现脚本见[视觉报告第8节](README_QWEN35_TEST_RECORDS.md#record-47)。

## 17. 图文输入校准与 partial prefill state 修复（当前4K手机版本）

2026-10-15完成修复并部署。原4K产物来自只含8段Wiki文本的校准，图片未进入统计；本轮使用实际W8视觉特征，CUDA统计8段文本和猫/女神像两张图，狗图留作验证，同时覆盖AR128和带历史AR1。

Decoder输入范围由 `[-0.1377, 0.1533]` 扩为 `[-4.6405, 1.6444]`。图像特征通过输入QDQ的附加SQNR从约3.75提升到76 dB，狗图越界降到0%；这不是视觉encoder自身变成76 dB。全部48路cache IO编码和初始化zero point同步，语言W4/W8权重及GPTQ codes/scales保持原值。

还修复了两个使“只扩范围”仍失败的问题：49个decoder RMSNorm改成原生二维RMSNorm，避免20个epsilon被宽网格量化成0；AR128最后一个block按last_token_index停止GDN state更新，conv cache读取有效末尾4个值。Gather保留匹配编码后独立转换到cache网格。5项回归测试通过，两个方法均单个QNN delegate、无CPU回退，没有新DSP op。

实际完整量化PTE：hybrid猫图输出“两只猫睡在粉色毯子上，旁边有遥控器”，KV输出相同关键信息；未参与校准的狗图在两种模式下均描述白色狗、绿草和伸出的舌头，四项均正常EOS。`2+3`恢复为`5`并EOS。单次hybrid图文首token269/288ms，decode约36–38 token/s；不是长程性能benchmark，AR128保持。

| 同一1K WikiText测试子集 | HF FP32 | 原PTE | 新PTE |
|---|---:|---:|---:|
| Prefill AR128 PPL | 10.87851 | 11.72747 | **11.74834** |
| KV AR1 PPL | 10.87851 | 13.70082 | **11.62953** |

Prefill基本持平（+0.18%），KV降低15.12%；仍有约7–8%的FP32差距，未当作完整测试集PPL或全面多模态精度通过。

当前手机默认模型已切换，新语言PTE SHA256为 `e0b2ae5a...baaeb45`，视觉仍为已测试W8A16 `d8fc472a...fe19a02`。原语言PTE与FP16视觉版本保留回退。产物、手机输出、PPL、三个问题的演化和复现见[完整修复报告](README_QWEN35_TEST_RECORDS.md#record-48)、当前release清单（历史产物已清理）。本轮修复的是4K图，冻结短prompt FA256性能产物未重导。

## 18. 图文修复后的性能复测

2026-10-15，在同一SM8750手机上显式选择原版4K PTE和当前修复版，使用相同W8A16视觉塔、embedding、tokenizer、快速UINT16 argmax runner及DeltaNet包。AR128保持；各prefill用例交替测试3轮，16→4080长decode按原版、修复版、修复版、原版顺序各2轮。模型加载不计时，忽略EOS并校验实际输入/输出数量，预热不计入均值。

| 测量 | 原版4K PTE | 修复版4K PTE |
|---|---:|---:|
| 纯文本128→1 prefill | 127.33 ms / 1005.24 token/s | **128.33 ms / 997.40 token/s** |
| 纯文本4096→1 prefill | 3998.33 ms / 1024.43 token/s | **3965.67 ms / 1032.87 token/s** |
| 纯文本16→4080 decode | 36.9466 token/s | **36.9047 token/s** |
| 图文217→64首token | 282.33 ms | **281.33 ms** |
| 图文4096→1首token | 4003.33 ms | **4019.67 ms** |

Prefill各项平均耗时变化均在±1%内，长decode吞吐降低0.11%，本轮没有测到明显性能退化。短图文的视觉编码仍约29.28ms。Decoder step约26.50ms/token、采样0.175ms/token，快速argmax仍生效。

本轮**两版均是FA内部长度4096**，此前1684–1707 token/s为冻结的FA256短图，不能把两者差异归因于本次图文精度修复。连续长decode的电池温度从32.7升至38.4°C，保留全部样本，不将小幅均值差当作严格等速或确定加速的证明。28次正式测量、2次预热全部完成；本轮没有重导模型或改动权重。

逐轮结果、阶段拆分、计时口径与复现见[性能复测报告](README_QWEN35_TEST_RECORDS.md#record-49)、本轮汇总（历史产物已清理）。

## 19. 普通 llama.py 入口集成（2026-10-15）

当前版本标记为 `qwen35-mainline-optimized-20261015`，没有创建 Git commit/tag。普通入口直接导出，不再导入 `llama-output` 中的实验模块或修改模型注册表。导出命令见export.sh（历史产物已清理），通用用法见[README](README.md)。

| 范围 | 固定采用的配置和修复 |
|---|---|
| 投影与量化 | K/z 合并 W8 per-channel，q/v/a/b 合并 W4 group32；其余大 Linear W4；18 个 GDN out_proj 自动 fixed-grid GPTQ |
| Prefill | 对外 AR128；FP16 chunk64 / inverse leaf8；仅投影最后选定位置的 logits；padding 不更新有效 recurrent/conv state |
| FA 与 cache | GQA 不复制 KV；同一 PTE 自动选择 FA256 短图或完整 4K 图；48 路 UINT16 cache 的 scale/zero point 来自校准，并对齐输入输出编码 |
| 校准 | GPU 上串行校准一个 AR128 模型，派生 AR1；合并 AR128 与 warm AR1 的范围；文本和图片按样本配对，decoder 使用 QDQ 视觉特征及正确 M-RoPE |
| 视觉与 norm | 51 个视觉 Linear 为 W8A16；保留 GELU(tanh) 等价展开和 INT32 bias 范围保护；Q/K、decoder 与 gated RMSNorm 保留原生归一化 |
| Decode 与部署 | 精确整数 VTCM core、UINT16 greedy argmax、默认 shared buffer；普通入口支持纯文本和图文；本地模型目录同时用于权重、tokenizer 和预处理 |

已删除 GDN kernel/state/conv/cache dtype 的实验 CLI 开关，以及投影拆分/合并、全 W4 等环境变量分支；OP 包内实现和实验变体保留。`--gdn_op_package_dir` 指定源码目录，入口按 SDK 和 SoC 增量构建 host、Android、DSP 库。

### 算子库一致性修复

本次接入发现工作目录里的旧 DSP 构建仍默认丢弃归约输入的低4位。同一份第一版 PTE，只替换 DSP 库，1K KV PPL 从13.9376降至11.6735。现已将精确 radix-16 拆分归约设为默认，并实际重建；新构建与原验证库在该测试的 NLL 完全一致。最终 DSP SHA256 为 `59ca74c1109bbc10c6a6e5234f3647729681ed664c573954c54c933c75e4ab36`。同时去掉 state 输入多余的 `.float()`，避免额外 observer 破坏共享 state 编码。

### 最终设备验收

SM8750 / Qwen3.5-0.8B，temperature=0，不计加载时间。三轮性能测试固定输出预算并忽略 EOS；功能测试另外遵循 EOS。

| 用例 | 本次最终 PTE |
|---|---:|
| 文本128→1，自动 FA256 | 85/87/86 ms，平均 **86 ms / 1488.37 token/s** |
| 文本4096→1，完整 FA4K | 3892/4084/4021 ms，平均 **3999 ms / 1024.26 token/s** |
| Decode 输入16、输出预算496 | 实测495次decode forward，**38.35 token/s** |
| 图文217→64首token，含视觉编码 | **200.33 ms**；视觉约30 ms；后续decode **37.98 token/s** |

历史1684–1707 token/s来自较早的冻结短图；本次完整校准后的实测为上表数值，不能用历史峰值代替本轮成绩。第18节当前修复版的128-token成绩为997.40 token/s，但两轮不是交替配对测试。

另对较早冻结短图与本次最终PTE交替配对三轮，使用同一新runner、embedding、tokenizer及128-token输入：冻结短图76/78/80 ms（平均78 ms），本次86/84/86 ms（平均85.33 ms）。本次耗时高约9.40%，尚不能宣称保住了旧短图的全部性能。两份图的校准和精度修复不同；仅凭计时不能把增量归因于某个算子，具体样本见配对短输入结果（历史产物已清理）。本次保留的是通过当前图文及PPL验收的优化组合。

同一 WikiText-2 raw test 的前1024个token、1023个预测目标，本次使用新 embedding PTE 的实际设备输出：

| 方法 | 原修复版PPL | 本次PPL | HF FP32参考 |
|---|---:|---:|---:|
| KV AR1 | 11.6295 | **11.5479** | 10.8785 |
| Prefill AR128 | 11.7483 | **11.6818** | 10.8785 |

这是同一片段的回归评测，不是完整测试集PPL。视觉输出对HF的SQNR：校准猫图23.73 dB、留出狗图17.88 dB，均为有限值，与此前修复版相当。猫图、狗图分别在hybrid/KV生成中正常结束；普通 `llama.py --pre_gen_pte` 的文本和图文入口也通过。Python说明能够生成，但仍有事实性错误，不能据此宣称全面生成质量通过。

75项测试及50项子测试通过。最终 `kv_forward`、`prefill_forward`、`prefill_forward_256` 都是单个QNN delegate，没有CPU fallback；AR128的实际hidden输入为 `[1,128,1024]`。最终语言PTE为499,518,720字节，其他资产包括W8A16视觉PTE、embedding PTE及tokenizer。

复现和证据：PTE目录（历史产物已清理）、配置及校验值（历史产物已清理）、性能（历史产物已清理）、KV PPL（历史产物已清理）、prefill PPL（历史产物已清理）、测试（历史产物已清理）、普通图文入口（历史产物已清理）。

### Review 后修复（2026-10-15）

四项问题已修复：OP 增量构建跟踪有效编译配置，普通入口固定精确 radix-16 归约；hybrid 的内部 FA 容量按 AR128 向上对齐，并在加载模型前检查物理容量；通用多模态校准支持同批样本图片数量不等；KV-only 跳过生成用 prefill 图、FA256 桶及 AR128 embedding 导出，保留必要的 AR128 校准。

第二轮检查还将 prefill 容量限制限定于 hybrid，避免误限制 KV-only 的显式上下文。详细修改、验证结果和验证范围见 [修复及复审记录](README_QWEN35_TEST_RECORDS.md#record-03)。本轮没有重导完整 PTE 或重跑手机性能，上述性能及 PPL 仍是第19节原始验收结果。

### Lincal3.5 接入与跨模型修复（2026-10-15）

按用户确认，Lincal3.5 0.8B/2B 与 Qwen3.5 共用 K/z W8、其余 Linear W4、out_proj GPTQ、AR128 FP16 ChunkGDN 和 AR1 VTCM。补齐 SWA 双 mask 校准、固定窗口 cache 的 AR1 派生、混合 FA/SWA 的 metadata 与选择性裁剪；纯 SWA 跳过 FA256。另修复通用 runner 的末尾 AR 分块越界，以及 query 行数恰好等于 context 时 reshape 裁剪误选轴的问题。

103项测试、12项子测试通过；小型 Lincal PTE 的 prefill/decode 均为单 QNN delegate，无 CPU fallback，公开双 mask/cache 均为 UINT16；两个 Android runner 编译成功。完整 Lincal PPL 与手机性能尚未验收，不能沿用 Qwen 的数值作为 Lincal 成绩。详见[接入与修复报告](README_QWEN35_TEST_RECORDS.md#record-01)。

---

<a id="record-03"></a>

## 记录 03：review-fixes / REPORT.md

原始记录位置：`llama-output/qwen3_5-mainline-integration/review-fixes/REPORT.md`。

# 普通 Qwen3.5 入口：Review 修复与复审

日期：2026-10-15。范围：前次 review 的四项问题及其调用路径。

## 修复内容

| 问题 | 修复 | 验证 |
|---|---|---|
| 切换编译参数后继续复用旧 OP 库，可能带回低精度归约 | Makefile 用每个目标自己的 `.build-config` 跟踪有效编译/链接参数、编译器版本、SDK 路径和源文件列表；配置变化才更新时间戳。普通 `llama.py` 入口通过 make 命令行固定 `FULL_PRECISION_REDUCTION=0 / SPLIT_REDUCTION=1 / KARATSUBA_REDUCTION=0`，使用此前验证过的精确 radix-16 拆分归约。 | 四个平台的增量构建回归测试；真实 v79 SDK 编译；普通入口实际构建 x86、Android、DSP，覆盖环境残留 `SPLIT_REDUCTION=0`。 |
| 非 AR128 整数倍的上下文预算，最后一个分块超出内部 FA 容量 | hybrid 内部容量采用 `max(256, ceil(max_seq_len/128)*128)`，不改变对外 cache 布局。显式物理容量不足，在 tokenizer/权重加载前报错。 | 511、1000、1024 长度的真实 FX 注意力图，比较裁剪前后最后分块的有效输出和全部 cache；测试 CLI 提前校验。 |
| 通用多模态校准 batch_size=2，样本图片数不同，默认 collate 堆叠失败 | `ModalityOutputCollator` 按样本顺序展开媒体输出，不再要求每个样本有相同图片数。 | 图片数量 `(1,0)`、`(0,1)`、`(1,2)`、`(0,0)`，实际运行 encoder 校准收集和 decoder embedding 合并；与逐样本期望精确比较。 |
| KV-only 先构造/导出生成 prefill 图再丢弃，额外耗时和占内存 | 将生成图准备放到独立方法；KV 分支只保留 AR1 decoder/embedding，跳过 logits 行选择、padding 改写、chunk 替换、FA256 桶和 AR128 embedding 导出。AR128 校准及可选 QDQ 评估文件仍保留。 | 故障注入确保 KV 不调用任何 prefill 改写；校准后的四层小模型分别运行 KV 与 hybrid 路径。 |

例如预算1000、prompt999：最后分块起点896；旧内部容量1000只允许起点872，新容量1024允许起点896。物理缓存默认仍是1128。对齐的1024/4096配置保持原来的内部 FA 容量。

## 验证证据

最终回归：**74项测试、9项子测试全部通过**。

- 回归测试（历史产物已清理）：相关量化、cache、GDN、GQA、视觉、导出及 SWA 路径；包含下述7项构建测试。
- 构建回归测试（历史产物已清理）：实际 GNU Make + 记录命令和输出的编译器替身，验证参数/编译器版本/SDK 变化会重建，相同配置不重建，删除源文件会重新链接，`make -n` 不创建文件，以及普通入口覆盖实验归约参数。
- 真实 SDK 构建记录（历史产物已清理）、对象/库哈希与时间戳（历史产物已清理）：先用低精度参数编译，再切换精确归约；对象和库均改变；第三次相同配置的哈希和时间戳完全不变。
- 普通入口实际构建（历史产物已清理）、配置和哈希（历史产物已清理）：QNN 2.49.40.260810、Hexagon SDK 6.7.0.0、工具21.0.04、目标v79；x86/Android/DSP均成功。
- 小模型生成图验证（历史产物已清理）、输入输出形状（历史产物已清理）：4层、3个GDN+1个FA、AR128、物理容量384；KV仅1个运行方法，hybrid有3个。所有输出有限，cache ABI保持；prefill不含自定义 recurrence core，decode保留3个。
- DSP 函数对比（历史产物已清理）：重建库与此前手机上的验证库解析 PLT 跳转目标后，21个DeltaNet函数的反汇编一致。完整动态库哈希因链接布局不同而变化，未把二进制哈希相等当作前提。
- 格式检查（历史产物已清理）通过；Flake8检查（历史产物已清理）通过，命令显式排除了C901复杂度度量，wrapper中的两个既有复杂方法没有在本轮重新拆分。`git diff --check`通过。

## 第二轮 Review

复查了构建配置失效、导出分支、FA容量与runner的`max_prefill_past`约束、多媒体顺序、cache编码与embedding生命周期。补充修正了“hybrid的容量限制误用于KV-only”：KV无需生成用prefill桶，不受其最小256/向上取整要求限制；CLI回归用例覆盖该区别。

清理了两个失效的模型导入和一个冗余`getattr`。上述四项问题未发现剩余阻断项。

## 验证范围

本轮没有进行完整0.8B/2B模型重新校准、PTE重导或手机性能/PPL复测；没有更新手机上的模型、runner或算子库。修复主要作用于增量构建、校准合批和导出控制流程；不据此宣称新的吞吐提升、完整导出峰值内存下降量或新的PPL结果。上一轮短prefill相对冻结短图的约9.4%耗时增量尚未重新评估。

---

<a id="record-04"></a>

## 记录 04：qwen3_5-0_8b-layer-analysis / HIDDEN_STATE_ERROR_REPORT.md

原始记录位置：`llama-output/qwen3_5-0_8b-layer-analysis/HIDDEN_STATE_ERROR_REPORT.md`。

# Qwen3.5-0.8B PTQ hidden-state error audit

## Scope

The hidden-state numbers below come from the CPU PT2E QDQ graph generated with the same W8A16 recipe, VTCM GDN path, CUDA calibration device, and Qwen3.5 decoder graph used for the device export. The comparison is FP32 graph versus the converted QDQ graph at the selected internal nodes. It is not an intermediate tensor dump from the HTP device.

The PPL numbers are from the actual SM8750 PTE runs.

## First failure points

| location | UInt16 GDN/state | FP16 GDN/state A/B | interpretation |
|---|---:|---:|---|
| layer 0 GDN norm output | 12.11 dB | — | initially healthy |
| layer 1 GDN norm output | -4.30 dB | — | error is already visible early |
| layer 10 GDN norm output | **-13.80 dB** | -12.49 dB | largest local GDN collapse |
| layer 10 `attention.out_proj` | **-15.82 dB** | -14.58 dB | propagates/amplifies the layer-10 GDN error |
| layer 10 block output | **-6.70 dB** | -5.31 dB | residual stream is now badly contaminated |
| final `norm` output (`mul_423`) | **-1.63 dB** | -0.72 dB | final hidden state before `lm_head` |
| output projection (`output.conv`) | about -0.40 dB | — | not the first large error source |

Other GDN norm outputs remain negative in later layers, for example layer 12 -3.53 dB, layer 13 -2.31 dB, layer 14 -1.90 dB, and layer 21 -1.53 dB. This is consistent with recurrent error accumulation after the layer-10 failure.

The per-node data is in `hidden_stage_metrics.json`.

## What the PPL says

| configuration | device PPL |
|---|---:|
| HF FP32 reference | 16.80 |
| W8A16 + UInt16 GDN/state + VTCM | 123.74 |
| W8A16 + FP16 GDN/state + DDR | 51.65 |

Keeping the GDN/state path in FP16 lowers PPL by 58.3%, while the final hidden SQNR improves by about 0.90 dB in the CPU QDQ comparison. This isolates the recurrent path as a major contributor, although the FP16 experiment also changes the core input and accumulation path, so it is not a state-buffer-only experiment.

## Calibration findings

1. `vision.json` contains two samples, but the export command used the default `--calib_limit 1`. The export lineage records `calib_limit: 1`; only one sample was used for the original export.
2. Repeating the hidden-state analysis with `--calib_limit 2` did not change the reported layer-10 or final-hidden metrics. Therefore, increasing this particular image count from one to two is not sufficient to explain or fix the collapse.
3. The scale and zero point are observer/QNN encoding results, not constants in the DeltaNet QHPI kernel. The wrapper requests `MinMaxObserver` and ties recurrent-state encodings with `share_state_qparams=True`; after conversion, the cache IO alignment pass also aligns the 36 cache outputs to the matching inputs.
4. The custom core annotation still emits repeated `Could not resolve quant range for node: deltanet_core_*` messages. A core node has two outputs, so the warning can be caused by asking for the range on the tuple-producing node whose output qspec is intentionally `None`; it does not by itself prove that both getitem outputs lack encodings. The two getitem outputs still need to be checked directly.
5. The deployed context contains layer-specific recurrent-state encodings. In the inspected UInt16 context, the layer-10 state encoding is approximately `scale=1.3998e-5`, `offset=-30015`, giving a representable range of about `[-0.420, 0.497]`; layer 13 is similarly narrow at about `[-0.388, 0.563]`. These ranges are a concrete saturation risk if the real recurrent state exceeds them. The current artifacts do not yet contain a direct state min/max versus this range, so this must be measured before declaring saturation the sole cause.

## Diagnosis

The evidence points to a combination of three effects, in this order:

1. **GDN recurrent/state path:** the layer-10 GDN norm and its following `out_proj` are the first catastrophic local errors; the FP16-core A/B and device PPL both show that this path matters substantially.
2. **Layer-specific projection error:** later `in_proj_qkv_z` nodes fall from good early-layer SQNR to roughly 4–6 dB and about 1 dB in the deepest groups. These errors feed the recurrence and accumulate across layers.
3. **Calibration coverage:** the original export calibrated on only one image conversation. This is too narrow for a recurrent state whose range changes by layer and token. However, two image samples produced the same hidden metrics, so calibration count alone is not the complete explanation.

The `lm_head` is not where the main error starts. The whole-vocabulary logits SQNR was therefore a poor primary diagnostic; the final hidden state and per-layer residual/GDN nodes show where the damage enters.

## Next isolation experiment

The next useful measurement is not another whole-vocabulary SQNR. Capture, for every GDN layer, the FP32 recurrent-state input/output and the QDQ recurrent-state input/output, then report:

- min, max, RMS, and saturation percentage against that layer's scale/zero point;
- SQNR before the core, after the core attention output, and after the state output;
- the same run with the layer-10 `in_proj_qkv_z`, `in_proj_a`, and `out_proj` temporarily kept at 16-bit weights.

That will distinguish state-range saturation from projection quantization. The current evidence says both should be tested, with layer 10 first.

---

<a id="record-05"></a>

## 记录 05：qwen3_5-0_8b-layer-analysis / QUANTIZATION_DIAGNOSIS_REPORT.md

原始记录位置：`llama-output/qwen3_5-0_8b-layer-analysis/QUANTIZATION_DIAGNOSIS_REPORT.md`。

# Qwen3.5-0.8B 量化误差定位报告

更新时间：2026-10-13。目标是先判断误差是否来自 PTQ、GDN recurrent state 或 KV cache，再决定是否引入 GPTQ。当前对照运行采用 SM8750、QNN SDK 2.49.40.260810、KV 模式、上下文 1024、W8A16 decoder、GDN VTCM、UInt16 state；参考模型是本地 Qwen3.5-0.8B FP32。

## 最新结论（2026-10-13）

静态 PyTorch FP32 与 HF 使用同一转换权重，在 128/1024 token 回归中逐 token 对齐；全 FP16 PTE 在 SM8750 上的 PPL 为 16.8295，而 HF 为 16.8024。模型结构、权重转换、RoPE、GDN 数学、cache 逻辑和 PPL driver 已排除为主因。

整型 PTQ 是当前 PPL 差距的主因。W8A16 + FP16 GDN/state 的 PPL 为 51.6529，W16A16 + FP16 GDN/state 为 37.2138，只有全 FP16 回到 16.8295。提高权重位宽不能消除 activation/scale 传播误差。GDN/state 量化是重要放大器：UInt16 state 到 FP16 state 把 W8 PPL 从 123.7376 降到 51.6529；WikiTrain 校准进一步得到 37.8642，但仍远高于全 FP16。

因此先继续做 activation PTQ、state/cache 的直接张量捕获和饱和率闭环；GPTQ 只能优化 Linear 权重，放在这些检查完成之后。

## 结论摘要

1. 误差首先在 GDN 的 recurrent/state 路径暴露，layer 1 已出现负 SQNR，layer 10 GDN norm 为 -13.80 dB，随后 layer 10 `out_proj` 为 -15.81 dB，最终 hidden state 为 -1.63 dB。
2. 把 GDN/state 路径改成 FP16 后，设备端 WikiText 子集 PPL 从 123.74 降到 51.65；这说明 state/core 的量化是主要原因之一，但 FP16 实验同时改变了 core 输入和累加路径，不能把收益全部归因给 state buffer。
3. 校准数据确实影响结果。原始单图校准的 full PPL 为 123.74；短文本/短图 decode-like 校准为 77.16；8 条 WikiTrain 多模态校准为 79.03。校准范围不足是第二个确定因素。
4. 目前还没有直接从 HTP 运行时读取每一步 state 的 min/max；本报告先给出 QNN 实际 state 量化范围和 CPU PT2E QDQ 的 FP32/QDQ 对照。直接诊断导出已做过一次 1024 长度尝试，但 PT2E observer 插入和 FX 图重编译耗时 2201 秒后仍未进入中间张量捕获；256 长度尝试也在同一阶段停止。因此下面明确区分“已实测”和“待补”，不把推断写成饱和率。
5. GPTQ 暂不作为第一步。当前 QNN recipe 是 activation-aware W8A16，GDN state/cache 还有独立 UInt16 编码；TorchAO GPTQ 是 Linear 权重 Int4 weight-only，直接套到现有 QNN graph 不能定位 state/cache 问题，应该在 state/cache 误差确认后只对投影和 FFN 做受控 A/B。

## 1. FP32 min/max

已有 CPU PT2E hidden-state capture 的参考输出在 `hidden_stage_metrics.json`（历史产物已清理）。目前确认的最早大误差点如下：

| 节点 | FP32/QDQ 对照 SQNR | FP32 RMS | QDQ RMS | max abs error |
|---|---:|---:|---:|---:|
| layer 0 GDN norm `to_16` | 12.11 dB | — | — | — |
| layer 1 GDN norm `to_35` | -4.30 dB | — | — | — |
| layer 10 GDN norm `to_192` | -13.80 dB | 0.0570 | 0.2989 | 3.3138 |
| layer 10 GDN `out_proj` `conv2d_82` | -15.81 dB | 0.0223 | 0.1499 | 0.9781 |
| final hidden `mul_423` | -1.63 dB | — | — | — |

直接 state/output 的逐层 min/max 将由 `run_gdn_diagnostics.py`（历史产物已清理） 生成；这部分必须从同一个 FP32/QDQ forward 对中读取，不能用 logits 反推。

## 2. 量化范围

QNN graph 中 recurrent state 输出是 UInt16，编码形式为 `x ~= (q + offset) * scale`，所以可表示范围为：

`[(offset) * scale, (65535 + offset) * scale]`。

各校准导出的逐层范围已经保存到 `state_quant_ranges.json`（历史产物已清理）。典型范围如下：

| layer | 原始单图校准 | WikiTrain 8 样本 | 短样本 decode-like |
|---:|---:|---:|---:|
| 0 | [-3.845, 4.105] | [-4.684, 5.024] | [-4.026, 4.063] |
| 4 | [-6.854, 11.518] | [-6.981, 11.097] | [-6.902, 11.515] |
| 10 | [-0.281, 0.529] | [-0.324, 0.548] | [-0.389, 0.488] |
| 13 | [-0.311, 0.635] | [-0.417, 0.731] | [-0.292, 0.727] |
| 22 | [-0.491, 0.974] | [-0.580, 1.420] | [-0.440, 2.286] |

layer 10、13 的 state 编码范围窄，尤其需要检查真实 state 是否越界；只看 scale/zero point 不能直接断言已经饱和。

## 3. 饱和比例

当前已有 artifact 只保存了 QNN scale/offset，没有保存 HTP 每一步的原始 state。因此“饱和比例”不能用全词表 logits SQNR 或最终 PPL 代替，也不能把零初始化的 state input 当成运行时 state。

本次诊断的定义是：

`below = mean(fp32 < quant_range_min)`，`above = mean(fp32 > quant_range_max)`，`saturation = below + above`。

统计对象分为：

- GDN recurrent state output：校准序列结束时的 `[1,16,128,128]`；
- GDN recurrent state input：进入下一步 core 前的 state；
- causal-conv cache：`[1,6144,4]`；
- full-attention KV cache：`[1,2,1,256]`。

逐层结果需要一次能完成 PT2E 中间捕获的导出后写入 `gdn_state_attention_metrics.json`。如果 saturation 接近 0 但 SQNR 仍低，原因更可能是量化步长过大、core 内部计算/累加精度或投影误差；如果在 layer 10/13 出现明显越界，则先扩大校准覆盖或单独提高 state bit width。

## 4. state 输入 SQNR

当前 CPU PT2E 报告已经证明 GDN 路径早期发生误差，但旧脚本没有成功捕获 custom op 的 tuple 输入/输出；它只捕获了 norm、out_proj、残差和 final hidden。因此目前不能把 `state input SQNR` 写成一个假数字。

诊断脚本会对每个 `deltanet_core` 的 state 参数做：

- FP32 core state input 与 QDQ dequantized core state input 对照；
- min/max、RMS、量化范围和 saturation；
- SQNR、cosine、max absolute error。

如果 state input SQNR 从 layer 0 开始已经很低，先修 cache IO/qparams；如果 input 尚可而 output 突然下降，重点检查 core 内部 state update 和输出重编码。

## 5. state 输出 SQNR

已有结果显示 layer 10 GDN norm 和其后 `out_proj` 是最差局部位置，但这是 state 输出经过 reshape/norm/投影后的间接证据。诊断脚本会直接比较每层 core 返回的 recurrent state output，形状 `[1,16,128,128]`，并把它与实际 QNN UInt16 范围关联。

FP16 state A/B 的设备结果：

| 配置 | WikiText 子集 PPL | mean NLL |
|---|---:|---:|
| W8A16 + UInt16 GDN/state + VTCM | 123.7376 | 4.8182 |
| W8A16 + FP16 GDN/state + DDR | 51.6529 | 3.9445 |

这已经把 state/core 路径列为第一优先级，但还需要直接 state output SQNR 来区分“state buffer 饱和”和“core 内部量化误差”。

## 6. attention output SQNR

attention output 这里定义为 GDN core 的第 0 个返回值，即形状 `[1, seq, 16, 128]` 的 core attention output；同时保留 attention `out_proj` output 作为传播检查。旧 CPU QDQ 结果中，layer 10 `out_proj` 为 -15.81 dB，是目前最差的 attention-path 节点；layer 10 GDN norm 为 -13.80 dB，说明问题在 out_proj 之前已经存在。

完整逐层 core attention output SQNR 会在诊断文件中生成。判断方式：

- core attention output 先掉：投影、q/k/v/z 量化或 core 输入量化优先；
- core attention output 尚可、state output 掉：state 更新/重编码优先；
- 两者都尚可、out_proj 掉：out_proj 权重/激活量化优先。

## 7. KV cache 检查

KV cache 需要单独看两类：

| cache | 形状 | 关联路径 |
|---|---|---|
| GDN causal-conv cache | `[1,6144,4]` | causal conv 输入/输出 |
| GDN recurrent state | `[1,16,128,128]` | DeltaNet core 输入/输出 |
| full-attention K/V | `[1,2,1,256]` | full attention 层的 KV IO |

KV 模式运行时是 AR-1 逐 token decode，状态会跨 token 回写；它不是单独的 AR-32 prefill runtime graph。现有 cache IO 检查通过 6 项测试，PTE 的 31 个 state/cache 输入和 31 个输出保持 UInt16 编码，18 个 FP16 recurrent-state 输入/输出保持 HALF；这说明 cache 的 dtype/IO 对齐机制本身已覆盖，但不等价于数值范围没有饱和。短样本 calibration 的 full PPL 从原始 123.74 降到 77.16，且第 33–128 token 区间 PPL 约 32.9，说明 cache/state 累积和校准覆盖对长 decode 有明显影响。第一个 token 的 PPL 仍很差，说明初始输入/首步量化也要保留检查。

## 8. 是否是量化问题

当前证据支持“主要是量化路径问题”，但还没有证据支持“只有一个 scale 写死”或“只有饱和导致”。QNN context 中已经有逐层 state scale/offset，且 cache IO 会对齐 qparams；问题更可能是：

1. recurrent state 的分布随层和 token 漂移，单图校准覆盖不足；
2. layer 10/13 的 state 范围较窄，导致量化误差或饱和；
3. `in_proj_qkv_z`、`in_proj_a/b` 和 `out_proj` 的 W8A16 误差进入 recurrence 后累积；
4. cache 输入、core 输出和下一步输入的编码是否完全一致，需要直接 tensor 对照确认。

## 9. GPTQ 的进入条件

只有在上面六项数据完成，并确认 state/cache 没有独立的编码错误后，才做 GPTQ A/B：

1. 固定 state/cache 为当前最佳 UInt16 或 FP16；
2. 只替换 `in_proj_qkv_z`、`in_proj_a/b`、`out_proj` 和 FFN Linear；
3. 使用同一批校准 token，比较权重重构误差、attention output SQNR、state output SQNR 和 PPL；
4. 先做 layer 10、13 的局部 GPTQ，再决定是否全模型应用。

TorchAO 的 `Int4WeightOnlyGPTQQuantizer` 是 Linear 权重-only 方案；当前 QNN graph 主要是 Conv2d/W8A16 和独立 GDN cache 编码，不能把 GPTQ 的改善直接解释成 state 修复。

## 10. 本次直接诊断执行记录

直接诊断脚本是 `run_gdn_diagnostics.py`（历史产物已清理）。它已经实现了对 `deltanet_core` 的 state input、attention output、state output，以及 `[1,6144,4]` causal-conv cache 的节点选择、QDQ 参数解析、min/max、saturation 和 SQNR 计算。执行结果暴露的是导出器瓶颈：

- 1024 长度：`TextDecoder::quantize` 在 `prepare_pt2e`/FX observer 插入阶段运行 2201.24 秒，停在 `get_new_attr_name` 的重复属性扫描；没有生成中间捕获文件。
- 256 长度：同一阶段仍然需要数分钟以上，停止时没有生成中间捕获文件。
- 这不是设备运行失败，也不是 GDN kernel 报错；是当前 W8A16 PT2E 图准备阶段的时间开销。已有 CPU QDQ hidden-state 结果和设备 PPL 仍然有效。

因此本轮先不进入 GPTQ，也不报告虚假的 state saturation 数字。后续若要拿到逐层 state min/max，建议先把诊断导出改成只构造 text decoder、关闭视觉分支和完整 QDQ recipe，或复用已完成的 QDQ GraphModule，避免再次触发整套 multimodal PT2E 准备。

## 11. 评测语料校准复核记录

为检查“校准分布不足”是否可以单独解释 PPL 差距，构造了 8 条固定 WikiText 语料切片，每条约 256 token，并加入本地图片与显式 image placeholder。纯文本样本被多模态数据集构建器拒绝；补齐多模态格式后，量化阶段完成，但 decoder QNN 编译在 `tensor_generator.h:89: Lookup in 1 tensor types failed` 处结束并段错误，没有生成可运行的 decoder PTE。该失败发生在编译器类型表/资源阶段，不能计入 PPL 对照。

可用的已完成校准 A/B 仍是 WikiTrain 8 样本：W8A16 + FP16 state 为 37.8642，W16A16 + FP16 state 为 37.2138。两者都显著高于全 FP16 的 16.8295，故“只增加校准样本”不是当前唯一修复路径；下一步应先让 QNN graph 能支持敏感 activation 的浮点或更合适的 scale 传递，再对 state input/output 做直接饱和率捕获。

另外尝试了 `--use_fp16 --gdn_kernel ddr --gdn_state_dtype fp16`，不经过 PTQ 仍在 decoder QNN 编译的同一 `tensor_generator.h:89` 路径段错误；已完成的 16.8295 基线使用的是 PyTorch GDN 分解。故目前不能把“全 FP16 + DDR fused GDN”作为正确性对照，fused GDN 的浮点编译路径需要单独修复后再比较。

## 现有证据与脚本

- CPU hidden state：[`HIDDEN_STATE_ERROR_REPORT.md`](README_QWEN35_TEST_RECORDS.md#record-04)、`hidden_stage_metrics.json`（历史产物已清理）
- QNN state qparams：`state_quant_ranges.json`（历史产物已清理）
- 原始 PPL 对照：[`../qwen3_5-0_8b-ppl/REPORT.md`](README_QWEN35_TEST_RECORDS.md#record-36)
- decode-like 校准 PPL：`../qwen3_5-0_8b-decodecalib_full_ppl.json`（历史产物已清理）、`../qwen3_5-0_8b-decodecalib128_ppl.json`（历史产物已清理）
- 诊断脚本：`run_gdn_diagnostics.py`（历史产物已清理）

## 12. 单层 128-token 对照

为避免整套 24 层模型掩盖误差来源，单独构造了 `Qwen35DecoderLayer(0)`，只输入 128 个 token，使用真实 Qwen3.5-0.8B 权重和 8 个 WikiText 128-token 校准批次。比较的是 FP32 exported graph 与 Qualcomm PT2E QDQ graph，未经过设备编译，因此这一步只回答“量化图的数值误差从哪里开始”。

生产路径的 fused projection、depthwise Conv2d、FFN Conv2d 和 fused DeltaNet core 结果见 [`one_layer_128_fused_w4/REPORT.md`](README_QWEN35_TEST_RECORDS.md#record-28) 和 [`one_layer_128_fused_w8/REPORT.md`](README_QWEN35_TEST_RECORDS.md#record-33)。主要结果：

| 配置 | layer output SQNR | recurrent state SQNR | causal-conv state SQNR |
|---|---:|---:|---:|
| W4A16 + fused GDN | -2.19 dB | -0.18 dB | 4.97 dB |
| W8A16 + fused GDN | 27.00 dB | 37.36 dB | 45.69 dB |
| W8A16 + PyTorch GDN decomposition | 17.28 dB | 27.26 dB | 30.83 dB |

W4A16 的误差排序如下：

1. GDN q/k L2 normalization 的 `rsqrt_1`、`rsqrt_2` 出现 `inf`。对应的 `sum(x*x)+eps` 在 uint16 QDQ 后可能量化为 0，`eps=1e-6` 小于该节点的量化步长；这是激活量化范围与非线性 `rsqrt` 组合造成的数值问题。
2. 有限值中，FFN 门控乘法约 -3.81 dB，`feed_forward.w2_conv` 约 -3.36 dB，`w3_conv` 约 -2.29 dB，`w1_conv` 约 -1.49 dB。它们直接决定该层最终残差输出约 -2.19 dB。
3. GDN 输入侧的 fused `in_proj_qkv_z` 约 4.35 dB，causal depthwise conv 约 8.83 dB；fused core 返回的 recurrent state 约 -0.18 dB。说明 state 的坏结果已经由输入投影/卷积误差和 state 编码共同形成。

因此当前单层证据把优先级定为：先处理 `rsqrt` 前的量化和 GDN state 编码，再处理 W4 FFN 的 `w1/w3/w2`，最后再看 `out_proj`。`reshape`、`transpose`、`to` 等低 SQNR 记录多数只是携带上游误差，没有独立量化参数，不能作为根因。

---

<a id="record-06"></a>

## 记录 06：all_linear_w4_128 / REPORT.md

原始记录位置：`llama-output/qwen3_5-0_8b-layer-analysis/all_linear_w4_128/REPORT.md`。

# 所有 Linear 回退 W4：128-token 误差与性能

日期：2026-10-13；Qwen3.5-0.8B layer 0；SM8750/v79；QNN 2.49.40.260810。

## 1. 实际回退范围

- `in_proj_qkv_z` 恢复为原合并布局（8192×1024），取消 K/z W8 拆分；a/b 两个输入投影均 W4。
- `out_proj` 与 FFN w1/w2/w3 均 W4；每块 32，实际 code 范围 [-7,7]。本轮使用原始权重的 PTQ，未应用 GPTQ patch。
- causal depthwise conv 继续 W8；激活、conv cache、recurrent state 均 UINT16。因此这里的“全 W4”精确定义为所有 Linear 权重 W4。
- GDN 默认恢复快速 VTCM / int32 缩位归约；完整精度 FP32 归约保留为可选实验分支，默认宏值已从 1 恢复为 0。
- L2 非有限值修复、custom op 各输出独立使用量化参数的修复继续保留。

Recipe 使用 `Qwen3_5_0_8BQuantRecipe(all_linear_w4=True)`，与原混合精度选项显式区分；常规导出使用何种 recipe 仍由调用方指定。

## 2. 性能：统一为纯 linear-attention 的三个输出

输入为已经过前置 RMSNorm 的 hidden、conv cache、recurrent state。输出仅 attention/out_proj、conv cache、recurrent state。图包含输入投影、causal conv、Q/K L2、128 步 GDN、RMSNorm/z 门控和 out_proj。FFN、残差、前置 input RMSNorm 和精度检查点均从计时图中删除。

三个性能版本使用完全相同的 8 段 WikiText 输入（每段 128 token），每个输入先 warmup 5 次，再计时 100 次，交替跑三轮，profile_level=0。一次只导出一个 PTE，三种版本依次运行。

| 版本 | 三轮平均 / ms | 总平均 / ms |
|---|---|---:|
| 所有 Linear W4 + 快速 VTCM | 3.07125, 3.08782, 3.07673 | 3.07860 |
| K/z W8 + V W4 GPTQ + 快速 VTCM | 3.11004, 3.10979, 3.10683 | 3.10889 |
| K/z W8 + V W4 GPTQ + 完整精度 FP32 归约 | 4.66850, 4.66730, 4.66538 | 4.66706 |

当前同图下，FP32 归约相对快速混合精度版增加约 50.1% 的耗时；把 Linear 全改为 W4，仅比快速混合精度版减少约 1.0% 的时间。此次主要速度代价来自归约方式。

### 原约 2.7 ms 图的回归核对

使用设备上原始 VTCM PTE、原输入和当前回退库，warmup 5、计时 100 次，实测 **2.68531 ms**。三个输出全部与当时保存的基线逐字节一致，最大差为 0，确认旧快速 kernel 的行为已恢复。

当前重新校准的新 W4 图为 3.07860 ms，尚未达到 2.73 ms。历史图和新图采用不同输入/校准数据、L2 计算图及 W4 量化配置，不能将两者的全部差异归因于权重位宽，也不能用历史图的 2.69 ms 代替新图的实际结果。

上次给出的 4.73/6.33 ms 包含 FFN 和 22 个检查点，性能口径与原 2.7 ms 不一致。本报告使用三个原始输出的图重新比较。

## 3. 精度测试设置

校准在 CUDA 上完成，WikiText 8×128 token。评估使用相同 8 段文本的后续 128 token，每个窗口独立零 cache；FP32 和 PTQ 使用相同评估输入。FP32 权重保持原始本地模型，没有替换参考模型。

完整层检查点图用于量化误差分析，另导出三输出的纯 attention PTE 计时。剪裁后的 CPU QDQ 三个输出与完整检查点图对应值逐元素完全一致（最大差 0）。另在设备执行三输出 PTE 检查实际精度，结果与检查点图接近。

| 三输出性能 PTE 的实测 SQNR | dB |
|---|---:|
| attention/out_proj 输出 | 14.459 |
| conv cache 输出 | 24.756 |
| 最终 recurrent state | 20.897 |

## 4. FP32 min/max、量化范围、饱和比例

范围由同一评估集 FP32 检查点与实际 CUDA 校准参数计算；饱和比例指落在可表示实数范围外的比例。

| 张量 | FP32 min/max | 量化实数范围 | 范围外比例 |
|---|---|---|---:|
| core_input_query | [-0.086693369, 0.087773368] | [-0.085455813, 0.08781688] | 4.7683716e-05% |
| core_input_key | [-0.99902123, 0.98502505] | [-0.99690216, 0.9873646] | 9.5367432e-05% |
| core_input_value | [-0.27846456, 5.5828161] | [-0.27845217, 5.5479727] | 0.18630028% |
| core_input_state | [0, 0] | [-3.2021129, 2.574118] | 0% |
| gdn_core_output | [-0.26395759, 0.19815163] | [-0.25338209, 0.17697673] | 0.00014305115% |
| recurrent_state_output | [-3.1699224, 2.5329218] | [-3.2021129, 2.574118] | 0% |
| rms_weighted | [-9.2018738, 9.3015947] | [-9.0650895, 9.3019112] | 0.00014305115% |
| z_input | [-9.1924429, 7.1938391] | [-7.5258259, 7.1945143] | 4.7683716e-05% |
| gdn_norm_z_gate | [-2.5572574, 2.5596676] | [-2.5615983, 2.5176533] | 0.00028610229% |

## 5. state 输入、state 输出、attention 输出 SQNR

初始 state 输入全零，两条路径完全相同（误差为 0）；这不是非零跨窗口 state 的精度验证。最终 state 取每个 128-token 窗口的最后一步。attention output 取 out_proj 后输出，使用 hidden state 维度，没有使用全词表 logits。所有 dB 均为先累计信号/噪声功率再计算，非逐样本 dB 的算术平均。

| 阶段 | 全 W4 CPU QDQ / dB | 全 W4 快速设备 / dB | 前一轮 K/z W8 + V GPTQ + FP32 归约设备 / dB |
|---|---:|---:|---:|
| state 输入（全零） | 零误差 | 零误差 | 零误差 |
| 最终 state 输出 | 21.499 | 20.896 | 31.152 |
| attention/out_proj 输出 | 15.692 | 14.454 | 25.756 |
| conv cache 输出 | 25.221 | 24.757 | 27.414 |
| Q 输入 | 22.585 | 22.136 | 22.145 |
| K 输入 | 21.062 | 20.654 | 39.740 |
| V 输入 | 32.195 | 31.382 | 34.534 |
| z 输入 | 22.354 | 21.925 | 43.040 |
| GDN 输出 | 30.652 | 29.924 | 32.789 |
| RMSNorm × gamma | 33.552 | 30.093 | 35.521 |
| norm/z 门控 | 23.034 | 20.968 | 29.609 |
| FFN 输出 | 10.873 | 9.600 | 17.038 |
| 完整层 hidden state | 13.948 | 12.735 | 21.002 |

前一轮和本轮同时改变了 K/z、out_proj、FFN down 的精度、投影布局、V GPTQ 和归约方式；该表用于展示方案总变化，不能把完整层误差差值全归因于 K/z。CUDA 与前一轮 CPU 校准也有小的数值舍入差，但文本窗口与 FP32 原模型一致。

全 W4 后，K 输入与最终 state 精度下降明显。norm/z 门控回到约 20.97 dB。out_proj 和 FFN down 也降为 W4，因此 attention 输出与完整层 hidden state 的下降进一步增加。

## 6. 图与 kernel 验证

- 默认宏 `DELTANET_U16_FULL_PRECISION_REDUCTION=0`；x86、Android arm64、Hexagon v79 三个包增量构建成功，运行时 `_HTP.so` 同步更新。
- 原历史 VTCM PTE 的三个结果与原基线逐字节一致。
- 新三输出 PTE 与完整检查点图的 CPU QDQ 相同输入输出严格一致；重新在设备跑三个输出得到 attention 14.459 dB、state 20.897 dB，与检查点图的 14.454/20.896 dB 接近。
- 性能图只有一个 QNN delegate，数学操作无 CPU fallback；图中已删除前置 input RMSNorm、FFN 及导出时用于类型检查的残留断言依赖。
- 所有评估窗口设备值均为有限值。
- `test_gdn_split_key_projection.py` 共 7 个测试通过，覆盖原两种混合布局，以及全 W4 选项的 a/b/out_proj/FFN 与 depthwise conv 实际配置。

## 7. 复现与文件

完整重新校准、检查点精度与性能导出按独立进程串行执行：

```bash
source /home/hunter/anaconda3/etc/profile.d/conda.sh
conda activate et_npu
source env.sh
export CUDA_VISIBLE_DEVICES=0
python llama-output/qwen3_5-0_8b-layer-analysis/run_all_linear_w4_128.py
```

已保存 CPU/calibration 结果时，可从纯 attention 性能导出开始：

```bash
python llama-output/qwen3_5-0_8b-layer-analysis/run_all_linear_w4_128.py --phase perf_w4
```

- `all_w4_linear_attention_128.pte`：三输入、三输出的纯 linear-attention，约 5.94 MiB。
- `merged_linear_attention_ffn_128.pte`：包含 FFN 的完整层检查点图，用于精度诊断。
- `performance.json` / `timing_*.log`：三轮、八输入的统一性能数据。
- `summary.json` / `metrics.csv` / `outputs/`：完整层及逐 token 精度数据。
- `three_output_accuracy/`：真实三输出性能 PTE 的精度复核。
- `all_w4_attention_equivalence.json`：剪裁图和原图对应输出的严格一致性。
- `cached_vtcm_retest.log`：原历史图 2.68531 ms 与三个输出逐字节一致的记录。
- `fp32_reduction_op_package/`：回退前完整精度实现的快照，用于性能对照。
- `artifact_sha256.json`：当前纯 attention PTE 与快速 DSP 库的哈希。

本轮仅验证 layer 0 / 128-token prefill，未执行完整模型 PPL 或跨窗口非零 cache decode。当前新图 3.08 ms 的约束差距仍存在；历史 2.69 ms 的原图结果单独保留。

## 8. attention/out_proj 的进一步拆分

见 [out_proj 误差拆分报告](README_QWEN35_TEST_RECORDS.md#record-08)。保持干净 FP32 输入、仅替换设备实际 LPBQ W4 权重，输出为 16.601 dB；使用实际设备门控输入、权重恢复 FP32，输出为 18.582 dB。二者叠加的 LPBQ 回放为 14.45423 dB，实际设备为 14.45438 dB，同输入回放与设备符合度为 70.040 dB。该拆分定位到 W4 权重与上游门控输入两个主要来源，并解释了 QDQ 和设备之间的 block scale 编码差异。

## 9. out_proj 的 W4 GPTQ 实验

见 [GPTQ 设备验证报告](README_QWEN35_TEST_RECORDS.md#record-07)。仅在 QNN 实际有效 LPBQ 网格上优化 out_proj 的 W4 code，设备 attention 输出从 14.454 提升到 17.412 dB，完整层 hidden 从 12.735 提升到 15.064 dB。上游 14 个设备检查点逐元素不变。匹配计时原 W4 为 3.10189 ms、GPTQ 为 3.08478 ms，两个 PTE 大小相同；当前没有观察到性能代价。未执行完整模型 PPL。

## 10. 完整 24 层、128-token PTE 的逐层误差

见 [完整 decoder 逐层报告](README_QWEN35_TEST_RECORDS.md#record-09)。完整 PTE 包含 24 层、最终 norm、LM head 和缓存；上下文 1024、prefill 128，18 个 GDN out_proj 应用 W4 GPTQ。第 0 层设备输出 15.066 dB，第 7 层 full attention 从输入 10.105 降到输出 4.939 dB；第 23 层输出 6.378 dB，最终 norm 后 5.701 dB。相邻层边界逐元素一致。该结果表明单层 GPTQ 收益尚未消除整条设备路径的累计误差。

## 11. K/z W8 完整模型对照

见 [K/z W8 的完整设备误差与匹配性能](README_QWEN35_TEST_RECORDS.md#record-10)。第 0 层 K 输入 39.741 dB、z 输入 43.037 dB、state 输出 28.557 dB，但 attention 输出仅改善至 17.616 dB；最终 norm 为 5.414 dB。同条件纯 attention 耗时 W4 3.0968 ms、K/z W8 3.0981 ms。局部误差改善尚未修复完整设备模型。

---

<a id="record-07"></a>

## 记录 07：gptq_out_proj / REPORT.md

原始记录位置：`llama-output/qwen3_5-0_8b-layer-analysis/all_linear_w4_128/gptq_out_proj/REPORT.md`。

# out_proj 的 W4 GPTQ：128-token 设备验证

日期：2026-10-13；Qwen3.5-0.8B layer 0；SM8750/v79；QNN 2.49.40.260810。

## 1. 本次实际修改

仅优化 `layers.0.attention.out_proj` 的 1024×2048 权重。仍为 **W4 / group=32 / code [-7,7] / zero point=0**，激活、conv cache 和 recurrent state 仍 UINT16，GDN 使用同一个快速 VTCM 库。

没有修改 qkv_z、a/b、depthwise conv、FFN 权重或任何激活 scale/zero point。没有拆分投影矩阵、修改算子实现或加入 FP32 归约路径。

GPTQ 使用原本的 QNN **有效 LPBQ block scale 网格**，通过输入 Hessian 对权重逐列量化误差进行补偿。没有重新搜索 scale，没有启用 act-order，维持原有输入列顺序及 block 布局。

Torch 保存图的 out_proj scale buffer 改为 QNN 实际有效 scale，使 CPU QDQ 与设备使用同一网格。它由已有校准参数计算；再次按 QNN 规则编码后逐元素保持不变，因此设备实际使用的基础 scale 和 block scale 系数不改变。这项 buffer 改动不是重新校准出更宽的范围。

使用已有通用实现 gptq_fixed_grid（历史产物已清理）。算法参考为 [GPTQ 原始实现](https://github.com/IST-DASLab/gptq/blob/main/gptq.py)。

## 2. 校准与评估

- 沿用原 WikiText 的 8 段 256-token 文本，每段前 128 token 用作校准，后 128 token 用作评估。
- 总校准量 1024 token；Hessian 输入为 `[1024,2048]`。
- 在 **CUDA** 上运行原始 FP32 layer 0，采集 out_proj 的实际输入，再使用已有输入 A16 编码构建 GPTQ 校准输入；GPTQ 同样运行 CUDA。
- `damping=0.01`，处理 block=128，量化 group=32；没有用评估窗口调整这些参数。
- 参考仍为原 safetensors 模型的 FP32 输出；未把 GPTQ 权重放入参考模型。
- 每个 128-token 窗口使用独立的零 conv/recurrent state。这次验证不覆盖非零跨窗口 cache 或完整模型 decode。

约 **23.931%** 的 out_proj 权重 code 改变。实际 GPU 权重优化约 **0.202 s**，这个时间不包含采集校准输入、导出或设备执行。

## 3. 先隔离权重误差

参考为原 FP32 out_proj 输出，表中预测都使用设备实际 LPBQ grid 和已有输出 A16 编码。

| 权重处理方式 | FP32 门控输入 / dB | 原设备门控输入 / dB |
|---|---:|---:|
| 原 W4 code + 实际 LPBQ scale | 16.601 | 14.454 |
| 在实际 LPBQ grid 上重新最近舍入 | 16.631 | 14.419 |
| 在同一实际 LPBQ grid 上做 GPTQ | **23.884** | **17.412** |

直接在有效 grid 上重做最近舍入，不能解决当前输入上的投影误差。GPTQ 的主要收益来自考虑输入相关性后对 code 的补偿，而非改变位宽或 scale 网格。

干净输入下，GPTQ 把 out_proj 输出提升 **7.282 dB**。原设备门控输入下，预测为 **17.41193 dB**，实际设备检查点为 **17.41210 dB**。

## 4. 真实设备：完整层变化

完整层包含 FFN、残差和 22 个精度检查点。逐窗口累计信号/误差功率后计算 SQNR；没有使用全词表 logits。

| 检查点 | 原全部 Linear W4 / dB | 仅 out_proj W4 GPTQ / dB | 变化 / dB |
|---|---:|---:|---:|
| GDN core 输出 | 29.924 | 29.924 | 0 |
| 最终 recurrent state | 20.896 | 20.896 | 0 |
| norm/z 门控输出 | 20.968 | 20.968 | 0 |
| attention/out_proj | 14.454 | **17.412** | **+2.958** |
| attention 残差 | 14.181 | 17.139 | +2.958 |
| FFN 前置 norm | 12.555 | 15.659 | +3.104 |
| FFN 输出 | 9.600 | 11.507 | +1.907 |
| 完整层 hidden state | 12.735 | **15.064** | **+2.329** |

attention 输出误差功率下降 **49.39%**，完整层 hidden state 误差功率下降 **41.50%**。

所有上游 14 个设备检查点在 8 个窗口中逐元素完全一致，最大差为 0：包括 Q/K/V、decay/beta、GDN、RMSNorm、z、门控、conv cache 和最终 recurrent state。FFN 的权重没有修改；其输入改善后，FFN 和完整层输出也随之改善。

## 5. 128 个 token 的变化

128 个位置分别累计 8 个评估窗口后，attention 输出 SQNR **每个位置都提升**，提升区间为 **1.598–4.114 dB**。

设备逐 token 误差变化（历史产物已清理）

完整原始数据在 `token_comparison.csv`。图中两条路径使用相同的 FP32 参考和评估输入。

## 6. 同时验证实际计时 PTE 的精度

纯 attention PTE 去掉 FFN、残差、前置输入 RMSNorm 和额外检查点，包含三个输出，使用已归一化的 hidden 输入。

| 三输出 PTE 的设备 SQNR | 原 W4 / dB | GPTQ / dB |
|---|---:|---:|
| attention/out_proj | 14.459 | **17.418** |
| conv cache | 24.756 | 24.756 |
| recurrent state | 20.897 | 20.897 |

与完整检查点图的 17.412 dB 接近。三输出图的输入 RMSNorm 位于调用方，完整层图含输入 RMSNorm 的量化路径，两种图精度的微小差别沿用原测试口径。

## 7. 性能与文件大小

使用同一快速 DSP/Android op package、同一批 8×128-token 输入。每个输入 warmup 5 次、计时 100 次；原 W4 和 GPTQ 两个 PTE 交替跑三轮，profile_level=0。

| 版本 | 第 1 轮 / ms | 第 2 轮 / ms | 第 3 轮 / ms | 总平均 / ms |
|---|---:|---:|---:|---:|
| 原全部 Linear W4 | 3.09034 | 3.08826 | 3.12706 | **3.10189** |
| 仅 out_proj W4 GPTQ | 3.09313 | 3.08548 | 3.07571 | **3.08478** |

GPTQ 均值比此次匹配基线低 0.55%，属于这个测量中的小差异；证据支持 **未观察到耗时增加**，不将其解释为稳定的性能优化。

两个纯 attention PTE 都为 **6,223,824 bytes**，文件大小完全相同。图中只有一个 QNN delegate，没有数学操作 CPU fallback。

历史约 2.73 ms 与当前重新校准图的约 3.08 ms 差距仍存在；本轮 GPTQ 没有解决这一性能差距。

## 8. 校验和剩余问题

保存图仅两个 buffer 改变：out_proj 权重 code `_frozen_param9` 和对应 scale `_scale3`。全部其他 buffer、recipe、激活编码完全相同。有效 scale 再编码严格一致，代码范围检查通过。

GPTQ 的改进已经通过真实 PTE 验证，但 attention 输出仍为 17.41 dB、完整层 hidden 为 15.06 dB，**不能据此声称完整模型 PPL 已恢复**。本轮没有执行全模型 PPL。

当前原设备门控输入经过完全 FP32 out_proj 的对照为 18.582 dB；上游门控误差的影响仍然明显。FFN 输出还只有 11.507 dB，需要继续拆解并处理 FFN 自身的 W4 误差。

本轮结果保存在独立实验目录，原 W4 PTE 和原权重数据保留。常规全模型导出流程尚未自动应用此 patch。

## 9. 复现与产物

在仓库根目录执行，按独立进程依次进行优化、检查点导出/设备评估、三输出 PTE 导出、计时：

```bash
source /home/hunter/anaconda3/etc/profile.d/conda.sh
conda activate et_npu
source env.sh
export CUDA_VISIBLE_DEVICES=0
python llama-output/qwen3_5-0_8b-layer-analysis/optimize_out_proj_gptq.py
python llama-output/qwen3_5-0_8b-layer-analysis/report_out_proj_gptq.py
```

- `out_proj_weight_patch.pt`：GPTQ code、有效 scale、原 code/scale、buffer 名称和校准记录。
- `calibration_inputs.pt`：1024×2048 的 GPTQ 校准输入。
- `qdq.pt2`：只有 out_proj 应用 GPTQ 的完整层 QDQ 图。
- `merged_linear_attention_ffn_128.pte`：包含 FFN 和检查点的设备评估图。
- `gptq_linear_attention_128.pte`：三个输出的纯 attention 性能图。
- `summary.json` / `metrics.csv` / `outputs/`：真实设备和 CPU 的完整层结果。
- `isolated_summary.json` / `isolated_metrics.csv`：RTN/GPTQ 权重隔离实验。
- `comparison.csv` / `token_comparison.csv` / `token_comparison.png`：与原 W4 的比较。
- `three_output_accuracy/`：真实性能 PTE 的三个输出和精度结果。
- `performance.json` / `timing_*.log`：三轮匹配性能、PTE 字节数、PTE/库 SHA256。
- `validation.json`：buffer 范围、激活参数、14 个上游检查点及纯 attention 精度校验。

---

<a id="record-08"></a>

## 记录 08：out_proj_analysis / REPORT.md

原始记录位置：`llama-output/qwen3_5-0_8b-layer-analysis/all_linear_w4_128/out_proj_analysis/REPORT.md`。

# attention/out_proj 误差拆分：全 Linear W4、快速 VTCM

2026-10-13；Qwen3.5-0.8B layer 0；SM8750/v79；128-token prefill。

## 1. 结论

本轮原设备结果：norm/z 门控输出 **20.968 dB**，out_proj 输出 **14.454 dB**。

损失可以拆为两个主要来源：

1. **out_proj 自身的 W4 权重误差**。保持输入完全 FP32，仅替换为设备实际使用的 W4 权重，输出为 **16.601 dB**。
2. **上游门控输入误差经投影后的影响**。使用实际设备门控输入，out_proj 权重恢复 FP32，输出为 **18.582 dB**。门控输入的 20.968 dB 经这个矩阵投影后下降 2.386 dB。

二者叠加，并纳入 QNN 对 block scale 的压缩及输出 A16，预测为 **14.45423 dB**，设备为 **14.45438 dB**。相同输入下预测值与设备值之间为 **70.040 dB**。

这解释了当前 out_proj 的设备结果。单纯修改 VTCM recurrence kernel 或给 out_proj 更宽的激活范围，不能消除已测得的 W4 投影权重误差。当前检查也没有显示 out_proj 的 layout 或量化参数关联错误。

## 2. 测试范围与实际参数

使用上一轮保存的同一个校准 QDQ 图、原始本地 safetensors 权重、8 个评估窗口及设备检查点，未重新校准或导出。每个窗口 128 token，初始 cache/state 为零。校准来自 WikiText 的前 128 token，评估来自后续 128 token。

当前输入投影 qkv_z、a/b、out_proj 和 FFN 均 W4；depthwise conv 为 W8；激活和缓存为 UINT16；recurrence 使用已经回退的快速 VTCM 分支。本轮没有应用 GPTQ、修改位宽或修改 kernel。

| 参数 | 实际值 |
|---|---|
| 权重 | `model.language_model.layers.0.linear_attn.out_proj.weight` |
| 权重形状 | `[1024, 2048]`，无 bias |
| token 输入 / 输出形状 | `[1, 128, 2048]` / `[1, 128, 1024]` |
| 原 FP32 权重 min/max | `[-0.283203125, 0.1953125]` |
| W4 code | `[-7, 7]`，zero point = 0 |
| block | 每个输出通道按输入维分组，每组 32；共 1024×64 个 scale |
| 原 block scale min/max | `[0.0008021763642, 0.04045759141]` |
| 输入激活 scale / zero point | `7.750441000098363e-05` / `33051` |
| 输出激活 scale / zero point | `3.036231100850273e-05` / `46182` |

输入/输出激活参数直接从保存的校准图读取，非人为设定。W4 权重使用对称量化，因此 zero point 为 0。

## 3. FP32 min/max → 量化范围 → 饱和比例

“范围外比例”指未量化张量超出校准实数范围的比例，不是量化 code 恰好位于端点的比例。

| 张量 | 未量化 min/max | 实数量化范围 | 范围外比例 |
|---|---|---|---:|
| 原 FP32 门控输入 | `[-2.5572574, 2.5596676]` | `[-2.5615983, 2.5176533]` | 0.0002861023% |
| 原 FP32 out_proj 输出 | `[-1.2801794, 0.59704369]` | `[-1.4021922, 0.58760180]` | 0.0000953674% |
| 实际设备输入 × 实际 LPBQ 权重，输出量化前 | `[-1.2494684, 0.60355993]` | 同上 | 0.0000953674% |

最后两行各有 1 个元素在 8×128×1024 个输出中越界。门控原输入有 6 个元素在 8×128×2048 个输入中越界。

输入 A16 单独造成的输出误差为 **69.243 dB**；实际 LPBQ 路径的输出 A16 误差分量为 **67.176 dB**。它们远小于 W4 权重及上游输入误差，因此 out_proj 当前的主要问题不是激活严重饱和或范围过宽。

## 4. 控制变量：输入和权重分别替换

除“相同输入核对”外，表中参考均为原 FP32 层的 attention/out_proj 输出。各窗口先累计信号/噪声功率再计算 SQNR，不平均 dB。

| 输入 | 权重 / 输出处理 | out_proj SQNR / dB | 用途 |
|---|---|---:|---|
| 原 FP32 门控输入 | 原 FP32 权重，双精度重算 | 134.637 | 核对取权重、维度和矩阵方向 |
| 原 FP32 门控输入，量化为 A16 | 原 FP32 权重 | 69.243 | 单独测输入 A16 |
| 原 FP32 门控输入 | 原 QDQ 图 W4 权重 | 16.883 | 测普通 W4 舍入误差 |
| 原 FP32 门控输入 | 设备实际 LPBQ W4 权重 | 16.601 | 测实际设备权重总误差 |
| 实际设备门控输入 | 原 FP32 权重 | 18.582 | 测上游输入误差经投影的影响 |
| 实际设备门控输入 | 原 QDQ 图 W4 权重 + 输出 A16 | 14.641 | 尚未模拟 LPBQ scale 压缩 |
| 实际设备门控输入 | 实际 LPBQ W4 权重 + 输出 A16 | 14.45423 | 完整数值回放 |
| 实际设备输出 | 设备 PTE 的检查点 | 14.45438 | 实测结果 |

另一个参考口径：直接把实际设备门控输入传入原 CPU QDQ 图，与设备 out_proj 比较，仅为 **29.216 dB**。加入 LPBQ 的有效 scale 后，同输入 CPU 回放与设备的符合度上升到 **70.040 dB**。

前面的 FP32 输入/权重替换是误差归因实验，未把生产图改为 FP32，不能据此声称优化了性能。

## 5. 新定位：导出时 block scale 还会被量化一次

对应实现是 NodeVisitor.make_qnn_per_block_config（历史产物已清理）。

Torch QDQ 用各 block 独立的浮点 scale，而 QNN 的 LPBQ 编码把同一输出通道的 64 个 scale 压缩到共享基础 scale 加整数系数：

```python
base_scale = original_block_scales.amax(dim=-1) / 16
block_scale_code = round(original_block_scales / base_scale[:, None]).clamp(1, 16)
effective_block_scale = base_scale[:, None] * block_scale_code
effective_weight = weight_code * effective_block_scale
```

这里的 `16` 来自后端当前固定的 4-bit block scale 编码约束，不是写死激活的 scale/zero point。`original_block_scales` 来自权重量化 observer；激活 scale/zero point仍取校准结果。

回放严格使用源码相同的 float32 运算、舍入规则和 clamp；权重 code 使用原图已有整数，未重新量化 code。

| block scale 相对误差 | 数值 |
|---|---:|
| 中位数 | 2.871% |
| P90 | 6.398% |
| P99 | 19.616% |
| 最大 | 33.333% |
| 原 scale / 有效 scale SQNR | 29.113 dB |

纯权重 SQNR 从原 QDQ 的 **19.868 dB** 降为实际 LPBQ 的 **19.330 dB**。保持干净输入时，输出从 16.883 降为 16.601 dB；实际设备输入时，从预测的 14.641 降到 14.454 dB。

因此 LPBQ scale 压缩解释了 CPU/设备剩余差异，但它只造成约 0.19 dB 的当前总输出变化；最大的权重损失已经在普通 W4 舍入阶段发生。不能将整个 14.45 dB 归咎于 scale 压缩。

## 6. 为什么输入 20.97 dB，使用 FP32 out_proj 仍会降到 18.58 dB？

线性投影会对信号和误差的方向施加不同增益。令 `X` 为原门控输入，`E = X_device - X`，即使权重完全 FP32，也有：

```text
reference_output = X @ W.T
input_error_output = E @ W.T
```

实测这个矩阵对参考信号的功率增益为 **0.06417**，对输入误差的功率增益为 **0.11117**，后者相对前者约为 **1.732 倍**。因此 SQNR 下降：

```text
10 * log10(0.11117 / 0.06417) = 2.386 dB
20.968 dB - 2.386 dB = 18.582 dB
```

这不是新增了量化算子，而是原来输入误差经这个矩阵后，相对于输出参考信号变得更明显。2048 项加权求和中的信号抵消、输入各维的相关性都会影响这一比值。只比较全权重张量 SQNR，无法预测真实输入上的投影误差。

## 7. 精确误差分解和交叉项

`W` 为原 FP32 权重，`dW` 为实际 LPBQ 权重减去 `W`，`dX` 为实际设备输入减去 FP32 输入：

```text
Y_device - Y_fp32
 = dX @ W.T
 + X @ dW.T
 + dX @ dW.T
 + output_A16_error
 + device_replay_residual
 + fp32_reference_rounding
```

逐元素验证上述等式，双精度分解与实测设备误差一致。以下各行使用同一原 FP32 输出信号功率 **1749.82904**。

| 误差项 | 误差功率 Σe² | 该项等效 SQNR / dB |
|---|---:|---:|
| 上游输入经 FP32 权重投影 | 24.254782 | 18.582 |
| 实际 LPBQ W4 权重，干净输入 | 38.270062 | 16.601 |
| 输入误差与权重误差相乘 | 0.363004 | 36.831 |
| 输出 A16，包括极少量饱和 | 0.000335 | 67.176 |
| 同输入 LPBQ 回放后设备剩余误差 | 0.000173 | 70.042 |
| FP32 参考的数值舍入 | 约 6.02e-11 | 134.637 |

单独误差项平方功率合计 **62.888356**，全部两两交叉项合计 **-0.146475**，实际总误差功率 **62.741881**。上游输入误差与权重误差的余弦相关系数约 **0.00840**，两者在本评估集接近不相关。

以实际总误差功率作分母，权重项的独立平方功率约为 **61.00%**，输入项约为 **38.66%**；该数字是分量功率比，不是忽略交叉项的严格可加因果百分比。完整交叉项保存在 `physical_cross_terms.csv`。

## 8. 128 个位置是否越来越差？

每个 token 位置累计 8 个窗口的功率后计算：

| token 位置 | 门控输入 | 设备输入 + FP32 权重 | FP32 输入 + 实际 W4 权重 | 实际设备 out_proj |
|---:|---:|---:|---:|---:|
| 1 | 21.811 | 20.674 | 16.686 | 15.197 |
| 32 | 20.431 | 18.122 | 16.667 | 14.352 |
| 64 | 21.813 | 18.890 | 16.236 | 14.311 |
| 96 | 21.613 | 19.586 | 16.811 | 14.922 |
| 128 | 21.039 | 18.262 | 16.342 | 14.173 |

| 区间 | 实际设备 out_proj SQNR / dB |
|---|---:|
| 1–32 | 14.468 |
| 33–64 | 14.466 |
| 65–96 | 14.444 |
| 97–128 | 14.440 |

区间间没有明显随 token 数增加的累积恶化。out_proj 是逐 token 的线性投影，本身也不维护跨 token state；这里不能凭曲线排除上游 recurrence 或跨窗口非零 state 的问题，但当前这一步的主要误差表现为持续存在的投影损失。

逐 token 的 SQNR 与同输入 LPBQ 回放核对（历史产物已清理）

## 9. 是否只有几个通道出问题？

按“干净输入 × 权重误差”的输出通道误差功率排序：

| 最大误差的输出通道数量 | 覆盖的 W4 权重输出误差功率 |
|---:|---:|
| 8 / 1024 | 4.587% |
| 32 / 1024 | 13.455% |
| 128 / 1024 | 34.140% |
| 512 / 1024 | 75.676% |

误差分布较广。误差功率最大的通道依次为 382、46、216、478、6、0、249、631。通道 382 的干净输入 W4 输出仅 3.101 dB，通道 0 为 33.916 dB：各通道参考信号功率差异很大，不能只看单通道 dB 来决定全局问题。

全部 1024 个通道、8 个窗口的分量功率保存在 `physical_channels.csv`。

## 10. 对后续修复的约束

当前证据支持先优化 **out_proj 的 W4 权重选择**，同时继续处理上游门控输入误差。符合性能约束的候选是维持现有 group=32 / W4 结构，用校准输入进行 GPTQ 补偿，并在 **LPBQ 实际有效 scale 网格**上选择 code。仅针对 Torch 原浮点 scale 优化，可能被后端的二次 scale 压缩改变目标。

即使 out_proj 权重完全恢复 FP32，使用当前设备输入也只有 18.582 dB。因此 out_proj 权重优化不能单独恢复到上游门控输入的 20.968 dB，更不能据此声称完整模型 PPL 已修复。

本轮是归因实验，没有应用新权重或生成新优化 PTE，没有新增性能收益；现有快速 W4 图仍使用上一轮测得的 3.07860 ms。未执行完整模型 PPL，也未验证跨窗口非零缓存 decode。

## 11. 复现和原始数据

在仓库根目录执行，三个脚本依次运行：

```bash
source /home/hunter/anaconda3/etc/profile.d/conda.sh
conda activate et_npu
source env.sh
python llama-output/qwen3_5-0_8b-layer-analysis/diagnose_out_proj_w4.py
python llama-output/qwen3_5-0_8b-layer-analysis/check_out_proj_lpbq.py
python llama-output/qwen3_5-0_8b-layer-analysis/report_out_proj_w4.py
```

复用 `../evaluation.pt`、`../qdq.pt2`、`../metadata.json` 和 `../outputs/`。

- `summary.json` / `metrics.csv`：原 QDQ 网格的各控制变量与逐 token SQNR。
- `lpbq_summary.json` / `lpbq_metrics.csv` / `lpbq_scales.pt`：设备有效 block scale、回放及设备核对。
- `physical_summary.json` / `physical_components.csv` / `physical_cross_terms.csv`：精确分量功率与交叉项。
- `physical_channels.csv`：逐窗口、逐输出通道分析。
- `physical_tokens.csv` / `token_summary.csv` / `out_proj_tokens.png`：128 个 token 位置的原始及累计结果。
- `ranges.csv` / `physical_ranges.csv`：激活实数范围、FP32 min/max 与范围外比例。
- `captures.pt`：使用设备门控输入运行 CPU QDQ 图的 conv 输入/输出。

布局核对：CPU 图的实际 conv 输入与设备门控输入经 reshape 后逐元素一致，QDQ 解码后的权重与手动解码逐元素一致，门控到 conv 之间的激活编码一致；原权重双精度重算与保存的 FP32 输出达到 134.637 dB。

---

<a id="record-09"></a>

## 记录 09：full_gptq_prefill128 / REPORT.md

原始记录位置：`llama-output/qwen3_5-0_8b-layer-analysis/full_gptq_prefill128/REPORT.md`。

# 完整 24 层 PTE：128-token hidden state SQNR

2026-10-13；Qwen3.5-0.8B；SM8750/v79；上下文 1024；prefill 128 token。

## 主要结果

第 0 层设备输出 15.066 dB，与之前单层测试接近。第 7 层（第二个 full attention 层）输入 10.105 dB、输出 4.939 dB，是从第 1 层起最明显的下降位置；同层 GPTQ CPU QDQ 输出为 13.557 dB。

最后一层设备输出 6.378 dB，最终 RMSNorm 后 5.701 dB。相邻层边界在所有八个窗口逐元素一致，全部设备检查点均为有限值。GPTQ 的局部收益没有消除完整设备路径上的累计误差，需要优先对第 7 层做同输入拆解。

第 7 层输入/输出的 FP32 参考值均落在对应的校准范围内，范围外比例为 0%；这只核对层边界，尚不能排除该层内部张量的饱和或上游误差经过 attention 放大。

## 模型与测试范围

PTE 包含完整文本 decoder 的 24 层（18 GDN、6 full attention）、最终 RMSNorm、LM head 和 48 个 cache 输出，并增加逐层 hidden 检查点。输入为原 embedding 生成的 hidden、causal mask、位置和缓存；沿用常规文本 decoder 与 embedding 分离的接口。视觉 encoder 未参与此次文本精度测试。

全部 Linear（包括 FFN、full attention 和 LM head）使用 W4/group=32，depthwise conv W8，激活及 GDN 缓存 UINT16。仅 18 个 GDN out_proj 使用固定 QNN 实际 LPBQ scale 网格上的 GPTQ，其他 Linear 未应用 GPTQ。快速 VTCM op package 保持原实现。

校准和 GPTQ 均使用 CUDA；WikiText 每段前 128 token 校准，后 128 token 评估，共 8 个窗口。FP32 参考运行 CUDA，关闭 TF32。每个评估窗口从零 cache 开始，不验证跨窗口 decode。SQNR 使用 hidden state，未使用全词表 logits。

## 逐层输入与输出

每一层接收上游实际量化路径的输出，误差沿完整模型传播。表中的输入取该层残差分支使用的 hidden，输出取 FFN 残差之后、下层 norm 之前的 hidden。额外的量化边界可能使输入与上一层输出存在微小差异，见 validation.json。

以下数值先累加 8×128 token 的信号/误差功率，再计算 dB，未对各窗口 dB 取算术平均。原 W4 一列是 CPU QDQ 对照，本轮没有导出原 W4 完整设备 PTE。

| 层 | 类型 | 原 W4 CPU 输出 / dB | GPTQ CPU 输出 / dB | GPTQ 设备输入 / dB | GPTQ 设备输出 / dB |
|---:|---|---:|---:|---:|---:|
| 0 | GDN | 13.948 | 16.893 | 84.170 | 15.066 |
| 1 | GDN | 12.809 | 15.165 | 15.066 | 12.947 |
| 2 | GDN | 12.878 | 14.925 | 12.947 | 11.408 |
| 3 | full attention | 12.240 | 13.876 | 11.408 | 10.153 |
| 4 | GDN | 12.388 | 13.814 | 10.153 | 9.924 |
| 5 | GDN | 12.593 | 13.889 | 9.924 | 10.044 |
| 6 | GDN | 13.281 | 14.634 | 10.044 | 10.105 |
| 7 | full attention | 12.362 | 13.557 | 10.105 | 4.939 |
| 8 | GDN | 12.423 | 13.584 | 4.939 | 3.681 |
| 9 | GDN | 12.378 | 13.494 | 3.681 | 4.507 |
| 10 | GDN | 12.883 | 13.969 | 4.507 | 5.392 |
| 11 | full attention | 12.142 | 13.181 | 5.392 | 5.233 |
| 12 | GDN | 12.430 | 13.462 | 5.233 | 5.454 |
| 13 | GDN | 12.685 | 13.694 | 5.454 | 5.317 |
| 14 | GDN | 13.536 | 14.546 | 5.317 | 5.534 |
| 15 | full attention | 12.454 | 13.234 | 5.534 | 5.206 |
| 16 | GDN | 12.730 | 13.557 | 5.206 | 5.739 |
| 17 | GDN | 12.799 | 13.622 | 5.739 | 6.112 |
| 18 | GDN | 13.501 | 14.312 | 6.112 | 6.896 |
| 19 | full attention | 12.914 | 13.600 | 6.896 | 6.695 |
| 20 | GDN | 13.122 | 13.833 | 6.695 | 7.080 |
| 21 | GDN | 13.061 | 13.828 | 7.080 | 6.846 |
| 22 | GDN | 12.520 | 13.323 | 6.846 | 6.500 |
| 23 | full attention | 12.025 | 12.726 | 6.500 | 6.378 |

最终 RMSNorm 后的设备 hidden SQNR：**5.701 dB**。设备检查点非有限值计数：**0**。

## 128 个位置的分布

逐层输出与逐 token SQNR（历史产物已清理）

token_sqnr.csv 包含每层输入/输出、128 个位置在八个窗口上累计的 SQNR。不能把输出减输入的 dB 差直接解释为该层单独新增的量化误差：输出参考信号和误差经过投影、门控、残差及 norm 后，功率都发生变化。本次指标是完整链路的累计误差，若要严格隔离某层，应再做同输入控制变量实验。

## FP32 范围、量化范围、饱和比例

ranges.csv 按每层 input/output 记录 FP32 min/max、校准 scale/zero point、可表示的实数范围和范围外元素比例。参数直接取本次校准图。GPTQ 只改变 out_proj code，并把该权重的 scale 表示规范到既有 QNN 有效网格，未重新校准激活。

## 导出与运行验证

PTE 大小：423,216,480 bytes（403.61 MiB）；18 个 fused GDN core；单个 QNN delegate，无数学算子 CPU fallback。PTE SHA256：`8d3ddd37dea53dbbf81810e19b8e747fc49cdc85b65de2ef03684a3569421d5e`。

诊断 PTE 增加了大量检查点输出，包含完整 LM head logits 输出，不用它的执行时间与之前纯 attention 三输出图比较。该程序用于逐层诊断，增加输出后应使用通用 qnn_executor_runner 和保存的输入运行。

## 文件与复现

- qwen35_full_gptq_prefill128_checkpoints.pte：本次完整 decoder 诊断 PTE。
- layer_sqnr.csv：每层输入/输出的汇总 SQNR。
- token_sqnr.csv：每层 128 个 token 位置的结果。
- ranges.csv：FP32 min/max、量化范围、饱和比例。
- metrics.csv / summary.json / outputs/：CPU 与真实设备结果。
- metadata.json / validation.json：精度配置、层与输出索引、连续边界核对、程序哈希。
- baseline_qdq.pt2 / gptq_qdq.pt2：本次原 W4 与 GPTQ 完整层图。
- out_proj_patches.pt / gptq_calibration.pt：18 个 out_proj 的 code patch、校准输入及原 FP32 权重。

```bash
source /home/hunter/anaconda3/etc/profile.d/conda.sh
conda activate et_npu
source env.sh
export CUDA_VISIBLE_DEVICES=0
python llama-output/qwen3_5-0_8b-layer-analysis/export_full_gptq_prefill128.py
python llama-output/qwen3_5-0_8b-layer-analysis/report_full_gptq_prefill128.py
```

编译和评估按独立进程串行执行。已保存 QDQ 时可以通过 --phase export 重试编译；已保存 PTE 时可以通过 --phase run 重跑设备测试。本轮未计算完整模型 PPL，不能仅由 SQNR 宣称生成质量已修复。

## K/z W8 后续对照

见 [K/z W8 完整模型与匹配性能报告](README_QWEN35_TEST_RECORDS.md#record-10)。复用相同 out_proj GPTQ code 和有效 scale，K/z 输入明显改善，但最终 norm 的设备 hidden SQNR 为 5.414 dB，未超过本报告的 5.701 dB；纯 linear attention 的匹配耗时为 W4 3.0968 ms、K/z W8 3.0981 ms。

---

<a id="record-10"></a>

## 记录 10：full_kz_w8_gptq_prefill128 / REPORT.md

原始记录位置：`llama-output/qwen3_5-0_8b-layer-analysis/full_kz_w8_gptq_prefill128/REPORT.md`。

# K/z W8：完整 24 层 PTE 的误差与匹配性能

2026-10-13；Qwen3.5-0.8B；SM8750/v79；context 1024；prefill 128。层号从 0 开始。

## 结果

K/z W8 明显提高第 0 层 K、z 和 recurrent state 的 SQNR，但完整设备模型没有改善：最终 RMSNorm hidden 从 **5.701 dB** 变成 **5.414 dB**。同条件纯 linear-attention 三输出图的耗时几乎相同：W4 **3.0968 ms**，K/z W8 **3.0981 ms**。

## 精度配置与控制条件

- 18 个 GDN 层：K/z 合并为 4096×1024，W8/per-channel；q/v/a/b 合并为 4128×1024，W4/group=32。
- GDN out_proj、FFN、6 个 full attention 层的 Linear 和 LM head 均为 W4/group=32；depthwise causal conv W8。
- 激活、conv cache 和 recurrent state 保持 UINT16。每个窗口从零 cache 开始。
- 复用上一轮 18 个 GDN out_proj 的 GPTQ code 与 QNN 实际 LPBQ 有效 scale 网格；本轮没有重新训练 GPTQ，也没有给 V 应用 GPTQ。
- 快速 VTCM op package 保持相同二进制；使用原有 int32 移位 reduction，未切换到较慢的全浮点 reduction。
- CUDA 校准使用相同 WikiText 文本的 8×128 token；评估使用同八段文本的后续 128 token。两个完整模型使用相同评估输入和同一份已保存 FP32 hidden 参考。
- FP32 合并投影图与原图在 49 个层边界上逐窗口验证，最低 SQNR **64.36 dB**，并非逐 bit 一致。

169 个 conv 权重节点实际检查为 36 个 W8（18 K/z、18 depthwise）和 133 个 W4。配置直接取校准后 QDQ 图，scale/zero point 未手写固定值。

逐元素核对了两图中共有的 133 个权重节点，code、scale、zero point 全部一致。合并 q/v/a/b 的 18 组矩阵经行重排后，这三个参数也全部一致，只有 K/z 更改权重精度。重新校准后的激活 encoding 可能随图结构或上游精度变化。

SQNR=10·log10(Σreference²/Σ(actual−reference)²)。先累加八个窗口的功率再转 dB，没有平均窗口 dB；使用 hidden/probe 张量，未用整个词表 logits 评价 hidden 精度。

## 第 0 层：沿 GDN 路径逐项对比

W4 内部 probe 来自此前同校准设置的单层 GPTQ PTE；K/z W8 probe 来自本次完整 PTE。下表完整层输出和后续逐层表均比较两个完整 PTE。

| 张量 | 原 W4 设备 / dB | K/z W8 设备 / dB | 变化 / dB | K/z W8 CPU QDQ / dB |
|---|---:|---:|---:|---:|
| state 输入（零 cache） | 零误差 | 零误差 | — | 零误差 |
| conv_state_output | 24.757 | 26.359 | +1.602 | 26.823 |
| core_input_query | 22.136 | 22.136 | +0.000 | 22.585 |
| core_input_key | 20.654 | 39.741 | +19.087 | 39.818 |
| core_input_value | 31.382 | 31.383 | +0.001 | 32.194 |
| core_input_decay | 42.852 | 42.859 | +0.007 | 42.953 |
| core_input_beta | 29.995 | 29.997 | +0.002 | 30.658 |
| gdn_core_output | 29.924 | 31.314 | +1.390 | 32.294 |
| recurrent_state_output | 20.896 | 28.557 | +7.661 | 29.898 |
| rms_weighted | 30.093 | 29.407 | -0.686 | 34.320 |
| z_input | 21.925 | 43.037 | +21.113 | 42.899 |
| gdn_norm_z_gate | 20.968 | 22.719 | +1.751 | 28.839 |
| attention_output | 17.412 | 17.616 | +0.204 | 21.718 |
| 完整层 hidden 输出 | 15.066 | 15.183 | +0.116 | 18.152 |

K/z 输入的改善没有等比例传到 attention 输出：门控仍受 q/v/a/b、GDN reduction 和 RMSNorm 路径影响；out_proj 和 FFN 也保持 W4。仅根据相邻检查点 dB 的差不能断言某个算子单独产生了多少误差，需要同输入替换实验来隔离。

## FP32 min/max、量化范围、饱和比例，再看 state 与 attention SQNR

以下展示关键 probe，ranges.csv 含全部层边界和第 0 层 probe。范围取本次校准后的实际 encoding。FP32 范围外比例表示干净参考值落在量化范围之外的比例；设备端点比例表示输出等于可表示上下界的比例，不能直接反推算子量化前有多少值被裁剪。

| 张量 | FP32 min | FP32 max | 可表示实数 min | 可表示实数 max | FP32 范围外 / % | 设备端点 / % | 设备 SQNR / dB |
|---|---:|---:|---:|---:|---:|---:|---:|
| probe0.core_input_key | -0.999021 | 0.985025 | -0.996902 | 0.987365 | 9.53674e-05 | 9.53674e-05 | 39.741 |
| probe0.core_input_state | 0 | 0 | -3.20219 | 2.57402 | 0 | 0 | inf |
| probe0.gdn_core_output | -0.263981 | 0.198145 | -0.253382 | 0.176977 | 0.000143051 | 0.000190735 | 31.314 |
| probe0.recurrent_state_output | -3.16996 | 2.533 | -3.20219 | 2.57402 | 0 | 0 | 28.557 |
| probe0.gdn_norm_z_gate | -2.55726 | 2.55968 | -2.5616 | 2.51765 | 0.000286102 | 0.000619888 | 22.719 |
| probe0.z_input | -9.19222 | 7.19446 | -7.52583 | 7.19451 | 4.76837e-05 | 0.00100136 | 43.037 |
| probe0.attention_output | -1.28018 | 0.597062 | -1.40218 | 0.587598 | 9.53674e-05 | 9.53674e-05 | 17.616 |

## 完整 24 层：实际上游输入，128 token

每层包含 attention、残差、FFN、残差。输入/输出检查点为残差 hidden 的量化边界；所有 23 对相邻边界在八个窗口上逐元素一致。误差沿整条路径传播，以下是累计误差。

| 层 | 类型 | W4 设备输入 | W4 设备输出 | K/z W8 设备输入 | K/z W8 设备输出 | 设备输出变化 | W4 CPU 输出 | K/z W8 CPU 输出 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | GDN | 84.170 | 15.066 | 84.170 | 15.183 | 0.116 | 16.893 | 18.152 |
| 1 | GDN | 15.066 | 12.947 | 15.183 | 11.997 | -0.950 | 15.165 | 16.140 |
| 2 | GDN | 12.947 | 11.408 | 11.997 | 11.099 | -0.309 | 14.925 | 15.723 |
| 3 | full attention | 11.408 | 10.153 | 11.099 | 9.780 | -0.373 | 13.876 | 14.423 |
| 4 | GDN | 10.153 | 9.924 | 9.780 | 9.351 | -0.573 | 13.814 | 14.359 |
| 5 | GDN | 9.924 | 10.044 | 9.351 | 9.219 | -0.824 | 13.889 | 14.459 |
| 6 | GDN | 10.044 | 10.105 | 9.219 | 10.553 | 0.449 | 14.634 | 15.213 |
| 7 | full attention | 10.105 | 4.939 | 10.553 | 5.300 | 0.361 | 13.557 | 14.021 |
| 8 | GDN | 4.939 | 3.681 | 5.300 | 1.615 | -2.066 | 13.584 | 14.071 |
| 9 | GDN | 3.681 | 4.507 | 1.615 | 2.710 | -1.797 | 13.494 | 13.992 |
| 10 | GDN | 4.507 | 5.392 | 2.710 | 4.042 | -1.350 | 13.969 | 14.453 |
| 11 | full attention | 5.392 | 5.233 | 4.042 | 3.925 | -1.307 | 13.181 | 13.598 |
| 12 | GDN | 5.233 | 5.454 | 3.925 | 4.469 | -0.985 | 13.462 | 13.871 |
| 13 | GDN | 5.454 | 5.317 | 4.469 | 4.727 | -0.590 | 13.694 | 14.103 |
| 14 | GDN | 5.317 | 5.534 | 4.727 | 4.912 | -0.622 | 14.546 | 14.935 |
| 15 | full attention | 5.534 | 5.206 | 4.912 | 4.590 | -0.616 | 13.234 | 13.549 |
| 16 | GDN | 5.206 | 5.739 | 4.590 | 5.138 | -0.601 | 13.557 | 13.925 |
| 17 | GDN | 5.739 | 6.112 | 5.138 | 5.494 | -0.617 | 13.622 | 13.994 |
| 18 | GDN | 6.112 | 6.896 | 5.494 | 6.241 | -0.655 | 14.312 | 14.708 |
| 19 | full attention | 6.896 | 6.695 | 6.241 | 5.995 | -0.701 | 13.600 | 13.939 |
| 20 | GDN | 6.695 | 7.080 | 5.995 | 6.431 | -0.649 | 13.833 | 14.191 |
| 21 | GDN | 7.080 | 6.846 | 6.431 | 6.160 | -0.686 | 13.828 | 14.179 |
| 22 | GDN | 6.846 | 6.500 | 6.160 | 5.854 | -0.646 | 13.323 | 13.668 |
| 23 | full attention | 6.500 | 6.378 | 5.854 | 6.015 | -0.363 | 12.726 | 13.037 |
| 最终 RMSNorm | hidden | — | 5.701 | — | 5.414 | -0.287 | 11.994 | 12.257 |

### token 1 与 token 128 的变化

每个位置先汇总八个窗口的功率。第 7、8 层的设备误差随 token 位置明显恶化；下表不是额外 decode 测试，均来自同一次 128-token prefill。

| 层 | token 位置 | W4 设备输出 / dB | K/z W8 设备输出 / dB | K/z W8 CPU 输出 / dB |
|---:|---:|---:|---:|---:|
| 7 | 1 | 18.351 | 18.496 | 18.923 |
| 7 | 128 | 3.477 | 4.255 | 13.747 |
| 8 | 1 | 19.018 | 19.209 | 19.615 |
| 8 | 128 | 2.549 | 0.351 | 13.729 |

逐层、逐 token 输出 hidden SQNR（历史产物已清理）

第 7 层 full attention 后两种设备路径都只有约 5 dB；K/z W8 在第 8 层进一步降至 1.615 dB，而 CPU QDQ 路径没有同样下降。本次完整模型结果不支持把 K/z W8 当作端到端精度修复。下一步应对第 7 层 full attention 和第 8 层 GDN/FFN 做同输入的 CPU QDQ、NPU、FP32 对照，分别检查内部范围及算子误差；目前不能把全部差异归因于 VTCM 算子。

第 7、8 层 input/output 边界的 FP32 范围外比例与设备端点比例均为 0%；这排除了所观察边界的裁剪，尚未排除层内部激活或 state 的饱和。

## 匹配耗时与大小

从两个本次完整 QDQ 图中提取第 0 层纯 linear attention，仅保留 attention、conv cache、state 三个输出。使用相同归一化 hidden 输入；不包含输入 RMSNorm、残差、FFN 和诊断输出。两个提取图的 CPU 三输出均与源完整图逐元素一致。

设备上交替测试三轮，每轮每种配置八个输入，每输入预热 5 次、执行 100 次。以下耗时不是完整 24 层延迟，也不是纯 GDN core 延迟。

| 项目 | 全 Linear W4 | K/z W8 |
|---|---:|---:|
| 纯 linear attention 平均耗时 / ms | 3.096776 | 3.098110 |
| 三输出 attention PTE / MiB | 5.936 | 7.822 |
| 完整诊断 PTE / MiB | 403.611 | 436.913 |

完整诊断 PTE 的输出数不同（本次增加第 0 层内部 probe），其大小差也含诊断输出/编译布局的影响。K/z 权重从 W4 增至 W8 的纯权重增量理论上为 18×4096×1024×0.5 byte = 36 MiB，其他布局变化影响实际 PTE 差值。

所有被检查设备张量均无 NaN/Inf。两个纯 attention PTE 和完整 PTE 均为单个 QNN delegate，完整图包含 18 个 fused GDN core，未发生数学算子 CPU fallback。libQnnDeltaNetOpPackage_HTP.so 的 SHA256 为 `00ab32f78a8e5a9e340993bb48adf7e0bc7b2b656d115a3b5a66b754f4f085fd`。

## 文件与复现

- qwen35_full_gptq_prefill128_checkpoints.pte：K/z W8 完整文本 decoder 诊断 PTE，目录标明配置；未包含视觉 encoder。
- layer_comparison.csv / token_comparison.csv：24 层输入输出与各 128 个位置的 SQNR。
- layer0_probe_comparison.csv / ranges.csv：GDN 内部误差、实际校准参数、范围及端点比例。
- summary.json / validation.json / metadata.json / outputs/：汇总、控制条件和原始设备输出。
- weight_comparison.json：其余权重与 q/v/a/b 重排前后的逐元素一致性检查。
- performance/performance.json：匹配性能、三轮原始均值和 op package 哈希。

```bash
source /home/hunter/anaconda3/etc/profile.d/conda.sh
conda activate et_npu
source env.sh
export CUDA_VISIBLE_DEVICES=0
python llama-output/qwen3_5-0_8b-layer-analysis/export_full_kz_w8_prefill128.py
python llama-output/qwen3_5-0_8b-layer-analysis/benchmark_kz_w8_prefill128.py
python llama-output/qwen3_5-0_8b-layer-analysis/report_full_kz_w8_prefill128.py
```

导出按独立进程串行完成。此处评估冷 cache 的 128-token prefill；未测跨窗口 decode，也未重测完整模型 PPL，不能据此宣称生成质量或 PPL 已解决。

## 后续 PPL 检验

见 [相同 128-token 窗口的 PPL 对照](README_QWEN35_TEST_RECORDS.md#record-13)。共 1016 个目标，HF FP32 41.3273、静态 FP32 41.3275、W4 CPU QDQ 48.6054、K/z W8 CPU QDQ 47.7622、W4 设备 94.4445、K/z W8 设备 102.3431。浮点基线一致，但 K/z W8 的设备 PPL 仍为同配置 CPU QDQ 的 2.143 倍；这是冷 prefill 的短窗口诊断结果，尚未评估跨 chunk cache 和标准长上下文 PPL。

## 第 7、8 层定位与精度修复

见 [ReduceMean 与 GDN 低位丢失的设备验证报告](README_QWEN35_TEST_RECORDS.md#record-12)。Q/K 改为二维 native RMSNorm 后 PPL 为 87.398；保留 GDN 归约低位后为 50.076，state 仍是 UINT16。同轮纯 linear attention 耗时为原缩位版 2.947 ms、完整 FP32 归约 4.527 ms、新整数归约 3.550 ms。准确结果需要修复后的 PTE 搭配新的整数归约 HTP 库。

---

<a id="record-11"></a>

## 记录 11：complete_prefill128 / REPORT.md

原始记录位置：`llama-output/qwen3_5-0_8b-layer-analysis/full_kz_w8_gptq_prefill128/complete_prefill128/REPORT.md`。

# 完整 Qwen3.5-0.8B：128-token prefill 性能与精度

## 配置与比较口径

- 设备：SM8750/v79，序列容量 1024，24 层（18 GDN + 6 full attention）。
- K/z 合并 W8 per-channel；q/v/a/b 合并 W4 group32；其余 Linear W4 group32，causal depthwise conv W8。
- 原有 18 个 GDN out_proj 的 GPTQ 权重码及 QNN LPBQ 有效 scale 保留；不重新校准。原校准在 CUDA 上完成。
- 激活、conv cache、recurrent state 仍为 UINT16；精确版使用整数分解恢复点积中的低 4 bit，并在归约结束后合并为 FP32。
- 精度：同一批 8 个冷 cache 的 128-token 窗口，对比 FP32；SQNR 汇总先累加 signal/noise power 再取 dB，不平均 dB。
- 耗时：49 个常规输出（全部 128-token logits + 48 个 cache），移除所有额外诊断输出。三轮交替测，每轮 8 个样本、3 次 warmup、20 次执行。
- 计时覆盖完整 decoder，包括 FFN、final norm、LM head、cache 输出；不含 host embedding lookup、模型加载、ADB 传输和结果文件落盘。

## 实际 runner PTE：UINT16 接口（主要性能结论）

完成 PTE 类型检查：hidden 输入为 Float32，position 为 Int32，mask 与全部 48 个 cache 输入为 UINT16，logits 与全部 48 个 cache 输出为 UINT16；embedding 输出为 Float32。

| 配置 | 128-token 完整 decoder 平均耗时 | 增加 |
|---|---:|---:|
| 修复前快速 VTCM | 112.756 ms | — |
| Q/K native norm + 精确整数 GDN | 126.968 ms | **14.212 ms / 12.60%** |

同一批 8 个样本、三轮交替测，每轮 3 次 warmup、20 次执行。包含 24 层、FFN、final norm、LM head 和 cache 输出，排除加载、host embedding、ADB 传输和输出文件落盘。修复前对照只包含 `prefill_forward`；修复后直接运行已验证生成的 hybrid PTE 的 `prefill_forward`。两边使用同样的权重、cache 编码与输出对齐、编译选项。方法数量的差别已在原始汇总记录。

下面原先的 141.772 → 160.423 ms 表是 **Float32 边界的独立 decoder 对照**，不能与这个 UINT16 接口表混用。正常 runner 单次 128-token prompt 的 prefill 为 104 ms、decode 为约 28.9 token/s；该数字包含 host prompt/embedding 处理且输入文本不同，不作为三轮配对性能结论。

实际 hybrid runner PTE 在同批 1016 个 next-token 目标上的 **PPL = 50.076208**，与诊断 PTE 的 50.076208 一致。

## Float32 边界：128-token 完整 decoder 耗时

| 配置 | 平均 ms | 相对原版增加 |
|---|---:|---:|
| 修复前快速 VTCM | 141.772 | 0.000 ms / 0.00% |
| 仅 Q/K native RMSNorm 修复 | 149.934 | 8.162 ms / 5.76% |
| Q/K norm 修复 + 精确整数 GDN | 160.423 | 18.651 ms / 13.16% |

精确版相对仅 norm 修复版增加 10.489 ms。

这是完整 24 层的时间。此前约 3 ms 的数字对应单层纯 linear-attention，不含 FFN，不能直接与这一表比较。

## 各层输出 hidden 与 recurrent state SQNR

| 层（从 0 开始） | hidden 原版 dB | hidden 修复 dB | state 原版 dB | state 修复 dB |
|---|---:|---:|---:|---:|
| 0 | 15.18 | 17.80 | 28.56 | 28.63 |
| 1 | 12.00 | 15.75 | 14.71 | 17.88 |
| 2 | 11.10 | 15.36 | 12.79 | 17.12 |
| 3 | 9.78 | 14.02 | — | — |
| 4 | 9.35 | 13.75 | 15.50 | 19.67 |
| 5 | 9.22 | 13.87 | 13.80 | 18.45 |
| 6 | 10.55 | 14.67 | 9.72 | 14.76 |
| 7 | 5.30 | 13.49 | — | — |
| 8 | 1.61 | 13.45 | 7.02 | 15.51 |
| 9 | 2.71 | 13.39 | 2.77 | 14.50 |
| 10 | 4.04 | 13.85 | 2.51 | 13.04 |
| 11 | 3.93 | 13.05 | — | — |
| 12 | 4.47 | 13.29 | 7.72 | 16.73 |
| 13 | 4.73 | 13.53 | 5.31 | 15.14 |
| 14 | 4.91 | 14.36 | 5.58 | 14.97 |
| 15 | 4.59 | 13.04 | — | — |
| 16 | 5.14 | 13.39 | 7.42 | 13.81 |
| 17 | 5.49 | 13.47 | 8.51 | 14.58 |
| 18 | 6.24 | 14.19 | 9.48 | 13.73 |
| 19 | 5.99 | 13.51 | — | — |
| 20 | 6.43 | 13.76 | 10.10 | 14.35 |
| 21 | 6.16 | 13.74 | 9.98 | 14.62 |
| 22 | 5.85 | 13.24 | 11.75 | 16.30 |
| 23 | 6.01 | 12.61 | — | — |

最终 norm hidden：**5.41 → 11.84 dB**。

## 128-token 内的位置变化

| 位置（从 1 开始） | final norm 原版 dB | final norm 修复 dB |
|---|---:|---:|
| 1 | 3.13 | 3.13 |
| 32 | 5.77 | 13.05 |
| 64 | 5.20 | 12.28 |
| 96 | 5.19 | 12.36 |
| 128 | 4.45 | 11.32 |

第一个 token 仍只有约 3.13 dB，修复并未改善该位置。总体误差下降不代表每个位置、每个算子都已解决。

## 第一层 GDN 内部

| 输出 | 原版 dB | 修复 dB |
|---|---:|---:|
| gdn_core_output | 31.31 | 32.10 |
| recurrent_state_output | 28.56 | 28.63 |
| rms_weighted | 29.41 | 33.79 |
| gdn_norm_z_gate | 22.72 | 28.20 |
| attention_output | 17.62 | 21.39 |
| conv_state_output | 26.36 | 26.36 |

这些窗口的初始 recurrent state 全为零，state 输入 SQNR 为无穷大。它不能用于判断跨 chunk 的 state 回写是否正确；上表 state SQNR 是处理完 128 token 后的输出。

## 同批 PPL

| 对照 | PPL |
|---|---:|
| HF FP32 | 41.327 |
| CPU QDQ + K/z W8 + GPTQ | 47.768 |
| 修复前设备 | 102.343 |
| norm 修复 + 精确整数 GDN 设备 | 50.076 |

这一 PPL 对应 8 × 128 token 的冷 cache 窗口，共 1016 个 next-token 目标；不是完整 WikiText 或 1024-token 连续上下文 PPL。精确版比 CPU QDQ 高约 4.83%，比 FP32 高约 21.17%，仍有量化误差。

## 数据文件

- 耗时原始汇总（历史产物已清理）
- hidden SQNR 汇总（历史产物已清理）
- 逐样本 hidden SQNR（历史产物已清理）
- 逐层 cache SQNR（历史产物已清理）
- 逐样本 cache min/max 与误差（历史产物已清理）

## 完整 PTE 与实际生成

- hybrid decoder PTE（历史产物已清理）：`prefill_forward`（128 token）+ `kv_forward`（1 token），context capacity 1024。
- embedding PTE（历史产物已清理）。
- 实际 PTE 的接口类型检查（历史产物已清理）。
- 正常 runner 性能与 PPL（历史产物已清理）。
- 实际生成、HF FP32 对照与命令（历史产物已清理）。
- [运行说明](README_QWEN35_TEST_RECORDS.md#record-14)。

128-token prompt 的摘要输出：

> In 2006, the character Boulter starred alongside Ben Whishaw in the play Citizenship, which was written by Mark Raven.

26-token 短 prompt（KV 模式）“What is 2 + 2? Reply with only the number.”：设备和 HF FP32 均输出 `4`。

生成测试是文本模式。runner 加载已有 vision encoder，但本次没有执行图像编码。128-token hybrid prompt 恰好占满一个 chunk，没有 padding；任意非 128 倍数的 hybrid prompt 和跨多个 prefill chunk 的质量尚未验证。短 prompt 已通过 KV 模式验证。

---

<a id="record-12"></a>

## 记录 12：layer7_8_replay / REPORT.md

原始记录位置：`llama-output/qwen3_5-0_8b-layer-analysis/full_kz_w8_gptq_prefill128/layer7_8_replay/REPORT.md`。

# PPL 误差定位与修复：Q/K ReduceMean 与 GDN 低位丢失

2026-10-13；SM8750/v79；QNN 2.49.40.260810；Qwen3.5-0.8B。层号从 0 开始。

## 完整模型 PPL

同一批八个 WikiText 预留窗口，各 128 token、零 cache，共 1016 个 next-token 目标。编译容量 1024；此结果尚未覆盖跨 chunk cache 或长窗口 decode。所有阶段保留 K/z W8、q/v/a/b W4、FFN/full-attention/LM-head W4/group=32、18 个 out_proj GPTQ 和 UINT16 conv/state。

| 配置 | PPL ↓ |
|---|---:|
| HF FP32 | 41.3273 |
| 同配置 CPU QDQ | 47.7683 |
| 原 K/z W8 设备 | 102.3431 |
| Q/K native RMSNorm，原快速归约 | 87.3979 |
| Q/K 修复 + 完整 FP32 归约 | 50.1004 |
| Q/K 修复 + 高低位整数归约（64 个 V/块） | 50.0762 |

没有重新校准、改变投影权重 code、升级 state 到 FP16 或追加 FFN GPTQ。后两个设备结果使用同一份完整 PTE，仅替换 HTP op package 实现，按库/PTE SHA256 留存证据。修复后设备与 CPU QDQ 的额外 PPL 差距显著缩小，剩余量化质量仍低于 FP32。

## 1. 第 7 层：误差来自 Q/K 方差归约

从已保存完整 QDQ 图提取第 7 层，输入量化边界、mask、position、KV cache、权重和校准 encoding 原样保留。分别回放八个干净 FP32 hidden 和八个原完整设备 hidden。设备输入回放的独立层输出与原完整 PTE 在八个窗口为 61.8–62.5 dB，能重现原来的下降。

把所有已观察的设备检查点注入 CPU 图，逐段重新计算，并将 W4 scale 换算到 NodeVisitor 使用的实际 LPBQ 网格。这样每段都使用其上游实际设备张量，避免把上游误差算到本段。

| 第 7 层分段回放（设备输入轨迹） | 设备与同输入 CPU LPBQ SQNR / dB |
|---|---:|
| q_projection | 83.117 |
| k_projection | 74.909 |
| v_projection | 73.446 |
| q_norm_fn.rsqrt_input | 8.014 |
| q_norm_fn.rsqrt_output | 104.484 |
| q_norm_fn.output | 81.348 |
| k_norm_fn.rsqrt_input | 8.904 |
| k_norm_fn.rsqrt_output | 104.599 |
| softmax | 63.863 |
| context | 43.955 |

额外导出 square 和 mean 进一步分离：Q 的平方输出约 33 dB、K 约 36 dB（该对照仍含 LPBQ 与原 block scale 的差异），mean 降至约 8 dB。直接对观察到的设备 square 张量在 CPU 计算 mean，Q 仅 8.139 dB、K 8.831 dB，确认大误差发生在本图的量化 ReduceMean 执行路径。

示例：Q head 0 的首 token，观察到的 square 在 CPU 归约为 1.025848，设备 mean 为 0.531790；K 对应为 1.704602 与 0.598617。这些平均值落在校准范围内。当前证据没有确定 SDK 内部究竟是 requantization、布局还是优化实现导致，因此报告不把未观察到的内部机制当作事实。

给定设备的 variance+eps，CPU rsqrt 与设备约 104 dB；softmax 在同 masked_scores 下约 63.9 dB，行概率和接近 1、被 mask 的概率质量为 0。它们的累计 SQNR 低主要是接收了上游错误的归一化值。

修复：Q/K 的四维输入 [B,H,T,D] 先展成二维 [-1,D]，用原生 RMSNorm，再恢复形状。生产 Qwen35RMSNorm 仍保留 1+weight 的语义。独立层 weighted-native 与 scaleless-native 两种设备验证均通过；scaleless-native 与 CPU QDQ 的完整层输出约 29.4 dB，而原 mean 路径约 6 dB。

完整 PTE 使用冻结校准图的替换版本：把已校准 gamma 烘焙为静态输入，保留原输入/输出 encoding，替换全部六个 full attention 层的 12 个 Q/K norm。完整模型第 7 层 hidden 从 5.300 提升到 10.490 dB，PPL 从 102.343 降至 87.398。

## 2. 第 8 层：GDN 的低位被归约丢弃

第 8 层输入投影/conv/L2 与 CPU LPBQ 接近，逐段回放 GDN core 输出只有约 23.4 dB。同 RMSNorm 输入下 native RMSNorm 为 67.2 dB；同 RMS/z 输入下 gate 为 69.0 dB。门控的累计误差主要继承了 core 的误差，并被低幅值向量的归一化放大。

原快速 uint16 kernel 在计算 KᵀS、QᵀS 前，state 和 K/Q 均缩掉四位，再把结果乘回 2⁸；这些低位无法恢复。缩位不只影响表示的最小噪声，低幅值 readout 经过 RMSNorm/z 后会成为明显的预测误差。完整精度归约库保留全部中心化整数，以 FP32 归约，state 仍按 UINT16 存储及更新。

| 同设备 hidden 输入，第 8 层累计比较 | 原快速归约 / dB | 完整归约 / dB | 新整数归约 / dB |
|---|---:|---:|---:|
| gdn_output | 23.365 | 42.523 | 42.523 |
| state_output | 34.183 | 37.975 | 37.975 |
| rms_weighted | 13.702 | 36.622 | 36.622 |
| norm_z_gate | -0.566 | 23.029 | 23.029 |
| attention_output | -5.171 | 19.585 | 19.585 |
| output | 4.053 | 30.329 | 30.329 |

第 8 层独立诊断图与原完整 PTE 的输出为约 28–29 dB，尚非逐 bit 一致。完整模型的同 PTE、仅换归约库控制实验用于验证该改动在实际整条路径上的 PPL 收益，避免仅凭独立诊断图下结论。

### FP32 范围、量化范围、范围外比例

| 张量 | FP32 min | FP32 max | 可表示 min | 可表示 max | FP32 范围外 / % |
|---|---:|---:|---:|---:|---:|
| core_input_state | 0 | 0 | -3.10833 | 4.18375 | 0 |
| state_output | -2.99964 | 4.28899 | -3.10833 | 4.18375 | 0.000143051 |
| gdn_output | -0.14168 | 0.15722 | -0.127144 | 0.14604 | 0.000572205 |
| norm_z_gate | -2.36727 | 4.77341 | -2.36557 | 4.98625 | 4.76837e-05 |
| attention_output | -0.314754 | 0.375603 | -0.331369 | 0.325386 | 0.000190735 |

state 输入为零 cache，两种路径均为零误差。state 输出和 attention/readout SQNR 见上表。范围/scale/zero point 均取校准图，未手写默认参数；segment_ranges.csv 还记录同输入各段量化前 min/max、范围外比例及设备端点比例。范围外比例不能直接等同于设备内部真实裁剪率。

## 3. 保留低位的整数归约与性能

把 state 与系数拆成高/低位：S=16·S_hi+S_lo，K=16·K_hi+K_lo。分别累计 A=ΣS_hiK_hi、B=Σ(S_hiK_lo+S_loK_hi)、C=ΣS_loK_lo，最后以 FP32 合成 256A+16B+C。K、Q 两个投影复用同一次 state 读取。

K≤128 时，各 int32 累加器不会溢出：最大 |A|≤128×4095×4096 < INT32_MAX；B、C 的界更小。更大的 K 使用完整 FP32 归约保底。已对四个 K 长度、五个 zero point、随机及正/负端点系数的 60 组情况验证与 int64 dot 完全一致，并在真实完整 PTE 上测得相同 PPL。

最初同时处理两个 64-V 块的整数版需更多寄存器，实测 7.291 ms；改为一次一个块后为 3.550 ms，PPL 保持 50.076。这里测量确认了块大小的性能影响，没有把寄存器溢出当作已从汇编证明的结论。

| 单层纯 linear attention，三输出同 PTE | 匹配耗时 / ms |
|---|---:|
| 原缩位快速版 | 2.947185 |
| 完整 FP32 归约 | 4.526751 |
| 高低位整数归约，64-V/块 | 3.550387 |

新整数版本相对同轮原快速版增加 **20.5%** 耗时。三轮交替测试，每轮八个输入，每输入预热 5 次、执行 100 次。测量是单层 attention，未包含 FFN、残差、输入 RMSNorm或诊断检查点。不能把它当作完整 24 层延迟，也尚未达到历史 2 ms 的目标。

## 修改、产物和使用

- 生产 layernorm.py：四维 Q/K 改为二维 native RMSNorm；FP32/BF16/FP16、两种 head 数的浮点公式检查通过，原 add_unit_offset 保留。
- DeltaNetQhpi.cpp：增加 DELTANET_U16_SPLIT_REDUCTION=1 选项，原快速默认路径保留；默认、FP32 控制与整数控制的 v79 编译均通过。
- qk_native_rms/qwen35_full_gptq_prefill128_checkpoints.pte：Q/K 修复后的完整 decoder 诊断 PTE，包含 24 层、LM head、缓存和 hidden 检查点。
- 精准整数运行库：layer7_8_replay/split_reduction_v1_build/hexagon-v79/libQnnDeltaNetOpPackage_HTP.so。此 PTE 必须搭配该库，才能重现 50.076 的 PPL；搭配原缩位库为 87.398。
- 原 ARM op package 接口保持相同；每次控制实验结束已恢复设备工作区的原库。
- 完整 PPL、库与 PTE 哈希：qk_native_rms/split_reduction_v1_control/summary.json 与 identity.json。
- 各算子、各段误差：summary.csv、segment_summary.csv、segment_ranges.csv、norm_variance/metrics.csv。
- 匹配性能：reduction_performance_v1/summary.json；旧双块版性能在 reduction_performance/。

构建时从 SDK 环境运行原 Makefile，以单独 WORK 保存变体，并指定 HEXAGON_CPPFLAGS=-DDELTANET_U16_SPLIT_REDUCTION=1。此次实际构建命令及结果保存在 split_reduction_v1_build.log。test_full_precision_reduction.py --split 复测 64-V 单块产物；--split-two-blocks 指向旧双块快照。

目前结果解决了本批窗口上主要的设备额外误差，剩余 PPL 相对 FP32 约高 21%。尚需长上下文及跨 chunk/decode 验证才能评价正常应用的完整生成质量；此处没有宣称所有 PPL 误差已经消失。

---

<a id="record-13"></a>

## 记录 13：ppl_prefill128 / REPORT.md

原始记录位置：`llama-output/qwen3_5-0_8b-layer-analysis/full_kz_w8_gptq_prefill128/ppl_prefill128/REPORT.md`。

# 128-token prefill：PPL 对照

Qwen3.5-0.8B；8 个相同文本窗口；每窗 128 token，零 cache；共 1016 个 next-token 目标。

模型编译容量为 1024，但本次每窗实际上下文最多 127 token，未验证跨 chunk cache，也不是标准完整 WikiText-2 test-set PPL。评估文本为此前校准文章各段预留的后 128 token；没有 padding、聊天模板、BOS、采样或 repetition penalty。

PPL = exp(sum(token NLL) / 1016)。原始 logits 以 float64 logsumexp 计分，logits[t] 对齐参考 tokens[t+1]，忽略没有后续目标的最后一个 logits。

## 结果

| 配置 | PPL | mean NLL | 相对 HF PPL | 目标 top-1 准确率 | 与 HF top-1 一致率 |
|---|---:|---:|---:|---:|---:|
| hf_fp32 | 41.3273 | 3.721522 | 1.000× | 36.12% | 100.00% |
| fp32_static | 41.3275 | 3.721528 | 1.000× | 36.12% | 100.00% |
| all_w4_cpu_qdq | 48.6054 | 3.883734 | 1.176× | 33.46% | 77.26% |
| kz_w8_cpu_qdq | 47.7622 | 3.866234 | 1.156× | 33.56% | 78.54% |
| all_w4_device | 94.4445 | 4.548013 | 2.285× | 26.87% | 55.31% |
| kz_w8_device | 102.3431 | 4.628331 | 2.476× | 25.98% | 51.87% |

## 判断

HF FP32 与静态 FP32 的 mean NLL 仅差 0.00000604，1016 个目标上的 top-1 预测全部一致，浮点转换路径与计分对齐得到独立交叉验证。

CPU QDQ 上 K/z W8 从原 W4 的 48.605 改善为 47.762；设备上却从 94.445 恶化为 102.343。本次设备 PPL 是 HF 的 2.476 倍，也是同配置 CPU QDQ 的 2.143 倍。

需要优先隔离 CPU QDQ 与 QNN 下沉后实际执行之间的额外差异，包括有效 LPBQ scale、融合计算/累加精度及内部张量饱和。CPU QDQ 仍保留原 block scale 表示，而 QNN 使用实际 LPBQ 网格，这两者存在语义差异；当前 PPL 差距不能直接证明硬件算子实现有错。

上一轮 hidden 检查点显示第 7 层 full attention 和第 8 层 GDN/FFN 是明显的设备误差位置。这次 PPL 进一步确认整条预测路径仍有实质性的质量损失；它不能单独定位到其中某一个算子，也未测试跨 chunk cache。

## 按 token 位置拆分

以下分桶 PPL 使用各桶目标 NLL，不是逐窗口 PPL 的算术平均。

| logits 位置 | HF FP32 | 静态 FP32 | W4 CPU QDQ | K/z W8 CPU QDQ | W4 设备 PTE | K/z W8 设备 PTE |
|---|---:|---:|---:|---:|---:|---:|
| 1..32 | 63.471 | 63.471 | 73.027 | 71.680 | 112.118 | 116.957 |
| 33..64 | 41.815 | 41.814 | 50.667 | 49.154 | 96.707 | 93.564 |
| 65..96 | 37.071 | 37.073 | 43.892 | 43.579 | 100.942 | 110.669 |
| 97..127 | 29.332 | 29.333 | 33.985 | 33.519 | 72.084 | 90.233 |

## 来源与复现

设备结果直接读取刚完成完整模型真实设备测试的 outputs/output_N_0.raw，按已校验 shape 读取 float32 logits；PTE 哈希、每个 raw 哈希及对应目标 ID 保存在 manifest 和各 case JSON。

静态 FP32 使用上一轮完整图保存的最终 norm hidden，接原始 checkpoint 的 FP32 LM head。CPU QDQ 使用各完整 QDQ 图保存的最终 norm hidden，接该图原有 LM head 子图；没有重新校准或改变任何量化参数。HF FP32 独立加载原模型，作为浮点基线。

- summary.json：PPL、NLL、目标准确率和比较口径。
- 各 case JSON：逐窗口、逐 token NLL、目标 ID、top-1 ID、目标排名。
- position_buckets.csv：按 token 位置的误差变化。

```bash
source /home/hunter/anaconda3/etc/profile.d/conda.sh
conda activate et_npu
source env.sh
export CUDA_VISIBLE_DEVICES=0
python llama-output/qwen3_5-0_8b-layer-analysis/evaluate_full_prefill128_ppl.py
```

各阶段通过独立进程串行运行，复用既有 PTE 和保存的 hidden；没有并行加载多个完整模型。

## 后续修复

见 [第 7、8 层同输入定位报告](README_QWEN35_TEST_RECORDS.md#record-12)。定位到 Q/K 的量化 ReduceMean 及 GDN 归约前缩位；修复组合将同口径设备 PPL 降至 50.076。新的整数归约保留 UINT16 state，匹配纯 attention 耗时 3.550 ms，比同轮原缩位版 2.947 ms 增加约 20.5%。

---

<a id="record-14"></a>

## 记录 14：complete_hybrid / README.md

原始记录位置：`llama-output/qwen3_5-0_8b-layer-analysis/full_kz_w8_gptq_prefill128/qk_native_rms/complete_hybrid/README.md`。

# Qwen3.5-0.8B 完整 hybrid PTE

## 运行配置

- SM8750/v79；context 1024；prefill AR=128；decode AR=1。
- K/z W8，q/v/a/b 与其余 Linear W4 group32，causal conv W8。
- 18 个 GDN out_proj 使用原 GPTQ 权重；原 CUDA 校准参数保留。
- 激活、conv cache 与 recurrent state 是 UINT16。hidden 是 Float32，mask/cache/logits 的 runner 接口是 UINT16。
- HTP op 使用 `DELTANET_U16_SPLIT_REDUCTION=1` 的精确整数归约库；state 存储保持 UINT16。

## 文件

- hybrid_llama_qnn.pte（历史产物已清理）：469,642,752 bytes。
- tok_embedding_qnn.pte（历史产物已清理）：508,639,744 bytes。
- tokenizer.json（历史产物已清理）：12,807,982 bytes。
- libraries/libQnnDeltaNetOpPackage_HTP.so（历史产物已清理）：1,082,216 bytes。
- libraries/libQnnDeltaNetOpPackage.so（历史产物已清理）：873,808 bytes。

已有 vision encoder 位于 `llama-output/qwen3_5-0_8b-diagnose/kv-cache-fixed/vision_encoder_qnn.pte`。标准 multimodal runner 需要加载它；文本测试传空 image list，不执行 encoder。

## 复现与部署

在仓库根目录执行：

```bash
source /home/hunter/anaconda3/etc/profile.d/conda.sh
conda activate et_npu
source env.sh
export CUDA_VISIBLE_DEVICES=0
python llama-output/qwen3_5-0_8b-layer-analysis/run_complete_hybrid.py
```

该脚本推送配套 PTE、tokenizer、encoder 和 op 库，运行精确 128-token 的 hybrid prompt 与短 prompt 的 KV 模式，并计算同 prompt 的 HF FP32 greedy 对照。实际设备 shell 命令、输出和 SHA256 保存在 generation/summary.json（历史产物已清理）。

仅部署、不重复运行：

```bash
python llama-output/qwen3_5-0_8b-layer-analysis/run_complete_hybrid.py --deploy-only
```

设备 workspace 为 `/data/local/tmp/executorch/qwen35_complete_hybrid`，op package 路径已经绑定在 PTE 中。请保持该目录与 `libraries/` 中的两份配套库。默认包目录下的 HTP 库仍是原快速近似版本；替换回默认库会失去本次精度修复。

## 实测结果

- 128-token 完整 decoder，UINT16 接口：**112.756 → 126.968 ms**，增加 **14.212 ms（12.60%）**。
- 最后一层 hidden SQNR：6.015 → 12.610 dB；final norm hidden：5.414 → 11.837 dB。
- 实际本 PTE 的 PPL：**50.076208**；同批 HF FP32 为 41.327、CPU QDQ 为 47.768。
- 128-token hybrid prompt 输出连贯英文摘要；短 prompt 的 KV 模式输出 `4`，与 FP32 一致。

[完整性能/逐层 SQNR 报告](README_QWEN35_TEST_RECORDS.md#record-11)。

## 测试范围

PPL 使用八个冷 cache 的 128-token 窗口、1016 个 next-token 目标；不是完整 WikiText 或连续 1024-token PPL。生成测试包括恰好 128-token 的 hybrid prompt 与 26-token 的 KV prompt。非整 chunk 的 hybrid prompt、跨多个 prefill chunk 和图像输入的质量尚未验证。第一个 token 的 final norm SQNR 仍约 3.13 dB，量化误差没有全部消除。

---

<a id="record-15"></a>

## 记录 15：context_performance_20261014 / REPORT.md

原始记录位置：`llama-output/qwen3_5-0_8b-layer-analysis/full_kz_w8_gptq_prefill128/qk_native_rms/context_performance_20261014/REPORT.md`。

# Qwen3.5-0.8B 上下文长度性能

- 设备 SM8750，HTP v79，序列号 3B1F57E9B8L74X5D；QNN 2.49.40.260810；HTP Burst；runner 为 RelWithDebInfo（O2）构建。
- Decode：输入 16 token，输出 n−16 token；强制忽略 EOS。首个输出由 prompt 阶段产生，decode 计时包含 n−17 次 KV forward。
- Prefill：输入 n token，输出 1 token；prefill AR=128，无 padding；此项不执行 KV decode forward。
- 每档分别编译 context/cache 容量 n+1，为 prefill 的首个输出保留一个位置；不是共用 8K 静态图。
- 完整 24 层，K/z W8，其余 Linear W4 group32，out_proj GPTQ，精确整数 GDN，state/conv cache 与 logits 接口 UINT16。
- 原 CUDA 校准与权重保持；RoPE 表按同一公式扩展，原 1K 部分的量化码逐项一致。
- 时间取 runner 的阶段时间戳，不含模型加载和 ADB；prefill 包含 tokenizer、embedding、CPU cache 更新，decode 包含 embedding、forward、采样、cache 更新。
- Decode 报告整个生成过程的平均速度，有效上下文从 16 增长到 n；不是单独测最后一步。

| n | 实际容量 | Prefill ms | Prefill token/s | Decode ms/token | Decode token/s | 重复次数 |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1025 | 938.00 | 1091.68 | 51.378 | 19.46 | 2 |
| 2048 | 2049 | 2159.00 | 948.59 | 80.332 | 12.45 | 2 |
| 3072 | 3073 | 4151.00 | 740.06 | 106.967 | 9.35 | 1 |
| 4096 | 4097 | 6539.00 | 626.40 | 132.882 | 7.53 | 1 |

## 每轮结果

| n | 轮次 | Prefill ms | 折合每 128 token ms | Decode 总时间 s | Decode ms/token | 电池温度（decode 前→后）°C |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1 | 927.00 | 115.875 | 51.318 | 50.961 | 29.0→未记录 |
| 1024 | 2 | 949.00 | 118.625 | 52.158 | 51.795 | 29.2→未记录 |
| 2048 | 1 | 2155.00 | 134.688 | 162.181 | 79.853 | 29.4→未记录 |
| 2048 | 2 | 2163.00 | 135.188 | 164.128 | 80.811 | 30.2→未记录 |
| 3072 | 1 | 4151.00 | 172.958 | 326.785 | 106.967 | 30.8→32.9 |
| 4096 | 1 | 6539.00 | 204.344 | 542.027 | 132.882 | 33.1→36.7 |

## Runner 边界修复

原 runner 将 prompt 限制为 context_len−prefill_ar_len。最后一个完整 chunk 写回时，先将 full-attention cache 从 prefill 布局扩展到已分配的 decode 布局，并保留 128-token 输出的读取步幅。GDN 完整 state 更新方式不变。
`backends/qualcomm/tests/test_llama_swa.py::test_native_kv_manager` 已通过，包含双头 K/V、最终 chunk 写回、token 选择和 ASan/UBSan 检查。

各目录的 records.json 保存每次结果与原始计时；*.log、*.command.txt 保存 runner 日志和确切命令。生成文本仅用于检查执行完成；本次测性能，长生成忽略 EOS，不作为质量评价。

---

<a id="record-16"></a>

## 记录 16：performance_regression_20261014 / REPORT.md

原始记录位置：`llama-output/qwen3_5-0_8b-layer-analysis/full_kz_w8_gptq_prefill128/qk_native_rms/performance_regression_20261014/REPORT.md`。

# Qwen3.5-0.8B 性能回退定位与修复

日期：2026-10-14。设备：SM8750，serial `3B1F57E9B8L74X5D`，QNN 2.49.40.260810，Hexagon v79。

## 1. 已交付结果

采用现有校准、GPTQ 权重和精确 VTCM GDN，实现并导出了修复版完整 hybrid PTE：

- 模型容量 4096；prefill AR=128，decode AR=1；24 层 decoder，含 FFN、final norm、LM head。
- 短 decode：输入 16 token，生成 64 token，其中首 token 属于 prompt forward，后续 63 次 decode 计时。
- 同一短生成口径：问题版 **7.604 token/s → 修复版 37.861 token/s**。物理容量分别为旧 4097 和修复后的 4096，二者差异属于本次修复。
- 另外两次修复版短生成：41.584 和 41.888 token/s。综合本次短测为 **约 38–42 token/s**，不是长达 4K token 的生成均值。
- 完整 128-token prompt：144 ms，**888.9 token/s**。
- 同一 4K PTE，输入 1024 / 3840 token、输出 1 token：分别 **879.0 / 862.0 token/s**。
- 同一组留出窗口的 PPL：**50.0762 → 50.0248**，未见精度回退。
- 所有生成测试均为空图片列表；vision encoder 被 runner 加载，但未进行图片编码。加载耗时不计入 prefill/decode 指标。

不再进行已取消的 1K–8K 长生成扫测。本报告中的短生成、两个 prefill 点及 PPL 是针对修复的验证。

## 2. 主要根因：历史 KV 被重复并反复转换

这个模型并非每层都是 linear attention。配置共有 18 个 GDN 层、6 个 full-attention 层，每四层一次 full attention。

### 原图做法

full attention 的 2 个 KV head 经 `unsqueeze → expand → reshape` 重复成 8 个 query head；然后，把整段历史 cache 从 UINT16 非对称网格转换到 MatMul 需要的对称 INT16 网格。每个 token 都重复做，历史越长，代价越大。

4K 容量下，一份重复后的 K 或 V 是：

`1 × 8 × 4096 × 256 × 2 bytes = 16 MiB`

6 层的 K+V，单计这类重复张量的体积就是 192 MiB；转换、读写、中间布局变换还会增加实际流量。原始 2-head K/V 总体积只需 48 MiB。这里的计算是张量体积估算，不是 DDR 硬件计数器测量。

详细 QNN profile 中，第一组 V-cache 转换 `aten__to_copy_default_56` 从约 1.37M cycles（1K）升到约 27M cycles（4K）；GDN 单步核没有同量级增长。问题主要发生在全注意力的 cache 路径。

### 修复做法

1. 把 query 从 `[B,8,T,D]` 重排为 `[B,2,4*T,D]`，K/V 保留 2 head，MatMul 后恢复原 query-head 形状。
2. 新 token 写 cache 前，只对新条目做原有校准转换。
3. cache 存储采用原来 MatMul RHS 的校准 scale，继续通过 UINT16 接口传输；zero point=32768 对应有符号中心的偏移。
4. 下一步直接消费已转换的历史条目，避免对整段历史执行浮点比例换算。
5. prefill、decode 和 runner 常量方法使用相同的新 FA-cache 编码；新 cache 使用对应 zero point 初始化。

**scale 来自现有校准图，未写死。** 32768 是 UINT16 与 INT16 编码中心的类型映射。12 个 FA K/V cache 编码改变；36 个 GDN state/conv 编码、全部权重码值、Q/K native RMSNorm、精确整数 GDN 内核均保留。

对应实现：`examples/qualcomm/oss_scripts/llama/quantize/generation.py::optimize_grouped_attention`。

## 3. 两项额外开销

### 3.1 上下文被错误设置成 n+1，破坏对齐

此前为了容纳输入 n、输出 1，我把物理容量设成 1025 / 2049 / 4097 等。这样虽然绕过了 runner 的长度限制，却导致 QNN 的 full-attention 布局和融合路径变差。这是测试脚本引入的问题。

控制变量短生成：相同权重、精确 GDN、同一 runner、相同 16-token prompt / 64-token 输出：

- capacity=1024：39.57 / 42.67 ms/token。
- capacity=1025：54.05 / 54.08 ms/token。

因此，不能通过加 1 构造静态 context。当前修复 PTE 使用 4096；通用扫测脚本也已改为按 128 对齐分配容量。容量选取仍须包含计划输出空间。

### 3.2 prefill 把 128 个 token 都送入巨大 LM head

词表为 248320。原输出 `[1,128,248320] UINT16` 是 **60.625 MiB**；普通生成只消费一个有效 token 的 logits。

修复在 LM head **前**选择 hidden 的 token 行，再做投影；生成输出变成 `[1,1,248320]`，约 **0.474 MiB**。因此既减少 LM head 计算，也减少输出搬运。它不是仅在完整 logits 后切片。

prefill 最后多一个 `INT32[1] last_token_index` 输入；runner 为每块传入实际有效 token 的索引，cache 索引和共享缓冲大小同步修正。校准和需要所有 token logits 的 PPL 图保留原接口。

对应实现：`generation.py::select_generation_logits`，以及 `runner/multimodal_runner/multimodal_prompt_processor.cpp`、`runner/utils.h`。

### 3.3 runtime 未开启共享缓冲

原完整 PTE 为 `shared_buffer=False`。本次启用 QNN shared buffer，并在 runner 传入 `--shared_buffer`，降低 host/RPC 侧的复制和注册开销。

将原 PTE 的 runtime compile spec 改为 shared 后，检查全部 backend payload 的 SHA256 相同；已编译图和权重未变。修复版 plain/shared 的 49 个输出在 decode、prefill 样本上全部逐位相同。

仅在命令行增加 `--shared_buffer` 不足以保证生效，PTE 侧也必须启用。

## 4. 分阶段耗时

以下是相同真实 hidden 样本、逻辑空 cache、warmup=2 后 5 次 forward 的均值。是完整 decoder 图执行耗时，不是生成 token/s；冷 cache 指内容为空，不代表减少静态计算范围。

| 配置 | decode forward ms | prefill 128 forward ms |
|---|---:|---:|
| 原图：容量4097，全部prompt logits | 127.640 | 206.276 |
| 对齐4096，只投影所选token | 105.955 | 164.003 |
| 上项加共享缓冲 | 90.765 | 145.726 |
| 对齐、所选token、GQA/cache转换 | 38.802 | 161.575 |
| 全部修复 | 21.745 | 142.137 |

各行来自先后执行的短测，包含设备波动；不能把相邻差值解释为严格可加的独立硬件成本。profile=2 的逐算子追踪有明显开销，只用于找热点，表格采用 profile=0。

旧版精度较低的 GDN 与当前精确 GDN，在同一原始 1K PTE 的短生成对照中分别为 23.46 / 22.90 token/s，差约 2.5%。不足以解释这次数量级明显更大的回退，故本次没有换回有 PPL 问题的旧核。

## 5. 长度增长验证

同一修复版 capacity=4096，prompt 为 128 的整数倍，输出 1 token，包含 embedding、所有 decoder 块和 runner cache 提交：

| 输入 tokens | prefill ms | tokens/s | 每 128 tokens ms |
|---:|---:|---:|---:|
| 128 | 144 | 888.89 | 144.00 |
| 1024 | 1165 | 878.97 | 145.625 |
| 3840 | 4455 | 861.95 | 148.500 |

1024→3840，每块耗时增加约 1.97%。这里控制了编译容量，避免把容量变化和实际 prompt 长度混在一起。未声称完全消除 full attention 的理论复杂度，也未重新测试 8K。

## 6. 数值验证

- 小图参考测试：8 次非零 cache 连续更新，原重复-KV 图与分组图 attention 最大绝对误差为 0；覆盖 cache 写回编码。
- 生成行选择测试：验证第 0、中间、最后行的结果，cache 输出保持相同。
- 设备单步 decode 样本：logits 和全部 36 个 GDN state/conv 输出与对齐后的旧图逐位相同。
- 设备 prefill128 样本：改变了 MatMul 形状，设备结果存在差异；相对旧图的 GDN state SQNR 最低约 29.0 dB，最后一个 GDN 层 state 约 39.47 dB。该指标是优化前后差异，不是相对 FP32 的精度。
- PPL 用原来的 8 个独立、冷 cache、128-token 留出窗口，共 1016 个 next-token 目标。用所选行 PTE 依次选择 0..126，未把上一轮输出 cache 回写，因此每行仍属于同一次 causal prefill 的语义。
- 原 PPL 50.076208，新 PPL 50.024808。它是本地留出窗口的回归检查，不是标准全量 WikiText PPL，也不证明 8K state 质量。

Python 两项语义测试通过；native KV-manager 边界/索引测试在 ASAN+UBSAN 下通过；Android `qnn_multimodal_runner` 已重新构建。新增 Python 文件 ruff 通过，`git diff --check` 通过。

## 7. 官方数字的设备口径

用户给出的 1496.44 / 18.62 token/s，对应官方 Qwen3.5-2B、4096、NPU、**Snapdragon X2 Elite CRD**，后端为 `geniex_llamacpp`、Q4_0。官方 Galaxy S25 / Dragonwing Q-8750 的对应 NPU 数字为 593.10 / 22.64 token/s。来源：[Qualcomm 官方 perf.yaml](https://raw.githubusercontent.com/qualcomm/ai-hub-models/main/src/qai_hub_models/models/qwen3_5_2b/perf.yaml)。

本地是 SM8750、0.8B、QNN/ExecuTorch、混合 W4/W8，与上述不同模型和配置不能直接计算框架性能倍数。本地旧版回退已通过本地控制变量复现并修复。

## 8. 使用产物与重建

生成 PTE：`generation_c4096_gqa/hybrid_llama_qnn_shared.pte`。

- SHA256：`e084f2c53d6a3ca355dc8910cd19a89c4bbbb43ed00053f1f8fe3eacbbf1ea4c`。
- 精确 GDN DSP SO SHA256：`490d117374851b0e5b6fc2d3b69f264b41b703c5bd9c79ad4326b8c35d435d9e`。
- 设备目录：`/data/local/tmp/executorch/qwen35_generation_c4096_gqa_shared`。
- 编译的 op-package 路径位于 `/data/local/tmp/executorch/qwen35_generation_c4096_gqa`，两处均已部署同一个精确 SO；手动迁移时应一并配置。
- `run_text.sh` 可直接运行已部署版本的短文本生成。
- 原始校准 QDQ、全-token logits 模型均保留。本次没有修改用户 `mybuild.sh`。

重建当前完整模型（使用此前保存的 CUDA 校准 QDQ；一次只编译一个模型）：

```bash
conda activate et_npu
source env.sh
export CUDA_VISIBLE_DEVICES=0
python llama-output/qwen3_5-0_8b-layer-analysis/export_generation_performance_fix.py --capacity 4096 --grouped-query
python llama-output/qwen3_5-0_8b-layer-analysis/enable_generation_shared_buffer.py \
  llama-output/qwen3_5-0_8b-layer-analysis/full_kz_w8_gptq_prefill128/qk_native_rms/performance_regression_20261014/generation_c4096_gqa/hybrid_llama_qnn.pte \
  llama-output/qwen3_5-0_8b-layer-analysis/full_kz_w8_gptq_prefill128/qk_native_rms/performance_regression_20261014/generation_c4096_gqa/hybrid_llama_qnn_shared.pte
```

优化已用于本次交付 PTE；普通 `llama.py` 导出入口尚未默认调用这两个后校准变换，直接重跑旧命令不会自动得到相同图。所选行接口需要本次新编译的 runner；prefill 验证均使用完整 128-token 块，未覆盖非整块 GDN padding 的状态语义。

## 9. 原始证据

- `summary.json`：汇总。
- `paired_generation_4k/summary.json`：同 prompt、同输出数的前后端到端对照。
- `generation_validation/summary.json`：短 KV/hybrid 输出文本及时间。
- `prefill_scaling_validation/summary.json`：同 4K 容量的多块提交验证。
- `gqa_prefill_ppl/summary.json`：1016 个目标的 PPL 与逐 token NLL。
- `gqa_numerical_comparison.json`：逐层 GDN state/conv 优化前后差异。
- 各 `*_kv_forward` / `*_prefill_forward` 目录：无 profile 时间、basic profile、详细算子 cycles 和 raw 输出。

---

<a id="record-17"></a>

## 记录 17：gdn_128_after_l2fix / REPORT.md

原始记录位置：`llama-output/qwen3_5-0_8b-layer-analysis/gdn_128_after_l2fix/REPORT.md`。

# Qwen3.5 单层 GDN / 128-token 输入、输出和 state 误差报告

## 结论

原生 RMSNorm 等价 L2 norm 修复后，8 段评估文本的 128-token HTP attention output 整体 SQNR 为 **30.026 dB**，第 128 步 recurrent state 整体 SQNR 为 **21.384 dB**。

state 在第一步就有明显误差，此后在约 21–24 dB 附近波动；观察到的误差没有在第 128 步突然发散。逐路替换实验显示 key 路径对 state 影响最大。

同时发现并修复了多输出 custom op 的量化参数串用问题：attention 和 state 在 QNN tensor wrapper 中曾共用了 state 编码。修复后 attention 使用自己的校准编码，state 数值保持不变。

## 测试范围

- 只加载 Qwen3.5-0.8B layer 0；前面无其他层，初始 conv/state 为零。
- 保留 `layers.0` 模块路径，确认正式 recipe 命中：qkv/b 为 block-W4，a/causal conv/out_proj 为 channel-W8；FFN 配置未改。
- 8 段 WikiText 来源文本，每段前 128 token 校准，后 128 token 评估。两个窗口独立零 state 起步；不是连续 256-token cache 测试。
- 这里的 attention output 是 `deltanet_core` 的原始输出 `[1,128,16,128]`，位于 z gating、output norm 和 out_proj 之前。state 是 `[1,16,128,128]`。
- 同一输入对照固定从正式单层校准图中提取的 scale/zero point，不在小算子图上重新拟合。`FixedQParamsObserver` 在探针中只是保留这些校准结果；数字不是硬编码。
- 设备 `3B1F57E9B8L74X5D`，SM8750/v79；QNN SDK 2.49.40.260810；映射 `DeltaNetCoreVtcm`。
- SQNR=10log10(sum(reference²)/sum(error²))。多样本结果先累加 signal/noise power，再计算 dB，不做 dB 的简单平均。

128-token trace（历史产物已清理）

## 1. FP32 min/max → 量化范围 → 饱和比例

饱和比例定义为原 FP32 值落在校准可表达范围之外的元素比例；不是把边界 code 的出现频率当作饱和。表中的 state 范围和饱和比例覆盖所有中间步骤。

| tensor | FP32 min | FP32 max | scale | zero point | quant real min | quant real max | out-of-range % |
|---|---:|---:|---:|---:|---:|---:|---:|
| query | -0.08669337 | 0.08777337 | 2.6439718e-06 | 32321 | -0.08545581 | 0.08781688 | 0.00004768 |
| key | -0.9990212 | 0.985025 | 3.0278003e-05 | 32925 | -0.9969032 | 0.9873657 | 0.00009537 |
| value | -0.2784646 | 5.582816 | 8.8903398e-05 | 3132 | -0.2784454 | 5.547839 | 0.23279190 |
| decay | 9.007723e-05 | 1 | 1.5259022e-05 | 0 | 0 | 1 | 0.00000000 |
| beta | 0.02051976 | 0.9988014 | 1.5242311e-05 | 0 | 0 | 0.9989048 | 0.00000000 |
| state_output | -3.347821 | 2.887232 | 8.8138688e-05 | 36330 | -3.202079 | 2.57409 | 0.00003465 |
| output | -0.2639576 | 0.1981516 | 6.5670429e-06 | 38583 | -0.2533762 | 0.1769949 | 0.00014305 |

value 的约 0.23% 越界主要需要结合幅度解读：SiLU 的负极小值贴近校准负下界。state 的少量越界仅影响极少数坐标，无法解释整体约 21 dB 的 state 误差；纯边界量化对照和 key 替换实验提供了更直接的归因。

## 2. state 输入 SQNR → state 输出 SQNR → attention output SQNR

以下为评估样本 0 的实际 HTP 轨迹，每一行来自真实前缀执行后的 state。state input 是该 token 更新前的 state，state output 是更新后的 state。

| token | state input SQNR / dB | state output SQNR / dB | attention output SQNR / dB |
|---:|---:|---:|---:|
| 1 | 初始全零，误差 0 | 23.029 | 24.687 |
| 2 | 23.029 | 22.403 | 29.469 |
| 4 | 21.604 | 22.271 | 27.653 |
| 8 | 23.150 | 23.421 | 30.620 |
| 16 | 23.438 | 23.597 | 30.259 |
| 32 | 23.036 | 22.972 | 31.317 |
| 64 | 22.618 | 22.617 | 34.312 |
| 96 | 22.550 | 22.643 | 36.340 |
| 128 | 21.841 | 21.935 | 32.404 |

逐 token 的 SQNR 会随信号幅度变化，不能期待单调下降。例如早期 attention 幅度小，固定量化步长会造成更低的相对 SQNR；后期信号变大，即便 RMS error 增大，SQNR 也可能提高。原始 CSV 同时保存范围、RMS error 和 max error。

### 每步 state 如何观测

T=1 VTCM 小图曾在 QNN host Graph Optimizations 阶段 segfault；没有使用那次失败的数据。改为复用同一个已经通过的 T=128 PTE：第 p 次运行保留前 p 个真实 token，其后 query/key/value/beta=0、decay=1。尾部是恒等 state 更新。当前实际校准 decay 可以精确编码 1、beta 可以精确编码 0；没有改 scale。

128 次 prefix 的有效 attention 输出全部与完整 128-token prefill 逐元素一致；dummy attention 全零；prefix=128 的 state 也逐元素一致，最大差值都为 0。INT16 更新系数在 decay=1 时恰好是 8192/2^13，尾部不会增加 state 舍入误差。这保证了取样对应同一个 prefill kernel 的真实递推。

## 3. 上游误差与 GDN 内部误差拆分

| 对照 | 128-token attention output SQNR / dB | 最终 state SQNR / dB |
|---|---:|---:|
| 量化投影/预处理输入，FP32 recurrence，无 core IO QDQ | 30.535 | 21.453 |
| 原 FP32 core 输入，仅 core IO QDQ，内部 state FP32 | 50.196 | 60.329 |
| 正式 PT2E CPU QDQ，内部 state FP32 | 30.540 | 21.502 |
| 量化输入，每步 INT16 state 舍入，精确 FP32 归约 | 30.455 | 21.401 |
| HTP VTCM，与原 FP32 layer 的 core 输入参考对比 | 30.026 | 21.384 |
| HTP VTCM，与相同 core 输入的 FP32 recurrence 对比 | 40.665 | 36.312 |

`device_same_inputs` 包括输出量化、每步 state 舍入/饱和、DSP 整数和浮点近似带来的全部误差。这些因素的影响不能通过两个 dB 数字直接相减来分配。主 kernel 的 state/q/k 归约仍有右移 4 位、乘积尺度补回 256，以及 13-bit 更新系数。

进一步将 DSP 与同一输入、同样每步 INT16 state 舍入和相同输出编码的精确 FP32 归约对照：attention SQNR `41.848 dB`，最终 state `42.548 dB`。这些剩余误差来自实现差异，当前测试尚未把右移归约和更新系数量化进一步分开。

### q/k 已修复的范围

上轮修复并验证的是 L2 normalization 本身。本轮将 W4 投影、causal conv 和预处理都算进输入误差，core 输入 query/key 的整体 SQNR 约 22.60/21.06 dB；归一化修复不会消除它前面已经形成的投影误差。

## 4. 一路输入替换实验：哪个输入影响最大

每个实验都从原 FP32 core 输入开始，只把所列的一路替换为量化单层生成的对应输入，其余输入和 recurrence 都保持 FP32。因此该表直接测量各输入带来的传播误差。

| 仅替换输入 | attention output SQNR / dB | 第 128 步 state SQNR / dB |
|---|---:|---:|
| query | 33.947 | 完全一致 |
| key | 35.423 | 22.318 |
| value | 37.591 | 30.901 |
| decay | 75.066 | 69.907 |
| beta | 45.382 | 37.830 |

key 对 state 的影响最大；query 只用于读出 attention，不参与 state 更新，所以只替换 query 时 state 完全一致。value 次之，beta 再次，decay 的误差很小。

## 5. 发现并修复的多输出编码错误

正式 PT2E QDQ 图里，attention scale/zero point = `6.567042873939499e-6 / 38583`，state = `8.813868771540001e-5 / 36330`。它们校准正确且不同。但是 AnnotateQuantAttrs 在 tuple 的父节点上顺序写属性，最后留下 state 的属性；CustomOp builder 为两个输出都用父节点构造 TensorWrapper，导致 attention 实际使用了 state 的编码。

从设备原始输出的整数格点也确认了这一点：修复前 attention 恰好落在 state scale 的格点上。修复后每个输出读取自己的 getitem 量化元数据；TensorWrapper 名字、index、形状保持一致。state 输出编码仍与输入 cache 编码一致。

修复前后总体 attention SQNR：`29.955 → 30.026 dB`。这次修复对小幅值的早期 token 更明显；总体仍由上游投影误差主导，因此整体改善有限。state 数值保持不变。

代码：`backends/qualcomm/builders/op_custom_op.py`、`backends/qualcomm/builders/node_visitor.py`。增加两个实际 QNN wrapper 回归测试：两个输出不同 scale/zero point，以及量化 attention + FP16 state；连同 L2 tests 共 15 passed。未改 VTCM C++ 内核或 FFN recipe。

## 6. 产物和复现

- `input_metrics.csv`：8 样本 ×128 token 的 q/k/v/decay/beta 输入指标。
- `token_metrics.csv`：逐 token state_in/state_out/attention 及内部 stages，逐样本误差和幅度。
- `token_summary.csv`：按 token 汇总的 SQNR。
- `ablation_metrics.csv`、`ablation_summary.csv`：逐路输入替换结果。
- `encodings.json`、`core_inputs.pt`：正式校准参数及 FP32/量化输入。
- `range_stats.json`：精确越界元素数和比例；`dsp_vs_ideal_int16.json`：每步 state 同精度的 DSP 对照。
- `device_full128/gdn_vtcm_128.pte`：128-token VTCM 单算子图。
- `device_prefix_states/prefix_outputs`：128 个真实 state 检查点及 attention 输出。
- `device_prefix_states/consistency.json`：prefix 与完整 prefill 一致性验证。
- `before_output_encoding_fix`：修复前设备数据和 CSV，供对照。
- `gdn_128_trace.png/.svg`：曲线图。

```bash
conda activate et_npu
source env.sh
python llama-output/qwen3_5-0_8b-layer-analysis/diagnose_gdn_128.py --device 3B1F57E9B8L74X5D
python llama-output/qwen3_5-0_8b-layer-analysis/gdn_input_ablation.py
python llama-output/qwen3_5-0_8b-layer-analysis/gdn_range_check.py
python llama-output/qwen3_5-0_8b-layer-analysis/report_gdn_128.py
```

这些结果限于 layer 0、零初始 state、当前八段评估文本和当前校准范围。尚未测试跨 1024-token 窗口的 cache 累积，也未复测全模型 PPL。下一步应优先检查 key 上游的 W4 projection/causal conv 误差，再单独量化 DSP 缩位归约的影响。

---

<a id="record-18"></a>

## 记录 18：gdn_128_key_w8_rest_w4 / REPORT.md

原始记录位置：`llama-output/qwen3_5-0_8b-layer-analysis/gdn_128_key_w8_rest_w4/REPORT.md`。

# GDN 128-token：key W8，q/v/z/a/b 混合精度实验

## 结果

key 输入 SQNR：21.065 → 39.816 dB。

## 配置和测试范围

- 原基线：合并 qkv_z W4、a W8、b W4。
- 本次：key W8、合并 q/v/z W4、a W4、b W4。
- W4 为 block size=32；W8 为 per-channel；激活均 A16。
- causal conv/out_proj 仍为 W8A16；conv cache/recurrent state 仍为 UINT16。
- 沿用相同 layer 0、8 段 WikiText、前 128-token 校准/后 128-token 评估，初始 cache 为零。
- 每个配置独立校准，再从单层 QDQ 图读取实际 scale/zero point。没有沿用基线硬填的新参数。
- 拆分权重前后 FP32 projection 输出已核对；评估 core FP32 输入也与基线逐项核对。
- CPU QDQ 覆盖完整上游投影和预处理；设备 PTE 直接测 GDN VTCM core，投影/conv 的输入由 CPU QDQ 提供。
- attention output 指 GDN core 输出，位于 z gating/out_proj 之前；未测全模型 PPL，也未测投影的设备性能。
- 此实验在诊断脚本中实现，正式全模型导出默认配置未改。

## Core 输入 SQNR

| 输入 | 基线 / dB | 本次 / dB |
|---|---:|---:|
| query | 22.595 | 22.595 |
| key | 21.065 | 39.816 |
| value | 32.205 | 32.205 |
| decay | 67.498 | 43.246 |
| beta | 30.645 | 30.645 |

## GDN 输出 SQNR

attention 合并 8×128 token 的 signal/noise power；state 合并 8 个最终 cache。

| 对照 | 基线 attention / dB | 本次 attention / dB | 基线 state / dB | 本次 state / dB |
|---|---:|---:|---:|---:|
| core_boundary_only | 50.196 | 50.196 | 60.329 | 60.329 |
| device_same_inputs | 40.665 | 40.839 | 36.312 | 37.517 |
| device_total | 30.026 | 31.717 | 21.384 | 29.315 |
| int16_state_each_step | 30.455 | 32.258 | 21.401 | 29.385 |
| production_cpu_qdq | 30.540 | 32.312 | 21.502 | 29.969 |
| upstream_fp | 30.535 | 32.353 | 21.453 | 29.847 |

实际 VTCM core 的最终 state 改善 7.932 dB，attention 改善 1.691 dB。

key W8 明显改善 state；q 的误差维持原值，因此 attention 的改善幅度小于 state。a 改为 W4 使 decay 精度下降，但本配置的整体 state 仍有明显改善。此对照同时改变 key 和 a，尚未单独测量 a 改为 W4 对最终输出的影响。

## 实际命中的 recipe

| 模块 | 精度 | 粒度 |
|---|---|---|
| `layers.0.attention.in_proj_qkv_z.in_proj_qvz.conv` | use_16a4w_block | PER_BLOCK |
| `layers.0.attention.in_proj_qkv_z.in_proj_k.conv` | use_16a8w | PER_CHANNEL |
| `layers.0.attention.in_proj_b.conv` | use_16a4w_block | PER_BLOCK |
| `layers.0.attention.in_proj_a.conv` | use_16a4w_block | PER_BLOCK |
| `layers.0.attention.causal_conv2d` | use_16a8w | PER_CHANNEL |
| `layers.0.attention.out_proj.conv` | use_16a8w | PER_CHANNEL |
| `layers.0.feed_forward.w1_conv` | use_16a4w_block | PER_BLOCK |
| `layers.0.feed_forward.w3_conv` | use_16a4w_block | PER_BLOCK |
| `layers.0.feed_forward.w2_conv` | use_16a8w | PER_CHANNEL |

## 权重存储开销和复现

当前模型 key 投影为 2048×1024：理想打包 W4 为 1 MiB、W8 为 2 MiB；升为 W8 时每个 GDN 层增加约 1 MiB 权重，另有 scale 元数据。a 为 16×1024，从 W8 改 W4 只节约约 8 KiB。q/v/z 继续合并为 W4 投影。

```bash
conda activate et_npu
source env.sh
python -u llama-output/qwen3_5-0_8b-layer-analysis/try_gdn_key_w8.py \
  --key-bits 8 --a-bits 4 \
  --out llama-output/qwen3_5-0_8b-layer-analysis/gdn_128_key_w8_rest_w4 --device 3B1F57E9B8L74X5D
```

## 校准编码和逐 token 数据

校准后的 GDN 参数见 `encodings.json`；逐 token 输入、state、attention 指标见 `input_metrics.csv` / `token_metrics.csv`。

测试中非有限值计数：0。

比较曲线（历史产物已清理）

---

<a id="record-19"></a>

## 记录 19：kz_gate_drop / REPORT.md

原始记录位置：`llama-output/qwen3_5-0_8b-layer-analysis/kz_gate_drop/REPORT.md`。

# norm/z 门控 SQNR 下降：定位、修复与设备对照

后续已按性能要求恢复默认快速 VTCM 归约，并完成所有 Linear W4 的独立测试，
见 [回退报告](README_QWEN35_TEST_RECORDS.md#record-06)。下文保留回退前的完整精度实验数据。

日期：2026-10-13；模型：本地 Qwen3.5-0.8B；层：layer 0；设备：SM8750 / Hexagon v79；QNN SDK：2.49.40.260810。

## 1. 已验证的结果

原来的 21.04 dB 来自只有 K 为 W8、z 为 W4 的版本。K/z 合并为 W8 后为 23.77 dB；本次完整精度 HVX 归约修复后为 28.17 dB；再针对 V 的 W4 权重做 GPTQ 后为 29.61 dB。后两项都实际运行了设备 PTE。

主要发现：

- z W8 后，z 分支已不是本次大幅下降的主要来源；门控误差主要来自 GDN 读出。
- DSP 快速归约在乘法前丢掉 Q/K/state 的 4 个低位；乘 256 不能恢复信息。完整中心化值转 FP32 后以 HVX 累加，设备门控改善 4.39 dB。
- 剩余误差的单输入修复实验定位到 V，而非 Q。保持 V 为 W4，在实际已有 block-32 量化网格上用 GPTQ 做误差补偿，门控再改善 1.44 dB。
- 本轮验证范围是单层 128-token prefill；尚未重新测量完整模型 PPL。

## 2. 精度配置与测试范围

所有列均使用相同的原始模型、文本和 FP32 参考。K/z 合并 W8A16（4096×1024），Q/V/A/B 合并 W4A16（4128×1024），W4 分块为 32，code 范围 [-7,7]。FFN gate/up W4，down W8；out_proj、causal conv W8。激活、conv cache、recurrent state 保持 UINT16。

校准使用 WikiText 的 8 个独立 128-token 窗口，共 1024 token；评估使用各文本后续 128-token 窗口，共 1024 token。评估窗口单独从零 cache 开始，未将校准窗口的 state 传入评估窗口。

GPTQ 只更新 merged Q/V/A/B 中 rows [2048:4096] 的 V 整数 code；Q/A/B 的 code、K/z 的权重、所有权重 scale/zero point、所有激活和 cache 的校准参数均保持相同。V 为 W4，没有改为 W8。优化在 CUDA 上执行，评估 token 不参与 Hessian 估计。

DSP 修复对照复用相同 PTE，只替换运行时 op package。GPTQ 变更了权重，因此单独生成了完整单层 PTE。两个 PTE 都包含投影、卷积、GDN、门控、out_proj、残差、FFN，全部在 QNN delegate 执行；额外输出用于误差检查。

## 3. 设备各阶段 SQNR（相对原始 FP32）

| 阶段 | 原快速归约 | 完整精度归约 | 完整精度 + V GPTQ |
|---|---:|---:|---:|
| Q 输入 | 22.145 dB | 22.145 dB | 22.145 dB |
| K 输入 | 39.740 dB | 39.740 dB | 39.740 dB |
| V 输入 | 31.437 dB | 31.437 dB | 34.534 dB |
| z 输入 | 43.040 dB | 43.040 dB | 43.040 dB |
| GDN 输出 / norm 输入 | 31.601 dB | 32.146 dB | 32.789 dB |
| 最终 recurrent state | 28.658 dB | 28.725 dB | 31.152 dB |
| RMSNorm × gamma | 30.428 dB | 33.736 dB | 35.521 dB |
| norm/z 门控输出 | 23.773 dB | 28.167 dB | 29.609 dB |
| out_proj 后 attention 输出 | 20.034 dB | 24.662 dB | 25.756 dB |
| FFN 后完整层 hidden state | 17.589 dB | 20.498 dB | 21.002 dB |

## 4. 为什么输入看起来很高，门控后却低

两个门控输入分别是 u = gamma × RMSNorm(x) 和 g = SiLU(z)，输出 y = u × g。z 输入的 43.04 dB 不能代表另一分支 u 的精度。原快速版本的 x 为 31.60 dB，u 为 30.43 dB，而 z 为 43.04 dB。

忽略很小的 z 误差时，输出误差约为 g × (u_device - u_ref)。门控后的 SQNR 是：

```text
SQNR(y) ≈ 10 log10[sum(g_ref² × u_ref²) / sum(g_ref² × error_u²)]
```

g 对不同 token/head/channel 的加权不同。如果原来误差较大的位置恰好有较大的 |g|，乘法后的整体 SQNR 会降低。RMSNorm 对每个 head 独立归一化，也会重新改变 head 之间的权重；eps=1e-6 时，小幅值输入的增益最高接近 1000。这并不意味着可以忽略下降：本轮确实找到并修复了其中的额外 DSP 数值损失。

对原快速版本做严格代数分解（使用实际设备输出），四项之和等于总误差：

```text
error_y = (u_dev-u_ref)*g_ref
          + u_ref*(g_dev-g_ref)
          + (u_dev-u_ref)*(g_dev-g_ref)
          + (y_dev-u_dev*g_dev)
```

| 独立误差项，用同一输出信号功率计算 | 等效 SQNR |
|---|---:|
| RMS/GDN 分支 | 23.814 dB |
| z 分支 | 44.482 dB |
| 交叉项 | 67.515 dB |
| 乘法/输出 QDQ 的额外残差 | 72.911 dB |

各项存在相关性，不能将其噪声功率简单相加当作百分比。门控乘法本身的数值实现并未造成主要损失。

SQNR 的 dB 是对数尺度，不等于有效位数减半。例如门控 23.77→29.61 dB，对应归一化 RMS 误差约 6.48%→3.31%，在同一输出信号上噪声功率约下降 3.83 倍。

## 5. 第一处修复：GDN 归约保留完整低位

历史快速实现的 state 乘法输入近似为：

```text
q12 = (q_code - q_zero) >> 4
k12 = (k_code - k_zero) >> 4
s12 = (s_code >> 4) - (s_zero >> 4)
Rq ≈ 256 × sum(q12 * s12)
Rk ≈ 256 × sum(k12 * s12)
```

state 的移位与中心化顺序还与 Q/K 不同。这是一种额外近似，既不是校准产生的 scale，也不是 UINT16 的必然要求。仅把 shift 改成 0 会让 int32 乘积累加存在溢出风险，因此不能这样修。

当前默认实现：先将 U16 展开为 int32 并减完整 zero point，再将完整中心化值转为 FP32，沿 K 用 HVX SIMD 累加 Rq/Rk；累加器仍跨 K 留在寄存器，只在末尾写一次 scratch。state 仍为 UINT16，VTCM 驻留与 head 并行分工保持原结构；update 仍逐 token 执行。

修复位置为 `deltanetQhpiHvxReduceStateU16`，其输出与 P2 helper 的 scratch 类型联动修改为 FP32，尺度补偿改为 1。完整 tile 不再丢低位；非整 tile 在完整精度配置下使用 int64 保底计算。`DELTANET_U16_FULL_PRECISION_REDUCTION=0` 可编译历史缩位分支，用于复现旧测量。

### 同一设备输入的数学 recurrence 对照

| recurrence 方式 | GDN 输出 | RMS × gamma | 门控 |
|---|---:|---:|---:|
| 历史 DSP | 31.601 | 30.428 | 23.773 |
| FP32 内部 state，输入/输出用原 A16 网格 | 32.154 | 34.117 | 28.617 |
| 每步 UINT16 state 舍入，完整精度数学 recurrence | 32.109 | 33.746 | 28.181 |

修复后设备门控 28.167 dB，已接近每步 UINT16 舍入的数学对照 28.181 dB。由此可区分 DSP 近似损失与本轮 PTQ 本身的损失。

## 6. 第二处定位：V 输入的因果替换实验

以下使用历史 PTE 的实际设备 core 输入，完整精度 recurrence，每一步仍量化 state 为 UINT16，输出仍走同一 A16 编码和门控。每次只把指定 core 输入换为原始 FP32 参考，其他输入和 z 保持设备值；这是定位对照，不能当作真实可部署的精度配置。

| 替换项 | 门控 SQNR |
|---|---:|
| 不替换 | 28.181 dB |
| 只恢复 Q | 28.431 dB |
| 只恢复 K | 28.187 dB |
| 只恢复 V | 34.376 dB |
| 只恢复 decay | 28.192 dB |
| 只恢复 beta | 28.307 dB |
| 恢复所有 core 输入，state/输出仍量化 | 36.643 dB |

尽管 Q 的整体 SQNR 只有 22.14 dB，它并非本轮门控下降的最大因素。只恢复 V 可提升约 6.20 dB；实际 GPTQ 的改善也验证了 W4 V 权重误差的重要性。V 的剩余误差还包含投影/卷积激活 QDQ 与 state 递推量化，不能全部归因于权重。

## 7. GPTQ 的实际结果与可复现入口

GPTQ 使用归一化投影输入构建输入 Hessian，对 W4 V 权重逐列舍入并向剩余列补偿误差，damping=0.01、处理 block=128；量化 group=32 与原图一致，scale 取已量化图 `_scale0` 的 V 行，不重新拟合。方法参考 [GPTQ 原始实现](https://github.com/IST-DASLab/gptq/blob/main/gptq.py)。

CPU QDQ 门控 28.774→30.888 dB；设备在完整精度归约下 28.167→29.609 dB。

通用实现：`examples/qualcomm/oss_scripts/llama/quantize/gptq.py`。单层实验脚本：`../optimize_gdn_value_gptq.py`。保存的 V code 和固定 scale 在 `gptq_value/value_weight_patch.pt`；可运行 PTE 为 `gptq_value/merged_linear_attention_ffn_128.pte`。常规全模型导出入口尚未自动应用这份单层 GPTQ patch，不能把单层实验结果当作全模型已部署结果。

```bash
source /home/hunter/anaconda3/etc/profile.d/conda.sh
conda activate et_npu
source env.sh
export CUDA_VISIBLE_DEVICES=0
python llama-output/qwen3_5-0_8b-layer-analysis/optimize_gdn_value_gptq.py --device 3B1F57E9B8L74X5D
```

## 8. FP32 范围、实际量化范围和饱和比例

下面来自相同评估集的 FP32 检查点和原图校准参数。比例按落在可表示实数范围之外计算；SiLU 最小值附近极小的 FP32/量化边界差也计入其中，不能仅凭比例推断误差能量。

| 张量 | FP32 min/max | 量化实数范围 | 范围外比例 |
|---|---|---|---:|
| core_input_state | [0, 0] | [-3.2020785, 2.5740904] | 0% |
| core_input_value | [-0.27846456, 5.5828161] | [-0.27844544, 5.5478388] | 0.2327919% |
| gdn_core_output | [-0.26395759, 0.19815163] | [-0.25337622, 0.17699494] | 0.00014305115% |
| recurrent_state_output | [-3.1699224, 2.5329218] | [-3.2020785, 2.5740904] | 0% |
| rms_weighted | [-9.2018738, 9.3015947] | [-9.0649106, 9.3017277] | 0.00019073486% |
| z_input | [-9.1924429, 7.1938391] | [-7.5256361, 7.1938936] | 4.7683716e-05% |
| z_silu | [-0.27846456, 7.1884389] | [-0.27846357, 7.1884392] | 0.079774857% |
| gdn_norm_z_gate | [-2.5572574, 2.5596676] | [-2.5615357, 2.5177454] | 0.00028610229% |

state 的实际 scale=8.81386877e-5，zero point=36330，code 范围 [0,65535]。这些参数来自校准，并未改为硬编码。初始 state 全零所以输入对照无误差；该项不代表非零跨窗口 cache 已得到验证。

## 9. 性能代价

同一带 22 个检查点输出的完整层 PTE，同一文本输入，warmup 10 次，每轮计时 100 次，交替跑两轮；全部保持 UINT16 state。

| 实现 | 两轮平均 / ms | 总平均 / ms |
|---|---|---:|
| 历史缩位归约 | 4.73333, 4.72886 | 4.73109 |
| 完整精度 HVX FP32 归约 | 6.25044, 6.41242 | 6.33143 |

完整层计时增加约 33.8%。这是带检查点且包含 FFN 的整层计时，不是仅 GDN core，也不能与之前去掉检查点的 linear-attention 性能表直接比较。当前修复优先消除可定位的数值损失；进一步性能优化应保留完整中心化信息。

## 10. 验证和剩余问题

- HTP x86、Android arm64、Hexagon v79 三个 op package 构建成功；历史宏分支另行编译检查。
- 8 个 128-token 评估窗口的完整层设备输出均为有限值；没有更换数据集或 FP32 参考。
- 三个设备版本的 attention_norm、Q/K/decay/beta、z/SiLU(z) 在全部 8 个窗口逐字节一致，见 `upstream_equivalence.json`。
- 新通用 GPTQ 函数的两个单元测试验证独立评估输入上的投影误差改善、输入/scale 不被修改、未观测列和末尾处理 block。
- 通用 GPTQ 实现复算并逐值核对，完全复现本次测量的 2,097,152 个 V code。
- 增量 make 原先只更新 `.so`，设备使用的 `_HTP.so` 可能仍为旧版；Makefile 已增加运行时库的依赖复制。
- 完整层 hidden state 已从 17.59→21.00 dB，但 FFN W4、V/conv 激活编码和每步 state 舍入仍有误差。门控 29.61 dB 不是问题全部解决，完整模型 PPL、非零跨窗口 state、decode 应分别继续验证。

## 原始数据

- `metrics.csv` / `contributions.csv` / `channel_metrics.csv`：历史门控误差分解及 recurrence replay。
- `input_causality/metrics.csv`：逐项替换输入的 128-token 轨迹。
- `full_precision_reduction/metrics.csv` / `outputs/`：DSP 修复后完整层实测。
- `gptq_value/metrics.csv` / `outputs/`：V GPTQ 后完整层实测。
- `performance/*.log`：相同 PTE 的重复计时。
- `original_op_package/`：实验前源码和二进制快照。

---

<a id="record-20"></a>

## 记录 20：full_precision_reduction / REPORT.md

原始记录位置：`llama-output/qwen3_5-0_8b-layer-analysis/kz_gate_drop/full_precision_reduction/REPORT.md`。

# 完整 Qwen3.5 layer 0 / 128-token prefill 误差

## 配置

k/z 合并 W8A16，权重为 4096×1024；q/v/a/b 合并 W4A16，权重为 4128×1024。FFN gate/up 为 W4A16，down 为 W8A16。激活、conv cache 和 recurrent state 均为 UINT16。

原始权重：qkv=[6144,1024]、z=[2048,1024]、a/b 各为 [16,1024]。q/k/v/z 各 2048 维，a/b 各 16 维。

8 段 WikiText：前 128 token 校准、后 128 token 评估，两个窗口独立零 cache。FP32、CPU QDQ、设备使用相同文本。attention 输出在 out_proj 之后；layer_output 为 FFN 后残差 hidden state，不包含词表 logits。

CPU 的 FFN same-input 对照只将 FFN 输入替换为 FP32 ffn_norm 输出再施加其真实 A16 编码，其他量化参数和 FFN 权重保持不变。

CPU QDQ 的 custom GDN 内部 recurrence 为 FP32，输入/输出施加校准 A16 QDQ；实际设备 VTCM kernel 每步将 state 舍入回 UINT16。因此设备相对 CPU 的差异同时包含每步 state 舍入和 DSP 实现差异。

设备使用带检查点输出的完整单层 PTE，投影、causal conv、VTCM GDN、门控、out_proj、残差、FFN 均由同一图执行；额外输出可能影响调度，本测试用于精度分析。

## 各阶段 SQNR

| 阶段 | CPU QDQ / dB | 设备 / dB | 设备相对 CPU QDQ / dB | FFN 同输入 / dB |
|---|---:|---:|---:|---:|
| attention_norm | 64.181 | 64.171 | 81.685 | — |
| core_input_query | 22.595 | 22.145 | 31.927 | — |
| core_input_key | 39.816 | 39.740 | 46.788 | — |
| core_input_value | 32.205 | 31.437 | 40.451 | — |
| core_input_decay | 43.246 | 43.144 | 56.130 | — |
| core_input_beta | 30.645 | 29.991 | 39.833 | — |
| core_input_state | inf | inf | inf | — |
| gdn_core_output | 32.312 | 32.146 | 40.323 | — |
| rms_weighted | 34.240 | 33.736 | 40.794 | — |
| z_input | 42.901 | 43.040 | 57.799 | — |
| z_silu | 43.432 | 43.432 | 71.857 | — |
| gdn_norm_z_gate | 28.774 | 28.167 | 34.509 | — |
| attention_output | 25.365 | 24.662 | 31.117 | — |
| attention_residual | 25.092 | 24.389 | 30.844 | — |
| ffn_norm | 24.195 | 23.350 | 29.185 | 67.731 |
| ffn_gate_projection | 20.241 | 19.720 | 28.221 | 21.790 |
| ffn_up_projection | 17.734 | 17.267 | 26.173 | 19.223 |
| ffn_down_projection | 17.133 | 16.621 | 25.146 | 18.746 |
| ffn_output | 17.133 | 16.621 | 25.146 | 18.746 |
| layer_output | 21.000 | 20.498 | 28.874 | — |
| conv_state_output | 26.836 | 26.367 | 36.528 | — |
| recurrent_state_output | 29.967 | 28.725 | 35.195 | — |

## FP32 min/max、量化范围和饱和比例

比例为评估 FP32 值落在校准可表达范围之外的元素比例。state 只统计最终 cache；其 128 步内部状态范围见之前的 GDN core 报告。

| 阶段 | FP32 min | FP32 max | 量化 min | 量化 max | 越界 % |
|---|---:|---:|---:|---:|---:|
| attention_norm | -6.7463 | 7.36656 | -8.0377 | 7.37203 | 0.000000 |
| core_input_query | -0.0866934 | 0.0877734 | -0.0854558 | 0.0878169 | 0.000048 |
| core_input_key | -0.999021 | 0.985025 | -0.996903 | 0.987366 | 0.000095 |
| core_input_value | -0.278465 | 5.58282 | -0.278445 | 5.54784 | 0.232792 |
| core_input_decay | 9.00772e-05 | 1 | 0 | 1 | 0.000000 |
| core_input_beta | 0.0205198 | 0.998801 | 0 | 0.998905 | 0.000000 |
| core_input_state | 0 | 0 | -3.20208 | 2.57409 | 0.000000 |
| gdn_core_output | -0.263958 | 0.198152 | -0.253376 | 0.176995 | 0.000143 |
| rms_weighted | -9.20187 | 9.30159 | -9.06491 | 9.30173 | 0.000191 |
| z_input | -9.19244 | 7.19384 | -7.52564 | 7.19389 | 0.000048 |
| z_silu | -0.278465 | 7.18844 | -0.278464 | 7.18844 | 0.079775 |
| gdn_norm_z_gate | -2.55726 | 2.55967 | -2.56154 | 2.51775 | 0.000286 |
| attention_output | -1.28018 | 0.597044 | -1.40206 | 0.587589 | 0.000095 |
| attention_residual | -1.25052 | 0.615598 | -1.34712 | 0.616526 | 0.000000 |
| ffn_norm | -16.4056 | 10.7934 | -16.0616 | 11.0016 | 0.000191 |
| ffn_gate_projection | -7.94095 | 4.4608 | -9.84849 | 4.56595 | 0.000000 |
| ffn_up_projection | -2.26577 | 3.20637 | -2.28334 | 3.12835 | 0.000027 |
| ffn_down_projection | -0.46517 | 0.597192 | -0.476861 | 0.567522 | 0.000095 |
| ffn_output | -0.46517 | 0.597192 | -0.476861 | 0.567522 | 0.000095 |
| layer_output | -1.02786 | 0.665625 | -0.861777 | 0.663792 | 0.000191 |
| conv_state_output | -27.3168 | 28.2632 | -29.0012 | 28.2629 | 0.000509 |
| recurrent_state_output | -3.16992 | 2.53292 | -3.20208 | 2.57409 | 0.000000 |

## 实际 recipe

| 模块 | 精度 | 粒度 |
|---|---|---|
| `layers.0.attention.in_proj_qvab.conv` | use_16a4w_block | PER_BLOCK |
| `layers.0.attention.in_proj_kz.conv` | use_16a8w | PER_CHANNEL |
| `layers.0.attention.causal_conv2d` | use_16a8w | PER_CHANNEL |
| `layers.0.attention.out_proj.conv` | use_16a8w | PER_CHANNEL |
| `layers.0.feed_forward.w1_conv` | use_16a4w_block | PER_BLOCK |
| `layers.0.feed_forward.w3_conv` | use_16a4w_block | PER_BLOCK |
| `layers.0.feed_forward.w2_conv` | use_16a8w | PER_CHANNEL |

## 使用与复现

完整模型导出可设置 `EXECUTORCH_QWEN35_FUSE_KZ_INT8=1`，选择合并 k/z W8、合并 q/v/a/b W4。

```bash
conda activate et_npu
source env.sh
python -u llama-output/qwen3_5-0_8b-layer-analysis/run_merged_gdn_layer128.py \
  --out llama-output/qwen3_5-0_8b-layer-analysis/kz_gate_drop/full_precision_reduction --mode kz --device 3B1F57E9B8L74X5D
```

测试覆盖 1/128-token、非零 cache、key/value head 数不同的完整 attention 等价性和实际 recipe 命中，参见 `backends/qualcomm/tests/test_gdn_split_key_projection.py`。尚未执行全模型 PPL。

## 产物

`metrics.csv` 保存 128 个 token 的阶段误差；`metadata.json` 保存节点、形状和校准编码；`qdq.pt2` 保存校准后图；`.pte` 和 `outputs/` 为实际设备运行产物。

128-token 阶段误差（历史产物已清理）

---

<a id="record-21"></a>

## 记录 21：gptq_value / REPORT.md

原始记录位置：`llama-output/qwen3_5-0_8b-layer-analysis/kz_gate_drop/gptq_value/REPORT.md`。

# 完整 Qwen3.5 layer 0 / 128-token prefill 误差

## 配置

k/z 合并 W8A16，权重为 4096×1024；q/v/a/b 合并 W4A16，权重为 4128×1024。FFN gate/up 为 W4A16，down 为 W8A16。激活、conv cache 和 recurrent state 均为 UINT16。

原始权重：qkv=[6144,1024]、z=[2048,1024]、a/b 各为 [16,1024]。q/k/v/z 各 2048 维，a/b 各 16 维。

8 段 WikiText：前 128 token 校准、后 128 token 评估，两个窗口独立零 cache。FP32、CPU QDQ、设备使用相同文本。attention 输出在 out_proj 之后；layer_output 为 FFN 后残差 hidden state，不包含词表 logits。

CPU 的 FFN same-input 对照只将 FFN 输入替换为 FP32 ffn_norm 输出再施加其真实 A16 编码，其他量化参数和 FFN 权重保持不变。

CPU QDQ 的 custom GDN 内部 recurrence 为 FP32，输入/输出施加校准 A16 QDQ；实际设备 VTCM kernel 每步将 state 舍入回 UINT16。因此设备相对 CPU 的差异同时包含每步 state 舍入和 DSP 实现差异。

设备使用带检查点输出的完整单层 PTE，投影、causal conv、VTCM GDN、门控、out_proj、残差、FFN 均由同一图执行；额外输出可能影响调度，本测试用于精度分析。

## 各阶段 SQNR

| 阶段 | CPU QDQ / dB | 设备 / dB | 设备相对 CPU QDQ / dB | FFN 同输入 / dB |
|---|---:|---:|---:|---:|
| attention_norm | 64.181 | 64.171 | 81.685 | — |
| core_input_query | 22.595 | 22.145 | 31.927 | — |
| core_input_key | 39.816 | 39.740 | 46.788 | — |
| core_input_value | 35.654 | 34.534 | 40.935 | — |
| core_input_decay | 43.246 | 43.144 | 56.130 | — |
| core_input_beta | 30.645 | 29.991 | 39.833 | — |
| core_input_state | inf | inf | inf | — |
| gdn_core_output | 32.949 | 32.789 | 40.344 | — |
| rms_weighted | 36.812 | 35.521 | 40.834 | — |
| z_input | 42.901 | 43.040 | 57.799 | — |
| z_silu | 43.432 | 43.432 | 71.857 | — |
| gdn_norm_z_gate | 30.888 | 29.609 | 34.504 | — |
| attention_output | 26.809 | 25.756 | 31.084 | — |
| attention_residual | 26.536 | 25.483 | 30.811 | — |
| ffn_norm | 26.100 | 24.736 | 29.147 | — |
| ffn_gate_projection | 20.789 | 20.156 | 28.145 | — |
| ffn_up_projection | 18.217 | 17.660 | 26.145 | — |
| ffn_down_projection | 17.647 | 17.038 | 25.138 | — |
| ffn_output | 17.647 | 17.038 | 25.138 | — |
| layer_output | 21.610 | 21.002 | 28.857 | — |
| conv_state_output | 27.988 | 27.414 | 36.496 | — |
| recurrent_state_output | 33.112 | 31.152 | 35.823 | — |

## FP32 min/max、量化范围和饱和比例

比例为评估 FP32 值落在校准可表达范围之外的元素比例。state 只统计最终 cache；其 128 步内部状态范围见之前的 GDN core 报告。

| 阶段 | FP32 min | FP32 max | 量化 min | 量化 max | 越界 % |
|---|---:|---:|---:|---:|---:|
| attention_norm | -6.7463 | 7.36656 | -8.0377 | 7.37203 | 0.000000 |
| core_input_query | -0.0866934 | 0.0877734 | -0.0854558 | 0.0878169 | 0.000048 |
| core_input_key | -0.999021 | 0.985025 | -0.996903 | 0.987366 | 0.000095 |
| core_input_value | -0.278465 | 5.58282 | -0.278445 | 5.54784 | 0.232792 |
| core_input_decay | 9.00772e-05 | 1 | 0 | 1 | 0.000000 |
| core_input_beta | 0.0205198 | 0.998801 | 0 | 0.998905 | 0.000000 |
| core_input_state | 0 | 0 | -3.20208 | 2.57409 | 0.000000 |
| gdn_core_output | -0.263958 | 0.198152 | -0.253376 | 0.176995 | 0.000143 |
| rms_weighted | -9.20187 | 9.30159 | -9.06491 | 9.30173 | 0.000191 |
| z_input | -9.19244 | 7.19384 | -7.52564 | 7.19389 | 0.000048 |
| z_silu | -0.278465 | 7.18844 | -0.278464 | 7.18844 | 0.079775 |
| gdn_norm_z_gate | -2.55726 | 2.55967 | -2.56154 | 2.51775 | 0.000286 |
| attention_output | -1.28018 | 0.597044 | -1.40206 | 0.587589 | 0.000095 |
| attention_residual | -1.25052 | 0.615598 | -1.34712 | 0.616526 | 0.000000 |
| ffn_norm | -16.4056 | 10.7934 | -16.0616 | 11.0016 | 0.000191 |
| ffn_gate_projection | -7.94095 | 4.4608 | -9.84849 | 4.56595 | 0.000000 |
| ffn_up_projection | -2.26577 | 3.20637 | -2.28334 | 3.12835 | 0.000027 |
| ffn_down_projection | -0.46517 | 0.597192 | -0.476861 | 0.567522 | 0.000095 |
| ffn_output | -0.46517 | 0.597192 | -0.476861 | 0.567522 | 0.000095 |
| layer_output | -1.02786 | 0.665625 | -0.861777 | 0.663792 | 0.000191 |
| conv_state_output | -27.3168 | 28.2632 | -29.0012 | 28.2629 | 0.000509 |
| recurrent_state_output | -3.16992 | 2.53292 | -3.20208 | 2.57409 | 0.000000 |

## 实际 recipe

| 模块 | 精度 | 粒度 |
|---|---|---|
| `layers.0.attention.in_proj_qvab.conv` | use_16a4w_block | PER_BLOCK |
| `layers.0.attention.in_proj_kz.conv` | use_16a8w | PER_CHANNEL |
| `layers.0.attention.causal_conv2d` | use_16a8w | PER_CHANNEL |
| `layers.0.attention.out_proj.conv` | use_16a8w | PER_CHANNEL |
| `layers.0.feed_forward.w1_conv` | use_16a4w_block | PER_BLOCK |
| `layers.0.feed_forward.w3_conv` | use_16a4w_block | PER_BLOCK |
| `layers.0.feed_forward.w2_conv` | use_16a8w | PER_CHANNEL |

## 使用与复现

完整模型导出可设置 `EXECUTORCH_QWEN35_FUSE_KZ_INT8=1`，选择合并 k/z W8、合并 q/v/a/b W4。

```bash
conda activate et_npu
source env.sh
python -u llama-output/qwen3_5-0_8b-layer-analysis/run_merged_gdn_layer128.py \
  --out llama-output/qwen3_5-0_8b-layer-analysis/kz_gate_drop/gptq_value --mode kz --device 3B1F57E9B8L74X5D
```

测试覆盖 1/128-token、非零 cache、key/value head 数不同的完整 attention 等价性和实际 recipe 命中，参见 `backends/qualcomm/tests/test_gdn_split_key_projection.py`。尚未执行全模型 PPL。

## 产物

`metrics.csv` 保存 128 个 token 的阶段误差；`metadata.json` 保存节点、形状和校准编码；`qdq.pt2` 保存校准后图；`.pte` 和 `outputs/` 为实际设备运行产物。

128-token 阶段误差（历史产物已清理）

---

<a id="record-22"></a>

## 记录 22：l2norm_fix / REPORT.md

原始记录位置：`llama-output/qwen3_5-0_8b-layer-analysis/l2norm_fix/REPORT.md`。

# Qwen3.5 GDN q/k L2 normalization 修复报告

## 本轮范围

只处理 GDN q/k 的 L2 normalization，也就是原来的 `rsqrt_1/2`；FFN 权重配置和 recurrent state 算法未改。本报告的 SQNR 是归一化输出张量的 SQNR，不是 logits，也不是整层累计 SQNR。

## 根因

原公式：`x * rsqrt(sum(x*x) + 1e-6)`。PT2E 在 reduction、加 epsilon、rsqrt、乘回输入之间插入激活 QDQ。即便 scale/zero point 来自正常校准，较小的正数也可以被舍入为零；`rsqrt(0)` 因而产生 Inf。

隔离校准得到 query denominator scale = `0.0014709224924445152`、key denominator scale = `0.0002694379072636366`，两者 zero point 都是 0，量化范围 0..65535。`1e-6` 小于这两个量化步长。问题是对分母做中间量化，不是 scale/zero point 被写死。

## 最终实现：现成 QNN RmsNorm，数学等价

对于归一化维度长度 D：

```python
F.rms_norm(x, (D,), eps=eps / D) * (D ** -0.5)
```

由于 `mean(x²)=sum(x²)/D`，这个表达式等于 `x/sqrt(sum(x²)+eps)`。D=128 时传给原生 RmsNorm 的 epsilon 为 `7.8125e-9`。这是原模型 epsilon 的代数换算，与校准 scale 不同。

reduction、epsilon 和 inverse sqrt 全部在 QNN 原生 RmsNorm 内部完成，避免在 denominator 上做独立 QDQ；输入和归一化输出仍按校准结果量化。无需改 VTCM recurrence op，也无需新增 DSP op。

## QNN L2Norm 查询结果

本地 QNN SDK 2.49.40.260810 的 `QnnOpDef.h` 和 HTP OpDef Supplement 都有 L2Norm，支持 rank<=4 和 INT16/FP16 等配置。但是当前 ExecuTorch 没有对应 builder，而且其公式是 `x/max(||x||₂, epsilon)`，与当前模型的 additive epsilon 不相同。因此使用仓库已经接入的 RmsNorm 保持公式一致。

本地依据：

- `/hunter/qnn_sdk/src/backends/qnn/2.49.40.260810/docs/QAIRT-Docs/QNN/OpDef/MasterOpDef.html`：L2Norm、RmsNorm
- `/hunter/qnn_sdk/src/backends/qnn/2.49.40.260810/docs/QAIRT-Docs/QNN/OpDef/HtpOpDefSupplement.html`：L2Norm、RmsNorm
- `backends/qualcomm/builders/op_rms_norm.py`：已存在的映射

## 测试方法与命名空间修正

仅加载 layer 0，128 tokens，八段 WikiText 来源文本。为准确命中正式 recipe，将单层保留在 `layers.0` 名字下。确认 qkv/b 使用 PER_BLOCK W4，a/causal_conv2d/out_proj 使用 PER_CHANNEL W8，FFN w1/w3 仍 W4、w2 仍 W8。

**此前直接导出裸 decoder layer 的分析脚本没有 `layers.0` 前缀，部分 attention regex 未命中；那些单层累计误差数字不能直接代表正式 recipe。** 本轮缓存 `layer0_qk_production_recipe.pt` 使用正确命名空间，实际命中配置保存在 `summary.json.upstream_recipe`。

校准输入取 FP32 layer-0 的 q/k；评估分别用同一批 FP32 q/k、同一批 W4 上游 q/k。每个归一化方案与相同输入直接运行原 FP32 公式对比，因此下面只衡量归一化新引入的误差，不把前面投影误差归给 rsqrt。

## 相同 W4 上游输入：隔离归一化对比

| 方案 | q SQNR / dB | k SQNR / dB | inverse sqrt 非有限值 |
|---|---:|---:|---:|
| 原始分解 + PT2E QDQ（CPU） | 20.024 | 26.617 | 4 |
| float 区域（CPU，仅诊断） | 80.235 | 61.450 | 0 |
| 原生 RmsNorm + PT2E QDQ（CPU） | 80.235 | 61.393 | 0 |
| 原生 RmsNorm PTE（SM8750 HTP） | 73.715 | 61.159 | 无独立 rsqrt 节点；输出非有限值 0 |

float 区域方案虽然消除了 Inf，但最初设备编译出现量化/浮点边界的 multiply 类型不匹配，部分节点回退 CPU。该方案仅留作诊断，没有接入生产。最终 native RmsNorm 方案的全部计算节点均下沉 HTP；编译日志没有 `| False` 和 `validateOpConfig failed`。

注意：基线最终输出可能仍然全部有限，因为 rsqrt 后面的 QDQ 可以掩盖/截断 Inf；这不说明内部计算正确。

## 零向量和极小向量

额外评估全零输入和所有元素=1e-5 的输入；这两种输入没有用于校准。最终 CPU QDQ 和设备输出均无 NaN/Inf。全零输入输出严格为 0。极小输入在 INT16 activation 的步长内被舍入为 0，因而输出 SQNR 为 0 dB，这个量化输入信息损失仍然存在；修复消除的是 denominator 被中途量化导致的非有限值，未声称任意幅值向量都能高 SQNR。

## 测试和产物

- `backends/qualcomm/tests/test_gdn_l2norm.py`：12 组 FP32 数学等价测试（含非末维、全零、1e-5、1e-3、普通随机输入），1 组 PT2E 图和有限值测试，13 passed。
- 现有 GDN cache 共享 observer 回归测试：1 passed；合计 14 项通过。
- 实际设备：`3B1F57E9B8L74X5D`，SM8750 / Hexagon v79。
- `native_rms_l2.pte`：独立 q/k 归一化图。
- `native_rms_qdq.py`：量化图；包含两个 `aten.rms_norm`，没有独立 `rsqrt`。
- `summary.json`：逐样本 SQNR、非有限值计数及实际上游 recipe。
- `runner.log`：设备输出记录。
- `../l2norm_native_rms.log`：编译及算子下沉证据。

复现命令（仓库根目录）：

```bash
source /home/hunter/anaconda3/etc/profile.d/conda.sh
conda activate et_npu
source env.sh
python llama-output/qwen3_5-0_8b-layer-analysis/diagnose_l2norm.py \
  --device 3B1F57E9B8L74X5D
```

本轮尚未重新导出全模型或测 PPL；上述结果证明归一化本身的问题已消除，不能据此声称全模型 PPL 已修复。

---

<a id="record-23"></a>

## 记录 23：merged_kz_layer128 / COMPARISON.md

原始记录位置：`llama-output/qwen3_5-0_8b-layer-analysis/merged_kz_layer128/COMPARISON.md`。

# 合并 k/z W8、合并 q/v/a/b W4：完整单层 128-token prefill 对照

## 配置和验证

| 投影 | 权重形状 | 精度 |
|---|---|---|
| k/z | 4096×1024 | W8A16，per-channel |
| q/v/a/b | 4128×1024 | W4A16，block size=32 |

q/k/v/z 各 2048 维，a/b 各 16 维；k/z 的行顺序为 k、z，另一矩阵为 q、v、a、b。所有原始权重行精确搬移。

- 沿用同一层 0、8 段 WikiText、前 128-token 校准/后 128-token 评估，独立零 cache。新旧 FP32 完整层检查点逐项核对通过，见 `fp32_equivalence.json`。
- QNN/PT2E 保留 `layers.0` 路径，确认两个矩阵实际命中 W8 per-channel / W4 per-block。
- causal conv/out_proj 仍为 W8A16；FFN gate/up 为 W4A16、down 为 W8A16；conv cache/recurrent state 为 UINT16。
- 各配置独立运行相同校准文本；scale/zero point 由校准图提取，见 `metadata.json`。
- 实际设备 SM8750/v79，完整投影、GDN VTCM、RMSNorm、z gate、out_proj、残差、FFN 均在单个 QNN delegate 中执行。额外输出用于精度分析，不能作为普通部署的性能基准。
- 1/128-token、非零 state、不同 key/value head 数的等价性以及实际量化 recipe 测试：6 passed。实测非有限数据计数：0。

## 新旧精度比较

旧配置：key W8，合并 q/v/a/b/z W4。新配置：合并 k/z W8，合并 q/v/a/b W4。

| 阶段 | 旧 CPU / dB | 新 CPU / dB | 旧设备 / dB | 新设备 / dB |
|---|---:|---:|---:|---:|
| core_input_query | 22.595 | 22.595 | 22.145 | 22.145 |
| core_input_key | 39.816 | 39.816 | 39.740 | 39.740 |
| core_input_value | 32.205 | 32.205 | 31.437 | 31.437 |
| gdn_core_output | 32.312 | 32.312 | 31.602 | 31.601 |
| recurrent_state_output | 29.967 | 29.967 | 28.658 | 28.658 |
| rms_weighted | 34.240 | 34.240 | 30.428 | 30.428 |
| z_input | 22.353 | 42.901 | 21.924 | 43.040 |
| z_silu | 22.840 | 43.432 | 22.380 | 43.432 |
| gdn_norm_z_gate | 23.144 | 28.774 | 21.041 | 23.773 |
| attention_output | 21.791 | 25.365 | 18.551 | 20.034 |
| ffn_norm | 20.509 | 24.195 | 16.642 | 17.976 |
| ffn_output | 15.549 | 17.133 | 12.774 | 13.679 |
| layer_output | 18.970 | 21.000 | 16.426 | 17.589 |

z 设备 SQNR 从 21.924 提升到 43.040 dB，门控输出从 21.041 提升到 23.773 dB。最终 hidden 从 16.426 提升到 17.589 dB。

不同阶段的 SQNR 以各自 FP32 信号为参考，不能通过节点间 dB 相减来划分误差贡献。state 统计第 128 步最终 cache；其他阶段合并全部 8×128 个 token 的 signal/noise power。

## 剩余误差

上一轮只把 GDN 输入替换为设备值、z 及后续运算保持 FP32时，门控输出为 23.834 dB。本次 z W8 后实测门控为 23.773 dB，与该对照接近。

这支持剩余门控误差主要受 GDN 输入误差传播影响。若要进一步定量，仍需在当前布局重做输入替换；此处没有假设不同误差完全独立。

GDN core 输出仍约 31.601 dB，recurrent state 约 28.658 dB，q 仍为 W4。FFN 使用干净输入的 CPU 对照仍只有 18.746 dB，因此 W4 FFN 自身误差仍然存在。

## 权重和 PTE 开销

理想打包权重：k/z W8 为 4 MiB，q/v/a/b W4 为 2.015625 MiB；相比 key W8 + q/v/a/b/z W4，增加约 1 MiB/GDN 层，不含 scale 元数据。

带检查点的实测 PTE：旧 15.614 MiB，新 16.568 MiB。这两个图的检查点数量不同，因此 PTE 大小差不完全等于权重差。

## 启用和复现

完整模型导出前设置：

```bash
export EXECUTORCH_QWEN35_FUSE_KZ_INT8=1
```

单层测试：

```bash
conda activate et_npu
source env.sh
python -u llama-output/qwen3_5-0_8b-layer-analysis/run_merged_gdn_layer128.py \
  --mode kz --out llama-output/qwen3_5-0_8b-layer-analysis/merged_kz_layer128 \
  --device 3B1F57E9B8L74X5D
python llama-output/qwen3_5-0_8b-layer-analysis/report_kz_layer128.py
```

`REPORT.md` 包含本配置的全部阶段、范围及饱和比例；`metrics.csv` / `trace.png` 保存逐 token 误差；`outputs/` 为设备原始输出。本轮尚未导出全部 24 层，也未复测全模型 PPL。

新旧逐 token 比较（历史产物已清理）

---

<a id="record-24"></a>

## 记录 24：merged_kz_layer128 / REPORT.md

原始记录位置：`llama-output/qwen3_5-0_8b-layer-analysis/merged_kz_layer128/REPORT.md`。

# 完整 Qwen3.5 layer 0 / 128-token prefill 误差

## 配置

k/z 合并 W8A16，权重为 4096×1024；q/v/a/b 合并 W4A16，权重为 4128×1024。FFN gate/up 为 W4A16，down 为 W8A16。激活、conv cache 和 recurrent state 均为 UINT16。

原始权重：qkv=[6144,1024]、z=[2048,1024]、a/b 各为 [16,1024]。q/k/v/z 各 2048 维，a/b 各 16 维。

8 段 WikiText：前 128 token 校准、后 128 token 评估，两个窗口独立零 cache。FP32、CPU QDQ、设备使用相同文本。attention 输出在 out_proj 之后；layer_output 为 FFN 后残差 hidden state，不包含词表 logits。

CPU 的 FFN same-input 对照只将 FFN 输入替换为 FP32 ffn_norm 输出再施加其真实 A16 编码，其他量化参数和 FFN 权重保持不变。

CPU QDQ 的 custom GDN 内部 recurrence 为 FP32，输入/输出施加校准 A16 QDQ；实际设备 VTCM kernel 每步将 state 舍入回 UINT16。因此设备相对 CPU 的差异同时包含每步 state 舍入和 DSP 实现差异。

设备使用带检查点输出的完整单层 PTE，投影、causal conv、VTCM GDN、门控、out_proj、残差、FFN 均由同一图执行；额外输出可能影响调度，本测试用于精度分析。

## 各阶段 SQNR

| 阶段 | CPU QDQ / dB | 设备 / dB | 设备相对 CPU QDQ / dB | FFN 同输入 / dB |
|---|---:|---:|---:|---:|
| attention_norm | 64.181 | 64.171 | 81.685 | — |
| core_input_query | 22.595 | 22.145 | 31.927 | — |
| core_input_key | 39.816 | 39.740 | 46.788 | — |
| core_input_value | 32.205 | 31.437 | 40.451 | — |
| core_input_decay | 43.246 | 43.144 | 56.130 | — |
| core_input_beta | 30.645 | 29.991 | 39.833 | — |
| core_input_state | inf | inf | inf | — |
| gdn_core_output | 32.312 | 31.601 | 37.999 | — |
| rms_weighted | 34.240 | 30.428 | 32.341 | — |
| z_input | 42.901 | 43.040 | 57.799 | — |
| z_silu | 43.432 | 43.432 | 71.857 | — |
| gdn_norm_z_gate | 28.774 | 23.773 | 25.055 | — |
| attention_output | 25.365 | 20.034 | 21.279 | — |
| attention_residual | 25.092 | 19.761 | 21.006 | — |
| ffn_norm | 24.195 | 17.976 | 18.916 | 67.731 |
| ffn_gate_projection | 20.241 | 16.869 | 19.312 | 21.790 |
| ffn_up_projection | 17.734 | 14.588 | 17.275 | 19.223 |
| ffn_down_projection | 17.133 | 13.679 | 15.942 | 18.746 |
| ffn_output | 17.133 | 13.679 | 15.942 | 18.746 |
| layer_output | 21.000 | 17.589 | 19.911 | — |
| conv_state_output | 26.836 | 26.367 | 36.528 | — |
| recurrent_state_output | 29.967 | 28.658 | 34.944 | — |

## FP32 min/max、量化范围和饱和比例

比例为评估 FP32 值落在校准可表达范围之外的元素比例。state 只统计最终 cache；其 128 步内部状态范围见之前的 GDN core 报告。

| 阶段 | FP32 min | FP32 max | 量化 min | 量化 max | 越界 % |
|---|---:|---:|---:|---:|---:|
| attention_norm | -6.7463 | 7.36656 | -8.0377 | 7.37203 | 0.000000 |
| core_input_query | -0.0866934 | 0.0877734 | -0.0854558 | 0.0878169 | 0.000048 |
| core_input_key | -0.999021 | 0.985025 | -0.996903 | 0.987366 | 0.000095 |
| core_input_value | -0.278465 | 5.58282 | -0.278445 | 5.54784 | 0.232792 |
| core_input_decay | 9.00772e-05 | 1 | 0 | 1 | 0.000000 |
| core_input_beta | 0.0205198 | 0.998801 | 0 | 0.998905 | 0.000000 |
| core_input_state | 0 | 0 | -3.20208 | 2.57409 | 0.000000 |
| gdn_core_output | -0.263958 | 0.198152 | -0.253376 | 0.176995 | 0.000143 |
| rms_weighted | -9.20187 | 9.30159 | -9.06491 | 9.30173 | 0.000191 |
| z_input | -9.19244 | 7.19384 | -7.52564 | 7.19389 | 0.000048 |
| z_silu | -0.278465 | 7.18844 | -0.278464 | 7.18844 | 0.079775 |
| gdn_norm_z_gate | -2.55726 | 2.55967 | -2.56154 | 2.51775 | 0.000286 |
| attention_output | -1.28018 | 0.597044 | -1.40206 | 0.587589 | 0.000095 |
| attention_residual | -1.25052 | 0.615598 | -1.34712 | 0.616526 | 0.000000 |
| ffn_norm | -16.4056 | 10.7934 | -16.0616 | 11.0016 | 0.000191 |
| ffn_gate_projection | -7.94095 | 4.4608 | -9.84849 | 4.56595 | 0.000000 |
| ffn_up_projection | -2.26577 | 3.20637 | -2.28334 | 3.12835 | 0.000027 |
| ffn_down_projection | -0.46517 | 0.597192 | -0.476861 | 0.567522 | 0.000095 |
| ffn_output | -0.46517 | 0.597192 | -0.476861 | 0.567522 | 0.000095 |
| layer_output | -1.02786 | 0.665625 | -0.861777 | 0.663792 | 0.000191 |
| conv_state_output | -27.3168 | 28.2632 | -29.0012 | 28.2629 | 0.000509 |
| recurrent_state_output | -3.16992 | 2.53292 | -3.20208 | 2.57409 | 0.000000 |

## 实际 recipe

| 模块 | 精度 | 粒度 |
|---|---|---|
| `layers.0.attention.in_proj_qvab.conv` | use_16a4w_block | PER_BLOCK |
| `layers.0.attention.in_proj_kz.conv` | use_16a8w | PER_CHANNEL |
| `layers.0.attention.causal_conv2d` | use_16a8w | PER_CHANNEL |
| `layers.0.attention.out_proj.conv` | use_16a8w | PER_CHANNEL |
| `layers.0.feed_forward.w1_conv` | use_16a4w_block | PER_BLOCK |
| `layers.0.feed_forward.w3_conv` | use_16a4w_block | PER_BLOCK |
| `layers.0.feed_forward.w2_conv` | use_16a8w | PER_CHANNEL |

## 使用与复现

完整模型导出可设置 `EXECUTORCH_QWEN35_FUSE_KZ_INT8=1`，选择合并 k/z W8、合并 q/v/a/b W4。

```bash
conda activate et_npu
source env.sh
python -u llama-output/qwen3_5-0_8b-layer-analysis/run_merged_gdn_layer128.py \
  --out llama-output/qwen3_5-0_8b-layer-analysis/merged_kz_layer128 --mode kz --device 3B1F57E9B8L74X5D
```

测试覆盖 1/128-token、非零 cache、key/value head 数不同的完整 attention 等价性和实际 recipe 命中，参见 `backends/qualcomm/tests/test_gdn_split_key_projection.py`。尚未执行全模型 PPL。

## 产物

`metrics.csv` 保存 128 个 token 的阶段误差；`metadata.json` 保存节点、形状和校准编码；`qdq.pt2` 保存校准后图；`.pte` 和 `outputs/` 为实际设备运行产物。

128-token 阶段误差（历史产物已清理）

---

<a id="record-25"></a>

## 记录 25：merged_qvabz_layer128 / REPORT.md

原始记录位置：`llama-output/qwen3_5-0_8b-layer-analysis/merged_qvabz_layer128/REPORT.md`。

# 完整 Qwen3.5 layer 0 / 128-token prefill 误差

## 配置

key 独立 W8A16；q/v/a/b/z 合并 W4A16，权重为 6176×1024。FFN gate/up 为 W4A16，down 为 W8A16。激活、conv cache 和 recurrent state 均为 UINT16。

原始权重：qkv=[6144,1024]、z=[2048,1024]、a/b 各为 [16,1024]。q/v/z 各 2048 维，a/b 各 16 维，因此合并 q/v/a/b/z 的输出宽度为 2048+2048+16+16+2048=6176；key 单独为 2048×1024。

8 段 WikiText：前 128 token 校准、后 128 token 评估，两个窗口独立零 cache。FP32、CPU QDQ、设备使用相同文本。attention 输出在 out_proj 之后；layer_output 为 FFN 后残差 hidden state，不包含词表 logits。

CPU 的 FFN same-input 对照只将 FFN 输入替换为 FP32 ffn_norm 输出再施加其真实 A16 编码，其他量化参数和 FFN 权重保持不变。

CPU QDQ 的 custom GDN 内部 recurrence 为 FP32，输入/输出施加校准 A16 QDQ；实际设备 VTCM kernel 每步将 state 舍入回 UINT16。因此设备相对 CPU 的差异同时包含每步 state 舍入和 DSP 实现差异。

设备使用带检查点输出的完整单层 PTE，投影、causal conv、VTCM GDN、门控、out_proj、残差、FFN 均由同一图执行；额外输出可能影响调度，本测试用于精度分析。

## 各阶段 SQNR

| 阶段 | CPU QDQ / dB | 设备 / dB | 设备相对 CPU QDQ / dB | FFN 同输入 / dB |
|---|---:|---:|---:|---:|
| attention_norm | 64.181 | 64.171 | 81.685 | — |
| core_input_query | 22.595 | 22.145 | 31.927 | — |
| core_input_key | 39.816 | 39.740 | 46.788 | — |
| core_input_value | 32.205 | 31.437 | 40.451 | — |
| core_input_decay | 43.246 | 43.144 | 56.130 | — |
| core_input_beta | 30.645 | 29.991 | 39.833 | — |
| core_input_state | inf | inf | inf | — |
| gdn_core_output | 32.312 | 31.602 | 37.999 | — |
| gdn_norm_z_gate | 23.144 | 21.041 | 24.476 | — |
| attention_output | 21.791 | 18.551 | 21.029 | — |
| attention_residual | 21.518 | 18.278 | 20.758 | — |
| ffn_norm | 20.509 | 16.642 | 18.694 | 67.731 |
| ffn_gate_projection | 18.819 | 16.041 | 19.087 | 21.790 |
| ffn_up_projection | 16.214 | 13.699 | 17.077 | 19.223 |
| ffn_down_projection | 15.549 | 12.774 | 15.724 | 18.746 |
| ffn_output | 15.549 | 12.774 | 15.724 | 18.746 |
| layer_output | 18.970 | 16.426 | 19.637 | — |
| conv_state_output | 26.836 | 26.367 | 36.528 | — |
| recurrent_state_output | 29.967 | 28.658 | 34.944 | — |

## 原布局与新布局：完整单层 CPU QDQ 对照

新旧 FP32 图在 8 段评估文本的全部检查点逐项核对通过，最大差值见 `fp32_equivalence.json`。

| 阶段 | 原布局 / dB | key W8 + merged W4 / dB |
|---|---:|---:|
| core_input_key | 21.065 | 39.816 |
| gdn_core_output | 30.540 | 32.312 |
| recurrent_state_output | 21.502 | 29.967 |
| gdn_norm_z_gate | 23.078 | 23.144 |
| attention_output | 21.720 | 21.791 |
| ffn_output | 15.489 | 15.549 |
| layer_output | 18.923 | 18.970 |

key W8 的改善主要体现在 recurrent state。norm/z gate 之后仍存在其他误差，完整层 hidden state 从 18.923 变为 18.970 dB，改善有限。FFN 同输入输出为 18.746 dB，说明其自身量化误差也需要处理；此测试尚未把 FFN 的权重与激活误差进一步分离。

设备相对 CPU QDQ：core 输出为 37.999 dB，而 norm/z gate 后降至 24.476 dB。这段设备差异需要进一步拆分，当前不能仅凭复合阶段指标归因于 z 权重或某个 norm 算子。

## FP32 min/max、量化范围和饱和比例

比例为评估 FP32 值落在校准可表达范围之外的元素比例。state 只统计最终 cache；其 128 步内部状态范围见之前的 GDN core 报告。

| 阶段 | FP32 min | FP32 max | 量化 min | 量化 max | 越界 % |
|---|---:|---:|---:|---:|---:|
| attention_norm | -6.7463 | 7.36656 | -8.0377 | 7.37203 | 0.000000 |
| core_input_query | -0.0866934 | 0.0877734 | -0.0854558 | 0.0878169 | 0.000048 |
| core_input_key | -0.999021 | 0.985025 | -0.996903 | 0.987366 | 0.000095 |
| core_input_value | -0.278465 | 5.58282 | -0.278445 | 5.54784 | 0.232792 |
| core_input_decay | 9.00772e-05 | 1 | 0 | 1 | 0.000000 |
| core_input_beta | 0.0205198 | 0.998801 | 0 | 0.998905 | 0.000000 |
| core_input_state | 0 | 0 | -3.20208 | 2.57409 | 0.000000 |
| gdn_core_output | -0.263958 | 0.198152 | -0.253376 | 0.176995 | 0.000143 |
| gdn_norm_z_gate | -2.55726 | 2.55967 | -2.56154 | 2.51775 | 0.000286 |
| attention_output | -1.28018 | 0.597044 | -1.40206 | 0.587589 | 0.000095 |
| attention_residual | -1.25052 | 0.615598 | -1.34712 | 0.616526 | 0.000000 |
| ffn_norm | -16.4056 | 10.7934 | -16.0616 | 11.0016 | 0.000191 |
| ffn_gate_projection | -7.94095 | 4.4608 | -9.84849 | 4.56595 | 0.000000 |
| ffn_up_projection | -2.26577 | 3.20637 | -2.28334 | 3.12835 | 0.000027 |
| ffn_down_projection | -0.46517 | 0.597192 | -0.476861 | 0.567522 | 0.000095 |
| ffn_output | -0.46517 | 0.597192 | -0.476861 | 0.567522 | 0.000095 |
| layer_output | -1.02786 | 0.665625 | -0.861777 | 0.663792 | 0.000191 |
| conv_state_output | -27.3168 | 28.2632 | -29.0012 | 28.2629 | 0.000509 |
| recurrent_state_output | -3.16992 | 2.53292 | -3.20208 | 2.57409 | 0.000000 |

## 实际 recipe

| 模块 | 精度 | 粒度 |
|---|---|---|
| `layers.0.attention.in_proj_qvabz.conv` | use_16a4w_block | PER_BLOCK |
| `layers.0.attention.in_proj_k.conv` | use_16a8w | PER_CHANNEL |
| `layers.0.attention.causal_conv2d` | use_16a8w | PER_CHANNEL |
| `layers.0.attention.out_proj.conv` | use_16a8w | PER_CHANNEL |
| `layers.0.feed_forward.w1_conv` | use_16a4w_block | PER_BLOCK |
| `layers.0.feed_forward.w3_conv` | use_16a4w_block | PER_BLOCK |
| `layers.0.feed_forward.w2_conv` | use_16a8w | PER_CHANNEL |

## 使用与复现

完整模型导出可设置 `EXECUTORCH_QWEN35_SPLIT_K_INT8=1`，选择独立 key W8、合并 q/v/a/b/z W4。

```bash
conda activate et_npu
source env.sh
python -u llama-output/qwen3_5-0_8b-layer-analysis/run_merged_gdn_layer128.py \
  --out llama-output/qwen3_5-0_8b-layer-analysis/merged_qvabz_layer128 --mode merged --device 3B1F57E9B8L74X5D
```

新增测试覆盖 1/128-token、非零 cache、key/value head 数不同的完整 attention 等价性和实际 recipe 命中；3 passed。尚未执行全模型 PPL。

## 产物

`metrics.csv` 保存 128 个 token 的阶段误差；`metadata.json` 保存节点、形状和校准编码；`qdq.pt2` 保存校准后图；`.pte` 和 `outputs/` 为实际设备运行产物。

128-token 阶段误差（历史产物已清理）

---

<a id="record-26"></a>

## 记录 26：merged_qvabz_layer128_baseline / REPORT.md

原始记录位置：`llama-output/qwen3_5-0_8b-layer-analysis/merged_qvabz_layer128_baseline/REPORT.md`。

# 完整 Qwen3.5 layer 0 / 128-token prefill 误差

## 配置

key 独立 W8A16；q/v/a/b/z 合并 W4A16，权重为 6176×1024；FFN gate/up 为 W4A16，down 为 W8A16。激活、conv cache 和 recurrent state 均为 UINT16。

8 段 WikiText：前 128 token 校准、后 128 token 评估，两个窗口独立零 cache。FP32、CPU QDQ、设备使用相同文本。attention 输出在 out_proj 之后；layer_output 为 FFN 后残差 hidden state，不包含词表 logits。

CPU 的 FFN same-input 对照只将 FFN 输入替换为 FP32 ffn_norm 输出再施加其真实 A16 编码，其他量化参数和 FFN 权重保持不变。

设备使用带检查点输出的完整单层 PTE，投影、causal conv、VTCM GDN、门控、out_proj、残差、FFN 均由同一图执行；额外输出可能影响调度，本测试用于精度分析。

## 各阶段 SQNR

| 阶段 | CPU QDQ / dB | 设备 / dB | 设备相对 CPU QDQ / dB | FFN 同输入 / dB |
|---|---:|---:|---:|---:|
| attention_norm | 64.181 | — | — | — |
| core_input_query | 22.595 | — | — | — |
| core_input_key | 21.065 | — | — | — |
| core_input_value | 32.205 | — | — | — |
| core_input_decay | 67.498 | — | — | — |
| core_input_beta | 30.645 | — | — | — |
| core_input_state | inf | — | — | — |
| gdn_core_output | 30.540 | — | — | — |
| gdn_norm_z_gate | 23.078 | — | — | — |
| attention_output | 21.720 | — | — | — |
| attention_residual | 21.447 | — | — | — |
| ffn_norm | 20.413 | — | — | 67.731 |
| ffn_gate_projection | 18.757 | — | — | 21.790 |
| ffn_up_projection | 16.162 | — | — | 19.223 |
| ffn_down_projection | 15.489 | — | — | 18.746 |
| ffn_output | 15.489 | — | — | 18.746 |
| layer_output | 18.923 | — | — | — |
| conv_state_output | 25.226 | — | — | — |
| recurrent_state_output | 21.502 | — | — | — |

## 实际 recipe

| 模块 | 精度 | 粒度 |
|---|---|---|
| `layers.0.attention.in_proj_qkv_z.conv` | use_16a4w_block | PER_BLOCK |
| `layers.0.attention.in_proj_b.conv` | use_16a4w_block | PER_BLOCK |
| `layers.0.attention.in_proj_a.conv` | use_16a8w | PER_CHANNEL |
| `layers.0.attention.causal_conv2d` | use_16a8w | PER_CHANNEL |
| `layers.0.attention.out_proj.conv` | use_16a8w | PER_CHANNEL |
| `layers.0.feed_forward.w1_conv` | use_16a4w_block | PER_BLOCK |
| `layers.0.feed_forward.w3_conv` | use_16a4w_block | PER_BLOCK |
| `layers.0.feed_forward.w2_conv` | use_16a8w | PER_CHANNEL |

## 产物

`metrics.csv` 保存 128 个 token 的阶段误差；`metadata.json` 保存节点、形状和校准编码；`qdq.pt2` 保存校准后图；`.pte` 和 `outputs/` 为实际设备运行产物。

128-token 阶段误差（历史产物已清理）

---

<a id="record-27"></a>

## 记录 27：one_layer_128 / REPORT.md

原始记录位置：`llama-output/qwen3_5-0_8b-layer-analysis/one_layer_128/REPORT.md`。

# Qwen3.5 layer 0 / 128-token PTQ error localization

- Graph: one `Qwen35DecoderLayer(0)` only; no other decoder layers are present.
- Calibration: eight WikiText-derived 128-token batches.
- Quantization: Qualcomm PT2E W8A16 with MinMaxObserver; linear and convolution weights use per-channel quantization.
- SQNR is computed at each common FP32/QDQ intermediate node.

## Operator groups with the lowest SQNR

| group | nodes | worst SQNR (dB) | worst node | max abs error | mean RMS error |
|---|---:|---:|---|---:|---:|
| `rsqrt` | 4 | -inf | `rsqrt_1` | inf | inf |
| `mul` | 782 | 14.404 | `mul_783` | 1.14519 | 0.00113626 |
| `linear` | 8 | 14.410 | `linear_7` | 0.856692 | 0.0254894 |
| `to.dtype` | 19 | 17.189 | `to_20` | 0.809039 | 0.0128986 |
| `add` | 138 | 17.277 | `add_138` | 0.277442 | 0.00154525 |
| `select` | 640 | 17.823 | `select_515` | 0.199579 | 0.00210899 |
| `expand_as` | 512 | 17.838 | `expand_as_415` | 0.199574 | 0.00236807 |
| `unsqueeze` | 643 | 17.838 | `unsqueeze_546` | 0.199574 | 0.0026741 |
| `sum` | 386 | 19.445 | `sum_140` | 0.27755 | 0.00253287 |
| `default` | 263 | 21.845 | `stack_1` | 0.230313 | 0.0034582 |
| `reshape` | 7 | 21.845 | `reshape_4` | 0.720417 | 0.0138704 |
| `sub` | 128 | 27.581 | `sub_87` | 0.126771 | 0.00171426 |
| `mean` | 2 | 28.602 | `mean_2` | 0.000207903 | 3.05347e-05 |
| `slice` | 129 | 30.500 | `slice_30` | 0.172483 | 0.0348941 |
| `cat` | 1 | 30.843 | `cat` | 0.172483 | 0.0346115 |
| `transpose` | 2 | 30.843 | `transpose` | 0.172483 | 0.0186812 |
| `exp` | 3 | 35.357 | `exp_1` | 0.0411927 | 0.00103017 |
| `sigmoid` | 1 | 47.006 | `sigmoid` | 0.00979513 | 0.0023324 |
| `dim` | 1 | 76.455 | `squeeze` | 1.88798e-05 | 1.09519e-05 |

## Layer outputs

| output | mean SQNR (dB) | worst SQNR (dB) | max abs error | mean RMS error |
|---|---:|---:|---:|---:|
| `layer_output` | 17.277 | 17.054 | 0.0821569 | 0.00533066 |
| `conv_state_output` | 30.834 | 29.806 | 0.151394 | 0.035173 |
| `recurrent_state_output` | 27.256 | 26.999 | 0.147895 | 0.00120134 |

## Worst individual nodes

| node | module path | target | shape | avg SQNR (dB) | max abs error | mean RMS error | cosine |
|---|---|---|---|---:|---:|---:|---:|
| `rsqrt_1` | `attention` | `aten.rsqrt.default` | `[1, 128, 16, 1]` | -inf | inf | inf | nan |
| `mul_783` | `feed_forward` | `aten.mul.Tensor` | `[1, 128, 3584]` | 14.404 | 0.73661 | 0.00773417 | 0.981901 |
| `linear_7` | `feed_forward.w2` | `aten.linear.default` | `[1, 128, 1024]` | 14.410 | 0.114168 | 0.00424807 | 0.981971 |
| `linear_6` | `feed_forward.w3` | `aten.linear.default` | `[1, 128, 3584]` | 15.790 | 0.390129 | 0.0357062 | 0.986846 |
| `to_20` | `ffn_norm` | `aten.to.dtype` | `[1, 128, 1024]` | 17.189 | 0.809039 | 0.127359 | 0.990445 |
| `mul_782` | `ffn_norm` | `aten.mul.Tensor` | `[1, 128, 1024]` | 17.189 | 0.808872 | 0.127359 | 0.990445 |
| `add_138` | `` | `aten.add.Tensor` | `[1, 128, 1024]` | 17.277 | 0.0821494 | 0.00533086 | 0.991048 |
| `mul_161` | `attention` | `aten.mul.Tensor` | `[1, 16, 128, 128]` | 17.822 | 0.004053 | 9.28094e-06 | 0.993541 |
| `select_515` | `attention` | `aten.select.int` | `[1, 16, 128]` | 17.823 | 0.0294277 | 0.00104746 | 0.991979 |
| `expand_as_415` | `attention` | `aten.expand_as.default` | `[1, 16, 128, 128]` | 17.838 | 0.0294277 | 0.00104562 | 0.992006 |
| `unsqueeze_546` | `attention` | `aten.unsqueeze.default` | `[1, 16, 128, 1]` | 17.838 | 0.0294277 | 0.00104562 | 0.992000 |
| `mul_166` | `attention` | `aten.mul.Tensor` | `[1, 16, 128, 128]` | 17.930 | 0.0064322 | 9.84691e-06 | 0.992959 |
| `linear_5` | `feed_forward.w1` | `aten.linear.default` | `[1, 128, 3584]` | 18.158 | 0.856692 | 0.0545989 | 0.992408 |
| `select_145` | `attention` | `aten.select.int` | `[1, 16, 128]` | 18.348 | 0.0254233 | 0.000980264 | 0.993085 |
| `select_540` | `attention` | `aten.select.int` | `[1, 16, 128]` | 18.424 | 0.0249625 | 0.00102379 | 0.991770 |
| `rsqrt_2` | `attention` | `aten.rsqrt.default` | `[1, 128, 16, 1]` | 18.430 | 10.3294 | 0.588794 | 0.997017 |
| `mul_156` | `attention` | `aten.mul.Tensor` | `[1, 16, 128, 128]` | 18.439 | 0.00357992 | 7.63936e-06 | 0.994226 |
| `mul_781` | `ffn_norm` | `aten.mul.Tensor` | `[1, 128, 1024]` | 18.439 | 1.14519 | 0.119651 | 0.992839 |
| `select_30` | `attention` | `aten.select.int` | `[1, 16, 128]` | 18.492 | 0.0304707 | 0.000989073 | 0.992598 |
| `select_105` | `attention` | `aten.select.int` | `[1, 16, 128]` | 18.499 | 0.0208085 | 0.000960373 | 0.993228 |
| `expand_as_119` | `attention` | `aten.expand_as.default` | `[1, 16, 128, 128]` | 18.524 | 0.0196385 | 0.000959583 | 0.993361 |
| `unsqueeze_250` | `attention` | `aten.unsqueeze.default` | `[1, 16, 128, 1]` | 18.524 | 0.0196385 | 0.000959583 | 0.993351 |
| `expand_as_435` | `attention` | `aten.expand_as.default` | `[1, 16, 128, 128]` | 18.526 | 0.0249636 | 0.00101786 | 0.991816 |
| `unsqueeze_566` | `attention` | `aten.unsqueeze.default` | `[1, 16, 128, 1]` | 18.526 | 0.0249636 | 0.00101786 | 0.991806 |
| `to_18` | `ffn_norm` | `aten.to.dtype` | `[1, 128, 1024]` | 18.597 | 0.0506775 | 0.00461458 | 0.993355 |
| `add_135` | `` | `aten.add.Tensor` | `[1, 128, 1024]` | 18.597 | 0.0506915 | 0.00461458 | 0.993356 |
| `mul_151` | `attention` | `aten.mul.Tensor` | `[1, 16, 128, 128]` | 18.645 | 0.00368546 | 6.18901e-06 | 0.995207 |
| `expand_as_87` | `attention` | `aten.expand_as.default` | `[1, 16, 128, 128]` | 18.654 | 0.0208083 | 0.000945392 | 0.993395 |
| `unsqueeze_218` | `attention` | `aten.unsqueeze.default` | `[1, 16, 128, 1]` | 18.654 | 0.0208083 | 0.000945392 | 0.993391 |
| `mul_171` | `attention` | `aten.mul.Tensor` | `[1, 16, 128, 128]` | 18.716 | 0.00377038 | 9.74649e-06 | 0.994204 |
| `expand_as_27` | `attention` | `aten.expand_as.default` | `[1, 16, 128, 128]` | 18.768 | 0.0249411 | 0.000959462 | 0.992964 |
| `unsqueeze_158` | `attention` | `aten.unsqueeze.default` | `[1, 16, 128, 1]` | 18.768 | 0.0249411 | 0.000959462 | 0.992959 |
| `mul_191` | `attention` | `aten.mul.Tensor` | `[1, 16, 128, 128]` | 18.861 | 0.00838115 | 1.33214e-05 | 0.993632 |
| `select_505` | `attention` | `aten.select.int` | `[1, 16, 128]` | 18.880 | 0.0211229 | 0.000944501 | 0.993319 |
| `linear_4` | `attention.out_proj` | `aten.linear.default` | `[1, 128, 1024]` | 18.911 | 0.0506895 | 0.00461458 | 0.993829 |
| `select_80` | `attention` | `aten.select.int` | `[1, 16, 128]` | 18.936 | 0.0259482 | 0.000951169 | 0.993003 |
| `select_550` | `attention` | `aten.select.int` | `[1, 16, 128]` | 18.950 | 0.0237495 | 0.000921026 | 0.993862 |
| `select_605` | `attention` | `aten.select.int` | `[1, 16, 128]` | 18.970 | 0.0239225 | 0.000906684 | 0.994113 |
| `select_215` | `attention` | `aten.select.int` | `[1, 16, 128]` | 18.988 | 0.0300431 | 0.00096749 | 0.992414 |
| `select_160` | `attention` | `aten.select.int` | `[1, 16, 128]` | 18.991 | 0.0235339 | 0.000908368 | 0.993954 |

---

<a id="record-28"></a>

## 记录 28：one_layer_128_fused_w4 / REPORT.md

原始记录位置：`llama-output/qwen3_5-0_8b-layer-analysis/one_layer_128_fused_w4/REPORT.md`。

# Qwen3.5 layer 0 / 128-token fused-GDN W4A16 PTQ error localization

- Graph: one `Qwen35DecoderLayer(0)` only; no other decoder layers are present.
- Calibration: eight WikiText-derived 128-token batches.
- Quantization: Qualcomm PT2E W4A16 using the production Qwen3.5 recipe (per-block W4 for qkv/FFN projections and per-channel W8 for sensitive projections), with the fused DeltaNet core annotation path.
- SQNR is computed at each common FP32/QDQ intermediate node.

## Operator groups with the lowest SQNR

| group | nodes | worst SQNR (dB) | worst node | max abs error | mean RMS error |
|---|---:|---:|---|---:|---:|
| `rsqrt` | 4 | -inf | `rsqrt_1` | inf | inf |
| `mul` | 14 | -3.814 | `mul_15` | 43.145 | 0.213974 |
| `conv` | 8 | -3.362 | `conv2d_7` | 9.71278 | 0.288143 |
| `transpose` | 8 | -3.362 | `transpose_7` | 17.082 | 0.255124 |
| `reshape` | 13 | -3.362 | `reshape_12` | 17.082 | 0.223205 |
| `add` | 10 | -2.189 | `add_10` | 42.9692 | 0.399214 |
| `to.dtype` | 18 | -1.741 | `to_19` | 17.082 | 0.18264 |
| `permute` | 4 | 0.248 | `permute_3` | 7.55767 | 0.321162 |
| `dim` | 8 | 0.248 | `squeeze_7` | 7.55767 | 0.190529 |
| `mean` | 2 | 2.221 | `mean_2` | 0.00390255 | 0.000636266 |
| `cat` | 1 | 4.742 | `cat` | 7.55767 | 0.698652 |
| `unsqueeze` | 4 | 4.742 | `unsqueeze` | 7.55767 | 0.19043 |
| `slice` | 2 | 4.742 | `slice_2` | 7.55767 | 0.696645 |
| `sum` | 2 | 5.095 | `sum_2` | 42.9688 | 1.73832 |
| `default` | 5 | 6.004 | `contiguous` | 2.67082 | 0.175841 |
| `exp` | 3 | 12.140 | `exp_1` | 0.556057 | 0.0154516 |
| `sigmoid` | 1 | 21.613 | `sigmoid` | 0.189204 | 0.0433877 |

## Layer outputs

| output | mean SQNR (dB) | worst SQNR (dB) | max abs error | mean RMS error |
|---|---:|---:|---:|---:|
| `layer_output` | -2.189 | -2.418 | 1.06833 | 0.0501241 |
| `conv_state_output` | 4.966 | 4.344 | 5.53201 | 0.691977 |
| `recurrent_state_output` | -0.177 | -0.401 | 3.41467 | 0.0282698 |

## Worst individual nodes

| node | module path | target | shape | avg SQNR (dB) | max abs error | mean RMS error | cosine |
|---|---|---|---|---:|---:|---:|---:|
| `rsqrt_1` | `attention` | `aten.rsqrt.default` | `[1, 128, 16, 1]` | -inf | inf | inf | nan |
| `rsqrt_2` | `attention` | `aten.rsqrt.default` | `[1, 128, 16, 1]` | -inf | inf | inf | nan |
| `mul_15` | `feed_forward` | `aten.mul.Tensor` | `[1, 3584, 1, 128]` | -3.814 | 3.28476 | 0.0629945 | 0.038160 |
| `conv2d_7` | `feed_forward.w2_conv` | `aten.conv2d.default` | `[1, 1024, 1, 128]` | -3.362 | 0.732305 | 0.03287 | 0.070176 |
| `transpose_7` | `feed_forward` | `aten.transpose.int` | `[1, 128, 1, 1024]` | -3.362 | 0.732309 | 0.03287 | 0.070176 |
| `reshape_12` | `feed_forward` | `aten.reshape.default` | `[1, 128, 1024]` | -3.362 | 0.732309 | 0.03287 | 0.070176 |
| `conv2d_6` | `feed_forward.w3_conv` | `aten.conv2d.default` | `[1, 3584, 1, 128]` | -2.295 | 3.17248 | 0.286417 | 0.280814 |
| `add_10` | `` | `aten.add.Tensor` | `[1, 128, 1024]` | -2.189 | 1.06834 | 0.0501241 | 0.093588 |
| `mul_14` | `ffn_norm` | `aten.mul.Tensor` | `[1, 128, 1024]` | -1.741 | 17.0821 | 1.12602 | 0.281593 |
| `transpose_6` | `feed_forward` | `aten.transpose.int` | `[1, 1024, 1, 128]` | -1.741 | 17.082 | 1.12602 | 0.281593 |
| `to_19` | `ffn_norm` | `aten.to.dtype` | `[1, 128, 1024]` | -1.741 | 17.082 | 1.12602 | 0.281593 |
| `reshape_11` | `feed_forward` | `aten.reshape.default` | `[1, 128, 1, 1024]` | -1.741 | 17.082 | 1.12602 | 0.281593 |
| `conv2d_5` | `feed_forward.w1_conv` | `aten.conv2d.default` | `[1, 3584, 1, 128]` | -1.491 | 9.71278 | 0.524434 | 0.158155 |
| `mul_13` | `ffn_norm` | `aten.mul.Tensor` | `[1, 128, 1024]` | -1.207 | 26.5835 | 1.14879 | 0.262360 |
| `to_12` | `attention` | `aten.to.dtype` | `[1, 16, 128, 128]` | -0.177 | 3.41467 | 0.0282698 | 0.426704 |
| `to_17` | `ffn_norm` | `aten.to.dtype` | `[1, 128, 1024]` | -0.066 | 1.18847 | 0.0395638 | 0.254940 |
| `add_7` | `` | `aten.add.Tensor` | `[1, 128, 1024]` | -0.066 | 1.18847 | 0.0395638 | 0.254941 |
| `mul_8` | `attention` | `aten.mul.Tensor` | `[1, 128, 16, 128]` | -0.001 | 1.27466 | 0.0883993 | 0.432805 |
| `mul_12` | `ffn_norm` | `aten.mul.Tensor` | `[1, 128, 1024]` | 0.115 | 1.78958 | 0.0164516 | 0.266872 |
| `conv2d_4` | `attention.out_proj.conv` | `aten.conv2d.default` | `[1, 1024, 1, 128]` | 0.248 | 1.18848 | 0.0395638 | 0.332933 |
| `permute_3` | `attention.out_proj` | `aten.permute.default` | `[1, 128, 1024, 1]` | 0.248 | 1.18847 | 0.0395638 | 0.332933 |
| `squeeze_7` | `attention.out_proj` | `aten.squeeze.dim` | `[1, 128, 1024]` | 0.248 | 1.18847 | 0.0395638 | 0.332933 |
| `mul_11` | `attention.norm` | `aten.mul.Tensor` | `[2048, 128]` | 0.420 | 4.92778 | 0.109376 | 0.318726 |
| `transpose_5` | `attention.out_proj` | `aten.transpose.int` | `[1, 2048, 1, 128]` | 0.420 | 4.83948 | 0.109372 | 0.318787 |
| `to_16` | `attention.norm` | `aten.to.dtype` | `[2048, 128]` | 0.420 | 4.83948 | 0.109372 | 0.318787 |
| `reshape_9` | `attention` | `aten.reshape.default` | `[1, 128, 2048]` | 0.420 | 4.83948 | 0.109372 | 0.318787 |
| `reshape_10` | `attention.out_proj` | `aten.reshape.default` | `[1, 128, 1, 2048]` | 0.420 | 4.83948 | 0.109372 | 0.318787 |
| `rsqrt_4` | `ffn_norm` | `aten.rsqrt.default` | `[1, 128, 1]` | 1.001 | 54.6991 | 24.0088 | 0.967025 |
| `to_11` | `attention` | `aten.to.dtype` | `[1, 128, 16, 128]` | 1.073 | 0.258473 | 0.00488822 | 0.512604 |
| `reshape_7` | `attention` | `aten.reshape.default` | `[2048, 128]` | 1.073 | 0.258473 | 0.00488822 | 0.512604 |
| `mul_7` | `attention` | `aten.mul.Tensor` | `[1, 128, 16, 128]` | 1.941 | 8.86016 | 0.0711045 | 0.761602 |
| `mean_2` | `ffn_norm` | `aten.mean.dim` | `[1, 128, 1]` | 2.221 | 0.00390255 | 0.00127253 | 0.846829 |
| `add_8` | `ffn_norm` | `aten.add.Tensor` | `[1, 128, 1]` | 2.226 | 0.00390257 | 0.00127253 | 0.847201 |
| `reshape_5` | `attention` | `aten.reshape.default` | `[1, 128, 16, 128]` | 2.292 | 2.35554 | 0.0613886 | 0.736809 |
| `to_8` | `attention` | `aten.to.dtype` | `[1, 128, 16, 128]` | 2.292 | 2.35554 | 0.0613886 | 0.736809 |
| `reshape_2` | `attention` | `aten.reshape.default` | `[1, 128, 16, 128]` | 2.663 | 4.0437 | 0.666774 | 0.689917 |
| `reshape_8` | `attention` | `aten.reshape.default` | `[2048, 128]` | 2.663 | 4.0437 | 0.666774 | 0.689917 |
| `to_15` | `attention.norm` | `aten.to.dtype` | `[2048, 128]` | 2.663 | 4.0437 | 0.666774 | 0.689917 |
| `mul_5` | `attention` | `aten.mul.Tensor` | `[1, 128, 16, 128]` | 2.731 | 1.64086 | 0.064544 | 0.729609 |
| `mul_6` | `attention` | `aten.mul.Tensor` | `[1, 128, 16, 128]` | 2.740 | 0.145033 | 0.00569958 | 0.729919 |

---

<a id="record-29"></a>

## 记录 29：one_layer_128_fused_w4_ffnw8 / REPORT.md

原始记录位置：`llama-output/qwen3_5-0_8b-layer-analysis/one_layer_128_fused_w4_ffnw8/REPORT.md`。

# Qwen3.5 layer 0 / 128-token W4A16 PTQ error localization

- Graph: one `Qwen35DecoderLayer(0)` only; no other decoder layers are present.
- Calibration: eight WikiText-derived 128-token batches.
- Quantization: Qualcomm PT2E W4A16 using the production Qwen3.5 recipe (per-block W4 for qkv/FFN projections and per-channel W8 for sensitive projections).
- SQNR is computed at each common FP32/QDQ intermediate node.

## Operator groups with the lowest SQNR

| group | nodes | worst SQNR (dB) | worst node | max abs error | mean RMS error |
|---|---:|---:|---|---:|---:|
| `rsqrt` | 4 | -inf | `rsqrt_1` | inf | inf |
| `mul` | 14 | -3.739 | `mul_15` | 43.145 | 0.213935 |
| `transpose` | 8 | -3.287 | `transpose_7` | 17.082 | 0.255088 |
| `reshape` | 13 | -3.287 | `reshape_12` | 17.082 | 0.223183 |
| `conv` | 8 | -3.287 | `conv2d_7` | 9.56258 | 0.287717 |
| `add` | 10 | -2.144 | `add_10` | 42.9692 | 0.399188 |
| `to.dtype` | 18 | -1.741 | `to_19` | 17.082 | 0.18264 |
| `permute` | 4 | 0.248 | `permute_3` | 7.55767 | 0.321162 |
| `dim` | 8 | 0.248 | `squeeze_7` | 7.55767 | 0.190529 |
| `mean` | 2 | 2.221 | `mean_2` | 0.00390255 | 0.000636266 |
| `cat` | 1 | 4.742 | `cat` | 7.55767 | 0.698652 |
| `unsqueeze` | 4 | 4.742 | `unsqueeze` | 7.55767 | 0.19043 |
| `slice` | 2 | 4.742 | `slice_2` | 7.55767 | 0.696645 |
| `sum` | 2 | 5.095 | `sum_2` | 42.9688 | 1.73832 |
| `default` | 5 | 6.004 | `contiguous` | 2.67082 | 0.175841 |
| `exp` | 3 | 12.140 | `exp_1` | 0.556057 | 0.0154516 |
| `sigmoid` | 1 | 21.613 | `sigmoid` | 0.189204 | 0.0433877 |

## Layer outputs

| output | mean SQNR (dB) | worst SQNR (dB) | max abs error | mean RMS error |
|---|---:|---:|---:|---:|
| `layer_output` | -2.144 | -2.372 | 1.05779 | 0.0498667 |
| `conv_state_output` | 4.966 | 4.344 | 5.53201 | 0.691977 |
| `recurrent_state_output` | -0.177 | -0.401 | 3.41467 | 0.0282698 |

## Worst individual nodes

| node | module path | target | shape | avg SQNR (dB) | max abs error | mean RMS error | cosine |
|---|---|---|---|---:|---:|---:|---:|
| `rsqrt_1` | `attention` | `aten.rsqrt.default` | `[1, 128, 16, 1]` | -inf | inf | inf | nan |
| `rsqrt_2` | `attention` | `aten.rsqrt.default` | `[1, 128, 16, 1]` | -inf | inf | inf | nan |
| `mul_15` | `feed_forward` | `aten.mul.Tensor` | `[1, 3584, 1, 128]` | -3.739 | 3.28396 | 0.0624558 | 0.039021 |
| `transpose_7` | `feed_forward` | `aten.transpose.int` | `[1, 128, 1, 1024]` | -3.287 | 0.695465 | 0.0325854 | 0.070438 |
| `reshape_12` | `feed_forward` | `aten.reshape.default` | `[1, 128, 1024]` | -3.287 | 0.695465 | 0.0325854 | 0.070438 |
| `conv2d_7` | `feed_forward.w2_conv` | `aten.conv2d.default` | `[1, 1024, 1, 128]` | -3.287 | 0.695467 | 0.0325854 | 0.070438 |
| `conv2d_6` | `feed_forward.w3_conv` | `aten.conv2d.default` | `[1, 3584, 1, 128]` | -2.251 | 3.15927 | 0.284971 | 0.282516 |
| `add_10` | `` | `aten.add.Tensor` | `[1, 128, 1024]` | -2.144 | 1.05779 | 0.0498667 | 0.095763 |
| `mul_14` | `ffn_norm` | `aten.mul.Tensor` | `[1, 128, 1024]` | -1.741 | 17.0821 | 1.12602 | 0.281593 |
| `transpose_6` | `feed_forward` | `aten.transpose.int` | `[1, 1024, 1, 128]` | -1.741 | 17.082 | 1.12602 | 0.281593 |
| `to_19` | `ffn_norm` | `aten.to.dtype` | `[1, 128, 1024]` | -1.741 | 17.082 | 1.12602 | 0.281593 |
| `reshape_11` | `feed_forward` | `aten.reshape.default` | `[1, 128, 1, 1024]` | -1.741 | 17.082 | 1.12602 | 0.281593 |
| `conv2d_5` | `feed_forward.w1_conv` | `aten.conv2d.default` | `[1, 3584, 1, 128]` | -1.463 | 9.56258 | 0.522759 | 0.159472 |
| `mul_13` | `ffn_norm` | `aten.mul.Tensor` | `[1, 128, 1024]` | -1.207 | 26.5835 | 1.14879 | 0.262360 |
| `to_12` | `attention` | `aten.to.dtype` | `[1, 16, 128, 128]` | -0.177 | 3.41467 | 0.0282698 | 0.426704 |
| `to_17` | `ffn_norm` | `aten.to.dtype` | `[1, 128, 1024]` | -0.066 | 1.18847 | 0.0395638 | 0.254940 |
| `add_7` | `` | `aten.add.Tensor` | `[1, 128, 1024]` | -0.066 | 1.18847 | 0.0395638 | 0.254941 |
| `mul_8` | `attention` | `aten.mul.Tensor` | `[1, 128, 16, 128]` | -0.001 | 1.27466 | 0.0883993 | 0.432805 |
| `mul_12` | `ffn_norm` | `aten.mul.Tensor` | `[1, 128, 1024]` | 0.115 | 1.78958 | 0.0164516 | 0.266872 |
| `conv2d_4` | `attention.out_proj.conv` | `aten.conv2d.default` | `[1, 1024, 1, 128]` | 0.248 | 1.18848 | 0.0395638 | 0.332933 |
| `permute_3` | `attention.out_proj` | `aten.permute.default` | `[1, 128, 1024, 1]` | 0.248 | 1.18847 | 0.0395638 | 0.332933 |
| `squeeze_7` | `attention.out_proj` | `aten.squeeze.dim` | `[1, 128, 1024]` | 0.248 | 1.18847 | 0.0395638 | 0.332933 |
| `mul_11` | `attention.norm` | `aten.mul.Tensor` | `[2048, 128]` | 0.420 | 4.92778 | 0.109376 | 0.318726 |
| `transpose_5` | `attention.out_proj` | `aten.transpose.int` | `[1, 2048, 1, 128]` | 0.420 | 4.83948 | 0.109372 | 0.318787 |
| `to_16` | `attention.norm` | `aten.to.dtype` | `[2048, 128]` | 0.420 | 4.83948 | 0.109372 | 0.318787 |
| `reshape_9` | `attention` | `aten.reshape.default` | `[1, 128, 2048]` | 0.420 | 4.83948 | 0.109372 | 0.318787 |
| `reshape_10` | `attention.out_proj` | `aten.reshape.default` | `[1, 128, 1, 2048]` | 0.420 | 4.83948 | 0.109372 | 0.318787 |
| `rsqrt_4` | `ffn_norm` | `aten.rsqrt.default` | `[1, 128, 1]` | 1.001 | 54.6991 | 24.0088 | 0.967025 |
| `to_11` | `attention` | `aten.to.dtype` | `[1, 128, 16, 128]` | 1.073 | 0.258473 | 0.00488822 | 0.512604 |
| `reshape_7` | `attention` | `aten.reshape.default` | `[2048, 128]` | 1.073 | 0.258473 | 0.00488822 | 0.512604 |
| `mul_7` | `attention` | `aten.mul.Tensor` | `[1, 128, 16, 128]` | 1.941 | 8.86016 | 0.0711045 | 0.761602 |
| `mean_2` | `ffn_norm` | `aten.mean.dim` | `[1, 128, 1]` | 2.221 | 0.00390255 | 0.00127253 | 0.846829 |
| `add_8` | `ffn_norm` | `aten.add.Tensor` | `[1, 128, 1]` | 2.226 | 0.00390257 | 0.00127253 | 0.847201 |
| `reshape_5` | `attention` | `aten.reshape.default` | `[1, 128, 16, 128]` | 2.292 | 2.35554 | 0.0613886 | 0.736809 |
| `to_8` | `attention` | `aten.to.dtype` | `[1, 128, 16, 128]` | 2.292 | 2.35554 | 0.0613886 | 0.736809 |
| `reshape_2` | `attention` | `aten.reshape.default` | `[1, 128, 16, 128]` | 2.663 | 4.0437 | 0.666774 | 0.689917 |
| `reshape_8` | `attention` | `aten.reshape.default` | `[2048, 128]` | 2.663 | 4.0437 | 0.666774 | 0.689917 |
| `to_15` | `attention.norm` | `aten.to.dtype` | `[2048, 128]` | 2.663 | 4.0437 | 0.666774 | 0.689917 |
| `mul_5` | `attention` | `aten.mul.Tensor` | `[1, 128, 16, 128]` | 2.731 | 1.64086 | 0.064544 | 0.729609 |
| `mul_6` | `attention` | `aten.mul.Tensor` | `[1, 128, 16, 128]` | 2.740 | 0.145033 | 0.00569958 | 0.729919 |

---

<a id="record-30"></a>

## 记录 30：one_layer_128_fused_w4_l2fixed / REPORT.md

原始记录位置：`llama-output/qwen3_5-0_8b-layer-analysis/one_layer_128_fused_w4_l2fixed/REPORT.md`。

# Qwen3.5 layer 0 / 128-token W4A16 PTQ error localization

- Graph: one `Qwen35DecoderLayer(0)` only; no other decoder layers are present.
- Calibration: eight WikiText-derived 128-token batches.
- Quantization: Qualcomm PT2E W4A16 using the production Qwen3.5 recipe (per-block W4 for qkv/FFN projections and per-channel W8 for sensitive projections).
- SQNR is computed at each common FP32/QDQ intermediate node.

## Operator groups with the lowest SQNR

| group | nodes | worst SQNR (dB) | worst node | max abs error | mean RMS error |
|---|---:|---:|---|---:|---:|
| `rsqrt` | 4 | -39.041 | `rsqrt_2` | 999.049 | 118.786 |
| `mul` | 14 | -3.813 | `mul_15` | 43.145 | 0.214182 |
| `conv` | 8 | -3.362 | `conv2d_7` | 9.71642 | 0.288139 |
| `transpose` | 8 | -3.362 | `transpose_7` | 17.0894 | 0.255109 |
| `reshape` | 13 | -3.362 | `reshape_12` | 17.0894 | 0.223192 |
| `add` | 10 | -2.188 | `add_10` | 42.9657 | 0.404944 |
| `to.dtype` | 18 | -1.740 | `to_19` | 17.0894 | 0.182626 |
| `permute` | 4 | 0.250 | `permute_3` | 7.55767 | 0.32116 |
| `dim` | 8 | 0.250 | `squeeze_7` | 7.55767 | 0.190528 |
| `mean` | 2 | 2.222 | `mean_2` | 0.00390515 | 0.000636186 |
| `cat` | 1 | 4.742 | `cat` | 7.55767 | 0.698652 |
| `unsqueeze` | 4 | 4.742 | `unsqueeze` | 7.55767 | 0.19043 |
| `slice` | 2 | 4.742 | `slice_2` | 7.55767 | 0.696645 |
| `sum` | 2 | 5.094 | `sum_2` | 42.9657 | 1.73828 |
| `default` | 5 | 6.004 | `contiguous` | 2.67082 | 0.175841 |
| `exp` | 3 | 12.140 | `exp_1` | 0.556057 | 0.0154516 |
| `sigmoid` | 1 | 21.613 | `sigmoid` | 0.189204 | 0.0433877 |

## Layer outputs

| output | mean SQNR (dB) | worst SQNR (dB) | max abs error | mean RMS error |
|---|---:|---:|---:|---:|
| `layer_output` | -2.188 | -2.415 | 1.07203 | 0.0501186 |
| `conv_state_output` | 4.966 | 4.344 | 5.53201 | 0.691977 |
| `recurrent_state_output` | -0.139 | -0.350 | 3.41529 | 0.0281462 |

## Worst individual nodes

| node | module path | target | shape | avg SQNR (dB) | max abs error | mean RMS error | cosine |
|---|---|---|---|---:|---:|---:|---:|
| `rsqrt_2` | `attention` | `aten.rsqrt.default` | `[1, 128, 16, 1]` | -39.041 | 999.049 | 439.154 | 0.831973 |
| `rsqrt_1` | `attention` | `aten.rsqrt.default` | `[1, 128, 16, 1]` | -13.942 | 363.633 | 11.9911 | 0.600898 |
| `mul_15` | `feed_forward` | `aten.mul.Tensor` | `[1, 3584, 1, 128]` | -3.813 | 3.28476 | 0.0629909 | 0.038178 |
| `conv2d_7` | `feed_forward.w2_conv` | `aten.conv2d.default` | `[1, 1024, 1, 128]` | -3.362 | 0.718131 | 0.0328683 | 0.070170 |
| `transpose_7` | `feed_forward` | `aten.transpose.int` | `[1, 128, 1, 1024]` | -3.362 | 0.718126 | 0.0328683 | 0.070170 |
| `reshape_12` | `feed_forward` | `aten.reshape.default` | `[1, 128, 1024]` | -3.362 | 0.718126 | 0.0328683 | 0.070170 |
| `conv2d_6` | `feed_forward.w3_conv` | `aten.conv2d.default` | `[1, 3584, 1, 128]` | -2.295 | 3.15225 | 0.286414 | 0.280835 |
| `add_10` | `` | `aten.add.Tensor` | `[1, 128, 1024]` | -2.188 | 1.07203 | 0.0501185 | 0.093693 |
| `mul_14` | `ffn_norm` | `aten.mul.Tensor` | `[1, 128, 1024]` | -1.740 | 17.0895 | 1.12595 | 0.281730 |
| `to_19` | `ffn_norm` | `aten.to.dtype` | `[1, 128, 1024]` | -1.740 | 17.0894 | 1.12595 | 0.281730 |
| `reshape_11` | `feed_forward` | `aten.reshape.default` | `[1, 128, 1, 1024]` | -1.740 | 17.0894 | 1.12595 | 0.281730 |
| `transpose_6` | `feed_forward` | `aten.transpose.int` | `[1, 1024, 1, 128]` | -1.740 | 17.0894 | 1.12595 | 0.281730 |
| `conv2d_5` | `feed_forward.w1_conv` | `aten.conv2d.default` | `[1, 3584, 1, 128]` | -1.491 | 9.71642 | 0.524408 | 0.158253 |
| `mul_13` | `ffn_norm` | `aten.mul.Tensor` | `[1, 128, 1024]` | -1.206 | 26.5952 | 1.14861 | 0.262686 |
| `mul_8` | `attention` | `aten.mul.Tensor` | `[1, 128, 16, 128]` | -0.201 | 1.2696 | 0.0904599 | 0.421729 |
| `to_12` | `attention` | `aten.to.dtype` | `[1, 16, 128, 128]` | -0.139 | 3.41529 | 0.0281462 | 0.427527 |
| `to_17` | `ffn_norm` | `aten.to.dtype` | `[1, 128, 1024]` | -0.064 | 1.19701 | 0.0395567 | 0.255308 |
| `add_7` | `` | `aten.add.Tensor` | `[1, 128, 1024]` | -0.064 | 1.197 | 0.0395567 | 0.255308 |
| `mul_12` | `ffn_norm` | `aten.mul.Tensor` | `[1, 128, 1024]` | 0.117 | 1.79222 | 0.0164477 | 0.270395 |
| `permute_3` | `attention.out_proj` | `aten.permute.default` | `[1, 128, 1024, 1]` | 0.250 | 1.19701 | 0.0395567 | 0.333747 |
| `squeeze_7` | `attention.out_proj` | `aten.squeeze.dim` | `[1, 128, 1024]` | 0.250 | 1.19701 | 0.0395567 | 0.333747 |
| `conv2d_4` | `attention.out_proj.conv` | `aten.conv2d.default` | `[1, 1024, 1, 128]` | 0.250 | 1.197 | 0.0395567 | 0.333748 |
| `mul_11` | `attention.norm` | `aten.mul.Tensor` | `[2048, 128]` | 0.423 | 4.92813 | 0.109337 | 0.319489 |
| `transpose_5` | `attention.out_proj` | `aten.transpose.int` | `[1, 2048, 1, 128]` | 0.423 | 4.83948 | 0.109334 | 0.319550 |
| `to_16` | `attention.norm` | `aten.to.dtype` | `[2048, 128]` | 0.423 | 4.83948 | 0.109334 | 0.319550 |
| `reshape_9` | `attention` | `aten.reshape.default` | `[1, 128, 2048]` | 0.423 | 4.83948 | 0.109334 | 0.319550 |
| `reshape_10` | `attention.out_proj` | `aten.reshape.default` | `[1, 128, 1, 2048]` | 0.423 | 4.83948 | 0.109334 | 0.319550 |
| `rsqrt_4` | `ffn_norm` | `aten.rsqrt.default` | `[1, 128, 1]` | 1.004 | 54.6737 | 23.9998 | 0.967050 |
| `to_11` | `attention` | `aten.to.dtype` | `[1, 128, 16, 128]` | 1.082 | 0.258466 | 0.00488333 | 0.510620 |
| `reshape_7` | `attention` | `aten.reshape.default` | `[2048, 128]` | 1.082 | 0.258466 | 0.00488333 | 0.510620 |
| `mul_7` | `attention` | `aten.mul.Tensor` | `[1, 128, 16, 128]` | 1.941 | 8.86016 | 0.0711045 | 0.761602 |
| `mean_2` | `ffn_norm` | `aten.mean.dim` | `[1, 128, 1]` | 2.222 | 0.00390515 | 0.00127237 | 0.846968 |
| `add_8` | `ffn_norm` | `aten.add.Tensor` | `[1, 128, 1]` | 2.227 | 0.00390517 | 0.00127237 | 0.847341 |
| `reshape_5` | `attention` | `aten.reshape.default` | `[1, 128, 16, 128]` | 2.292 | 2.35554 | 0.0613886 | 0.736809 |
| `to_8` | `attention` | `aten.to.dtype` | `[1, 128, 16, 128]` | 2.292 | 2.35554 | 0.0613886 | 0.736809 |
| `mul_5` | `attention` | `aten.mul.Tensor` | `[1, 128, 16, 128]` | 2.588 | 1.65322 | 0.0656201 | 0.724347 |
| `mul_6` | `attention` | `aten.mul.Tensor` | `[1, 128, 16, 128]` | 2.600 | 0.146125 | 0.00579217 | 0.724778 |
| `reshape_2` | `attention` | `aten.reshape.default` | `[1, 128, 16, 128]` | 2.663 | 4.0437 | 0.666774 | 0.689917 |
| `reshape_8` | `attention` | `aten.reshape.default` | `[2048, 128]` | 2.663 | 4.0437 | 0.666774 | 0.689917 |
| `to_15` | `attention.norm` | `aten.to.dtype` | `[2048, 128]` | 2.663 | 4.0437 | 0.666774 | 0.689917 |

---

<a id="record-31"></a>

## 记录 31：one_layer_128_fused_w4_l2float / REPORT.md

原始记录位置：`llama-output/qwen3_5-0_8b-layer-analysis/one_layer_128_fused_w4_l2float/REPORT.md`。

# Qwen3.5 layer 0 / 128-token W4A16 PTQ error localization

- Graph: one `Qwen35DecoderLayer(0)` only; no other decoder layers are present.
- Calibration: eight WikiText-derived 128-token batches.
- Quantization: Qualcomm PT2E W4A16 using the production Qwen3.5 recipe (per-block W4 for qkv/FFN projections and per-channel W8 for sensitive projections).
- SQNR is computed at each common FP32/QDQ intermediate node.

## Operator groups with the lowest SQNR

| group | nodes | worst SQNR (dB) | worst node | max abs error | mean RMS error |
|---|---:|---:|---|---:|---:|
| `rsqrt` | 4 | -39.041 | `rsqrt_2` | 999.049 | 118.789 |
| `mul` | 14 | -3.814 | `mul_15` | 43.145 | 0.213971 |
| `conv` | 8 | -3.362 | `conv2d_7` | 9.71283 | 0.288143 |
| `transpose` | 8 | -3.362 | `transpose_7` | 17.0845 | 0.255123 |
| `reshape` | 13 | -3.362 | `reshape_12` | 17.0845 | 0.223204 |
| `add` | 10 | -2.189 | `add_10` | 42.9657 | 0.404945 |
| `to.dtype` | 18 | -1.741 | `to_19` | 17.0845 | 0.182639 |
| `permute` | 4 | 0.248 | `permute_3` | 7.55767 | 0.321162 |
| `dim` | 8 | 0.248 | `squeeze_7` | 7.55767 | 0.190529 |
| `mean` | 2 | 2.221 | `mean_2` | 0.00390263 | 0.000636269 |
| `cat` | 1 | 4.742 | `cat` | 7.55767 | 0.698652 |
| `unsqueeze` | 4 | 4.742 | `unsqueeze` | 7.55767 | 0.19043 |
| `slice` | 2 | 4.742 | `slice_2` | 7.55767 | 0.696645 |
| `sum` | 2 | 5.094 | `sum_2` | 42.9657 | 1.73828 |
| `default` | 5 | 6.004 | `contiguous` | 2.67082 | 0.175841 |
| `exp` | 3 | 12.140 | `exp_1` | 0.556057 | 0.0154516 |
| `sigmoid` | 1 | 21.613 | `sigmoid` | 0.189204 | 0.0433877 |

## Layer outputs

| output | mean SQNR (dB) | worst SQNR (dB) | max abs error | mean RMS error |
|---|---:|---:|---:|---:|
| `layer_output` | -2.189 | -2.418 | 1.06868 | 0.0501239 |
| `conv_state_output` | 4.966 | 4.344 | 5.53201 | 0.691977 |
| `recurrent_state_output` | -0.178 | -0.402 | 3.41511 | 0.028271 |

## Worst individual nodes

| node | module path | target | shape | avg SQNR (dB) | max abs error | mean RMS error | cosine |
|---|---|---|---|---:|---:|---:|---:|
| `rsqrt_2` | `attention` | `aten.rsqrt.default` | `[1, 128, 16, 1]` | -39.041 | 999.049 | 439.154 | 0.831973 |
| `rsqrt_1` | `attention` | `aten.rsqrt.default` | `[1, 128, 16, 1]` | -13.942 | 363.633 | 11.9911 | 0.600898 |
| `mul_15` | `feed_forward` | `aten.mul.Tensor` | `[1, 3584, 1, 128]` | -3.814 | 3.28476 | 0.0629942 | 0.038161 |
| `conv2d_7` | `feed_forward.w2_conv` | `aten.conv2d.default` | `[1, 1024, 1, 128]` | -3.362 | 0.732448 | 0.0328698 | 0.070179 |
| `transpose_7` | `feed_forward` | `aten.transpose.int` | `[1, 128, 1, 1024]` | -3.362 | 0.732453 | 0.0328698 | 0.070179 |
| `reshape_12` | `feed_forward` | `aten.reshape.default` | `[1, 128, 1024]` | -3.362 | 0.732453 | 0.0328698 | 0.070179 |
| `conv2d_6` | `feed_forward.w3_conv` | `aten.conv2d.default` | `[1, 3584, 1, 128]` | -2.295 | 3.17257 | 0.286417 | 0.280816 |
| `add_10` | `` | `aten.add.Tensor` | `[1, 128, 1024]` | -2.189 | 1.06869 | 0.0501239 | 0.093589 |
| `mul_14` | `ffn_norm` | `aten.mul.Tensor` | `[1, 128, 1024]` | -1.741 | 17.0843 | 1.12602 | 0.281596 |
| `to_19` | `ffn_norm` | `aten.to.dtype` | `[1, 128, 1024]` | -1.741 | 17.0845 | 1.12602 | 0.281595 |
| `reshape_11` | `feed_forward` | `aten.reshape.default` | `[1, 128, 1, 1024]` | -1.741 | 17.0845 | 1.12602 | 0.281595 |
| `transpose_6` | `feed_forward` | `aten.transpose.int` | `[1, 1024, 1, 128]` | -1.741 | 17.0845 | 1.12602 | 0.281596 |
| `conv2d_5` | `feed_forward.w1_conv` | `aten.conv2d.default` | `[1, 3584, 1, 128]` | -1.491 | 9.71283 | 0.524433 | 0.158158 |
| `mul_13` | `ffn_norm` | `aten.mul.Tensor` | `[1, 128, 1024]` | -1.207 | 26.5874 | 1.14879 | 0.262365 |
| `to_12` | `attention` | `aten.to.dtype` | `[1, 16, 128, 128]` | -0.178 | 3.41511 | 0.028271 | 0.426700 |
| `to_17` | `ffn_norm` | `aten.to.dtype` | `[1, 128, 1024]` | -0.066 | 1.18856 | 0.0395637 | 0.254944 |
| `add_7` | `` | `aten.add.Tensor` | `[1, 128, 1024]` | -0.066 | 1.18856 | 0.0395637 | 0.254945 |
| `mul_8` | `attention` | `aten.mul.Tensor` | `[1, 128, 16, 128]` | 0.002 | 1.26961 | 0.0883726 | 0.432873 |
| `mul_12` | `ffn_norm` | `aten.mul.Tensor` | `[1, 128, 1024]` | 0.115 | 1.78961 | 0.0164517 | 0.266827 |
| `conv2d_4` | `attention.out_proj.conv` | `aten.conv2d.default` | `[1, 1024, 1, 128]` | 0.248 | 1.18857 | 0.0395637 | 0.332993 |
| `permute_3` | `attention.out_proj` | `aten.permute.default` | `[1, 128, 1024, 1]` | 0.248 | 1.18857 | 0.0395637 | 0.332993 |
| `squeeze_7` | `attention.out_proj` | `aten.squeeze.dim` | `[1, 128, 1024]` | 0.248 | 1.18857 | 0.0395637 | 0.332993 |
| `mul_11` | `attention.norm` | `aten.mul.Tensor` | `[2048, 128]` | 0.420 | 4.92821 | 0.109374 | 0.318749 |
| `transpose_5` | `attention.out_proj` | `aten.transpose.int` | `[1, 2048, 1, 128]` | 0.420 | 4.83948 | 0.109371 | 0.318810 |
| `to_16` | `attention.norm` | `aten.to.dtype` | `[2048, 128]` | 0.420 | 4.83948 | 0.109371 | 0.318810 |
| `reshape_9` | `attention` | `aten.reshape.default` | `[1, 128, 2048]` | 0.420 | 4.83948 | 0.109371 | 0.318810 |
| `reshape_10` | `attention.out_proj` | `aten.reshape.default` | `[1, 128, 1, 2048]` | 0.420 | 4.83948 | 0.109371 | 0.318810 |
| `rsqrt_4` | `ffn_norm` | `aten.rsqrt.default` | `[1, 128, 1]` | 1.001 | 54.6864 | 24.0092 | 0.967025 |
| `to_11` | `attention` | `aten.to.dtype` | `[1, 128, 16, 128]` | 1.073 | 0.258466 | 0.00488844 | 0.512522 |
| `reshape_7` | `attention` | `aten.reshape.default` | `[2048, 128]` | 1.073 | 0.258466 | 0.00488844 | 0.512522 |
| `mul_7` | `attention` | `aten.mul.Tensor` | `[1, 128, 16, 128]` | 1.941 | 8.86016 | 0.0711045 | 0.761602 |
| `mean_2` | `ffn_norm` | `aten.mean.dim` | `[1, 128, 1]` | 2.221 | 0.00390263 | 0.00127253 | 0.846825 |
| `add_8` | `ffn_norm` | `aten.add.Tensor` | `[1, 128, 1]` | 2.226 | 0.00390263 | 0.00127254 | 0.847197 |
| `reshape_5` | `attention` | `aten.reshape.default` | `[1, 128, 16, 128]` | 2.292 | 2.35554 | 0.0613886 | 0.736809 |
| `to_8` | `attention` | `aten.to.dtype` | `[1, 128, 16, 128]` | 2.292 | 2.35554 | 0.0613886 | 0.736809 |
| `reshape_2` | `attention` | `aten.reshape.default` | `[1, 128, 16, 128]` | 2.663 | 4.0437 | 0.666774 | 0.689917 |
| `reshape_8` | `attention` | `aten.reshape.default` | `[2048, 128]` | 2.663 | 4.0437 | 0.666774 | 0.689917 |
| `to_15` | `attention.norm` | `aten.to.dtype` | `[2048, 128]` | 2.663 | 4.0437 | 0.666774 | 0.689917 |
| `mul_5` | `attention` | `aten.mul.Tensor` | `[1, 128, 16, 128]` | 2.731 | 1.65322 | 0.0645502 | 0.729531 |
| `mul_6` | `attention` | `aten.mul.Tensor` | `[1, 128, 16, 128]` | 2.738 | 0.146125 | 0.00570088 | 0.729787 |

---

<a id="record-32"></a>

## 记录 32：one_layer_128_fused_w4_native_l2 / REPORT.md

原始记录位置：`llama-output/qwen3_5-0_8b-layer-analysis/one_layer_128_fused_w4_native_l2/REPORT.md`。

# Qwen3.5 layer 0 / 128-token W4A16 PTQ error localization

- Graph: one `Qwen35DecoderLayer(0)` only; no other decoder layers are present.
- Calibration: eight WikiText-derived 128-token batches.
- Quantization: Qualcomm PT2E W4A16 using the production Qwen3.5 recipe (per-block W4 for qkv/FFN projections and per-channel W8 for sensitive projections).
- SQNR is computed at each common FP32/QDQ intermediate node.

## Operator groups with the lowest SQNR

| group | nodes | worst SQNR (dB) | worst node | max abs error | mean RMS error |
|---|---:|---:|---|---:|---:|
| `mul` | 12 | -3.813 | `mul_13` | 26.5952 | 0.22595 |
| `conv` | 8 | -3.362 | `conv2d_7` | 9.71642 | 0.288139 |
| `transpose` | 8 | -3.362 | `transpose_7` | 17.0894 | 0.255109 |
| `reshape` | 13 | -3.362 | `reshape_12` | 17.0894 | 0.223192 |
| `add` | 8 | -2.188 | `add_8` | 2.2815 | 0.07161 |
| `to.dtype` | 18 | -1.740 | `to_19` | 17.0894 | 0.182626 |
| `default` | 7 | -0.201 | `rms_norm_1` | 18.7041 | 0.377864 |
| `int` | 4 | -0.201 | `movedim_3` | 2.62498 | 0.0734462 |
| `permute` | 4 | 0.250 | `permute_3` | 7.55767 | 0.32116 |
| `dim` | 8 | 0.250 | `squeeze_7` | 7.55767 | 0.190528 |
| `rsqrt` | 2 | 1.004 | `rsqrt_2` | 54.6737 | 12.0002 |
| `mean` | 2 | 2.222 | `mean_2` | 0.00390515 | 0.000636186 |
| `cat` | 1 | 4.742 | `cat` | 7.55767 | 0.698652 |
| `unsqueeze` | 4 | 4.742 | `unsqueeze` | 7.55767 | 0.19043 |
| `slice` | 2 | 4.742 | `slice_2` | 7.55767 | 0.696645 |
| `exp` | 3 | 12.140 | `exp_1` | 0.556057 | 0.0154516 |
| `sigmoid` | 1 | 21.613 | `sigmoid` | 0.189204 | 0.0433877 |

## Layer outputs

| output | mean SQNR (dB) | worst SQNR (dB) | max abs error | mean RMS error |
|---|---:|---:|---:|---:|
| `layer_output` | -2.188 | -2.415 | 1.07196 | 0.0501186 |
| `conv_state_output` | 4.966 | 4.344 | 5.53201 | 0.691977 |
| `recurrent_state_output` | -0.139 | -0.350 | 3.41529 | 0.0281462 |

## Worst individual nodes

| node | module path | target | shape | avg SQNR (dB) | max abs error | mean RMS error | cosine |
|---|---|---|---|---:|---:|---:|---:|
| `mul_13` | `feed_forward` | `aten.mul.Tensor` | `[1, 3584, 1, 128]` | -3.813 | 3.28476 | 0.0629908 | 0.038178 |
| `conv2d_7` | `feed_forward.w2_conv` | `aten.conv2d.default` | `[1, 1024, 1, 128]` | -3.362 | 0.718131 | 0.0328683 | 0.070170 |
| `transpose_7` | `feed_forward` | `aten.transpose.int` | `[1, 128, 1, 1024]` | -3.362 | 0.718126 | 0.0328683 | 0.070170 |
| `reshape_12` | `feed_forward` | `aten.reshape.default` | `[1, 128, 1024]` | -3.362 | 0.718126 | 0.0328683 | 0.070170 |
| `conv2d_6` | `feed_forward.w3_conv` | `aten.conv2d.default` | `[1, 3584, 1, 128]` | -2.295 | 3.15225 | 0.286414 | 0.280835 |
| `add_8` | `` | `aten.add.Tensor` | `[1, 128, 1024]` | -2.188 | 1.07197 | 0.0501185 | 0.093693 |
| `mul_12` | `ffn_norm` | `aten.mul.Tensor` | `[1, 128, 1024]` | -1.740 | 17.0895 | 1.12595 | 0.281730 |
| `transpose_6` | `feed_forward` | `aten.transpose.int` | `[1, 1024, 1, 128]` | -1.740 | 17.0894 | 1.12595 | 0.281730 |
| `to_19` | `ffn_norm` | `aten.to.dtype` | `[1, 128, 1024]` | -1.740 | 17.0894 | 1.12595 | 0.281730 |
| `reshape_11` | `feed_forward` | `aten.reshape.default` | `[1, 128, 1, 1024]` | -1.740 | 17.0894 | 1.12595 | 0.281730 |
| `conv2d_5` | `feed_forward.w1_conv` | `aten.conv2d.default` | `[1, 3584, 1, 128]` | -1.491 | 9.71642 | 0.524408 | 0.158253 |
| `mul_11` | `ffn_norm` | `aten.mul.Tensor` | `[1, 128, 1024]` | -1.206 | 26.5952 | 1.14861 | 0.262686 |
| `rms_norm_1` | `attention` | `aten.rms_norm.default` | `[1, 128, 16, 128]` | -0.201 | 14.3638 | 1.02344 | 0.421729 |
| `mul_6` | `attention` | `aten.mul.Tensor` | `[1, 128, 16, 128]` | -0.201 | 1.26858 | 0.0904569 | 0.421725 |
| `movedim_3` | `attention` | `aten.movedim.int` | `[1, 128, 16, 128]` | -0.201 | 1.26858 | 0.0904569 | 0.421725 |
| `to_12` | `attention` | `aten.to.dtype` | `[1, 16, 128, 128]` | -0.139 | 3.41529 | 0.0281462 | 0.427527 |
| `to_17` | `ffn_norm` | `aten.to.dtype` | `[1, 128, 1024]` | -0.064 | 1.19701 | 0.0395567 | 0.255308 |
| `add_5` | `` | `aten.add.Tensor` | `[1, 128, 1024]` | -0.064 | 1.197 | 0.0395567 | 0.255308 |
| `mul_10` | `ffn_norm` | `aten.mul.Tensor` | `[1, 128, 1024]` | 0.117 | 1.79222 | 0.0164477 | 0.270395 |
| `permute_3` | `attention.out_proj` | `aten.permute.default` | `[1, 128, 1024, 1]` | 0.250 | 1.19701 | 0.0395567 | 0.333747 |
| `squeeze_7` | `attention.out_proj` | `aten.squeeze.dim` | `[1, 128, 1024]` | 0.250 | 1.19701 | 0.0395567 | 0.333747 |
| `conv2d_4` | `attention.out_proj.conv` | `aten.conv2d.default` | `[1, 1024, 1, 128]` | 0.250 | 1.197 | 0.0395567 | 0.333749 |
| `mul_9` | `attention.norm` | `aten.mul.Tensor` | `[2048, 128]` | 0.423 | 4.92821 | 0.109337 | 0.319489 |
| `transpose_5` | `attention.out_proj` | `aten.transpose.int` | `[1, 2048, 1, 128]` | 0.423 | 4.83948 | 0.109334 | 0.319550 |
| `to_16` | `attention.norm` | `aten.to.dtype` | `[2048, 128]` | 0.423 | 4.83948 | 0.109334 | 0.319550 |
| `reshape_9` | `attention` | `aten.reshape.default` | `[1, 128, 2048]` | 0.423 | 4.83948 | 0.109334 | 0.319550 |
| `reshape_10` | `attention.out_proj` | `aten.reshape.default` | `[1, 128, 1, 2048]` | 0.423 | 4.83948 | 0.109334 | 0.319550 |
| `rsqrt_2` | `ffn_norm` | `aten.rsqrt.default` | `[1, 128, 1]` | 1.004 | 54.6737 | 23.9998 | 0.967050 |
| `to_11` | `attention` | `aten.to.dtype` | `[1, 128, 16, 128]` | 1.082 | 0.258466 | 0.00488333 | 0.510620 |
| `reshape_7` | `attention` | `aten.reshape.default` | `[2048, 128]` | 1.082 | 0.258466 | 0.00488333 | 0.510620 |
| `mean_2` | `ffn_norm` | `aten.mean.dim` | `[1, 128, 1]` | 2.222 | 0.00390515 | 0.00127237 | 0.846968 |
| `add_6` | `ffn_norm` | `aten.add.Tensor` | `[1, 128, 1]` | 2.227 | 0.00390517 | 0.00127237 | 0.847341 |
| `reshape_5` | `attention` | `aten.reshape.default` | `[1, 128, 16, 128]` | 2.292 | 2.35554 | 0.0613886 | 0.736809 |
| `to_8` | `attention` | `aten.to.dtype` | `[1, 128, 16, 128]` | 2.292 | 2.35554 | 0.0613886 | 0.736809 |
| `movedim_2` | `attention` | `aten.movedim.int` | `[1, 128, 16, 128]` | 2.294 | 2.35554 | 0.0613745 | 0.736819 |
| `rms_norm` | `attention` | `aten.rms_norm.default` | `[1, 128, 16, 128]` | 2.588 | 18.7041 | 0.742407 | 0.724347 |
| `movedim_1` | `attention` | `aten.movedim.int` | `[1, 128, 16, 128]` | 2.600 | 1.65321 | 0.0655309 | 0.724778 |
| `mul_5` | `attention` | `aten.mul.Tensor` | `[1, 128, 16, 128]` | 2.600 | 0.146125 | 0.00579217 | 0.724778 |
| `mul_4` | `attention` | `aten.mul.Tensor` | `[1, 128, 16, 128]` | 2.600 | 1.65321 | 0.0655309 | 0.724778 |
| `reshape_2` | `attention` | `aten.reshape.default` | `[1, 128, 16, 128]` | 2.663 | 4.0437 | 0.666774 | 0.689917 |

---

<a id="record-33"></a>

## 记录 33：one_layer_128_fused_w8 / REPORT.md

原始记录位置：`llama-output/qwen3_5-0_8b-layer-analysis/one_layer_128_fused_w8/REPORT.md`。

# Qwen3.5 layer 0 / 128-token fused-GDN W8A16 PTQ error localization

- Graph: one `Qwen35DecoderLayer(0)` only; no other decoder layers are present.
- Calibration: eight WikiText-derived 128-token batches.
- Quantization: Qualcomm PT2E fused-GDN W8A16 with MinMaxObserver; linear and convolution weights use per-channel quantization.
- SQNR is computed at each common FP32/QDQ intermediate node.

## Operator groups with the lowest SQNR

| group | nodes | worst SQNR (dB) | worst node | max abs error | mean RMS error |
|---|---:|---:|---|---:|---:|
| `rsqrt` | 4 | -inf | `rsqrt_1` | inf | inf |
| `mul` | 14 | 20.183 | `mul_5` | 0.791879 | 0.00692907 |
| `to.dtype` | 18 | 21.628 | `to_11` | 0.540031 | 0.00393745 |
| `reshape` | 13 | 21.628 | `reshape_7` | 0.540031 | 0.00453066 |
| `transpose` | 8 | 24.177 | `transpose_7` | 0.540031 | 0.0059377 |
| `conv` | 8 | 24.177 | `conv2d_7` | 0.868255 | 0.00746756 |
| `add` | 10 | 26.998 | `add_10` | 0.67952 | 0.00603922 |
| `permute` | 4 | 29.263 | `permute_3` | 0.136795 | 0.00717842 |
| `dim` | 8 | 29.263 | `squeeze_7` | 0.136795 | 0.00415628 |
| `mean` | 2 | 32.406 | `mean_2` | 0.000141676 | 1.97246e-05 |
| `exp` | 3 | 39.961 | `exp_1` | 0.0258459 | 0.000606729 |
| `sum` | 2 | 43.989 | `sum_2` | 0.679039 | 0.0197377 |
| `default` | 5 | 44.885 | `contiguous` | 0.0993087 | 0.0062493 |
| `slice` | 2 | 45.693 | `slice_1` | 0.120328 | 0.00630572 |
| `cat` | 1 | 45.736 | `cat` | 0.120328 | 0.00623086 |
| `unsqueeze` | 4 | 45.736 | `unsqueeze` | 0.120328 | 0.00216212 |
| `sigmoid` | 1 | 49.629 | `sigmoid` | 0.00721443 | 0.00172455 |

## Layer outputs

| output | mean SQNR (dB) | worst SQNR (dB) | max abs error | mean RMS error |
|---|---:|---:|---:|---:|
| `layer_output` | 26.998 | 25.876 | 0.0747057 | 0.00174969 |
| `conv_state_output` | 45.688 | 44.858 | 0.120263 | 0.00636089 |
| `recurrent_state_output` | 37.359 | 35.680 | 0.0669981 | 0.000377426 |

## Worst individual nodes

| node | module path | target | shape | avg SQNR (dB) | max abs error | mean RMS error | cosine |
|---|---|---|---|---:|---:|---:|---:|
| `rsqrt_1` | `attention` | `aten.rsqrt.default` | `[1, 128, 16, 1]` | -inf | inf | inf | nan |
| `rsqrt_2` | `attention` | `aten.rsqrt.default` | `[1, 128, 16, 1]` | 18.277 | 10.3294 | 0.599369 | 0.996942 |
| `mul_5` | `attention` | `aten.mul.Tensor` | `[1, 128, 16, 128]` | 20.183 | 0.376975 | 0.00867136 | 0.995959 |
| `mul_6` | `attention` | `aten.mul.Tensor` | `[1, 128, 16, 128]` | 20.184 | 0.03332 | 0.00076637 | 0.995961 |
| `to_11` | `attention` | `aten.to.dtype` | `[1, 128, 16, 128]` | 21.628 | 0.057658 | 0.000459624 | 0.996884 |
| `reshape_7` | `attention` | `aten.reshape.default` | `[2048, 128]` | 21.628 | 0.057658 | 0.000459624 | 0.996884 |
| `mul_15` | `feed_forward` | `aten.mul.Tensor` | `[1, 3584, 1, 128]` | 23.937 | 0.347204 | 0.00259628 | 0.997991 |
| `transpose_7` | `feed_forward` | `aten.transpose.int` | `[1, 128, 1, 1024]` | 24.177 | 0.120181 | 0.00139013 | 0.998060 |
| `reshape_12` | `feed_forward` | `aten.reshape.default` | `[1, 128, 1024]` | 24.177 | 0.120181 | 0.00139013 | 0.998060 |
| `conv2d_7` | `feed_forward.w2_conv` | `aten.conv2d.default` | `[1, 1024, 1, 128]` | 24.177 | 0.12018 | 0.00139012 | 0.998060 |
| `conv2d_6` | `feed_forward.w3_conv` | `aten.conv2d.default` | `[1, 3584, 1, 128]` | 26.438 | 0.354727 | 0.010502 | 0.998873 |
| `mul_8` | `attention` | `aten.mul.Tensor` | `[1, 128, 16, 128]` | 26.553 | 0.202177 | 0.00416113 | 0.999097 |
| `add_10` | `` | `aten.add.Tensor` | `[1, 128, 1024]` | 26.998 | 0.0746994 | 0.00174972 | 0.999035 |
| `to_19` | `ffn_norm` | `aten.to.dtype` | `[1, 128, 1024]` | 28.294 | 0.540031 | 0.0355158 | 0.999258 |
| `reshape_11` | `feed_forward` | `aten.reshape.default` | `[1, 128, 1, 1024]` | 28.294 | 0.540031 | 0.0355158 | 0.999258 |
| `transpose_6` | `feed_forward` | `aten.transpose.int` | `[1, 1024, 1, 128]` | 28.294 | 0.540031 | 0.0355158 | 0.999258 |
| `mul_14` | `ffn_norm` | `aten.mul.Tensor` | `[1, 128, 1024]` | 28.294 | 0.540208 | 0.0355155 | 0.999258 |
| `conv2d_5` | `feed_forward.w1_conv` | `aten.conv2d.default` | `[1, 3584, 1, 128]` | 28.314 | 0.868255 | 0.0170207 | 0.999288 |
| `to_17` | `ffn_norm` | `aten.to.dtype` | `[1, 128, 1024]` | 28.949 | 0.0454814 | 0.00140371 | 0.999434 |
| `add_7` | `` | `aten.add.Tensor` | `[1, 128, 1024]` | 28.949 | 0.0454958 | 0.00140369 | 0.999435 |
| `mul_13` | `ffn_norm` | `aten.mul.Tensor` | `[1, 128, 1024]` | 29.182 | 0.791879 | 0.0347885 | 0.999395 |
| `permute_3` | `attention.out_proj` | `aten.permute.default` | `[1, 128, 1024, 1]` | 29.263 | 0.0454977 | 0.00140369 | 0.999491 |
| `squeeze_7` | `attention.out_proj` | `aten.squeeze.dim` | `[1, 128, 1024]` | 29.263 | 0.0454977 | 0.00140369 | 0.999491 |
| `conv2d_4` | `attention.out_proj.conv` | `aten.conv2d.default` | `[1, 1024, 1, 128]` | 29.263 | 0.0454982 | 0.00140366 | 0.999491 |
| `mul_12` | `ffn_norm` | `aten.mul.Tensor` | `[1, 128, 1024]` | 29.821 | 0.0492687 | 0.000538892 | 0.999832 |
| `mul_11` | `attention.norm` | `aten.mul.Tensor` | `[2048, 128]` | 31.158 | 0.433495 | 0.00318696 | 0.999699 |
| `to_16` | `attention.norm` | `aten.to.dtype` | `[2048, 128]` | 31.159 | 0.433468 | 0.00318643 | 0.999699 |
| `reshape_9` | `attention` | `aten.reshape.default` | `[1, 128, 2048]` | 31.159 | 0.433468 | 0.00318643 | 0.999699 |
| `reshape_10` | `attention.out_proj` | `aten.reshape.default` | `[1, 128, 1, 2048]` | 31.159 | 0.433468 | 0.00318643 | 0.999699 |
| `transpose_5` | `attention.out_proj` | `aten.transpose.int` | `[1, 2048, 1, 128]` | 31.159 | 0.433468 | 0.00318643 | 0.999699 |
| `mean_2` | `ffn_norm` | `aten.mean.dim` | `[1, 128, 1]` | 32.406 | 0.000141676 | 3.94458e-05 | 0.999934 |
| `add_8` | `ffn_norm` | `aten.add.Tensor` | `[1, 128, 1]` | 32.413 | 0.000141669 | 3.94369e-05 | 0.999933 |
| `rsqrt_4` | `ffn_norm` | `aten.rsqrt.default` | `[1, 128, 1]` | 37.313 | 1.25237 | 0.367596 | 0.999983 |
| `to_12` | `attention` | `aten.to.dtype` | `[1, 16, 128, 128]` | 37.359 | 0.0669981 | 0.000377426 | 0.999953 |
| `mul_7` | `attention` | `aten.mul.Tensor` | `[1, 128, 16, 128]` | 38.338 | 0.53501 | 0.00109229 | 0.999952 |
| `exp_1` | `attention` | `aten.exp.default` | `[1, 128, 16]` | 39.961 | 0.0258459 | 0.00146567 | 0.999952 |
| `reshape_5` | `attention` | `aten.reshape.default` | `[1, 128, 16, 128]` | 41.436 | 0.0993087 | 0.000677763 | 0.999986 |
| `to_8` | `attention` | `aten.to.dtype` | `[1, 128, 16, 128]` | 41.436 | 0.0993087 | 0.000677763 | 0.999986 |
| `reshape_2` | `attention` | `aten.reshape.default` | `[1, 128, 16, 128]` | 43.048 | 0.136795 | 0.00637843 | 0.999978 |
| `reshape_8` | `attention` | `aten.reshape.default` | `[2048, 128]` | 43.048 | 0.136795 | 0.00637843 | 0.999978 |

---

<a id="record-34"></a>

## 记录 34：one_layer_128_w4 / REPORT.md

原始记录位置：`llama-output/qwen3_5-0_8b-layer-analysis/one_layer_128_w4/REPORT.md`。

# Qwen3.5 layer 0 / 128-token W4A16 PTQ error localization

- Graph: one `Qwen35DecoderLayer(0)` only; no other decoder layers are present.
- Calibration: eight WikiText-derived 128-token batches.
- Quantization: Qualcomm PT2E W4A16 using the production Qwen3.5 recipe (per-block W4 for qkv/FFN projections and per-channel W8 for sensitive projections).
- SQNR is computed at each common FP32/QDQ intermediate node.

## Operator groups with the lowest SQNR

| group | nodes | worst SQNR (dB) | worst node | max abs error | mean RMS error |
|---|---:|---:|---|---:|---:|
| `rsqrt` | 4 | -inf | `rsqrt_1` | inf | inf |
| `mul` | 654 | -4.958 | `mul_355` | 43.145 | 0.015999 |
| `transpose` | 8 | -3.355 | `transpose_7` | 17.1689 | 0.255119 |
| `reshape` | 13 | -3.355 | `reshape_12` | 17.1689 | 0.223237 |
| `conv` | 8 | -3.355 | `conv2d_7` | 9.70382 | 0.288103 |
| `sub` | 128 | -2.725 | `sub_53` | 4.56858 | 0.0580994 |
| `add` | 138 | -2.185 | `add_138` | 42.9692 | 0.0522121 |
| `to.dtype` | 18 | -1.737 | `to_19` | 17.1689 | 0.182718 |
| `default` | 262 | -0.602 | `expand_107` | 2.67082 | 0.0585066 |
| `unsqueeze` | 516 | -0.602 | `unsqueeze_218` | 7.55767 | 0.052637 |
| `select` | 640 | -0.275 | `select_581` | 2.19981 | 0.0383515 |
| `expand_as` | 512 | -0.219 | `expand_as_342` | 1.11768 | 0.0361917 |
| `sum` | 258 | 0.103 | `sum_4` | 42.9688 | 0.0424206 |
| `permute` | 4 | 0.252 | `permute_3` | 7.55767 | 0.321158 |
| `dim` | 5 | 0.252 | `squeeze_4` | 7.55767 | 0.292229 |
| `mean` | 2 | 2.220 | `mean_2` | 0.0039042 | 0.000636311 |
| `cat` | 1 | 4.742 | `cat` | 7.55767 | 0.698652 |
| `slice` | 2 | 4.742 | `slice_2` | 7.55767 | 0.696645 |
| `exp` | 3 | 12.140 | `exp_1` | 0.556057 | 0.0154516 |
| `sigmoid` | 1 | 21.613 | `sigmoid` | 0.189204 | 0.0433877 |

## Layer outputs

| output | mean SQNR (dB) | worst SQNR (dB) | max abs error | mean RMS error |
|---|---:|---:|---:|---:|
| `layer_output` | -2.185 | -2.418 | 1.08891 | 0.0501 |
| `conv_state_output` | 4.966 | 4.344 | 5.53201 | 0.691977 |
| `recurrent_state_output` | -0.624 | -0.807 | 3.76458 | 0.0297593 |

## Worst individual nodes

| node | module path | target | shape | avg SQNR (dB) | max abs error | mean RMS error | cosine |
|---|---|---|---|---:|---:|---:|---:|
| `rsqrt_1` | `attention` | `aten.rsqrt.default` | `[1, 128, 16, 1]` | -inf | inf | inf | nan |
| `rsqrt_2` | `attention` | `aten.rsqrt.default` | `[1, 128, 16, 1]` | -inf | inf | inf | nan |
| `mul_355` | `attention` | `aten.mul.Tensor` | `[1, 16, 128, 128]` | -4.958 | 1.83234 | 0.00636441 | 0.313900 |
| `mul_610` | `attention` | `aten.mul.Tensor` | `[1, 16, 128, 128]` | -4.951 | 1.94203 | 0.00615342 | 0.298013 |
| `mul_605` | `attention` | `aten.mul.Tensor` | `[1, 16, 128, 128]` | -4.875 | 1.809 | 0.00606854 | 0.372970 |
| `mul_350` | `attention` | `aten.mul.Tensor` | `[1, 16, 128, 128]` | -4.738 | 1.7954 | 0.00626605 | 0.332087 |
| `mul_330` | `attention` | `aten.mul.Tensor` | `[1, 16, 128, 128]` | -4.601 | 2.29867 | 0.00577722 | 0.339828 |
| `mul_360` | `attention` | `aten.mul.Tensor` | `[1, 16, 128, 128]` | -4.577 | 1.71318 | 0.00597655 | 0.385431 |
| `mul_485` | `attention` | `aten.mul.Tensor` | `[1, 16, 128, 128]` | -4.569 | 2.10325 | 0.00609257 | 0.389317 |
| `mul_420` | `attention` | `aten.mul.Tensor` | `[1, 16, 128, 128]` | -4.483 | 1.79995 | 0.0059787 | 0.353621 |
| `mul_445` | `attention` | `aten.mul.Tensor` | `[1, 16, 128, 128]` | -4.472 | 2.33447 | 0.0062665 | 0.423127 |
| `mul_340` | `attention` | `aten.mul.Tensor` | `[1, 16, 128, 128]` | -4.441 | 1.51323 | 0.0057713 | 0.401746 |
| `mul_40` | `attention` | `aten.mul.Tensor` | `[1, 16, 128, 128]` | -4.434 | 0.956579 | 0.00250117 | 0.330492 |
| `mul_565` | `attention` | `aten.mul.Tensor` | `[1, 16, 128, 128]` | -4.379 | 1.78805 | 0.00599137 | 0.407860 |
| `mul_590` | `attention` | `aten.mul.Tensor` | `[1, 16, 128, 128]` | -4.375 | 2.36363 | 0.00629672 | 0.412420 |
| `mul_365` | `attention` | `aten.mul.Tensor` | `[1, 16, 128, 128]` | -4.320 | 1.96718 | 0.00592944 | 0.362678 |
| `mul_130` | `attention` | `aten.mul.Tensor` | `[1, 16, 128, 128]` | -4.309 | 2.11203 | 0.00514977 | 0.367940 |
| `mul_435` | `attention` | `aten.mul.Tensor` | `[1, 16, 128, 128]` | -4.294 | 1.80558 | 0.00602902 | 0.408340 |
| `mul_45` | `attention` | `aten.mul.Tensor` | `[1, 16, 128, 128]` | -4.278 | 1.07724 | 0.00286817 | 0.387448 |
| `mul_585` | `attention` | `aten.mul.Tensor` | `[1, 16, 128, 128]` | -4.266 | 1.86066 | 0.00593673 | 0.379258 |
| `mul_190` | `attention` | `aten.mul.Tensor` | `[1, 16, 128, 128]` | -4.253 | 2.38368 | 0.00540965 | 0.327527 |
| `mul_510` | `attention` | `aten.mul.Tensor` | `[1, 16, 128, 128]` | -4.248 | 1.93525 | 0.00569712 | 0.416438 |
| `mul_525` | `attention` | `aten.mul.Tensor` | `[1, 16, 128, 128]` | -4.248 | 1.9669 | 0.00611121 | 0.372369 |
| `mul_480` | `attention` | `aten.mul.Tensor` | `[1, 16, 128, 128]` | -4.235 | 1.75208 | 0.00591367 | 0.392707 |
| `mul_375` | `attention` | `aten.mul.Tensor` | `[1, 16, 128, 128]` | -4.221 | 1.96276 | 0.00588174 | 0.366185 |
| `mul_635` | `attention` | `aten.mul.Tensor` | `[1, 16, 128, 128]` | -4.211 | 1.94642 | 0.00603026 | 0.372071 |
| `mul_470` | `attention` | `aten.mul.Tensor` | `[1, 16, 128, 128]` | -4.195 | 1.57543 | 0.00589767 | 0.417773 |
| `mul_25` | `attention` | `aten.mul.Tensor` | `[1, 16, 128, 128]` | -4.195 | 0.655144 | 0.0014812 | 0.383199 |
| `mul_265` | `attention` | `aten.mul.Tensor` | `[1, 16, 128, 128]` | -4.167 | 1.71601 | 0.00566074 | 0.407206 |
| `mul_395` | `attention` | `aten.mul.Tensor` | `[1, 16, 128, 128]` | -4.161 | 1.84958 | 0.00579391 | 0.359673 |
| `mul_305` | `attention` | `aten.mul.Tensor` | `[1, 16, 128, 128]` | -4.158 | 1.47016 | 0.00567347 | 0.423159 |
| `mul_500` | `attention` | `aten.mul.Tensor` | `[1, 16, 128, 128]` | -4.152 | 1.75323 | 0.00602554 | 0.387356 |
| `mul_380` | `attention` | `aten.mul.Tensor` | `[1, 16, 128, 128]` | -4.151 | 1.67328 | 0.00598458 | 0.353633 |
| `mul_595` | `attention` | `aten.mul.Tensor` | `[1, 16, 128, 128]` | -4.136 | 1.6741 | 0.00577913 | 0.379225 |
| `mul_475` | `attention` | `aten.mul.Tensor` | `[1, 16, 128, 128]` | -4.134 | 1.84496 | 0.00588016 | 0.416979 |
| `mul_540` | `attention` | `aten.mul.Tensor` | `[1, 16, 128, 128]` | -4.124 | 2.2225 | 0.00596879 | 0.400945 |
| `mul_405` | `attention` | `aten.mul.Tensor` | `[1, 16, 128, 128]` | -4.121 | 1.51568 | 0.00575922 | 0.386908 |
| `mul_460` | `attention` | `aten.mul.Tensor` | `[1, 16, 128, 128]` | -4.119 | 1.97965 | 0.006056 | 0.363554 |
| `mul_320` | `attention` | `aten.mul.Tensor` | `[1, 16, 128, 128]` | -4.117 | 1.95667 | 0.0057757 | 0.401703 |
| `mul_290` | `attention` | `aten.mul.Tensor` | `[1, 16, 128, 128]` | -4.110 | 1.54101 | 0.00570751 | 0.401626 |

---

<a id="record-35"></a>

## 记录 35：rms_z_gate_analysis / REPORT.md

原始记录位置：`llama-output/qwen3_5-0_8b-layer-analysis/rms_z_gate_analysis/REPORT.md`。

# GDN RMSNorm / z 门控误差归因：Qwen3.5 layer 0，128-token prefill

## 结论

这段是带学习权重的 RMSNorm，epsilon=1e-6，之后乘 SiLU(z)。RMSNorm 原生算子精度良好，norm/z 复合阶段的低 SQNR 主要来自上游输入误差：CPU PTQ 中 z 的 W4 投影误差占主要影响，设备上 GDN 输出误差经归一化和门控加权后也有明显贡献。

同设备实际输入的 FP32 公式对照：RMSNorm 输出 **67.392 dB**，SiLU(z) **73.510 dB**，最终乘法输出 **59.769 dB**。因此原生 RMSNorm/SiLU 的数值误差无法解释总体门控输出的 21.041 dB。

## 实现和测试方法

```python
variance = (x.float() ** 2).mean(-1, keepdim=True)
unit = x.float() * torch.rsqrt(variance + 1e-6)
normalized = unit * gamma
output = normalized * F.silu(z.float())
```

- normalization 维度为每个 token/head 的 128 个通道；gamma 是 128 个学习参数，这里使用 gamma 本身。
- 与 q/k 的 L2 norm 分别检查：本段按平方均值归一化，并使用学习权重；本段 epsilon 为 1e-6。
- PT2E `RecomposeRmsNorm` 已把原始 pow/mean/add/rsqrt/mul/weight 子图合为原生 `aten.rms_norm(input, [128], gamma, 1e-6)`；平方均值和逆平方根没有独立的量化节点。下文的 mean/rsqrt 是 FP32 数学中间量。
- 保持 key W8，合并 q/v/a/b/z W4 block=32，所有边界激活 A16。gamma 实际按 A16 配置量化，并非 W4；其权重 SQNR 为 105.625 dB。
- 沿用已经校准的完整单层 `qdq.pt2`。所有 scale/zero point 从该图读取，不重新拟合或写入新常数。
- 8 段 WikiText：前 128 token 校准，后 128 token 评估，独立零 cache。模型层 0，SM8750/v79。
- 新增设备检查点直接导出 norm 输入、z、RMSNorm 输出、sigmoid、SiLU、乘法结果；设备仍使用同一个 VTCM GDN kernel。
- 新探针与上轮设备图的 GDN 输出、norm/z 输出、最终 state 逐元素比较，最大差值均为 0，详见 `probe_consistency.json`。
- SQNR 先合并 8×128 token 的 signal/noise power，再计算 dB；不是逐样本 dB 的均值。不同节点的 dB 差值不能直接分配误差贡献。

## 1. 各阶段总体 SQNR

| 阶段 | CPU QDQ / dB | 设备 / dB |
|---|---:|---:|
| GDN / RMSNorm 输入 | 32.312 | 31.602 |
| RMSNorm 含 gamma 输出 | 34.240 | 30.428 |
| z 投影输出 | 22.354 | 21.924 |
| z cast/requant 后 | 22.353 | 21.924 |
| sigmoid(z) | 30.553 | 30.155 |
| SiLU(z) | 22.840 | 22.380 |
| RMSNorm × SiLU(z) | 23.144 | 21.041 |

CPU 的 RMSNorm 输出比输入 SQNR 更高；z 在进入门控前就已经只有约 22 dB。把归一化和门控合并观察，会遮住 z 分支的误差。

## 2. 同输入：算子自身误差

对 CPU 或设备实际收到的 x/z，计算相同 gamma 的 FP32 数学输出，再与量化图/设备输出比较。这个参考消除了前面的投影和 GDN 与原模型输入之间的差异。

| 算子输出 | CPU QDQ 同输入 / dB | 设备同输入 / dB |
|---|---:|---:|
| RMSNorm 含 gamma 输出 | 68.311 | 67.392 |
| SiLU(z) | 74.768 | 73.510 |
| RMSNorm × SiLU(z) | 62.279 | 59.769 |

gamma 单独替换为量化 gamma 时，门控输出 SQNR 为 104.197 dB，影响很小。

CPU 上直接给这一段原 FP32 x/z，再施加既有边界 QDQ，门控输出仍有 44.966 dB；这项包括输入/输出量化和少量范围越界。

## 3. 一路替换：GDN 输入还是 z？

其余输入保持 FP32、RMSNorm/SiLU/乘法均用 FP32，仅替换指定输入，统计最终 gated 输出。设备列使用设备实际 x 或 z 做这项数学对照。

| 仅替换的输入 | CPU 路径 / dB | 设备路径 / dB |
|---|---:|---:|
| GDN 输出 x | 28.903 | 23.834 |
| z | 24.473 | 24.227 |

CPU PTQ 中 z 的影响更大；设备路径中 GDN 输入和 z 的影响接近，都需要处理。这些独立替换实验不能按 dB 相加。

## 4. z 投影：权重还是激活/校准范围？

从已转换 QDQ 图取出实际反量化后的合并投影权重，切出末尾 2048 行 z 权重。分别只量化权重或只采用实际 A16 投影输入；其他部分保持 FP32。

| 对照 | z 输出 SQNR / dB | 最终 gated SQNR / dB |
|---|---:|---:|
| 仅 W4 z 权重 + FP32 输入 | 22.354 | 24.474 |
| FP32 z 权重 + 实际 A16 输入 | 69.767 | 69.351 |
| 正式 QDQ z 分支 | 22.354 | 24.473 |

z 权重本身 SQNR=19.957 dB，最大权重误差=0.03125。W4 的实际 integer range 为 [-7,7]，沿输入通道每 32 个权重一组。仅量化权重便复现了正式 z 分支约 22.35 dB 的输出误差；仅 A16 输入则约 69.77 dB。

因此当前这段的主要 PTQ 问题是 W4 z 权重离散化误差。当前评估文本上的 A16 校准范围和 RMSNorm 原生算子没有表现出同量级的误差。结论限于这 8 段文本，不能直接推广到全部任务。

## 5. FP32 min/max → 量化范围 → 饱和比例

越界比例按原 FP32 值是否超出可表达范围计数。

| 阶段 | FP32 min | FP32 max | 量化 min | 量化 max | 越界 % |
|---|---:|---:|---:|---:|---:|
| GDN / RMSNorm 输入 | -0.263958 | 0.198152 | -0.253376 | 0.176995 | 0.000143 |
| RMSNorm 含 gamma 输出 | -9.20187 | 9.30159 | -9.06491 | 9.30173 | 0.000191 |
| z 投影输出 | -9.19244 | 7.19384 | -29.3738 | 28.263 | 0.000000 |
| z cast/requant 后 | -9.19244 | 7.19384 | -7.52564 | 7.19389 | 0.000048 |
| sigmoid(z) | 0.000101796 | 0.999249 | 0 | 0.999985 | 0.000000 |
| SiLU(z) | -0.278465 | 7.18844 | -0.278464 | 7.18844 | 0.079012 |
| RMSNorm × SiLU(z) | -2.55726 | 2.55967 | -2.56154 | 2.51775 | 0.000286 |

FP32 RMSNorm 公式内的范围，未在 PT2E 中单独量化：

| 中间量 | min | max |
|---|---:|---:|
| mean_square | 3.711128e-16 | 0.0007276726 |
| denominator | 1e-06 | 0.0007286726 |
| inverse_rms | 37.04536 | 999.9999 |

epsilon 保证 denominator 为正，inverse RMS 上界为 1000。本次比较没有 NaN/Inf 数据元素；SQNR 表中的 inf 表示参考与实际完全一致。

## 6. 低幅值 token/head 的误差

按 FP32 x 的每 token/head RMS 分组，用设备 x 计算 FP32 RMSNorm，单独统计输入误差传播；不包括原生 RMSNorm 自身误差。

| FP32 RMS 分组 | 行数 | 行比例 % | 输入 SQNR / dB | norm SQNR / dB | norm 噪声占比 % |
|---|---:|---:|---:|---:|---:|
| <1e-4 | 1927 | 11.761 | 11.310 | 10.988 | 3.768 |
| 1e-4..1e-3 | 3363 | 20.526 | 22.718 | 22.643 | 36.323 |
| 1e-3..1e-2 | 9773 | 59.650 | 29.657 | 31.770 | 58.209 |
| >=1e-2 | 1321 | 8.063 | 33.304 | 39.900 | 1.700 |

聚合 GDN SQNR 更受大幅值 head 的 signal power 影响；归一化改变各 head 的权重，之后 SiLU(z) 再次按元素加权。低幅值 head 的输入误差因此不能只用整体 GDN SQNR 判断。这里观察到的是已有输入误差的传播，并非 RMSNorm 算子失去精度。

## 7. 产物与复现

- `metrics.csv`：各阶段、每 token 的误差及一路替换实验。
- `head_metrics.csv` / `head_summary.json`：每个 token/head 的 RMS、输入及 norm 误差功率。
- `metadata.json`：实际校准参数、节点、gamma 和 z 权重量化指标。
- `full_path_probe/`：有效设备 PTE、runner log、原始检查点输出。
- `captures.pt` / `z_quantized_weight.pt`：FP32/QDQ 中间值和实际反量化 z 权重。

额外尝试导出 standalone norm/gate 小图曾在 QNN Graph Optimizations 阶段 segfault，失败输出未用于本报告。这里的设备同输入指标来自有效全路径图的实测输入/输出与 FP32 公式对照；该全路径图与上轮完整层共同检查点逐元素一致。诊断脚本已只保留有效全路径设备导出和 CPU isolated 对照。

```bash
conda activate et_npu
source env.sh
python -u llama-output/qwen3_5-0_8b-layer-analysis/diagnose_gdn_rms_gate.py \
  --device 3B1F57E9B8L74X5D
python llama-output/qwen3_5-0_8b-layer-analysis/report_gdn_rms_gate.py
```

下一步应优先改善 z 的 W4 投影误差，并继续检查 GDN 的低幅值输出误差。此次只定位原因，未改变模型权重精度、RMSNorm 公式或 FFN 配置，未测试全模型 PPL。

128-token 分支与算子精度（历史产物已清理）

---

<a id="record-36"></a>

## 记录 36：qwen3_5-0_8b-ppl / REPORT.md

原始记录位置：`llama-output/qwen3_5-0_8b-ppl/REPORT.md`。

# Qwen3.5-0.8B：PTE teacher-forced PPL 对照

## 结果

2026-10-12，SM8750 设备，QNN SDK 2.49.40.260810。数值越低越好。

| 配置 | PPL | mean NLL (nats/token) | 实际下一词 top1 命中率 | 与 HF top1 一致率 |
|---|---:|---:|---:|---:|
| 原始权重 HF FP32 / CUDA | 16.8024 | 2.8215 | 45.62% | 100.00% |
| 静态 PyTorch FP32（同一转换权重，CPU） | 16.8024（1024 token：10.8777） | — | 45.62%（1024 token：56.21%） | 与 HF 一致 |
| 原 recipe，融合 qkv/z=W4，UInt16 state，VTCM | 153.4462 | 5.0334 | 25.93% | 36.65% |
| 全 decoder W8A16，融合 qkv/z，UInt16 state，VTCM | 123.7376 | 4.8182 | 26.64% | 39.78% |
| 全 decoder W8A16，融合 qkv/z，FP16 GDN/state，DDR | 51.6529 | 3.9445 | 35.60% | 53.31% |
| 全 decoder W8A16，FP16 GDN/state，WikiTrain 8 样本校准 | 37.8642 | — | — | — |
| 全 decoder W16A16，FP16 GDN/state，WikiTrain 8 样本校准 | 37.2138 | — | — | — |
| 全 decoder FP16，PyTorch GDN 分解，FP16 cache | **16.8295** | 2.8231 | — | — |

- 原 recipe 部分权重本来就是 W8，第一条 PTE 对照不能称为全部 decoder 权重都是 W4。
- W8 + UInt16 的 PPL 相比原 recipe 降低约 19.4%。
- W8 + FP16 GDN/state 相比 W8 + UInt16 的 PPL 降低 **58.3%**，mean NLL 减少 **0.8736 nats/token**，真实下一词 top1 命中率增加 **8.96 个百分点**。
- FP16 GDN/state 版 PPL 仍是 FP32 参考的 **3.07 倍**，尚未恢复原模型精度。
- FP16 版沿用上一轮实现：core 的 q/k/v/decay/beta 输入及 recurrent state 都是 FP16，内部采用 FP32 累加；上游投影保持 W8A16。它同时更换了 core 的数据路径，不能把全部收益单独归因为某一个 state buffer 的存储格式。
- 静态 PyTorch FP32 和 HF 在 128/1024 token 回归中逐 token 对齐，说明模型结构、转换权重、RoPE 和 GDN 数学实现不是 PPL 根因。
- 全 FP16 PTE 在设备上得到 16.8295，和 HF 16.8024 只相差 0.16%；因此当前 153.45/123.74/37.21 的大幅劣化来自整型 PTQ 执行路径，而不是设备 runner 或评测语料。
- 把 GDN 改成 FP16 后，W8 PPL 为 51.6529；再把 decoder 权重升到 W16，只有 37.2138，说明单独提高权重位宽不能恢复精度，激活量化和 scale 传播仍是主问题。

## 测量口径

- 语料：本地缓存的 `Salesforce/wikitext`、`wikitext-2-raw-v1` **test** 集，按 `"\n\n".join(dataset["text"])` 拼接，取本模型 tokenizer 的前 **4096 个 token**。不使用 PTQ 校准图片，也不使用模型自己生成的回答。
- 这是子集对照，**不是完整 WikiText-2 test 集 PPL**；也不评估图片理解质量。
- 冻结后的原始文本、tokenizer 快照、token IDs 和窗口在 `corpus/`。token 文件 SHA256：`2ebdb31fd4394419aabae16a635375c8e70d306c14462051d849056893cd2b27`。
- 窗口长度 1024，步长 512，共 7 个窗口。每个窗口从零 position 和新 cache 开始，重叠部分只作为上下文。
- 4095 个目标（原始位置 1..4095）每个恰好计分一次；第一个 token 不计分。不能平均各窗口的 PPL。
- 标准公式：`PPL = exp(sum(-log p(reference_next_token | reference_prefix)) / 4095)`。
- 每个 PTE 窗口把真实前缀逐 token 输入 kv/AR1 decoder，最后一个目标不作为输入。没有聊天模板、额外 BOS/EOS、图片输入、thinking controller、重复惩罚或模型生成反馈。
- HF 使用严格加载的本地 `Qwen3_5ForConditionalGeneration`、FP32 / CUDA、eager attention，禁用 TF32。
- PTE 使用原来的 embedding 与 decoder PTE，未重新导出或改变权重。vision PTE 由当前 runner 加载，但纯文本评测不调用 vision forward。
- NLL 使用完整 248320 维词表。UInt16 logits 通过实际 PTE 的 `get_logits_scale` 计分；公共 zero point 在稳定 logsumexp 中抵消，未指定固定量化参数。设备端用 float64 和指数 LUT 求和，主机重新核对全部逐 token NLL、计数与加权总和。
- 三种 PTE、embedding、vision、tokenizer 的本地及设备 SHA256 一致，明细在 `model_hashes.json`。

## 为什么 −1.283 dB 仍能生成文本

SQNR 衡量所有 logits 的数值差异，生成选词依赖候选词的相对大小。给全部 logits 加一个相同常数，会改变 SQNR，却不改变 softmax 或 argmax。这是指标性质的例子，不是在断言此前误差全部来自常数偏移。

即使部分选词排序改变，语言模型仍可连续选择常见词，形成语法通顺的句子；这不能证明内容、概率或原模型精度被保留。
PPL 针对固定参考文本的真实下一词概率，避免用“能吐字”作为质量判断。本次 FP16 GDN/state 的 PPL 收益明显，说明上一轮仅依据原图像 prompt logits SQNR 作出的精度判断不充分。
两种测试的输入也不同：此前是图片 chat prompt，这次是未加模板的纯文本测试语料，不能把两项指标当作同一份样本的可互换分数。

## 输入一致性与验证

初次文本路径运行在第三个窗口捕获到 C++ tokenizer 与 HF token IDs 不一致，立即停止；这次未完成结果不计入本表。
独立检查显示 WikiText 的 W000002、W000003、W000006 有分词差异，其他四个窗口一致。此前图片 chat prompt 的 214 个 token IDs **完全一致**，所以此前 SQNR 测量不受这项差异影响。

当前 multimodal runner 已支持 `MultimodalInput` 的预分词 token 输入。PPL driver 直接传入冻结的 uint64 IDs，跳过 C++ 二次分词；当前 CLI 的 `--tokenized_prompt` 前端没有在本次改动中接入。
32-token 回归验证中，直接 IDs 路径与原文本路径的每个 NLL 和 top1 预测完全一致。正式结果使用新 driver 从所有窗口重新运行。

## 新增隔离实验（2026-10-13）

### 静态 FP32 对照

使用同一份转换后的 321 个 state-dict 参数，直接调用静态 Qwen3.5 PyTorch 模型：128 token 的 PPL 为 21.6445（HF 21.6466），1024 token 的 PPL 为 10.8777（HF 10.8785），top1 结果逐 token 一致。该对照排除了模型转换、位置编码、GDN 公式和权重加载错误。

### 全 FP16 设备对照

KV、上下文 1024、PyTorch GDN 分解、FP16 recurrent state/conv cache/full-attention KV 的 PTE 已在 SM8750 完整跑完 4095 个目标，PPL 为 16.8295。7 个窗口分别为 10.8927、18.7513、22.1091、15.2547、21.1899、17.4716、23.1448，与 HF 窗口结果一致到 FP16 舍入误差范围。

### GDN-only 与 mixed FP16 尝试

已有“GDN FP16 activation/state + W8 权重”PTE 的 PPL 为 51.6529，没有达到全 FP16 的 16.83。尝试保留 depthwise convolution 为整型、只把其他节点切到 `use_fp16a8w` 时，QNN graph validation 失败：HTP 的 `DepthWiseConv2d` 不接受 FP16 activation + SFixed8 weight 的混合 IO；保留该卷积整型后，普通 Conv2d/Reshape/Transpose 又出现 mixed dtype 校验错误。因此当前 SDK/backend 不能用这种局部 mixed-FP16 recipe 直接修复 PPL。

### 同一评测语料校准的复核

我另外生成了 8 条来自固定 WikiText token 语料的 256-token 切片，并按 Qwen3.5 多模态格式加入本地图片和显式 image placeholder。直接使用纯文本消息会被多模态数据管线拒绝；补齐 placeholder 后，校准阶段可以完成，但 W16A16 + FP16 recurrent state 的 decoder 在 QNN 编译阶段触发 `tensor_generator.h:89: Lookup in 1 tensor types failed`，进程随后段错误，没有生成可运行 decoder PTE。这是当前 QNN 编译/混合类型路径的失败，不能把它当作 PPL 结果。

因此本报告采用已经成功编译的 WikiTrain 多模态校准结果：W8 PPL 37.8642、W16 PPL 37.2138。继续增加同分布校准样本前，必须先修复该编译器失败，否则无法形成有效 A/B。

同样尝试了不做 PTQ 的 `--use_fp16 --gdn_kernel ddr --gdn_state_dtype fp16`，decoder 在同一个 QNN `tensor_generator.h:89` 路径段错误；当前 16.8295 的正确性基线是 PyTorch GDN 分解。也就是说，DDR fused GDN 的 FP16 编译/类型路径本身还不能作为独立的浮点对照。

当前可复现的结论是：

1. 先修整型 activation PTQ/scale 传播和 state/cache 编码，不能只继续改 GDN C++ kernel；
2. GPTQ 只能优化 Linear 权重，不能修复已确认的整型 activation 路径，放在 state/cache 和 activation 误差闭环之后；
3. 全 FP16 是数值正确性基线，后续每个量化改动都必须以它和 HF 的 PPL 作为回归门槛。

- `comparison.json`：完整汇总、逐窗口 PPL、命中率、模型与 driver 指纹。
- `hf_fp32/`、`w4_u16/`、`w8_u16/`、`w8_fp16/`：各 7 个窗口的全部 token NLL、top1 和 summary。
- `native_token_comparison.json`：独立 C++ / HF tokenizer 比较。
- `native-tokenizer-partial/`：未计入汇总的首次未完成尝试。
- `smoke/`、`native-tokenizer-smoke/`：32-token 两条输入路径的数值回归。
- `hf.log`、`w4_u16.log`、`w8_u16.log`、`w8_fp16.log`：实际运行日志。

## 复跑

仓库根目录激活 `et_npu` 并 `source env.sh` 后：

```bash
export CUDA_VISIBLE_DEVICES=0
python llama-output/qwen3_5-0_8b-ppl/hf_reference.py
python llama-output/qwen3_5-0_8b-ppl/build_driver.py
```

构建复用当前 Android multimodal runner 的对象和库，只增加独立 PPL 入口，不采样后续模型输入。
若重新构建时修改了 runner 源码，先增量编译 `cmake --build build-android/examples/qualcomm --target qnn_multimodal_runner -j8`。

设备上三套原 PTE workspace 已就绪。以下是 FP16 版完整复跑；UInt16 版用 `full_fused_w8a16` 和 `w8_u16`，原 recipe 用 `full_fused_matched1024` 和 `w4_u16`：

```bash
adb -s 3B1F57E9B8L74X5D push \
  llama-output/qwen3_5-0_8b-ppl/qwen35_ppl_driver \
  llama-output/qwen3_5-0_8b-ppl/corpus/tokens.u64 \
  llama-output/qwen3_5-0_8b-ppl/w8_fp16.json \
  /data/local/tmp/hunter/executorch/qwen35_ppl/

adb -s 3B1F57E9B8L74X5D shell \
  'cd /data/local/tmp/hunter/executorch/full_fused_w8a16_fp16state && \
   export LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. && \
   /data/local/tmp/hunter/executorch/qwen35_ppl/qwen35_ppl_driver \
   /data/local/tmp/hunter/executorch/qwen35_ppl/w8_fp16.json \
   /data/local/tmp/hunter/executorch/qwen35_ppl/tokens.u64 \
   /data/local/tmp/hunter/executorch/qwen35_ppl/w8_fp16'

adb -s 3B1F57E9B8L74X5D pull \
  /data/local/tmp/hunter/executorch/qwen35_ppl/w8_fp16 \
  llama-output/qwen3_5-0_8b-ppl/

python llama-output/qwen3_5-0_8b-ppl/audit_results.py
```

脚本会覆盖自己的 case 输出，不具备跨版本 resume；需要保留历史结果时先复制或改用新的结果目录。

---

<a id="record-37"></a>

## 记录 37：qwen3_5-0_8b-split-z-sqnr / comparison.md

原始记录位置：`llama-output/qwen3_5-0_8b-split-z-sqnr/comparison.md`。

# Qwen3.5 qkv/z 拆分 SQNR 实测

后续不拆分 qkv/z 的全 decoder W8A16，以及 FP16 recurrent state 整模型对比见 [w8a16_fp16state.md](README_QWEN35_TEST_RECORDS.md#record-38)。

## 条件

- 模型：本地 Qwen3.5-0.8B，原始 FP32 权重作为参考。
- 独立 attention：第 0 层、batch=1、128 token，hidden=1024，16 个 value head，state=[1,16,128,128]。
- 与原 PTQ 误差报告对齐：随机种子 666、667，初始 conv/recurrent state 都为零；两个校准样本也用于测量，结果不代表未见样本的泛化精度。
- PT2E 校准使用 RTX 3090/CUDA，QNN 设备为 SM8750，SDK 2.49.40.260810。
- 设备上的四个独立 attention PTE 均使用当前 DeltaNetCoreVtcm op package。
- 生产 recipe：qkv/int4 LPBQ、block=(1,32,1,1)，a/out_proj/causal_conv 为 int8；z 在拆分版为 int8，融合版 z 包含于 int4 qkv_z。
- 所有 scale/zero point 都来自 observer 校准和实际 PTE 的编码；测量过程不手动指定量化范围。

## PT2E：与原表相同的计算口径

| 配置 | output seed666 (dB) | state seed666 (dB) | output seed667 (dB) | state seed667 (dB) |
|---|---:|---:|---:|---:|
| 原 uniform 16a16w | 18.1954 | 17.8627 | 18.4053 | 16.0435 |
| 原 uniform 16a8w | 17.9092 | 17.3924 | 18.1593 | 16.3613 |
| 原 uniform 16a4w | 5.5996 | 9.4771 | 5.3780 | 9.8815 |
| uniform int4，仅 z 改 int8 | 6.0828 | 9.4779 | 5.7988 | 9.8824 |
| 生产 recipe，未拆分 qkv/z | 11.0100 | 9.2839 | 11.6221 | 10.7257 |
| 生产 recipe，拆分 qkv=int4 / z=int8 | 11.5811 | 9.2839 | 12.2753 | 10.7257 |

## 独立 attention PTE：设备实测

| 配置 | output seed666 (dB) | state seed666 (dB) | output seed667 (dB) | state seed667 (dB) |
|---|---:|---:|---:|---:|
| 原 uniform 16a4w | 5.7035 | 10.2552 | 5.4861 | 9.6651 |
| uniform int4，仅 z 改 int8 | 6.1640 | 10.2623 | 5.8919 | 9.6640 |
| 生产 recipe，未拆分 qkv/z | 11.1365 | 11.4345 | 11.4104 | 12.1335 |
| 生产 recipe，拆分 qkv=int4 / z=int8 | 11.7487 | 11.4346 | 12.1198 | 12.1336 |

以上 PTE 是独立 128-token attention 测试图，使用生产 recipe 重新校准；不是从整模型 kv PTE 中直接读取的中间 attention 输出。

## 解释

- 相同生产 recipe 下，PT2E output 收益为 +0.5711/+0.6533 dB，设备 output 收益为 +0.6122/+0.7094 dB。对应两样本设备输出误差功率降低约 13.15%/15.07%。
- recurrent state 在相同 recipe 的融合/拆分对比中基本一致。z 用在 GDN 输出后的 RMSNorm + SiLU 门控，不进入 recurrence，不改变该层 state。
- 从原 uniform int4 的约 5.6 dB 到当前 recipe 的约 11.6 dB，包含 int4 分组量化、其他敏感权重 int8 和 observer 配置的收益，不能全部归因于 z=int8。

## 复跑独立 attention

在仓库根目录激活 et_npu 并 source env.sh 后运行：

```bash
CUDA_VISIBLE_DEVICES=0 python llama-output/qwen3_5-0_8b-split-z-sqnr/compare_attention.py --device
```

每种配置在独立 Python 进程中编译，避免单一进程连续编译图时出现的 QNN 图优化崩溃。

数值明细见 results.json；整模型结果见 full_pte_results.json。

## 真实整模型 kv PTE：相同 1024 配置对照

- 重新导出未拆分对照版：kv、max_seq_len=1024、VTCM、seq_mse_candidates=0、同一 vision.json/CUDA 校准。
- 设备推理使用相同图片预处理输入和精确的 214 个 chat token；参考为本地 HF FP32 原始权重模型。
- logits 按每个 PTE 自己保存的 scale/zero point 反量化。实际两个 1024 PTE 的 logits 编码也完全相同。
- multimodal runner 的 tokenized_prompt 目前未实现，因此测试显式传入与 FP32 参考一致的 Describe this image.，并断言设备 prompt token 数为 214。

| PTE | 全 prompt logits SQNR (dB) | 最后 prompt token logits SQNR (dB) | 平均 cosine | top1 一致数 |
|---|---:|---:|---:|---:|
| 未拆分 qkv_z=int4 | 0.8739 | -0.9668 | 0.5691 | 13/214 |
| 拆分 qkv=int4 / z=int8 | 0.9675 | 0.4119 | 0.5978 | 14/214 |

整模型 logits SQNR 增加 0.0936 dB，提升小，仍低于 1 dB。单层输出改善不能视为整模型精度问题已解决。

旧的 512 PTE 比较仅作为历史记录保留在 JSON，主结论使用重新导出的 1024 对照版。

复跑同配置整模型对照（不重新导出）：

```bash
CUDA_VISIBLE_DEVICES=0 python llama-output/qwen3_5-0_8b-split-z-sqnr/compare_full_pte.py --matched-control --reuse-reference
```

---

<a id="record-38"></a>

## 记录 38：qwen3_5-0_8b-split-z-sqnr / w8a16_fp16state.md

原始记录位置：`llama-output/qwen3_5-0_8b-split-z-sqnr/w8a16_fp16state.md`。

# Qwen3.5：融合 qkv/z、W8A16 和 FP16 recurrent state

后续已完成 [WikiText-2 文本子集 PPL 对照](README_QWEN35_TEST_RECORDS.md#record-36)：HF FP32 为 16.80，原 recipe 为 153.45，W8 + UInt16 state 为 123.74，W8 + FP16 GDN/state 为 51.65。FP16 版有明显的文本 PPL 收益；下面的 SQNR 结论只适用于此前那一条图片 prompt。

## 测量条件

- 日期：2026-10-12；模型：`/hunter/model/Qwen/Qwen3.5-0.8B`。
- 设备：SM8750，序列号 `3B1F57E9B8L74X5D`；QNN SDK 2.49.40.260810。
- 三个配置都是未拆分的 qkv/z、kv 模式、AR=1、max_seq_len=1024。
- 相同 `vision.json`、CUDA 校准，`seq_mse_candidates=0`。
- 原 recipe 的 qkv/z 为 W4 分组量化，部分其他权重已经是 W8；新实验把所有 decoder conv2d 权重改为 W8、每输出通道量化，其他激活采用 A16 和 MinMaxObserver。
- PTE 设备输入为同一张图片的相同预处理像素、相同的 214 个 chat token，prompt 为 `Describe this image.`。其中 196 个位置是图片 token。
- 参考为本地 HF 原始权重 FP32 模型。设备 logits 按每个 PTE 自己保存的 `get_logits_scale` / `get_logits_zero_point` 反量化，没有手动指定 scale 或 zero point。
- SQNR = `10*log10(sum(reference**2)/sum((reference-device)**2))`，对全部 248320 个词表 logits 计算。

## 真实整模型 PTE 结果

| 配置 | 全 prompt logits SQNR (dB) | 最后 token logits SQNR (dB) | 平均 cosine | top1 一致数 |
|---|---:|---:|---:|---:|
| 原 recipe，qkv/z=W4，UInt16 state | 0.8739 | -0.9668 | 0.5691 | 13/214 |
| 全 decoder W8A16，UInt16 state，VTCM | 0.8289 | 0.5548 | 0.5576 | 10/214 |
| 全 decoder W8A16，FP16 state，DDR | 0.8755 | -1.2825 | 0.5367 | 10/214 |

补充指标：只统计 18 个非图片 prompt 位置的 logits SQNR，依次为 0.9788 / 1.8316 / 2.2420 dB。
最后 token 的 softmax KL(reference || device) 依次为 3.6045 / 0.6108 / 8.2509；最后 token 的 top10 交集为 3 / 3 / 2。
两版新 PTE 都能连续生成英文，但上述数值没有证明其 logits 与 FP32 一致。只有这一张图片与这一条 prompt，不能作为通用模型质量评估。

**结论：提高权重位宽、再改为 FP16 recurrent state，在本次输入上都没有显著提高全 prompt logits SQNR。FP16 版最后 token 的数值一致性也比 UInt16 版差。此测试尚不能定位剩余误差来自哪个层或哪个操作。**

## FP16 state 实际如何实现

- 参数：`--gdn_kernel ddr --gdn_state_dtype fp16`。
- 复用现有 `DeltaNetCore` 的 FP16 QHPI kernel，FP16 存储、FP32 乘加/累加；没有使用逐次 FP16 算术舍入的 `DeltaNetCoreFp16Accum`。
- q/k/v/decay/beta 也在 core 边界转为 FP16。上游投影和激活仍采用 W8A16。因此这不是只修改 UInt16 VTCM kernel 内某一个 state buffer 的严格单变量实验。
- PTE 核实：18 个 recurrent state 输入和输出都是 `HALF`，这些 cache 没有整数 scale/zero point 方法。其他 30 个 cache 仍使用校准得到的 UInt16 编码。
- runner 根据每一层 K/V 各自的 dtype 初始化和传递 cache。FP16 state 初始化为真正的浮点零，下一步读取上一步的 FP16 state。
- 包装器必须在保存 `example_input` 和 `export_input` 之前设置 FP16 标志。首次尝试漏掉这个初始化顺序，产生 UInt16 输入 / FP16 输出的无效 PTE，runner 正确拒绝了它；修复后重新导出才用于本表。
- 此版有 37 个 QNN delegate；18 个 core 在 HTP 上执行。每个 decode step 有 90 次 `FLOAT->HALF` 和 18 次 `HALF->FLOAT` CPU cast。已检查 PTE 指令，CPU cast 不包含原始 UInt16 编码到浮点的直接数值转换。
- 此版复用 DDR FP16 kernel，不能视作最快 UInt16 VTCM kernel 的最终性能版。

类型检查明细见 `fp16_state_pte_io.json`，6 项 cache 初始化/编码对齐测试通过。

## 连续生成

相同图片与 prompt，temperature=0，seq_len=278，两版均生成了 63 个后续 token，因长度上限停止。

- W8A16 + UInt16：以 `This image doesn't matter for the description of you, because it's a picture of a cat.` 开头。
- W8A16 + FP16：以 `**Image Description:**` 开头，描述两只猫、床、电视和遥控器，随后出现不准确的细节。

完整输出分别在 `full_fused_w8a16/generation.txt` 和 `full_fused_w8a16_fp16state/generation.txt`。
生成结果会因第一个不同 token 而分叉，不能替代相同输入下的 logits 数值比较。

## 导出与复跑

在仓库根目录运行。实验脚本明确关闭 qkv/z 拆分，并使用独立 W8A16 recipe；原生产 W4 recipe 没有被整体替换。

```bash
source /home/hunter/anaconda3/etc/profile.d/conda.sh
conda activate et_npu
source env.sh
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH="$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/lib/x86_64-unknown-linux-gnu:$LD_LIBRARY_PATH"

python -u llama-output/qwen3_5-0_8b-split-z-sqnr/export_w8a16.py \
  -a ./llama-output/qwen3_5-0_8b-fused-w8a16-kv1024-fp16state \
  --build_folder build-android \
  --soc_model SM8750 \
  --model_mode kv \
  --gdn_kernel ddr \
  --gdn_state_dtype fp16 \
  --gdn_op_package_dir ./examples/qualcomm/custom_op/deltanet_op_package_htp/DeltaNetOpPackage \
  --max_seq_len 1024 \
  --prefill_ar_len 32 \
  --decoder_model qwen3_5-0_8b \
  --calibration_device cuda \
  --prompt "Describe this image." \
  --calib_samples examples/qualcomm/oss_scripts/llama/assets/samples/vision.json \
  --source_safetensors /hunter/model/Qwen/Qwen3.5-0.8B \
  --skip_qdq_save \
  --compile_only

python llama-output/qwen3_5-0_8b-split-z-sqnr/inspect_fp16_state.py

python llama-output/qwen3_5-0_8b-split-z-sqnr/compare_full_pte.py \
  --reuse-reference \
  --pte-dir llama-output/qwen3_5-0_8b-fused-w8a16-kv1024-fp16state \
  --case full_fused_w8a16_fp16state
```

UInt16 state 版：将导出目录改为 `qwen3_5-0_8b-fused-w8a16-kv1024-vtcm`，参数改为 `--gdn_kernel vtcm --gdn_state_dtype uint16`，比较 case 用 `full_fused_w8a16`。

## 文件

- `full_pte_results.json`：设备测量、PTE 实际量化参数、实验条件。
- `logits_additional_metrics.json`：非图片位置、去均值 SQNR、softmax KL 等补充指标。
- `w8a16_export.log` / `w8a16_eval_new_runner.log`：W8A16 导出与设备测试。新 runner 复跑结果与修改 cache 支持之前完全一致。
- `fp16_state_export.log` / `fp16_state_eval.log`：修复后 FP16 state 导出与设备测试。
- `fp16_cache_tests.log` / `fp16_runner_build.log`：Python 验证与 Android runner 编译。
- `fp16_state_export.input-u16-invalid.log` / `fp16_state_pte_io.input-u16-invalid.json`：未用于 SQNR 表的首次无效尝试。

---

<a id="record-39"></a>

## 记录 39：cross-model-review / REPORT.md

原始记录位置：`llama-output/qwen3_5-mainline-integration/cross-model-review/REPORT.md`。

# 其他模型兼容性 Review

日期：2026-10-15。检查当前工作区的集成改动，重点为模型配置、MHA/SHA、SWA、OP选择、mask容量及截断。**本轮没有修改实现代码。**

后续状态：用户已确认 Lincal3.5 采用 Qwen3.5 的优化 GDN 与量化流程。下文保留初次 review 的发现和当时的建议；最终采用共享优化流程并适配 SWA，三个问题的修复及验证见[修复报告](README_QWEN35_TEST_RECORDS.md#record-01)。

## Findings

### P1：Qwen 专属物理 padding 被应用到 Lincal3.5，默认校准失败

位置：`examples/qualcomm/oss_scripts/llama/llama.py:1240`，关联 `gdn.py:78`、`wrappers/base_component.py:103`、`model/static_llama.py:1628`。

`is_qwen35_model()`同时包含`qwen3_5`和`lincal3_5`，入口因此给Lincal默认上下文加128；但`process_model_args()`仅对`qwen3_5`采用AR128校准及物理长度，其余仍使用完整上下文校准、逻辑max_seq_len。Lincal3.5自动启用true SWA，SWA模型的causal mask用逻辑max_seq_len创建。

已复现两个注册版本：逻辑长度1024，默认物理长度1152，校准AR1152，创建causal mask时请求`[1152,-128]`，报`RuntimeError: Trying to create tensor with negative dimension -128`。这是默认配置即可触发的导出阻断。

建议将“需要GDN算子包”和“使用优化Qwen导出流程”分开判断，Qwen专属padding/AR限制只作用于后一类，保留Lincal原有校准和SWA配置。不要直接把Lincal改送到假设单mask/full-attention输入的Qwen生成图路径。

### P1：通用多模态 runner 接受了最后一个 AR 分块越界的 prompt

位置：`examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/multimodal_prompt_processor.cpp:271`，关联同文件的`prepare_io()`和`KVManager::update_cache()`。

入口从检查`context_len-ar_len`改为检查更大的decode cache容量，扩容代码解决了最后分块输出的cache存储，但没有让普通多模态模型的prefill图获得更多history槽位或RoPE位置。Qwen优化入口有额外物理padding和内部容量检查；其他模型的`max_prefill_past`仍默认INT32_MAX。

例：Qwen2.5-VL等采用`LlamaModelWithoutEmbedding`的模型，context1000、AR128、prompt999会被新检查接受。最后一块从896开始，需要896个past槽位，但图只有872个；`prepare_io`还生成896..1023的位置，超过长度1000的RoPE表。已用同一模型类缩小复现：context15、AR4、prompt14被新条件接受，最后位置12..15导致`index 15 is out of bounds for dimension 0 with size 15`。

建议对所有prefill图同时检查最后分块的起点和图的有效history/位置范围；若要放宽普通模型的prompt长度，需要同步物理padding和图布局。仅扩大cache写入容量不足以满足这些条件。

### P2：共享量化 recipe 使 Lincal3.5 的敏感投影同步降至 W4，未接入对应 GPTQ

位置：`examples/qualcomm/oss_scripts/llama/static_llm_quant_recipe.py:883`，关联`__init__.py:650/660`和`wrappers/llm_wrappers.py:2776`。

两个Lincal3.5配置继承Qwen3.5基础配置，直接共享修改后的recipe。相对于Git HEAD，`attention.out_proj`、FFN `w2_conv`、`output.conv`均从W8 per-channel改为W4 group32。实际匹配两版策略验证了权重编码范围由`[-127,127]`变成`[-7,7]`。

GPTQ out_proj优化只在`quantize_qwen35()`中运行，Lincal走`HybridTextDecoder`，不会执行这套补偿。因此“Qwen低比特优化”会改变另一模型的量化精度，却没有对应GPTQ和Lincal PPL验收。没有跑完整Lincal权重，不能据此给出其PPL下降幅度。

建议给Lincal显式保留原recipe，或为其完成独立GPTQ与PPL验证后再启用新的低比特配置。

## 检查结果与作用范围

| 范围 | 当前行为 | 本轮结论 |
|---|---|---|
| Qwen3.5 MHA/SHA | 强制关闭MHA→SHA，走原生GQA优化 | 限定于`qwen3_5`，未覆盖普通模型的开关。 |
| Lincal3.5及其他模型 MHA/SHA | 保留默认MHA→SHA，可用`--disable_mha2sha`关闭 | 混合3层GDN+1层FA/SWA的FP32小图转换后，8路cache全部相同，logits最大误差2.38e-7；未复现这条路径的新数值错误。 |
| SWA环形cache | 继续由SWA专用更新与mask路径处理 | 现有跨AR、回绕、partial update和原生C++ sanitizer测试通过；默认Lincal校准的问题发生在建图阶段，见P1。 |
| GDN OP | `qwen3_5/lincal3_5`注册VTCM包，普通Llama/Gemma/Qwen2.5-VL不注册 | 未看到把无GDN模型切换到DeltaNet的分支。Qwen生成prefill使用chunk图，Lincal仍走通用流程，不应把两者视为同一配置。 |
| FA256/internal mask裁剪 | 只在Qwen35TextDecoder生成图准备中执行 | 不会直接裁剪其他模型的mask；通用runner容量检查的间接影响见P1。 |
| 通用QNN builder/observer | custom tuple输出独立encoding、quant/float Convert、ConcatObserver | 相关回归通过；没有对其他完整模型做端到端精度验收。 |

## 验证证据

- 相关回归（历史产物已清理）：**20项测试、12项子测试通过**，包含MHA→SHA、mask tagging、SWA cache长度、原生C++环形cache/AR切换/mask sanitizer测试、custom op输出encoding、quant/float转换、HTP/LPAI concat observer。
- 小模型复现脚本（历史产物已清理）、日志（历史产物已清理）、结果（历史产物已清理）：上述两个P1及混合GDN/FA/SWA的SHA检查。
- 两版recipe匹配结果（历史产物已清理）：两个Lincal配置、三个敏感投影逐项比较；W4仍用int8容器保存代码，编码范围和block_size证明其量化精度为4bit。

这次review未重新导出或部署任何完整模型，未测其他模型的完整PPL及手机性能。通过的测试不覆盖上面新复现的默认配置和临界容量问题，不能用这些通过结果排除两个P1。

---

<a id="record-40"></a>

## 记录 40：qwen3_5-prefill-optimization / REPORT.md

原始记录位置：`llama-output/qwen3_5-prefill-optimization/REPORT.md`。

# Qwen3.5 prefill 优化记录（2026-10-14）

> 2026-10-14：用户确认保留1684–1707 token/s版本并停止继续优化。最终配置与采用项见[FINAL_SUMMARY.md](README_QWEN35_TEST_RECORDS.md#record-02)：对外prefill AR=128，内部chunk=64。本文是阶段历史，不作为当前版本的唯一说明。

## 版本保存

基线标记：`qwen35-prefill-baseline-20261014`。这是带 SHA256 的文件快照，没有创建 Git commit/tag。原有未提交的工作区内容一并保存。

- 基线清单（历史产物已清理）：0.8B/2B decoder、embedding、量化元数据、DSP/ARM SO、runner、旧性能记录。
- `baseline/source.tar.gz`、`baseline/working-tree.patch`、`baseline/git-status.txt`：当时源码与工作区状态。
- 可用优化版清单（历史产物已清理）、decoder PTE（历史产物已清理）、元数据（历史产物已清理）。

## 已验证的完整模型结果

SM8750，同一设备；0.8B 文本输入；4K cache 接口；AR=128；相同原权重/校准、K/z W8、其余大投影 W4group32、GDN out_proj GPTQ、UINT16 cache。模型加载不计入 prefill。两轮交错顺序测试，温度 0。

| 测试 | 原版 token/s（两轮） | 优化版 token/s（两轮） | 平均吞吐提升 |
|---|---:|---:|---:|
| prefill128 → 1 token | 847.68 / 876.71 | 1207.55 / 1230.77 | 41.4% |
| prefill1024 → 1 token | 891.21 / 890.43 | 1250.31 / 1214.71 | 38.4% |
| prefill1536 → 1 token | 867.31 / 896.67 | 1127.75 / 1110.63 | 26.9% |

**这一阶段四方法版本的完整模型结果约 1200–1250 token/s。** 1536-token 输入越过 1024 桶后回到原 4K 图，平均速度约 1119 token/s。

数值验证：

- 所测 128/1024/1536 prompt、16→64 KV decode、128→64 hybrid 生成文本逐字一致。
- 冷 cache、非零 past=128、past=896 的可用桶与原图，49 个输出逐元素完全相同；后两项为 cache 边界语义测试。
- 8 个独立 128-token 留出窗口、1016 个预测目标，抽样 PPL 前后均为 **50.0248078862**。这是回归检查，不能当作完整标准 WikiText PPL。
- 16→64 端到端 decode：原版 37.46/40.38 token/s，优化版 36.04/37.08 token/s。短样本上略低。固定同一输入的 graph-only 配对检查：原版 21.49–22.19 ms，新 PTE+新 SO 21.76–22.17 ms，49 个输出完全一致；不能据此声称 decode 已提升。

原始证据：生成记录（历史产物已清理）、PPL（历史产物已清理）、decode 控制实验（历史产物已清理）。

## 主要改动与原因

### 1. Prefill 桶裁剪 FA 中被 mask 的 padding

该模型包含 18 层 GDN 和 6 层 full attention。虽然 GDN 是线性递推，FA 分支仍然会对静态 4K cache 长度执行 matmul，短 prompt 的多数 cache 位置为被 mask 的 padding。

新增 `prefill_forward_256` 和 `prefill_forward_1024`，内部 FA 只计算有效范围；仍保留原 4K 外部 cache、相同量化参数和输出格式。runner 按实际 `past + AR` 选择桶，超过范围自动使用原 `prefill_forward`。保留 `kv_forward`。四个方法一起编译且共享权重，保证 QNN context 内的 graph 名称与 PTE 方法名一致。

独立 graph：4K 原版约 140–144 ms；1024 桶约 102 ms；256 桶约 98 ms。编译器估算 DDR spill/fill 从约 977/1038 MB 降至 562/571 MB；这些是编译期估算，不是实测总线计数。

### 2. GDN 精确整数归约减少乘积

DSP `DeltaNetQhpi.cpp` 新增可选宏 `DELTANET_U16_KARATSUBA_REDUCTION=1`，默认关闭。保留 radix16 和原 FP32 合成顺序，把每次投影 4 组整数乘积降为 3 组：

```text
A = sh * kh
C = sl * kl
B = (sh + sl) * (kh + kl)
cross = B - A - C
dot = 256*A + 16*cross + C
```

中间 B 可能超过有符号 INT32，使用 HVX 的模 2^32 累加/减法恢复有界的 cross；最终 A/cross 在 K≤128 下可表示。极值/随机端点验证 588 case、37632 lane，其中 15 例中间发生 wrap，均与 INT64 参考完全一致。整个模型 49 个输出也完全一致。O2 独立收益约 2%；O3 与 256 桶组合约 91.33 ms/128 token（graph-only）。

### 3. Runner 错误处理与独立部署

- 每次 prefill 调用检查执行结果，失败立即报错，避免未执行成功却上报虚假的高吞吐。
- 新版本部署在 `/data/local/tmp/executorch/qwen35_prefill_optimized`，原 baseline 路径保留。
- `deploy.py` 仅修改 PTE compile spec 的 SO 路径，验证所有 backend payload SHA256 不变。
- 新 PTE 487503872 bytes，比基线 470540032 bytes 增加约 16.2 MiB。

## 2000 token/s 的时间预算

128 / 2000 = **64 ms/完整模型 block**，含 24 层和选定行 LM head。单层 linear-attention 的约 3 ms 不能当作整模型时间。当前可用版 graph-only 约 91 ms，端到端约 102–106 ms。

故意跳过 GDN 计算的代价拆分实验，256 桶仍约 57.85 ms；这张图输出错误，只用于估计非 GDN 成本，并非硬件严格下界，更不是可用的 2213 token/s 模型。这说明仅优化 recurrence 的收益空间有限，还需要压缩其他算子/内存调度。

## 试验记录

| 候选 | 范围 / 结果 | 处理 |
|---|---|---|
| radix32 三乘积归约 | 改变 FP32 合成舍入，最低约 30 dB | 拒绝，改为精确 radix16 |
| native 模型 RMSNorm | graph 约 96.7 ms，数值有变化 | 收益小，未采用 |
| DLBC | graph 约 96.8 ms | 收益小，未采用 |
| custom core 内部 FP16 state | graph 约 166 ms | 比原版慢，拒绝 |
| AR512 + 1024 桶 | 重复真实 embedding 的 graph-only 测试，约 1285 token/s | 未到目标，未集成 |
| 首版 32-token Neumann chunk GDN | 第一层单核约 0.98 ms；完整图约 68.16 ms，但后续层矩阵幂溢出 | 精度失败，拒绝 |
| 首版 128-token Neumann chunk GDN | 单核约 1.13 ms，精度失败 | 拒绝 |

chunk GDN 使用原已校准的输入/输出 UINT16 边界，内部 FP16 native QNN matmul；公式通过 nilpotent 三角逆并行计算 token 间校正。修复 `op_to.py` 的 quantized/float 边界选择，改用 `Convert` 执行 affine scale/zero-point 转换。小图实机约 87.8 dB，无 CPU fallback。初次 chunk 实机精度失败定位在大负值 FP16 exp 的非有限结果；将指数限制到 [-16,0] 后单核通过。后续稳定chunk版本已完成完整模型性能和抽样PPL验证，见文末及最终总结。

## 复现入口

```bash
conda activate et_npu
source env.sh
python llama-output/qwen3_5-prefill-optimization/benchmark_generation.py
python llama-output/qwen3_5-prefill-optimization/check_decode.py
python llama-output/qwen3_5-prefill-optimization/check_ppl.py
```

测试脚本需要对应已部署目录。设备空间较少，PPL 的设备输出已在拉回主机后清理。重新跑 PPL 需要预留约 0.5 GB 输出空间。原始主机数据保留。

## 无效记录

`generation_invalid_graph_names/` 中的早期包装试验无效：独立编译的 QNN binary graph 名称没有随 PTE method 改名，执行失败，旧 runner 未检查错误而给出异常高吞吐。已修复 runner，正确四方法重新编译。该目录、`.invalid_so_path.json` 和跳过计算的 ablation 均不能用作最终性能结果。

## 已验证版本的热点 profile

单次详细 profile 的算子周期占比：

| 算子组 | 周期占比 |
|---|---:|
| `qnn_custom_ops_deltanet_core_default` | 56.06% |
| `aten_convolution_default` | 20.00% |
| `aten_mul_tensor` | 6.71% |
| `aten_permute_copy_default` | 3.95% |
| `aten_slice_copy_tensor` | 3.93% |
| `aten_rms_norm_default` | 1.90% |
| `aten_squeeze_copy_dims` | 1.35% |
| `aten__to_copy_default` | 1.25% |

详细 profile 的执行时间约 783 ms，远高于关闭 profiling 时约 91 ms，因此以上比例只定位热点，不能按比例换算正常执行的毫秒数。证据：分组（历史产物已清理）、原始事件（历史产物已清理）。

## 稳定 chunk 求逆版本（已验证、选定）

直接 Neumann 矩阵幂在后续层溢出，约 68 ms 的完整图不满足数值要求，不能作为可用速度。改用分块三角逆后消除大矩阵幂，初始版本因 Gather 结果直接接 Concat 出现错误的分块读取；对 odd gather 加入最小 FP16 subnormal 的加法，强制生成独立结果后设备输出恢复。这个 workaround 的影响已计入实测 SQNR。

随后将最底部 8×8 块在内部求逆，只对大块做合并，减少 Gather/Concat。真实第一层 calibrated 输入、无 CPU fallback：

| chunk | 单核 ms | output SQNR | state SQNR |
|---|---:|---:|---:|
| 32 | 1.26828 | 65.31 dB | 55.02 dB |
| 64 | 1.22858 | 65.21 dB | 52.63 dB |
| 128 | 1.41304 | 64.95 dB | 54.63 dB |

64-token 候选全层参考检查 786 个 matmul/exp 中间值，全部有限。完整 QNN PTE、设备性能、抽样PPL和真实生成检查均已完成。以上单核精度仅代表这份第一层输入，尚未代表完整模型或长序列质量。

## Runner 分段计时与官方比较口径

新增 `Prefill timing` 日志，实测（ms）：

| 配置 / prompt | prepare | execute | cache/mask |
|---|---:|---:|---:|
| baseline_prefill128_0 | 0.060 | 139.033 | 0.319 |
| buckets_prefill128_0 | 0.060 | 96.943 | 0.229 |
| baseline_prefill1024_0 | 0.267 | 1109.059 | 1.843 |
| buckets_prefill1024_0 | 0.442 | 766.363 | 1.607 |

CPU cache/mask 更新在 128 token 时仅约 0.23 ms，主要瓶颈在模型执行，不能把当前差距归因于 CPU cache memcpy。

2026-10-14 查阅 [Qualcomm AI Hub Qwen3.5-2B 官方页面](https://aihub.qualcomm.com/compute/models/qwen3_5_2b)：当前 4096 / 1496 token/s prefill / 18.6 token/s decode 卡片选择的是 Snapdragon X2 Elite CRD，runtime 为 GenieX llama.cpp，精度为 Q4_0 GGUF；本地为 SM8750 + ExecuTorch QNN + W4group32/KzW8，两者设备、runtime 和量化配方不同。2000 仍是本机优化目标，但不能由这条官方记录推导出必须达到的本机阈值。

官方指标还已用 [Qualcomm 源码 perf.yaml](https://github.com/qualcomm/ai-hub-models/blob/main/src/qai_hub_models/models/qwen3_5_2b/perf.yaml) 交叉核对：X2 Elite CRD、4096、NPU 条目为 prefill=1496.439169、decode=18.620861，GenieX v0.8.0。本地保存 下载副本（历史产物已清理）。

## 稳定 chunk 完整 prefill 图（设备检查）

64-token chunk + 8×8 内部求逆，完整 24 层 prefill graph 配对测试：原版 4K 图平均 145.7952 ms，新图平均 **73.0440 ms/128 token = 1752.37 token/s**。这是 graph-only，端到端会包含 embedding、sampling 等开销，不能直接当作端到端吞吐。当前单图裁剪 FA history，适用 past≤128；同一组 8×128-token 窗口、1016 targets 的 PPL=49.8616637907（旧版 50.0248078862），未恶化。证据：完整图性能/49输出比较（历史产物已清理）。

## 接回 KV decode 的短 prompt 实测

诊断 hybrid PTE（历史产物已清理） 将旧版 KV graph 和新 chunk prefill graph 配对；两者的 QNN 内部图名保持原样。当前这份诊断 PTE 只适用于 hybrid prompt≤256 token，runner 会检查历史范围；KV capacity 仍为4096。两份 context 未共享权重，文件为941852424 bytes，尚未作为常规部署产物替换可用的四方法 PTE。

| 测试 | 原版两轮 | chunk 两轮 |
|---|---:|---:|
| prefill128 prefill token/s | 876.71 / 847.68 | 1684.21 / 1706.67 |
| hybrid128_64 prefill token/s | 895.11 / 859.06 | 1580.25 / 1641.03 |

128→1 的端到端 prefill 约1684–1707 token/s，尚未达到2000。128→64 的 decode 约37.28/37.75 token/s，原版40.03/39.55；原KV graph未修改，但输入状态/生成文本及context布局有变化，短测不应直接归因于kernel变慢。生成文本均为连贯英文，chunk版本与基线的措辞不同；因测试设置了ignore_eos，遇到终止符后仍继续到固定输出预算，不能将后面的重新生成当作普通结束策略。首token相同。

证据：生成及统计（历史产物已清理）、PPL（历史产物已清理）、单层core（历史产物已清理）、源码（历史产物已清理）。用户已接受这一性能并要求停止后续优化，对外prefill AR固定为128。

## 继续优化：分块预条件求逆

新 profile 中投影/卷积约占算子周期28.1%，MatMul约14.4%，Gather约7.8%，转置约7.1%，切片约6.0%。这些仍是详细 profile 的周期占比，不能换算普通运行毫秒数。证据：chunk profile（历史产物已清理）。

增加一个候选：先求8×8对角块的逆D，再对块间耦合D·E求逆，得到 `(I-D·E)^(-1)D`；把大矩阵的64阶nilpotent问题压为8个块之间的8阶问题。全程使用mask和MatMul，移除inverse里的Gather/Concat，以及之前odd-gather的微小扰动workaround。18层实际输入的cold/warm reference检查全部有限，逐层output约53–68 dB、state最低约41 dB（cold）；单核设备约0.930 ms、output/state SQNR约65.31/52.62 dB，但没有完成整模/PPL验证，按用户要求不纳入选定版本。源码：chunk_preconditioned.py（历史产物已清理）。

## 停止后续优化与固定版本

用户确认1684–1707 token/s已经够用，停止继续优化。最终采用`chunk_gdn.py`的稳定chunk64 / 8×8块逆版本，对外AR128。后续preconditioned/fused候选和SLC均不采用。SLC设备错误见slc_runtime_failure.log（历史产物已清理）：本机当前SDK/固件不支持该选项，context创建报5005；不是有效性能结果。

完整采用项、继承设置、源码位置、测量口径、短prompt限制及冻结版本见[FINAL_SUMMARY.md](README_QWEN35_TEST_RECORDS.md#record-02)。

---

<a id="record-41"></a>

## 记录 41：baseline / REPORT.md

原始记录位置：`llama-output/qwen3_5-prefill-optimization/baseline/REPORT.md`。

# Qwen3.5-0.8B / 2B 完整模型性能对照

日期：2026-10-14。设备：SM8750，serial `3B1F57E9B8L74X5D`，QNN 2.49.40.260810，Hexagon v79。

## 1. 结果

本地 `/hunter/model/Qwen/Qwen3.5-2B` 已完成 GPU 校准、18 层 GDN out_proj GPTQ、完整 hybrid PTE 编译和设备文本推理。

同一设备、同一物理 context 容量 4096、prefill AR=128 / decode AR=1、相同精确 VTCM GDN 内核，端到端测试结果如下。每个点跑两次，速度用总 token 数除以总耗时计算；模型加载时间不计入。

| 测试 | 0.8B 两次耗时 ms | 0.8B token/s | 2B 两次耗时 ms | 2B token/s |
|---|---:|---:|---:|---:|
| 输入128，输出1，hybrid prefill | 155 / 155 | **825.81** | 146 / 146 | **876.71** |
| 输入1024，输出1，hybrid prefill | 1182 / 1184 | **865.60** | 1183 / 1181 | **866.33** |
| 输入16，输出64，KV decode | 1719 / 1648 | **37.42** | 2590 / 2602 | **24.27** |

decode 的首个输出由 prompt forward 产生，后续 63 次 decode forward 进入表中计时。因此该行是 63 / decode 秒数，完整输出仍为 64 token。

**核心发现：在 1024-token prefill 上，0.8B 和 2B 的速度相差约 0.08%；decode 则是 0.8B 快约 54.2%。** 当前 prefill 很明显没有随模型参数量缩小而获得对应收益。128-token 短测的少量逆向差异，不应推广成 2B 普遍更快。

这次结果证明需要继续优化两者共同使用的 prefill 路径；还没有达到 2000 token/s。若目标是 2000 token/s，完整 128-token 块的端到端预算为 **64 ms**，本轮每块约 146–155 ms，需要约 2.3–2.4 倍整体提速。

原始记录：`generation/summary.json`，包含每次计时、命令、生成文本；`summary.json` 汇总了速度、profile、PTE hash。

## 2. 两个模型哪些地方相同

核对本地模型配置和实际导出图：

| 结构 | 0.8B | 2B |
|---|---:|---:|
| decoder 层数 | 24 | 24 |
| GDN 层 / full-attention 层 | 18 / 6 | 18 / 6 |
| hidden dim | 1024 | 2048 |
| FFN intermediate dim | 3584 | 6144 |
| GDN key/value heads | 16 / 16 | 16 / 16 |
| GDN key/value head dim | 128 / 128 | 128 / 128 |
| 每层 recurrent state | `[1,16,128,128]` | 相同 |
| 每层 conv state | `[1,6144,4]` | 相同 |
| FA query / KV heads | 8 / 2 | 8 / 2 |
| FA head dim | 256 | 256 |
| vocab size | 248320 | 248320 |
| 合并 K/z 投影权重 | `4096×1024` | `4096×2048` |
| 合并 q/v/a/b 投影权重 | `4128×1024` | `4128×2048` |

因此两者都要执行 18 层、每层 128 步、相同 state 大小的 GDN recurrence；6 层 full attention 的 head 形状和 4K KV cache 大小也相同。模型缩小主要减少投影、FFN、LM head 的参数和计算，不能按总参数量比例缩小这些共同成本。

## 3. 无逐算子追踪的完整 decoder 耗时

`qnn_executor_runner`，profile=0，warmup=2，之后 5 次 forward 均值。输入是相同留出文本 token 对应的各模型真实 embedding；cache 内容初始化为空，静态 capacity 仍为 4096。

| 方法 | 0.8B ms | 2B ms |
|---|---:|---:|
| 完整 decode forward，AR=1 | **22.084** | **33.380** |
| 完整 prefill forward，AR=128 | **142.548** | **138.322** |

这些是 decoder 图执行时间。端到端生成还包括 embedding、采样和 runner/cache 管理等开销。profile=0 的冷 cache forward 也不等于长序列持续 decode。

basic profile 的 accelerator execute：

| 方法 | 0.8B ms | 2B ms |
|---|---:|---:|
| decode | 21.620 | 32.308 |
| prefill128 | 140.495 | 134.670 |

basic profile 是另一次执行，只用来核对耗时主要落在 accelerator；不能把两张表相减当成精确 host 成本。两张 decoder 图都检查了只有一个 QNN delegate，没有 CPU 算子回退。

## 4. prefill 热点

以下百分比来自 **profile=2 的逐算子 instrumented cycles**，分母为所有 `OpId` 事件的 cycles 之和。卷积按导出图的模块来源分组：169 个 convolution，其中包含投影、FFN、causal conv 和 LM head。

| 算子类别 | 0.8B M cycles | 占比 | 2B M cycles | 占比 |
|---|---:|---:|---:|---:|
| 18 个 GDN recurrence | **451.19** | **50.07%** | **450.96** | **45.93%** |
| norm / gating / layout / 其他 | 250.44 | 27.79% | 277.56 | 28.27% |
| FFN 三个投影 | 48.36 | 5.37% | 113.37 | 11.55% |
| LM head，仅选择的1个token | 72.65 | 8.06% | 49.90 | 5.08% |
| full-attention MatMul | 44.91 | 4.98% | 43.11 | 4.39% |
| attention 输出 / FA 输入投影 | 16.10 | 1.79% | 22.73 | 2.31% |
| GDN 两个输入投影 | 11.80 | 1.31% | 17.77 | 1.81% |
| GDN causal conv | 5.67 | 0.63% | 6.56 | 0.67% |

这里能直接观察到：

1. GDN recurrence 的 cycles 几乎相同，差约 0.05%；单层平均约 25.1M instrumented cycles。
2. 2B 的 FFN cycles 是 0.8B 的约 2.34 倍，确实体现了模型变宽。
3. full-attention MatMul 的形状相同，cycles 也接近。
4. GDN 输入投影只占约 1–2% 的 instrumented cycles，单纯减少这一组投影参数，无法解释或消除当前整图热点。
5. 同样词表下，2B 的 LM head 在本次追踪中反而更低。不同矩阵形状、QNN 执行选择和设备波动都会影响该值；尚未通过控制实验确定原因，不能仅按权重大小推算耗时。

**限制：profile=2 会显著改变执行时间。** 0.8B / 2B prefill 的逐算子追踪分别约 877 / 960 ms，正常 forward 约 143 / 138 ms。因此不能把上述百分比乘正常耗时，声称 GDN 在正常执行中准确占 50% 或某个毫秒数。这些数据用于定位、比较同一类算子的热点；要测优化收益，必须再用 profile=0 和端到端测试。

`norm/gating/layout/其他` 也需要继续拆分。0.8B 的大项包括 mul 46.91M、view_copy 44.32M、add 30.31M、slice_copy 30.07M、permute_copy 28.39M cycles；它们包含算术、量化转换、布局相关的追踪事件，不能全都归为 DDR 流量。

详细事件保存在 `profiles/{模型}_{方法}/events2.json`；`analyze.py` 负责分组，分组规则和完整操作统计可查。

## 5. decode 热点随参数量增大

同样为 instrumented cycles：

| 算子类别 | 0.8B M cycles | 2B M cycles |
|---|---:|---:|
| FFN 投影 | 29.81 | 80.87 |
| LM head | 32.21 | 49.28 |
| GDN recurrence | 22.67 | 23.94 |
| GDN 输入投影 | 9.61 | 16.14 |

decode 每次只有一个 token，更受大投影的影响；2B 的无 profile forward 增加约 51.2%，与端到端 decode 变慢的方向一致。当前 0.8B 的短 decode 约 37.4 token/s，已经恢复到此前约 35 token/s 的水平。

## 6. 本次导出设置

- 2B 使用 `et_npu` 环境和 CUDA 校准；校准集是与 0.8B 相同的 8 段 Wiki 文本，各取 128 token，共 1024 calibration tokens。
- K/z 合并为 W8；q/v/a/b 合并为 W4；其他 Linear W4 group32；causal conv 的原有 W8 设置保留。
- 对 18 层 GDN out_proj 使用固定 QNN 有效网格的 GPTQ；未把整模型标成 GPTQ。
- activation、recurrent state、conv state、KV cache、logits 接口为 UINT16；hidden 接口为 FLOAT32，positions 为 INT32。
- Q/K 使用 native RMSNorm 路径，attention output 和 state 各自使用校准编码。
- GQA 不复制 KV heads；新 token 写入 FA cache 前转换一次，历史 cache 直接使用已对齐的校准网格。
- prefill 在 LM head 前选择所需 token 行；避免对全部 128 行执行 LM head。
- decoder PTE 和 runner 均开启 shared buffer。
- 两模型使用同一个精确 split-reduction VTCM DSP SO，SHA256：
  `490d117374851b0e5b6fc2d3b69f264b41b703c5bd9c79ad4326b8c35d435d9e`。
- 全程串行导出，未同时跑多个大模型导出任务。

## 7. 文本推理结果

2B 的两次短生成输出一致，开头为：

> **Python** is a high-level, interpreted, general-purpose programming language known for its readability, dynamic typing, and extensive standard library.

输出可读，未出现此前乱码现象。本次是性能对照，没有对 2B 做完整 PPL 评估，因此不能仅凭该样例证明量化精度已全面达标。

测试为空图片列表。为兼容当前 multimodal runner 的加载接口，设备上放置了已有 0.8B vision encoder；**它没有被执行，也未参与 2B embedding 或 decoder**。本次产物用于 2B 文本推理，未验证 2B 图片输入。

## 8. 官方数字的对应口径

官方 Qwen3.5-2B、context4096、NPU、GenieX llama.cpp / Q4_0 数据：

| 官方设备 | prefill token/s | decode token/s |
|---|---:|---:|
| Snapdragon X2 Elite CRD | 1496.44 | 18.62 |
| Samsung Galaxy S25 / Dragonwing Q-8750 | 593.10 | 22.64 |

来源：[Qualcomm 官方 perf.yaml](https://raw.githubusercontent.com/qualcomm/ai-hub-models/main/src/qai_hub_models/models/qwen3_5_2b/perf.yaml)，2026-10-14 核对。

本地 2B 在 SM8750 上约 866 prefill / 24.3 decode，数值超过上述手机条目，但量化方案、后端和实际 prompt/生成计时范围不同，不能据此认定同口径领先官方。1496 的设备条目是 X2 Elite CRD。

本地 0.8B 和 2B 的同机对照，已经直接暴露出 prefill 的共同热点。是否能把 0.8B 做到 2000 token/s，仍需要以实际优化后的整图结果验证。

## 9. 下一步优化顺序

1. 首先针对实际模型量化参数下的精确 GDN recurrence 做优化，保留当前数值精度，用同一真实输入对照 profile=0。两模型共同使用 18 个相同大小的 state，优先处理这里才能让两者一起受益。
2. 拆分 GDN 周围的 norm/gate、切分和布局转换，识别可消除的物化与重复转换；结合实际 buffer 和 VTCM/DDR 计数器判断访存，不能只看 `view_copy` 名称。
3. 分别核对 LM head 的 QNN 执行选择和 decode FFN 热点，避免把 prefill、decode 的瓶颈混在一起。
4. 所有候选优化都在完整 PTE 上复测本报告的三个点；保留生成样例，并做与现有量化基线的数值回归。2000 token/s 以整块 64 ms 为验证目标。

这些是根据本次追踪确定的后续方向；本次没有新修改 DSP 内核，也没有声称已完成这一轮提速。

## 10. 产物和复现

2B 产物：

- `2B/hybrid_llama_qnn.pte`：1,114,504,704 bytes，包含24层、prefill/decode、FFN、final norm、LM head。
- `2B/tok_embedding_qnn.pte`：1,017,195,008 bytes。
- `2B/libraries/`：本次实际部署的两个 op-package SO。
- `2B/metadata.json`：模型配置、48个 cache 编码、hidden 编码、workspace。
- `2B/hybrid_llama_qnn.io.json`：输入输出类型。
- `2B/run_text.sh`：运行已部署设备目录的短文本生成。
- 设备目录 `/data/local/tmp/executorch/qwen35_size_2B`，PTE 的 op-package 路径也指向该目录。迁移设备目录时须同时处理编译时路径。

decoder SHA256：`0d4270212d609b1172d95d9970708c35f60980d5eb919366ba77ab73df37b0bc`。

embedding SHA256：`a3513ce219e3d582330ade80307b9020a566c52458d4f81242a24470030a63d3`。

环境初始化后重建、部署和对照：

```bash
conda activate et_npu
source env.sh
export CUDA_VISIBLE_DEVICES=0
python llama-output/qwen3_5-size-comparison/export_model.py --size 2B --capacity 4096
python llama-output/qwen3_5-size-comparison/benchmark.py --phase deploy
python llama-output/qwen3_5-size-comparison/benchmark.py --phase generation
python llama-output/qwen3_5-size-comparison/benchmark.py --phase profile --size all
python llama-output/qwen3_5-size-comparison/analyze.py
```

generation 会跳过已有完整记录；要新跑一轮，应先重命名 `generation/summary.json`，保留原始证据。

当前脚本复用 `llama-output/qwen3_5-0_8b-layer-analysis` 下既有导出、校准、GPTQ 和编译 helper；需要保留这些依赖。本次没有修改 `mybuild.sh`，普通旧 `llama.py` 命令也不会自动得到这里的后校准 GQA/所选行优化。

---

<a id="record-42"></a>

## 记录 42：context4096_ar128 / REPORT.md

原始记录位置：`llama-output/qwen3_5-prefill-optimization/context4096_ar128/REPORT.md`。

# Qwen3.5-0.8B：4096 token 性能实测

设备 SM8750 / HTP v79，串行运行；文本推理，不运行图像编码器。

保持已选定的 prefill AR=128、GDN chunk=64 / inverse leaf=8 / 内部 FP16；KV decode 使用原精确整数 VTCM 核。K/z W8，其余大 Linear W4，原 GPTQ 权重及校准参数不变。

物理容量 4224，以 128 对齐并为第一个生成 token 留位置；4096-token prompt 分 32 个 AR128 block 执行。Full-attention 保留 4096 的计算长度，支持最后一个 block 的 past=3968。

时间来自 runner 阶段时间戳，包含 embedding、模型执行、采样和 cache 更新；不含模型加载或 ADB。强制忽略 EOS 以跑满输出数，这不是生成质量评估。

| 测试 | 输入 token | 输出 token | 阶段平均耗时 | 速度 | 次数 |
|---|---:|---:|---:|---:|---:|
| Prefill | 4096 | 1 | 3899.00 ms | 1050.53 token/s | 2 |
| Decode | 16 | 4080 | 123.397 s | 33.06 token/s | 2 |

Decode 的首 token 在 prompt 阶段产生，decode 时间覆盖后续 4079 次 forward；有效上下文从 16 增长到 4095，速度是这整个过程的平均值。

## 每轮结果

| 轮次 | 测试 | Prompt 阶段 ms | Decode 阶段 s | Prefill token/s | Decode token/s | 电池温度 °C |
|---:|---|---:|---:|---:|---:|---|
| 1 | prefill4096_1 | 3890.00 | 0.013 | 1052.96 | — | 未记录 → 29.1 |
| 1 | decode16_4080 | 403.00 | 122.893 | 39.70 | 33.19 | 29.1 → 29.6 |
| 2 | prefill4096_1 | 3908.00 | 0.013 | 1048.11 | — | 29.6 → 29.6 |
| 2 | decode16_4080 | 430.00 | 123.901 | 37.21 | 32.92 | 29.6 → 31.0 |

## 与128-token短prompt结果的区别

先前1684–1707 token/s来自内部FA长度256的短prompt专用图。本次是FA长度4096的长prompt图，6个full-attention层每个block按静态4096长度计算；18个GDN层仍采用相同chunk64算法。两项结果的FA工作量不同。

| Prefill轮次 | 32个block prepare合计 ms | execute合计 ms | cache合计 ms | 单block execute平均 ms |
|---:|---:|---:|---:|---:|
| 1 | 3.760 | 3778.778 | 17.621 | 118.087 |
| 2 | 3.726 | 3811.101 | 18.602 | 119.097 |

prepare/execute/cache是prompt processor的累计时间，端到端prompt计时还包括文本处理、embedding等。这些计时没有进一步分离FA与GDN，不能仅据此量化各自的耗时占比。

## 原始证据

- 编译配置：metadata.json（历史产物已清理）
- PTE / runner / 算子库校验：deployment.json（历史产物已清理）
- 原始计时和阶段统计：records.json（历史产物已清理）
- 逐轮日志：`prefill4096_1_*.log`、`decode16_4080_*.log`。

---

<a id="record-43"></a>

## 记录 43：decode_diagnosis / PREFILL_REPORT.md

原始记录位置：`llama-output/qwen3_5-prefill-optimization/decode_diagnosis/PREFILL_REPORT.md`。

# 采样修复后的prefill复测

2026-10-14，SM8750 / HTP v79，Qwen3.5-0.8B，纯文本，temperature=0，输出1个token。对外AR始终为128；使用相同PTE分别配对运行旧runner与直接量化argmax的runner，交替测试顺序，模型加载不计入时间。

## 结果

128输入使用冻结的短prompt PTE（FA内部长度256）；4096输入使用长上下文PTE（物理容量4224，FA内部长度4096）。两个PTE的GDN均保持chunk64 / leaf8 / FP16。短prompt初次两轮有较大执行波动，因此额外补了四轮交替对照，以下保留全部样本。

| 输入→输出 | Runner | 次数 | 平均prompt耗时 | 平均吞吐 | 中位耗时 | 每轮耗时ms |
|---|---|---:|---:|---:|---:|---|
| 128→1 | 原版 | 6 | 82.33 ms | 1554.66 token/s | 81 ms | 82, 80, 84, 80, 91, 77 |
| 128→1 | 直接量化argmax | 6 | **80.83 ms** | **1583.51 token/s** | **79 ms** | 80, 92, 79, 78, 79, 77 |
| 4096→1 | 原版 | 2 | 4001.00 ms | 1023.74 token/s | 4001 ms | 3943, 4059 |
| 4096→1 | 直接量化argmax | 2 | **4092.50 ms** | **1000.86 token/s** | 4092.5 ms | 4135, 4050 |

这里的平均吞吐使用输入token数除以平均耗时。新runner的128输入中位吞吐为1620.25 token/s。所有新旧输出文本逐字节一致。

## 如何理解

- 本次修复主要改善长decode。4096→1只在prompt末尾采样一次，不能将每次约4 ms的节省乘以32个prefill block。
- 4K原版模型执行平均3907.69 ms，新runner平均4002.66 ms；这部分图与权重未改，但本轮执行时间增加约95 ms。端到端均值增加91.5 ms，而执行之外的开销减少约3.47 ms。因此，本轮均值不能作为采样修复导致模型图变慢的证据，也没有测到prefill总体提速。
- 新runner的4K模型执行占prompt总时间约97.8%，cache更新约18 ms。剩余优化需要进一步分离模型图中的FA、GDN和投影耗时，采样已经不是prefill的主要开销。
- 原冻结记录1684–1707 token/s以及较早4K测试1050.53 token/s属于早先测量。本次短prompt有77–92 ms的波动；报告保留全部数据，没有用最好一轮替代均值。

作为同代平台的参考，官方Galaxy S25 / Snapdragon 8 Elite For Galaxy的Q4_0、GenieX llama.cpp记录：4K NPU prefill为**889.51 token/s**；512容量NPU为979.81 token/s。手机、运行时、权重精度及测试口径不同，不能直接据此声称严格配对的加速比例。[官方性能表](https://github.com/qualcomm/ai-hub-models/blob/main/src/qai_hub_models/models/qwen3_5_0_8b/perf.yaml)

## 证据

- 首次配对结果（历史产物已清理）、原始阶段数据（历史产物已清理）
- 短prompt追加配对（历史产物已清理）、追加阶段数据（历史产物已清理）
- 全部样本汇总（历史产物已清理）
- 测试脚本（历史产物已清理），原始日志、命令和文本均在本目录。

本次只复测，未修改AR、模型、校准、GDN或runner实现。

---

<a id="record-44"></a>

## 记录 44：decode_diagnosis / REPORT.md

原始记录位置：`llama-output/qwen3_5-prefill-optimization/decode_diagnosis/REPORT.md`。

# Qwen3.5-0.8B decode：定位并修复CPU采样开销

2026-10-14，设备SM8750 / HTP v79，纯文本。沿用此前同一个长上下文PTE，输入16 token、输出4080 token、忽略EOS；首token在prompt阶段产生，decode阶段计时覆盖后续4079次forward。物理容量4224，prefill AR保持128。

## 1. 官方数据核对

官方Galaxy S25 / Snapdragon 8 Elite For Galaxy、Q4_0、GenieX llama.cpp的记录如下，读取日期2026-10-14：

| 后端 | Context capacity | Decode token/s |
|---|---:|---:|
| NPU | 4096 | 37.724579 |
| NPU | 512 | 42.218670 |
| CPU | 4096 | 52.200623 |
| CPU | 512 | 83.816643 |

来源：[Qualcomm官方perf.yaml](https://github.com/qualcomm/ai-hub-models/blob/main/src/qai_hub_models/models/qwen3_5_0_8b/perf.yaml)，本地快照official-perf.yaml（历史产物已清理）。本机与S25属于同代平台；手机、量化格式、运行时和生成长度口径不同，不能将这些记录当作严格配对测试。

## 2. 已定位的开销

旧runner在temperature=0时，仍然把248320个UINT16 logits逐个转换为float，再由通用sampler扫描词表找argmax。原两轮4K输出测试中，采样计时平均16.927秒，占123.397秒decode时间的约13.7%。

为验证开销位置，给`TokenGenerator::generate`加入`steady_clock`分阶段累计计时。使用同一个容量4224的PTE，输入16、输出496的短程配对结果为：

| 阶段 | 旧runner ms/token | 直接argmax ms/token |
|---|---:|---:|
| Embedding与输入准备 | 0.235 | 0.218 |
| Decoder step | 25.910 | 24.908 |
| **采样** | **4.116** | **0.173** |
| Cache / mask更新 | 0.158 | 0.168 |
| Tokenizer decode / callback | 0.013 | 0.013 |
| 端到端decode token/s | 32.86 | 39.24 |

`Decoder step`包括delegate/runtime开销，不是独立的HTP硬件计时。短程两轮的decoder执行时间也有波动，因此不能将全部吞吐差异归因于采样。

## 3. 修复

- temperature=0且repetition penalty关闭时，直接在量化logits上求argmax，跳过float数组转换。
- 对UINT16使用ARM NEON向量归约取最大值，再按位置顺序找到第一个匹配值，保留旧sampler的并列值规则。
- 正的per-tensor scale保持大小顺序；原旧greedy路径也直接比较量化codes对应的float整数值，此修改不改变排序。
- 随机采样、repetition penalty、Float logits继续走原路径。

代码：[quantized_logits.h](runner/quantized_logits.h)、[decoder_runner.h](runner/decoder_runner.h)、[token_generator.cpp](runner/token_generator.cpp)。量化权重、校准参数、PTE、GDN及外部cache格式没有改变。

## 4. 完整16→4080结果

| Runner | 第1轮 token/s | 第2轮 token/s | 平均decode耗时 | 平均token/s |
|---|---:|---:|---:|---:|
| 原runner | 33.1915 | 32.9214 | 123.397秒 | 33.0559 |
| **直接UINT16 argmax** | **37.3395** | **37.3029** | **109.2945秒** | **37.3212** |

吞吐提高**12.90%**，decode耗时减少约14.10秒。此处只比较相同PTE、相同prompt与输出预算的本地结果。

新runner长程平均每token：decoder step **26.205 ms**，采样**0.173 ms**，embedding/输入准备0.232 ms，cache/mask更新0.170 ms，tokenizer/callback 0.013 ms。剩余主要耗时在decoder step；当前计时没有进一步分离GDN、投影、FA或HTP/runtime开销。

## 5. 一致性验证与保存

- 主机标量参考测试和设备NEON测试均通过：覆盖UINT16/UINT8、非16字节对齐指针、向量边界、尾部、重复最大值、全零、全最大值及完整248320词表。
- 短程旧/新生成文本逐字节一致；两轮4080-token生成文本均与原版长程输出逐字节一致。这里保存的是文本比对，没有额外记录逐位置token ID。
- PTE SHA256仍为`e619d39b951151e05d48d85f7e695565009181a449663aab34671943378f1320`。
- 新runner已部署到`/data/local/tmp/executorch/qwen35_context4096_ar128/qnn_multimodal_runner_u16_argmax`，原runner和冻结快照保留。
- 新runner SHA256为`582ad62caa8b6b6022cc9d7314a1fd07fcf1887e7132fe7dda633b7ddfab906a`。

证据：manifest.json（历史产物已清理）、短程旧计时（历史产物已清理）、短程新计时（历史产物已清理）、完整4K新计时（历史产物已清理）。测试源码：quantized_logits_test.cpp（历史测试代码已清理）。原始日志和生成文本保存在本目录。

重新运行已部署的新版runner：

```bash
python llama-output/qwen3_5-prefill-optimization/decode_diagnosis/benchmark_decode.py \
  --runner qnn_multimodal_runner_u16_argmax --label rerun4096 --repeats 2
```

本轮decode验证没有重新导出模型，没有修改prefill AR或GDN算法。后续按用户要求补测prefill，结果见[PREFILL_REPORT.md](README_QWEN35_TEST_RECORDS.md#record-43)：采样修复没有带来明显prefill整体提速，4K输入仍约1000–1040 token/s。

---

<a id="record-45"></a>

## 记录 45：image_benchmark / REPORT.md

原始记录位置：`llama-output/qwen3_5-prefill-optimization/image_benchmark/REPORT.md`。

# Qwen3.5-0.8B：图片输入的性能影响

2026-10-14，SM8750 / HTP v79。之前的性能测试全部为纯文本，`empty_images.txt`为空，未执行视觉编码器。

本次所有对照使用同一个4K长上下文PTE、修复采样后的runner，prefill AR=128、GDN chunk64、KV精确整数VTCM核。每项两轮，输出64 token，忽略EOS；decode阶段统计首token之后的63次forward。没有重新导出或校准模型。

## 1. 图片与计时口径

- 图片：本地COCO两只猫照片（历史产物已清理），原始640×480。
- 为匹配已有静态视觉编码器，bicubic缩放为448×448，用本地checkpoint的image processor做标准化和patch排列。
- 编码器输入Float32 `[1,784,1536]`，输出Float32 `[1,196,1024]`，即**196个图像token**。
- 短文字是`Describe this image in one sentence.`。无图19 token，加图后217 token：增加196个图像token和2个视觉边界token。
- 等长对照将纯文本和图文合并后的decoder输入都固定为4096 token。有图版本减少对应文字长度，并非4096文字再加图片。
- 首token时间包含视觉编码、文字embedding、decoder prefill和采样；**不含模型加载、JPEG预处理与raw文件读取**。预处理在主机离线完成，一次记录15.25 ms，不能当作设备预处理性能。
- 短文字也使用FA计算长度4096的同一个长图；不能与此前FA长度256的短prompt专用PTE混用吞吐结果。

## 2. 两轮平均结果

| 对照 | 合并输入token | AR128调用次数 | 视觉编码ms | Decoder prefill execute ms | 首tokenms | Decode token/s |
|---|---:|---:|---:|---:|---:|---:|
| 同一句短文字，无图 | 19 | 1 | 0 | 122.897 | **125.0** | **37.29** |
| 同一句短文字，加1张图 | 217 | 2 | 25.845 | 247.429 | **279.0** | **37.23** |
| 固定4096总token，纯文本 | 4096 | 32 | 0 | 3947.473 | **4035.0** | **37.04** |
| 固定4096总token，包含1张图 | 4096 | 32 | 25.988 | 3930.186 | **4041.5** | **37.08** |

同一句短文字加图后，首token增加**154 ms**，达到原来的2.23倍。主要增加一次约26 ms的视觉编码和一个AR128 block的decoder计算。

总输入固定4096时，图片本身仍需要约26 ms的视觉编码；端到端均值仅增加6.5 ms，受其余阶段的波动和文字长度变化影响，不能由此把视觉编码耗时估成6.5 ms。两组首token逐轮分别是4013/4057 ms与4028/4055 ms。

图像只在prompt阶段编码。本样本、63步decode的配对结果没有显示明显decode速度变化，两种输入均约37 token/s。这不是重新跑了一遍4080-token长decode测试。

视觉编码时间从runner的`Encoding tensor`到`Encoder execution completed`日志的微秒时间戳相减，包含`module.forward`和少量包装开销；不是硬件内部单算子计时。两轮视觉编码范围为25.68–26.21 ms。

## 3. 实际推理质量存在问题

图片的196个embedding确实经过编码、合并并进入decoder；实测输入数量和AR128调用次数均已检查。但本图片没有通过描述正确性检查：

- 短图文hybrid输出开头为`**<30.42.0.8.0.1...`，没有正确描述两只猫。
- 4K背景文字加图时，输出描述了所提供的演员/戏剧背景，没有描述图片。
- 补跑相同图片和短文字的**KV模式**，输入217 token，prompt耗时5765 ms、decode约37.08 token/s；输出开头为`Juan 200 million dollars...`，同样错误。

因此，这份完整量化PTE的图文精度目前存在问题。仅凭这些对照还不能判断错误来自视觉编码器、图文量化范围、cache或其他环节；KV对照表明错误不只发生在chunk64 prefill路径。本报告的耗时数据不能用作图文精度通过的证明。强制忽略EOS用于固定性能预算，生成全文中的后续重复内容也包含这个因素。

## 4. 保存与复现

- 图片及预处理元数据（历史产物已清理）、确切prompt和token数量（历史产物已清理）
- 两轮原始统计（历史产物已清理）、汇总（历史产物已清理）、KV对照（历史产物已清理）
- PTE / runner / 图片校验与结论（历史产物已清理）
- 预处理脚本（历史产物已清理）、设备测试脚本（历史产物已清理）
- 各项原始日志、命令和生成文本保存在本目录。

设备数据保存在`/data/local/tmp/executorch/qwen35_context4096_ar128/image_outputs/`，图片输入为同目录`cats448.raw`和`image_inputs.txt`。原纯文本模型、冻结快照和测试结果保留。

## 5. 后续视觉encoder精度检查

同一张图片、相同预处理，对比encoder和merger之后的196×1024图像embedding：当前静态PyTorch FP32相对Transformers FP32为98.81 dB，而设备正在使用的视觉PTE只有**−3.46 dB**、cosine **0.541126**。

另外，当前decoder输入量化只能表示[-0.13769,0.15332]，设备图像embedding有**55.90%**越界；即使使用HF正确的视觉输出，也有**24.09%**越界，输入QDQ后SQNR只剩**3.78 dB**。视觉PTE输出误差和decoder输入范围过窄两处均已测到。具体视觉出错算子仍待CPU QDQ与设备逐层定位，完整口径和证据见[encoder_accuracy/REPORT.md](README_QWEN35_TEST_RECORDS.md#record-46)。

## 6. 后续视觉encoder修复

视觉塔改为FP16，并显式保留GELU(tanh)的数学语义，以等价sigmoid公式降低开销。相同猫图输入，设备SQNR从−3.46提高到**34.68 dB**，cosine从0.541提高到**0.999830**；三张图均在33–35 dB。最终配对encoder耗时从25.62增至41.30 ms，PTE从107 MB增至206 MB。新视觉PTE已部署，旧版保留。

同一个HF FP32语言模型接入新特征后能描述两只猫和遥控器，但完整量化PTE仍产生错误内容：现有decoder输入范围仍截断新特征的24.08%。本轮完成视觉encoder修复，后续需要处理图文输入校准及量化语言路径。全部候选、代码、验证与部署见[视觉修复报告](README_QWEN35_TEST_RECORDS.md#record-47)。

---

<a id="record-46"></a>

## 记录 46：encoder_accuracy / REPORT.md

原始记录位置：`llama-output/qwen3_5-prefill-optimization/image_benchmark/encoder_accuracy/REPORT.md`。

# 视觉 encoder 输出精度与 decoder 输入量化检查

本文记录修复前的旧PTE诊断。后续视觉encoder已重建并切换到FP16修复版，原数值与产物保留；新旧配对结果及仍未解决的decoder输入问题见[修复报告](README_QWEN35_TEST_RECORDS.md#record-47)。

2026-10-14，Qwen3.5-0.8B，SM8750 / HTP v79。

## 1. 视觉 encoder 输出已经有明显误差

同一张两只猫照片、同一份预处理后的像素输入，对比 **视觉塔与 merger 完成后、送进语言模型之前** 的图像 embedding。输出形状是 `[1,196,1024]`，共196个图像token。

参考取 Transformers `Qwen3_5VisionModel.pooler_output`，不是 merger 之前 `[784,768]` 的 `last_hidden_state`。参考模型使用本地同一 checkpoint、FP32、eager attention、CUDA，关闭TF32。

| 输出相对HF FP32 | SQNR | Cosine | RMSE |
|---|---:|---:|---:|
| 当前静态 PyTorch FP32 encoder | **98.81 dB** | **0.999999999934** | 0.000001902 |
| 当前设备实际使用的视觉 encoder PTE | **−3.46 dB** | **0.541126** | 0.246954 |

**当前静态浮点实现能对齐HF；部署的视觉PTE没有对齐。** 大误差在进入语言模型之前就存在。现有证据尚不能区分是视觉PTQ本身、后端降低/编译，还是旧导出产物内部的问题，需要进一步比较CPU QDQ和设备输出。

SQNR使用 `10*log10(sum(reference²)/sum((actual-reference)²))`，以HF为参考，对全部196×1024元素合并能量后计算；不是逐token dB的平均。

### 输入和产物核对

- 图片：`llama-output/qwen3_5-0_8b-m512-kv/calibration-images/000000039769.jpg`，640×480，统一bicubic缩放为448×448。
- 使用本地HF image processor、`do_resize=False`，输入FP32 `[1,784,1536]`，grid=`[1,28,28]`。
- 主机和设备实际输入raw SHA256均为 `285dd25333ac795ff235352337cb2bd287a37873ef70f0f4756ebb17613969af`。
- 仓库校准预处理 `_qwen_vl_image_to_patches` 与这份HF预处理输入对比：最大差异 **5.91×10⁻⁸**，RMSE **2.38×10⁻⁸**。没有看到足以解释此次误差的预处理差异。
- 直接运行设备当前 `vision_encoder_qnn.pte`，其输出是FP32，全部有限值。本次单次执行26.025 ms，只用作确认实际模型正常完成执行，不作为新增性能统计。
- 设备PTE SHA256：`2fa0f90e6ed5013e63512304db683c4144fdd73165352c57695342456c1f3930`。已从设备保存为本目录 `vision_encoder_current.pte`。
- 该校验值与 `llama-output/qwen3_5-0_8b-diagnose/kv-cache-fixed/vision_encoder_qnn.pte` 一致。其导出记录指向本地Qwen3.5-0.8B checkpoint；对应日志显示视觉Linear采用 **W8A16、MinMax、per-tensor**。不是拿另一份本地PTE替代当前部署版本。
- 原视觉校准配置包含两张图，其中包括本次猫照片。配置相同不等于PTQ精度通过；旧导出设置 `skip_qdq_save=True`，没有保留可直接复用的视觉CPU QDQ图。

## 2. decoder 输入量化范围也不适合图像 embedding

从当前长上下文模型的 `decode_qdq.pt2` 和 `prefill_qdq.pt2` 读取首个输入量化节点，两种方法一致：

```text
scale      = 4.440613793121884e-06
zero_point = 31008
codes      = [0, 65535]
实数范围    = [-0.1376945525, 0.1533210724]
```

| 图像embedding来源 | FP32 min / max | 低于下界 | 高于上界 | 超出范围总比例 |
|---|---|---:|---:|---:|
| HF FP32 encoder | −4.43045 / 1.64649 | — | — | **24.09%** |
| 当前设备 encoder | −4.64042 / 1.64648 | 30.22% | 25.68% | **55.90%** |

这里能够直接观察量化前的值，因此比例是实际越界值比例，不只是统计输出端点码。

### 单独隔离输入量化的影响

| 路径，相对原始HF图像embedding | SQNR | Cosine |
|---|---:|---:|
| **HF正确encoder输出 → 当前decoder输入QDQ** | **3.78 dB** | **0.793927** |
| 设备encoder输出 → 当前decoder输入QDQ | **0.0024 dB** | **0.366513** |

第一行绕过了错误的视觉PTE，仍然发生大幅损失：**即使先修好视觉encoder，当前decoder输入网格也会截断正确的图像特征。**

第二行SQNR比直接设备encoder的−3.46 dB更高，是截断同时压低了部分错误幅度；cosine继续下降，不能把这个数值变化解释为精度恢复。

输入QDQ结果是在主机按当前图中实际scale/zero point执行PyTorch量化、反量化得到，**不是设备中间张量dump**。视觉encoder输出本身则来自真实设备。Runner将image embedding加入合并buffer后，直接按FP32复制进decoder输入，没有在这段路径重新缩放图像特征。

相关代码：

- `examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/multimodal_runner.cpp:1012`：encoder输出加入embedding merger。
- `examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/multimodal_prompt_processor.cpp:243`：合并后的embedding按FP32复制进decoder输入。

## 3. 下一步定位顺序

1. **先只处理视觉encoder。** 单独导出视觉图并保留CPU QDQ，比较patch embedding、12个视觉block、merger的FP32/QDQ/设备结果，找到首个明显偏离位置；无需同时重新导出整个语言模型。
2. CPU QDQ若已经有大误差，检查对应节点的校准范围、饱和、权重量化；CPU QDQ若准确而设备偏离，再检查QNN对应算子和降低方式。当前末端输出对照还不能确定具体出错算子。
3. 视觉输出对齐后，**使用真实图文输入重新校准decoder输入和后续激活**，覆盖图像embedding范围，并保留纯文本PPL对照。不能只恢复视觉encoder而继续沿用现在这份过窄的输入网格。

本轮完成比较和诊断，没有替换生产模型或修改量化配方。以上是单张图片的受控对照，不能代替多图片精度评估。

## 4. 保存的证据与复现

- 汇总数值（历史产物已清理）、196个图像token的误差（历史产物已清理）。
- HF参考：`hf_fp32.pt`，包含merger之后的pooler output及merger之前的hidden；静态FP32：`static_fp32.pt`。
- 设备实际输出：`outputs/output_0_0.raw`；设备日志：`device.log`。
- 输入QDQ后的主机分析结果：`after_decoder_input_qdq.raw`。
- 比较脚本（历史产物已清理）：加载同一checkpoint和保存的实际设备输出，计算SQNR、cosine及输入截断比例。

激活 `et_npu`、`source env.sh` 后，在仓库根目录计算：

```bash
python llama-output/qwen3_5-prefill-optimization/image_benchmark/encoder_accuracy/compare.py
```

重新抓取当前设备encoder输出的命令：

```bash
adb -s 3B1F57E9B8L74X5D shell 'cd /data/local/tmp/executorch/qwen35_context4096_ar128 && mkdir -p encoder_accuracy/outputs && env LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./embedding_check/embedding_runner --model_path vision_encoder_qnn.pte --method_index 0 --input_list_path image_inputs.txt --output_folder_path encoder_accuracy/outputs --warm_up 0 --iteration 1 --profile_level 0'
adb -s 3B1F57E9B8L74X5D pull /data/local/tmp/executorch/qwen35_context4096_ar128/encoder_accuracy/outputs llama-output/qwen3_5-prefill-optimization/image_benchmark/encoder_accuracy
```

此处 `embedding_check/embedding_runner` 是前一轮保存的通用 `qnn_executor_runner` 剥离调试符号副本，支持单输入视觉模型；没有使用teacher-forcing专用runner。

---

<a id="record-47"></a>

## 记录 47：repair / REPORT.md

原始记录位置：`llama-output/qwen3_5-prefill-optimization/image_benchmark/encoder_accuracy/repair/REPORT.md`。

# Qwen3.5-0.8B：视觉 encoder 修复记录

2026-10-14，SM8750 / HTP v79。只重建视觉模型，所有导出串行执行；没有重导语言模型。

## 1. 已修复和仍未修复的范围

**视觉 encoder 已重建、验证并部署。** 相同猫照片、相同输入raw、相同HF FP32参考，encoder和merger之后的196×1024图像embedding：

| 项目 | 旧视觉PTE | 选定修复版 |
|---|---:|---:|
| SQNR | **−3.45698 dB** | **34.68497 dB** |
| Cosine | 0.541126 | **0.999830** |
| RMSE | 0.246954 | **0.003059** |
| 最大绝对误差 | 4.551888 | **0.074232** |
| Encoder执行时间 | **25.6223 ms** | **41.2954 ms** |
| PTE文件大小 | 106,965,944 bytes | 205,551,296 bytes |

该表来自最后一轮配对测试：先warm-up 3次，再执行10次，两个版本使用SHA256完全相同的输入。输入SHA256为 `285dd25333ac795ff235352337cb2bd287a37873ef70f0f4756ebb17613969af`。不是从不同尺寸或不同预处理的图片中挑选结果。

**精度改善有成本**：视觉模型改为FP16，PTE从约107 MB增至206 MB；本轮配对的encoder耗时增加15.67 ms，约61%。这不是语言模型prefill的性能变化；语言PTE、权重量化和AR128没有改动。

**完整量化PTE的带图生成仍然不正确。** 新视觉特征进入现有decoder时还有24.08%的值超出输入量化范围。视觉encoder的数值修复已完成，图文输入校准和量化语言路径的修复仍待继续，不能将本报告当作完整多模态精度通过。

## 2. 本轮修改

### 视觉塔采用FP16

原W8A16路径在CPU QDQ中已经有明显误差。直接使用per-channel又暴露了patch投影bias的int32容量问题；修复该问题后，设备精度仍弱于浮点视觉图。因此选定版本保留FP32外部接口，内部权重和激活使用FP16。

- `VisionModalityConfig`增加 `use_fp16=False`；`Qwen35VLEncoder`设置为True。
- `llama.py::compile`读取这一配置，为视觉组件跳过PTQ并生成FP16 HTP compiler spec。
- 语言模型的 `args.use_fp16` 没有改变；K/z W8、其余大Linear W4、GPTQ及GDN路径继续使用原生产PTE。
- 旧W8视觉recipe仍保留用于对照，没有删除或冒充当前选定精度。

### 保留checkpoint的GELU(tanh)语义

checkpoint使用 `gelu_pytorch_tanh`。现有 `op_gelu.py` 不读取 `aten.gelu` 的approximate参数，直接创建QNN `Gelu`；本机SDK的MasterOpDef定义该算子为erf公式，且没有approximate参数。

当前视觉block改用显式 `StaticGELUTanh`，避免将tanh版本交给这个native Gelu：

```text
u = clamp(x, -10, 10)
gelu_tanh(x) = x * sigmoid(u * (1.5957691216057308 + 0.07135481642948435*u*u))
```

来自 `1+tanh(v)=2*sigmoid(2v)` 的等价形式，避免FP16中 `1+tanh(v)` 接近零时的消减误差，并减少中间运算。只限制内部多项式的输入来避免FP16三次方溢出，外层仍保留原x。FP32源图与HF的三张图对照分别为89.79、104.61、97.75 dB。

MLP的权重名称和加载方式保持兼容。这是普通PyTorch图，整个视觉图降低为一个QNN delegate，没有新增DSP自定义op，也没有CPU算子回退。

代码：

- `examples/qualcomm/oss_scripts/llama/model/vision_encoder.py`：`StaticGELUTanh`、`StaticQwen35VLVisionBlock`。
- `examples/qualcomm/oss_scripts/llama/encoder/encoder_config.py`：视觉精度配置。
- `examples/qualcomm/oss_scripts/llama/llama.py`：组件级FP16编译选择。
- `backends/qualcomm/tests/test_vision_gelu_tanh.py`：数学一致性、FP16尾部有限值、导出图避免native Gelu的回归测试。

## 3. 对照过程与未采用项

| 方案 | 猫图CPU QDQ SQNR | 猫图设备SQNR | 设备耗时说明 |
|---|---:|---:|---|
| 旧部署W8A16 | 未保存旧QDQ | −3.46 dB | 配对25.62 ms |
| 重建W8A16 per-tensor | 12.95 dB | 12.36 dB | 原生量化图 |
| 直接W8 per-channel | 7.27 dB | 未导出 | patch bias截断，停止该候选 |
| W8 per-channel + bias范围保护 | 23.27 dB | 19.40 dB | 三图平均26.21 ms |
| FP16 + native Gelu | 不做PTQ | 23.10 dB | 三图平均28.21 ms |
| FP16 + 显式tanh公式 | 不做PTQ | 34.53 dB | 三图平均95.37 ms，未采用 |
| **FP16 + 等价sigmoid公式** | 不做PTQ | **34.68 dB** | **三图平均40.02 ms，选定** |

三图均使用3次warm-up、10次计时；表中早期重建per-tensor仅做诊断，不将其单次时间混为同口径性能结果。最终配对单猫图为41.30 ms，与三图均值存在正常测量差异，原始值全部保留。

### bias问题的具体证据

per-channel权重本身的patch权重SQNR从36.08改善到41.15 dB，但派生bias scale为 `activation_scale * weight_scale[channel]`，768个输出通道中有3个通道的int32范围不足。

patch bias原值范围为[-2.078125,2.796875]，未经保护的QDQ范围缩成[-1.6484375,2.109375]；bias SQNR只有16.25 dB。候选中按照实际校准activation scale和真实bias，扩大必要通道的weight observer范围，使派生bias能落在int32内；没有写死激活scale或zero point。该保护提高了W8候选精度，但最终选择了更稳定的FP16视觉图。最终FP16不需要这项bias量化保护。

### 对旧PTE的归因限制

旧视觉导出设置了 `skip_qdq_save=True`，所以不能逐节点恢复旧产物的CPU QDQ。当前重建per-tensor图已不同于旧PTE的−3.46 dB结果；本轮没有把这一全部差距强行归因于某个旧算子。另做了legacy Cast选择对照，24处转换都同时具有量化属性，结果没有变化，未将之前的单侧Cast/Convert问题认定为本例原因。

最终修复依据是可复现的FP32源图、实际设备结果、数学语义检查和跨图片验证；旧PTE及各候选保留，便于继续追溯。

## 4. 三张图片与功能验证

| 图片 | 选定版设备SQNR | Cosine | 执行时间 |
|---|---:|---:|---:|
| 两只猫 | 34.68376 dB | 0.999830 | 39.7188 ms |
| 自由女神像 | 33.03200 dB | 0.999751 | 40.2175 ms |
| 狗图 | 33.78964 dB | 0.999791 | 40.1199 ms |

前两张是旧W8候选的校准图片；狗图未参与其校准。最终FP16图不依赖这些图片进行PTQ。三图样本不能代替完整视觉数据集评估。

使用同一个Transformers FP32语言模型、同一图文prompt，仅替换视觉特征做隔离验证：

| 视觉特征来源 | 生成描述 |
|---|---|
| HF FP32 | Two cats are sleeping on a pink blanket with remote controls nearby. |
| 旧设备encoder | A close-up of a fluffy, multi-colored cat lying on a soft pink blanket. |
| 新设备encoder | Two cats are sleeping on a pink couch with remote controls nearby. |

新视觉特征恢复了“两只猫”和“遥控器”信息。**这一表使用HF FP32语言模型，不是完整量化PTE的生成结果。** prompt/token IDs和完整输出保存在 `caption_check.json`。

3个回归测试通过。FP32图与HF对照、所有设备输出均检查了形状和有限值。

## 5. 部署与仍需继续的decoder问题

选定模型已部署到：

```text
/data/local/tmp/executorch/qwen35_context4096_ar128/encoder_accuracy/selected/vision_encoder_qnn.pte
```

当前workspace的 `vision_encoder_qnn.pte` 链接已切换到它，现有runner无需修改encoder参数。原始链接目标 `/data/local/tmp/executorch/qwen35_generation_c4096_gqa_shared/vision_encoder_qnn.pte` 保留；冻结版本未覆盖。

- 新encoder SHA256：`2542c3575a93f180673e00e7dd767ce5c03190e348137be37d6786ac299db9bc`。
- 原encoder SHA256：`2fa0f90e6ed5013e63512304db683c4144fdd73165352c57695342456c1f3930`。
- 语言PTE仍为 `e619d39b951151e05d48d85f7e695565009181a449663aab34671943378f1320`，AR128未改。
- 常规 `llama.py` 以后导出Qwen3.5时按新配置选择FP16视觉塔；4K benchmark重新部署时优先使用已保存的选定视觉PTE，避免重新链接旧encoder。

完整PTE已做一次带图smoke test，217个输入token、最多64个输出token、正常遵循EOS。仍输出 `Thegis Capt ...` 等错误内容。该次TTFT为289 ms、decode约37.06 token/s，只记录运行情况，不作为完整性能结论。

当前decoder输入实数范围仍只有[-0.13769455,0.15332107]，**新视觉输出有24.0817%被截断**。因此下一步需要修复图文输入校准，并继续验证此前已经测到的量化语言路径误差；不能承诺只改输入范围就能恢复全部整模精度。

## 6. 证据与复现

- 当前模型、源码快照及部署记录：../selected/manifest.json（历史产物已清理）、选定PTE（历史产物已清理）。
- 完全相同输入的最终配对结果：release_validation/summary.json（历史产物已清理）。
- 三图设备结果：fp16_sigmoid_gelu/device_comparison.json（历史产物已清理）。
- 同一HF语言模型的视觉特征对照：caption_check.json（历史产物已清理）。
- 完整量化PTE的真实输出：../selected/full_pte_output.txt（历史产物已清理）、日志（历史产物已清理）。
- bias分析：`baseline/parameter_error.json`、`per_channel/parameter_error.json`、`per_channel_bias_safe/bias_protection.json`。
- 单次导出、编译、设备执行的日志及命令保存在各候选目录；清理过的3个设备临时候选PTE都先校验了主机备份，审计见 `storage_cleanup.json`。

激活 `et_npu` 并 `source env.sh` 后，视觉图可单独重建，按顺序执行：

```bash
python llama-output/qwen3_5-prefill-optimization/image_benchmark/encoder_accuracy/repair.py prepare_fp16 --case fp16_sigmoid_gelu
python llama-output/qwen3_5-prefill-optimization/image_benchmark/encoder_accuracy/repair.py compile --case fp16_sigmoid_gelu --fp16
python llama-output/qwen3_5-prefill-optimization/image_benchmark/encoder_accuracy/run_repair_device.py --case fp16_sigmoid_gelu --iteration 10 --warm-up 3
pytest -q backends/qualcomm/tests/test_vision_gelu_tanh.py
```

已部署版本的同输入新旧对照可直接运行 `validate_release.py`。没有创建Git commit。

## 7. 保留 GELU 修正后重新测试 W8

2026-10-14，按用户要求单独重建两种 W8 视觉图。语言模型没有重导，prefill AR128、GDN 和 KV cache 配方不变。下面的 W8 是视觉 Linear 权重精度；不代表将语言模型全部改成 W8。

两版都保留 `StaticGELUTanh` 的 sigmoid 等价公式。全部 51 个 Linear（含 patch 投影和 merger）的权重均为 INT8、按输出通道对称量化，范围 [-127,127]；已经从保存的 QDQ 图逐个检查。两版都降低为一个 QNN delegate，没有 CPU 算子回退。

- **W8A16**：激活为 uint16，MinMaxObserver 在 GPU 上用女神像和猫图校准，狗图留作未参与校准的测试。bias 使用派生 int32 参数；保留 patch 投影 3/768 个通道的 bias 容量保护，最大 weight scale 调整倍数为 1.868976。
- **W8＋FP16 激活**：使用仓库已有 `use_fp16a8w` 配置，Linear 权重仍为 INT8；激活、bias、LayerNorm 等走浮点路径，HTP 使用 FP16。CPU QDQ 中有 51 个 INT8 权重节点，没有激活 quantize_per_tensor 节点。FP32 外部接口保留。权重量化的 min/max 来自权重 observer，无须依赖激活校准范围。
- 原先的 FP16 视觉模型继续作为默认部署版本，两份 W8 PTE 留在独立实验目录。

### 7.1 相同输入的配对结果

三个模型都在本轮重新执行，使用与原修复报告完全相同的猫图 raw、HF FP32 输出、3 次 warm-up 和 10 次计时。输入 SHA256 仍为 `285dd25333ac795ff235352337cb2bd287a37873ef70f0f4756ebb17613969af`。SQNR 比较 encoder＋merger 之后的 196×1024 图像 embedding。

| 视觉配置 | 设备 SQNR | Cosine | Encoder 执行时间 | PTE 大小 |
|---|---:|---:|---:|---:|
| FP16 基线，保留 GELU 修正 | **34.68497 dB** | 0.999830 | **40.1745 ms** | 205,551,296 bytes |
| W8A16 per-channel＋bias 保护 | **23.73154 dB** | 0.997968 | **28.1828 ms** | 108,477,424 bytes |
| W8＋FP16 激活 per-channel | **23.52160 dB** | 0.997876 | **45.8386 ms** | 107,170,272 bytes |

W8A16 的执行时间减少 **29.85%**，PTE 缩小 **47.23%**，但 SQNR 降低约 10.95 dB。W8＋FP16 激活仅缩小体积，本轮未测到性能优势，也没有恢复 FP16 基线精度。不能把“W8 跑通”当作与 FP16 等精度。

### 7.2 三张图与 CPU QDQ

本表来自另一轮三图执行，预处理与三个候选保持一致，同样先 warm-up 3 次再计时 10 次。与上表严格配对的 raw 存在约 6e-8 的预处理数值差异，量化后可能改变少量 codes，所以分别保存两轮结果，不混用两轮猫图数值。

| 图片 | FP16 设备 SQNR（上一轮基线） | W8A16 CPU QDQ / 设备 SQNR | W8＋FP16 激活 CPU QDQ / 设备 SQNR |
|---|---:|---:|---:|
| 猫图，参与 W8A16 校准 | 34.68376 | 23.47782 / **23.79687** | 23.58435 / **23.52138** |
| 女神像，参与 W8A16 校准 | 33.03200 | 24.10326 / **23.86624** | 24.11785 / **22.65786** |
| 狗图，未参与校准 | 33.78964 | 17.88016 / **17.87568** | 20.71055 / **20.26948** |

三图平均执行时间分别为上一轮 FP16 **40.0187 ms**、本轮 W8A16 **28.8502 ms**、本轮 W8＋FP16 激活 **45.2254 ms**。正式速度比较优先使用 7.1 的本轮相同输入配对数据。

**误差已存在于权重量化后的 CPU 图。** 即使完全去掉激活量化，W8＋FP16 激活的 CPU QDQ 最终仍只有约 20.71–24.12 dB。因此不能只归因于设备执行或激活校准；INT8 权重量化本身及误差在网络中的传播仍然明显。

CPU QDQ 的累计边界 SQNR 示例：

| 边界 | W8A16 猫图 / 狗图 | W8＋FP16 激活 猫图 / 狗图 |
|---|---:|---:|
| patch_embed | 49.995 / 42.713 | 49.998 / 48.503 |
| block00 残差输出 | 36.643 / 32.222 | 36.645 / 37.351 |
| block11 残差输出 | 23.087 / 20.027 | 22.770 / 23.971 |
| merger LayerNorm 输出 | 18.916 / 13.667 | 19.097 / 16.827 |
| merger 最终输出 | 23.478 / 17.880 | 23.585 / 20.711 |

这些是相对 FP32 的累计误差，不能据此把某一个算子的 SQNR 下跌全部解释为该算子自身的局部量化误差。浮点激活版本中 LayerNorm 也会放大上游误差。W8A16 的狗图还受激活范围影响：去掉激活量化后，patch SQNR 提高约 5.79 dB、最终 CPU 输出提高约 2.83 dB。

### 7.3 图片描述与完整模型限制

同一个 Transformers FP32 语言模型、同一张猫图、同一组 prompt/token IDs，只替换视觉特征，greedy 生成最多 64 token：

| 特征来源 | 图片描述 |
|---|---|
| HF FP32 | Two cats are sleeping on a pink blanket with remote controls nearby. |
| FP16 设备视觉塔 | Two cats are sleeping on a pink couch with remote controls nearby. |
| W8A16 设备视觉塔 | Two cats are sleeping on a pink blanket with remote controls nearby. |
| W8＋FP16 激活设备视觉塔 | Two cats are sleeping on a pink blanket with remote controls nearby. |

两版 W8 都能恢复“两只猫”和“遥控器”，但这是单张图片的隔离验证，不能替代完整视觉集评估，也不是量化整模 PTE 的生成结果。

沿用现有 decoder 输入量化网格时，相同猫图的 FP16 特征仍有 **24.0817%** 越界，W8A16 为 **24.5536%**，W8＋FP16 激活为 **24.5870%**。因此换 W8 没有解决图文输入校准问题。此阶段尚未重新跑 W8 的量化整模生成，也没有覆盖已验证的 FP16 默认部署；后续整模手机实测见第 8 节。

### 7.4 产物和复现

- 本轮配对结果：w8_validation/summary.json（历史产物已清理）。
- 参数、哈希、文件大小、配对精度和 decoder 输入截断比例：w8_comparison.json（历史产物已清理）。
- W8A16：PTE（历史产物已清理）、CPU 逐层结果（历史产物已清理）、设备结果（历史产物已清理）、权重检查（历史产物已清理）。
- W8＋FP16 激活：PTE（历史产物已清理）、CPU 逐层结果（历史产物已清理）、设备结果（历史产物已清理）、权重检查（历史产物已清理）。
- 图片描述隔离验证：w8_caption_check.json（历史产物已清理）。

两份 PTE 的主机/设备 SHA256 已一致核对。W8A16 为 `d8fc472a62a5120afb70425a1cc748770401d0c5e99c4c24bc5fbe9e8fe19a02`；W8＋FP16 激活为 `36d244152d92f12a4ec0e3fee1883d17aa3a7fae685df9a972151910c32e2bfc`。设备空间不足时仅删除了有同哈希主机备份的旧 direct-tanh 临时候选，记录见 `w8_storage_cleanup.json`。

激活 `et_npu` 并 `source env.sh` 后串行执行：

```bash
python llama-output/qwen3_5-prefill-optimization/image_benchmark/encoder_accuracy/repair.py prepare --case w8a16_sigmoid_gelu --per-channel --protect-bias
python llama-output/qwen3_5-prefill-optimization/image_benchmark/encoder_accuracy/repair.py compile --case w8a16_sigmoid_gelu
python llama-output/qwen3_5-prefill-optimization/image_benchmark/encoder_accuracy/run_repair_device.py --case w8a16_sigmoid_gelu --warm-up 3 --iteration 10

python llama-output/qwen3_5-prefill-optimization/image_benchmark/encoder_accuracy/repair.py prepare --case w8fp16_sigmoid_gelu --per-channel --fp16-activations
python llama-output/qwen3_5-prefill-optimization/image_benchmark/encoder_accuracy/repair.py compile --case w8fp16_sigmoid_gelu --fp16-activations
python llama-output/qwen3_5-prefill-optimization/image_benchmark/encoder_accuracy/run_repair_device.py --case w8fp16_sigmoid_gelu --warm-up 3 --iteration 10

python llama-output/qwen3_5-prefill-optimization/image_benchmark/encoder_accuracy/validate_release.py --w8
python llama-output/qwen3_5-prefill-optimization/image_benchmark/encoder_accuracy/caption_check.py --cases fp16_sigmoid_gelu w8a16_sigmoid_gelu w8fp16_sigmoid_gelu --output llama-output/qwen3_5-prefill-optimization/image_benchmark/encoder_accuracy/repair/w8_caption_check.json
```

## 8. W8A16 接完整量化语言模型的手机实测

2026-10-14，按用户要求将已编译的 W8A16 视觉 PTE 接入手机上的完整 ExecuTorch 图文 runner。此次实际运行的是 `qnn_multimodal_runner_u16_argmax`，分别加载视觉、token embedding 和语言 PTE，不使用 Transformers 生成。W8 指视觉塔，语言权重继续采用现有混合 W4/W8 配方。

W8 视觉文件已经在上一阶段编译并部署，本次直接使用同一份产物，没有重新导出未修改的语言图。主机/设备 SHA256 重新核对：视觉 `d8fc472a...fe19a02`，语言 `e619d39b...78f1320`；完整值和三个 PTE 的位置保存在 artifact_manifest.json（历史产物已清理）。宿主目录中放置了指向实际模型文件的链接，避免重复复制。

runner 通过 `--encoder_path encoder_accuracy/repair/w8a16_sigmoid_gelu/vision_encoder_qnn.pte` 显式加载 W8A16，所以本次测试不受默认 FP16 encoder 链接影响。

### 手机结果

图片为同一张 448×448 猫照片，输出 196 个图像 token，加模板/文字后输入总长 217 token；prompt 为 `<image>Describe this image in one sentence.`。greedy，temperature=0，正常遵循 EOS，最多生成 64 token。两次图文生成均未达到 EOS，最终由 seq_len=281 限制停止。

| 测试 | Prompt 处理方式 | 视觉编码 | 首 token 时间 | Decode 速度 | 生成内容 |
|---|---|---:|---:|---:|---|
| Hybrid 图文 | AR128，2 个 block | 28.132 ms | 271 ms | 39.0819 token/s | `** 1. 1 1 1 ...`，错误 |
| KV 图文 | AR1，217 次 forward | 30.101 ms | 5763 ms | 36.9285 token/s | `(1) 3 3 3 ...`，错误 |
| Hybrid 纯文本对照 | `What is 2 + 3? Answer with only the number.`，26 个输入 token | 不运行视觉塔 | 129 ms | 不用于速度比较 | `5`，随后正常 EOS |

两项图文 decode 时间覆盖首 token 之后的 63 次 forward。每项仅运行一次，用于功能诊断，不是重复计时的性能结论；不能将 KV 模式处理 217 个 prompt token 的时间当作 AR128 prefill 性能。纯文本只生成了答案和结束 token，其速度样本过短，不作比较。

Hybrid 图文的 decoder prefill execute 为 238.870 ms；KV 图文为 5693.019 ms。两个模式都成功加载 W8 文件、执行视觉塔并生成文本，没有编译或运行失败，但**图文内容仍未通过验证**。

### 结论与当前边界

W8A16 的视觉编码延迟与单独测试相符，但仅换视觉 PTE 无法修复量化整模的图文输出。上一节同一个 FP32 HF decoder 使用这份 W8 特征能正确描述猫图，本节量化语言 PTE 仍重复输出数字。

已确认现有 decoder 输入网格会截断这张图 W8 特征的 **24.5536%**，这是下一步必须修复的误差点；本次测试没有证明它是所有整模误差的唯一原因。纯文本 `2+3=5` 只证明该简单对照能正确生成，不能替代此前的 PPL/hidden SQNR 验证。

所有原始命令、日志、生成文本、速度文件和 结构化结果（历史产物已清理） 都保存在 `w8a16_full_pte/`。独立 W8 运行脚本保留了显式 encoder 路径；默认 FP16 部署链接未被覆盖。

激活 `et_npu` 并 `source env.sh` 后，可重新执行全部三项：

```bash
python llama-output/qwen3_5-prefill-optimization/image_benchmark/encoder_accuracy/run_w8_full_pte.py
# 只运行 AR128 图文：
python llama-output/qwen3_5-prefill-optimization/image_benchmark/encoder_accuracy/run_w8_full_pte.py --cases hybrid_image
```

该脚本复用当前设备 workspace，核对模型和图片 SHA256，不清空 workspace，也不重新上传大型语言模型。

---

<a id="record-48"></a>

## 记录 48：multimodal_calibration / REPORT.md

原始记录位置：`llama-output/qwen3_5-prefill-optimization/multimodal_calibration/REPORT.md`。

# Qwen3.5-0.8B 图文输入校准与 prefill state 修复

设备 SM8750 / HTP v79，视觉使用已验证的 W8A16 per-channel PTE。语言权重保持 K/z W8、其余大 Linear W4 group32，以及已有 out_proj GPTQ codes/scales。prefill AR128、chunk64/leaf8，KV 仍使用现有 VTCM op；没有新增 DSP op。

## 1. 原始导出缺少图文校准数据

原 4K PTE 是从此前 `export_complete_hybrid.py` 的纯文本 QDQ 图扩展得到的。上游 `export_full_gptq_prefill128.py::load_model` 只读 8 段 Wiki 文本的前 128 token，从未合并视觉 embedding。`export_context4096.py` 仅扩展 context/RoPE 并保留原编码，无法补上图像分布。

原 decoder 输入范围只有 `[-0.13769455, 0.15332107]`。同一猫图约 24.55% 的 W8 特征越界。W8 视觉特征接 HF FP32 decoder 可描述猫和遥控器，接原量化语言 PTE 则重复数字。

这次修复的是实际选定导出产物的校准来源。标准 `PTQStrategy` 本身有合并 vision intermediate outputs 的路径，不能据此认定所有标准图文 PTQ 都漏掉了图片。

## 2. 重新统计激活范围

`prepare_data.py` 使用手机 W8 视觉 PTE 的实际浮点输出和本地 checkpoint 的 tokenizer、三维 M-RoPE 位置构造输入。猫图、女神像参与校准，狗图不参与；另保留 8 段原文本校准文章。文本每段使用前 128 token 做 prefill、后续 16 token 做带历史 decode。图文 decode 包括 prompt 剩余部分和 HF 教师答案。

`recalibrate.py` 保留原权重 Q/DQ、INT4/INT8 codes 和 GPTQ scales，仅在激活 Q/DQ 处放置透明 MinMaxObserver。AR128 与 AR1 共用按原 encoding 关联的 observer；执行时传递真实更新的 conv/recurrent/FA cache。原文本范围与新增统计取并集，避免缩小已有覆盖。

全部校准在 CUDA 上完成，TF32 关闭，串行处理模型；共重新统计 1149 组共享激活网格。FP32 静态参数、权重 codes、weight scales 在改写激活参数前后逐字节一致。后续原生 RMSNorm 改写也保留既有权重。

### Decoder 输入网格

| 项目 | 原始 | 图文校准后 |
|---|---:|---:|
| scale | 4.440613793e-6 | 9.590110130e-5 |
| zero point | 31008 | 48388 |
| code 范围 | 0–65535 | 0–65535 |
| 实数范围 | −0.137695～0.153321 | −4.640462～1.644416 |

以下是保存的三图 W8 输出，分别通过原、新输入 QDQ 的离线检查；SQNR 的参考是 **QDQ 之前的 W8 特征**，不是 HF FP32 视觉输出，也不是整模最终 hidden SQNR。W8 视觉塔自身的约 18–24 dB 误差没有被这一表消除。

| 图片 | 原越界比例 | 新越界比例 | 原输入 QDQ SQNR | 新输入 QDQ SQNR |
|---|---:|---:|---:|---:|
| 猫 | 24.5371% | 0.000498% | 3.7559 dB | 76.2788 dB |
| 女神像 | 32.6501% | 0% | 3.7849 dB | 76.9739 dB |
| 狗，未参与校准 | 30.1588% | 0% | 3.7446 dB | 76.6935 dB |

猫图只有 1 个值略超新端点，误差小于半个量化步长。另一次严格相同 raw 的猫图运行输出略有不同，原比例为24.5536%；各轮证据分别保留，不混合其数字。

## 3. Cache 和 metadata 也必须同步

48 路 cache 输入/输出的 scale、zero point 都从改写后的图读取，包含 conv state、recurrent state 和 FA K/V。只更新12路 FA metadata 会让36路 GDN cache 用旧 zero point 初始化，因此本轮显式审计全部48路。

两种方法的 cache 编码逐项相同，每路输出编码与对应输入一致。logits scale/zero point 也从新输出读取；PPL runner 使用新 logits scale。原始 production PTE 保留作对照。

## 4. 只扩大量化范围仍不足

首版图文校准 PTE 已上手机：KV 猫图从重复数字变为猫在粉色毯子上的句子，狗图能识别白狗和草地，但出现指令幻觉；hybrid 分别只输出 `Two`、`A` 就结束，纯文本 `2+3` 首 token 为 `5`，随后重复中文片段。

该版连续1K文本 KV PPL为 **13.5281**，原始为 **13.7008**，HF FP32为 **10.8785**。输入是同一组 WikiText-2 raw test 前1024 token及实际 embedding PTE 输出；这不是完整测试集 PPL。它没有显示长文本 KV 退化，却暴露了 partial prefill 的问题。

### RMSNorm 分母被量化到零

49个 attention/FFN/final RMSNorm 中，20个的 epsilon 在扩展后的分母网格上会舍入成0。例如 layer0 attention_norm 的 epsilon为约9.994659e-7，分母步长为3.578635e-6；零输入的 `variance + epsilon` 经该QDQ后为0，`rsqrt` 原始计算为无穷大。

`Qwen35RMSNorm` 改用展平至二维的原生 `F.rms_norm`，将 variance/epsilon 保持在算子内部。既有选定 QDQ 图中的49个分解 norm 也改为同一原生算子，保留校准输入/输出网格和 gamma 运算。已有 Q/K 原生 norm 保留。

### 最后一个 AR128 block 的39个 padding token

217-token图文输入需要2个AR128 block，最后89个有效token之后补39个零。旧图依然更新完整128步GDN state，并从包含padding的conv输入末尾取4个值；runner的 `n_update=89` 无法修正已经聚合到单个张量里的最终state。

新 `mask_chunk_prefill_padding` 使用已有 `last_token_index`：有效token保持原decay/beta，padding设为 `decay=1, beta=0`，使state停止更新；conv cache从有效末尾Gather4个值。该变换在FP16 chunk替换前执行，decode AR1不加此mask。

QNN Gather要求输入/输出编码相同，而conv cache需要原输出编码。本轮在Gather两端显式保留输入网格，再独立转换到cache输出网格。最初遗漏这一步导致18个Gather留在CPU，导出检查拒绝该版本；对应失败日志保留。修正后的小图已完整委托到QNN。

5项回归测试通过，覆盖原生RMSNorm数学一致性、partial/full block的GDN state和conv state，以及不同输入/输出量化网格的conv cache。

## 5. 证据与复现

- 数据来源与划分（历史产物已清理）、`cats.pt`/`statue.pt`/`dog.pt` 中的真实特征、位置和HF参考。
- 激活范围（历史产物已清理）、输入边界检查（历史产物已清理）。
- 校准日志（历史产物已清理）、48路cache审计（历史产物已清理）、参数与哈希不变检查记录（历史产物已清理）。
- 首版实际手机生成结果（历史产物已清理）、KV文本PPL（历史产物已清理）。
- 修正版模型与归一化变换记录（历史产物已清理）、Gather小图编译（历史产物已清理）、回归测试（历史产物已清理）。
- 存储清理仅移除了已核对主机同哈希备份的旧诊断PTE，记录在 `storage_cleanup.json`。

在 `et_npu` 环境并 `source env.sh` 后串行执行：

```bash
python llama-output/qwen3_5-prefill-optimization/multimodal_calibration/prepare_data.py
python llama-output/qwen3_5-prefill-optimization/multimodal_calibration/recalibrate.py calibrate
python llama-output/qwen3_5-prefill-optimization/multimodal_calibration/recalibrate.py prepare
python llama-output/qwen3_5-prefill-optimization/multimodal_calibration/recalibrate.py audit
python llama-output/qwen3_5-prefill-optimization/multimodal_calibration/stability_fix.py prepare
python llama-output/qwen3_5-prefill-optimization/multimodal_calibration/stability_fix.py audit
python llama-output/qwen3_5-prefill-optimization/multimodal_calibration/stability_fix.py export
```

## 6. 最终手机结果与当前 release

2026-10-15，修正版完整量化 PTE 已通过手机验证并设为当前4K workspace默认模型。prefill、decode均为单个QNN delegate，没有CPU算子回退。

同一448×448猫图、217输入token，temperature=0，正常遵循EOS，最多输出64token：

| 模式 | 输出 | 首token | Decode速度 |
|---|---|---:|---:|
| Hybrid AR128 | Two cats are sleeping on a pink blanket with remote controls nearby. | 269 ms | 38.2353 token/s |
| KV AR1 | Two cats are lying on a pink blanket with remote controls beside them. | 5775 ms | 36.7454 token/s |

未参与校准的狗图，同样217输入token：

| 模式 | 输出 | 首token | Decode速度 |
|---|---|---:|---:|
| Hybrid AR128 | A fluffy white dog sits on green grass with its tongue out, looking up at the camera. | 288 ms | 36.1217 token/s |
| KV AR1 | A fluffy white dog sits on green grass with its tongue out. | 5883 ms | 36.3128 token/s |

四项图文生成都正常EOS，没有数字重复或提前输出一两个词即结束。纯文本 `2+3` 输出 `5` 后正常EOS；`Explain Python.` 恢复连续英文输出，因64token上限被截断，其中仍有不准确的知识表述，不将其视为全面事实正确性验证。原文保存在日志，不作修饰。

以上每项一次功能诊断；decode只有13–19次forward，不能当作稳定长程性能benchmark。视觉编码约29ms，AR保持128，长上下文FA容量仍为4096。没有重测或覆盖原冻结FA256短prompt的1684–1707 token/s成绩。

### 相同1K文本的实际设备 PPL

| 来源 | Prefill AR128 | KV AR1 |
|---|---:|---:|
| HF FP32 | 10.87851 | 10.87851 |
| 原纯文本校准PTE | 11.72747 | 13.70082 |
| 本轮完整修正版 | **11.74834** | **11.62953** |

Prefill相对原版增加0.18%，基本持平；KV降低15.12%，相对FP32的差距从25.94%缩到6.90%。两种模式现在相近，但仍有量化误差。这里是与此前相同WikiText-2 raw test前1024token、1023个teacher-forced目标和实际embedding PTE输出，不是完整测试集标准PPL。

数据：手机生成原文与计时（历史产物已清理）、prefill PPL（历史产物已清理）、KV PPL（历史产物已清理）。

### 已部署产物

- 当前完整模型清单（历史产物已清理）。
- 语言PTE（历史产物已清理）：480,366,592 bytes，SHA256 `e0b2ae5a75d8c66dbbb354fe5ada277ef107d6ccaee9535cbb402ee27baaeb45`。
- W8A16视觉PTE（历史产物已清理）：108,477,424 bytes，SHA256 `d8fc472a62a5120afb70425a1cc748770401d0c5e99c4c24bc5fbe9e8fe19a02`。
- token embedding和tokenizer继续使用与本轮所有测试完全相同的文件，清单记录了来源和SHA256。

手机默认 `hybrid_llama_qnn.pte`、`vision_encoder_qnn.pte` 已切到这两份文件，四个关键文件的主机/设备哈希已核对一致。原语言模型保存为 `hybrid_llama_qnn.before_multimodal_calibration.pte`，原FP16视觉文件也保留。重新部署4K benchmark时优先使用本轮selected产物；没有创建Git commit。

切换默认路径后另做一次猫图smoke test，生成文本与上述hybrid逐字相同并正常EOS，首token289ms、decode35.91 token/s。追加记录见 selected/runs/results.json（历史产物已清理），保留此次波动，不将单次最快值作为稳定性能。

在现有手机workspace复现图文与简单文本测试：

```bash
python llama-output/qwen3_5-prefill-optimization/image_benchmark/encoder_accuracy/run_w8_full_pte.py
# 验证未参与校准的狗图：
python llama-output/qwen3_5-prefill-optimization/image_benchmark/encoder_accuracy/run_w8_full_pte.py --cases hybrid_dog kv_dog
```

该脚本检测到本轮selected清单后，默认核对并运行新的语言模型，把新结果写到selected/runs，保留旧失败记录。原始编译脚本、全部中间QDQ图、calibration-only失败生成、Gather验证失败日志及小图修复证据均保留，可分别回溯三类问题。

## 7. 修复后的配对性能复测

2026-10-15，原版与修复版4K PTE使用相同视觉塔及快速runner，串行交替测试，各prefill用例3轮、16→4080长decode各2轮，AR128保持，固定生成数量并忽略EOS。28次正式测量和2次预热全部完成。

| 项目 | 原版 | 修复版 |
|---|---:|---:|
| 128→1纯文本prefill | 127.33 ms | 128.33 ms |
| 4096→1纯文本prefill | 3998.33 ms / 1024.43 token/s | 3965.67 ms / 1032.87 token/s |
| 16→4080纯文本decode | 36.9466 token/s | 36.9047 token/s |
| 217-token图文首token | 282.33 ms | 281.33 ms |
| 4096-token图文首token | 4003.33 ms | 4019.67 ms |

本轮未测到明显性能退化：prefill平均耗时变化均在±1%内，长decode吞吐变化−0.11%；视觉编码约29ms，快速采样仍约0.175ms/token。两版均为FA4096长图，不能直接与此前FA256短图约1700token/s比较。完整逐轮数据、阶段拆分、升温记录和复现见[性能报告](README_QWEN35_TEST_RECORDS.md#record-49)。

---

<a id="record-49"></a>

## 记录 49：performance / REPORT.md

原始记录位置：`llama-output/qwen3_5-prefill-optimization/multimodal_calibration/performance/REPORT.md`。

# 图文校准与 state 修复后的性能复测

2026-10-15，Qwen3.5-0.8B，SM8750 / HTP v79，设备 `3B1F57E9B8L74X5D`。

## 1. 比较对象与计时口径

原版为修复前的4K语言PTE `e619d39b...f1320`；修复版为当前手机默认release `e0b2ae5a...aeb45`。本轮只测性能，没有重新导出或改动模型。修复版包含图文激活校准、cache编码同步、49个native RMSNorm以及partial prefill的GDN/conv state修复。语言权重、GPTQ codes/scales、K/z W8与其余大Linear W4配置沿用原值。

两版共用同一个W8A16视觉PTE、token embedding、tokenizer、快速UINT16 argmax runner及DeltaNet op package。两版都是FA内部容量4096、物理容量4224、prefill AR128、GDN chunk64 / leaf8 / FP16及KV快速VTCM算子。7个模型/输入/runner文件在测试前核对SHA256，op package另记录结束时SHA256；完整路径和哈希见本轮清单（历史产物已清理）。

- 串行运行；每版先做一次16→64的KV预热，预热不计入结果。每个测量都是新的runner进程，模型加载时间排除，prompt阶段的首次graph调用仍计时。
- 四种prefill/图文用例各3轮配对，长decode各2轮配对。偶数轮原版→修复版，奇数轮修复版→原版；长decode顺序为原版、修复版、修复版、原版。
- temperature=0、忽略EOS、固定输入/输出数量，避免修复前后生成内容和结束位置不同造成性能比较偏差。忽略EOS后的文本不作为生成质量评测。
- 纯文本prefill：输入128或4096，输出1。图文短用例：猫图196个视觉token加21个文本/模板token，共217输入、64输出；长图文用例总输入4096（含196个视觉token）、输出1。
- 长decode严格按用户确认的口径：**输入16、输出4080**。第一个输出token由prompt阶段生成，decode计时覆盖剩余4079次forward，吞吐为 `4079 / decode耗时`。
- 首token/端到端prefill计时从推理开始到prompt evaluation结束，包含embedding、图文合并、视觉编码（有图时）、prefill执行和cache处理；不包含模型加载。吞吐用token数除以平均耗时，不取最好一轮。

## 2. Prefill 与图文首 token

| 用例 | 原版平均首token | 修复版平均首token | 耗时变化 | 原版吞吐 | 修复版吞吐 |
|---|---:|---:|---:|---:|---:|
| 纯文本128→1 | 127.33 ms | **128.33 ms** | +0.79% | 1005.24 token/s | **997.40 token/s** |
| 纯文本4096→1 | 3998.33 ms | **3965.67 ms** | -0.82% | 1024.43 token/s | **1032.87 token/s** |
| 图文217→64（首token） | 282.33 ms | **281.33 ms** | -0.35% | 768.60 token/s | **771.33 token/s** |
| 图文4096→1 | 4003.33 ms | **4019.67 ms** | +0.41% | 1023.15 token/s | **1018.99 token/s** |

上述四项平均耗时变化均在±1%内，本轮未测到明显prefill性能退化。三轮样本不能证明严格等速，也不将其中小幅提速解释为修复带来的确定收益。

### 逐轮首 token，单位 ms

| 用例 | 原版三轮 | 修复版三轮 |
|---|---|---|
| 纯文本128→1 | [127, 128, 127] | [130, 129, 126] |
| 纯文本4096→1 | [4036, 3996, 3963] | [4101, 3849, 3947] |
| 图文217→64（首token） | [284, 280, 283] | [281, 281, 282] |
| 图文4096→1 | [4110, 3931, 3969] | [4041, 3989, 4029] |

### Prompt 阶段拆分，均值 ms

这里的execute是runner围绕模型调用的墙钟计时，包含delegate/runtime开销，不是独立的HTP硬件计时。端到端计时还包含没有列入这三个子阶段的embedding、图文合并等工作。

| 用例 | 版本 | 视觉编码 | prepare | execute | cache |
|---|---|---:|---:|---:|---:|
| 纯文本128→1 | 原版 | 0.000 | 0.063 | 122.819 | 0.194 |
| 纯文本128→1 | 修复版 | 0.000 | 0.066 | 124.126 | 0.186 |
| 纯文本4096→1 | 原版 | 0.000 | 2.777 | 3914.540 | 14.710 |
| 纯文本4096→1 | 修复版 | 0.000 | 2.899 | 3880.585 | 15.160 |
| 图文217→64（首token） | 原版 | 29.257 | 0.083 | 248.957 | 0.347 |
| 图文217→64（首token） | 修复版 | 29.275 | 0.085 | 248.056 | 0.348 |
| 图文4096→1 | 原版 | 30.181 | 2.880 | 3890.906 | 14.554 |
| 图文4096→1 | 修复版 | 29.399 | 3.044 | 3908.351 | 15.305 |

217-token图文用例的视觉编码为原版29.26 ms、修复版29.28 ms；视觉PTE没有改变。固定63次后续decode的平均吞吐两版均为37.78 token/s；该短用例不能替代下面的4K长decode。

## 3. 长 decode：16→4080

| 版本 | 第1轮 | 第2轮 | 平均decode耗时 | 平均decode吞吐 |
|---|---:|---:|---:|---:|
| 原版 | 37.0125 token/s | 36.8810 token/s | 110.4025 s | **36.9466 token/s** |
| 修复版 | 36.6629 token/s | 37.1497 token/s | 110.5280 s | **36.9047 token/s** |

修复版吞吐变化为-0.114%，4079次decode的平均总耗时增加0.1255秒，约0.0308 ms/step。相对每次运行的波动，这不是明显降速。

### 每个 decode token 的阶段均值

| 阶段 | 原版 ms/token | 修复版 ms/token |
|---|---:|---:|
| Embedding/输入准备 | 0.23355 | 0.23244 |
| Decoder step（含delegate/runtime） | 26.48685 | 26.50489 |
| 量化logits argmax | 0.16679 | 0.17507 |
| Cache/mask更新 | 0.16547 | 0.16899 |
| Tokenizer/callback | 0.01238 | 0.01469 |

快速argmax仍正常生效，采样约0.17 ms/token，没有回到此前约4 ms/token的float转换路径。耗时主要仍在decoder step，本轮没有进一步用硬件profile拆分GDN、FA与各投影，不能从这些数据把0.02 ms左右的decoder差异归因到某一个修复。

## 4. 与此前约1700 token/s的关系

**本轮是FA4096长上下文PTE；此前1684–1707 token/s来自FA内部容量256的短prompt PTE。** 两者对外AR都为128，内部FA计算长度不同。本轮在同一个FA4096配置下，原版128→1也是约1005 token/s，修复版约997 token/s；4096→1则分别约1024、1033 token/s。

因此，不能把1700与997直接相除，得出本次精度修复损失四成性能。本轮没有重导或覆盖冻结的FA256短图，不能声称已经验证它接入图文修复后的性能。短图和4K图的既有区分及旧配对数据见[前次prefill报告](README_QWEN35_TEST_RECORDS.md#record-43)。

## 5. 波动、环境与结论

Prefill测量期间电池温度32.4–32.7°C；连续长decode期间从32.7升至38.4°C。这里记录的是Android battery温度，不是HTP结温或硬件频率。长decode采用ABBA顺序减轻随时间升温的影响，未将其当作严格恒温测量，也不挑选最快一轮。

两版prompt后的RSS均约566–567 MiB，没有观察到明显内存增量。30次串行执行全部成功，其中2次预热、28次正式测量；每次都校验实际prompt token数、输出forward数及hybrid AR128 block数。

**结论：当前4K修复版的图文精度改善没有带来可见的整体性能损失。AR128保持，4K纯文本prefill约1033 token/s，长decode约36.90 token/s，单图视觉编码约29 ms、217-token图文首token约281 ms。** 此处仅给出本机配对实测，不代表所有prompt、所有图片或恒温条件下的普遍结论。

## 6. 证据与复现

- 逐轮原始记录（历史产物已清理）、汇总（历史产物已清理）、文件哈希与测试口径（历史产物已清理）。
- 每次命令、完整runner日志、生成文本均保存在`paired_20261015/`；本次执行的脚本快照（历史产物已清理）。
- 精度与正常EOS生成另见[图文修复报告](README_QWEN35_TEST_RECORDS.md#record-48)，本轮强制生成长度的文本用于计时。

在保持现有手机workspace及两版PTE的情况下重测；使用新的输出目录以保留历史结果：

```bash
python llama-output/qwen3_5-prefill-optimization/multimodal_calibration/benchmark_performance.py \
  --out llama-output/qwen3_5-prefill-optimization/multimodal_calibration/performance/paired_rerun \
  --prefill-repeats 3 --decode-repeats 2
```

脚本显式选择旧备份或修复版PTE，不依赖当前默认decoder别名，不重新部署、不清空设备workspace。

---

<a id="record-50"></a>

## 记录 50：release_ar128 / README.md

原始记录位置：`llama-output/qwen3_5-prefill-optimization/release_ar128/README.md`。

# 选定版本：prefill AR128 / GDN chunk64

版本：`qwen35-prefill-ar128-chunk64-20261014`。按用户要求停止继续优化。

- 128-token prompt端到端prefill：1684–1707 token/s。
- 对外prefill AR=128，内部为两个64-token GDN块；decode AR=1。
- hybrid prefill当前最大prompt为256 token，KV容量4096。
- 抽样PPL为49.8617，原版50.0248；8个独立128-token窗口，非标准全量评测。
- `hybrid_chunk_qnn.pte`、`prefill_qdq.pt2`、runner和SO均为独立保存的副本，校验值在`manifest.json`。
- 选定版本使用稳定分块三角逆；没有采用SLC、后续preconditioned或fused候选。
- `source.tar.gz`保存相关源码快照；`working-tree.patch`包含当时整个未提交工作区的差异。

使用已部署版本，采用已测试的128-token prompt：

```bash
python llama-output/qwen3_5-prefill-optimization/release_ar128/run_selected.py
python llama-output/qwen3_5-prefill-optimization/release_ar128/run_selected.py --output-tokens 1
```

脚本不重新导出模型，默认正常遵循EOS。PTE运行依赖设备上现有的embedding、tokenizer、vision encoder、QNN SDK库和op package目录，路径在manifest中。

优化说明：[FINAL_SUMMARY.md](README_QWEN35_TEST_RECORDS.md#record-02)。历史记录：[REPORT.md](README_QWEN35_TEST_RECORDS.md#record-40)。

---

<a id="record-51"></a>

## 记录 51：text_accuracy / REPORT.md

原始记录位置：`llama-output/qwen3_5-prefill-optimization/text_accuracy/REPORT.md`。

# 当前 Qwen3.5-0.8B PTE 与 Transformers 的纯文本精度对比

测试日期：2026-10-14。设备：SM8750 / HTP v79，序列号 `3B1F57E9B8L74X5D`。

## 1. 结论

**纯文本也存在精度损失，逐 token KV 路径比 AR128 prefill 更明显。** 本轮使用同一批真实 token、相同位置和因果历史，设备采用实际 embedding PTE 的输出。PPL 直接取原生产 PTE 的输出；SQNR 取 LM head 之前的最终 hidden state。

| 输入 | Transformers FP32 PPL | PTE prefill PPL | 相对FP32增加 | PTE KV PPL | 相对FP32增加 |
|---|---:|---:|---:|---:|---:|
| WikiText 连续1024 token | 10.87851 | 11.72747 | 7.80% | 13.70082 | 25.94% |
| 8个冷启动128-token窗口 | 41.32729 | 49.85514 | 20.63% | 67.40884 | 63.11% |

| 输入 | Prefill 最终hidden SQNR | KV 最终hidden SQNR | Prefill 与HF的top-1一致率 | KV 与HF的top-1一致率 |
|---|---:|---:|---:|---:|
| WikiText 连续1024 token | 11.22 dB | 7.31 dB | 84.36% | 77.61% |
| 8个冷启动128-token窗口 | 11.88 dB | 6.99 dB | 76.38% | 64.96% |

之前保留版本的冷窗口 PPL 49.86，是与旧量化版本 50.02 对照；本轮补上 Transformers FP32 基准约41.33，说明两份量化版本接近并不等于已经恢复浮点精度。

## 2. 公平比较的口径

- 同一份本地 checkpoint：`/hunter/model/Qwen/Qwen3.5-0.8B`，Transformers `Qwen3_5ForConditionalGeneration`，FP32、eager attention、关闭TF32，CUDA执行。
- 当前生产模型为 `../context4096_ar128/hybrid_llama_qnn.pte`。K/z投影W8；q/v/a/b合并投影W4；其余大Linear沿用原W4 group32和已保留的GDN out_proj GPTQ；state、conv、FA cache的外部接口UINT16。没有重新校准或修改这些权重。
- Prefill：AR128、GDN内部FP16 chunk64 / leaf8，跨128-token调用携带state和FA cache。KV：AR1、原整数VTCM core，每个token更新cache。两种模式各自独立从零历史开始，不在本轮PPL中混合切换。
- 实际embedding PTE在设备执行同一批token；其FP32输出直接输入decoder。Transformers使用checkpoint原始embedding，embedding量化误差因此包含在PTE的最终结果中。
- token IDs保存在 `cases.json`。两个模型的三组position IDs一致，纯文本三组都是 `0..T-1`；因果mask允许同样的历史。每个独立样本清空conv/state/FA cache，连续1K流不在128边界清空。
- PPL采用teacher forcing：位置t的输出预测真实token[t+1]，不把各自自由生成的不同文本拿来评分。完整1K共1023个目标；8个冷窗口各127个目标，共1016个。聚合先累加NLL再除以目标数，最后取exp。
- 原PTE的LM head只输出选中位置。prefill评分时，对同一个128-token块保持输入cache不变，依次选取每个位置的logits；完成整块后更新一次cache。重复执行只用于精度计分，不能用这轮耗时作为prefill性能。
- UInt16 logits使用PTE实际 `get_logits_scale` 计算稳定logsumexp。共同zero point在softmax中抵消，没有凭空假设scale或zero point。计分公式与保存的原始logits交叉核对过。
- **数据范围限制**：连续1K为WikiText-2 raw test拼接文本的前1024 token，不是完整测试集的标准PPL。8个冷窗口来自之前校准文章的保留后缀，不是独立测试文章。两组结果分别报告，不混为一个PPL。

## 3. 最后一个RMSNorm之前和之后

SQNR定义为 `10*log10(sum(HF_hidden²)/sum((PTE_hidden-HF_hidden)²))`，参考是Transformers相同位置的hidden。跨样本累加信号/噪声能量后计算，不对各样本dB直接平均。

`pre_norm`是第24层残差输出；`final_norm`是最终RMSNorm输出、LM head输入。这里的主要SQNR不是整个词表logits的SQNR。

| 输入 | 模式 | RMSNorm之前 SQNR | RMSNorm之后 SQNR | 最终hidden cosine | 最终hidden RMSE |
|---|---|---:|---:|---:|---:|
| WikiText 连续1024 token | Prefill AR128 | 11.91 dB | 11.22 dB | 0.961960 | 1.091565 |
| WikiText 连续1024 token | KV AR1 | 7.96 dB | 7.31 dB | 0.930013 | 1.712056 |
| 8个冷启动128-token窗口 | Prefill AR128 | 12.66 dB | 11.88 dB | 0.967451 | 1.026664 |
| 8个冷启动128-token窗口 | KV AR1 | 8.09 dB | 6.99 dB | 0.929337 | 1.803675 |

末端RMSNorm之前已经有明显误差，不能把最终hidden的差距全部归因于LM head或最后一个RMSNorm。

### 连续1K的输出范围

| 张量 | HF FP32 min / max | 量化可表示范围 | Prefill端点码比例 | KV端点码比例 |
|---|---|---|---:|---:|
| pre_norm | -11.57942 / 7.50899 | [-12.60516, 7.66772] | 0.000000% | 0.000191% |
| final_norm | -47.68611 / 41.59077 | [-51.65412, 43.16543] | 0.000000% | 0.003242% |

这些编码从现有QDQ图读取：`pre_norm` scale=0.0003093443810939789、zero point=40748；`final_norm` scale=0.0014468536246567965、zero point=35701。两种方法使用相同网格。

端点码比例是输出codes等于0或65535的比例，不能代替量化前真实clipping比例。此处范围覆盖HF最终hidden；prefill的末端没有端点码，仍有上述误差，下一步应检查前面各层产生的误差，而不是只扩大最终hidden的scale。这里没有测所有中间张量的饱和情况。

## 4. 连续128-token分段变化

输入是一条连续1K流，各块携带前面的历史；位置按1开始编号。分段PPL按该位置的下一token预测计分，最后一块只有127个目标。

| token位置 | HF PPL | Prefill PPL | KV PPL | Prefill最终hidden SQNR | KV最终hidden SQNR |
|---|---:|---:|---:|---:|---:|
| 1-128 | 21.6809 | 21.8063 | 24.9349 | 12.38 dB | 7.37 dB |
| 129-256 | 24.5047 | 28.0991 | 36.3390 | 11.77 dB | 7.55 dB |
| 257-384 | 17.4251 | 21.9766 | 28.6190 | 11.37 dB | 7.42 dB |
| 385-512 | 6.2132 | 6.5009 | 6.8365 | 11.41 dB | 7.50 dB |
| 513-640 | 12.7483 | 13.3150 | 15.2638 | 11.33 dB | 7.53 dB |
| 641-768 | 2.3526 | 2.3769 | 2.4235 | 10.21 dB | 6.99 dB |
| 769-896 | 13.4798 | 14.2862 | 17.5621 | 10.91 dB | 7.18 dB |
| 897-1024 | 8.4178 | 9.0211 | 10.7597 | 10.70 dB | 7.00 dB |

连续1K hidden SQNR（历史产物已清理）

图中每32个token合并信号/噪声能量后计算SQNR，灰线是128-token边界；逐token数据见 `hidden_per_token.csv`。

首token从零历史开始，最终hidden SQNR已分别是prefill **0.49 dB**、KV **-5.07 dB**。这说明差异在首次forward就存在，不能只用长上下文cache累计来解释；仍需逐层定位其来源。

## 5. 测量有效性与embedding核对

hidden诊断PTE来自当前生产PTE对应的两份QDQ图，仅追加已有量化hidden输出：原logits和48个cache输出仍保留，额外输出49为final_norm、50为pre_norm。没有重新量化权重或换GDN核，每个方法仍是一个QNN delegate，没有CPU算子回退。

原生产PTE与诊断PTE在 **44个位置** 的全词表UInt16 logits codes **100%逐元素一致，最大code差=0**；KV的完整teacher-forced PPL也一致。这是仪器有效性检查，主要模型SQNR仍使用hidden。

实际embedding PTE相对HF原始embedding的SQNR为 **84.24 dB**。最初直接喂HF embedding时，重入decoder量化网格发现 **90个值** 与设备embedding路径不同，最大只差1个code；最终本报告全部结果已经改用实际设备embedding输出。最初对照也保留，见 `hf_embedding_summary.json` 和不带 `device_embedding_` 前缀的结果目录。

### 附：之前无图短prompt

额外核对了之前性能测试的同一19-token ChatML prompt：`Describe this image in one sentence.`，没有图片。这里仅检查数值路径，不把对不存在图片的描述当作模型精度证据。

这个prompt的teacher-forced PPL：HF 4.99572、prefill 7.61958、KV 10.60394；最终hidden SQNR分别为 5.38 / 2.40 dB。它只有18个计分目标且包含格式token，不与Wiki结果合并。HF对此无图prompt也生成了虚构景物，原始输出见 `hf_chat_generation.json`。

## 6. 后续定位顺序

1. 先固定首token、零state/conv/FA cache，对齐HF与KV图的逐层输出，找到第一个明显掉SQNR的位置。首token无需重放长历史，能缩小排查范围。
2. 对该层的投影、causal conv、Q/K norm、GDN输入/输出、门控、out_proj、FFN分别测相同输入；区别算子本身的局部误差与前层传来的累计误差。
3. 再对照同一组输入的AR1整数GDN和AR128 FP16 chunk路径，检查实际编码、舍入和state接口。当前数据只证明KV整条路径更差，尚不能断言某个GDN核或某个cache就是唯一原因。
4. 首token局部误差明确后，再查128边界和连续decode的state/FA cache误差积累；最终用相同Wiki token集复测PPL。

本轮只增加诊断脚本、诊断PTE和报告，原生产PTE、量化配方、prefill AR128及冻结快照保持原样。

## 7. 证据与复现

- 原PTE SHA256：`e619d39b951151e05d48d85f7e695565009181a449663aab34671943378f1320`。
- 诊断PTE SHA256：`25477817a7a0fc1e6b516bfc2e30e11e2717848b1aa25cd3e272e4404c29f0a2`。
- HF输入/结果：`cases.json`、`hf_scores.json`、`hf_summary.json`、`*_hf.pt`。
- 原生产PTE计分：`device_embedding_production_{cold8,wiki1024,chat19}_{kv_forward,prefill_forward}/sample_*.csv`。
- hidden与对照logits：对应 `device_embedding_probe_*/sample_*_{final_norm,pre_norm}.raw` 及 `sample_*_logits_*.raw`。
- 反量化编码与IO：`probe_metadata.json`。汇总：`summary.json`。逐tokenSQNR：`hidden_per_token.csv`。
- 脚本执行命令及设备日志均保存在各结果目录的 `command.txt`、`runner.log`；编译日志是 `build_runner.log`、`export_probe.log`。

在仓库根目录激活 `et_npu` 并 `source env.sh` 后，单独复测已部署模型：

```bash
python llama-output/qwen3_5-prefill-optimization/text_accuracy/run_eval.py --device-embedding
python llama-output/qwen3_5-prefill-optimization/text_accuracy/run_eval.py --device-embedding --probe
python llama-output/qwen3_5-prefill-optimization/text_accuracy/analyze.py
python llama-output/qwen3_5-prefill-optimization/text_accuracy/write_report.py
```

两个设备测试命令按顺序执行。`reference.py` 可重新生成HF FP32参考；`check_embedding.py` 可重新生成并部署设备embedding输入。

---

<a id="record-52"></a>

## 记录 52：qwen3_5-size-comparison / REPORT.md

原始记录位置：`llama-output/qwen3_5-size-comparison/REPORT.md`。

# Qwen3.5-0.8B / 2B 完整模型性能对照

日期：2026-10-14。设备：SM8750，serial `3B1F57E9B8L74X5D`，QNN 2.49.40.260810，Hexagon v79。

## 1. 结果

本地 `/hunter/model/Qwen/Qwen3.5-2B` 已完成 GPU 校准、18 层 GDN out_proj GPTQ、完整 hybrid PTE 编译和设备文本推理。

同一设备、同一物理 context 容量 4096、prefill AR=128 / decode AR=1、相同精确 VTCM GDN 内核，端到端测试结果如下。每个点跑两次，速度用总 token 数除以总耗时计算；模型加载时间不计入。

| 测试 | 0.8B 两次耗时 ms | 0.8B token/s | 2B 两次耗时 ms | 2B token/s |
|---|---:|---:|---:|---:|
| 输入128，输出1，hybrid prefill | 155 / 155 | **825.81** | 146 / 146 | **876.71** |
| 输入1024，输出1，hybrid prefill | 1182 / 1184 | **865.60** | 1183 / 1181 | **866.33** |
| 输入16，输出64，KV decode | 1719 / 1648 | **37.42** | 2590 / 2602 | **24.27** |

decode 的首个输出由 prompt forward 产生，后续 63 次 decode forward 进入表中计时。因此该行是 63 / decode 秒数，完整输出仍为 64 token。

**核心发现：在 1024-token prefill 上，0.8B 和 2B 的速度相差约 0.08%；decode 则是 0.8B 快约 54.2%。** 当前 prefill 很明显没有随模型参数量缩小而获得对应收益。128-token 短测的少量逆向差异，不应推广成 2B 普遍更快。

这次结果证明需要继续优化两者共同使用的 prefill 路径；还没有达到 2000 token/s。若目标是 2000 token/s，完整 128-token 块的端到端预算为 **64 ms**，本轮每块约 146–155 ms，需要约 2.3–2.4 倍整体提速。

原始记录：`generation/summary.json`，包含每次计时、命令、生成文本；`summary.json` 汇总了速度、profile、PTE hash。

## 2. 两个模型哪些地方相同

核对本地模型配置和实际导出图：

| 结构 | 0.8B | 2B |
|---|---:|---:|
| decoder 层数 | 24 | 24 |
| GDN 层 / full-attention 层 | 18 / 6 | 18 / 6 |
| hidden dim | 1024 | 2048 |
| FFN intermediate dim | 3584 | 6144 |
| GDN key/value heads | 16 / 16 | 16 / 16 |
| GDN key/value head dim | 128 / 128 | 128 / 128 |
| 每层 recurrent state | `[1,16,128,128]` | 相同 |
| 每层 conv state | `[1,6144,4]` | 相同 |
| FA query / KV heads | 8 / 2 | 8 / 2 |
| FA head dim | 256 | 256 |
| vocab size | 248320 | 248320 |
| 合并 K/z 投影权重 | `4096×1024` | `4096×2048` |
| 合并 q/v/a/b 投影权重 | `4128×1024` | `4128×2048` |

因此两者都要执行 18 层、每层 128 步、相同 state 大小的 GDN recurrence；6 层 full attention 的 head 形状和 4K KV cache 大小也相同。模型缩小主要减少投影、FFN、LM head 的参数和计算，不能按总参数量比例缩小这些共同成本。

## 3. 无逐算子追踪的完整 decoder 耗时

`qnn_executor_runner`，profile=0，warmup=2，之后 5 次 forward 均值。输入是相同留出文本 token 对应的各模型真实 embedding；cache 内容初始化为空，静态 capacity 仍为 4096。

| 方法 | 0.8B ms | 2B ms |
|---|---:|---:|
| 完整 decode forward，AR=1 | **22.084** | **33.380** |
| 完整 prefill forward，AR=128 | **142.548** | **138.322** |

这些是 decoder 图执行时间。端到端生成还包括 embedding、采样和 runner/cache 管理等开销。profile=0 的冷 cache forward 也不等于长序列持续 decode。

basic profile 的 accelerator execute：

| 方法 | 0.8B ms | 2B ms |
|---|---:|---:|
| decode | 21.620 | 32.308 |
| prefill128 | 140.495 | 134.670 |

basic profile 是另一次执行，只用来核对耗时主要落在 accelerator；不能把两张表相减当成精确 host 成本。两张 decoder 图都检查了只有一个 QNN delegate，没有 CPU 算子回退。

## 4. prefill 热点

以下百分比来自 **profile=2 的逐算子 instrumented cycles**，分母为所有 `OpId` 事件的 cycles 之和。卷积按导出图的模块来源分组：169 个 convolution，其中包含投影、FFN、causal conv 和 LM head。

| 算子类别 | 0.8B M cycles | 占比 | 2B M cycles | 占比 |
|---|---:|---:|---:|---:|
| 18 个 GDN recurrence | **451.19** | **50.07%** | **450.96** | **45.93%** |
| norm / gating / layout / 其他 | 250.44 | 27.79% | 277.56 | 28.27% |
| FFN 三个投影 | 48.36 | 5.37% | 113.37 | 11.55% |
| LM head，仅选择的1个token | 72.65 | 8.06% | 49.90 | 5.08% |
| full-attention MatMul | 44.91 | 4.98% | 43.11 | 4.39% |
| attention 输出 / FA 输入投影 | 16.10 | 1.79% | 22.73 | 2.31% |
| GDN 两个输入投影 | 11.80 | 1.31% | 17.77 | 1.81% |
| GDN causal conv | 5.67 | 0.63% | 6.56 | 0.67% |

这里能直接观察到：

1. GDN recurrence 的 cycles 几乎相同，差约 0.05%；单层平均约 25.1M instrumented cycles。
2. 2B 的 FFN cycles 是 0.8B 的约 2.34 倍，确实体现了模型变宽。
3. full-attention MatMul 的形状相同，cycles 也接近。
4. GDN 输入投影只占约 1–2% 的 instrumented cycles，单纯减少这一组投影参数，无法解释或消除当前整图热点。
5. 同样词表下，2B 的 LM head 在本次追踪中反而更低。不同矩阵形状、QNN 执行选择和设备波动都会影响该值；尚未通过控制实验确定原因，不能仅按权重大小推算耗时。

**限制：profile=2 会显著改变执行时间。** 0.8B / 2B prefill 的逐算子追踪分别约 877 / 960 ms，正常 forward 约 143 / 138 ms。因此不能把上述百分比乘正常耗时，声称 GDN 在正常执行中准确占 50% 或某个毫秒数。这些数据用于定位、比较同一类算子的热点；要测优化收益，必须再用 profile=0 和端到端测试。

`norm/gating/layout/其他` 也需要继续拆分。0.8B 的大项包括 mul 46.91M、view_copy 44.32M、add 30.31M、slice_copy 30.07M、permute_copy 28.39M cycles；它们包含算术、量化转换、布局相关的追踪事件，不能全都归为 DDR 流量。

详细事件保存在 `profiles/{模型}_{方法}/events2.json`；`analyze.py` 负责分组，分组规则和完整操作统计可查。

## 5. decode 热点随参数量增大

同样为 instrumented cycles：

| 算子类别 | 0.8B M cycles | 2B M cycles |
|---|---:|---:|
| FFN 投影 | 29.81 | 80.87 |
| LM head | 32.21 | 49.28 |
| GDN recurrence | 22.67 | 23.94 |
| GDN 输入投影 | 9.61 | 16.14 |

decode 每次只有一个 token，更受大投影的影响；2B 的无 profile forward 增加约 51.2%，与端到端 decode 变慢的方向一致。当前 0.8B 的短 decode 约 37.4 token/s，已经恢复到此前约 35 token/s 的水平。

## 6. 本次导出设置

- 2B 使用 `et_npu` 环境和 CUDA 校准；校准集是与 0.8B 相同的 8 段 Wiki 文本，各取 128 token，共 1024 calibration tokens。
- K/z 合并为 W8；q/v/a/b 合并为 W4；其他 Linear W4 group32；causal conv 的原有 W8 设置保留。
- 对 18 层 GDN out_proj 使用固定 QNN 有效网格的 GPTQ；未把整模型标成 GPTQ。
- activation、recurrent state、conv state、KV cache、logits 接口为 UINT16；hidden 接口为 FLOAT32，positions 为 INT32。
- Q/K 使用 native RMSNorm 路径，attention output 和 state 各自使用校准编码。
- GQA 不复制 KV heads；新 token 写入 FA cache 前转换一次，历史 cache 直接使用已对齐的校准网格。
- prefill 在 LM head 前选择所需 token 行；避免对全部 128 行执行 LM head。
- decoder PTE 和 runner 均开启 shared buffer。
- 两模型使用同一个精确 split-reduction VTCM DSP SO，SHA256：
  `490d117374851b0e5b6fc2d3b69f264b41b703c5bd9c79ad4326b8c35d435d9e`。
- 全程串行导出，未同时跑多个大模型导出任务。

## 7. 文本推理结果

2B 的两次短生成输出一致，开头为：

> **Python** is a high-level, interpreted, general-purpose programming language known for its readability, dynamic typing, and extensive standard library.

输出可读，未出现此前乱码现象。本次是性能对照，没有对 2B 做完整 PPL 评估，因此不能仅凭该样例证明量化精度已全面达标。

测试为空图片列表。为兼容当前 multimodal runner 的加载接口，设备上放置了已有 0.8B vision encoder；**它没有被执行，也未参与 2B embedding 或 decoder**。本次产物用于 2B 文本推理，未验证 2B 图片输入。

## 8. 官方数字的对应口径

官方 Qwen3.5-2B、context4096、NPU、GenieX llama.cpp / Q4_0 数据：

| 官方设备 | prefill token/s | decode token/s |
|---|---:|---:|
| Snapdragon X2 Elite CRD | 1496.44 | 18.62 |
| Samsung Galaxy S25 / Dragonwing Q-8750 | 593.10 | 22.64 |

来源：[Qualcomm 官方 perf.yaml](https://raw.githubusercontent.com/qualcomm/ai-hub-models/main/src/qai_hub_models/models/qwen3_5_2b/perf.yaml)，2026-10-14 核对。

本地 2B 在 SM8750 上约 866 prefill / 24.3 decode，数值超过上述手机条目，但量化方案、后端和实际 prompt/生成计时范围不同，不能据此认定同口径领先官方。1496 的设备条目是 X2 Elite CRD。

本地 0.8B 和 2B 的同机对照，已经直接暴露出 prefill 的共同热点。是否能把 0.8B 做到 2000 token/s，仍需要以实际优化后的整图结果验证。

## 9. 下一步优化顺序

1. 首先针对实际模型量化参数下的精确 GDN recurrence 做优化，保留当前数值精度，用同一真实输入对照 profile=0。两模型共同使用 18 个相同大小的 state，优先处理这里才能让两者一起受益。
2. 拆分 GDN 周围的 norm/gate、切分和布局转换，识别可消除的物化与重复转换；结合实际 buffer 和 VTCM/DDR 计数器判断访存，不能只看 `view_copy` 名称。
3. 分别核对 LM head 的 QNN 执行选择和 decode FFN 热点，避免把 prefill、decode 的瓶颈混在一起。
4. 所有候选优化都在完整 PTE 上复测本报告的三个点；保留生成样例，并做与现有量化基线的数值回归。2000 token/s 以整块 64 ms 为验证目标。

这些是根据本次追踪确定的后续方向；本次没有新修改 DSP 内核，也没有声称已完成这一轮提速。

## 10. 产物和复现

2B 产物：

- `2B/hybrid_llama_qnn.pte`：1,114,504,704 bytes，包含24层、prefill/decode、FFN、final norm、LM head。
- `2B/tok_embedding_qnn.pte`：1,017,195,008 bytes。
- `2B/libraries/`：本次实际部署的两个 op-package SO。
- `2B/metadata.json`：模型配置、48个 cache 编码、hidden 编码、workspace。
- `2B/hybrid_llama_qnn.io.json`：输入输出类型。
- `2B/run_text.sh`：运行已部署设备目录的短文本生成。
- 设备目录 `/data/local/tmp/executorch/qwen35_size_2B`，PTE 的 op-package 路径也指向该目录。迁移设备目录时须同时处理编译时路径。

decoder SHA256：`0d4270212d609b1172d95d9970708c35f60980d5eb919366ba77ab73df37b0bc`。

embedding SHA256：`a3513ce219e3d582330ade80307b9020a566c52458d4f81242a24470030a63d3`。

环境初始化后重建、部署和对照：

```bash
conda activate et_npu
source env.sh
export CUDA_VISIBLE_DEVICES=0
python llama-output/qwen3_5-size-comparison/export_model.py --size 2B --capacity 4096
python llama-output/qwen3_5-size-comparison/benchmark.py --phase deploy
python llama-output/qwen3_5-size-comparison/benchmark.py --phase generation
python llama-output/qwen3_5-size-comparison/benchmark.py --phase profile --size all
python llama-output/qwen3_5-size-comparison/analyze.py
```

generation 会跳过已有完整记录；要新跑一轮，应先重命名 `generation/summary.json`，保留原始证据。

当前脚本复用 `llama-output/qwen3_5-0_8b-layer-analysis` 下既有导出、校准、GPTQ 和编译 helper；需要保留这些依赖。本次没有修改 `mybuild.sh`，普通旧 `llama.py` 命令也不会自动得到这里的后校准 GQA/所选行优化。

<!-- QWEN35_CONTEXT_MATRIX_20261016_BEGIN -->
## 2026-10-17：0.8B／2B，1K–8K，INT4 embedding／KV8性能矩阵

SM8750／sun、PKR110、设备431e3c37，QNN SDK 2.49.40.260810。纯文本，正常生产QNNMultimodalRunner，temperature=0，无profile。

两模型统一INT4/HQQ/group32 embedding、full-attention K/V UINT8（MatMul RHS UINT8），GDN conv/recurrent state UINT16。K/z投影与depthwise conv W8，其余Linear W4/group32，保留既有out_proj GPTQ。Prefill AR128，FP16 chunk64/leaf8；decode使用DeltaNetCoreVtcm。

每模型五套独立hybrid PTE。逻辑长度n=1024/2048/3072/4096/8192，物理容量n+128，便于完整输入n并输出一个token。每个decoder的PTE及QNN context都只有kv_forward、prefill_forward，没有256等短上下文图。

2B/8K的QNN原生MatMul配置另行核对：每图12个FA MatMul，prefill的RHS长度均为8192，decode均为8320。

Prefill输入n、输出1；decode输入16、输出n−16。首个输出来自prefill，decode测速分子为实际n−17次AR1。每项三轮取中位数，共60次测量；同一份8192-token WikiText-2 raw test输入，各进程预热16→32，固定长度忽略EOS。计时含embedding/cache，排除模型加载、tokenization和图片编码；vision PTE加载但未执行。

三轮均确认测前thermal=0，部分长decode结束为1/2；未锁频，因此这些是手机持续生成实测值。1K–4K最长等待300秒，8K最长900秒，冷却不计入推理时间。


| 上下文 token | 0.8B Prefill token/s | 0.8B Decode token/s | 2B Prefill token/s | 2B Decode token/s |
|---:|---:|---:|---:|---:|
| 1024 | 1726.81 | 67.35 | 1196.26 | 35.25 |
| 2048 | 1605.02 | 63.36 | 1130.87 | 34.98 |
| 3072 | 1455.23 | 64.74 | 1042.77 | 36.23 |
| 4096 | 1261.86 | 63.51 | 926.70 | 35.05 |
| 8192 | 1293.75 | 39.97 | 773.63 | 27.43 |

0.8B/8K的decode明显低于1K–4K；本轮没有进行算子profile，以上数据只确定端到端性能变化。PPL/SQNR未重测，2B的新INT4/KV8配置不能仅凭性能测试认定精度达标。

校准统一CUDA、seed42、两条图文样本+8条WikiText-2 train序列，每条文本最大1024token。导出和设备测试全串行，单次导出MemoryMax=36GiB、MemorySwapMax=0；导出进程裁剪FX NodeSource的递归调试历史。矩阵测量时2B在测试进程覆盖配方；完成测量后，INT4 embedding／原生KV8和FX内存优化已接入普通`llama.py`入口，见下节。

产物根目录：`llama-output/qwen35-context-performance-20261016/`。模型目录分别为`0_8b/m1024`等和`2b/m1024`等；各目录有decoder、embedding、vision三个PTE和tokenizer。0.8B/4K复用此前验证的无短图PTE，测量是本轮重新执行；另外9套新导出。

`REPORT.md`记录三轮范围、PTE大小与导出峰值；`RESULTS.csv`记录10项中位数，`RAW_MEASUREMENTS.csv`保留60次原始测量，`verification.json`记录IO shape/dtype、两图结构、token计数和全部冷启动状态核对。`performance.png`和`performance.svg`是性能曲线。
<!-- QWEN35_CONTEXT_MATRIX_20261016_END -->

<a id="qwen35-cli-integration-20261017"></a>
## 2026-10-17：接入普通llama.py、review和清理

### 接入内容

0.8B和2B现在直接使用主README中的普通`python examples/qualcomm/oss_scripts/llama/llama.py ...`命令。不再通过测试脚本替换模型注册项或临时修改2B配方；本地权重、tokenizer仍由`--source_safetensors`指定。

| 项目 | 正式配置 |
|---|---|
| Qwen3.5 0.8B／2B embedding | `aten.embedding.default`节点规则，INT4/HQQ/group32；PTE保持packed，token输入INT64 |
| Full-attention K/V | 原生`annotate_kv_8bit`；存储和MatMul RHS都是UINT8，Query/概率/输出仍UINT16 |
| GDN conv/recurrent state | UINT16，保持校准后的独立scale/zero point |
| 投影/FFN | K/z及depthwise conv W8，其余Linear W4/group32；保留out_proj GPTQ |
| Prefill | AR128，FP16 chunk64/leaf8，完整上下文图，保留有效token状态更新和末行LM-head选择 |
| Decode | 快速`DeltaNetCoreVtcm` |
| Hybrid decoder图数 | `kv_forward`和`prefill_forward`两张；无256等额外短图 |
| 校准 | 原AR128＋warm AR1共享激活网格，图文＋文本校准；支持CUDA |

将矩阵脚本中的FX内存优化提取为`export_memory.py::compact_fx_provenance()`，由`llama.py::export_llama`仅在Qwen3.5-family编译期间调用。新生成的`NodeSource`保留直接来源，不再递归deepcopy祖先调试历史；节点的ModuleStack、张量、量化参数和计算图语义不变。上下文使用锁和线程局部上下文标记，退出、异常、KeyboardInterrupt时恢复原函数。`--pre_gen_pte`路径不进入该上下文；其他模型族编译仍使用原始来源记录。

矩阵导出时2B最大进程RSS为34.73GiB。这是此前完整导出的实测值，本轮没有重导十套完整模型，也没有重新测量接入后的完整导出峰值。

### Review结论

发现并修正了配置继承隐患：Qwen4B和Lincal2B同样继承原2B基类，如果直接修改共享基类配方，会顺带改变这两个尚未完成本轮消融的模型。现在公共`Qwen3_5QuantRecipe`保留原FP16 embedding／KV16配置，只有注册的Qwen2B显式选择优化配方。Qwen/Lincal0.8B沿用已有INT4/KV8配置。

| 检查范围 | 结论 |
|---|---|
| 0.8B／2B默认配置 | 均命中embedding INT4节点规则和原生KV8；没有UINT8历史cache扩展到UINT16计算网格 |
| Qwen4B／Lincal2B | 保留FP16 embedding／KV16，其他权重量化规则保持原配置 |
| MHA/SHA、SWA及mask | 本轮没有改变选择条件、窗口长度、mask更新或容量裁剪规则；现有SWA/mask回归通过 |
| 量化与cache | AR128/AR1、分组attention前后cache编码一致；GDN state未被FA KV8规则改为8bit |
| FX来源历史 | 正常退出、嵌套、异常、中断恢复和其他线程隔离验证通过；普通模型的递归来源记录保留 |
| 构建配置 | 新helper已加入`targets.bzl`的`llama_lib`源文件列表；本轮没有新增C++接口或改变runner ABI |

上述范围复查后未发现新的阻断问题。2B INT4/KV8的完整PPL/SQNR仍未补测；短生成验证和此前性能矩阵不能代替完整精度评估。

### 本轮验证

- Python/量化/embedding/cache/内存/SWA/mask等共42个不同用例通过，16项可选测试因环境条件跳过。
- 沙箱内C++ KV/SWA测试首次受LeakSanitizer线程检查限制；在主机上保持AddressSanitizer／UndefinedBehaviorSanitizer配置原样重跑，1项通过。以上42项包含此重跑通过的用例。
- 在正式的压缩FX来源上下文内额外复测AR128/AR1校准、分组attention和prefill容量裁剪：1项通过，覆盖0.8B/2B KV8与旧KV16三种配方，GDN state保持16bit。临时测试补上了与生产流程相同的`remove_export_checks`预处理。
- 新helper的5项恢复/线程隔离/数值测试复跑通过；小模型128-token导出及decomposition输出与原来源记录模式逐元素一致。
- 普通`llama.py --help`通过。普通`llama.py --pre_gen_pte`在SM8750／PKR110／431e3c37上加载现有完整2B/4K INT4/KV8 PTE，通过`prefill_forward`处理短中文问题，正常输出`2`并EOS结束，无实验注册覆盖。
- `git diff --check`与本轮修改Python文件的Black格式检查通过。

验证日志在`llama-output/qwen35-context-performance-20261016/integration-review/`，手机输出在同根目录`integration-cli-2b/outputs/outputs.txt`。这次手机运行是功能验证，没有作为新的性能或PPL成绩。

### 清理范围

删除88个Qwen相关临时导出、benchmark、审计、画图脚本及测试driver源码/可执行文件，4个本轮开发的临时回归测试文件，并清理37个对应pyc。保留原有受版本控制的SWA/mask/C++回归测试和正式自定义op源码。

删除清单及各文件SHA256保存在`llama-output/qwen35-context-performance-20261016/test_code_cleanup.json`。清理前后核对77个PTE路径及大小完全一致；所有PTE、输入/输出、日志、报告、CSV、配置与测量结果均保留。后续导出使用[主README的正式命令](README.md#standard-qwen35-export-entry-point-2026-10-17)。上述验证完成时尚未commit或push，后续代码提交由`preview`分支记录跟踪。

> 2026-10-17 收尾说明：以下GDN state8、归一化byte cache、StateCacheUnpack/Pack和整数MatMul内容均为历史实验记录。按用户要求，相关源码、标注入口、op注册和临时测试代码已撤销，不能照下方历史片段启用。当前仅full-attention K/V采用KV8；GDN recurrent/conv state保持UINT16，chunk MatMul保持FP16，INT4 embedding及通用prefill优化保留。详见文末清理记录。

<a id="qwen35-gdn-state8-20261017"></a>
## Qwen3.5-0.8B：GDN recurrent state UINT8 实验（2026-10-17）

结论：**量化标注、混合 dtype、原生 HVX/VTCM、PTE和设备推理均已打通，但当前 per-tensor UINT8 state 的精度损失很大，默认配置继续保留 UINT16。**

### 范围

只修改 recurrent state；conv state、Q/K/V、decay/beta和core输出保持UINT16，FA KV8和embedding INT4的生产默认配置不变。实验输入来自本地Qwen3.5-0.8B第0层的真实权重，形状为state [1,16,128,128]。对照采用相同FP32 embedding、input RMSNorm、输入投影、causal conv、Q/K L2和gate，以隔离state位宽误差。这里**没有重新量化整个0.8B模型，也没有把W4投影/INT4 embedding误差混入state消融**；不代表完整模型PPL或端到端性能。

校准使用已缓存WikiText-2 raw train的连续1024个token，8个128-token窗口，真实传递非零conv/recurrent state；测试使用此前性能矩阵相同WikiText-2 raw test的前1024个token。两个版本的Q/K/V、decay/beta和core输出的校准scale/zero point完全相同，只有state网格不同。

### 实现

新增`annotate_gdn_state_8bit`，只将core第5个输入和第1个输出改成8bit，并通过SharedQuantizationSpec共享校准网格。默认core标注先执行，recipe的override随后执行；scale/zero point来自MinMax校准，非写死。正常导出报告按实际calibrated cache encoding记录conv/recurrent/FA各自dtype。仅实验时，在所选recipe实例的自定义标注列表追加：

```python
from executorch.examples.qualcomm.custom_op.deltanet import annotate_gdn_state_8bit

recipe.recipe.custom_quant_annotations.append(annotate_gdn_state_8bit)
```

当前默认recipe不追加此标注。

`DeltaNetState8.cpp::deltanetCoreQhpiState8`为独立的混合精度实现，注册到现有DeltaNetCore和DeltaNetCoreVtcm：五个U16输入＋一个U8 state，输出U16 attention＋U8 state。state在DDR和VTCM中均为byte code；只在HVX寄存器中展开，使用INT32 MAC及FP32缩放/更新。不存在完整state扩展成UINT16量化网格的中间张量。

U16 Q/K先在整数MAC中以32768为中心，再通过sum(S)精确补回实际calibrated zero point；32768是算术重编码常数，不是硬编码state zero point。归约按最多128个K分块避免INT32溢出，支持与原内核相同的K/V<=256及尾部。旧U16内核和注册顺序保留。

### 按指定顺序核对误差

| 指标（非零 state 单步 decode，手机实测对齐 FP32） | UINT16 state | UINT8 state |
|---|---:|---:|
| FP32 min/max（state 输入） | [-2.38344, 2.17460] | 相同 |
| FP32 min/max（state 输出） | [-2.38645, 2.17800] | 相同 |
| 量化范围 | [-3.57608, 2.69304] | [-3.56479, 2.70432] |
| 饱和比例（原 FP32 state 超出量化范围） | 0% | 0% |
| state 输入 SQNR | 60.34 dB | 17.98 dB |
| state 输出 SQNR | 58.20 dB | 18.00 dB |
| attention output SQNR（core＋RMSNorm/z＋FP32 out_proj） | 37.38 dB | 8.81 dB |

该单步输入使用FP32预热128token后的非零state；它单独检查state保存/加载的损失，没有混入之前128步的UINT8递推误差。attention output包含core、RMSNorm/z门控和FP32 out_proj；后两部分在主机用设备core输出计算，不是整个attention在手机上的整图误差。

UINT16校准scale为0.0000956605727、zero point=37383；UINT8 scale为0.02458476834、zero point=145。步长扩大约257倍。首128token FP32 state的绝对值中位数仅0.00098467，P90为0.00860886；**93.0542%**的元素低于UINT8半步长，会落入零码，而原FP32的精确零元素只有0.3418%。因此主要问题是小信号和小更新被舍入掉，扩大校准范围无法解决。饱和接近零不代表量化精度好。

### Prefill与decode区别

| 128-token 冷启动 prefill，手机 FP16 ChunkGDN | UINT16 state | UINT8 state |
|---|---:|---:|
| state 输出 SQNR | 56.05 dB | 17.97 dB |
| core output SQNR | 61.33 dB | 61.33 dB |
| attention output SQNR | 37.63 dB | 37.63 dB |
| state 大小/层 | 512 KiB | 256 KiB |

冷启动state为0，因此第一块prefill的attention输出可以完全相同，损失出现在写回的state。正式prefill仍使用FP16 chunk64/leaf8，块内部不逐token量化state；每128token块之间才保存为UINT8。连续8块的主机参考中，最后一块attention output SQNR从37.50 dB降到14.42 dB；不能只根据第一块输出判定state8无误差。

| 连续 recurrence | UINT16 state SQNR | UINT8 state SQNR | UINT16 attention output SQNR | UINT8 attention output SQNR |
|---|---:|---:|---:|---:|
| 128步，手机 PTE 内真实逐步回写 | 42.36 dB | 9.17 dB | 33.11 dB | 6.80 dB |
| 1024步，PyTorch 逐步量化参考 | 33.99 dB | 6.33 dB | 30.27 dB | 5.57 dB |

128步原生recurrence PTE用于闭环验证逐token回写，不是正式prefill实现。UINT8设备core输出相对逐步量化参考SQNR为75.13 dB，state为55.39 dB；state code一致率99.9996%。单步非零state的UINT8输出code与量化参考100%一致，进一步说明大的FP32误差来自state8本身。

### 性能

SM8750/PKR110/431e3c37，HTP v79，QNN SDK 2.49.40.260810；正常qnn_executor_runner、shared buffer、20次warmup，每轮AR1执行1000次，AR128执行100次，交替运行两精度，三轮取中位数。没有锁频或profile；数据用于功能及微基准，不是整模型吞吐结论。

| 单算子 PTE execute，3轮中位数 | UINT16 state | UINT8 state | 延迟变化 |
|---|---:|---:|---:|
| 正式 FP16 ChunkGDN prefill，AR128 | 1.19301 ms | 1.11605 ms | −6.45% |
| 原生 VTCM core decode，AR1 | 0.248197 ms | 0.262874 ms | +5.91% |
| 原生逐步 recurrence，128步 | 2.17043 ms | 11.69697 ms | +438.92% |

这些是单算子PTE的execute延迟，含独立delegate调用开销，排除加载/ADB，不含输入投影、norm/z、out_proj和FFN；不能直接乘18换算整个模型耗时。原生state8 recurrence是功能原型，尚未优化到U16内核水平；正式prefill使用FP16 ChunkGDN，不能把上表11.70ms当作prefill时间。

每层state存储减半。0.8B共有18个GDN层，仅recurrent state由9 MiB降到4.5 MiB；conv与FA KV大小不变。

### 验证与限制

x86、Android、Hexagon v79构建通过。HVX byte展开、lane顺序及饱和打包验证通过；35组不同K/V、尾部及K>128的归约/更新检查通过。两层GDN＋FA小图确认AR128/AR1的cache位宽分别为Conv16/FA8/Recurrent8/FA8，输入输出编码共享；默认16位配方仍为Conv16/FA8/Recurrent16/FA8。

共6个微型PTE：原生core的AR1/128各两精度，正式FP16 ChunkGDN的AR128两精度。均只有一个QNN delegate且无CPU算子；state8图的state输入/输出为BYTE，其他IO为UINT16。Chunk路径的state Convert为UINT8→FP16及FP16→UINT8，使用真实浮点计算；不是已放弃的UINT8→UINT16计算网格方案。

C++ runner已按每个输入/输出的实际nbytes和dtype初始化及回写，无需修改运行时代码；补充UINT16 conv＋UINT8 recurrent＋UINT8 FA的native KVManager regression，并保留sanitizer检查。当前大误差下不切换默认配方，不继续做完整模型PPL/全量导出。若后续坚持state8，应先研究按head/block缩小量化范围，并相应适配kernel的量化参数布局。

### 文件

`accuracy.json`为逐步量化参考；`prefill-accuracy.json`为真实FP16 chunk参考；`device-accuracy-vs-fp32.json`为手机输出相对FP32；`device-results.json`及`device-prefill-results.json`记录各轮性能和实现一致性。`qnn-audit-*.json`记录混合IO dtype，PTE和原始输出保留。临时实验脚本置于/tmp，未加入代码库。当前新增实现未commit/push。

<a id="qwen35-gdn-state8-optimization-20261017"></a>

> 性能比较更正：以下“普通UINT8原库／新库”实验没有同步修改PTE内部绝对库路径，不能用于比较库代码优化。精度/PPL消融仍有效；正确绑定路径后的内核性能及修复见[后续记录](#qwen35-state8-performance-20261017)。

## GDN UINT8 state 通道归一化优化（2026-10-17）

## 结论

原生 UINT8 state 已能运行；本轮解决的是全张量共用量化步长引起的小值丢失。选定按 head/value 通道归一化缓存，复用现有 DeltaNetCoreVtcm，不增加新的 op 类型。

实际手机连续 1024 token：prefill attention 累计 SQNR **37.17 dB**，逐 token recurrence **25.91 dB**。
生产默认仍是 UINT16 recurrent state；归一化 UINT8 作为经过小 PTE 和 GDN core PPL 消融验证的可选原型保留。完整生产 W4／INT4 embedding／FA KV8 PTE 的 PPL、图文输入和端到端性能本轮未测。

## 实现与数学关系

`examples/qualcomm/custom_op/deltanet.py::ValueScaledDeltaNetCore`：`from_calibration` 观察训练样本每个 token 的 FP32 state 更新，计算 `c[h,v] = max_train,b,k,t |S[b,h,k,v]|`。缩放系数与量化网格只从训练集拟合。首个测试窗口用于比较方案，额外三个1K窗口在选定方案后验证，未重新拟合参数。

```text
R[k,v] = S[k,v] / c[v]
delta_R = beta * (V/c - decay * K^T R_old)
R_new = decay * R_old + K * delta_R
Y = (Q^T R_new) * c
```

每个 head 独立应用上述关系。c 与 token/key 坐标无关，因此是可逆的状态表示变换；在非零 warm state 上 FP32 输出／state 与原公式 SQNR 为 136.97／126.94 dB。

Q/K/decay/beta 的计算方式不变。core 内部使用归一化 V 和 R，输出再按通道恢复 Y。PT2E 自动校准归一化 V、R 和 Y 的量化网格；R 的输入输出共享同一组 UINT8 scale/zero point。缩放乘除节点也必须使用 MinMaxObserver；实验最初误用默认 MovingAverageMinMaxObserver 时范围收窄，已修正。

cache 保存的是 R。warm-start 的物理 S 必须先除以 c；之后直接传递返回的 R。prefill/decode 必须使用同一份 c 和共享 cache 编码，不能与旧的 S-cache 图直接交接。零 state 初始化保持为量化 zero point。

每层 Python FP32 缩放 buffer 8 KiB（16×128×4），state 仍为 256 KiB；18 层 recurrent state 由 9 MiB 降到 4.5 MiB，不含 conv/FA cache。没有创建 UINT16 state 计算网格；原生路径只在寄存器中展开 byte，prefill 延续原 FP16 ChunkGDN。

`src/ops/DeltaNetState8.cpp::outputState8` 将 delta 与 U16 输出的中心化、乘加、舍入和打包向量化。35 组 K/V shape 含尾部、K>128 的 libnative 对照通过。AR1 的微小速度差与测量波动重叠，128 步原生循环没有可靠提速；主要精度改善来自通道归一化。UINT16 内核和旧 kernel 索引未改变。

## 单层精度：按指定顺序检查

固定本地 0.8B 第0层真实 FP32 embedding／投影／causal conv／L2／gate 输入，独立 train1024 校准、test1024 验证。不能将这组结果解释为 W4 投影精度。

### 1. FP32 min/max

| 位置 | 原始物理 S 最小值 | 最大值 | 归一化 R 最小值 | 最大值 |
|---|---:|---:|---:|---:|
| 128 | -2.383440 | 2.174600 | -1.110773 | 1.126003 |
| 1024 | -3.059072 | 2.831918 | -1.720514 | 1.794810 |

### 2. 量化范围

普通 UINT8 的统一物理步长为 0.02458477。新 R 的 step=0.0078431414、zero=128、code=[0,255]，R 可表示范围 [-1.003922,0.996079]。
训练得到 c 范围 [0.00365009,3.65685916]，因此物理 S 每通道有效步长为 [0.0000286282,0.0286812633]。小幅值通道不再被最大通道决定步长。

### 3. 饱和比例

以对应 FP32 物理 state 检查可表示范围：第128 token 超范围 **0.002670%**，第1024 token **0.085831%**。这是 FP32 超范围比例，不把合法零值或实际 endpoint code 一概算成饱和。
手机冷 AR128 prefill 输出中，zero-point code 占比由普通 UINT8 的 93.04% 降到 15.81%；该指标包含真实小值量化成零，不等于饱和率。

### 4–6. state 输入 SQNR、state 输出 SQNR、attention output SQNR

下表为手机结果。attention 是 core 输出继续通过同一 FP32 RMSNorm/z/out_proj 后与 FP32 对照比较；不含残差/FFN。非零 AR1 输入为精确 FP32 前128 token state 一次量化，不是经历128次 byte舍入的 state。

| 路径／配置 | state 输入 dB | state 输出 dB | attention 输出 dB |
|---|---:|---:|---:|
| ar1／state16 | 60.34 | 58.20 | 37.38 |
| ar1／state8-new | 17.98 | 18.00 | 8.81 |
| ar1／scaled8 | 37.84 | 36.66 | 35.88 |
| prefill／state16 | — | 56.05 | 37.63 |
| prefill／state8-new | — | 17.97 | 37.63 |
| prefill／scaled8 | — | 37.79 | 37.58 |
| recurrence／state8-new | — | 9.17 | 6.80 |
| recurrence／scaled8 | — | 18.77 | 28.83 |

原生 recurrence128 是逐 token 舍入的诊断图，正式 prefill 使用 FP16 chunk64/leaf8，两者不能混用名称。

连续8次128-token手机调用，直接将原始 UINT8 state 输出反馈到下次输入，host 不做反量化/再量化：

| 路径 | 最后一块 state 输入 dB | 最后 state 输出 dB | 最后一块 attention dB | 全1024 attention累计 dB |
|---|---:|---:|---:|---:|
| prefill | 33.82 | 33.37 | 36.82 | 37.17 |
| recurrence | 13.48 | 13.03 | 24.82 | 25.91 |

## 粒度消融

按 head 单独设置 affine grid 提升有限；沿 value 维分组更有效。以下为量化参考实现的全1024 attention累计 SQNR：

| state方案 | 每token舍入 decode dB | 每128token舍入 prefill dB |
|---|---:|---:|
| global | 5.57 | 15.81 |
| head-k128-v128 | 6.77 | 18.88 |
| head-k128-v8 | 18.17 | 29.99 |
| head-k16-v16 | 16.25 | 29.83 |
| head-k4-v4 | 23.74 | 35.30 |
| global16 | 30.27 | 37.75 |
| 选定 value 通道归一化，实际 PT2E QDQ图 | 25.72 | 37.27 |

未选用直接的4×4 affine block方案：还需要改变 QHPI 量化参数表示和归约实现，且 held-out clipping 升到约1.96%。选定方案复用原6输入／2输出 core，新增的只是 V/c 与 Y×c 两个图节点。

## PPL 与最终 hidden SQNR

**这是全0.8B文本模型上的 GDN core 消融，不是生产 W4 PTE 评估。** embedding、所有 Linear/FFN、FA、conv 路径保持 FP32；仅 GDN core 的Q/K/V/decay/beta/Y为校准 A16，state为对应位宽。归一化方案另校准 V/c 与 Y/c。

prefill 模拟每128token持久化一次 state，内部FP16 ChunkGDN；decode 模拟每token量化一次 state。模型投影/FA采用整段因果计算以降低测试开销，因此这里不报告GPU执行时间作为decode性能。

WikiText-2 raw train前1024用于校准；test前4096切成四个独立冷启动1024窗口，各1023个teacher-forced目标，共4092。下表先汇总NLL再取exp，不平均PPL。不是标准全测试集，也不是连续4K cache实验。

| 方案 | 前1024 PPL | 四窗口合并 PPL | 四窗口最终 hidden SQNR dB |
|---|---:|---:|---:|
| fp32-decode | 10.8785 | 19.0496 | — |
| state16-decode | 10.8610 | 19.1786 | 23.23 |
| state8-decode | 85.7426 | 166.8354 | 1.85 |
| scaled8-decode | 11.1978 | 19.9046 | 16.14 |
| state16-prefill | 10.9081 | 19.1048 | 36.81 |
| state8-prefill | 11.7138 | 20.4664 | 11.81 |
| scaled8-prefill | 10.9572 | 19.1559 | 26.39 |

每个窗口的 PPL：

| 起始token | FP32 | UINT16 decode | UINT8 decode | 归一化 UINT8 decode | UINT16 prefill | UINT8 prefill | 归一化 UINT8 prefill |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 10.8785 | 10.8610 | 85.7426 | 11.1978 | 10.9081 | 11.7138 | 10.9572 |
| 1024 | 24.9312 | 25.1244 | 383.9310 | 25.4320 | 24.9989 | 26.5915 | 24.8758 |
| 2048 | 20.3502 | 20.4309 | 173.7796 | 21.7753 | 20.4383 | 22.2210 | 20.5301 |
| 3072 | 23.8597 | 24.2670 | 135.4268 | 25.3126 | 23.9029 | 25.3491 | 24.0625 |

四窗口合并后，归一化 UINT8 相比 UINT16：decode PPL **+3.79%**，prefill **+0.27%**。decode仍有逐步量化累计损失；prefill更接近基准。

Transformers FP32同一首1024输入 PPL=10.87851560，本地 FP32 PPL相同；最终 hidden SQNR=115.98dB，逐位置 NLL最大差=3.4213066e-05。新PPL基准已与HF核对，不与历史不同语料、完整量化PTE的PPL直接比较。GPU峰值约5303MiB；全程串行单模型。

## 手机耗时

SM8750／431e3c37；独立小PTE的execute，AR1 warmup20＋1000次，AR128 warmup20＋100次；三轮交替顺序取中位数。排除加载/ADB，包含单delegate调用；未包含投影/norm/z/out_proj/FFN。

| 路径 | UINT16 | 普通UINT8，原库 | 普通UINT8，新库 | 归一化UINT8 |
|---|---:|---:|---:|---:|
| ar1 | 0.245624 ms | 0.269605 ms | 0.260276 ms | 0.231372 ms |
| prefill | 1.180240 ms | — | 1.126620 ms | 1.153190 ms |

各轮原始耗时见 `device-summary.json`。归一化的额外缩放没有造成明显回退，但这些小幅收益不能推算成整模型token/s。128步原生循环只有一轮诊断数，不作为优化速度结论；正式prefill只看chunk路径。

## 验证与产物

三份归一化PTE：AR1、AR128逐token诊断、AR128正式chunk。均一个QNN delegate、零CPU算子，外部IO为5×UINT16＋UINT8 state，输出UINT16 attention＋UINT8 state。AR1 QNN图为ElementWiseDivide→DeltaNetCoreVtcm→ElementWiseMultiply，core签名保持5×UFIX16＋UFIX8。

FP32非零state等价性、batch2/K3/V7、拆段state交接、无效scale及空校准拒绝通过；native35组HVX归约/更新/输出对照通过；完整UINT16路径未改。详见 `verification.json`。

产物目录：`llama-output/qwen35-gdn-state8-optimization-20261017/`。主要文件：`group-sweep.json`、`equalize.json`、`scaled-accuracy.json`、`scaled-state-statistics.json`、`device-results.json`、`device-sequence.json`、`ppl.json`、`ppl-extra.json`、`ppl-aggregate.json`、`ppl-hf-reference.json`、`verification.json`。旧对照产物保留在 `llama-output/qwen35-gdn-state8-20261017/`；两版DSP库SHA保存于 `library-sha256.json`。

未提交或推送本轮修改。下一步若接入正式模型，应在生产W4/INT4 embedding/FA KV8的校准路径中学习每层c，再同时派生共享R-cache的prefill/decode；当前原型结果不能替代该完整PTE精度验证。

<a id="qwen35-state8-performance-20261017"></a>

## UINT8 GDN 性能定位与修复（2026-10-17）

## 结论

小幅提升包含两个不同问题：实验换库没有替换 PTE 真正引用的路径；UINT8 native kernel 存在临时向量落栈、逐K行重复准备系数及加载delta的开销。已修正测试路径并优化内核。正式AR128 prefill依旧是FP16 chunk，缩减state持久化位宽不能使其矩阵计算同步减半。

同一台SM8750／431e3c37、相同输入、相同QNN编译图，仅运行时算子库切换；三轮交替顺序，warmup20，AR1迭代1000、128步诊断迭代50。均为独立小PTE execute时间，不是整模型token/s。

| 路径 | UINT16对照 | 归一化UINT8旧内核 | 归一化UINT8最终内核 | 相对UINT8旧版耗时下降 |
|---|---:|---:|---:|---:|
| ar1 | 0.245855 ms | 0.228294 ms | 0.198000 ms | 13.27% |
| recurrence | 2.166420 ms | 6.599960 ms | 3.135540 ms | 52.49% |

AR1相对UINT16耗时下降约19.5%。128步原生诊断仍比UINT16原生内核慢，不把它当作正式prefill；正式prefill仍走约1.15ms的FP16 chunk。完整模型decode/prefill端到端性能本轮未测。

## 1. 换库路径问题

`QnnBackendCommon.cpp` 按PTE compile spec里的`op_package_path`原样注册库。旧micro-PTE写入`/data/local/tmp/hunter/qwen35-gdn-state8/`；归一化micro-PTE写入`/data/local/tmp/hunter/qwen35-state8-optimization/`。把新so推到另一个测试workspace不会覆盖这些路径。

因此此前标记“普通UINT8原库／新库”的换库AB比较无效，两次仍可能加载同一旧路径；不能用这些数字判断SIMD修改有没有收益。此前FP32/QDQ精度、PPL消融及归一化PTE本身的执行结果仍有效。

此次只重写测试PTE的runtime op-package路径到独立`qwen35-state8-perf-before`和`qwen35-state8-perf-final`目录。重新反序列化后校验QNN delegate data SHA完全相同、IO不变；校验每轮手机对应路径so的SHA与host版本一致。没有重新编译QNN计算图。

第一次未修路径的本轮测试已移到`invalid-path-runs/`，不作为优化结果。寄存器版本和试验性I16系数分支的结果分别保存在`register-runs/`、`exact16-runs/`。

## 2. 保留的内核修改

代码：`examples/qualcomm/custom_op/deltanet_op_package_htp/DeltaNetOpPackage/src/ops/DeltaNetState8.cpp`。

- `reduceState8Tile<Aligned>`：把indexed accumulator数组改为显式HVX寄存器变量，减少归约结束后的数组落栈与重载；<=128 keys一组，保持完整Q/K位数与I32溢出界限。
- `storeStateProjection`：直接恢复Q/K零点补偿并保存投影，不保留跨K循环的八个FP32总和向量。
- `updateStateRows8<Aligned>`：V tile在外、K行在内，四个delta向量跨所有K行复用；旧实现每K行调用一次update、重载delta并把word/code临时数组放入栈。
- `deltanetCoreQhpiState8`：把state/key的scale除法与固定系数准备移到K循环外，避免128行重复计算。
- 已对齐且V stride是128倍数时使用VMEM路径，其他shape保留未对齐/尾部路径。

试过“所有Q/K系数可放进I16时省去sum(S)零点补偿”的额外分支。约95%的样本head/token满足条件，但三轮结果没有稳定额外收益，最终移除，保留更简单的寄存器/复用版本。UINT16 native内核未改。

## 3. QNN profile解释prefill收益为什么小

profile使用runtime level2，仅用于定位。它显著增加测量开销，不用profile运行的绝对耗时代替上面的无profile性能。下列比例按每节点cycles的中位数加总，属于算子周期分布，不等于并行图关键路径或DDR带宽分布。

归一化UINT8的AR128 FP16 chunk图约208个profile节点：

| 分类 | 节点cycles占比 |
|---|---:|
| 36个MatMul | 31.91% |
| 20个Gather/index_select | 20.22% |
| 全部8个dtype/layout Convert | 6.34% |
| 其中state入口和出口两次Convert | **1.80%** |
| 额外V/c和Y×c两次缩放 | 7.64% |

UINT16 prefill对应state两次Convert约2.46%，普通UINT8约1.98%。换成byte持久化后，进入ChunkGDN的工作state仍为真实FP16，K@S、Q@S、state更新及三角逆等矩阵/中间张量访问没有减半。不要把全部state相关计算或VTCM访问都算成UINT8节省。

因此正式prefill只有小幅收益有依据。若继续降低prefill延迟，应优先减少Gather/布局物化、三角逆相关中间操作及缩放节点开销；单改state存储位宽不能替代这些计算优化。

AR1在修正绑定后采集的native core周期：

| 图／库 | core节点中位cycles |
|---|---:|
| state16／before | 1195626 |
| scaled8／before | 1071490 |
| scaled8／final | 739728 |

归一化UINT8 core自身周期下降约31%，无profile的execute下降约13%。delegate、输入输出和DSP调度等其他成本仍在，不能把core周期收益直接当作整模型吞吐收益。

QNNProfiler把cycles/us/count等数值直接写入ETDump；默认Inspector按ns→ms统一缩放会改变显示量级。本轮bound profile使用原始delegate数值，事件单位以名称后缀为准；旧prefill CSV恢复cycles时乘回1e6。没有把cycles直接换成毫秒。

## 4. 精度与验证

测试中AR1和128步诊断的attention输出、state输出与旧库均逐code100%一致、最大差0。calibration、scale/zero point、R表示和网络量化规则未改变。该结果针对本轮固定输入，不是任意浮点输入都逐bit一致的承诺。

libnative随机90组涵盖aligned/unaligned、K到256、V尾部、独立输出与原地更新，通过；有1处靠近FP32半步舍入边界的差异为1code，已记录，其他更新与参考一致。x86/Android/v79构建通过。完整生产W4 PTE PPL没有重测；此前state精度消融记录保持独立。

## 产物

目录：`llama-output/qwen35-state8-performance-20261017/`。最终数据为`device-results.json`、`device-summary.json`、`accuracy-vs-before.json`、`profile-bound-node-summary.json`、`verification.json`；包含每轮runner日志、raw输出、已绑定PTE及before/final库SHA。汇编对照为`before.asm`、`after.asm`，数学检查为`math-final.log`。

本轮没有提交或推送。

<a id="qwen35-prefill-graph-20261017"></a>

# Qwen3.5 AR128 GDN prefill 优化（2026-10-17）

> 后续整模型复测发现，本节的 dense32 高次幂三角逆在后续层会产生非有限值。0.83219/0.87959 ms 是第0层历史小图测量，不能作为稳定完整模型的性能。当前实现已改为32×32矩阵内嵌8×8叶块，保留Gather消除；修复、UINT8专属融合及重新测量见[稳定版记录](#qwen35-state8-cacheio-stable-20261017)。下方保留原始过程以便追溯。

## 结果与测量范围

已接入 `examples/qualcomm/oss_scripts/llama/model/chunk_gdn.py`，由标准 `llama.py` 的 `replace_prefill_cores()` 自动使用。部署 AR 保持 128，内部仍是两个顺序更新 state 的 64-token chunk。没有增加实验开关；AR1 native VTCM 内核不在本次修改范围内。

同一 SM8750／431e3c37，真实 Qwen3.5-0.8B 第 0 层 Q/K/V、decay、beta 和非零初始 state；B=1、H=16、K=V=128。共享 buffer，warmup20、iteration100、三轮交替顺序，中位数。QNN SDK2.49.40.260810；测试全部串行导出。

| state 路径 | 原版 AR128 | 新版 AR128 | 耗时下降 | 加速比 |
|---|---:|---:|---:|---:|
| UINT16 | 1.19436 ms | **0.83219 ms** | **30.32%** | 1.435× |
| 归一化 UINT8，R=S/c | 1.16197 ms | **0.87959 ms** | **24.30%** | 1.321× |

这是独立 GDN 小 PTE 的 execute 延迟，含 delegate 调用，排除加载和 ADB；不含投影、norm/z、out_proj、FFN、视觉编码器。不能直接作为完整模型 token/s，也不能将它乘层数视为整模型实测。本轮没有重导完整模型、重测端到端吞吐或完整生产 W4 PTE PPL。旧完整 PTE 不会自动改变，需要重新导出。

## 保留的修改

1. **Gather 改静态切片。** 原来把 64×64 lower 展平成 4096 元素，按 int32 索引提取对角/左下块，再按 even/odd 索引提取逆矩阵。现在直接切出两个 32×32 对角块和左下块，逆矩阵以 `[:,0::2]`、`[:,1::2]` 取两块。删除 `leaf_indices`、`inverse_indices` 及运行时 arange。
2. **三角逆叶块8→32。** 对两个 32×32 严格下三角块做有限幂级数，再只合并一级。每个叶块用 `I+L` 开始，四轮 `power=power@power; inverse=inverse@(I+power)` 得到至 L³¹ 的逆；最终左下块仍为 `B⁻¹ T A⁻¹`。MatMul调用数保持不变，但逆矩阵的拼接/重排减少；本次 profile 中 MatMul 节点周期加总也降低约14.1%。这不是减少理论 FLOPs 的算法。
3. **减少布局物化。** AR128 中 Concat19→7、Reshape33→19、Split2→0。dtype Convert仍有8个、Transpose仍有16个，没有声称它们全部消失。
4. **mask/符号融合。** `exp(clamp(g_i-g_j))*lower` 末尾已经清掉上三角，删除 clamp 前的重复 `*lower`；把 strict mask 的负号预合并到常量 `negative_strict`。动态 Multiply22→20、Neg2→0。beta仍在原位置，所有校准得到的 QDQ边界参数保持一致。
5. 保留 odd block 的 `+2**-24` 物化：直接让切片进入 QNN Concat 曾出现 stride 错误。实机验证后保留这项兼容处理。

| 原始 QNN wrapper 类型 | 原版数量 | 新版数量 |
|---|---:|---:|
| Gather | 20 | **0** |
| StridedSlice | 14 | 30 |
| Pack | 0 | 2 |
| Concat | 19 | 7 |
| Reshape | 33 | 19 |
| Split | 2 | 0 |
| MatMul | 36 | 36 |
| Convert / Transpose | 8 / 16 | 8 / 16 |
| Multiply / Neg | 22 / 2 | 20 / 0 |
| wrapper总数 | 204 | 170 |

归一化 UINT8 图另外各有一个 `V/c` Divide 和 `Y*c` Multiply；它们仍存在。

## 精度：连续8×128 token

用相同测试流，从零 state 开始，每步把手机 raw state 输出直接作为下一步输入；没有用 FP32 state 重置，也没有 host DQ/Q。指标参考 FP32 第0层，attention output 包含 FP32 RMSNorm/z/out_proj，未加入 W4投影误差。

| 路径 | core output 累计 SQNR，旧→新 | attention output 累计 SQNR，旧→新 | 第1024 token state SQNR，旧→新 |
|---|---:|---:|---:|
| UINT16 | 50.2855→50.2729 dB | **37.7526→37.7556 dB** | 41.3269→41.3313 dB |
| 归一化 UINT8 | 43.3874→43.5530 dB | **37.1684→37.1673 dB** | 33.3664→33.6227 dB |

逐128token的 state输入、state输出、core输出、attention输出 SQNR 在 `accuracy-*.csv` / `analysis.json`。优化只改变 FP16 运算组织，未声称逐bit一致。UINT16最后一个块的core SQNR下降在旧版同样存在（43.0905→43.0881dB），不是此次新引入的现象。

GPU检查覆盖8个真实128-token块：新版相对旧版 core/state SQNR均大于60dB。随机shape覆盖 `[B,H,T,K,V]=(1,1,64,32,64)、(2,3,128,64,96)、(1,4,256,128,128)`，对FP32 recurrence SQNR均>57dB；三角逆对FP64 inverse的最大绝对误差<5.5e-5。导出无index_select/arange。两种PTE均一个QNN delegate、零CPU fallback；IO为6个UINT16→2个UINT16，或5个UINT16+BYTE→UINT16+BYTE。state8不是扩展到UINT16计算网格，内部chunk仍是真实FP16。

## 未合入的候选

以下部分是筛选阶段的单轮，不能与最终三轮中位数混作同一组AB；原始记录保存在对应device JSON/runner日志。

| 候选 | 测量 | 决定 |
|---|---|---|
| 只改切片、保留8叶块 | 三轮中位0.96407ms，配对原版1.19014ms | 有效，但32叶块更快 |
| 最后一级改为两块直接solve | 单轮0.99437ms，配对切片0.96315ms | 更慢 |
| 合并Q/K相关MatMul并solve | 三轮中位0.95374ms | 不如最终方案 |
| 两chunk准备合并、head/chunk折叠 | solve三轮1.05055ms；完整inverse单轮1.00764ms | 布局开销抵消减少调用的收益 |
| 合并batch+MatMul / 连续batch轴 | 单轮1.02445ms / 1.38608ms | 更慢 |
| 16叶块 / 32叶块 | 同轮0.92606ms / 0.84212ms | 选择32 |
| 64叶块 | 单轮0.87690ms，对照32为0.84750ms；精度也略差 | 选择32 |
| beta移到inverse列上 | 三轮0.85637ms，对照mask32为0.82900ms | 保留原beta位置 |
| `V/c`改量化倒数乘法 | 同轮三轮0.87811→0.86762ms，仅降1.19% | 冷启动core SQNR60.27→52.43dB，未合入 |

倒数乘法试验保留原来的激活网格，倒数取自校准后DQ(c)，另建UINT16常量编码；它引入额外倒数量化误差。1K累计core SQNR43.5530→42.8570dB，attention37.1673→37.1309dB。没有为约1%的收益更改生产归一化参数，也没有跨QDQ边界直接删除`V/c`或`Y*c`。

## Profile与剩余瓶颈

runtime profile level2、warmup2、iteration10。Inspector保留原始delegate数值，事件后缀为cycles/us/count。详细profile大幅扰动耗时（UINT16约9.80→5.82ms），不用这些绝对数值报告性能。以下是每节点周期中位数的加总，不等于并行图的关键路径，也不等于DDR/VTCM字节数。

| UINT16节点类别 | 原版周期加总 | 新版周期加总 |
|---|---:|---:|
| Gather + StridedSlice | 2,126,347 | 133,513.5 |
| MatMul | 3,334,480 | 2,864,036.5 |
| Convert | 732,525 | 705,665.5 |
| Transpose | 294,800.5 | 289,002.5 |
| Concat | 89,582.5 | 23,588.5 |
| 所有节点 | 9,604,640.5 | 7,038,102 |

新版归一化UINT8的`V/c`仍约330,498cycles，占节点周期4.29%；新版MatMul约37.37%、Multiply约20.07%、Add约18.32%、Convert约8.37%。UINT8新版仍略慢于UINT16新版，R归一化缩放和内部FP16计算没有随byte持久化消失。后续需要在保持校准/量化边界的前提下融合缩放与邻接算子，或优化MatMul/逐元素链；不能将直接删QDQ或低精度倒数当成无损融合。

## 复查与产物

`ChunkGDN`的标准替换接口、input/output顺序、cache网格及state反馈协议不变。没有改FA/SWA mask、MHA/SHA、量化recipe或runner选图。只保存最终实现，实验候选代码在`/tmp`，未加入仓库源码。

`analysis.json`包含性能、精度、profile、PTE SHA和source SHA；`boundary-verification.json`核对前后QDQ参数；`source-validation.json`保存数值/shape检查；`*-audit.json`保存QNN算子dtype/shape；`device-production-*.json`保存最终三轮；`feedback-*`保存连续反馈raw；`profile-*`保存ETDump和CSV；`baseline-source.py`保留原版源码供复现。

没有提交或推送。

<a id="qwen35-state8-cacheio-stable-20261017"></a>

# GDN UINT8 cache IO 融合与三角逆稳定性修复（2026-10-17）

## 范围与结果

SM8750／431e3c37，Qwen3.5-0.8B第0层真实输入，AR128，B=1、H=16、K=V=128，非零初始state。三个小PTE使用相同稳定三角逆；共享buffer、warmup20，每轮500次，五轮交替顺序，中位数。QNN2.49.40.260810，Hexagon6.7／Tools21.0.04，手机实际库SHA与主机一致。

| 路径 | 五轮execute耗时（ms） | 中位数 | 对UINT16耗时变化 |
|---|---|---:|---:|
| UINT16 | 0.883452 / 0.890244 / 0.892334 / 0.892984 / 0.897644 | 0.892334 ms | 基准 |
| 旧归一化UINT8 | 0.919218 / 0.920312 / 0.920994 / 0.920228 / 0.924284 | 0.920312 ms | +3.14% |
| 融合UINT8 | 0.850580 / 0.866480 / 0.851366 / 0.860464 / 0.852160 | **0.852160 ms** | **−4.50%** |

融合UINT8相对旧UINT8耗时下降7.41%。这次没有把共有的Gather/三角逆优化计入UINT8专属收益。测量为独立GDN delegate execute，不含投影、norm/z、out_proj、FFN或视觉编码器，不能作为完整模型吞吐。

## 原因与实现

旧byte路径存R=S/c，在R坐标中做FP16 recurrence：`V/c → ChunkGDN → Y*c`。因此除了cache转换，还多了输入Divide、输出Multiply及中间量化，抵消了byte存储收益。

新路径在边界融合转换与缩放：

```text
byte R → StateCacheUnpack → physical FP16 S
physical V + S → FP16 ChunkGDN → physical Y / new physical S
new physical S → StateCachePack → byte R
```

`StateCache.cpp`两个QHPI算子直接计算`S=(code−zp)*step*c`与`code=round_even(S*(1/c)/step+zp)`。c取自已校准的冻结常量DQ，step/zp取运行时QHPI tensor metadata；不是写死的量化网格。倒数用FP32常量，未再量化为UINT16。HVX每次处理128个value，按32个key row分任务，保留非整齐维度尾部。

`generation.py::replace_prefill_cores()`识别已校准的归一化UINT8图，保留physical V/Y与byte R输入输出的原网格，去掉中间`V/c`及`Y/c`量化和恢复Multiply。每层有独立c，两个不同层scale的替换验证通过。普通UINT16图和非匹配图继续用普通ChunkGDN；未更改默认state16配置、FA/SWA mask、MHA/SHA或runner选图。

| QNN节点 | UINT16 | 旧UINT8 | 融合UINT8 |
|---|---:|---:|---:|
| MatMul | 36 | 36 | 36 |
| Convert | 8 | 8 | 6 |
| Divide | 0 | 1 | 0 |
| Multiply | 26 | 27 | 26 |
| StateCacheUnpack / Pack | 0 / 0 | 0 / 0 | 1 / 1 |
| Gather / Concat | 0 / 7 | 0 / 7 | 0 / 7 |

state持久存储从每层512KiB减至256KiB；18层一个逻辑state合计9→4.5MiB。AR128全部外部输入/输出逻辑字节数仅从3.0078125→2.5078125MiB，下降16.62%，因为Q/K/V/Y等仍是16bit。内部MatMul与工作state在本版仍FP16，每128token才读写一次外部state，内部流量并未减半。新c/倒数常量合计16KiB/层，比旧单份UINT16 c增加12KiB/层；不能把所有模型内存都说成减半。

## 修正上一版数值稳定性结论

上一轮dense32高次幂三角逆的0.83219/0.87959ms，仅覆盖第0层小图。整模型state16对照也出现NaN：本次参考输入中第5个GDN（transformer层索引5）、第二个64-token chunk的inverse首次非有限，输入lower仍有限。不能把这个问题归因于UINT8。

aligned keys、beta=1时dense32的L²/L⁴最大值为30/4060，L⁸溢出FP16，而最终数学逆是有界的。当前32×32操作数内只保留独立8×8叶块，执行到L⁴，再以mask合并8→16→32，最后合并32→64；避开高阶dense幂。仍无Gather、Concat7、MatMul36，相比不稳定版增加6个Multiply。保留odd slice的`+2**−24`物化兼容处理。

aligned keys beta=1、0.999、0.99、0.9均有限；beta=1的128步output SQNR53.21dB。其他beta的最大inverse绝对误差达0.0197，因此没有将旧随机小矩阵的5.5e−5界推广到全部输入。见`inverse-validation.json`、`overflow-location.json`。

## 精度与兼容验证

连续8×128token，手机raw byte state直接反馈，无host DQ/Q或FP32重置。物理state按前次诊断约定用原始FP32训练c恢复；实际算子使用冻结DQ(c)。attention指标为设备core输出加CPU FP32 RMSNorm/z/out_proj，不包含W4投影误差。

| 指标 | 旧UINT8 | 融合UINT8 |
|---|---:|---:|
| 累计core SQNR | 43.3865 dB | 44.0044 dB |
| 累计attention output SQNR | 37.1698 dB | 37.2678 dB |
| 第1024token state SQNR | 33.3711 dB | 33.5134 dB |

GPU整模型参考测试：四个独立cold WikiText-2 test窗口，各1024token，4092个预测目标；复用train校准，所有模型权重与其他路径保持FP32，仅GDN输入/gate/output A16与state路径变化。该测试使用原始FP32 c，未模拟小PTE冻结c的常量量化，**不是生产W4／INT4 embedding／FA KV8 PTE的设备PPL**。

| 路径 | PPL | 最终hidden SQNR |
|---|---:|---:|
| FP32 | 19.04965 | reference |
| UINT16 | 19.10795 | 36.8345 dB |
| 旧归一化UINT8 | 19.14796 | 26.4378 dB |
| 融合UINT8 | 19.07242 | 28.7002 dB |

这批样本PPL接近，但UINT8 hidden精度仍低于UINT16；PPL略低不能证明UINT8更准确。生产默认保留UINT16，完整W4 PTE精度和端到端性能尚未复测。

新prefill输出raw byte state已直接接现有AR1图，未改delegate数据，输出/state对CPU QDQ SQNR72.42/63.59dB。三个prefill PTE均一个QNN delegate、零CPU算子；UINT8 IO为五个UINT16加一个BYTE输入、UINT16加BYTE输出。

x86、Android arm64、Hexagon v79库编译通过。libnative校验499851个值及800个舍入边界：7个FP16差异、最大1ULP；2个byte差异、最大1code，来自系数乘法重结合。覆盖V=1/31/64/96/127/128/129/256/257、row尾部和非对齐指针。未做SM8850设备验证。

## 证据与剩余工作

`analysis.json`汇总性能、逐128token精度、QNN节点、详细profile及PTE/source/library SHA；`final-compare-results.json`和`final-compare-*`保存五轮实测；`accuracy-*.csv`保存输入/输出state和attention指标；`ppl-reference.json`为参考PPL；`decode-handoff.json`为AR1兼容验证；`rewrite-validation.json`验证每层scale和fallback；build/stress日志同目录。

详细profile只使用原始cycles比较节点，不将周期加总当作图关键路径或DDR/VTCM字节计数。旧UINT8 Divide约325760cycles，新图无该Divide；内部36个MatMul仍占新图节点周期约42.76%。若要进一步获取8bit计算收益，需要另测integer MatMul和相应校准，而不是仅缩小持久state。

未提交或推送。

<a id="qwen35-gdn-integer-matmul-20261017"></a>

# GDN 整数 MatMul 对照（2026-10-17）

## 结论

可以使用与KV8相同的整数MatMul：UINT16左输入、UINT8右输入、UINT16输出，计算图不先把右矩阵还原为FP16。单层AR128已经导出PTE并在SM8750运行。但是目前整数候选没有超过最新融合FP16版本，且精度下降，因此没有替换生产默认路径。

## 测量范围与配置

Qwen3.5-0.8B第0层真实Q/K/V/gates/state，B=1、H=16、K=V=128，外部AR128、内部两个64-token chunk，手机431e3c37。校准只用保存的WikiText train 8×128输入；测试为独立连续8×128输入。MinMax校准，physical输入/输出以及R=S/c byte cache网格复用原来的校准结果。未使用test值重新定scale。

持久state采用原来的归一化R，未直接将物理S用单一8bit网格再量化；后一种做法会丢掉value通道归一化收益。所有可运行PTE均一个QNN delegate、零CPU算子，state输入输出均BYTE，其余五输入和attention输出UINT16。

最终计时：shared buffer、warmup20、iteration500，五轮交替顺序，execute中位数；不包含加载和ADB，也不包含投影、norm/z、out_proj、FFN、视觉编码器。精度采用手机连续raw byte cache反馈，core参考FP32；attention额外使用CPU FP32 RMSNorm/z/out_proj。FP16精度复用同一PTE前轮已保存raw，未将测试state重置为FP32。

| 路径 | AR128耗时 | 1K累计core SQNR | 1K累计attention SQNR | 第1024token state SQNR |
|---|---:|---:|---:|---:|
| 旧归一化R、FP16 MatMul | 0.923438 ms | 43.3865 dB | 37.1698 dB | 33.3711 dB |
| 最新融合cache IO、FP16 MatMul | **0.853852 ms** | **44.0044 dB** | **37.2678 dB** | **33.5134 dB** |
| 完整整数chunk | 0.895760 ms | 37.2444 dB | 32.3472 dB | 26.4963 dB |

完整整数版本相对旧FP16路径耗时下降约3.00%，但比最新融合FP16路径慢约4.91%，累计attention SQNR下降约4.92dB。不能选旧基准宣称它超过当前最快版本。本轮没有导出完整模型、测完整模型token/s或PPL。

五轮耗时：

| 路径 | 各轮execute（ms） |
|---|---|
| 旧FP16 | 0.925742 / 0.920152 / 0.923438 / 0.920314 / 0.925734 |
| 融合FP16 | 0.853852 / 0.851460 / 0.864872 / 0.851402 / 0.854438 |
| 完整整数 | 0.890806 / 0.895760 / 0.893480 / 0.902296 / 0.895920 |

## 试验演进与QNN审计

1. **只改36个MatMul。** 在FP16逐元素图中建立校准后的integer边界，36个MatMul实际全部UFIX16×UFIX8→UFIX16。Convert从8增到116。同轮三次iteration100测量：旧FP16为0.94552ms，整数MatMul候选1.11048ms；warm非零state单块core/state SQNR为35.94/27.36dB，而FP16为47.84/36.17dB。
2. **只改K@state、Q@state。** 四个MatMul整数化，其余32个FP16。Convert20；同轮0.94650→1.04376ms。warm core/state SQNR46.31/36.04dB。这个局部方案误差较小，但没有性能收益。
3. **周围运算一起量化。** 所有逐元素、布局、矩阵运算均为量化tensor；QNN wrapper审计浮点tensor出现次数为0。Convert降到38，避免每个MatMul周围反复转换。32个MatMul为16×8→16；4个`power@power`为8×8→8，随后按需要requantize至16位。

三角逆平方的两输入来自同一个FX节点，现有量化annotation按输入节点而非输入序号建map，不能直接给同一节点同时配置16位和8位。若强行覆盖，会形成QNN不支持的8×8→16；本试验将四个平方统一为8×8→8，保持真实整数计算，没有FP16 MatMul。其余32个继续用KV8式16×8→16。

完整integer PT2E图末尾的同dtype cast曾在lowering时被消除，导致输出错误保留UINT16；增加明确的cache转换边界后，最终PTE确认输出为BYTE，量化参数仍取原校准网格。编译后该边界是integer Convert，QNN审计不存在浮点tensor。

临时部分整数图触发了`FuseConsecutiveCast`对Convert分支的断言；只在该实验进程中停用此融合，保留全部边界，未修改全局backend代码。最终完整整数图走标准pass，不需要这个临时patch。

把36个MatMul全部设为8×8→8的候选在本机QNN graph optimization中段错误退出，未得到可运行PTE。四个小平方使用8×8的完整整数候选则编译并实机运行成功；不能把一次候选编译失败扩大为INT8 MatMul整体不支持。

曾有两个无效实验：直接使用FP16输出dtype进行UINT16 QDQ参考会出现数值转换异常，已改为FP32 QDQ再转换；在DQ与MatMul之间保留FP16 cast会导致MatMul实际仍为FP16，已通过QNN dtype审计剔除。这些数值和计时没有纳入结论。

## Profile与限制

profile level2、warmup2、iteration10。只对每节点周期中位数加总，不将它当作关键路径、实际DDR流量或普通execute毫秒数。

| 节点周期加总 | 仅MatMul整数化 | 完整整数 |
|---|---:|---:|
| Convert | 3,771,249 | 570,985.5 |
| MatMul | 3,832,313.5 | 3,571,850 |
| Multiply | 1,124,019 | 2,492,009 |
| Divide | 326,179 | 323,935 |
| 所有节点 | 10,195,300.5 | 8,712,151 |

只改MatMul时Convert约占37%；完整整数时转换开销明显下降，但Multiply变重，且MatMul没有显示出足以弥补其他开销的收益。矩阵右输入是动态state、key、correction等，存在运行时量化/重排成本；仅依据8bit位宽不能推断这些算子整体必然更快。

完整整数warm单块设备对CPU QDQ的core/state SQNR为44.71/28.98dB，存在需要继续定位的backend/整数算术差异；不能把全部设备误差都归因于权重或cache量化。CPU参考已出现量化损失：连续1K core SQNR约37.5dB，未满足与FP16近似无损的目标。未将该候选合入标准llama.py默认选路。

## 产物

最终完整整数小PTE：`integer/fullinteger8square-scaled8.pte`。

`integer/analysis.json`保存五轮数据、IO类型、QNN MatMul类型、PTE SHA、profile和逐128token精度；`integer/device-fullinteger-results.json`为最终交替计时；`integer/feedback-fullinteger8square-scaled8/`为连续反馈raw；`integer/accuracy-*.csv`为state输入/输出、core、attention指标；`integer/*-audit.json`为QNN tensor dtype/shape；`integer/calibration.json`及`encodings-*.json`为MatMul校准范围/网格。

`normalized/`为早期FP32参考图与未成功lowering的尝试，`normalized-half/`为DQ后误插入FP16 cast的候选；这些不是最终integer性能结果。临时脚本在`/tmp`，没有添加新的生产GDN op或测试源码。未提交或推送。


<a id="qwen35-gdn-experiment-cleanup-20261017"></a>

## 撤销GDN低位实验，保留KV8（2026-10-17）

按用户要求撤回GDN recurrent state8及integer MatMul实验。删除DeltaNetState8.cpp/.h、StateCache.cpp、ValueScaledDeltaNetCore、ScaledStateChunkGDN、annotate_gdn_state_8bit及byte cache pattern替换；包恢复原四个DeltaNet op及原UINT16 HVX/VTCM内核。移除新增的state8测试fixture、试验专属wrapper标注顺序/报告逻辑和HVX byte转换工具；原有KV8、embedding INT4、SWA/mask与runner逻辑不变。

保留与位宽无关、已经验证的通用prefill改进：静态slice代替Gather、mask/符号常量折叠、32×32操作数内嵌稳定8×8三角逆叶块，避免dense32高阶幂溢出。移除为byte融合拆分的forward_fp16辅助入口。当前标准导出为AR128，GDN state/conv UINT16、内部FP16 MatMul；full-attention K/V沿用annotate_kv_8bit。

实验结论保留在本README，不再提供GDN state8或integer MatMul的生产入口。旧实验PTE仅为历史产物，不适合搭配清理后op库部署；正式模型需按当前代码重新导出。

清理验证：移除46个`/tmp`试验源码/driver/可执行文件及缓存，清除byte算子旧object和依赖；x86、Android arm64、Hexagon v79算子库重编通过，三个库均无StateCacheUnpack/Pack、state8.hvx或DeltaNetState8符号。0.8B/2B recipe核对保留annotate_kv_8bit。清理后AR128 UINT16的两个校准输出与已保存稳定版reference逐元素完全一致；导出仍为36个FP16 MatMul、零Gather。源码无state8入口残留，`git diff --check`通过；没有重新运行完整模型性能或PPL。


<a id="qwen36-moe-layer0-20261017"></a>

## Qwen3.6-35B-A3B 真实单层 MoE 可行性测试（2026-10-17）

模型 `/hunter/model/Qwen/Qwen3.6-35B-A3B` 的配置类型仍为 qwen3_5_moe。只读取第0层，hidden=2048，256个routed experts、Top8、intermediate=512，保留真实router和shared expert。输入取真实embedding→GDN→残差→post-attention RMSNorm的hidden，参考为Transformers 5.8.1 FP32；设备计时仅包括MoE FFN，不包括GDN、norm、视觉或其他层。

专家权重为对称INT4 group32、FP16 scale、RTN；router/shared expert FP16。运行时Top8→Gather选中专家的packed byte和scale→解包为FP16→gate/up和down MatMul→加权合并shared expert。实际是W4存储、FP16计算，没有原生INT4稀疏GEMM，没有固定专家编号，也没有把所有256个专家的FFN都算一遍。

手机431e3c37、SM8750，QNN 2.49.40.260810，shared buffer，关闭profile。计时为warmup之后execute，排除加载、ADB和文件IO，三轮中位数：

| 测试 | 耗时 | 对FP32 MoE输出SQNR | Top8集合重合率 |
|---|---:|---:|---:|
| AR1，warmup2/iteration20 | 34.40425 ms | 22.6694 dB | 100% |
| AR128，32次AR4调用，warmup1/iteration2 | 6700.6235 ms | 22.5342 dB | 99.9023% |

AR128单QNN图编译约14分钟仍在SDK GraphPrepare::build_graph_deps，RSS约12.2GiB，主动停止，没有发生OOM。最终AR128实验PTE是同一AR4 QNN图顺序执行32次，CPU仅slice_copy/cat；一个delegate条目、32次调用、一份context，不能当作原生AR128单图prefill性能。初版32个delegate条目导致设备重复内存占用，已去重；外层输出顺序按QNN原始IDs/probabilities/hidden修正，重新导出并重跑。生产backend及llama.py未改。

另外8×128独立test输入，实际命中249个专家，全部输出有限；综合输出SQNR为22.3240dB，Top8集合重合率99.8901%（8192次选择中9次成员不同）。对只量化专家权重的FP32参考为38.4630dB；该W4参考自身对FP32为22.4233dB。主要输出精度损失已存在于RTN权重量化，未测完整模型PPL。

AR1详细profile按去除warmup后的每节点周期中位数加总：解包/scale83.53%、专家MatMul8.63%、Gather6.20%、大tensor布局/cast1.43%、其余0.21%。这是节点周期统计，不是普通execute的时间分摊或DDR/VTCM实测流量。当前通用算子拼接性能不适合整模，需要融合专家读取/INT4解包/矩阵计算，并按expert聚合prefill token。

每层routed expert INT4数据384MiB、scale48MiB，合计432MiB；40层仅这部分为16.875GiB，超过当前手机约10.86GiB物理RAM，整模需要额外的权重换入设计或更大内存设备。A3B不能用来估算全部驻留权重。

产物位于 `llama-output/qwen36-moe-layer0-20261017/`：README详细说明方法和限制，analysis.json汇总最终结果与SHA，source/保存复现代码；ar1.pte为441.71MiB，ar128-tiled.pte为448.80MiB。没有将35B MoE加入正式支持列表，没有修改0.8B/2B或GDN/KV量化默认值，未提交或推送。


### 后续：动态A8/W4与8槽位外存验证（2026-10-17）

MoE不要求FP16。已在SM8750/QNN2.49.40实机运行动态W4/A8 MatMul，一个delegate、零CPU算子、量化IO，图中只有MatMul；同PTE更换三套权重，均对CPU整数参考逐元素一致。权重使用SFIX8容器加BW_SCALE_OFFSET(bitwidth=4)。直接packed SFIX4作为MatMul输入的动态和静态配置均未通过SDK校验，不能把逻辑W4支持等同于packed4无需布局适配或已证明硬件4bit吞吐。

真实路由expert117的gate/up子矩阵（64输入、32输出、group32）使用整数部分和乘真实group scale，再求和，避免浮点权重矩阵。Activation/partial网格仅用train校准；设备对CPU QDQ SQNR39.94dB、对W4权重FP32参考29.91dB，最大绝对差0.00085449，动态scale减半时输出恰好减半。这是子矩阵能力验证，未给出完整整数MoE精度/PPL。

全部256个专家存432MiB外存bank，每个专家1.6875MiB。Android C++ O_DIRECT加载器仅分配8槽、13.5MiB缓冲，进程峰值约17MiB。使用真实128-token路由轨迹：逐token请求1024次、miss921次，读1554.19MiB，pread892.031ms；按expert归并后只读213个不同专家、359.44MiB，pread159.113ms。每种调度独立进程、初始应用cache为空，均只持有8个槽位；此为一层纯IO测量，不包括router、整数解包、NPU计算、ADB或完整模型推理。

A3B是每token激活规模，Top8随层和token变化；公共权重和热专家可常驻，缺失项换入。128-token prefill需要处理专家并集，但可8个一批，不必全部驻留。旧34.40ms/6.70s不能代表此整数/外存路径。两段尚未合并为完整MoE端到端PTE，未更改正式llama.py或0.8B/2B/GDN量化默认值。

详细记录及源码位于 `llama-output/qwen36-moe-streaming-20261017/README.md` 和 `source/`；JSON保留ABI校验、设备数值、IO字节/命中/时间/RSS。未提交或推送。


### AR128 完整 MoE 整数计算与 5 ms 目标（2026-10-17）

用户明确性能目标为128-token prefill。使用同一真实第0层输入、group32 W4权重及Top8路由，在SM8750上建立独立GGML/Hexagon基准，复用本地高通llama.cpp fork的HVX Q4×Q8点积。完整图包含router、TopK/softmax、routed gate/up、SwiGLU、down、加权求和及shared expert；没有CPU算子fallback，未计入attention/GDN、图加载、权重读取/重排或SSD IO。

真实128-token输入命中213个专家，单专家最多承接31个token。没有固定8个专家，没有为每个专家计算全部token。专家权重始终为packed Q4 group32，激活动态Q8，INT32组内点积后乘scale并以FP32累加；router/shared保留FP16权重，优化版允许它们走HMX。Routed专家的MUL_MAT_ID限制为HVX整数，不是FP16专家MatMul。

| 候选，profiling打开 | 完整MoE中位数 | 对原FP32 MoE输出SQNR |
|---|---:|---:|
| 原始整数核，一次图内4×32，router/shared也用HVX | 33.0519 ms | 22.5792 dB |
| 紧凑激活副本，整块128 | 50.1345 ms | 22.5792 dB |
| 同专家2 token复用Q4解包，4×32 | 27.0193 ms | 22.4897 dB |
| 保持128 token，gate/up K切2×1024、down K切4×128 | 26.0235 ms | 22.4897 dB |
| K分块，加4 token复用解包 | 23.5029 ms | 22.4897 dB |
| 限制4 token内核循环展开，尝试降寄存器压力 | 23.6052 ms | 22.4897 dB |

各行2次warmup后5次图调用，表中均打开profiling。保留4 token展开版本，关闭profiling后两轮各7次计时中位数为23.120729/22.484115ms。warm compute目标5ms尚未达到，没有将不成功的紧凑布局或限制展开候选当作收益。

K分块保持group32完整，权重值/scale不变，每个子矩阵仍看到全部128-token路由；不同K部分的权重互不重叠，避免沿token分块重复读取权重。紧凑激活副本虽然减少VTCM，但每次点积要额外广播，反而变慢。4 token版本复用一次权重解包执行四份整数点积，尾部按2/1处理。该版本K循环仍有向量spill/reload，未声称已无spill。

最终输出全部有限，对W4权重FP32计算参考SQNR37.7256dB；逐token对原FP32 SQNR最小/中位/最大17.4258/22.6723/25.6805dB。Router/shared HMX舍入使1024次选择中1个成员变化，Top8集合重合率99.9023%；不是全部误差都来自专家点积。未测完整35B模型PPL。

最终profile的平均gate/up约10.14ms、down6.82ms，占NPU节点时间约84%；router/加权0.63ms、shared约0.51ms，其余约2.15ms。此为逐节点NPU分组，不直接等于host调用时间。当前主要瓶颈仍在HVX整数专家投影，尚未获得整数HMX矩阵吞吐。

此warm基准把单层256专家约432MiB驻留DDR，**不满足最终8槽位流式执行器的内存目标**。独立8槽位加载器已经证明只用13.5MiB专家缓冲可读同一213专家并集，但冷读取359.44MiB用时159.113ms；两段尚未合并执行，不能将22.48–23.12ms叫作含SSD端到端，也不能简单相加称作实测执行器延迟。5ms内仅读这些权重就要求约75.38GB/s payload带宽，这是数据量推算，不是实测DDR带宽或严格不可实现证明。

源码、构建/运行脚本、各版本输出/计时/反汇编、hash和完整方法说明位于`llama-output/qwen36-moe-ar128-5ms-20261017/README.md`及`source/`。只修改本次实验副本，没有修改外部SDK/llama.cpp源码或正式llama.py/attention/GDN/量化默认值，未提交或推送。


### AR128 MoE 热点定位与预排、线程调度优化（2026-10-17）

继续使用同一真实35B-A3B第0层输入、213个命中专家及group32 W4权重，在SM8750进行单层MoE warm测试；没有改attention/GDN或量化规则。

上一版NPU逐算子profile中，gate/up 10.1424ms、down 6.8230ms，合计约占84%。添加诊断周期计数后，gate/up计算区段8.9056ms、暴露DMA等待0.8965ms；down计算区段5.0061ms、等待1.1353ms。激活DMA及Q8量化总计约0.2221ms，优先处理专家投影热循环。

分段时间选择每个投影部分“计算+等待”最大的worker，再按gate/up、down求和，未把六个并行worker简单相加。计算区段包含寻址、解包、INT32点积、group scale、VTCM读取/spill及写回，不能称为纯MAC时间；DMA pop等待也不是全部DDR成本。level3的host时间被大量trace日志放大，未用来做性能比较。

第一项优化在载入时按点积顺序重排INT4 nibble，使一个word的四个K值直接对应vrmpy lane；运行时以mask/shift替代多条shuffle/align。INT4整数、scale、DDR字节数、激活Q8方法及FP32累加均不变；没有重校准，没有专家FP16矩阵。1/2/4 token路径和host逆排列同步修改。反汇编hvx_mm_id静态vshuff从16处降到4处、valign从8处降到0处，向量spill仍存在；不能当作动态指令计数或声称已无spill。

第二项优化根据真实router结果按token数排序命中专家，6个worker通过原子计数各自领取完整专家，替代每线程遍历所有专家、分摊输出行。减少专家级短预取/寻址序列，并改变并行访存方式；总权重数据量没有减少6倍。任务状态每次调用初始化，实验任务数组上限256，仅用于当前独立基准。

关闭profiling，每轮2次warmup、7次完整图调用：

| 配置 | 第一轮中位数 | 第二轮中位数 | 两轮中位数均值 |
|---|---:|---:|---:|
| 上一版，6 worker | 23.0376ms | 23.1492ms | 23.0934ms |
| 预排nibble，6 worker | 19.7552ms | 20.0226ms | 19.8889ms |
| **预排nibble+专家任务，6 worker** | **19.5298ms** | **19.2959ms** | **19.4129ms** |

先交替测原版/预排，再测预排/专家任务。原版到最终版延迟均值下降15.94%，加速约1.19×。额外减少为4个worker反而24.1064ms，没有采用。最终profile中gate/up为7.5247ms、down5.7390ms，两个专家投影仍占约80%；所有这些NPU节点时间与host调用时间为不同口径。

所有候选与上一轮final128输出逐元素一致，最大绝对差0、全部有限。对原FP32 MoE输出SQNR保持22.4897dB，对W4/FP32参考37.7256dB，路由未新增变化；没有测完整35B模型PPL。

5ms目标仍未达到。这些结果排除SSD IO，且单层256专家约432MiB仍驻留DDR。此前8槽位加载器的159.113ms冷IO成本未重测，也没有与计算合成执行器，不能将19ms当作含SSD端到端性能。仅保留本次独立源码副本，未接入标准llama.py/量化配置，未修改外部SDK/llama.cpp，未提交或推送。

完整记录、代码生成/构建/运行脚本、分段周期、逐次耗时、输出hash及反汇编位于`llama-output/qwen36-moe-ar128-hotspots-20261017/README.md`、`source/`、`phase-breakdown.json`和`comparison.json`。


### AR128 MoE 专家数组直接索引（2026-10-18）

256个专家权重原本已按expert维连续存储，使用`base + expert_id * stride`取址。本次保留权重布局，在独立GGML实验增加显式`experts[256]`描述符，router的Top8 ID直接索引权重地址、该专家token列表及数量；不复制选中的完整专家矩阵。gate/up的2个K部分及down的4个K部分分别取各自的专家基址。

路由改为CSR列表，提前算好量化激活在VTCM中的byte offset及输出byte offset，4/2/1 token内核省去每个输出tile重复计算slot取模/地址。仍按token数安排六个worker领取专家ID；未重编号专家或修改Top8 slot/合并顺序，每次调用重建，避免复用过期路由。v79路由scratch由63KiB降为11KiB，不含共有的队列/任务状态或临时栈；权重仍432MiB全部驻留DRAM。

同机关闭profiling，旧/新交替两轮，每进程2次warmup后15次完整AR128 MoE调用。旧版两轮中位数19.490834/19.175886ms，均值19.333360ms；新版19.282605/19.354531ms，均值19.318568ms。差异约0.08%，没有明显提速，不宣称已解决专家投影瓶颈或达到5ms。MoE输出、router logits、IDs、scores、shared输出逐字节一致；对原FP32 SQNR22.4897dB、对W4/FP32为37.7256dB，真实命中213个专家，无新增PPL。

代码及构建入口在`llama-output/qwen36-moe-ar128-hotspots-20261017/source/indexed-experts.h`、`prepare_variant.py`、`build.sh packed-indexed`；`indexed-comparison.json`记录全部样本与输出hash，`experts-build/`保留旧版库。仅修改独立实验副本，未接入正式llama.py；计时不含attention/GDN、图加载或SSD，尚未与8槽位加载器合并，未提交或推送。


### AR128 MoE 专家融合流水线（2026-10-18）

借鉴按专家组织中间结果、融合后处理的做法，在独立GGML/Hexagon副本实现完整routed FFN的HTP操作。仍为真实第0层256专家/Top8、group32 packed INT4、动态Q8、INT32点积及FP32跨组累加；router/shared沿用FP16/HMX。专家计算仍是HVX整数核，未改为FP16，没有接入HMX整数核。

每次调用重建CSR路由；输入只量化一次，在VTCM保存2MiB紧凑结果。六个worker各自连续处理同一专家的小批token，gate/up的2个K部分→SwiGLU→量化/布局→down的4个K部分及部分和均留在VTCM。每条路由加权后写入互不重叠的8MiB持久scratch，再按原Top8顺序合并。完整融合cap24占用7.87695MiB VTCM，只计算有效token；超过24条时分块，未修改专家ID或合并顺序。

SM8750同一构建关闭profiling，串行未融合/融合/未融合/融合，每进程2次warmup后15次完整MoE调用。未融合两轮中位数19.080104/19.028021ms，均值19.0540625ms；融合17.658802/17.666094ms，均值17.662448ms，延迟下降7.30%、约1.079×。修复后的两阶段cap32候选另测一轮18.091146ms，保留为对照。

连续8组真实128-token输入覆盖249个专家，每块单专家最多26–38个token，已覆盖cap24尾部。完整版及两阶段版的输出和IDs与未融合版逐字节一致，最大差值0，全部有限；首块MoE输出SQNR对原FP32为22.4897dB，8块合并22.3431dB，对W4/FP32参考39.3512dB，没有新增整模PPL。

首次完整融合动态分配scratch约21.9ms，加入输入紧凑缓存单独未明显提速；采用context持久scratch后保留上述最终候选。两阶段早期直接DMA读取尚在DDR缓存中的gate/up结果导致SQNR12.9377dB，已修复为缓存一致的HVX搬运；该早期无效结果未参与精度或性能结论。level3分段只用于诊断，不使用被日志放大的host时间。

代码、复现、候选历史及SHA在`llama-output/qwen36-moe-expert-pipeline-20261018/README.md`、`source/`、`comparison.json`、`manifest.json`。仍是单层432MiB专家权重驻留DRAM，计时排除attention/GDN、加载/重排及SSD IO；未合并8槽位外存执行器，未接入正式llama.py/其他模型，未提交或推送。5ms目标尚未达到。

### AR128 MoE 整数HMX与40层成本诊断（2026-10-18）

在SDK的`Tools/target/hexagon/include/hmx_hexagon_protos.h`找到并打通真正的packed INT4 × UINT8 HMX路径。模拟器和SM8750手机的32×32×32探针均1024/1024输出与CPU整数参考相同，包含负权重和输入零点修正，没有展开成专家FP16矩阵。完整group32混合候选使用持续HMX消费者及每HVX worker一个请求槽，至少8条路由的专家批次交给HMX，其余保留HVX。8组真实输入的MoE输出与原版逐字节一致，但探索性整层延迟20.417605ms，未采用。

group32独立scale要求逐组输出/缩放。当前HMX清累加器、加载乘加、UINT16输出的纯指令序列实测约20.65ns；若每专家只做一次32-row tile、不分块，213专家需要654336组，外推13.51ms，不含DMA、排列及缩放。这不是硬件峰值下界或完整MoE性能。大K合并的人工指令探针有吞吐潜力，但CPU真实层量化预检中，group32 W4A8为22.57dB，group128为20.62dB、group256为19.80dB、per-channel为18.67dB；采用RTN而非GPTQ，未选择精度折损，未测整模PPL。

另测读取真实213专家约359.44MiB权重的无计算DMA pass，六worker双缓冲，二维DMA单次诊断7.83495ms，连续1D为6.715ms，后者不含补齐；这些不是稳定性能基准或硬件下界。运行时连续读取并在VTCM反向补齐的完整MoE候选18.454479ms。加载时先按640-byte tile对齐，DRAM由432MiB升至480MiB；关闭profiling，原版/预对齐交替两轮，各2次warmup及15次调用，原版中位数17.653073/17.741146ms、均值17.6971095ms，预对齐17.447657/17.618802ms、均值17.5332295ms，仅0.93%收益，未采用内存代价。首轮预对齐曾16.859115ms，正式对照未维持该收益，不使用首轮作为结论。

进一步复用已有Q8工作区批量预取多个输出tile，不增加VTCM分配；根据首块真实路由，权重DMA请求由68480降至2096次，但完整MoE探索中位数17.846510ms，未见提速。上述四个候选各通过8组真实输入逐字节校验，首块设备SQNR保持22.4897dB，均未替换保留版17.662448ms。

40层按保留版同耗时外推，仅MoE热计算约706.5ms，排除attention/GDN与SSD，非整模实测。128 token真实命中213个专家，不能将A3B理解为跨token固定的同一组3B权重；40层按相同流量外推约14.04GiB权重读取。单层359.44MiB要在5ms内读完需要约75.4GB/s有效传输，尚未计其他流量。40层专家bank共16.875GiB，超过本机约10.86GiB总RAM；此前159.113ms/层的独立冷SSD读取成本尚未与计算合并。当前需要继续处理可合并K的整数计算/校准误差和整模专家缓存/换入，5ms目标未达到。

本轮完整记录与样本在`llama-output/qwen36-moe-integer-hmx-20261018/README.md`、`results.json`及`manifest.json`；预对齐及批量DMA副本位于相邻`qwen36-moe-padded-dma-20261018/`、`qwen36-moe-burst-dma-20261018/`。未修改正式模型/量化配置、外部llama.cpp或SDK，未提交或推送。

### AR128 MoE K128 原生整数 HMX 与 5ms 目标（2026-10-18）

继续使用本地Qwen3.6-35B-A3B真实第0层、256专家/Top8、hidden2048、expert/shared512及SM8750/v79。完整MoE计时包含router、Top8、routed FFN和shared expert，排除attention/GDN、SSD、加载及载入时重排。专家保持packed INT4，激活进入整数HMX；没有使用专家FP16 MatMul。首块128 token命中213个专家，校准与精度分别使用独立的1024-token train/test，测试集分8块且覆盖249个专家。

将权重/激活量化group32改为group128，4个K32连为一次K128深度HMX命令，减少累加器输出和缩放。累加输出先除4以适配UINT16，减去同步缩放的输入零点修正，FP32后处理乘4；SDK模拟验证整数恢复误差≤3。真实整层初版16.2776ms，8块设备输出SQNR20.4783dB，相对原版22.3431dB有精度损失。

每次HMX请求从一个N32输出块扩为4块、再扩为8块，六个HVX producer服务一个HMX consumer。载入时把每个N32/K128块预排为2048-byte原生N4、64-byte FP16 scale、64-byte INT16权重code和，共2176byte；运行时连续DMA，省去位重排/权重和计算/二维补齐。专家bank从432MiB降为408MiB，213专家实际读取339.46875MiB；八块版使用7,579,648byte VTCM。各RTN候选在8块真实输入上与K128初版逐字节一致。

为恢复精度，用train真实专家路由的GU输入及FP32 GU/SwiGLU产生的down输入统计E[x²]，对每个group128搜索1.00至0.80的截断比例，最小化激活加权权重平方误差。此为scale/截断校准，不是GPTQ，不增加推理图计算。CPU固定FP32路由参考中，RTN为20.5252dB、不加权MSE20.9662dB、激活加权21.6258dB；真实设备最终8块为21.5014dB，比原group32基准低0.8417dB。首块为21.6910dB。全部有限，最终8块Top8 ID与原版相同。没有测整模PPL，不能据此认定40层精度合格。

关闭profiling，baseline/candidate交替两轮，每进程2次warmup、16次计时，避免约10/12ms快慢分布下奇数样本中位数偏移：

| 配置 | 两轮合并中位数 | 两轮合并平均值 | 1024-token设备SQNR |
|---|---:|---:|---:|
| group32 W4A8，原HVX融合版 | 17.5255ms | 17.0907ms | 22.3431dB |
| group128 W4A8，原生N4/HMX，激活加权校准 | 10.6619ms | 11.0159ms | 21.5014dB |

中位数延迟下降39.16%，平均值下降35.54%，**仍未达到5ms**。五worker、成对输出转换/预计算scale以及提高HMX线程优先级没有形成更快的稳定候选，未采用。最低9.8ms探索样本不作为最终性能。

独立纯IO诊断只把真实213专家的339.47MiB权重由DRAM读入VTCM，完全跳过专家计算/重排/缩放，使用DSP qtimer计数。128/256/512/672KiB DMA块、六worker的10次中位数分别6.1614/6.1757/6.1693/6.1668ms；四worker、768KiB为6.1302ms，672KiB有一次5.5549ms但未稳定。5ms仅读这些payload就要求约71.19GB/s，本次约57.8GB/s。此为当前DMA实现的实测限制，不是所有算法/硬件的严格下界；不能将IO-only与完整图时间相加。当前先受权重读取量及传输吞吐约束，继续少量指令融合不能保证5ms。

新bank按40层相同尺寸仍约15.94GiB，超过本机约10.86GiB总RAM；旧独立加载器159ms/层的冷SSD成本尚未消除，也未与本内核合并。单层warm计时不能代替40层外存执行器性能。

完整过程、复现命令、全部样本、8块SQNR、IO计数与SHA位于`llama-output/qwen36-moe-native8-hmx-20261018/README.md`、`RESULTS.json`、`MANIFEST.json`；纯IO源代码在相邻`qwen36-moe-io-limits-20261018/`。新快路径保留为独立研究候选，原group32精度基准保留，未接入正式llama.py或修改量化默认值。

### AR128 MoE 减少实际权重读取（2026-10-18）

使用同一SM8750/v79、真实第0层256专家/Top8及上一轮激活加权校准的group128 W4A8。该checkpoint量化后的GU存在整行code为零的通道；只有完整gate或up行在全部K上均为零，才删除对应恒零SwiGLU通道。只在原group128内部置换，gate/up行与down列同步重排，保留到K32/N32边界，保持原量化scale和激活分组。未引入幅值剪枝或减少Top8；不能假定其他checkpoint有同样稀疏度。

256专家的有效中间通道最少93、最多510、平均300.06，原为512。将固定stride改为`offsets[256]`索引的可变长度INT4 bank；每专家128-byte头描述实际N/K块，连续DMA只读取保留payload。同步重算整数输入零点修正所需的权重code和。GGML I8 tensor仅承载不透明字节，每字节仍两个INT4权重，计算仍为UINT8×N4 HMX。随后缓存HMX消费者请求字段，只执行有效N32块的SwiGLU。

专家bank由408MiB降至289.0527MiB；首块实际213专家读取由339.46875MiB降至241.08960MiB，减少28.98%，包括专家头，不含额外约5KiB索引表读取。纯IO使用6 worker、128KiB双缓冲、DSP qtimer，两个串行交替轮次各16次：原bank中位数5.8827ms、紧凑bank4.3877ms，下降25.41%。收益来自实际少读字节，本轮有效字节带宽没有提高。IO-only不含计算，也不代表SSD或硬件严格下界。

另测已有`GGML_HEXAGON_OPPOLL=1`完成队列轮询，减少AP唤醒开销，但持续占用一个Android CPU核，不增加权重带宽。完整MoE关闭profiling，每进程2次warmup、16次计时，两个串行交替轮次合并32样本：

| 配置 | 合并中位数 | 平均值 |
|---|---:|---:|
| 原bank，阻塞等待 | 10.0123ms | 10.5475ms |
| 紧凑bank，阻塞等待 | 9.6981ms | 9.7360ms |
| 原bank，轮询 | 9.3589ms | 9.4066ms |
| **紧凑bank，轮询** | **8.3610ms** | **8.3655ms** |

相同轮询配置下，中位数下降10.66%；原阻塞版到最终轮询版平均延迟下降20.69%。两项收益分列，不能将轮询收益归给压缩。时延重复首块真实128-token输入；精度独立覆盖8块1024 token、249专家及超过32条路由的尾块。输出和Top8 ID与原group128版逐字节一致，最大绝对差0，全部有限，SQNR保持21.5014dB。本轮压缩没有新增量化误差，但不代表相对FP32逐位一致，也没有测整模PPL。

更大分组GU1024/down512的独立消融正式中位数5.6973ms，但设备SQNR降至19.9256dB，未采用。HVX广播、VTCM stride、原生TopK及CPU亲和性没有形成更好的保留结果。**5ms完整MoE目标仍未达到**；当前4.39ms仅纯搬运，不能与计算简单相加。所有MoE时延包含router/shared，排除attention/GDN、SSD、载入与离线打包，专家仍驻留DRAM。新bank按40层同尺寸约11.29GiB，仍超过本机约10.86GiB总RAM，未解决整模外存换入。

完整方法、复现命令、候选、正式样本和SHA在`llama-output/qwen36-moe-weight-io-fast-20261018/README.md`、`source/`、`RESULTS.json`、`MANIFEST.json`。代码保留在独立实验副本，未接入正式llama.py、改其他模型默认值或提交推送。

### AR128 MoE 参考 Houmo 路径（2026-10-18）

检查本地Houmo Qwen3.6-35B-A3B包装、AutoRound/GPTQ workflow及hmquant 1.4.0 wheel。其`MoeBlock`自定义OP携带三组专家qweight/scale及部分和格式，参考实现按专家分组、输入一次量化、grouped GEMM内部处理组scale并重排输出。可读Triton核为CUDA数值模拟，不能当作M50原生NPU内核移植或据其速度推断Hexagon性能。M50官方为最多双IPU、每IPU四tile，峰值153.6GB/s、最大48GB内存；这些规格及官方809-token示例与本手机AR128单层计时口径不同，没有取得同口径5ms的对应实测。

真实首块213专家平均4.8075条路由，53专家仅1条，129专家至多4条。当前32行HMX tile的未加权M维有效比例15.02%，不是NPU整体利用率。按bank头及路由静态计数共131,296次group输出，对应256.4375MiB原始部分和VTCM写入；这不是DDR落盘或硬件计数器值。需继续研究矩阵核内部分组缩放和输出粒度。

尝试每worker两套raw缓冲/请求，使下一批HMX与上一批HVX缩放重叠，VTCM由7,579,648增为8,366,080byte。相同紧凑group128 bank、6 HVX、轮询开启、profile关闭，原/新交替两轮，每进程2次warmup和16次计时：合并中位数8.35495→8.81919ms，慢5.56%，不采用。未单独证明变慢全部源自哪一项。

另按专家token数选择现有compact INT4 HVX整数核或HMX。相同构建串行扫描，每档2次warmup、7次计时：HVX阈值0/1/2/4/8的中位数分别8.53542/8.58578/8.67089/9.34984/11.43177ms，未形成分流收益。阈值0也含新增activation code和预计算，不能与旧保留版当作完全相同路径。该探索不能证明专用小M核没有潜力。

两项候选及五档阈值均通过连续8块真实输入校验，输出和Top8 ID与原版逐字节一致，全部有限，SQNR保持21.5014dB。手机已恢复原保留版，仍约8.36ms；完整MoE含router/shared，排除attention/GDN、SSD和加载，未测整模PPL，5ms未达到。后续优先研究细组scale/累加融合，及用GPTQ/AutoRound恢复约5.70ms大K候选的精度；本轮没有完成后者，也不承诺能恢复。

详情、源码、正式/探索样本及SHA位于`llama-output/qwen36-moe-houmo-study-20261018/README.md`、`RESULTS.json`、`MANIFEST.json`，小M候选位于相邻`qwen36-moe-small-m-20261018/`。正式模型、量化配置及外部Houmo/llama.cpp/SDK未改，未提交推送。


### AR128 MoE：GPTQ、全局压紧与融合后续（2026-10-18）

本地Qwen3.6-35B-A3B真实第0层、AR128、256专家/真实Top8；SM8750/v79、6HVX、1HMX、8MiB VTCM。专家packed INT4、整数A8×N4 HMX，router/shared沿用FP16。完整MoE包含router、专家FFN及shared，排除attention/GDN、SSD换入、载入与离线校准；仍为DRAM驻留实验。

| 配置 | 完整MoE正式中位数 | 1024-token设备输出SQNR |
|---|---:|---:|
| 原group128压紧版 | 8.3610ms | 21.5014dB |
| group128无新增误差优化 | 8.1116ms | 21.5014dB |
| 原粗组GU1024/down512、MSE | 5.6973ms | 19.9256dB |
| GPTQ粗组+全局压紧 | 5.2552ms | 21.1760dB |
| 兼容原同分规则的Top8+输出融合 | 5.1296ms | 21.1760dB |
| 再合并GU提交、shared投影和路由，保留候选 | **5.0207ms** | **21.1759dB** |

正式采集为关闭profile、每进程2次warmup+16次计时、两个串行交替轮次，最终候选共32样本，平均5.0224ms、范围4.9476–5.1213ms。相对原8.3610ms降低39.95%；**未稳定低于5ms，不能用最低一次称达标**。精度独立用8块/1024 token按总signal/noise能量计算，非八个dB平均，未测整模PPL。

优化及验证：

1. 原细组路径取消冗余逐组HMX clear，v79探针验证compound after输出后无需再次clear；交替控制8.3383→8.2027ms。激活移除四份旋转副本，全局compact由2MiB降512KiB；固定×4折入FP16 scale，扩大N块批量8→16，工作区8,316,928bytes，同轮控制8.1937→8.1116ms。8块输出/路由逐位一致，作为细组保留路径。
2. 对粗组实现完整Hessian GPTQ，block128、无act-order、1% damping、W4[-8,7]/FP16 scale；每专家90%真实路由统计+10%均匀先验，down使用已量化GU/SwiGLU/A8输入。CPU4线程串行256专家，独立train1024/test1024，479.75秒，无GPU或整模并发导出。CPU固定FP32 router/shared 20.2774→21.6322dB；手机真实整数输出19.9256→21.1760dB，二者不混用。
3. down变单一512通道量化组后，可以跨原128通道边界全局压紧GU恒零通道，GU行/down列同步重排，不改变已有GPTQ code/scale；8块逐位一致。GPTQ组内bank273.2227→全局240.0508MiB，原细组bank289.0527MiB。首块实际213专家读取由原细组241.0896MiB降到最终199.6165MiB，含实际调用专家头，排除未调用空占位bank及约1.7KiB offsets；不是硬件总线流量计数。首块最大31路由/专家，无cap32尾块重读。
4. 每专家两段GU连续打包、一次HMX请求，独立scale不合并；8块逐位一致。新partial Top8遇同分/非有限值回原完整排序，保持原tie order；1000组有限/重复/零值+3组非有限回退探针通过。首版不同tie order改变过一处专家集合，已舍弃。
5. Top8/Gather/原数值softmax融合为一次调用；shared sigmoid×输出及最终相加融合；shared gate/up/标量门控投影并为1056行FP16 HMX。后者改变标量门控累加路径，8块相对原GPTQ输出最大绝对差0.00017658，SQNR21.175953→21.175911dB，路由ID不变；最终构建再次验证8块输出与该候选一致，全部有限。

最后轻量profile仅取两次计时的OPBATCH，7个DSP节点平均4.8740ms，专家pipeline约4.2655ms，占87.5%，其中仍混合DMA、输入打包、HMX、缩放及SwiGLU，尚未逐项拆分。profile数字未混入正式时延。较大N批量、较少worker、cap24/28没有稳定收益；最终batch4约5.019ms、batch2约5.100ms，保留8。成对处理输出行保持逐位一致但约5.055ms，未采用。独立Top8/shared候选正式5.0267ms，与融合路由仅差约0.006ms，主要收益不来自这项融合。

最终关闭profiling恢复保留候选，轮询占用一个Android CPU核；原无新增误差group128的8.11ms版本也保留。专家仍在DRAM，本轮未集成SSD；240MiB按40层同尺寸约9.38GiB，另有其他权重/runtime，不能认定整模可全驻留本机约10.86GiB RAM。单层较细组仍低约0.326dB，整模PPL及40层误差未验证。

完整方法、样本、源码路径、SHA与复现命令：`llama-output/qwen36-moe-final-fuse-20261018/README.md`、`RESULTS.json`、`MANIFEST.json`。GPTQ在相邻`qwen36-moe-gptq-wide-20261018/`，最终构建/bank/路由核在`qwen36-moe-merged-gu-20261018/`。独立实验未接入正式llama.py、未改其他模型默认配置、未commit/push。


### AR128 MoE 内部optrace与进一步优化（2026-10-18）

继续在真实第0层、AR128、256专家/Top8、SM8750/v79、6HVX/1HMX/8MiB VTCM上测试，仍为DRAM驻留的完整MoE FFN，包含router/shared，排除attention/GDN、SSD、载入。W4/A8及FP16 router/shared、GPTQ权重和量化网格保持不变。

新增400–409成对周期时间戳，覆盖GU/down打包、DMA pop等待、HMX请求等待、输出修正/缩放、SwiGLU、down量化、写回及HMX命令服务；旧300–308只是累计cycles，不当作绝对时间。修正旧MOE_WEIGHT_PACK的误名，它实际记录DMA pop等待。最终level3 trace每worker平均：原/新DMA等待1.776/1.800ms、输出后处理0.901/0.914ms、输入打包0.504/0.204ms、down量化0.071/0.034ms、SwiGLU0.259/0.257ms、HMX请求等待0.150/0.144ms。worker时间会重叠，不能相加；HMX服务区间不是硬件busy计数器。DMA仍是最大等待，没有宣称已经消除或测到性能极限。

同一最终构建和bank、关闭profiling，每进程2次warmup+16次计时，按0/63/127交替两轮：

| 配置 | 32样本中位数 | 平均值 | 范围 |
|---|---:|---:|---:|
| tuning0：关闭本轮优化 | 4.985364ms | 4.991647ms | 4.958125–5.042084ms |
| tuning63：打包/预取/量化准备/1D DMA | 4.771380ms | 4.776947ms | 4.743282–4.883282ms |
| tuning127：再加寄存器SwiGLU，保留 | **4.763151ms** | **4.767292ms** | **4.715781–4.849115ms** |

同构建下降4.46%，保留版本本轮32次全部低于5ms。上一轮5.0207ms为独立测试，不把它与本轮控制4.9854ms之间的状态/构建差异全部归因优化。计时重复首块128-token输入，数值用8块1024 token；所有消融和最终构建输出文件/路由ID按字节完全一致，全部有限，SQNR保持21.175911dB，无新增量化误差，没有重跑整模PPL。

保留实现：跳过不消费的HMX M行清零，被消费行及dummy字节仍写完整；GU结束后提前一次读取down完整payload，与SwiGLU/A8准备重叠；HMX消费N4后先复制最多2KiB metadata，再让下一批DMA提前复用大buffer；每原GU1024/down512量化组只求一次scale/倒数，按全局压紧有效前缀计算等价的带零尾max，跳过恒零尾通道及冗余gate/down清零；连续权重显式使用1D DMA；SwiGLU中间量留寄存器。量化grid复用单独没有明显收益，需与有效前缀处理配合。tuning63到127约8微秒，未单独称显著提升。

尝试down缩放时直接路由写回、释放每worker256KiB，再把权重buffer557056扩至819200bytes以整块预取多数GU；8块同样按字节一致，但探索约4.76–4.78ms，没有进一步收益，未保留。最终tuning127不启用这两项，仍用原buffer、cap32/batch8、6worker及轮询，profile0；实际权重读取仍约199.6165MiB payload+头，没有再次压缩bank。

Level3改变调度并打印大量日志，host sample会被放大，未混入上述正式统计。可回放的Chrome trace JSON、PNG/SVG时间线、正式全部样本、源码/二进制hash、配置位及复现命令位于`llama-output/qwen36-moe-optrace-20261018/README.md`、`RESULTS.json`、`OPTRACE.json`、`MANIFEST.json`。实验代码未接入正式llama.py或quant recipe，未commit/push。后续仍应优先少读DDR字节、改善传输/消费布局；SDK有DLBC weights图选项，但未验证当前不透明bank的编码/解码，未计入任何硬件压缩收益。

### AR128 MoE 专家调度与提前DMA：4.733→4.542ms（2026-10-18）

继续使用真实Qwen3.6-35B-A3B第0层、128token、256专家/真实Top8、SM8750/v79、6HVX/1HMX/8MiB VTCM；INT4/N4权重和动态A8整数HMX、FP16 router/shared、GPTQ量化网格均保持原样。计时包含完整MoE FFN图，专家驻留DRAM，排除attention/GDN、SSD、载入及离线打包。

同一最终构建、原bank、profile0，每进程2次warmup+16样本，按127/1151/17535/50303和反向顺序两轮交替：

| 配置 | 32样本中位数 | 平均值 | 最小–最大 |
|---|---:|---:|---:|
| 上轮算法tuning127 | 4.732604ms | 4.732832ms | 4.697968–4.781198ms |
| 专家按bank顺序调度tuning1151 | 4.563177ms | 4.563161ms | 4.507864–4.593385ms |
| **再将GU DMA移至输入打包前tuning17535，保留** | **4.542317ms** | **4.544240ms** | **4.502656–4.605000ms** |
| 再合并初始两批DMA，未保留 | 4.584871ms | 4.624678ms | 4.525000–5.038594ms |

本轮同构建中位数降低4.0208%/0.190287ms，保留配置32/32次低于5ms；trace之后再次确认9样本中位数4.544531ms。上一轮4.763151ms为独立测试，本轮控制4.732604ms与其分列，避免将状态/构建差异算成收益。时间重复首块真实128token，精度另测8×128token。

保留两点：原按路由token数降序的任务排列改成专家ID升序，在当前GU/down bank内对应offset递增顺序；仍用原子取任务和原路由表，结果写固定token/slot，最终求和顺序不变。原GU先打包A8再提交前两批权重DMA，改成先提交相同请求再打包输入，读取与打包重叠；batch8、原双buffer、原VTCM与读取字节量均不变。调度贡献约0.169427ms，提前GU DMA再贡献约0.020860ms。

修正纯DMA probe，只读取实际src0/src2 payload，不读四个空part及128-byte头。首块213专家实际payload为209258496bytes/199.564453MiB，连426个头199.616455MiB。6worker、128/256/512/672KiB chunk的9样本中位数分别3.207552/3.219896/3.207865/3.202969ms；最终构建按路由数/地址顺序纯DMA为3.212448/3.207396ms，基本相同。不能把完整图收益称为物理内存带宽提高，更符合传输/计算调度配合改善；未用PMU继续区分TLB、总线和VTCM冲突，也不把纯DMA当硬下界。

最终同构建profile3每worker累计平均，原/新DMA等待2.071/1.600ms、输出解包修正缩放0.919/0.919ms、GU打包0.119/0.124ms、down打包0.087/0.084ms、HMX请求等待0.138/0.151ms。worker时间重叠；profiling改变时序和日志成本，不能用DSP trace的4.366/3.833ms替代正式完整图结果，也不能把HMX命令服务当硬件busy。DMA仍是最大等待，输出后处理仍值得继续拆。

其他消融均未保留：按权重大小降序4.754ms；GU/down相邻存储约4.55–4.58ms；worker额外2/4/6KiB padding约4.56–4.57ms；缩放metadata/correction外提未稳定获益；最后一批GU缩放时提前down约4.564ms、与GU提前组合约4.548ms；合并初始两批请求正式反而较慢。同专家相邻bank的256个段与原bank逐字节一致，但没有因此保留新格式。

统计现有N4码，GU/down零码比例20.8797%/19.1612%；1bit零位图+非零4bit编码约4.165/4.234bit，反而更大。8值分组的3bit基础码+溢出位图估计3.6897/3.7906bit，未计metadata与对齐，未实现codec、不计解压或DLBC收益。本轮读取总字节没有减少。

30组探索/正式/最终确认运行各8块，共240个验证块的最终输出/路由ID与上一轮逐字节一致且全部有限。1024token pooled输出SQNR由原独立FP32/BF16评估继承，仍21.175911dB；首块analyzer为21.640311dB，是不同统计口径。未重跑整模PPL。最终设备恢复原bank、tuning17535=127+1024+16384、cap32/batch8/6worker、profile0及轮询，未接入正式llama.py/recipe，未commit/push。详细记录、全部样本、Chrome trace、时间线及SHA位于`llama-output/qwen36-moe-bandwidth-20261018/README.md`、`RESULTS.json`、`OPTRACE.json`、`MANIFEST.json`。

### AR128 MoE shared/routed重叠：4.545→4.288ms（2026-10-18）

继续使用真实Qwen3.6-35B-A3B第0层、AR128、256专家/真实Top8、SM8750/v79、6HVX/1HMX/8MiB VTCM。专家仍为原GPTQ packed INT4/N4和动态A8，router/shared仍FP16，原量化网格与N4 bank未更改。范围是完整MoE FFN图，权重驻留DRAM，排除attention/GDN、SSD、载入和离线GPTQ/打包。

同一最终构建/数据目录、profile0，三轮正序/反序/移位串行，每进程2次warmup+32次计时，各96样本：

| 配置 | 中位数 | 平均值 | P95 | 最小–最大 |
|---|---:|---:|---:|---:|
| 上轮算法，同构建控制 | 4.545391ms | 4.545715ms | 4.598281ms | 4.421094–4.766094ms |
| shared/routed重叠 | 4.342135ms | 4.341658ms | 4.389557ms | 4.216875–4.547500ms |
| 再加shared预取 | 4.310730ms | 4.308675ms | 4.334895ms | 4.203281–4.495000ms |
| 再加成对输出缩放 | 4.292839ms | 4.287749ms | 4.315821ms | 4.178386–4.363073ms |
| **再加参数专门化/输入交织打包，保留** | **4.287969ms** | **4.287757ms** | **4.335287ms** | **4.176719–4.510000ms** |

同构建降低5.6634%/0.257422ms，共480正式样本。保留配置96/96次低于5ms；trace之后恢复profile0，额外16次中位数4.289271ms。上一轮独立4.542317ms与当前同构建控制4.545391ms分列，不将差异归因优化。最后专门化相对成对版本只约5µs，属于同一性能档，不单独称显著收益。时间重复首块真实128token，精度另外覆盖8×128token。

主要结构变化：原shared GU/SwiGLU/down与routed分开执行，共7个DSP计算节点；新图保留router和Top8，第三个CUSTOM完成shared/routed与合并，共3个DSP计算节点，optrace已确认。worker0先做shared，worker1–5同时做routed；shared结束后worker0加入专家任务队列。单个HMX服务在FP16和整数请求之间交替，分别加载bias并清累加器；不是两路MatMul同时占用同一HMX。重叠的是DMA、HVX准备/后处理及worker任务。

原每worker的integer raw预留128KiB，batch8/coarse2最多只消费32KiB，故每worker回收96KiB，6个合计576KiB。加原空余VTCM用于shared：512KiB FP16 activation、256KiB权重双buffer、8KiB raw、66KiB的16行GU暂存、256byte bias。GU阶段后，activation区复用为128KiB down activation和门控。新总VTCM8343808/8388608bytes，剩44800bytes，无越界扩容。

shared GU用两个128KiB buffer预取，down在相同256KiB区域容纳8批32KiB预取。SwiGLU到down的中间量留VTCM，shared down raw转FP32后直接乘门控，Top8求和与shared相加并入原pipeline combine。shared权重只作FP16 crouton无损重排，反解逐bit验证；虽然用I8 tensor装字节容器，里面仍为原FP16，不是INT8计算。专家N4读取字节不变。

成对后处理用INT16无饱和环绕减法代替INT32修正再截回INT16，两行scale共用一次64-lane FP16乘法。输入先交织两行U8，再扩展生成原`[0,u0,0,u1]`。此前28组局部探索均8块字节一致，但单独端到端收益很小：跨专家预取、大buffer无稳定收益，5/4worker约4.61/4.82ms。此次主要收益来自shared/routed重叠，仍用6worker。

曾定位并修复算子内部数据可见性：GU cached store写DDR后，后续cache-bypassing DMA读取了未提交内容。独立GU诊断与CPU FP16参考96.18dB。mode1/2显式DMA写回，mode4在GU全部写完后仅清理540672byte GU范围的cache，再DMA读回；检查API状态。修复前serial-v1/overlap-v1数值无效，未计入正式成绩。GGML builder的9参数限制通过显式填写已有src[9]处理，tensor/HTP协议原本支持10槽，没有改外部ABI。mode3仅诊断，不是有效FFN配置。

重新使用保存的独立FP32/BF16参考计算1024token pooled最终输出SQNR：旧21.175911028dB，新21.175912788dB，差约0.00000176dB，不称精度改善。新旧输出之间96.434389dB、最大绝对差4.87417e-5，**不是新旧逐字节一致**；全部路由ID与旧版逐字节一致。最终新serial/overlap/prefetch/后处理各配置之间的8块输出逐字节一致，均有限；没有重新校准或重跑整模PPL。正式15组各8块，共120验证块，均重复同一组1024真实token。

trace确认worker0的shared区间与其他5worker重叠，之后加入routed。profile3原/新每worker输出后处理0.938/0.457ms、GU+down打包0.211/0.133ms、DMA等待1.576/1.925ms。计算变快可更早到等待点，不能把等待增加当带宽下降。baseline CUSTOM不含图外shared，新CUSTOM含shared，窗口范围不同；profiling扰动调度，shared区间约1.83/2.21ms不可当profile0时延，worker时间不可相加，HMX服务区间不是硬件busy。DMA仍最大，没有宣称达到极限。

最终设备恢复shared_mode4、tuning6571135、cap32/batch8/6worker、profile0和op-poll；轮询仍占用一个Android CPU核。实验源码在`llama-output/qwen36-moe-shared-overlap-20261018/source/`，正式llama.py/quant recipe未修改，未commit/push。复现命令、全部正式样本、误差、Chrome trace、时间线和SHA见该目录`README.md`、`RESULTS.json`、`FORMAL.json`、`OPTRACE.json`、`MANIFEST.json`。


### AR128 MoE输出DMA/VTCM汇总：4.289→4.095 ms（2026-10-18）

延续上一轮真实Qwen3.6-35B-A3B layer0，AR128、hidden2048、FFN512、256专家/真实Top8。SM8750/v79/431e3c37，6 HVX、单HMX、8 MiB VTCM，原MAX vote、op-poll和权重bank不变。时间为完整MoE FFN（router/route/shared/routed/combine），不含attention/GDN、SSD/装载、离线GPTQ；专家仍常驻DRAM。正式llama.py/quant recipe没有整合，未commit/push。

三轮交错、每轮32次，加2次warmup，每配置96次，合计384正式样本；保留方案96/96低于5 ms。old为上一轮原二进制，control为本轮同构建关闭新逻辑。

| 配置 | 中位数 | 均值 | P95 |
|---|---:|---:|---:|
| old | 4.288724 ms | 4.292498 ms | 4.384362 ms |
| control | 4.293542 ms | 4.288837 ms | 4.321445 ms |
| token | 4.092422 ms | 4.114804 ms | 4.237331 ms |
| expert | 4.094791 ms | 4.094981 ms | 4.133672 ms |

保留expert-major布局，较原二进制降低 **4.5219% / 193.93 µs**，较同构建control降低4.6291%。token/expert中位数相差仅2.37 µs，属于同档，expert均值/P95更低。所有单点原样保留，未删长尾。正式时间重复首块128token，精度另跑8×128=1024不同token，各轮重复这组数据。

保留三项：

1. shared GU的128×1056 FP32结果放到worker0尚未启用的routed VTCM工作区。输入准备barrier之后原FP32输入暂存已无读者，shared结束后worker0才加入专家任务。省掉540672byte GU中间量DDR往返、range cache clean和重读。
2. routed down结果在原VTCM中原地乘路由score，通过DMA按expert-sorted route索引连续写8 MiB DDR bank；每chunk完成DMA后再复用工作区。建立token/slot→route position反向索引，保留原真实Top8和分数。route bank首次分配flush/invalidate，之后DMA读写、不经CPU缓存。当前隔离图没有其他MATMUL_ID复用这个通用ddr_spad；未来整合需明确独立生命周期。
3. 汇总前按反向索引把8路8 KiB结果DMA到连续VTCM token槽，再按slot0到slot7、最后shared的原顺序求和。每worker16槽/1 MiB，初始128 DMA descriptor<队列256；消费后环形复用。展开8个独立向量，但每元素加法顺序不变。

仍为原N4×动态A8整数HMX，GU1024/down512网格、FP16 router/shared和FP32 SwiGLU不变，专家实际权重读取约199.56 MiB/首块。总VTCM仍8343808/8388608byte，余44800byte；shared与combine复用不同生命周期的原worker区，不新增VTCM分配。

1024token最终输出及路由ID与上一轮4.288ms保留版本**逐字节一致**，全部有限；12正式运行均通过，独立参考pooled SQNR仍 **21.17591278795492dB**。与更早4.54ms旧输出仍是96.434389dB、max abs4.87417e-5，不能把这个跨两轮比较说成字节一致。没有重跑整模PPL或重新校准。

新增413号MOE_COMBINE_EXEC trace确认：profile3同构建控制/保留的pipeline CUSTOM（两者均含shared）4.0035/3.7820ms，worker平均combine0.3392/0.1549ms，store（含等待）0.0590/0.1709ms；仅worker0 shared完整区间2.2930/1.8885ms。DMA store显式等待提交，cached store返回较早，不能单看store变长判定更慢。worker时间不能求和当整层时间，HMX命令服务不是硬件busy，没有硬件总线字节PMU，不宣称量到了RFO或固定带宽提升。端到端收益以profile0正式结果为准。

本轮67组探索中，shared请求合并、缩放融合SwiGLU、第三GU buffer、大shared预取、stride padding、L2fetch、token就绪提前combine/跨专家预取都没有稳定大收益。FP16双行sigmoid使SQNR变为21.173134dB、旧新输出43.670164dB但没有明显性能优势，未保留。当前只保留有稳定收益且输出一致的路径，不称到达硬件极限。

最终配置：shared_mode4/tuning6571135/shared_local1/combine68/combine_window16，cap32/batch8/6worker、profile0、op-poll。trace后恢复配置再测16次，中位数 **4.088515ms**，8块输出再次一致。

复现与证据目录：`llama-output/qwen36-moe-shared-batch-20261018/`，见`README.md`、`RESULTS.json`、`FINAL_FORMAL.json`、`OPTRACE.json`、`trace-*.chrome.json`、时间线、精度bin及`MANIFEST.json`。源码关键函数为`source/shared-async.h::moe_shared_compute`及`source/expert-pipeline.h::moe_pipeline_worker/moe_combine_fetch/moe_pipeline_combine_tokens`。


### AR128 MoE VTCM FP32累加：4.088→3.844 ms（2026-10-18）

与上一轮相同统计范围：Qwen3.6-35B-A3B真实layer0，AR128、hidden2048、FFN512、256专家/真实Top8；含router、route/softmax、shared/routed FFN及合并，不含attention/GDN、SSD/装载、离线GPTQ。专家仍常驻DRAM。SM8750/v79/431e3c37，6 HVX、单HMX、原MAX vote/op-poll不变；每次计时覆盖graph_compute+synchronize。正式llama.py/quant recipe未整合，未commit/push。

| 配置 | 中位数 | 均值 | P95 |
|---|---:|---:|---:|
| old | 4.088020 ms | 4.090735 ms | 4.172734 ms |
| control | 4.102917 ms | 4.101393 ms | 4.139505 ms |
| accum19 | 3.862761 ms | 3.860546 ms | 3.916796 ms |
| packed22 | 3.844115 ms | 3.843725 ms | 3.891796 ms |
| packed23 | 3.847474 ms | 3.847853 ms | 3.886602 ms |
| packed24 | 3.855599 ms | 3.853530 ms | 3.923828 ms |

每配置3×32=96正式样本，合计576次。顺序为六配置正序/反序/从packed22旋转；先2次warmup，未删除长尾。old为上一版原二进制本轮重测，control为新构建关闭新功能。保留packed22，相对原二进制降低 **5.9663% / 243.91 µs**，相对同构建控制降低6.3077%。保留配置P95 3.891796ms、最大4.023594ms，不称每次低于4ms。

主要修改：把原8 MiB route DDR bank写出/读回改为1 MiB `[128,2048]` FP32 VTCM累加；专家仍完整贡献8路score×down，按到达顺序累加，最后加shared。每token一把QuRT PI mutex，保护完整2048维更新，任何时刻只持有一把。128锁约2048byte在context DDR heap中，首次使用初始化后复用，context teardown逐个destroy/free，不占VTCM。

为容纳累加区，先减小每worker chunk，继而无损去重重复scale：输入每1024维原grid只存一次，128token compact从524288降到294912byte；down每512维从1024降到640byte/row。Q8码、原scale、maximum、inverse及舍入过程不变。另回收shared不活跃的66 KiB GU暂存，把bias紧跟raw；在严格HMX/寄存器SwiGLU guard内移除不用的scalar partial暂存。单独packed布局、关闭累加的cap32/cap24对照，8块输出和路由均与旧版逐字节一致。

最终cap22 worker stride1013760byte，VTCM：packed input294912+integer bias2048+6worker6082560+shared794880+FP32 sum1048576=**8222976/8388608byte**，剩**165632byte**。cap23相近速度但余量79616，cap24余5888且未更快，保留cap22。首块213个活跃专家、最多31路/专家，cap22只让该31路专家分成22+9；推算N4 payload从199.564453增到约200.703125MiB（额外1.138672MiB），仍通过移除16MiB route中间量往返获得净收益。该payload是按descriptor/bank长度计算，非硬件字节PMU。

仍是原packed N4/INT4×动态A8整数HMX、GU1024/down512网格，router/shared FP16矩阵乘和原FP32 SwiGLU没有降精度；没有减少Top8或忽略小score。

本轮输出**不是字节一致**，FP32求和顺序变化且最低位可能随调度变化。18正式进程各8块，合计144验证块重复同一组1024不同token，全部有限、ID与上一版逐字节一致。保留方案对上一版输出 **143.653344–143.664645dB**，max abs **1.49011611938e-08**。独立参考pooled SQNR约 **21.1759128dB**，旧21.17591278795492，约1e-8dB变化不称精度改善。更早4.54ms版本与当前仍约96.4343dB，不混为当前比较。未重跑整模PPL。

同步演化必须保留记录：早期交换/TTAS自旋锁曾在cap23/24第7个验证块等待，首块timed能完成，故先前自旋版本时延均不作为最终有效结果。TTAS曾一次通过8块，正式复测仍复现等待；同cap24 packed但关闭累加，8块输出/ID完全与旧版一致。最终改用SDK `qurt_pimutex_lock/unlock`，争用时由QuRT挂起等待者并提供优先继承，三轮全部576计时和144验证块未超时。现有证据不能证明旧自旋停顿的全部底层原因。失败源代码/二进制/未完成正式数据归档为`FAILED_SPINLOCK*`，与保留PI结果分开。runner记录自身PID，60秒超时只清理对应PID。

新profile3控制/累加CUSTOM总区间3.7655/3.7645ms，接近；worker平均store（含DMA或mutex/累加）0.16031/0.07127ms、最终combine0.15672/0.03842ms、专家DMA等待1.80577/1.94779ms。局部计算缩短、等待增加会改变调度，不能简单求和当整层收益。约6%收益以576次profile0交错复测为准，不以两次trace总区间推导。HMX服务范围不是硬件busy，也没有读取总线字节PMU。

SIMD批量Top9/shared HMX优先调度没有稳定收益，保持关闭。最终配置cap22/shared_mode4/shared_local1/tuning6571135/combine0/accumulate/pack-grids，GU双buffer/batch8/6worker、原Top8/FP32非线性、profile0/op-poll。trace后额外16次中位数 **3.848229ms**，1024token相对上一版143.657147dB、max abs1.49011611938e-08，路由再次一致。

完整复现和证据：`llama-output/qwen36-moe-route-schedule-20261018/README.md`、`RESULTS.json`、`FORMAL_ACCUMULATE.json`、`OPTRACE.json`、`trace-*.chrome.json`、时间线、精度bin及`MANIFEST.json`。代码核心为`source/expert-pipeline.h`的累加/PI锁/内存guard、`source/group-quant.h`和`expert-hybrid.h`的无损布局、`shared-async.h`的活跃区复用及`prepare_pipeline.py`的上下文锁资源生命周期。

### AR128 MoE 2ms目标复核：权重搬运约3.20ms，保留原版（2026-10-18）

本轮未达到2ms，正式llama.py/recipe未修改，设备恢复上一轮实际二进制。真实Qwen3.6第0层、128token、256专家/Top8，INT4×动态A8、原量化网格、FP16 router/shared、原FP32 SwiGLU；包含完整MoE FFN，专家DRAM常驻，排除attention/GDN、SSD/装载。45组完整验证运行、832正式计时样本、360验证块重复同一组1024不同token，全部有限、路由ID一致；对上一轮输出最大绝对差7.4505805969e-09，独立参考SQNR21.1759127953–21.1759128000dB。没有重测整模PPL。

探索缓存DMA、L2fetch、HVX直接加载、跨专家GU预取、额外DDR/总线投票、3–6worker与完整GU权重缓冲、二维DMA、FastRPC uncached及无损gate/up成对布局。最终实际旧二进制/成对候选各64样本交错对照，中位数3.848282/3.838750ms，均值3.841824/3.838469ms，P95为3.874140/3.902678ms；均值只差3.35µs且候选尾延迟更高，未采用。本轮另一次原二进制64样本中位数3.780365ms，后续控制回到约3.85ms，没有将进程/设备波动算代码收益。

纯IO只读真实首块213专家各一份GU/down payload，共209258496byte=199.564453MiB；6worker两轮合并18个qtimer样本中位数3.203802ms、65.315675GB/s，2/3/4worker约3.198/3.199/3.205ms，二维DMA更慢。该结果是当前路径测量，不是硬件理论下界，也不能从完整图时间减去它当独立计算时间。

保留cap22因一个31-route专家重读，payload为210452480byte=200.703125MiB。在2ms读完仅该payload需105.22624GB/s；若吞吐保持本轮测量，则需把读取量降至约124.58MiB（至少少37.93%），尚未包括其他传输/解压/计算。host逐专家无损压缩并解压校验，最佳Zstd9实际加权payload172223069byte=164.245MiB（原81.835%），按同吞吐只读取仍约2.6368ms；没有实现设备解压，不能称NPU优化成绩，也不能据此证明所有无损编码的压缩率上限。当前尚无验证通过的2ms方案。

完整记录：`llama-output/qwen36-moe-transfer-20261018/README.md`、`RESULTS.json`、`FINAL_CROSSCHECK.json`、`IO.json`、`LOSSLESS_COMPRESSION.json`及源码/原始bin/日志/`MANIFEST.json`。实验副本保留用于复现，未commit/push。

### AR128 MoE 无损软件压缩与 QNN DLBC：未获得收益（2026-10-19）

继续使用真实 Qwen3.6-35B-A3B layer0、AR128、256 专家/Top8 的完整 MoE FFN 口径；SM8750/v79/431e3c37，专家 DRAM 常驻，排除 attention/GDN、SSD/加载与离线 GPTQ。原 N4×动态 A8、GU1024/down512 网格、FP16 router/shared、FP32 SwiGLU不变。未减少 token、Top8、FFN 宽度或量化位数，正式 `llama.py`/recipe 未整合。

C3 以 8 个 INT4 码为单位，在可无损表示为 24 bit 时存 3 byte，否则存原 4 byte，加 flags；GU 分批、down 整份 DMA 后在 VTCM 原地反向展开。全 bank host 独立/原地解码均逐字节还原，256 专家/2777 chunk 通过。真实首块按 cap22 次数加权，payload 从 210452480 降至 196265472 byte，仅少 6.741%。设备完整图 8 次中位数 **38.646485 ms**，远慢于原版。1024-token 输出有限、路由 ID 与旧版一致，max abs 7.45058e-09、对旧输出 SQNR155.880252dB、独立参考21.1759127974dB。profile3 两次 pipeline 平均38.2665ms，worker平均解码36.4693ms；反汇编有大量VLUT/位操作和寄存器溢出到栈，软件解码成为瓶颈。线程时间不能相加，DMA等待降低不等于带宽提升，失败方案未采用。

独立检查 SDK 的 `ENABLE_DLBC_WEIGHTS`（类型5，与现有 use_dlbc 类型4不同）：真实 GU 权重排除全零输出通道后，拼为 M/N/K=8/32768/1024 的 QNN FullyConnected。使用 QNN 自己的 W4/A8 校准：权重是 SFIXED8 容器加 BW_AXIS bitwidth=4、离线 context 约16MiB，未改成 W8。此微基准不是完整 MoE，也不与原动态 A8/分组网格宣称数值等价。SDK2.45的权重压缩前/后均17039360byte、压缩比100%；context从17117184增至17391616byte。手机off/on/on/off每组68次，丢弃前4次warmup，各配置128正式样本；accelerator execute中位数 **286/292μs**，原生UINT8输出逐字节一致，没有收益。2.45同时开启input/weights DLBC仍无压缩；2.49额外离线off/weights/both context也未缩小，后者未上机计时。没有给现有GGML DMA直接套QNN开关或集成原生MoE。

128-byte码块的离线hash字典分析，即使假设无限缓存、无查表/索引成本，真实路由最多省3833728byte（1.82166%），主要是down零块/同一专家重读；未实现设备字典，不是性能成绩。

最后恢复上一轮实际二进制、原bank、cap22/accumulate/pack-grids、profile0/op-poll。32次完整图中位数 **3.842396ms**、均值3.839046ms、P95 3.864690ms；此为旧版复测，不称新优化。恢复后1024token全部有限、ID一致，对旧输出159.441519dB、max abs7.45058e-09，独立参考21.1759127977dB。**2ms未达到，本轮未采用新候选，未重测整模PPL，未commit/push。** 详细格式、原始样本/输出、SDK配置与profiling CSV、源码/实际二进制哈希见`llama-output/qwen36-moe-c3-20261019/README.md`、`RESULTS.json`、`MANIFEST.json`。

### 真实 GDN attention 接完整 MoE：AR128 首层 11.278ms（2026-10-19）

使用真实 `/hunter/model/Qwen/Qwen3.6-35B-A3B` 第0层：hidden2048、GDN K/V heads16/32、head dim128、256专家/Top8、FFN512。SM8750/v79/431e3c37，8MiB VTCM。一个Android进程中串联当前ExecuTorch优化attention的QNN context与上一轮实际最快GGML/Hexagon MoE，包含input RMSNorm、完整GDN attention、第一次residual、post-attention RMSNorm、完整MoE、最后residual、runtime交接及同步。不是独立耗时相加，也不是统一QNN图或完整35B PTE。专家DRAM常驻，不计embedding、SSD/加载、其他层、视觉、最终模型norm/LM head。

RTX3090上CUDA校准1024个WikiText train token、关闭TF32，out_proj fixed-grid GPTQ。最初沙箱看不到NVIDIA设备节点，误判主机驱动不可用；主机CUDA实际tensor运算通过后，已用GPU重新导出正式图，早期CPU图未用于成绩。attention为K/z及depthwise conv W8，其他投影W4/group32，A16，chunk内部FP16；conv/recurrent边界UINT16，交给原MoE的normalized input/residual为FP32。MoE保持原INT4/N4×动态A8、GU1024/down512网格、FP16 router/shared、FP32 SwiGLU、cap22/accumulate/pack-grids，DSP二进制未改。

3轮各32次、每进程6次warmup，共96个完整层正式样本，全部保存：

| 同一次完整层调用的实际区间 | 中位数 | 均值 | P95 |
|---|---:|---:|---:|
| attention及两侧block RMSNorm、第一次residual | 6.316953ms | 6.246650ms | 6.519570ms |
| MoE及末尾residual | 4.985521ms | 5.013232ms | 5.122032ms |
| **完整第0层** | **11.277839ms** | **11.259882ms** | **11.600169ms** |

均值占比55.48%/44.52%；各区间中位数不可严格相加。逐轮完整层中位数11.313100/11.166667/11.281407ms，全部样本最小9.339218、最大24.534167ms，未删除第三轮长尾。单独连续运行控制各24次：attention中位5.861016ms，MoE+final residual中位4.111953ms。控制不逐次交替资源，不能相加冒充完整层，也不能将串联MoE区间4.99ms全部算作原3.84ms算术内核回退。默认QNN priority100曾造成约21ms的早期串联等待，正式使用公开QNN context priority200。QNN RAW IO可能有内部拷贝，未宣称零拷贝。

PTE仅包含一个QNN delegate、零CPU fallback；native context IO顺序由PTE DelegateCall自动映射。显式保留GGML输入/残差，防止最后add覆盖normalized input；早期有输入复用问题的MoE-only控制已排除。正式3进程与2控制进程的attention normalized output逐字节一致；首块实际触及208专家，cap22 payload207447040byte，区别于旧FP32 attention输入的213专家。

**精度仍有明显问题，不能称精度已解决。** 首块完整层hidden相对真实FP32 attention+MoE参考SQNR **13.004234dB**；attention residual14.895448dB、normalized MoE input11.631490dB。固定设备真实attention输入后，MoE output对FP32 MoE为20.842222dB，conditional layer30.698880dB，说明显著误差已在attention前端出现。8个独立cold-state AR128块共1024不同token全部有限，最后residual逐元素完全正确；conditional MoE output SQNR19.631–20.842dB，FP16 router对FP32的Top8 rank agreement97.56–99.32%，不宣称ID bitexact。未测连续warm-state前填或整模PPL，没有把单层时间乘40冒充整模测量。

源码、GPU导出、完整层runner、正式样本/输出及哈希见`llama-output/qwen36-full-layer-20261019/README.md`、`RESULTS.json`、`ACCURACY.json`、`MANIFEST.json`。正式`llama.py`/recipe/MoE内核未整合修改，未commit/push。

### 35B-A3B attention分段：共享IO与RPC修正，完整层11.294→7.982ms（2026-10-19）

对上一轮同一attention context进行基础/详细profile，并保留冻结量化参数提取五个子图。SM8750/v79、431e3c37，真实第0层AR128、hidden2048、GDN value heads32、Top8 MoE，原权重bank和DSP库不变。

此前6.32ms是QNN直调RAW IO原型的调用墙钟，包含较大IO/RPC开销；同图加速器段约3.4–3.5ms。独立attention三轮交替、每轮warmup20/96次：RAW5.785912ms→rpcmem+memRegister4.174844ms→共享IO加ExecuTorch Burst同款control100us/poll9999us配置 **3.448906ms**。每配置288样本，四个attention输出逐字节一致。

完整层同一个二进制三轮交替A/B，每配置96样本、每轮6次预热，包含两份FP32 hidden复制、runtime切换和同步：

| 配对完整层区间 | RAW原配置 | 共享IO+RPC | 中位数下降 |
|---|---:|---:|---:|
| attention+两侧norm+首次残差+边界复制 | 6.353698ms | **3.924037ms** | **38.24%** |
| MoE+最后残差 | 4.987136ms | 4.065339ms | 18.48% |
| **完整层** | **11.293594ms** | **7.982031ms** | **29.32%** |

五个独立子图的基础profile加速器段：input norm+投影0.5800ms，conv/布局/SiLU/QK归一化/gates准备1.2300ms，GDN core1.5365ms，norm/z门控0.2890ms，out_proj+残差+post norm0.2100ms。分图改变IO和融合边界，不能把子图相加冒充整图精确分账。整图详细profile映射确认较重的非core节点集中在conv两侧Transpose、cache拼接后的Slice、conv输出布局和大激活SiLU链；详细profile周期不换算为正常运行毫秒。

历史0.832ms对应0.8B/H16独立core及后来发现非有限值的dense32三角逆；当前稳定H16旧基准为0.892334ms。35B的H32 state元素数翻倍，这与额外IO开销是两个不同问题。

直接跨QNN/GGML注册同一fd发生fastrpc映射冲突，失败退出未计成绩；最终用独立QNN共享buffer，再显式复制共2MiB的normalized/residual到GGML，复制在计时内，不宣称零拷贝。三轮四个attention输出逐字节一致，最后残差逐元素正确；完整层对原量化版SQNR162.01–164.52dB、最大绝对差2.98e-8。原对全FP32参考约13dB的精度问题没有在本轮修复，也没有重测PPL。完整层仍未达到5ms；未测40层/SSD换入。

独立原型和完整证据：`llama-output/qwen36-attention-breakdown-20261019/README.md`、`RESULTS.json`、五个子图PTE/输入/profile、`full-qnn-audit.json`、`full-node-sources.json`、`layer-ab-*`。生产`llama.py`、量化recipe、attention/GDN数学实现与MoE库未修改，未commit/push。

### 真实前四层AR128：三层GDN＋一层FA、四套独立MoE（2026-10-19）

独立实验，未接入生产。SM8750/v79、431e3c37；本地Qwen3.6-35B-A3B真实层索引0/1/2/3，各自attention、router、shared和256专家bank。四套bank在同一GGML backend中同时驻留，四套QNN context常驻；真实层输出直接送下一层，不在中途换成参考hidden。沿用原最佳N4×动态A8整数HMX库及共享IO/RPC配置，cold AR128，排除加载和SSD换入。

最终同一二进制3个epoch，每epoch6次预热、32次正式计时，共96次四层调用：

| 层 | attention | attention含两侧norm/首次残差 | MoE含最后残差 | 完整层区间 |
|---|---|---:|---:|---:|
| 0 | GDN | 3.930860ms | 4.051979ms | 7.993073ms |
| 1 | GDN | 3.889844ms | 6.200703ms | 10.130469ms |
| 2 | GDN | 3.921433ms | 5.829063ms | 9.794297ms |
| 3 | full attention | 1.599896ms | 5.718281ms | 7.367553ms |

**四层整体中位35.312605ms，均值35.392557ms，P95 35.893593ms**，三个epoch中位35.316615/35.307683/35.313203ms。整体中位÷4为8.828151ms；各区间中位数不可严格相加。跨层hidden传递平均合计约0.110009ms，计时内包含QNN/GGML交接、显式复制及同步。第3层FA的FP32实现对Transformers SQNR132.596891dB，确认了partial RoPE、Q/K norm、query gate和因果mask。

MoE切换控制：实际量化链输入下，各层MoE独立运行中位数合计20.735417ms，四bank交替整体21.077630ms，相差0.342213ms/1.65%。额外使用各层真实FP32参考hidden重测：独立中位数合计20.270000ms，交替整体20.601250ms，相差0.331250ms/1.63%。差额包含缓存/调度/频率等影响，未声称是硬件分离出的纯bank切换；当前驻留四层未发现毫秒级放大。

实际bank含router/shared分别247.176/391.832/393.528/394.753MiB，四套权重缓冲合计1496620544B。量化后第0层零通道紧凑化省掉的权重更多，不能代表后续层。量化链实际选中208/215/193/177个专家，按cap22和header有效payload推算每块权重207447040/339720192/314382848/302657024B，不是DDR硬件流量计数；后续层负载明显更重。

四层bank与各自原生N4解码模拟匹配，MoE条件SQNR29.44/36.85/33.44/33.82dB，所有残差逐元素正确、输出有限。**精度尚不合格**：第1层GDN normalized对同输入FP32 attention仅−1.37545dB，四层累积hidden对FP32链约−0.044dB；该问题不是bank错绑，FP32输入的切换对照单独验证了性能结论。没有测PPL、40层、SSD换入、长历史cache或decode。

产物及复现：`llama-output/qwen36-four-layer-20261019/README.md`、`RESULTS.json`、`ACCURACY.json`、`FA_PARITY.json`、`MANIFEST.json`、逐层PTE/bank、`formal.log`、`teacher.log`。首轮旧二进制35.274063ms记录保存在`initial-formal/`，未混合进最终样本。生产`llama.py`/recipe/GDN/MoE内核本轮未改动，未commit/push。

<a id="qwen36-layer1-tiny-scales-20261019"></a>

### 第 1 层 GDN：极小权重 scale 引起设备端饱和（2026-10-19）

这里的第 1 层指 **layer index 1，即第二个 Transformer block**。使用 Qwen3.6-35B-A3B 真实权重，固定前述四层量化链送入该层的同一份 `[1,128,2048]` hidden，分别运行 FP32、冻结 PTQ Q/DQ 图、替换 ChunkGDN 后的 Q/DQ 图和手机 QNN context。SM8750/v79，设备 431e3c37；Q/V/a/b 和 out_proj 仍为 W4/block32，K/z、depthwise Conv 仍为 W8，conv/recurrent state 的外部 IO 仍为 UINT16。

#### 如何定位

同输入下 PyTorch 冻结 PTQ 图的 normalized SQNR 为 **20.12224dB**；替换 ChunkGDN 前后 normalized SQNR 为36.91552dB、state 为53.55317dB。但手机 normalized 对 FP32 只有 **−1.37545dB**。独立 core 使用同一组冻结 Q/DQ 输入时，设备 attention output 对原 recurrent Q/DQ 为59.85435dB、state 为52.94678dB。严重误差发生在设备执行路径，不能归因于普通 W4 PTQ 误差或直接判定三角逆失效。

继续把原量化图拆成输入投影、conv/QK/gates、core、norm/z、out_proj 五段，保留原校准参数。投影的第318、337输出通道在 PyTorch 量化结果中为零，手机却出现约−41.85754/+41.97395的饱和值。对应的投影与 depthwise Conv 权重 scale 约1e−9～1e−7；例如 K/z 的第318通道为4.54661e−9，第337通道为1.52590e−9。

给完整图额外添加中间输出后，修复前、修复后各自的四个最终输出与无额外输出的图**逐元素相等**，确认观察点没有消除原问题。完整图中按相同 FP32 输入比较：

| 位置 | 修复前 SQNR | 修复后 SQNR |
|---|---:|---:|
| Q/V/a/b 投影 | 4.32539dB | 24.26779dB |
| K/z 投影 | 4.87412dB | 45.15099dB |
| depthwise Conv 输出 | 2.90762dB | 34.30939dB |
| normalized query | 10.03998dB | 25.13047dB |
| normalized key | 10.51241dB | 41.86867dB |
| GDN attention output，门控前 | 28.14454dB | 30.47286dB |
| RMSNorm/z 门控输出 | −15.09651dB | 13.97249dB |
| out_proj 输出，残差前 | −11.87427dB | 16.02754dB |
| 残差及 post norm 后 normalized | −1.37545dB | **20.08072dB** |

GDN attention output 的全头 SQNR 掩盖了低能量头的异常：head4/5 的 FP32 RMS 分别为1.68547e−5/1.87366e−5，错误设备结果为1.02220e−3/1.03158e−3，两个头分别仅−35.55/−34.71dB。后面的 RMSNorm/z 门控把这些错误进一步放大，**head4/5 占门控总误差能量的99.894%**。修复后这两个头的 core RMS 恢复到1.73251e−5/1.91789e−5，gate RMS 也恢复到接近参考值。

#### 修复与消融

在 `quantize/qwen35.py::stabilize_gdn_weight_scales` 中，只处理 GDN 的 `in_proj_qvab.conv`、`in_proj_kz.conv` 和 `causal_conv2d` 冻结权重。对过小的权重 scale 使用 `torch.finfo(torch.float16).tiny`（2^−14）作为安全下限，并根据原已量化权重重新计算整数码：`new_code = round(old_code * old_scale / safe_scale)`，随后按原 W4/W8 范围截断。**不能只改 scale 而保留整数码**，否则会把这些小权重放大。该下限是本设备验证有效的数值保护；QNN 内部闭源，未把具体某条 FP16 或定点指令的下溢宣称为已证实原因。

修复放在 PT2E 转换后、out_proj GPTQ 和共享 AR128/AR1 激活范围采集之前。激活校准的 scale/zero point、UINT16 cache 网格及权重位宽没有被此函数替换。对本层共修改7117个 scale（W4 block6656、K/z W8 channel113、depthwise W8 channel348）；正式函数与已验证 PTE 的全部冻结 buffer 完全一致，重复调用不再修改任何值，out_proj 权重及全部 activation/cache Q/DQ 编码保持原样。FP32参考下的 PyTorch normalized SQNR仅从20.12224降至20.11893dB。

| 修复范围 | normalized SQNR | conv cache SQNR | state SQNR |
|---|---:|---:|---:|
| 原版 | −1.37545dB | 7.85019dB | 30.96707dB |
| 仅输入投影 | 6.39848dB | 24.93823dB | 31.63989dB |
| 仅 depthwise Conv | 20.08067dB | 7.85019dB | 31.65221dB |
| 两处一起修复 | **20.08072dB** | **24.93823dB** | **31.65221dB** |

因此本轮当前128-token输出的主要异常在 depthwise Conv，输入投影的异常还会写入 conv cache，仍需一起保护。无需改变 GDN state 位宽或重写 core。

额外两组固定 hidden 及非零 state 验证，逐组都比较同输入 FP32 attention：

| 输入 | 原版 normalized | 修复后 normalized |
|---|---:|---:|
| 原量化链实际输入，cold128 | −1.37545dB | 20.08072dB |
| 独立参考序列第128～255位置，cold128 | −0.97273dB | 20.62566dB |
| 独立参考序列第768～895位置，cold128 | −1.32846dB | 20.22309dB |
| 第128～255位置，承接首128步的非零FP32 conv/state | −0.96869dB | 20.82347dB |

非零输入单独编码后，conv 输入 SQNR70.09404dB、state 输入54.43325dB，两者超出量化范围的比例均为0；cold输入全零，不能用零信号的SQNR证明精度。该warm控制按输入各自的校准参数重新编码，没有验证直接复用原始UINT16缓存的长序列路径。本原型conv输入/输出scale分别为0.00127918646/0.000659527606，后续若直接回传原始缓存，需要检查编码对齐。

同一 hidden、cold cache、共享 IO/RPC poll、profile关闭，原版/修复版交替三轮，每轮64次、各192次正式样本：完整 attention front（含两侧 norm、out_proj、首次残差，不含MoE）中位 **3.480156→3.496693ms**，约+0.48%；P95 3.508953→3.532438ms。此处是独立 attention 对比，与上文四层共驻的3.889844ms不可混作同一组测试。修复没有增加运行时算子。

完整证据保存在 `llama-output/qwen36-layer1-fix-20261019/`：`RESULTS.json`、`HEADS.json`、`STAGES.csv`（FP32 min/max、量化范围、饱和比例、SQNR）、`PER_TOKEN.csv`（128个token逐位置SQNR）、`scale-floor/attention.pte`、原版/修复版中间输出，以及 `source/` 中的复现脚本。旧四层基准产物未覆盖。本轮修复了目标单层的严重错误；残留 W4/激活量化误差仍在，尚未据此宣称完整四层/40层或PPL已修复，MoE实验未整合进 `llama.py`，未commit/push。

#### 修复第 1 层后，完整四层 AR128 重测（2026-10-19）

按用户要求回到前四层真实链，**只替换 layer1 的修复版 attention context**；layer0/2/3 context、四套专家 bank、runner 二进制和运行参数沿用原版。上下文 SHA256 核对确认仅 layer1 不同，runner SHA256 与历史基准相同。使用独立目录及原 bank 的符号链接，没有覆盖旧产物或重复拷贝全部专家。输入仍为同一份128-token embedding，真实上层输出直接送下一层，全部权重常驻。各版本三轮，每轮6次预热、32次正式样本，共各96次；profile关闭。

计时从第0层输入 hidden 已写入缓冲后开始，到第3层输出完成并同步结束。包含两侧 norm、投影/attention、残差、MoE 路由/专家/shared、层间 hidden 传递、QNN/GGML交接及同步；排除第一层输入缓冲初次复制、模型加载、embedding、最终LM head、采样和SSD换入。输出落文件在计时外。

| 版本 | 四层整体中位 | 均值 | P95 |
|---|---:|---:|---:|
| 历史原版记录 | 35.312605ms | 35.392557ms | 35.893593ms |
| 本轮原版对照 | 35.375912ms | 35.416551ms | 35.892774ms |
| 本轮仅第1层修复 | **35.596172ms** | 35.637976ms | 36.009713ms |

同轮对照增加0.220260ms/0.623%。修复版三轮整体中位为35.667891/35.616641/35.573828ms。各段独立统计中位数，不能相加冒充整体中位：

| 层 | attention类型 | 原版attention→修复后 | 原版MoE→修复后 | 原版整层→修复后 |
|---|---|---:|---:|---:|
| 0 | GDN | 3.997057→3.999089ms | 4.058073→4.094532ms | 8.057969→8.095625ms |
| 1 | GDN | 3.891250→3.893959ms | 6.199011→5.965521ms | 10.141719→9.926511ms |
| 2 | GDN | 3.886406→3.878464ms | 5.802943→6.290521ms | 9.726432→10.210182ms |
| 3 | Full Attention | 1.637006→1.605859ms | 5.686719→5.665547ms | 7.381250→7.327917ms |

第1层attention耗时基本相同；修复改变了后续hidden和真实路由，MoE负载随之变化。选中专家数由208/215/193/177变为208/201/214/180；layer1的cap22/header推算有效payload从339720192降为322534912B，layer2由314382848增至343088128B。这是按路由和bank布局推算的有效权重payload，不是DDR硬件流量计数；整体差额还包含设备运行波动，不能全部解释为scale保护的计算开销。

逐层与同输入FP32 attention及完整FP32参考链比较：

| 层 | 原版normalized→修复后 | 原版累计hidden→修复后 |
|---|---:|---:|
| 0 | 11.63149→11.63149dB | 13.00423→13.00423dB |
| 1 | −1.37545→**20.08072dB** | −2.29869→**13.12396dB** |
| 2 | 15.99442→15.06053dB | −0.26735→12.37218dB |
| 3 | 24.61287→22.51170dB | −0.04411→**11.60127dB** |

后两层的“同输入”参考分别使用各版本实际到达该层的hidden，因此conditional SQNR不能当作输入完全不变的算子A/B；累计hidden统一对同一份FP32四层链。所有层、所有epoch的输出有限，最后残差逐元素正确；修复版MoE对各自原生N4模拟为29.44/28.75/29.78/30.88dB。本轮证实目标修复在真实四层链中生效，整体性能仍在35.6ms附近。未重测PPL、长历史cache、decode、SSD换入或40层；其余GDN层没有在本轮重新导出。

产物：`llama-output/qwen36-four-layer-retest-20261019/RESULTS.json`、逐层性能`RESULTS.csv`、`MANIFEST.json`、`baseline.log`、`fixed.log`、`baseline_ACCURACY.json`、`fixed_ACCURACY.json`及全部逐层输出。独立实验未整合进`llama.py`，本轮未新增生产代码变更，未commit/push。

#### 四层单层权重驻留：手机闪存按层换入（2026-10-19）

按“当前层MoE权重驻留DRAM，其余层bank存放手机文件系统，逐层按需换入”的口径测试。沿用真实Qwen3.6-35B-A3B的layer0/1/2/3、128-token输入、layer1已修复的attention、原INT4专家格式及计算内核。四套QNN attention context、图和计算scratch仍驻留；只把MoE权重改为单缓冲复用，不包含attention权重卸载。当前层加载的是该层全部256专家bank及router/shared，并未实现按实际选中专家读取或IO/计算重叠。实验未接入生产。

原四层MoE权重缓冲共1496620544B（1427.29MiB）；复用槽为414007296B（394.83MiB），减少72.34%。换层前等待上一层计算同步完成，复用同一物理缓冲；各层张量描述保留，使用可变缓冲属性。冷读额外需要8MiB常规对齐暂存，合计约402.83MiB权重及读取缓冲，不代表整个进程的总内存。

先加载layer0权重到DRAM，标准计时从首层hidden就绪开始；layer1/2/3的文件读取、写入NPU缓冲、计算及层间交接全部在计时内。另记录计入layer0重新加载的时间，用于说明下一次遍历也需要换回首层。文件句柄、图/context创建和输出落文件不在计时内，仍不含embedding、LM head、采样和SSD写回。以下为本手机文件系统的读取结果，并非PC NVMe SSD测试，也未实际运行40层。

| 模式 | 正式样本 | 四层整体中位 | P95 | 计入首层重新加载的整体中位 |
|---|---:|---:|---:|---:|
| 四层MoE全部驻留DRAM | 96 | **35.26823ms** | 36.11762ms | 35.26823ms |
| 单层缓冲，文件页缓存命中 | 8 | **174.67148ms** | 190.14406ms | 223.24758ms |
| 单层缓冲，O_DIRECT冷读 | 8 | **554.73047ms** | 570.50909ms | **660.30477ms** |

相对同轮全驻留，对应4.95倍和15.73倍耗时；冷读标准计时增加519.46ms。页缓存模式虽然只有一个应用权重槽，但全部bank的365406个文件页仍驻留内核页缓存，不能将其作为真实闪存速度或整机节省了同等DRAM的结果。

冷读通过`O_DIRECT`绕过页缓存，并在开始前对这些实验bank调用`POSIX_FADV_DONTNEED`；`mincore`测得bank页缓存前后均为0。每次标准计时内`/proc/self/io::read_bytes`为1237499904B，计入首层后为1496702976B，与文件按页对齐后的总量一致。其余两种模式计时内read_bytes均为0。

本手机对RPC共享内存直接`pread(O_DIRECT)`返回EFAULT，首轮失败不计成绩。最终实现先读入8MiB页对齐普通内存，再复制到NPU可访问的共享权重槽，完整统计两段开销；没有悄悄回退到缓冲文件读取。

| 换入层 | 文件读取中位 | 暂存→NPU共享DRAM复制中位 | 整体装载中位 |
|---|---:|---:|---:|
| 1 | 155.3077ms | 24.5645ms | 180.1988ms |
| 2 | 119.5443ms | 24.1548ms | 143.4357ms |
| 3 | 163.0199ms | 32.2170ms | 194.8220ms |

三层分别统计的中位数合计约437.87ms读取、80.94ms复制、518.46ms装载；各段中位数不能严格相加等同整体中位。按三层有效文件字节/读取时间估算约2.83GB/s，仍是该路径在本机的实测值，不是设备标称带宽或DRAM硬件流量计数。当前主要瓶颈已经变为按层搬运约1.24GB权重，计算仍是几十毫秒。

精度单独核对：全驻留、页缓存换入、O_DIRECT换入的32个权重张量FNV1a64哈希全部一致，哈希诊断只放在独立warmup，不计入正式性能。正式各模式首末两个快照的所有层输出有限，最后残差逐元素正确；最终hidden对同一FP32四层参考，resident为11.60127150dB，cached为11.60127150dB，direct为11.60367358/11.60194848dB，未观察到绝对SQNR下降。

**不能据此宣称换入与全驻留逐位一致**：cached与resident末快照为158.99572dB/maxdiff1.19209e−7，而direct与resident末快照为42.73822dB/maxdiff0.00571293，direct自身首末重复约42.06177dB。权重哈希一致，说明不是文件换入后权重字节变化；数值波动在计算/runtime路径中的具体原因尚未隔离，本轮不把它写成零额外误差，也没有测全模型PPL。该问题与读取带宽的实测结果分别保留。

结论：在这一“仅保留当前层、整层串行换入”的实现下，128-token四层耗时由约35ms升至约555ms，首层也需重新加载时约660ms。后续降低开销应比较只读取实际路由命中的专家、连续区间合并读取、多层/专家缓存、双缓冲预取与计算重叠；本轮没有把这些尚未实现的收益计入结果。

产物保存在`llama-output/qwen36-four-layer-storage-20261019/`：`RESULTS.csv`、`RESULTS.json`、`resident.log`、`cached.log`、`direct.log`、哈希诊断`hash-*.log`、各模式首末逐层输出，以及`source/storage_bench.cpp`、独立改写的`moe_factory.h`和复现脚本。手机端已拉回主机的初轮输出副本曾清理以释放空间；旧四层基准和全部专家bank未覆盖。本轮没有修改生产推理代码，未commit/push。

#### 四层闪存换入优化：真实路由选择、并发读取、有限专家缓存（2026-10-19）

延续上节的真实前四层、AR128、SM8750/v79手机`431e3c37`，优化MoE权重换入。仍使用原INT4专家bank和N4×A8整数HMX内核，未减少Top8、未丢弃低分专家、未修改attention精度；layer1继续使用scale修复版context。旧基准未覆盖，本轮仍为独立原型，未接入`llama.py`。

**实现变化**：

1. 将MoE执行图拆为router和专家计算两个子图。每层先执行本层attention和真实router，将当前128个token的Top8 ID读回，再求命中专家的并集。专家读取清单在正式计时内生成，没有使用预录的测试路由或预测下一层ID。保留原图的scratch分配、hidden/residual和router输出；子图UID设0，避免两个子图误复用同一个GGML图缓存。
2. 只把两个大bank（`gate_up_part0`、`down_part0`）放在可复用专家槽。四层router/shared/小占位bank分别保留8MiB元数据及权重缓冲，合计32MiB；前后端QNN attention context仍驻留。这里的“元数据”缓冲也包含router和shared权重，不仅是索引。
3. 根据原bank offset/header定位完整专家payload，按4KiB页对齐并合并重叠、相邻区间。只读实际命中的专家，保留原INT4格式和专家索引，不在装载时反量化。少量页边界会带入相邻专家的字节，物理读取量按实际页统计。
4. 使用持久IO线程池，各线程通过8MiB对齐普通缓冲执行`pread(O_DIRECT)`，随后复制到NPU共享专家槽。读取与复制在多个线程之间流水执行；没有把本机不支持的“直接O_DIRECT到RPC共享内存”当成成功路径。
5. 增加容量受限的CPU DRAM热门专家缓存，每层32或64个。名单来自各层**独立train/calibration序列的1024-token FP32路由频率**；正式测试使用另一段test序列。命中的缓存字节复制到复用槽，其余仍从手机闪存冷读。缓存范围与闪存读取范围严格切开，避免并发写同一目标页。没有用重复测试把全部bank偷偷暖进文件页缓存。
6. 另外试过在本层attention/router期间并发读取该层全部专家。实际选择性读取必须等待真实router，不能提前知道本次命中ID；本轮没有把预测预取算作真实路由收益。

**计时与缓存边界**：每组1次预热、8次正式样本，profile关闭；除单列的重叠实验只有3次正式样本。首先加载第0层完整专家bank并写入首层hidden，之后才开始标准四层计时；第1/2/3层实际路由、读取规划、闪存读取、CPU复制、attention、MoE及层间同步都在计时内。额外列出计入第0层重载的整体时间。context/图创建、一次性热门缓存初始化、embedding、LM head、采样及输出落文件不在标准计时内，不能把本表称为完整40层模型TTFT。

全部正式组的bank `mincore`页缓存均为`0→0`，`/proc/self/io::read_bytes`逐样本等于规划并执行的物理页读取总量。下表GB使用十进制，MiB使用2^20字节。

| 方案 | 四层中位 | P95 | 计入首层重载的中位 | 计时内闪存读取 | MoE权重及IO工作内存 |
|---|---:|---:|---:|---:|---:|
| 上节串行整层换入 | 554.73ms | 570.51ms | 660.30ms | 1.2375GB | 402.83MiB |
| 本轮串行整专家bank，router/shared驻留 | 535.05ms | 549.23ms | 630.35ms | 1.2146GB | 434.83MiB |
| 整专家bank，4线程 | 408.20ms | 426.98ms | 480.95ms | 1.2146GB | 458.83MiB |
| 真实命中专家，4线程，无热门缓存 | **366.86ms** | 371.56ms | 440.37ms | **0.9413GB** | **458.83MiB** |
| 真实命中专家，8线程，无热门缓存 | 366.31ms | 370.17ms | 444.62ms | 0.9413GB | 490.83MiB |
| 每层32热门专家，4线程 | **324.89ms** | 327.43ms | 398.43ms | **0.7951GB** | **635.71MiB** |
| 每层32热门专家，8线程 | 326.60ms | 359.00ms | 400.61ms | 0.7951GB | 667.71MiB |
| 每层64热门专家，4线程 | **283.28ms** | 286.22ms | 349.91ms | **0.6573GB** | **813.48MiB** |
| 每层64热门专家，8线程 | **280.74ms** | 292.51ms | 347.71ms | **0.6573GB** | **845.48MiB** |

相对上节554.73ms，低内存的“真实命中+4线程”减少33.87%，32专家/4线程减少41.43%，最快64专家/8线程减少49.39%，约1.98倍速度。计入首层重载，最快347.71ms相对660.30ms减少47.34%。本轮另有535.05ms串行对照，区分了router/shared驻留、拆图等基础变化与后续优化，不能把全部收益都归因于选择性读取。

四线程配置更适合内存紧张时使用：无热门缓存时，8线程仅快0.55ms却多32MiB缓冲；32专家时4线程反而略快；64专家时8线程只快2.55ms且P95更高。不是线程越多越好。

整bank预取与attention/router重叠的3样本探索中位396.41ms，同轮非重叠4线程为408.20ms；读取字节仍为1.2146GB，没有解决大头。此项样本较少、且并非交替AB，不把这11.78ms全部认定为稳定的纯重叠收益。正式保留的推荐配置仍是按真实路由读取加有限缓存。

**为什么仍明显慢于全驻留**：本测试每token选8个专家，但128个token合在一起，四层分别命中约208/201/214/180个不同专家，并非整个block只用8个。个别运行因已有的微小数值波动，后两层并集会变为213/181；全部样本都按当次真实ID读取。即使每层保留64个热门专家，后面三层仍需读取约657MB；热点缓存减少的是闪存字节，不会消除NPU槽内的复制和执行。

以64专家/4线程为例，各段分别统计的中位：

| 层 | attention front | router及ID取回 | 权重读取+复制wall time | 专家/shared及最终残差 |
|---|---:|---:|---:|---:|
| 0 | 4.107ms | 0.352ms | 计时前预装载 | 3.842ms |
| 1 | 4.029ms | 0.377ms | 84.874ms | 5.552ms |
| 2 | 4.275ms | 0.371ms | 82.405ms | 5.866ms |
| 3 | 1.890ms | 0.336ms | 78.448ms | 5.264ms |

三层装载中位合计245.73ms，约为283.28ms整体的87%，仅用于判断瓶颈；不同段的中位不能严格相加等同整体中位。日志`read_worker_ms`和`copy_worker_ms`是多个并发线程工作时间之和，不是可相加到关键路径的串行耗时。当前应继续处理实际读取字节和装载路径，单纯优化几毫秒attention无法追回这部分延迟。

**DRAM账目与40层限制**：四层复用一个394.83MiB专家槽。4线程需要32MiB IO暂存，四层router/shared等需32MiB；32专家缓存实际176.89MiB，64专家实际354.65MiB。因此上表对应635.71或813.48MiB，已经把CPU热门缓存计入，没有把文件页缓存漏掉。该数值是显式MoE权重/IO分配之和，**不是整个进程RSS**，未含attention权重/context、图scratch、激活、其他allocator和系统。

如果40层的每层缓存大小与本次四层均值相同，保持单个专家槽、4线程、每层8MiB router/shared缓冲，则MoE侧预算粗估：不放热门缓存约0.73GiB，每层32专家约**2.46GiB**，每层64专家约**4.19GiB**。这是容量外推，不是实测40层峰值；完整模型还要加attention/embedding/激活/系统，不能说“四层813MiB即可装下整个40层”。实际整合应使用**全局固定字节上限的缓存池**，在总DRAM预算内分配各层热门专家，未命中仍按真实路由从闪存读取，而不是给40层无限增加缓存。本轮实现和测试的是固定四层的每层32/64配置，尚未实现全模型动态淘汰。

热门缓存初始化属于冷启动成本：32专家/4线程本次242.01ms，64专家/4线程464.13ms，64专家/8线程443.05ms；均是进程启动时建立四层缓存的一次性时间，未计入标准四层执行，也未隐藏在第一轮预热中。后续连续遍历仍需重新换回第0层，本表已有含首层重载口径。

**精度验证**：所有正式组首末快照的四层hidden、MoE、normalized和cache均有限，最终残差逐元素正确，Top8 ID合法且分数和约为1。无缓存选择性读取、32专家/4线程、最快64专家/8线程的最终hidden对统一FP32四层链均约**11.60127dB**；相对本轮串行整bank对照，最快组两个快照为161.01/162.31dB，maxdiff均1.19209e−7。最省内存的4线程选择性读取为171.22/162.84dB，32专家/4线程为168.51/168.50dB。未观察到本轮装载优化引起的绝对FP32 SQNR下降。

仍不能宣称所有运行逐位一致：64专家/4线程首快照对整bank对照约42.35dB、maxdiff0.00686，末快照恢复约169.18dB；绝对FP32分别11.60367/11.60127dB。本轮整bank4线程也有一次类似小波动，上节O_DIRECT原型已有同类现象。未定位具体kernel/runtime原因，因此保留日志及重复性数据，不将其描述为已修复或零误差。本轮没有测试全模型PPL、AR1 decode或40层。

本轮可复用的结果：DRAM受限优先采用单槽、真实路由、4线程和受预算约束的32专家缓存；64专家模式作为更高内存换性能的选项。该原型把闪存换层耗时降低约一半，但尚未达到全驻留的35ms四层水平。

完整证据位于`llama-output/qwen36-moe-storage-optimization-20261019/`：`RESULTS.csv`、`RESULTS.json`（逐层耗时、物理读取、首末精度和重复性）、`HOT_CALIBRATION.json`（独立校准排名及命中统计）、`MANIFEST.json`（输入context来源、环境、源码及二进制哈希）、正式`final-*.log`和逐层快照。`source/stream_bench.cpp`实现装载优化，`source/moe_factory.h`实现单槽及子图准备，`source/build.sh`复用原优化GGML库构建Android可执行文件；`source/run_device.py`在原已部署bank上串行复现，`source/analyze.py`生成结果。正式配置使用8MiB读块和8次样本，原数据与旧模型未覆盖；本轮只追加测试记录，未改生产推理路径，未commit/push。

#### 专家权重读取与计算流水线：已装载专家先算，后续专家后台读取（2026-10-19）

用户提出在权重使用前提前装载。本轮在前述四层原型上实现了**同层专家级装载/计算重叠**：attention和真实router先执行；IO线程根据本次真实Top8读取专家，某专家的gate/up和down完整到达后发布就绪标记，DSP不再等待本层全部专家装载完成。仍复用同一个394.83MiB专家槽，未另加一整层权重缓冲，仍未接入生产`llama.py`。

原路径是`attention → router → 全部命中专家读完 → MoE计算 → 下一层`；新路径是`attention → router → 后台读取`，与之并行的专家内核按就绪状态执行，读线程继续装载后续专家。IO清单完全来自当次真实router，没有用上一轮测试ID、预测ID或丢弃低权重专家。INT4存储、N4×A8 HMX计算、K/z及attention量化配置不变。

**关键实现及正确性约束**：

1. `source/stream_bench.cpp::initialize_readiness`为所有读取/缓存复制job建立专家依赖计数。必须完成该专家两个bank的所有相关job，才能发布ready。页对齐导致一个job触及相邻专家时，两者都记录依赖，避免专家计算中又被另一个job写到相同payload页。
2. IO worker完成`O_DIRECT → 普通对齐暂存 → NPU共享槽`后，在ARM侧用`dc cvac`及`dsb ish`清理payload缓存；随后release写ready并清理标记所在缓存行。CPU DRAM热门缓存命中也走相同发布顺序。不是仅用普通bool就假定CPU/DSP自动一致。
3. 当前专用AR128/256专家协议复用router输出每128字节行中空闲的64字节，放256个ready word；没有改变Top8 ID和score。DSP `moe_storage_wait`按缓存失效和同步后可见的标记等待，配有超时。该padding协议只适用于本固定形状实验，未作为通用算子接口接入。
4. 在`ggml/src/ggml-hexagon/htp/expert-pipeline.h`的当前专家执行前增加等待；`expert-hybrid.h`原本存在跨专家DMA预取，本轮也在该预取入口检查ready。未就绪时保留下一任务但不发起DMA，轮到它时再等待，避免提前读到旧层/未装载payload。原DDR→VTCM→HMX流水保留。
5. 首版忙等待能正确执行，但增添开销；改为QuRT允许的最小100μs短休眠。逐专家交替两个bank的读取也降低了吞吐；最终保留按bank连续区间顺序装载，仍逐专家发布ready。程序保留对照选项，正式推荐`IO_EXPERT_GROUP=256`，不是要求256个专家全部ready才计算。
6. DSP首次完成一个专家后，在另一条空闲缓存行写完成标记并清理；IO worker在自己的时钟上观察该标记。这个观察时间是“至少已有一个专家完成”的时间上界，避免把CPU/DSP不同计数器直接相减，能证明不是只把线程启动并行、实际计算仍全程等待。

先前原型只试过读取和attention重叠；此处是在MoE专家计算内部重叠读取，不能混称同一项优化。这里也没有跨层提前得到下一层的精确专家名单：下一层router仍依赖当前层最终hidden。下一层固定权重/热门专家可提前装载；冷专家若要在router前预取，需要预测并处理误命中，或者额外整层/分组缓冲。这些跨层策略本轮没有实现或计入收益。

**正式性能**：同一SM8750/v79手机、前四层真实链、128-token输入、layer1 scale修复版context，profile关闭。32专家/4线程配置按`legacy→ordered→stream`、`stream→ordered→legacy`、`ordered→legacy→stream`三轮交替，每轮每组1次预热、8次正式样本，各24次。64专家/4线程另做`legacy→stream`与`stream→legacy`两轮，各16次。每组仍是4个IO线程、每线程8MiB暂存；热门名单沿用独立train/calibration频率。

| 每层热门缓存 | 本轮原路径四层中位 | 同读取/发布协议但串行 | 专家流水线四层中位 | 对原路径降低 | 流水线P95 | MoE权重及IO工作内存 |
|---|---:|---:|---:|---:|---:|---:|
| 32专家，4线程，各24样本 | 324.20896ms | 327.91864ms | **308.43763ms** | **4.86%** | 313.69671ms | **635.71MiB** |
| 64专家，4线程，各16样本 | 293.80435ms | 未单独测 | **273.45013ms** | **6.93%** | 288.70736ms | **813.48MiB** |

32专家对同协议串行减少19.48ms/5.94%；对已有原路径净减少15.77ms/4.86%，差额包括缓存清理、就绪协议和启动管理开销。三轮流水线中位308.577/307.929/309.049ms，原路径324.209/327.187/320.629ms；同协议串行某轮出现较高读取波动，不能只引用其355ms当对照来放大收益。64专家两轮流水线分别273.121/275.078ms，原路径298.145/290.406ms。本轮64专家串行与上节历史283.28ms有波动差异，收益使用本轮交替对照，不跨轮拼接。

计入第0层完整专家bank重载，32专家原路径396.20508→流水线**380.79193ms**；64专家360.11094→**341.86586ms**。这些时间仍不含context/图创建、一次性热门缓存初始化、embedding、LM head和采样，不是40层模型的完整TTFT。四个QNN attention context及router/shared缓冲继续驻留。

物理读取没有靠丢专家减少：32专家两种路径的计时内读取中位均795107328B；64专家均657313792B。个别样本因已有数值波动导致少量真实路由变化，按该次ID规划的字节数相应变化。逐样本`/proc/self/io::read_bytes`与实际job读取字节一致，bank页缓存均为`0→0`。工作内存与之前相同；ready/progress只复用padding和少量任务管理内存，没有再开394.83MiB整层槽。表中内存仍不是整个进程RSS，未含attention、scratch、激活及系统；不能将四层预算直接当作完整40层预算。

**直接重叠证据，32专家流水线的各段独立中位**：

| 换入层 | 首个任意专家ready | IO线程观察到已有专家完成 | 全部装载完成 | MoE调用耗时（含等待） |
|---|---:|---:|---:|---:|
| 1 | 61.174ms | **70.914ms** | 100.657ms | 101.087ms |
| 2 | 61.993ms | **62.488ms** | 95.450ms | 95.806ms |
| 3 | 58.468ms | **59.634ms** | 90.607ms | 90.964ms |

表前三列相对各层IO开始计时；不是完整四层开始的时间。所有正式流水线样本的换入层均观察到`0 < compute_observed_ms < io_ms`，且完成装载的专家数等于真实命中并集。由此验证后续权重还在读取时，已有专家完成了计算。`moe_ms`现在包含权重ready等待，和`io_ms`重叠，不能再按串行段相加；多线程`read/copy/clean_worker_ms`同样不是可相加的wall time。

收益只有几个百分点的原因是剩余瓶颈仍为**约0.66～0.80GB的实际冷读取**。32专家原路径三层装载中位合计约284.82ms，而对应MoE计算合计约18.06ms；流水线主要覆盖这18ms左右的计算，不能把280ms量级的存储搬运全部藏进几十毫秒计算。若要大幅继续降低完整长提示prefill开销，应同时研究在固定内存预算内提高缓存命中，或对多个AR128块按层复用已装载权重，减少同一权重反复读取；这些尚未实测，未计入本轮性能。

**精度**：正式各组首末快照的四层hidden、normalized、MoE及cache均有限，最终残差逐元素正确；32专家组还检查Top8 ID范围、非负score及score和约1。最终hidden对同一FP32四层参考仍约**11.60dB**。32专家流水线6份末层快照中5份约11.60127dB，1份11.60347dB；64专家4份约11.60127～11.60158dB，没有观察到绝对FP32 SQNR恶化。没有重测完整模型PPL。

仍保留此前native/runtime重复性限制：32专家流水线有一份对稳定原路径快照仅40.85dB/maxdiff0.00697，其余约158dB；64专家原路径自己也有首末约41dB的变化。不是所有组都逐位一致，不能据此宣称已修复历史运行波动或零附加误差。完整绝对SQNR和跨路径比较都保存在JSON中；64专家跨路径控制使用`cache64-b-legacy.sample0`，同时保留另一轮原路径的波动快照。

测试时手机文件系统一度剩余不足1GB；只删除了**已完整拉回主机的本轮/上一轮实验快照副本**，释放至约1.7GB，模型和专家bank未动。正式快照文件在计时外`fsync`，避免前一快照的后台写入直接干扰后续样本。设备仍接近满盘，实测包含本机当时存储状态，不当作设备标称吞吐上限。

产物：`llama-output/qwen36-moe-storage-pipeline-20261019/RESULTS.json`、`RESULTS.csv`（32专家正式交替组）、`CACHE64_RESULTS.json`、`MANIFEST.json`、`formal-r*.log`、`cache64-*.log`和首末逐层快照。`source/run_formal.py`复现32专家交替测试，`source/run_device.py`可设置`HOT_EXPERTS=64`、`IO_EXPERT_GROUP=256`复现64专家组。`source/build.sh`单独构建修改后的HTP库与Android原型，修改后的GGML源码在独立`ggml/`目录；旧库未覆盖。忙等待、细粒度读取顺序等探索日志保留，但不混入正式成绩。原型尚未整合到`llama.py`，未运行完整40层、AR1 decode或全模型PPL，未commit/push。

#### 手机存储带宽诊断：连续读、真实专家区间与完整流水线（2026-10-19）

用户追问流水线是否把“SSD带宽”用满。本轮没有直接将既有2.6～2.8GB/s装载吞吐当作硬件上限，而是增加独立只读探针，对同一手机、同一批专家bank测试连续读取、真实miss区间、CPU→RPC缓冲复制、缓存同步及Linux native AIO队列深度。

设备`431e3c37`的内部块设备路径含`1d84000.ufshc`，测的是**手机内部UFS文件系统路径**，不是主机NVMe或USB传输速度。`/proc/diskstats`、`/sys/block/sda/stat`均返回Permission denied，没有拿到硬件busy/util计数；下文百分比是实测文件吞吐之比，**不是SSD/NAND硬件利用率百分比，也没有证明理论硬件峰值**。

**测试控制**：不新建大数据文件、不改专家权重，仅对已部署bank执行`O_DIRECT`读取，所有测试串行。每组1次预热、5次正式样本；`mincore`在所有正式组开始/结束均为0，逐样本`/proc/self/io::read_bytes`与规划页字节数完全一致。该计数证明绕过文件页缓存的存储IO计账，不是NAND总线流量计。读取完成后只校验每块首尾字，属于诊断采样校验，未用它宣称全文件哈希一致。

三类输入：

- `full.plan`：layer1/2/3两个大bank的全部数据，1214631936B。连续区间分成最大8MiB请求，可跨层并发，用于观察本文件路径的大块读取能力。
- `miss64.plan`：来自前轮稳定真实路由快照及独立训练集top64缓存名单。保留原偏移、原bank、每层屏障，共657313792B冷读；按8MiB上限拆成**369个磁盘请求**，平均1.78MB、中位1.06496MB。装载对照另复制284016640B真实缓存页到RPC槽，总写入权重槽约941MB，匹配原访问清单。请求分布由实际命中/未命中空洞形成，不能用8MiB上限代表每次请求都连续读8MiB。
- `compact.plan`：仍读取原文件，只将每文件同样的冷读字节数改为连续前缀，保留相同三层屏障，总量仍657313792B、8MiB上限下81个请求。**这是地址布局对照，不是实际被选专家的权重，也没有导出或推理重排后的模型**，只量化访问地址形状的潜在损失。

**并发扩展结果，GB/s为十进制，每格为5个样本中位**：

| 同步pread线程数 | 整bank连续纯读 | 真实miss64区间纯读 |
|---|---:|---:|
| 1 | 2.8545GB/s | 1.8909GB/s |
| 2 | **3.3521GB/s** | 2.5609GB/s |
| 4 | 3.2997GB/s | 2.6677GB/s |
| 8 | 3.2537GB/s | 2.7085GB/s |
| 16 | 2.8144GB/s | 2.5970GB/s |

| Linux AIO应用队列深度 | 真实miss64区间纯读 |
|---|---:|
| 4 | 2.7358GB/s |
| 8 | **2.7401GB/s** |
| 16 | 2.7249GB/s |
| 32 | 2.7349GB/s |

同步线程数/异步应用队列深度表示应用最多提交的读请求，不等于物理NAND队列深度。并发继续增加没有得到显著增长。1MiB请求下，真实区间pread8为2.6498GB/s、AIO16为2.7125GB/s；2MiB请求下pread4为2.6607GB/s、AIO8为2.7329GB/s，均未突破8MiB上限方案的吞吐平台。因此本轮不以继续增加读线程、缩小读块或加大AIO深度作为已验证的优化。

**分离复制/同步开销**：

| 路径 | 数据范围 | 中位耗时 | 以冷读字节计算的有效吞吐 |
|---|---|---:|---:|
| 真实区间，AIO8，仅冷读 | 657.31MB | 239.889ms | **2.7401GB/s** |
| 真实区间，4线程，含热页复制到cached RPC槽 | 冷读657.31MB＋热页复制284.02MB | 245.728ms | 2.6750GB/s |
| 上项再加ARM缓存clean | 同上 | 244.807ms | 2.6850GB/s |
| 真实区间，8线程，含复制和clean | 同上 | 242.666ms | 2.7087GB/s |
| 等字节连续地址对照，AIO8，仅冷读 | 657.31MB，地址不同 | **192.926ms** | **3.4071GB/s** |

复制和clean探针通过与原路径相同heap/flags分配cached RPC缓冲，但未创建QNN context、DSP导入或执行计算；该表用于隔离装载路径，不是完整模型对照。copy与clean的几毫秒差异在设备运行波动内，不能写成“缓存同步反而加速”。相对纯读，复制和同步的wall-time增加较小，因为多线程执行已存在重叠；日志中的worker时间之和不能直接加到端到端时间上。

所有纯IO测试结束后，单独重跑原四层64专家缓存/4线程流水线，1预热＋8正式样本：整体中位**271.17234ms**、P95 277.48963ms；计入首层重载338.24719ms。逐样本`冷读字节 / 三个换入层io_ms之和`的中位为**2.62734GB/s**，仍保持真实路由、原量化及0→0文件页缓存，各换入层仍观察到专家计算早于IO完成。模型/权重未改，性能变化属于本轮设备状态下的重测，不能把273.45→271.17ms算作新增代码优化。

由此得到两个不同结论：

1. 当前完整装载约为本轮整bank连续纯读最高值的**78.38%**，相对等字节连续地址对照为**77.11%**。因此尚未达到该手机文件路径能实现的连续吞吐。
2. 当前完整装载约为本轮**真实专家区间纯读最好值的95.89%**。保留现在的访问布局时，IO、复制、同步和计算流水线已经接近本轮观测到的路径平台；仅把现有队列继续加深，没有观察到成倍空间。该比值受不同测试的时序、DVFS及负载影响，是近似路径比较，不是硬件%util。

当前应优先研究**减少实际冷读字节与改善bank地址布局**。热门专家在按ID排列的bank中分散，跳过缓存和未命中专家后形成许多空洞；可以离线把固定热点集中为连续区域，再按校准集共现关系排列其余专家，保留“逻辑expert ID→物理offset”索引。该建议由地址对照推断，本轮尚未实现重排bank或验证其精度/端到端收益。即使按本轮3.407GB/s连续读能力估算，657MB冷读本身仍约**193ms**，尚未计入其他开销；单靠补齐这部分带宽不能回到全驻留35ms四层。更大的prefill收益还需减少重复搬运，例如受总内存预算约束的缓存或多个AR128块的按层权重复用，不能仅依赖更早启动读取。

存储空间本轮约剩2.7GB，仍接近满盘；结果包含当时文件系统/映射/加密路径和设备时钟状态，不是裸UFS峰值。初轮探针有线程捕获错误导致进程退出，已修复并保存`failed-*`日志，失败样本及smoke结果均未混入正式统计。没有对手机执行root、修改系统队列或覆盖旧模型。

证据位于`llama-output/qwen36-storage-bandwidth-20261019/`：`RESULTS.csv`、`RESULTS.json`（全部31组正式IO对照及模型重测）、`PLANS.json`、四份`.plan`、`MANIFEST.json`、`DEVICE_INFO.txt`、`model-stream.log`及各配置日志；`source/make_plans.py`重建访问清单，`source/io_probe.cpp`实现同步/内核AIO及RPC装载隔离测试，`source/run_device.py`按配置串行测量。原量化、生产推理路径及专家bank未改，未commit/push。

#### 两层专家驻留：先处理完全部AR128块，再切换下一组（2026-10-19）

按用户提出的两层驻留方案，新增独立四层原型，在同一SM8750/v79手机上实测。输入使用真实Qwen3.6-35B-A3B前四层的同一段**1024-token文本embedding**，每次图调用仍为**AR128**，共8块；前三层为GDN，第4层为全注意力。不是8个互相独立的冷128-token请求，也不是完整40层模型结果。

两种调度使用相同的两个专家共享槽、四层attention context、原生INT4专家bank、router/shared权重和UINT16 cache。区别只在循环顺序：

```text
逐块换层：
  对每个128-token块：
    读取L0/L1专家 → 运行L0/L1
    读取L2/L3专家 → 运行L2/L3

两层驻留：
  读取L0/L1专家 → 全部8块依次运行L0/L1，保留各块hidden
  读取L2/L3专家 → 全部8块依次运行L2/L3
```

四层专家装载由**32次减为4次**。每次装载当前层全部256个专家；实际计算仍为每token的真实Top8以及shared expert，没有裁剪模型、预测路由或丢弃专家。两个大bank从手机内部UFS经2个`pread`线程、各8MiB对齐暂存复制到RPC专家槽；使用`O_DIRECT`。本轮没有热门专家额外缓存，也没有跨层预取/读算重叠，因此比较的是相同整bank装载策略下的调度收益，**不能把倍数直接套到上一节的top64热门缓存流水线**。

**先补齐因果计算及缓存反馈**：

- 沿用已修复layer1 scale的三个GDN context，跨块分别保留conv/recurrent state。conv输出与输入的量化网格不同，按各自校准scale/offset建立转换表，反馈前重新编码；没有直接复制不匹配的UINT16码值。state网格相同也走等价映射。
- 旧第4层只有冷128-token全注意力，本轮单独导出AR128＋896-token历史K/V输入的新context。历史右对齐，逐块更新K/V、RoPE绝对位置和因果mask；输出128条新K/V后保存最近896条供下一块读取。固定容量1024可覆盖本次完整提示。
- FP32下，分块normalized输出对完整1024-token参考的SQNR：GDN层0/1/2分别**125.865/128.234/128.684dB**，全注意力**125.724dB**；最大绝对误差均不超过1.24e-5。由此核对分块缓存和位置语义，没有用错误的冷图循环代替prefill。

**正式性能，1次预热＋3次计时，中位数**：两条换入路径的首次专家读取全部计入计时；context/图创建和一次性router/shared装载在计时外。额外四层全驻留组只作为不读闪存的计算对照，首次专家装载在计时外。最终轮关闭逐层trace；输出/cache快照在计时外关闭文件并`fsync`，完成后才开始下一样本。

| 调度 | 1024-token四层总耗时 | 最小～最大 | 计时内实际读取 | 专家槽分配 |
|---|---:|---:|---:|---:|
| 两槽，逐块换层 | **3929.764ms** | 3865.977～3974.172ms | **10.9241GiB** | 789.66MiB |
| 两槽，两层处理完全部8块再换组 | **783.511ms** | 781.142～795.753ms | **1.3655GiB** | 789.66MiB |
| 四槽，四层全常驻，计时内无权重读取 | **342.935ms** | 341.044～342.954ms | 0 | 1579.31MiB |

两层驻留对同装载策略逐块对照**加速5.016倍、耗时减少80.06%**，物理读取计账由11729698816B降至1466212352B，严格为八分之一。该计数来自`/proc/self/io::read_bytes`；少量页对齐差异相对逻辑bank总大小1466194944B单独保留。不是把文件页缓存读命中当作UFS读取，也不是USB或主机SSD吞吐。

两层驻留各段独立中位：attention（含两侧norm、首次残差及输入交接）**143.449ms**，MoE（含router/shared/末次残差）**193.052ms**，cache反馈**7.274ms**，专家读取及复制**438.240ms**。独立中位不能严格相加成总中位。闪存装载仍约占总时间56%；计算段与四层全驻留对照接近，主要收益来自避免反复搬相同专家权重。

**内存预算**：两个固定共享专家槽合计828014592B＝**789.66MiB**；四层router/shared等metadata槽合计32MiB；两路完整prompt hidden缓冲16MiB；IO暂存16MiB，上述明确分配合计**853.66MiB**。此外还有输入embedding副本8MiB、四层attention常量/context、GGML scratch、QNN IO、全部层cache以及运行库；853.66MiB不是总进程/总手机DRAM。RPC/ION分配也不能仅靠`VmRSS`完整统计。专家槽只保留两层，四层attention和router/shared继续常驻。没有复制一份四层专家作为CPU热缓存。

**数值重复性与精度**：原最快MoE使用多线程按完成顺序累加，逐层trace看到层0normalized逐位相同、层0hidden出现约1e-8差异，后续量化/递归会把差异放大；全驻留重复运行也能复现。因此本轮正式对照统一设置现有`MOE_ACCUMULATE=0`，采用固定Top8槽顺序合并，专家仍为原生INT4＋A8计算，未改成FP16专家MatMul。

最终带`fsync`的三组各3次测试，共9个完整1024-token hidden和所有最终cache，**相对逐块组sample0全部逐位一致、全部有限**。三组最终hidden对同一FP32四层参考均为**12.216008dB**。两层驻留按第0～7块分别为11.599/12.136/12.449/12.117/12.778/12.305/11.808/12.681dB，未出现随调度切换新增的误差。这证明本次受控样本下调度等价；12.216dB是已有量化链的绝对误差，不能称为无量化损失。

同时保留探索阶段的重复性限制：未固定合并顺序时，同一调度的完整prompt重复SQNR曾低至约22.95dB；固定合并顺序的较早一轮仍观察到一个MoE层输出差异，经后续层放大到58.96dB。最终9份逐位相同不等于已证明native内核所有输入/所有重复运行完全确定。本轮没有修改这些底层内核，没有重测完整模型PPL。

**适用范围**：本次减少读取的倍率来自**1024÷128＝8块的权重复用**。单次总输入只有128 token时，两种顺序都必须装载四层专家一次，不能得到八倍IO收益；AR1 decode也不能把尚未生成的后续token提前按层批处理。上下文更长时仍需维护全部历史KV和每层独立state；只测了1024和前四层，未把四层结果外推为完整40层端到端成绩。旧35.31ms对应冷128-token/不同全注意力图，本轮342.94ms对应带历史KV的1024-token四层，二者不能直接混用。

产物：`llama-output/qwen36-two-layer-residency-20261019/RESULTS.json`、`RESULTS.csv`、`STAGES.csv`、`MANIFEST.json`、`DEVICE_CONTEXTS.txt`、`FP32_GDN_CAUSAL_PARITY.json`、`final-{group,chunk,resident}.log`和`results/final-*`快照。探索日志和`before-fsync/`单独保留，不混入正式统计。`source/residency.cpp`实现两槽调度，`source/qnn_front.h`实现量化缓存反馈，`source/export_warm_fa.py`导出历史KV图；`source/build.sh`构建独立Android程序，`source/run_device.py --mode group --samples 3 --name final-group --ordered-combine`复现两层驻留，`source/analyze.py`生成统计。未整合到`llama.py`，生产量化配置、旧专家bank和原生算子库未改，未commit/push。

#### SM8975/V85：QNN 2.48.1＋设备ODM库兼容性实测（2026-10-20）

设备`A7SR026915000282`，`ro.soc.model=SM8975`、`ro.board.platform=art`，产品名WKL-AN20。按用户指定使用`/hunter/qnn_sdk/src/backends/qnn/2.48.1.260715`，重新构建隔离的Python 3.11 `PyQnnManagerAdaptor`，编译目标直接为`SM8975/HtpArch.V85`。没有将V85伪装成V81编译。运行时从以下只读系统路径复制库到独立测试目录：

```text
/odm/lib64/libQnnHtp.so
/odm/lib64/libQnnHtpV85Stub.so
/odm/lib64/libQnnSystem.so
/odm/lib64/rfsa/libQnnHtpV85Skel.so
```

哈希验证：ODM `libQnnHtp.so`和`libQnnSystem.so`分别与指定2.48.1 SDK的Android版本逐字节一致；HTP build ID为`v2.48.1.260715235057`。SDK确实不包含hexagon-v85/Stub，但主机编译器能够直接编译本轮V85图。设备目录中的V85 Stub/Skel补齐了运行时；`libQnnHtpPrepare.so`也来自相同2.48.1 SDK。未混用2.51运行库。设备端`LD_LIBRARY_PATH`指向独立运行目录，`ADSP_LIBRARY_PATH`包含该目录、`/odm/lib64/rfsa`及系统DSP搜索目录。

**兼容性测试**：每个输入3次预热＋20次计时，profile关闭，下表为同一轮20次调用的均值，不是跨轮性能中位数。客户端输入输出均为FP32，包含ExecuTorch调用、转换及拷贝；不含图创建和模型加载。所有新图均断言只有一个`QnnBackend` delegate，plan中无host operator。

| 测试 | 输入/形状 | 平均调用耗时 | 输出检查 |
|---|---|---:|---|
| FP16 Add | 三组`[1,4,8,8]`输入 | 1.107/0.857/0.970ms | 三组全部逐元素一致 |
| 当前`ChunkGDN` | AR128，32 heads，K/V维128，warm state | **4.1909ms** | 对独立FP32 recurrence：attention **59.962dB**，state **59.022dB** |
| 真实层0专家0，W4A8 SwiGLU | `[1,128,2048]`；GU `[1024,2048]`，down `[2048,512]` | **1.43495ms** | 全部有限；对PyTorch QDQ 25.873dB |
| 相同专家，W4A16 SwiGLU | 相同输入和真实权重 | **1.23025ms** | 全部有限；对PyTorch QDQ 69.763dB |

GDN测试使用当前生产`ChunkGDN`代码，chunk64/inverse_leaf32，随机归一化q/k、非零初始state；参考用独立逐token FP32 recurrence，另保存相同PyTorch混合精度图输出。device对该PyTorch输出的SQNR为attention62.550dB/state60.012dB，最大绝对误差对FP32为1.5685e-5/1.9995e-4。此处不是完整linear-attention，没有输入/输出投影或MoE；也不是之前UINT16 state、不同heads/IO/内核配置下0.832ms测试的同口径复测。

专家权重来自`/hunter/model/Qwen/Qwen3.6-35B-A3B`，只读取第0层第0个专家，测试整个GU→SwiGLU→down。使用12组随机输入校准、W4 per-channel；这是格式及执行兼容性检查，**不是生产GPTQ/group32精度评估，不含router、shared expert、Top8混合或256专家读取**。对源FP32输出的SQNR为W4A8 11.118dB、W4A16 11.321dB，主要包含本次粗粒度W4校准的误差，不能用来替代原整层/整模PPL或SQNR。W4A8本次慢于W4A16也不能外推完整MoE吞吐；未根据权重存储位宽推断具体HMX指令。

**已有MoE图的实际阻塞**：拉取设备`qwen35_moe_sm8975_20261020_probe/candidate.bin`，metadata显示编译版本`v2.49.40.260810104044`、coreAPI2.38。使用本轮2.48.1库加载时，ExpertGather和DeltaNet的包注册接口均返回0，但`contextCreateFromBinary`返回**5000**，明确报`Using newer context binary on old SDK`。因此这份图不能仅靠替换V85 Stub/Skel使用，必须拿到源图并用2.48.1重新导出。注册接口返回0只证明路径已登记，不证明DSP侧算子执行成功；该旧MoE图本轮没有执行推理。

本轮开始时，上节四层原型目录`llama-output/qwen36-two-layer-residency-20261019`以及依赖的GGML优化库/bank目录已不在当前工作区，已向用户询问迁移/归档路径。没有恢复原型前，不报告SM8975的完整四层、两层驻留、UFS读取或完整模型性能，也没有把单专家1.43ms当成完整MoE耗时。

产物位于`llama-output/sm8975-sdk2481-device-20261020/`：`RESULTS.json`汇总全部比较，`MANIFEST.json`保存源码/主机绑定/库哈希，`DEVICE_LIBS.txt`记录实际部署库和设备，`existing-moe-probe.log`保存5000失败原因，`candidate-metadata.json`保存旧图版本；各测试目录有独立PTE/context、输入、FP32/QDQ参考、device输出和runner日志。`source/build_host.sh`构建2.48.1绑定，`export_gdn.py`与`export_expert.py`重建新图，`run_case.py`部署并验证。生产SDK默认值、模型量化配置和用户已有改动未覆盖；未commit/push。

### SM8975/V85：重建真实四层 MoE 与两层驻留测试（2026-10-20）

用户确认旧原型已删除并允许重新导出后，从`/hunter/model/Qwen/Qwen3.6-35B-A3B`重新读取真实第0～3层权重，完成三层GDN＋一层full attention、四套router/shared和每层256专家/真实Top8的设备测试。SDK固定为`/hunter/qnn_sdk/src/backends/qnn/2.48.1.260715`，设备为`A7SR026915000282`、SM8975/V85；运行库继续使用上一节核对过的ODM HTP/System/V85 Stub/Skel。没有使用2.49生成的旧context。

**本轮重建版本与历史版本的区别**：已删除的不只有PTE，也包括原GGML专家优化内核及GPTQ bank。本轮重新实现了独立FastRPC整数专家服务，权重为packed INT4，激活动态A8，使用原生N4×UINT8 HMX；GU按1024维、down按512维量化，FP32 SIMD后处理和固定Top8槽顺序合并。专家没有改用FP16 MatMul。新bank采用CUDA激活加权MSE的RTN网格，**没有恢复成历史GPTQ bank**；router/shared使用FP16 QNN图。因此下面是新重建版本内部的受控比较，不能将与历史35.31ms或342.94ms的差异全部归因于SM8975硬件。历史35.31ms还使用了不同的冷full-attention图。

注意力沿用当前优化配置：K/z及depthwise conv W8，其他投影W4/group32、A16，稳定微小weight scale并对out_proj做fixed-grid GPTQ；prefill GDN使用当前`ChunkGDN`的FP16内部计算。conv/recurrent state和full-attention K/V边界均保持UINT16。四个attention PTE及四个router/shared PTE均检查为单个QNN delegate，无host operator；专家由独立原生服务执行，尚未拼成完整35B的单一PTE。

**计时范围**：从已生成的真实token embedding开始，经过四个完整attention/MoE block，包含两侧norm、两次残差、跨图IO、同步及cache反馈。128-token输入是一次完整四层prefill；1024-token输入按8个AR128块处理。tokenizer、embedding lookup、最终norm/LM head、context创建和一次性router/shared装载在计时外。全驻留专家预装载在计时外；两种换入调度包含第一次专家读取。每组1次预热＋3次计时，正式数据关闭细粒度profiling，输出和cache快照及`fsync`均在计时外。

| 输入及调度 | 四层整体中位 | 最小～最大 | 专家装载次数 | 计时内物理读取 | 专家槽分配 |
|---|---:|---:|---:|---:|---:|
| 128 token，四层全驻留 | **53.668ms** | 52.890～56.375ms | 0 | 0 | 1600.14MiB |
| 1024 token，四层全驻留 | **412.142ms** | 411.990～412.192ms | 0 | 0 | 1600.14MiB |
| 1024 token，两层处理完8块再换组 | **966.963ms** | 934.400～1023.812ms | 4 | **1.56264GiB** | **800.07MiB** |
| 1024 token，每块依次换入两组 | **4764.748ms** | 4751.118～4945.724ms | 32 | **12.50110GiB** | **800.07MiB** |

两层驻留相对相同换入策略的逐块对照**加速4.928倍**，物理读取严格减少为八分之一：`1677869056B`对`13422952448B`。读取计数来自`/proc/self/io::read_bytes`，不是CPU页缓存命中数。每次加载一层全部256专家，计算仍只执行真实Top8；没有热门专家缓存、预测路由、丢弃小score或跨层IO预取。专家bank文件通过手机内部UFS的`O_DIRECT`读取，两条`pread`线程各用8MiB暂存并复制到共享专家槽，CPU发布后同步DMA buffer，DSP换入后失效旧bank缓存。

| 1024-token调度 | attention中位 | MoE中位 | cache反馈中位 | 读取及CPU复制中位 |
|---|---:|---:|---:|---:|
| 四层全驻留 | 132.941ms | 270.299ms | 6.525ms | 0 |
| 两层驻留 | 136.275ms | 306.702ms | 9.444ms | 511.026ms |
| 逐块换入 | 143.974ms | 508.910ms | 17.176ms | 4089.124ms |

各段独立中位不可严格相加成总中位；MoE段包含router/shared、整数专家、末次残差和交接，DSP bank缓存失效也计入该段。两种换入方式的有效读取/复制吞吐都约**3.06GiB/s**，与本轮直接读取的字节数一致。这个数包含文件读取及复制，不是存储器裸峰值或硬件总线PMU。两层驻留的主要收益来自复用8个块的专家权重，未证明UFS已经达到硬件峰值。800.07MiB仅为两套专家槽，此外还有四层attention常量/context、router/shared、全部state/KV、原生工作区、IO暂存和hidden，不是进程或整机总DRAM。

**缓存与数值验证**：

- 三层GDN分别维护自己的conv/recurrent state；full attention保留896条历史K/V、使用绝对RoPE位置和因果mask。cache输出按各自校准scale/offset映射到输入网格；网格完全相同时直接复制，不把不同量化网格的码值直接反馈。
- CUDA FP32分块对完整1024-token参考的normalized SQNR，层0～3分别为**126.014/128.082/128.405/125.650dB**，最大绝对差不超过`1.24e-5`。这是同一真实权重FP32实现的因果分块检查，没有把量化输出当作FP32参考。
- 1024 token的三种调度各3次，**共9个完整hidden及全部最终cache逐位相同、输出全部有限**。两层驻留没有新增可见数值误差。
- 128-token最终hidden对FP32四层链为**12.924dB**；1024-token整体为**13.830dB**。1024-token按第0～7块分别为12.924/14.511/13.501/14.470/14.685/14.234/12.674/14.005dB。这里仍有原有量化误差，不能称为FP32精度；本轮未测完整模型PPL、图像输入、decode或40层。

**恢复执行路径时定位和修复的性能问题**：

1. 新QHPI实验的context在设备加载阶段返回5005，未用于正式结果；普通动态W4 bank输入还触发整块TCM迁移限制。最终使用FastRPC映射外部专家bank，QNN继续执行attention/router/shared，独立原生服务执行专家。没有据此断言V85不支持所有QHPI算子。
2. 初版原生专家服务使用DDR标量读取、普通复制和逐元素除法打包，层0专家计算段约123～126ms。分段profile显示主要时间在搬运/打包，HMX服务代码段仅约0.24～0.29ms；后者不是HMX硬件busy时间，重叠worker指标不可相加当总时间。
3. 改为DMA读取N4权重、输入及scale/code和，在VTCM做HVX最大值归约、A8打包、零点修正与输出缩放。修正了signed INT16解包的通道排列。验证样本上新输出对标量版**逐字节一致**，专家计算段降到约7.8～8.0ms；此数字不含QNN router/shared，不能当完整MoE或四层延迟。
4. 将主机cache反馈的网格相同路径改为直接复制，其他路径先批量复制再在普通主机内存中查表。四层AR128的cache反馈由探索版约35ms降到正式中位1.046ms。首次残差及跨层hidden拷贝仍包含在整体计时中。

新原型位于`llama-output/qwen36-four-layer-sm8975-20261020/`。`RESULTS.json`含范围、精度和汇总，`RESULTS.csv`和`SAMPLES.csv`保留汇总/逐样本时间，`CAUSAL_PARITY.json`保留FP32分块检查，`MANIFEST.json`记录源码、二进制、输入和bank的SHA256；正式日志为`final-resident128.log`、`final-resident.log`、`final-group.log`、`final-chunk.log`，`results/`保留9组完整输出/cache。

实现入口：`source/export_fronts.py`导出校准过的attention，`prepare_banks.py`生成真实专家N4 bank，`export_moe.py`导出router/shared；`IntegerMoE.cpp::execute/projection/prepare_activations`实现原生专家调度、DMA/HMX和HVX量化，`integer_rpc_impl.cpp`负责VTCM/QuRT/FastRPC，`residency.cpp`负责两层驻留及缓存反馈，`qnn_front.h`负责QNN元数据/共享IO。原生服务仍有进一步优化空间，例如持久worker及双缓冲预取；没有宣称恢复了已删除旧内核的全部优化。

环境按2.48.1设置后，重建依次运行`prepare_reference.py`、`export_fronts.py`、`prepare_banks.py`、`export_moe.py`、`build_native.sh`、`build_runner.sh`。`source/build_host.sh`可独立构建2.48.1绑定；本轮绑定副本与实际校准文本也保存在新目录内。已有产物可用以下命令复测两层驻留，另外将`--mode`改成`resident`或`chunk`进行相同口径对照：

```bash
conda activate et_npu
python llama-output/qwen36-four-layer-sm8975-20261020/source/run_device.py \
  --setup --mode group --samples 3 --tokens 1024 --name rerun-group
```

该脚本从设备ODM路径补齐四个运行库，并推送同一SDK的Prepare及本轮产物；原生库使用Hexagon SDK6.7.0.0/Tools21.0.04的V85编译器。`--setup`仅首次部署需要；默认不并行导出/测试。正式`llama.py`、生产量化配置和其他用户改动未覆盖；未commit/push。

### SM8975：纠正重建版运行路径的性能回退（2026-10-20）

用户指出上一节性能不符合预期后，保留原始二进制，在**同一台SM8975、同一SDK2.48.1、同一组四层PTE/N4 bank和真实输入**上重新建立优化前后对照。没有重新校准、更改量化精度、减少Top8或修改attention图。上一节的53.668ms来自另一批3个样本，本轮原始版5个AR128样本的中位为52.688ms；原记录保留，不用新的采样覆盖旧数据。

本轮发现的是重建实现的适配和搬运问题。它仍不能用于判断SM8975与历史SM8750的硬件优劣：历史35.31ms使用的原生内核及GPTQ/裁剪bank已经删除，本轮的MSE N4 bank与原生服务是另一套实现。

**实际资源与功耗配置检查**：设备API返回8个128-byte HVX单元、12个硬件线程、12MiB VTCM；此前重建版固定6个HVX worker、4MiB VTCM。诊断时Q6为2304MHz、HMX为1824MHz；补充原生服务独立的compute client/core/bus/HMX性能投票后，观测频率没有进一步上升。因此没有把问题归因于“没有升频”。投票在服务关闭时销毁，VTCM/HMX仍在每次调用完成后归还给QNN。

**排除的开销**：原始版profile中，VTCM申请约0.04～0.05ms、创建7个线程约0.07～0.09ms、释放资源约0.08ms，远小于整个专家调用。线程创建没有在本轮改成持久线程池。HMX服务代码段约0.24～0.28ms，这不是硬件busy计数，不能直接当作HMX利用率。worker计时互相重叠，也不能相加当总耗时。

**保留的修改**：

1. **批量DMA与双缓冲**：原来每个N32块同步搬运，再准备bias、等待HMX、做后处理。现在每个worker使用两块256KiB权重缓冲，当前批参与HMX计算/缩放时预取下一批。GU一次覆盖8个N32块，512维down一次覆盖32个N32块，减少小DMA事务和串行等待。DMA仍直接读取packed N4，专家仍为整数N4×A8 HMX。
2. **按设备资源配置工作区**：改为8个HVX worker＋1个HMX服务线程，每个worker划分1MiB，共申请8MiB VTCM。尝试过使用12MiB、512KiB双缓冲，未优于8MiB版本，因此没有采用更大的申请。6→8 worker的单独收益有限，主要收益来自搬运路径，不能把全部加速归因于并行度。
3. **只失效实际缓存的bank区域**：DMA描述符启用了source cache bypass，专家权重/scale/code和/专家header都由DMA读入VTCM。DSP标量访问DDR的区域只有全局magic和offset table，位于开头4KiB。换入时将原先整块400MiB bank的DCACHE invalidate缩到这4KiB；CPU发布共享buffer的同步保持不变。该修改针对当前DMA-only权重路径，不可直接推广到读取权重的其他实现。
4. **共享跨图IO**：router输入直接注册attention的normalized hidden输出buffer；原生专家直接使用同一hidden及router输出的ids/scores/shared buffer。去掉重复的主机复制，保留各context的内存注册及拥有者生命周期，并检查交接的shape/dtype一致。
5. **读取与CPU复制重叠**：旧路径的两条线程各自`pread→memcpy→pread`。新路径使用两条读取线程、两条复制线程、4个8MiB暂存块和有界队列，使下一块读取与上一块复制重叠。专家槽仍为两套800.07MiB，IO暂存从16MiB增加到32MiB；不是将四层专家全部常驻，也没有减少读取量。两层处理完所有AR128块后才换组的调度保持不变。

关闭细粒度profiling，核心8次、AR128全驻留5次、其他3次计时，各自先预热1次。范围继续是已生成embedding进入四个完整attention/MoE block至hidden输出，包含残差、交接、cache反馈；不包含tokenizer/embedding lookup/最终head/context创建。resident预装载在计时外，group包含全部换入。快照及fsync在计时外。

| 相同设备上的对照 | 原始版中位 | 修改后中位 | 耗时下降 | 原始/修改后最小～最大 |
|---|---:|---:|---:|---|
| 层0原生专家核心，AR128，不含QNN router/shared | 7.784ms | **6.381ms** | **18.0%** | 7.705～8.144 / 6.314～6.456ms |
| 四层完整AR128，全驻留 | 52.688ms | **45.608ms** | **13.4%** | 51.995～54.919 / 45.099～47.335ms |
| 四层1024 token，全驻留，8个AR128块 | 412.269ms | **356.963ms** | **13.4%** | 412.054～413.062 / 356.684～357.138ms |
| 四层1024 token，两层换入，含物理IO | 1061.035ms | **855.596ms** | **19.4%** | 1002.628～1065.088 / 850.876～876.035ms |

前两种全驻留结果对应`source/runtime-fixed/`二进制；加入IO流水线后保存为`source/runtime-fixed-io-pipeline/`，只改变计时外的resident预装载和group读取实现，专家内核与量化保持相同。后一项使用相邻重跑批次`verify-before-group1024`与`verify-after-pipeline-repeat1024`。IO波动明显，上一节原始版group中位为966.963ms；新版pipeline两批中位分别为**850.012/855.596ms**，不能把其中某批差值当作固定硬件收益。

进一步拆分1024-token全驻留，attention约132.5ms保持相近，MoE段从270.734ms降到215.495ms；其收益来自专家和交接。加入IO流水线前，同一新专家版本的group读取/复制中位601.642ms、整体972.093ms；加入后两批读取/复制中位481.234/483.661ms。每批始终读取4层、`1677869056B`，仍为1.56264GiB；有效读取/复制吞吐约3.23GiB/s，包含复制和同步，不是UFS裸峰值。没有在计时中利用页缓存命中冒充物理读取。

**为何增加算力没有按比例缩短时间**：层0纯专家测试使用的真实参考路由在128 token中选中212个不同专家，1024个token/expert组合平均只有4.83个token/专家。按当前bank和16-row分批推算，仅N4矩阵读取就约319.5MiB，另外还有metadata、输入、路由结果写回/合并。这里的212是纯专家参考输入统计，不代表量化四层链每一层的实际路由数。当前主要受权重搬运及细粒度调度限制，不能用芯片TOPS比例预测这个原型的速度，也没有依据宣称已达到SM8975的上限。

**数值与失败尝试**：

- 本轮分析脚本验证了原始/修改后共30条结果记录，其中完整四层28个hidden及全部最终cache、两个纯专家输出，全部与原始正确输出逐字节一致；hidden全部有限。包括6次新IO流水线运行，没有因缩小invalidate范围或共享buffer引入旧数据。
- 原有最终hidden SQNR保持AR128的12.924dB、1024 token的13.830dB，没有提高或降低；本轮没有重新测试完整模型PPL或decode。
- `O_DIRECT`直接读入rpcmem专家槽返回`EFAULT(14)`，没有采用；选用正常对齐暂存块＋共享槽复制流水线。
- 单独请求QNN `DDR_PERF_MODE`返回13006（`ERROR_FAILED`），没有启用，也没有据此判断V85普遍不支持该特性。该实验判断分支没有留在最终runner。

`RUNTIME_FIX_RESULTS.json`、`RUNTIME_FIX_SUMMARIES.csv`、`RUNTIME_FIX_SAMPLES.csv`保留本轮范围、逐样本值、精度和对照；`RUNTIME_FIX_MANIFEST.json`记录原始/修改后源码与二进制SHA256。原始bank/context/输入哈希继续参考上一节`MANIFEST.json`。`source/before-runtime-fix/`保存原始二进制及探索版本；`source/analyze_runtime_fix.py`可重新核对全部输出。细粒度日志包括`core-runtime-before-profile.log`、`hardware-capabilities-profile.log`、`core-8x256-profile.log`；这些开启profile的时间不用于上表。

本轮修改仅保留在上述四层原型与测试记录中，未接入正式`llama.py`、未改变生产模型量化配置，未commit/push。


### SM8975：补回冷图、VTCM融合与路由优化（2026-10-21）

继续处理用户指出的“历史SM8750四层约35.6ms，新SM8975反而慢”的问题。实验仍是本地`Qwen3.6-35B-A3B`真实第0～3层、三层GDN＋一层full attention、每层256专家/Top8、AR128，设备`A7SR026915000282`、V85、QNN SDK 2.48.1.260715及设备ODM库。全部修改保留在独立原型`llama-output/qwen36-four-layer-sm8975-20261020/source/`，没有接入生产`llama.py`，没有修改生产模型量化配置，未commit/push。

这轮确认存在可修复的软件开销，不能将上一轮45.6ms解释为硬件上限。校准参数、位宽和Top8保持不变；专家仍为原生N4×动态A8整数HMX，router/shared仍为FP16。下述量化后零通道压缩保留已有整数码和scale，未重新GPTQ或重新校准。

#### 找到并保留的改动

1. **补齐量化后零GU通道压缩**。旧重建bank每层400.035MiB，保留了很多已量化为零的gate/up行。扫描冻结N4码，只移除gate或up整行均为零的SwiGLU通道，并同步移除down对应列；保留其余行的顺序、码、scale与down动态量化网格。不依据“down列为零”单独删通道，避免改变中间激活最大值和A8 scale。layer0去除54257/131072个GU通道；四层bank文件共从1677869056B缩到1526194176B。layer0参考路由的N4矩阵读取推算由319.5MiB降至205.8125MiB（cap16→32也在其中），这不是DDR硬件计数。压缩bank仍按128B/4096B对齐，支持原有O_DIRECT换入。
2. **输入A8只准备一次**。128个token的hidden一次性动态量化，8个worker各处理16行；之后按专家路由gather和重排HMX输入，消除Top8重复读hidden及重复求最大值/量化。A8码在VTCM使用I16容器存放，节省工作区，实际计算仍是A8，没有改成16位激活运算。
3. **GU后处理与SwiGLU融合，复用临时空间**。gate块先写入工作区，up块到达时直接生成gate×sigmoid(gate)×up，覆盖gate；GU与down的metadata寿命不重叠，共用40KiB。每专家分批上限32行，匹配HMX的M32 tile，减少超过16行时的重复权重读取。
4. **使用全部12MiB VTCM保存固定Top8结果**。最终布局为8×448KiB worker区、512KiB输入码和8MiB路由结果。全部Top8槽留在片上，最后按slot0→7固定顺序相加，消除8MiB结果写DDR后再读回的路径。每worker双64KiB权重缓冲，GU/down分别最多批量提交2/8个N32块，减少HMX请求同步。曾试过1MiB到达顺序累加，但微小FP32顺序变化会被后续PTQ/TopK放大，最终未采用。
5. **修正cold AR128误走warm FA图**。原重建版即使只有128个新token且无历史，第3层仍计算896个masked历史位置＋128个新位置。本轮从冻结Q/DQ图去掉历史K/V concat，仅将相关KV形状改为128，权重、scale/zero point及因果语义保持。该独立cold图仅用于128-token冷测试，1024-token实验继续使用原1024列warm图。CPU冻结图对照误差约96～99dB、cache相等；设备上本次所有最终hidden/cache与原warm路径逐字节相同。没有在生产PTE中重新加入短上下文方法。
6. **FastRPC持久映射与描述符调用**。由每次传入多块MiB级sequence，改为初始化/首次使用时映射buffer，调用时只传fd、bank代次和槽位。复用QNN已注册的hidden、ids/scores/shared、residual；明确处理`AEE_EALREADY(26)`，其他错误仍检查。DSP共享/残差输入失效与输出flush、CPU读同步保留。最终残差在DSP按原括号顺序执行；中间三层hidden直接绑定原生输出供下一层QNN读取，CPU只读取最后一层，不再每层复制。
7. **持久QuRT线程与一次路由分组**。8个HVX worker和1个HMX服务线程在原生服务打开时创建，以信号量等待任务；VTCM/HMX仍逐调用申请/释放，空闲线程不持有HVX/HMX。原来每个专家都扫描128×8个索引，改为worker0一次构建按专家分组的CSR表，再按各专家的offset读取token/slot。表保留原token/slot顺序，放在worker0原有padding中，不增加VTCM申请。
8. **修复共享映射退出顺序**。持久HAP映射曾使QNN退出时报告`fastrpc memory failed to unmap`。最终先关闭原生服务、归还HAP引用，再由QNN注销共享buffer；只解除自己创建的FastRPC映射，不解除QNN已有映射。纯专家、cold、1024连续prefill和两层换入复测均正常退出，unmap警告为0。

#### 正式性能与口径

计时从预生成embedding进入四层开始，到最终hidden完成，包括两侧norm、首次及最终残差、router/shared/Top8专家、运行时交接和cache反馈。排除tokenizer、embedding lookup、最终LM head、模型/context创建及输出文件落盘。resident权重预加载在计时外；group在计时内完整读取四层bank，两层处理全部8个AR128块后再换下一组。没有实际运行完整40层、decode或图片。

表格采用最终源码重新构建后的`release-v85-csr-*`批次；此前`final-v85-csr-*`批次也保留，未覆盖。全部正式结果关闭profiling。cold32次、warm128与1024 resident各8次，均预热6次；纯专家16次/预热6次；group3次/预热1次。早期探索批次的样本/预热数不同，逐批保存在CSV，不把它们合并成一个统计总体。

| 测试 | 原始重建版 | 上轮修复 | 本轮最终中位 | 本轮最小～最大 |
|---|---:|---:|---:|---:|
| 层0纯原生专家核心，AR128，不含QNN router/shared | 7.784ms | 6.381ms | **3.521ms** | 3.511～3.563ms |
| 四层AR128，保持同一warm FA图 | 52.688ms | 45.608ms | **38.587ms** | 38.542～38.667ms |
| 四层cold AR128，使用修正的128列FA图 | — | — | **35.614ms** | 35.483～36.163ms |
| 四层1024 token，8个AR128块，全驻留 | 412.269ms | 356.963ms | **305.201ms** | 304.839～307.588ms |
| 四层1024 token，两层驻留，含全部物理IO | 1061.035ms | 855.596ms | **810.180ms** | 797.743～823.712ms |

相同warm图的四层AR128对原始版减少26.8%，连续1024减少26.0%。修正cold图后为35.614ms，相对原始52.688ms的总下降32.4%包含了cold/warm图差异，不能全部算成相同计算量的内核加速。最终cold的attention、MoE、cache反馈分别约13.169/21.531/0.802ms；这些是各段独立中位，不能直接相加冒充总时间。warm FA与cold FA差约3.0ms，定位到了之前比较中的实际多余计算。

1024 resident中，attention约128.888ms，MoE约169.061ms，cache反馈约6.346ms；attention计算图未改变，主要收益来自专家与数据交接。group读/复制中位487.553ms，物理读取1526194176B（与四个新bank文件完全一致），有效读/复制约2.92GiB/s。上一轮读取量为1677869056B，不能把group全部收益归因于吞吐提高。本轮较早的pool版本测得763.875ms，中间CSR批次为778.797ms，最终重新构建版本为810.180ms；IO波动存在，没有挑更快的一批替代最终记录。resident专家槽仍按最大bank大小分配，约1598.59MiB，group双槽约799.30MiB；文件变小不等于当前resident槽按同等比例减少。

#### 尚余瓶颈及历史SM8750对照

原生profile确认Q6 2304MHz、HMX 1824MHz、8个HVX单元、12个硬件线程和12MiB VTCM。持久线程任务发送约0.006～0.010ms；VTCM申请约0.040～0.049ms、释放约0.106～0.110ms，均逐层归还。以profile的layer1为例，worker观测的DMA等待约3.49～3.75ms，而其总专家处理区间约5.06～5.11ms；HMX请求等待约0.06～0.08ms，F32后处理约0.63～0.92ms。worker区间相互重叠，不能求和；DMA等待也包含竞争/调度，不是纯硬件DDR计数。HMX服务代码段约0.23～0.28ms，不等于硬件busy利用率。当前剩余重点仍是大量细粒度专家权重搬运，不能用TOPS比例预测速度。

历史SM8750修复后cold四层为35.596172ms，1024 resident为342.94ms；本轮SM8975对应35.614/305.201ms。cold已恢复到相近水平，**没有验证出显著的cold代际加速**；1024较历史数值低约11.0%。历史原生库/GPTQ bank已经删除，当前MSE N4 bank、量化输入与真实路由不同；历史计时从首层hidden就绪开始，本轮还计入首层buffer写入及显式cache反馈，因此这不是同二进制/同负载的芯片比较。保留原始总时间，不通过减去某段估计值声称严格同口径的硬件加速。

未采用的尝试：QNN HVX线程4/6/8覆盖没有收益；将输入码进一步压成I8、给权重换成双128KiB缓冲，数值相同但cold中位37.129ms，慢于I16/双64KiB版本36.459ms，已回退；DMA直接暂存shared/residual并写回，纯专家正确但完整链触发DSP异常，已回退。QNN详细profile明显扰动attention计时，所有profile开启的整体时间均不计入正式表格。

#### 正确性、文件和复现

最终正式cold32次、warm8次、1024 resident8次及group3次，共51个四层输出及全部408份cache，与原始正确参考逐字节一致；退出顺序修复后另12次四层输出及全部cache也一致，且无unmap警告。纯专家保存的最终输出逐字节一致。最终hidden对原FP32参考SQNR保持AR128 **12.923572dB**、1024 **13.830287dB**，全部有限。本轮解决性能回退，未声称提高量化精度、解决完整模型PPL或验证40层可用性。

`V85_SECOND_PASS_RESULTS.json`、`V85_SECOND_PASS_SUMMARIES.csv`和`V85_SECOND_PASS_SAMPLES.csv`保存各批原始样本/统计、精度及范围，`V85_SECOND_PASS_MANIFEST.json`保存最终源码、二进制、cold图和compact bank的SHA256。`source/analyze_v85_second_pass.py`重新检查所有正式输出；`source/v85-final-csr/`保存最终版本，`source/before-v85-second-pass/`保存探索对照。`final-v85-csr-profile-cold128.log`是原生阶段profile，正式计时参考`release-v85-csr-cold128.log`、`release-v85-csr-warm128.log`、`release-v85-csr-resident1024.log`和`release-v85-csr-group1024.log`。更早`final-v85-profile-cold128.log`因库目标文件名未覆盖，原生计数为零，不作为原生profile证据；部署工具已改为明确目标文件名。

示例（已存在设备输入/context/compact bank时）：

```bash
# 32次cold AR128；正式计时无profile
python llama-output/qwen36-four-layer-sm8975-20261020/source/run_device.py \
  --compact --chain-hidden --cold-fa --mode resident --tokens 128 \
  --samples 32 --warmups 6 --name cold128-repeat

# 连续8个AR128块；保留warm FA cache
python llama-output/qwen36-four-layer-sm8975-20261020/source/run_device.py \
  --compact --chain-hidden --mode resident --tokens 1024 \
  --samples 8 --warmups 6 --name resident1024-repeat
```

### SM8975：Qwen3.6-35B-A3B 全部40层导出及分组换入实测（2026-10-21）

本轮使用 `/hunter/model/Qwen/Qwen3.6-35B-A3B` 的全部真实40个decoder block，30层GDN＋10层full attention；没有复制前四层权重来代替后36层。每层256专家、Top8、hidden=2048，AR固定128。设备为 `A7SR026915000282`／SM8975 V85，QNN SDK `2.48.1.260715`，运行使用设备ODM的HTP/System/V85 Stub/Skel。

#### 导出产物和精度配置

- 独立原型目录：`llama-output/qwen36-full40-sm8975-20261021/`。每层保留 `attention.pte`、`moe-front.pte` 及提取的QNN context，专家以外部 `experts.n4` bank供原生整数kernel按索引访问。这是40层分层导出、串联执行的原型，尚不是可直接交给生产 `llama.py` 的单文件完整语言模型PTE。
- Attention维持W4A16/block32，合并K/z投影及depthwise conv为W8，GDN out_proj做GPTQ，conv/recurrent/full-attention K/V的外部IO均为UINT16。GDN沿用已优化的AR128路径。MoE专家为N4权重×动态A8整数HMX，router/shared为FP16，Top8及累加顺序保持原样。专家bank使用激活加权MSE量化，并非已删除的历史GPTQ专家bank；只压缩已有N4码中gate或up整行为零的通道，不改变保留码和scale。
- 复用原四层及其冻结校准结果，串行新增36层和9个cold FA context。新增153个阶段的墙钟记录合计 **1826.861秒（30.45分钟）**，未计ADB部署和已有四层的历史导出；最大单阶段进程RSS约 **2.831GiB**。不是同时加载35B模型或并行导出多个模型。
- 40个常规attention与40个router/shared图均检查输入输出顺序，导出检查为单QNN delegate。40个warm context文件共1237551152B，全部40层专家bank为 **16625393664B（15.484GiB）**。还有10个独立cold FA图，仅供无历史128-token测试。1024-token实验使用1024列warm FA图，不将cold图用于有历史的输入。

#### 全层运行遇到的问题及最终内存策略

最初同时保留80个QNN context时，context创建结束后首次400MiB专家bank的DSP映射失败，原生返回 `integer_rpc_run_mapped:25`，设备日志出现QuRT Error12。失败时系统MemAvailable约631MiB。此证据指向context并存引起的DSP映射资源问题，尚未区分地址空间上限、碎片与其他DSP映射配额，不能解释成芯片计算能力上限。SDK多context共享spill/fill试验没有消除故障，未纳入最终计时配置。

最终同时只保留 **两层QNN context＋两个专家bank槽**。每组两层把本次全部token处理完成，再释放context、换下一组；每层的cache和IO buffer保留，避免丢失状态。专家槽共 **838934528B（800.07MiB）**。1024-token测试中同一组完成8个AR128块后才换组，每轮仍只读40个bank，不重复读取8遍。

新增的切换接口先结束CPU输出读取、释放原生HAP IO引用，再让QNN注销同一DMA buffer并释放context；否则下一组重新注册共享hidden会报 `memRegister:8003`。两bank的映射和原生工作线程保持复用。原生IO映射表容量也由64扩到256，支持初始化全部40层IO的对照试验。最终两组正式测试均正常退出，无FATAL和FastRPC unmap错误。

#### 正式结果：40层串联，不外推

每项先预热1次，再记录3次。计时包含两侧norm、Attention投影/GDN或FA、首次残差、router/shared/Top8专家、最终残差、cache反馈、必要hidden交接、全部专家bank读取/复制/DMA同步，以及每组QNN context装载/注册/释放和原生映射交接。关闭profiling；输出文件保存、校验、初始元数据/IO buffer准备在计时外。

输入是已准备好的embedding。tokenizer、embedding lookup、最后的model norm、LM head、采样和图片encoder不在时间里；没有测试decode。下表“计算与交接”是逐样本从总时间减去实测bank和context区间，**不是已实现的40层全驻留性能**。各列为独立中位数，不能要求它们严格相加等于总时间。

| 输入 | 128 token，无历史 | 1024 token，8×AR128 |
|---|---:|---:|
| 40层总时间中位 | **7.908352s** | **10.610759s** |
| 总时间最小～最大 | 7.862950～8.246200s | 10.511542～10.666011s |
| 专家bank读取/复制/同步 | 4.779767s | 5.046029s |
| QNN context装载/注册/释放与映射交接 | 2.775319s | 2.766988s |
| 计算与数据交接 | 354.657ms | 2823.157ms |
| Attention，含投影、两侧norm和首次残差 | 143.297ms | 1327.372ms |
| Router/shared | 15.366ms | 109.135ms |
| 原生N4×A8 Top8专家与融合最终残差 | 170.689ms | 1219.565ms |
| Cache反馈 | 23.996ms | 115.304ms |
| 本次40层串联的总吞吐 | 16.19token/s | 96.51token/s |

全部6次正式样本的 `/proc/self/io` 物理读取量均为16625393664B，与40个bank的总文件大小完全一致。O_DIRECT读取，随后复制到双DMA槽并同步；这是整个读取/复制区间的有效吞吐，约3.24／3.07GiB/s，不能当成单独测得的UFS峰值。预热/初次验证有额外context文件物理读取，原日志保留。128-token原型首次跑通为9.596秒，随后context缓存等影响使正式运行降至7.86～8.25秒；这里报告固定预热1次后的全部3次，没有用最快单次替代中位数。

128-token最终总时间中约95.5%属于bank换入和context管理；1024-token通过每次换入处理更多AR128块摊薄开销，但总时间仍主要花在这两项。每轮换入约15.5GiB真实专家数据，没有因Top8而只从UFS读取8个专家；Top8只限制这轮DRAM→VTCM的专家计算。进一步优化应先减少实际换入字节、减少context重建或将其流水重叠，不能把这里的秒级总时间归因于GDN核心。

进程高水位RSS约1.294／1.309GiB，包含已触碰的context文件映射；不是整机或DSP内存峰值。正式运行记录到的最小系统MemAvailable约3.56／3.62GiB。前后thermalservice状态均为0；电池温度接口无输出，没有据此估计温度。

#### 正确性与记录

两层context窗口的四层128和1024结果（hidden＋全部8份cache）都与此前保留context版本逐字节一致。全部40层的两项正式测试，每项3次的最终hidden及全部80份cache均逐字节一致；输出全部有限。对同一输入的CUDA FP32分层参考，信噪比如下：

| 指标 | 128 token | 1024 token |
|---|---:|---:|
| 第40层block输出hidden SQNR | 11.486558dB | 10.773034dB |
| 用同一FP32最终RMSNorm归一化后hidden SQNR | 10.479537dB | 10.136652dB |

最终norm只用于离线比较，不计入时间。以上没有验证完整模型PPL、LM head输出或生成文本质量。

`RESULTS.json` 保存范围、位宽、内存/热状态及精度；`SAMPLES.csv`、`LAYERS.csv`、`LOADS.csv`、`SUMMARIES.csv` 保存逐次、逐层、逐bank数据；`EXPORT_STAGES.csv` 保存串行导出记录，`MANIFEST.json` 保存各层PTE、context、bank及源码/二进制SHA256。原始日志为 `full40-cold128.log`、`full40-prefill1024.log`。首次全context失败保存在 `initial-full-context-failure/`，共享spill/fill及映射失败试验也保留独立命名。

复测（需要已部署的该目录、指定设备和et_npu环境）：

```bash
python llama-output/qwen36-full40-sm8975-20261021/source/run_device.py \
  --layers 40 --tokens 128 --mode group --context-window --cold-fa \
  --breakdown --warmups 1 --samples 3 --name full40-repeat-cold128

python llama-output/qwen36-full40-sm8975-20261021/source/run_device.py \
  --layers 40 --tokens 1024 --mode group --context-window \
  --breakdown --warmups 1 --samples 3 --name full40-repeat-prefill1024
```

本轮只在独立原型中修改导出/运行脚本、QNN context生命周期和原生IO生命周期，未接入生产 `llama.py`，没有改动生产量化recipe，未commit/push。

### SM8975：40层MoE连续AR1 decode补测（2026-10-21）

在全部40层prefill原型上增加真实AR1图，模型仍为 `/hunter/model/Qwen/Qwen3.6-35B-A3B`，设备 `A7SR026915000282`／SM8975 V85，QNN SDK `2.48.1.260715`，运行使用设备ODM库。仍保留独立原型，未接入生产 `llama.py`，未commit/push。

#### 测试口径和实现

输入状态来自之前手机上实际完成的128-token全40层prefill：每层conv/recurrent或full-attention K/V cache均接入AR1。连续计算位置128～143，共16个新位置，每步依次经过全部40层，并反馈本步的全部cache。下一位置输入使用同一真实语料的已知embedding（teacher forcing）；这是decoder block链的AR1测试，不含LM head、采样、tokenizer、embedding lookup、最终model norm或图片encoder。没有执行文本生成或测量PPL。

40层的 `attention.pte`、`moe-front.pte` 均导出为AR1，每图检查为单QNN delegate，输入输出顺序保存于 `FRONT.json`／`io.bin`。Attention从已校准的AR128冻结QDQ图specialize而来，直接复用已有整数权重码、GPTQ结果、scale/zero point，没有重新校准。W4A16/block32、K/z与depthwise conv W8，以及UINT16 conv/recurrent/full-attention K/V IO保持。AR1 GDN编译了SDK2.48.1的V85 `DeltaNetCoreVtcm`，router/shared保持FP16。GDN自定义图与标准router图分进程编译，避开主机SDK混合配置连续编译时遇到的崩溃。

原生专家kernel专门执行一行：N4权重×动态A8整数HMX，Top8和固定槽位累加顺序不变。运行时先完成本层router，再依据真实Top8索引，从同一冻结 `experts.n4` 文件中O_DIRECT读取这8个专家的原始payload，重写槽内offset表，保留payload码、scale、sums和专家ID。读取按4096B对齐，计时包括对齐多读、复制、DMA同步和路由索引处理。专家双槽总大小 **26288128B（25.070MiB）**，没有跨层专家缓存，因此后续token仍会重新按路由读取。

AR1只需8×448KiB worker区、单行输入码及8×2048个F32路由结果。沿用4MiB路由区起点时，实际末端为4MiB＋64KiB；工作区由prefill的12MiB收紧到 **5MiB**。四层连续两步的hidden及8份cache与12MiB版本逐字节一致。

#### 先做的功能和资源验证

- 四层、连续两步，Top8选择性读取与完整bank读取的hidden和8份cache逐字节一致。对照的O_DIRECT字节数分别为95059968B和3052388352B。常驻context与两层切换的输出也逐字节一致。这些短测用于功能验证，不把四层速度外推成40层速度。
- 单层 warm-cache AR1 与同一初始状态的冻结QDQ CPU参考比较：post-attention norm输出35.112dB，首次残差39.049dB，conv输出46.871dB，recurrent state输出55.936dB，全部有限；详情 `AR1_CPU_QDQ_CHECK.json`。四层连续两步的最终hidden对FP32参考为16.931dB。
- 尝试将40层的80个QNN context都常驻。12MiB专家工作区版本返回原生error15；收紧到5MiB后仍在专家bank的DSP映射处返回error25。当时原生资源查询显示总/可用VTCM均为12MiB。该常驻试验没有得到有效完整输出，不列性能结果。现有context组织方式仍存在DSP映射资源限制，不能将失败解释为VTCM容量用满或芯片计算速度上限。
- 最终有效测试采用两层context窗口：每个AR1 token都依次换入20组context。切换前先释放原生HAP IO引用，再注销QNN buffer，保留40层的状态buffer。context装载/注册/释放、映射交接均纳入本次总时间。

#### 正式性能

预热1轮16步，再测3轮16步；关闭profiling和中间输出采集。每轮重置为同一真实prefill128状态，但轮内16步持续更新cache。中位结果如下，各分段独立取中位，不能要求严格相加等于总中位。

| 项目 | 16步整轮 | 平均每步 |
|---|---:|---:|
| **40层总耗时** | **33.746597s** | **2109.162ms** |
| context装载/注册/释放及映射交接 | 25.684607s | 1605.288ms |
| Top8专家读取/复制/同步 | 6.228693s | 389.293ms |
| 计算与其他交接 | 1.774207s | 110.888ms |
| Attention，含投影、两侧norm和首次残差 | — | 65.617ms |
| router/shared | — | 8.933ms |
| 原生Top8专家及融合最终残差 | — | 25.254ms |
| cache反馈 | — | 10.685ms |
| **本次decoder链decode吞吐** | **0.474122 token/s** | — |

三轮总耗时分别为34.664919／33.687506／33.746597s；平均每token范围2105.469～2166.557ms。每轮16步共有640次本层Top8读取：逻辑payload **8315122176B**，4096B对齐实际物理读取 **8335388672B**，三轮计数相同。平均每token逻辑519695136B、物理520961792B，约496.828MiB／token。当前reader每层串行读8段，区间有效吞吐约1.25GiB/s，包含复制和同步，未单独测存储峰值。

context管理约占76%，专家换入约占18%，其余计算与交接约占5%。因此本次0.474 token/s主要受分层原型的context重建和外存换入影响；“计算与其他交接”只是从同一实测扣除装载区间，不能当作已测得的全权重/context常驻40层速度。

与上一轮prefill放在一起，范围均为prepared embedding到40层block输出；prefill一次换入处理全部本次输入，decode每个新位置都经过全部40层：

| 测试 | 真实测量总时间中位 | 本次范围内吞吐 |
|---|---:|---:|
| prefill128，无历史 | 7.908352s | 16.185 token/s |
| prefill1024，8×AR128 | 10.610759s | 96.506 token/s |
| prefill128状态后，连续16步AR1 | 33.746597s | 0.474122 token/s |

#### 正确性与产物

3轮的16个最终hidden及每轮全部80份最终cache逐字节一致，输出全部有限。对同一语料、同一位置的CUDA FP32分层参考，最终block hidden SQNR为 **9.520855dB**；离线用同一FP32最终RMSNorm比较为 **9.161381dB**。参考自身从FP32状态开始，本次从设备量化prefill状态开始，误差包含prefill累计误差，不能全部归因于AR1。单层冻结QDQ对照单独隔离了实现误差。没有以hidden SQNR替代完整模型PPL或文本质量验证。

有效正式运行正常退出，QNN error和FastRPC unmap错误均为0，前后thermalservice状态均为0。原型目录 `llama-output/qwen36-full40-sm8975-decode-20261021/` 内保存 `RESULTS.json`、`SAMPLES.csv`、`STEPS.csv`、`MANIFEST.json`，原始正式日志 `full40-decode-window.log`。`before-ar1-vtcm-size/` 保留12MiB工作区的试验、被停止的初始window运行及二进制；5MiB常驻失败单独记录于 `validate-ar1-full40-retained.log`，未混入正式样本。80个AR1 PTE、QNN context及专家索引均保留，专家bank只读复用此前40层目录。

复测（需要已部署数据和指定设备）：

```bash
python llama-output/qwen36-full40-sm8975-decode-20261021/source/run_decode.py \
  --layers 40 --steps 16 --samples 3 --warmups 1 \
  --name full40-decode-repeat
```

默认即为两层context窗口和Top8选择性读取。`--retain-contexts` 是资源验证选项，本次40层运行失败；`--full-bank` 用于已通过的四层功能对照，不作为decode最终读策略。

### SM8975：decode四层有界FIFO与后台预加载（2026-10-21）

按用户提出的“初始化4层、消费后出队并提前加载后续层”实现独立原型，模型仍为 `/hunter/model/Qwen/Qwen3.6-35B-A3B` 全部真实40层。SM8975 V85，设备 `A7SR026915000282`，QNN SDK `2.48.1.260715`＋设备ODM库。128-token设备prefill状态后连续16步AR1，固定同一真实语料的teacher-forced输入；不含LM head、采样、embedding lookup、最终model norm或图片encoder。未接入生产 `llama.py`，未commit/push。

#### 四层队列如何运行

1. 初始装载L0、L1、L2、L3。每层有Attention和router/shared两个QNN context，最多4层／8个活跃context；正在装载的层也占槽位。
2. 主线程按原顺序完成当前层的Attention、router、Top8专家和cache反馈。完成L0后释放该层自己的原生HAP IO引用，将槽位标为待回收；全链共用的hidden输出映射保留。
3. 主线程进入L1的Top8专家读取区间时，唤醒后台线程卸载L0、装入L4；之后依次用同一槽位装入L5、L6……。先释放旧context，再创建新context，峰值不超过4层。
4. 同一个token走完40层后，继续同一环形队列，提前准备下个token的L0～L3。40层的conv/recurrent/full-attention K/V state和IO buffer保持各自状态，context回收不清空cache。第16步结束时排空队列，最后的卸载/drain计入总时间。
5. 每层的真实Top8需要本层router输出，所以专家仍按实际路由O_DIRECT读取原始N4 payload。没有提前猜测路由，没有把后续层的256专家整体读入DRAM。双专家槽仍为26288128B（25.070MiB），专家kernel仍为N4×动态A8整数HMX／5MiB VTCM，其他量化位宽、校准参数、GPTQ码及输入全部保持。

这里的4层上限指活跃QNN context，40层的状态/IO缓冲及OS文件页缓存另外保留。只读context文件继续mmap；本轮没有测整机RSS峰值，不能用“4层”推算整个进程只有4层权重大小。

#### SDK线程规则和共享映射

SDK本地文档 `docs/QAIRT-Docs/QNN/general/api_usage_guidelines.html#multi-threading` 明确规定，同一handle及其parent/dependent handle的QNN API调用需要外部同步。当前同一backend/device下，以SDK互斥锁串行化context创建/注销、graph执行和共享IO相关操作。原生专家执行也放在同一门控下，保持共享映射和计算资源交接的确定顺序。

后台SDK操作安排在主线程的专家读盘区间。直接在消费结束后立即抢SDK锁，会阻塞下一层的Attention API，减少可重叠的I/O；因此队列显式区分“IO窗口允许装载”和“主线程等待队列就绪”两种允许生产的时机。后者保证队列缺项时能继续装入，不会双方等待。

增加按fd释放HAP引用的原生RPC接口：每层退役只释放自己的post-norm hidden、ids/scores/shared、residual映射，公共链输出保持映射；最后一个token最后一层完成后，先归还公共HAP引用，再排空QNN context。同步/后台队列都沿用这一生命周期，避免后台注销正在使用的共享buffer。

`source/layer_queue.h` 保存4槽FIFO与后台线程，`source/decode.cpp` 控制IO窗口和SDK等待计时，`source/native_core.h`／`integer_rpc_impl.cpp` 实现按fd释放。状态机为ready→running→retired→loading→ready，最终进入done；取消路径会先停止并join生产线程。所有工作保持在独立目录 `llama-output/qwen36-full40-sm8975-queue4-20261021/source/`。

#### 正式结果

三组均先预热一轮16步，再测3轮16步；同一二进制、模型、原生算子和数据。正式关闭host trace及QNN详细profiling。队列稳态时间排除首批4层装载，包含最终drain；另外列出加回首批装载的时间。各项独立取中位。

| 方案 | 16步总时间 | 稳态平均每token | 吞吐 | 加回初始化后每token |
|---|---:|---:|---:|---:|
| 两层同步窗口，本轮复测 | 37.538312s | 2346.144ms | 0.426231 token/s | 2346.144ms |
| 四层同步FIFO | 40.422656s | 2526.416ms | 0.395818 token/s | 2532.560ms |
| **四层后台预加载FIFO** | **24.221452s** | **1513.841ms** | **0.660571 token/s** | **1520.066ms** |

后台队列相对本轮两层窗口平均延迟下降 **35.48%**，相对四层同步FIFO下降 **40.08%**。将首批4层初始化也计入后，对两层窗口的降幅仍为 **35.21%**。预热后首批4层装载中位 96.866ms；该进程第一次预热装载为214.831ms，这不包含tokenizer、embedding及全模型初始化。

各轮稳态每token最小～最大：两层窗口 2331.587～2361.743ms；四层同步 2523.690～2574.183ms；四层后台 1511.675～1521.360ms。

上一批两层窗口的历史中位为 2109.162ms/token，本批复测为 2346.144ms/token；存在跨批波动，本次对照的正式改善比例使用同批复测。相对历史批次，新后台队列的平均延迟下降约 28.23%。本轮前后thermalservice状态均为0；没有足够的频率/调度证据解释全部跨批差异。

#### 提前加载隐藏了哪些时间

四层后台FIFO的每token独立中位：

| 项目 | 每token时间 |
|---|---:|
| 队列缺项等待 | 0.011720ms |
| 主线程等待SDK门控 | 955.996ms |
| 后台context装载/回收的实际工作区间 | 1397.331ms |
| Top8专家读/复制/同步 | 443.116ms |
| Attention/router/专家/cache反馈 | 108.313ms |

后台context工作与专家I/O重叠，不能将两列相加算总耗时。队列本身几乎没有缺项等待，仍有约956ms/token在SDK门控前等待；当前主要剩余开销仍是context生命周期。

同样4层上限，后台线程的累计context工作区间也比同步FIFO更短（约1397对1935ms/token）。因此整体降幅同时包含后台执行位置、生命周期交错及I/O重叠的实测变化；没有进一步隔离CPU调度、缓存及SDK内部因素，不能把全部收益单独归因于重叠字节读取。

单独的40层两步host trace记录了80次上下文回收/装载和80段专家I/O：context工作 3065.238ms、专家I/O 836.127ms、两者交集 **821.428ms**。约 **98.24%** 的I/O与装载重叠，对应约 **26.80%** 的context区间被I/O覆盖。这是功能短测trace，未混入正式性能样本。

时间线保存在上述独立原型目录的 `QUEUE_TIMELINE.png`／`.svg`。

时间线展示初始化L0～L3之后，读取L1专家时卸载L0并装入L4，读取L2时装入L5。图中的空白不代表NPU空闲，包含未画出的实际计算和SDK等待；原始host时间区间另存 `QUEUE_TRACE.json`／`TRACE_SUMMARY.json`。没有据此声称获得DSP硬件busy或DDR计数。

#### 精度、容量和物理I/O验证

三种模式、各3轮，共9份完整16步hidden及全部720份最终cache，均与上一轮发布基准逐字节一致；全部输出有限。故这组输入的最终hidden SQNR仍为9.520855dB，离线最终norm后9.161381dB，没有因队列调度而改变。没有新增完整模型PPL或文本生成验证。

两种4槽模式每轮都记录峰值4层、共640次层装载和640次卸载（每层各含两个QNN context）。每轮专家逻辑读取8315122176B、实际O_DIRECT及 `/proc/self/io` 物理读取8335388672B，三种模式一致。没有通过减少层数、减少Top8、少读专家或更换位宽取得速度提升。正常退出，QNN error和FastRPC unmap错误均为0。

结果为 `RESULTS.json`、`SAMPLES.csv`、`STEPS.csv`；正式日志 `full40-window2.log`、`full40-queue4-sync.log`、`full40-queue4-async.log`。`MANIFEST.json` 保存本次源码、主机runner、原生DSP库和冻结模型manifest的哈希。原模型和专家bank只读复用前一轮目录。

复测（需要已部署的模型数据和指定设备）：

```bash
python llama-output/qwen36-full40-sm8975-queue4-20261021/source/run_decode.py \
  --layers 40 --steps 16 --samples 3 --warmups 1 --queue async \
  --name queue4-repeat

# 同步四槽对照：--queue sync；两层窗口对照：--queue off
```


### SM8975：40层decode固定context、state与KV buffer（2026-10-21）

独立原型 `/hunter/cpp_code/ET/npu/executorch/llama-output/qwen36-full40-sm8975-persistent-20261021`，真实Qwen3.6-35B-A3B全部40层，SM8975 V85，QNN2.48.1＋设备ODM库。从128-token设备prefill状态连续16步AR1，context1024、teacher forcing；不含embedding lookup、最终norm、LM head、采样和图片。未集成生产llama.py。

80个QNN context和470个IO注册handle仅初始化一次；各层GDN recurrent/conv state、full-attention K/V及对应输出buffer整个会话保留。临时交接buffer共用，专家仍按真实Top8读入两个固定槽。使用contextCreateFromBinaryListAsync的顺序资源共享模式，不再逐层contextFree/create。SDK的VA优化模式四层测试失败，最终使用SEQUENTIAL_WITHOUT_VA_OPTIMIZATION；设备警告VTCM context sharing不支持，未据此声称启用该功能。

| 方案，均为同批3轮16步中位 | 每token | 吞吐 |
|---|---:|---:|
| 四层后台context队列 | 1494.244ms | 0.669235 token/s |
| 固定context/IO，共享state工作区中间版 | 624.137ms | 1.602211 token/s |
| 固定context/IO，各层state/KV直接绑定最终版 | **619.679ms** | **1.613740 token/s** |

最终延迟下降58.53%，初始化一次约4.931s；加回初始化摊到16步约927.854ms/token。每token专家I/O约473.638ms，计算和cache反馈约145.562ms。

40层有效state/KV输入51.875MiB，共用临时交接buffer44KiB；两个专家槽25.070MiB。READY时原型rpcmem分配108.953MiB，不包含SDK内部权重与分配、模型文件页及CPU快照，不能当作完整RAM峰值。80个context binary文件合计约1.109GiB，不等于实际DRAM占用。

9份完整输出和720份cache与原40层基准逐字节一致；hidden SQNR仍9.520855dB。最终三轮循环中context创建/销毁、IO注册/注销和RPC buffer分配/释放全部为0；初始化与退出时context80/80、IO470/470平衡。专家读取字节数与旧方案一致，正式日志无QNN error或FastRPC unmap错误。详细记录见原型目录的REPORT.md、RESULTS.json、SAMPLES.csv及MANIFEST.json。


### SM8975：decode真实Top8并发IO与有界直读专家缓存（2026-10-21）

独立原型`llama-output/qwen36-full40-sm8975-decode-io-20261021`，真实Qwen3.6-35B-A3B全部40层、context1024，从128-token设备prefill状态连续16步AR1，teacher forcing；不含embedding、最终norm、head、采样和图片。PTE、N4 payload、路由和HMX整数算术保持。未接入生产llama.py。

串行基准的专家IO约471.635ms/token，其中pread约353.982ms、复制116.780ms、同步0.215ms。改为同层Top8并发pread/复制，并按层分区缓存原始N4权重；命中和新接纳专家直接从当前层RPC缓存DMA到VTCM，省去再次复制到临时槽。DSP只保留当前层缓存映射；context、state、KV和buffer依然整个会话固定保留。

同一二进制，各预热1轮＋正式3轮16步中位。每轮都清空专家缓存标签，第一token命中为0：

| 方案 | 实际专家缓存 | 每token | 吞吐 | IO墙钟/token | 物理读/token |
|---|---:|---:|---:|---:|---:|
| serial | 0.00MiB | 617.849ms | 1.618519 token/s | 471.635ms | 496.828MiB |
| parallel8 | 0.00MiB | 337.794ms | 2.960380 token/s | 191.530ms | 496.828MiB |
| mapped256 | 251.25MiB | 310.792ms | 3.217590 token/s | 155.061ms | 356.414MiB |
| mapped512 | 502.50MiB | 293.130ms | 3.411455 token/s | 137.474ms | 267.430MiB |

最快版总延迟下降52.56%，IO墙钟下降70.85%，物理读减少46.17%；命中率45.82%。输入来自真实当前router，未预测未来路由；全约15.48GiB专家bank继续保存在存储。约502MiB缓存为固定上限，不代表全模型常驻。

12份完整hidden与960份最终cache与原固定context基准逐字节一致，原SQNR保持。context/IO注册/RPC buffer生命周期操作在循环中均为0，结束80/80及470/470平衡；正式物理读与O_DIRECT字节相符，所有正式运行成功、无QNN/FastRPC或缓存unmap错误，前后thermal状态0。并行pread/copy时长为worker工作时间之和，不能和IO墙钟相加。当前未实现读盘与HMX跨层重叠。

最快版原型rpcmem合计611.453MiB，不含SDK内部权重和CPU/文件页，不是完整RAM峰值；一次setup约5.361s。复测使用`source/run_decode.py --layers 40 --steps 16 --samples 3 --warmups 1 --io-workers 8 --cache-mib 512 --mapped-cache --name repeat-fast`。详细结果、实现和限制见原型目录REPORT.md。

### SM8975：64步decode的专家局部性验证（2026-10-21）

同一独立原型、runner/DSP二进制、PTE和专家bank，将连续AR1扩到64步，位置128～191，context1024。涵盖真实40层，不含embedding、最终norm/head和采样；teacher forcing。两种容量各预热1轮＋正式3轮，每轮清空缓存标签，第一token命中为0。

| 实际专家缓存 | 每层槽数 | 命中率 | 直接读字节/token | IO墙钟/token | 总时间/token中位 |
|---|---:|---:|---:|---:|---:|
| 502.50MiB | 8 | 46.50% | 264.08MiB | 134.342ms | 289.551ms |
| 1005.00MiB | 16 | 59.01% | 201.66MiB | 118.793ms | 271.054ms |

增加502.50MiB缓存后，总延迟下降6.39%，直接读字节减少23.64%；这属于增加容量的收益，不是同等内存下更换策略的收益。两种容量三轮范围分别287.371～290.063和270.790～271.481ms/token。原型rpcmem分别611.453和1113.953MiB，不等于全进程RSS或整机峰值。默认配置保持原值。

相同8槽/层的因果路由模拟：LRU命中46.50%，累计LFU43.94%，每4token衰减的LFU46.17%，未替换当前策略。空间上，缓存未命中的10,957个读请求中只有171处可相邻合并，约减少1.56%的请求数，不代表相同比例的墙钟收益。不能按expert ID相邻盲目预读，也不能把跨几十MiB的文件区间合并读取。

三组单轮探索及两组各三轮正式测试，共9组输出；另外8组全部64步hidden和最终80份state/KV与无专家缓存对照逐字节一致。设备命中计数及直接读字节与路由重放模拟一致。所有运行成功、前后thermal状态0，循环内context/IO注册/RPC buffer生命周期操作均为0，退出80/80及470/470平衡。读盘与命中专家计算的重叠、按共同出现频率重排bank均尚未实现。

详细记录见`llama-output/qwen36-full40-sm8975-decode-io-20261021/LOCALITY.md`和`LOCALITY_ANALYSIS.json`；复测使用`source/run_decode.py --layers 40 --steps 64 --samples 3 --warmups 1 --io-workers 8 --cache-mib 1024 --mapped-cache --name repeat-locality1024`。

### SM8975：命中专家先算、miss并行读取（2026-10-21）

独立原型`llama-output/qwen36-full40-sm8975-overlap-20261021`。真实40层Qwen3.6-35B-A3B、context1024，从128-token设备prefill状态执行AR1；不含embedding、最终norm/head和采样，teacher forcing。保持原有路由、N4权重及动态A8 HMX计算。命中阶段只写原始Top8输出slot；miss阶段完成后按slot0～7原顺序求和，shared/residual只加一次。两阶段之间用固定64KiB DSP DDR route缓冲保留输出，专家缓存容量不增加。

502.5MiB缓存，在同一进程中预热关闭/开启各1轮，再交替各测3轮16步；每轮清空缓存标签。三组CPU上限边界一致，policy0/3/6分别1267.2/1267.2/806.4MHz，仍属于设备限频条件下的测量。

| 方式 | 总时间/token中位 | 范围 | MoE加载＋计算联合墙钟/token |
|---|---:|---:|---:|
| 关闭重叠 | 279.758ms | 266.558～280.423ms | 177.451ms |
| 开启重叠 | 252.458ms | 249.458～252.672ms | 153.417ms |

总时间下降9.76%，MoE联合墙钟下降13.54%；三组逐对改善17.101、27.751、27.300ms/token。中位轮命中阶段与IO批次区间重叠37.523ms/token，Native调用合计从47.505升到68.088ms/token，新增RPC/准备开销已计入。命中率和O_DIRECT请求字节不变，收益不来自少算专家或少读权重。

1005MiB组出现明显IO及CPU限频变化，不作为稳定提速结论；最初各方案顺序跑的64步结果也保留为环境波动记录，不将跨时间差值直接认作优化收益。`read_ms`排除命中阶段RPC时间，`io_wall_ms`则是与计算重叠的IO完整墙钟，两者不能重复相加。

四组64步测试共12份hidden＋960份最终cache、两组交替16步测试共12份hidden＋960份最终cache，均与相同输入的历史完整基准逐字节一致。DSP逐调用核对mask专家数，Host核对总执行次数及partial覆盖。所有循环内QNN context/IO注册/RPC buffer生命周期操作为0，退出80/80与470/470平衡。两个partial调用各自仍申请/释放HAP计算资源，其开销已计入；没有重建QNN context。

详见原型目录`REPORT.md`、`RESULTS.json`、`SAMPLES.csv`及`SEQUENTIAL_RESULTS.json`。复测使用`source/run_decode.py --layers 40 --steps 16 --samples 6 --warmups 2 --io-workers 8 --cache-mib 512 --mapped-cache --alternate --name repeat-alternating512`；单独开启用`--overlap`。尚未接入生产llama.py。

### SM8975：每层24专家缓存（2026-10-21）

同一独立重叠原型，真实40层、context1024、64步AR1（位置128～191），teacher forcing；不含embedding、最终norm/head、采样。每层仍只计算真实Top8，将保留历史专家的缓存容量从8扩到16、24。各容量单轮64步、从空缓存开始，不是三轮中位；顺序24→8→16，前后thermal状态均0，CPU上限边界均1843.2/1843.2/1036.8MHz。

| 每层缓存数 | 实际专家缓存 | 命中率 | O_DIRECT请求/token | 40层时间/token |
|---|---:|---:|---:|---:|
| 8 | 502.5MiB | 46.50% | 264.08MiB | 263.603ms |
| 16 | 1005.0MiB | 59.01% | 201.66MiB | 229.086ms |
| 24 | 1507.5MiB | 66.31% | 165.40MiB | 212.968ms |

24对比8，本轮耗时下降19.21%，读请求字节减少37.37%，增加1005MiB缓存；对比16再减少约16.118ms/token，增加502.5MiB。时间差仅为本批单轮观察，不与历史不同限频条件的数据直接计算加速。24槽不是Top24，也不等于恰好保留3个token。

三组全部64步hidden及最终80份state/KV与无专家缓存对照逐字节一致，实际路由完全相同，DSP均执行20480个专家，命中及请求字节与模拟一致。循环内context/IO注册/RPC buffer生命周期操作为0，退出80/80、470/470平衡。24槽原型rpcmem为1616.453MiB，峰值1621.684MiB，不是完整RAM峰值。

原型目录`llama-output/qwen36-full40-sm8975-overlap-20261021/CAPACITY24.md`、`CAPACITY_RESULTS.json`与`CAPACITY_SAMPLES.csv`保存细节；本模型使用`--cache-mib 1536 --mapped-cache --overlap`得到24槽/层。

### SM8975：按AR128 prefill四层一组、每层24专家槽执行（2026-10-21）

按用户明确的新口径实现于独立目录`llama-output/qwen36-prefill128-group4-experts24-20261021`。本次实际Attention/router输入shape均为`[1,128,2048]`，使用原AR128冻结图；128个token完整通过第0～3层后，才加载第4～7层，直至40层。不同于此前连续AR1的teacher-forcing测试，不能混用两者时间或缓存命中率。

同时4层执行context，每层最多24个专家权重槽，合计96槽、150.77MiB；换组复用这4个bank buffer。各层GDN state/conv和MHA K/V buffer保持。128个token各自仍选择Top8，对专家并集分批，每批最多24专家，一个专家服务分派给它的全部位置；最终按原token/slot顺序合并，不将专家集合截断到24。

40层本输入共5227个层号/专家组合、238批、40960次token–expert分派、40次最终合并。DSP和Host均检查覆盖计数。四层功能验证、40层功能验证及正式3轮的全部hidden和最终state/KV，均与原AR128全bank基准逐字节一致；正式共3份hidden、240份最终cache。

预热1轮＋正式3轮，每轮从空state/KV完成128-token prefill，包含全部专家IO及context加载/注册/卸载；不含embedding lookup、最终norm/head、采样和图片。正式总时间6.355711、6.436342、6.167889秒，中位**6.355711秒，20.139 token/s**。中位轮context换组3509.158ms，专家读盘/复制/同步2374.700ms，Native专家计算含批次RPC283.351ms，Attention145.020ms，router/shared15.033ms，cache反馈24.134ms，其余4.314ms。主要耗时是换组和IO，不是GDN核心计算。

各正式轮O_DIRECT及物理读均8458784768B（约7.878GiB），相较完整40层bank的16625393664B少49.12%；不据历史不同测试条件给出耗时加速比例。实际仍需超过24个专家参与每层计算，只保证同时驻留的权重槽不超过24。当前未实现批次间IO/计算重叠。

原型rpcmem合计452.082MiB、峰值454.090MiB，另有8MiB DSP route及约12.56MiB CPU staging，不等于完整RAM峰值。循环内RPC buffer分配/释放均0；执行context峰值8个＝4层，预热＋正式4轮结束context320/320、IO注册1800/1800平衡。测试前后thermal状态均0，CPU上限边界均1843.2MHz。

详见新原型`REPORT.md`、`RESULTS.json`、`SAMPLES.csv`、`GROUPS.csv`、`LAYERS.csv`和`BATCHES.csv`。复测：`source/run_device.py --layers 40 --tokens 128 --slots 24 --samples 3 --warmups 1 --name repeat-prefill128`。未接入生产llama.py。

### SM8975：AR128 prefill保留context，专家按四层换组（2026-10-21）

独立原型`llama-output/qwen36-prefill128-persistent-context-20261021`。沿用真实Qwen3.6-35B-A3B全部40层的冻结AR128图、N4×动态A8整数专家、UINT16 state/KV和原路由。初始化一次恢复80个Attention/router context、注册450个IO handle；context和各层state/KV保留至会话退出。四层分组只控制专家槽及执行顺序，全部128个token完成本组才进入下一组，专家仍为4×24槽、150.77MiB，并未将全部约15.48GiB专家bank放入DRAM。

共享层间临时IO，采用批量context恢复的`SEQUENTIAL_WITHOUT_VA_OPTIMIZATION`配置。设备仍提示不支持VTCM context sharing，不据此声称启用硬件VTCM共享。常驻路径不再换组注销IO或解除Native映射，退出时先关闭Native引用再释放QNN资源。

同一最终ARM runner和DSP SO，每组预热1轮、正式3轮，均从空state/KV完成128-token、40层prefill；不含embedding lookup、最终norm/head、采样、图片和输出保存。两组前后thermal均0、CPU policy0/3/6上限边界均1843.2MHz，顺序执行。

| 方案 | 三轮总时间 | 中位 | 吞吐 |
|---|---|---:|---:|
| 同二进制窗口context对照 | 6241.850／6312.761／6191.768ms | 6241.850ms | 20.507 token/s |
| context常驻 | 2993.969／3021.582／3013.586ms | **3013.586ms** | **42.474 token/s** |

同批中位延迟下降51.72%，吞吐2.071倍。窗口context恢复/注册/释放3355.935ms（加载注册2936.506ms、释放419.428ms），常驻循环内这些API调用均0；日志约0.002ms仅为分组计时标记。专家IO仍约2419.438ms，占总时间80.3%；Native阶段约388.486ms，对照272.328ms，本轮未定位其上升原因，已完整计入总耗时。

常驻版setup2609.227ms，其中context恢复/IO注册2094.241ms；这部分没有消失，只发生于初始化。setup＋首个prefill合计5522.384ms，对照7288.805ms，均非完整应用冷启动时间。常驻版计时内context/IO注册/RPC buffer生命周期操作均0，退出80/80与450/450平衡。

READY原型rpcmem合计254.371MiB、构建共享绑定前的累计峰值457.629MiB，另有DSP route及CPU staging；host RSS/HWM约1.278GiB，结束整机MemAvailable约2.760GiB。host RSS不覆盖全部DSP/共享分配，不能与rpcmem直接相加，也不代表完整内存峰值。context常驻有内存代价，不等于全部专家常驻。

四层和40层功能验证、两组三轮正式测试共576份完整hidden/最终state/KV与原AR128全bank基准逐字节一致；每轮仍执行5227个层号/专家组合、238批、40960次分派、40次最终合并。正式两组每轮物理读均8458784768B，未减少专家读取量。正式日志无QNN error、FastRPC unmap或生命周期不平衡。

详细数据、初始化与稳态口径、内存限制及复测命令见新原型`REPORT.md`、`RESULTS.json`、`SAMPLES.csv`、`LIFETIMES.csv`和`MANIFEST.json`。尚未接入生产llama.py。

### SM8975：AR128 prefill四层各36专家槽（2026-10-21）

同一独立原型`llama-output/qwen36-prefill128-persistent-context-20261021`，80个Attention/router context继续常驻，state/conv/KV仍为UINT16。128 token完整完成四层再换组，将每层24专家槽扩为36，共144槽。沿用完全相同的ARM runner、DSP SO、冻结图、N4×动态A8专家和Top8路由，无需重新导出或修改算子。这里是四层的权重槽容量，不是40层各36专家，也不是将每层路由截断成Top36。

四层功能验证通过后，同批重测24、36容量，各预热1轮＋正式3轮，全部40层、128-token prefill；不含初始化、embedding lookup、最终norm/head、采样和图片。两组顺序执行，前后thermal均0、CPU policy0/3/6上限边界均1843.2MHz。

| 配置 | 三轮时间（ms） | 中位 | 吞吐 | 专家槽内存 | Native批次 |
|---|---|---:|---:|---:|---:|
| 4×24 | 3023.197／2974.671／3017.680 | 3017.680ms | 42.417 token/s | 150.766MiB | 238 |
| 4×36 | 2805.656／2851.152／2854.801 | **2851.152ms** | **44.894 token/s** | 226.141MiB | 165 |

中位延迟下降5.52%，吞吐提高5.84%，增加75.375MiB专家槽。中位轮专家IO从2430.676降至2329.101ms，Native含RPC/资源申请从388.782降至322.079ms。批次少73次（30.67%），但两组均执行5227个层号/专家组合、40960次任务和40次合并；每轮物理读与O_DIRECT请求均8458784768B，约7.878GiB。增加容量没有减少本次128-token输入的专家读取量，不能当作跨请求LRU命中率收益。

36槽原型READY rpcmem合计329.746MiB、构建过程峰值533.004MiB，非完整RAM峰值。一次setup3123.086ms（24槽2718.762ms）；setup＋首个prefill为5853.935ms（24槽5640.848ms），本批冷setup口径并未提速，不与稳态中位混用。

36槽四层9份及两组正式各243份文件，共495份完整hidden/最终state/KV与原AR128全bank基准逐字节一致。每轮context/IO注册/RPC buffer生命周期操作均0，结束80/80和450/450平衡；正式日志无QNN error、FastRPC unmap或清理错误。

细节见原型`CAPACITY36.md`、`CAPACITY_RESULTS.json`、`CAPACITY_SAMPLES.csv`和`CAPACITY_MANIFEST.json`；复测使用`source/run_device.py --layers 40 --tokens 128 --slots 36 --samples 3 --warmups 1 --name repeat-capacity36`。生产llama.py未改，原型仍通过参数选择容量。

### SM8975：四层各64专家缓存，跨AR128块利用时间局部性（2026-10-21）

用户确认范围为当前四层各64专家，共256槽；独立原型`llama-output/qwen36-prefill128-cache64-20261021`。专家缓存含4个索引页固定402.016MiB，当前组的连续AR128块保留各层缓存，全部输入完成本组后再复用这四个buffer进入下四层。Attention及router/shared的全部80个QNN context、40层state/conv/KV继续常驻。没有分配40层各64专家的4020MiB缓存。

只改Host缓存调度，原冻结图、N4×动态A8整数HMX、Top8、UINT16 state/conv/KV及DSP SO保持。先计算本块所有命中专家，再按专家在当前块最后出现位置排序读取miss；按专家批次访问维护LRU和pin。命中直接使用原RPC/DRAM槽，不重复读取或CPU复制权重，仍走原DMA→VTCM计算路径。只使用当前块路由和历史缓存，没有读取未来块路由。每轮从空标签/state开始，四层换组只清空标签，不销毁buffer或context。

全部40层、单块128-token三轮中位**2668.024ms，47.976 token/s**，三轮2659.622／2668.024／2695.784ms。首块缓存命中0，仍读取8458784768B；本结果是64槽容量和批次调度的表现，不能归因于跨块时间局部性。

为验证时间局部性，以1024-token输入、8次AR128执行，对比相同二进制、相同4×64容量，只切换是否跨块保留专家。每组预热1轮＋正式3轮，均覆盖40层全部block、IO、Native RPC、反馈及hidden交接；不含setup、embedding lookup、最终norm/head、采样或图片。

| 配置，均4×64 | 三轮时间（ms） | 中位 | 吞吐 | 正式物理读 | 命中率 |
|---|---|---:|---:|---:|---:|
| 不保留跨块专家 | 20072.800／19981.682／20124.752 | 20072.800ms | 51.014 token/s | 58.678GiB | 0% |
| 保留跨块专家 | 14256.137／14239.249／14308.834 | **14256.137ms** | **71.829 token/s** | **37.514GiB** | **35.91%** |

本批观察延迟下降28.98%，读字节减少36.07%。命中13991/38959，分母为层/AR128块/专家请求，非327680次token–expert分派；同块聚合复用不重复算跨块命中。两组任务相同，均38959次层/块/专家计算、327680次token–expert任务、320次最终合并。命中优先拆批使Native调用752→804，额外开销全部计入；专家IO中位16587.930→10711.936ms，Native含RPC1771.816→1805.088ms。独立脚本逐批重放LRU/pin，命中和按bank索引计算的读字节与设备一致；正式物理读均等于O_DIRECT请求字节。

四层1024-token功能验证9份、单128正式243份、1024对照243份、1024缓存243份，共738份完整hidden/最终state/KV与原AR128全bank对应基准逐字节一致。1024两组每层每块专家集合相同，最终比较完整1024×2048 hidden。循环内context/IO注册/RPC buffer生命周期操作均0，单128退出80/80、450/450，1024退出80/80、470/470平衡。正式日志无QNN error、FastRPC unmap或任务覆盖错误。

1024缓存版READY原型rpcmem合计490.309MiB，构建过程累计峰值697.941MiB；单128为505.621MiB、708.879MiB。含缓存和state/IO，不含全部SDK内存、文件页、DSP heap或CPU staging，非完整RAM峰值。1024缓存版setup2848.291ms另计，结束host RSS约1.287GiB、整机MemAvailable约2.566GiB，不能与rpcmem简单相加。

各组前后thermal均0；128组与1024对照CPU policy0/3/6上限边界均1843.2MHz，1024缓存组结束policy6降到1036.8MHz，policy0/3保持1843.2MHz。两组顺序执行，未逐时采样实际频率；28.98%是本批观察值，不宣称严格控频加速比例，确定的命中及读字节减少不依赖该计时条件。

详见新原型`REPORT.md`、`RESULTS.json`、`SAMPLES.csv`、`CACHE.csv`、`BATCHES.csv`和`MANIFEST.json`。复测：`source/run_device.py --layers 40 --tokens 1024 --slots 64 --samples 3 --warmups 1 --name repeat-cache64`；对照增加`--no-reuse-experts`。仍保持AR128，未接入生产llama.py，也未实现本版命中计算与miss读取重叠。

### 旧全bank基准与4×64版本的1024-token性能回退核对（2026-10-21）

用户指出旧版128-token为7.908秒、1024-token为10.611秒；来源是`llama-output/qwen36-full40-sm8975-20261021/REPORT.md`、`SUMMARIES.csv`和原始`SAMPLES.csv`，数据有效。旧表各项分别取三轮中位，分项不保证严格加和。以下只比较总时间和确切物理读字节，不重新跑推理。

| 输入 | 旧整层bank复用 | 新4×64缓存 | 旧/新读盘量 |
|---|---:|---:|---:|
| 128 token | 7.908秒 | 2.668秒 | 15.484／7.878GiB |
| 1024 token，8×AR128 | **10.611秒** | **14.256秒** | **15.484／37.514GiB** |

旧版同时保留两层完整256专家bank，专家工作buffer800.070MiB；代码在8个AR128块循环外先各加载一次bank，整个prompt只加载40次完整bank。128与1024输入每轮均读取16625393664B，新增7块仅增加计算与交接，专家IO基本不变。

新版本当前四层各64槽，专家buffer402.016MiB；每个AR128块重新查询缺失专家，换出的权重被后续块需要时重读。即使命中35.91%，1024输入仍需24968次miss读取，累计40280723456B，是旧版2.423倍。4×64合计256槽按层分区，不能提供每层完整256专家的复用能力。上述内存仅指专家buffer，不能推断完整RAM峰值。

1024总耗时相比旧版增加34.36%，专家IO从5.046升至10.712秒，抵消了context常驻节省的约2.75秒；Native调用旧320次、新804次，计算/交接也增加。IO区间有效吞吐旧约3.068GiB/s、新约3.502GiB/s，主要问题是总读量增加。正式输出及state/KV与旧基准逐字节一致，不是更换输入、模型或量化精度造成。

此前28.98%提速仅为4×64内部“保留跨块缓存”相对“不保留”的改善；遗漏旧全bank全局基准，不能据此宣称长prefill整体优于此前。1024版目前是真实回退。应保持context常驻，修正专家加载批次覆盖全部相关AR128块的调度，或明确比较不同专家内存容量的整层驻留方案。当前严格4×64实现尚未实现跨块专家优先计算，不能把旧context时间直接扣掉当作新实测。

详细逐轮同样本拆分、代码循环位置及来源hash见`llama-output/qwen36-prefill128-cache64-20261021/COMPARISON_WITH_FULL_BANK.md`和同名JSON。新报告首页已补充回退提示，本轮未宣称该回退已修复。

### SM8975：历史Hot64预加载后由LRU更新（2026-10-21）

独立原型`llama-output/qwen36-prefill128-hot64-20261021`。真实40层Qwen3.6-35B-A3B，AR128，仍为当前四层×64专家槽、402.016MiB。全部80个Attention/router/shared QNN context和各层UINT16 state/conv/KV继续驻留；N4×动态A8 HMX、原Top8及DSP SO不变，仅改变Host缓存的初始内容。

从不重叠的WikiText校准片段各取1024 token，按校准FP32路由频率为每层选前64；评估片段不参与挑选。校准、评估分别从语料第0、2000行开始截取文本，行号不是token偏移。热点表40×64 int32、10KiB，校准集Top64分派覆盖平均82.01%，不等同于评估缓存命中率。

本组四层推理前预加载热点，随后正常批次LRU更新，热点不会永久锁定；先计算实际命中的专家，再按需读取缺失专家。换组复用同四个buffer。402MiB只能把首组热点提前到接收输入前；后续九组仍需换组读取。本次十组预加载全部计时，没有把成本藏到setup，也尚未实现预加载与初始化/计算重叠。

同一ARM runner/DSP二进制，每组预热1轮、正式3轮，表内为总时间中位轮的同样本分项。计时含40层读盘/复制/同步、Native RPC和反馈交接，不含固定setup、embedding lookup、最终norm/head、采样或图片。

| 输入 | 配置 | 预加载 | 后续推理 | 合计 | 全部专家请求读取 | 命中率 |
|---|---|---:|---:|---:|---:|---:|
| 128 | 冷LRU | 0 | 2.676秒 | **2.676秒** | 7.878GiB | 0% |
| 128 | Hot64＋LRU | 1.085秒 | 1.830秒 | **2.915秒** | 8.640GiB | 39.41% |
| 1024，8×AR128 | 冷LRU | 0 | 14.251秒 | **14.251秒** | 37.514GiB | 35.91% |
| 1024，8×AR128 | Hot64＋LRU | 1.086秒 | 13.632秒 | **14.718秒** | 38.918GiB | 40.12% |

本批计入预加载后128总耗时增加8.90%，1024增加3.28%；总专家请求读取分别增加9.68%、3.74%。40层预加载2560个专家，其中2060个在被换出前使用过，500个（19.53%）还没用就换出；这不表示它们在整个prompt中永远不被选择。预读改变LRU轨迹，且预加载流量不进入路由请求命中率的分母，所以命中率提高不能单独说明总IO获益。Native调用分别从97→112、804→840，全部计入。

128冷热请求总读取分别8458784768、9277243392B；1024分别40280723456、41787568128B。`RESULT direct_bytes`仅为路由miss，包含预加载应看`PRELOAD_RESULT total_direct_bytes`。部分hot轮存在少量额外文件页物理读；1024-hot中位轮实际41788755968B，比专家请求多1187840B。未强行将两类统计写成一致。

四层验证及四组正式共981份完整hidden和最终state/KV与原全bank基准逐字节一致；冷热每层每块专家集合及分派/合并计数相同。独立脚本回放LRU/pin、seed首次命中和按索引计算的读字节，均与设备一致。循环内context/IO注册/RPC buffer生命周期操作均0，退出80/80及450/450或470/470平衡。

全部前后thermal状态0。128两组与1024冷组CPU上限边界1843.2MHz；1024-hot结束policy6降至1036.8MHz，并出现文件页回收及少量额外IO。顺序测试、未逐时采样实际频率，时间差仅为本批观察；总请求读取增加不受该限制。单128原型READY RPC分配505.621MiB，1024为490.309MiB，不代表完整RAM峰值。

**该版没有整体提速，也未修复旧全bank的1024-token 10.611秒基准回退。** 目前保留为可复测实验，没有接入生产llama.py或改为默认。详细阶段、全部三轮时间、初始化边界与hash见原型`REPORT.md`、`RESULTS.json`、`PRELOAD_RESULTS.csv`和`MANIFEST.json`。复测：`source/run_device.py --layers 40 --tokens 1024 --slots 64 --hot-preload --samples 3 --warmups 1 --name repeat-hot`；对照去掉`--hot-preload`。

### SM8975：完整40层AR128 prefill PTE，专家外存加载（2026-10-21）

按用户确认的“完整40层，专家外存加载”导出独立原型`llama-output/qwen36-full-prefill-pte-sm8975-20261021/qwen36-full-prefill-ar128.pte`。一个`prefill_forward`方法，输入token IDs `[1,128]`，输出最后位置logits `[1,1,248320]`及完整最终hidden `[1,128,2048]`。包含真实embedding、40层Attention/norm/残差/router/shared/Top8 MoE和最终norm/LM head，不是单层或预先记录的路由回放。

完整PTE执行计划共122阶段：embedding 1、Attention 40、router/shared 40、外存MoE 40、final norm/head 1。实际复用原编译图和Native N4×动态A8 HVX/HMX内核，新编译末端W4 block32/A16 head。81个执行QNN context常驻，GDN/conv/MHA cache继续UINT16，专家仍为当前四层×64槽、402.016MiB。动态读盘由CPU调度，embedding从外部BF16表按行mmap，保持checkpoint值不变。本次没有引入INT4 embedding的额外误差。

**单个完整PTE包含多个QNN子图和Native MoE阶段，尚未融合成单个QNN图。** 需专用`source/pte_runner`注册`Qwen36PrefillStage` backend，尚不能用通用llama.py runner直接加载。只含固定128-token冷prefill，没有decode方法、额外短图或图片encoder；单会话串行原型。

PTE内嵌真实dense编译数据，大小1661562096B＝1.547GiB；另有1017118720B BF16 embedding表，专家复用原15.484GiB外存bank。另内嵌10份FA warm图用于cache量化元数据，不额外创建执行context。层间临时arena约4.955MiB，40层复用同一组activation槽。

SM8975/V85、QNN SDK2.48.1.260715和设备ODM库。同一最终PTE/ARM runner/DSP SO，每组预热1轮、正式3轮。计时为完整`module.execute`，含embedding、state复位、40层、全部预加载/专家IO、反馈及head；不含一次性PTE加载/context初始化、采样、图片或文件保存。

| 配置 | 三轮完整时间（ms） | 中位 | 吞吐 | 全部专家请求读取 |
|---|---|---:|---:|---:|
| 冷LRU | 2707.974／2804.950／2838.612 | **2804.950ms** | **45.634 token/s** | 7.878GiB |
| Hot64预加载＋LRU | 2969.796／2953.385／2931.220 | **2953.385ms** | **43.340 token/s** | 8.640GiB |

Hot64中位轮：embedding0.655ms、40层Attention含交接166.087ms、router/shared32.927ms、MoE2740.366ms、最终norm/head4.838ms、runtime/复位等8.511ms。MoE内部含预加载1087.625ms、按需读取/复制/同步1358.676ms、Native含RPC243.110ms、反馈19.640ms；子项不能与MoE总项重复相加。两类专家IO合计2446.302ms，占完整时间82.83%。扣除预加载后的1865.759ms不能当作完整prefill时间，打包PTE也未自动实现IO/计算重叠。

冷/热均5227个层/专家组合、40960次token–expert分派、40次最终合并；Native调用97/112次。Hot64命中2060/5227，预加载2560中500个使用前换出，读请求计数与原型一致。部分正式轮有少量额外PTE/embedding文件页物理读，原始统计保留。各组前后thermal0、CPU policy0/3/6上限边界1843.2MHz；顺序测试、未逐时采样实际频率，不将与历史原型的差值全部归因于PTE开销或融合加速。

两组正式共486份完整hidden及最终state/KV与原AR128量化全bank基准逐字节一致，冷热所有logits也逐字节一致。最终128-token hidden对FP32参考SQNR11.487dB，最后token hidden10.927dB；固定同一PTE hidden，隔离新norm/head的QNN对FP32 SQNR22.241dB，本输入两者argmax均为token1848。保留原型行为不等于无FP32误差，本次未重测完整PPL或多步生成。

READY原型RPC分配507.570MiB、构建累计峰值708.879MiB，非完整DRAM峰值。冷/热setup分别3232.206/4948.434ms，另计。循环内context/IO注册/RPC buffer生命周期操作均0，退出81/81 context、452/452 IO注册平衡。PTE分段在context销毁之后释放，异常路径亦保持该顺序。

报告、执行图及完整证据见新原型`REPORT.md`、`EXPORT.json`、`EXECUTION_GRAPH.txt`、`RESULTS.json`、`SAMPLES.csv`和`MANIFEST.json`。复测：`source/run_device.py --hot-preload --samples 3 --warmups 1 --name repeat-full-prefill`；去掉`--hot-preload`为冷LRU，增加`--deploy`重建独立设备目录。外部bank/索引/roPE及库通过符号链接复用既有原型，保留依赖后再整理旧目录。

### SM8975：完整40层AR1 decode PTE接续prefill（2026-10-21）

独立原型`llama-output/qwen36-full-decode-pte-sm8975-20261021/qwen36-full-decode-ar1.pte`，模型仍为真实Qwen3.6-35B-A3B。一个`decode_forward`方法，输入token IDs INT64 `[1,1]`和position INT64 `[1]` Tensor，输出logits `[1,1,248320]`及最终hidden `[1,1,2048]`。当前context1024，实际接续128-token设备prefill，连续执行位置128～143。新增的PTE完整覆盖embedding、40层Attention/norm/残差/router/shared/实际Top8 MoE及最终norm/head，共122阶段。

沿用已量化/编译的40个AR1 Attention和40个AR1 router/shared图，不重新校准主干。Attention为W4A16/block32，K/z及depthwise W8；AR1 GDN沿用V85 `DeltaNetCoreVtcm`，state/conv/KV UINT16。专家使用原N4×动态A8整数HMX、8线程O_DIRECT加载及按层LRU。最终norm/head先从相同校准输入获得冻结QDQ，再specialize并编译AR1；CPU端specialize前后结果与原prefill head参考逐元素一致。Embedding保持外部BF16行读取。

PTE大小1464092016B＝1.364GiB，内嵌81个实际执行QNN图及1296B cache网格元数据、各层IO顺序。单个完整ExecuTorch图中仍有多个QNN子图/Native阶段；需要专用`Qwen36DecodeStage` backend和`source/pte_runner`，未接入通用llama.py。没有图片encoder。

初始state为上一轮完整prefill runner完成cache反馈后保存的全部80份真实缓存快照及logits，来源SHA写入`SOURCE_PREFILL.json`。cache编码从原QNN二进制直接提取FP32 scale及INT32 offset，没有从显示精度有限的日志反推，也未硬编码量化网格。GDN保持原shape；MHA历史896位置扩为1023位置，左侧补127个量化零、原数据右对齐，按真实编码转换，后续随position更新RoPE/mask和K/V。每步保留状态，只有新测试序列开始才恢复快照。

本次prefill→decode为两个独立PTE/进程的一次性快照交接，尚未实现同进程零拷贝切换；快照只在初始化时读入，不在连续token循环读写。最初普通整数position被ExecuTorch常量检查拒绝129，最终改成动态INT64 Tensor后连续16步通过，失败日志保留在`initial-position-scalar.log`。

decode按层保留专家，默认cache budget384MiB得到40层×6槽、实际376.875MiB，另两个Top8临时bank25.070MiB，总专家buffer401.945MiB，低于prefill 4×64的402.016MiB。6是历史缓存容量，每个token仍实际计算8个专家。没有给每个token强制预读40×64个专家，也没有把15.484GiB专家bank全搬进DRAM。SDK context和state buffer长期保留，权重缓存DSP映射切层管理不等于context重建。

SM8975/V85、QNN SDK2.48.1.260715＋设备ODM库。同一最终PTE、ARM runner及原DSP库。固定输入为旧AR1基准相同WikiText token128～143，预热1轮＋正式3轮；贪心闭环另跑1轮16步。

| 口径 | ms/token | 吞吐 |
|---|---:|---:|
| 固定输入，模型图三轮中位 | **307.275** | **3.254 token/s** |
| 同一中位轮，含Host处理 | 311.559 | 3.210 token/s |
| 贪心闭环，模型图单轮 | 319.159 | 3.133 token/s |
| 贪心闭环，含Host处理单轮 | **323.425** | **3.092 token/s** |

固定输入三轮模型时间308.610／307.275／305.885ms/token。模型时间包含embedding、40层、全部专家IO、反馈和head；Host循环另含输入构造、argmax、有限值检查及结果拷贝，排除setup和循环后的文件保存。贪心序列路由不同，不将与固定输入的时间差都归因于采样。

固定输入中位轮每token：Attention70.487ms、router/shared17.525ms、专家IO144.433ms、Native50.242ms、反馈18.694ms、head4.457ms，其余约1.437ms。缓存命中1978/5120＝38.63%，16步读取5090627584B，约303.426MiB/token；贪心命中1627/5120＝31.78%，读取5658800128B。独立脚本逐层重放真实Top8、6槽LRU/pin及bank索引，命中和请求字节均一致。每步320次token–expert任务，16步5120次，没有减少计算量。

功能验证＋三轮正式共324份完整hidden/最终state/KV与旧AR1量化基准逐字节一致，正式所有logits相互逐字节一致。贪心首token取自prefill logits argmax1848，后续每步输入严格等于上步argmax，未使用teacher token充当生成输入。原始WikiText前缀的续写为` On 29 April 1915 , the High Seas Fleet sortied`（包含prefill首token和16次decode预测）；固定16步、未按EOS提前结束，没有以单条续写代替完整PPL评价。

正式setup7136.451ms，贪心setup5525.659ms，含快照导入及81个context初始化，另计。READY原型RPC分配486.785MiB、构建峰值487.844MiB，不含完整SDK/文件页/staging/DSP heap，非完整RAM峰值。全部连续token循环内context/IO注册/RPC buffer生命周期操作均0，退出81/81 context、472/472 IO注册平衡，无QNN error或清理失败。前后thermal0，CPU policy0/3/6上限边界均1843.2MHz，未逐时采实际频率，不与历史不同条件/容量数据直接算加速。

实现、全部阶段数据和证据见新原型`REPORT.md`、`EXPORT.json`、`SOURCE_PREFILL.json`、`GREEDY.json`、`RESULTS.json`、`SAMPLES.csv`、`STEPS.csv`及`MANIFEST.json`。设备已部署后复测：`source/run_device.py --greedy --steps 16 --samples 1 --warmups 0 --name repeat-greedy`；固定输入三轮去掉`--greedy`、使用`--samples 3 --warmups 1`。`--deploy`复用已有专家、prefill快照、embedding及库，清理旧目录前需保留依赖。未改生产llama.py，未commit/push。

### SM8975：完整40层PTE中文聊天验证（2026-10-21）

复用上述Qwen3.6-35B-A3B prefill/decode PTE，新建独立聊天输入和state。使用模型原生chat template、`enable_thinking=False`，通过正常系统提示构成恰好128个真实token，没有padding。用户问：“你好，请用中文介绍一下自己，再告诉我1加1等于几。”设备实际回答：“我是一个人工智能助手，旨在提供简洁、准确且自然的中文回答。1加1等于2。”正文20 token后产生`<|im_end|>`（248046），此条中文介绍和简单计算通过，没有用旧WikiText续写代替聊天验证。

首token来自这次prefill logits argmax，decode一次性导入这次prefill产生的80份state/KV，后续每步输入严格等于上步预测，脚本校验闭环关系。没有teacher forcing。原benchmark runner固定执行64步、不在EOS处提前停止；显示答案按首个EOS截断，原始完整预测仍保留。EOS后的额外预测不属于本条助手回答。

本次prefill图2826.139ms；全部64步decode图平均317.705ms/token，含Host处理321.779ms/token、3.108 token/s。decode统计含EOS后的额外步数，不作为恰好20-token回答延迟。setup、文件交接和保存另计。此为单条简短中文烟测，不代替PPL、多轮或复杂能力验证。完整提示、输入IDs、生成IDs、PTE/runner来源和原始日志见`llama-output/qwen36-chat-sm8975-20261021/REPORT.md`及同目录`INPUT.json`、`ANSWER.json`、`SOURCE.json`、`prefill.log`、`decode.log`；`run_chat.py`可独立复测。

同日增加聊天专用`source/stream_runner`：设备每次argmax后立即发送token及单调时钟时间，Host使用`Popen`逐行读取ADB并实时显示，原benchmark runner保留。默认显示Host接收间隔、Device预测间隔及完整图执行时间；`--text-only`显示连续正文。EOS来自模型generation config，遇到EOS立即停止，不再固定跑满64步。支持`--prompt`和`--max-new-tokens`，后者包含prefill首token和EOS；每次输入、日志、答案及`TOKEN_TIMINGS.jsonl`保存至独立`results/时间戳`目录。

以用户当前问题“什么是注意力机制？”实测，进程未结束前已逐步收到token；正文60 token＋EOS后停止，预测IDs与原相同输入的固定步数结果在EOS前完全一致。正常Host间隔中位315.192ms、范围281.838～359.481ms。首个decode间隔7209.623ms，其中独立decode context初始化6721.482ms；如实显示初始化停顿，不用约300ms的Graph耗时替代该间隔。循环内context/注册/RPC buffer生命周期仍0。细节及证据见该原型`REPORT.md`“实时吐字”小节和`results/20261021-171343-213092/VALIDATION.json`。

### SM8975：prefill四层×64专家池直接交给decode（2026-10-21）

按用户要求创建独立原型`llama-output/qwen36-prefill-expert-handoff-sm8975-20261021`。prefill执行后用`F_DUPFD`保留四块真实专家DMA-BUF，经`exec`交给AR1 decode；权重交接不写文件、不重新读盘或复制。仅写4232B ID/LRU/句柄元数据，四组缓存采样指纹核验一致。交接保留402.016MiB、256个真实专家，所属层为36～39。state/KV继续走原快照及真实量化网格转换，decode context仍需初始化。元数据/句柄导出0.613ms不代表整体阶段切换时间，首个decode Host间隔仍约7.7秒。

decode按原prefill四层窗口语义，从0～3层开始复用四个64槽池，随后依次换组。继承的256个专家在首次使用前被淘汰；16步共160次组切换、继承命中0、全部5120次专家请求miss。与相同中文输入、16步闭环的当前40层×6槽单轮对照：含Host315.863→371.402ms/token，3.166→2.693token/s；模型图309.970→365.597ms/token；专家读取/复制/同步144.077→203.110ms/token；请求读取287.017→496.832MiB/token；命中41.91%→0。两组前后thermal0，CPU上限边界1843200kHz，未逐时采实际频率，时间仅代表本次单轮顺序配对。

两版84个完整文件逐字节一致：input/output IDs、16步hidden/logits及80份最终state/KV。循环内context/注册/RPC buffer生命周期均0。内存交接成功，但四层窗口不能保留AR1下一token需要的其他层历史，未将此回退方案设为聊天默认。此次未施加4GiB总内存限制，也未重测PPL。复现入口`run_handoff.py --mode rolling4 --prompt "什么是注意力机制？" --max-new-tokens 17`，对照用`--mode baseline`；原理、阶段数据、全部SHA及限制见新原型`REPORT.md`与`RESULTS.json`。

### 更正及修正版：四层窗口按各层独立历史回填（2026-10-21）

用户指出回到0～3层应加载它们各自的时间局部性专家。上节实现只交接末组，换组清掉旧标签并按当前Top8读盘，没有保存40层历史及回填；0%命中只代表该简化实现，不能用于否定用户提出的历史回填方案。

新原型`llama-output/qwen36-prefill-history64-sm8975-20261021`在prefill每层完成后保存真实64槽ID/LRU及本层时钟，交接全部40层逻辑历史和末组四块物理缓存。decode进入0～3、4～7等组前，保存出组层的更新历史，再按入组层历史回填最多64专家；Top8缺失补读并更新本层LRU，下一token继续使用各层已更新的历史。逻辑历史跨组保留，物理权重仍仅当前四层驻留。本版先同步回填验证，未引入跨组IO/计算重叠。

相同128-token中文输入后8步贪心短测，history64命中2008/2560＝78.44%，当前每层6槽基准908/2560＝35.47%。但每token预加载3971.318MiB、补读105.159MiB，全部4076.477MiB（3.981GiB），比基准319.277MiB大。预加载1055.674ms/token计入完整图，含ARM循环320.292→1272.288ms/token（3.122→0.786token/s）；不能只看补读60.754ms或高命中率。此为单轮顺序短测，两组前后thermal0、CPU上限边界1843200kHz，未逐时采实际频率，不作为硬件极限。

两版84文件逐字节一致。独立重放各层历史/LRU得到8步命中259/277/239/237/247/229/264/256，与设备一致；按真实bank索引重算预加载33313832960B、补读882135040B，也与设备一致。循环内context/IO注册/RPC buffer生命周期操作均0。新原型保留为独立实验，未替换当前聊天默认；复测`run_history.py --mode history64 --prompt "什么是注意力机制？" --max-new-tokens 9`，对照用`--mode baseline`。完整实现、预加载计时边界及SHA见新原型`REPORT.md`与`RESULTS.json`。

### SM8975：64历史候选与8/16专家预取分离（2026-10-21）

独立原型`llama-output/qwen36-prefill-selective-prefetch-sm8975-20261021`保留各层64个真实历史ID/LRU，四层物理窗口换组时仅预取最近8或16个；候选存在但尚未加载仍按miss处理，缺失补读后更新本层LRU。选择只用过去路由，预取发生在当前组Attention/router之前。本次为同步预取，全部预取、补读、复制和同步均计入完整图；复用原完整40层PTE、量化网格和DSP库，仅重新编译ARM runner。

同一128-token中文输入后8步真实贪心decode，顺序各1轮、无预热：

| 配置 | 完整模型图 ms/token | 含ARM循环 ms/token | 吞吐 token/s | 实际Top8命中 | 全部专家请求读取/token |
|---|---:|---:|---:|---:|---:|
| 当前40层×6槽实际权重保留 | 316.291 | **322.406** | **3.102** | 35.47% | **319.277MiB** |
| 每层64历史、四层窗口预取8 | 455.959 | 463.784 | 2.156 | 43.13% | 777.882MiB |
| 每层64历史、四层窗口预取16 | 554.173 | 560.572 | 1.784 | 56.52% | 1207.987MiB |

预取8/16分别新增496.958/993.656MiB每token预取，只减少38.353/104.946MiB补读；预取实际使用1104/2560＝43.13%、1447/5120＝28.26%。专家IO总时间由154.350ms/token增至290.888/393.324ms/token，Attention约70ms、Native约49ms基本不变。减少预取显著优于上一轮全64同步回填，但仍未超过当前默认。四层窗口在AR1经过后切换，下一token需要重新读取本层权重；历史ID保留不等于实际权重持续驻留。此结果不替代尚未实施的异步IO实验。

两版各84个完整输出文件与本轮基准逐字节一致。独立重放真实prefill历史、逐层Top8及bank索引，对齐后全部预取/补读字节、每组预取数和内存命中均与设备一致，未把逻辑历史当作内存命中。循环内context/注册/RPC buffer生命周期均0。前后thermal0、CPU上限边界1843200kHz，未持续采实际频率，属于单轮短测。prefill未变，本轮完整128-token耗时约2.90～2.93秒；decode一次性初始化约6.66～6.80秒另计。未施加4GiB总内存限制，也未重测PPL。

保留当前较快的聊天默认，实验入口`run_prefetch.py --mode prefetch --preload-experts 8 --prompt "什么是注意力机制？" --max-new-tokens 9`，预取16改对应参数，基准使用`--mode baseline`。完整阶段拆分、计时边界、内存范围和复现见新原型`REPORT.md`、`RESULTS.json`、`MANIFEST.json`及`analyze_results.py`。

### SM8975：40层各保留16专家，prefill结束按LRU预加载（2026-10-21）

独立原型`llama-output/qwen36-decode-resident16-sm8975-20261021`按用户指定流程：128-token prefill结束保存各层真实ID/LRU，释放prefill四层池，再从每层历史选择最近16专家一次性加载到40层各自16槽；AR1只根据当前token的真实Top8查询、补读和LRU更新，没有未来token预测或四层窗口回填。原PTE/量化网格/DSP库复用，只编译ARM runner。

同一中文输入后16步真实贪心decode，三组顺序各1轮、无预热：

| 配置 | 完整图 ms/token | 含ARM循环 ms/token | 吞吐 token/s | 内存命中 | decode补读/token |
|---|---:|---:|---:|---:|---:|
| 6槽空缓存 | 309.296 | 317.580 | 3.149 | 41.91% | 287.017MiB |
| 16槽＋prefill LRU | **275.875** | **282.870** | **3.535** | **63.65%** | **178.410MiB** |
| 16槽空缓存 | 273.221 | 280.486 | 3.565 | 62.42% | 184.573MiB |

用户方案相较6槽循环耗时下降10.93%、吞吐提升12.27%、decode补读下降37.84%。一次性640专家预加载另读991.285MiB、耗时262.272ms；计入后16步累计平均299.262ms，比6槽累计低5.77%。同容量消融显示扩大缓存是主要收益：prefill预加载只比16槽空缓存多63次命中、少读98.617MiB，本轮没有额外净时间收益，不能把全部改善归因于继承prefill历史。两种16槽约2.4ms差值不当作稳定快慢结论。

16槽实际专家池1005MiB、全部已计数RPC buffer1114.910MiB，分配和连续执行成功；这不等于完整物理RAM峰值，未施加4GiB总内存限制。ready/结束采样显示预加载版进程VmSwap约13.0→6.5MiB，其余两组为0。前后thermal0、CPU上限边界1843200kHz，未持续采实际频率，数据为单轮短测。decode context初始化约6.7～6.9秒仍另计，首decode吐字间隔包含该停顿和预加载。

两个16槽版本各84个完整输出文件及真实路由与6槽逐字节一致；独立重算prefill最近16选择、逐步LRU/pin、bank对齐读取字节及最终cache ID全部通过。循环内context/注册/RPC buffer生命周期操作均0，退出81/81 context、472/472注册平衡。未重测PPL，未改原聊天默认。复测`run_resident.py --mode warm16 --prompt "什么是注意力机制？" --max-new-tokens 17`，对照用`baseline6`或`cold16`。详细计时边界、内存口径和证据见新原型`REPORT.md`、`RESULTS.json`、`STEPS.csv`、`MANIFEST.json`与`analyze_results.py`。

### SM8975：每层32专家驻留，与16槽同批对照（2026-10-21）

独立原型`llama-output/qwen36-decode-resident32-sm8975-20261021`沿用prefill后各层LRU一次性加载、AR1只按当前token真实Top8查询/补读/替换的流程，缓存从每层16扩到32。复用原完整40层PTE、量化网格和DSP库，只重新编译ARM runner。相同128-token中文输入后16步真实贪心，先16槽后32槽各1轮，无预热：

| 配置 | 模型图 ms/token | 含ARM循环 ms/token | 吞吐 token/s | 内存命中 | decode补读/token |
|---|---:|---:|---:|---:|---:|
| 16槽＋prefill LRU | 275.136 | 282.218 | 3.543 | 63.65% | 178.410MiB |
| 32槽＋prefill LRU | **246.694** | **253.363** | **3.947** | **74.75%** | **123.235MiB** |

32槽循环耗时下降10.22%、吞吐提升11.39%、补读下降30.93%；专家IO115.444→87.834ms/token，Attention/Native基本相当。第一步命中由40/320升到198/320。专家池1005→2010MiB、全部已计数RPC buffer1114.910→2119.910MiB，设备分配及连续执行通过；未施加4GiB总内存限制，不以进程RSS代替总物理峰值。

一次性预加载640→1280专家，991.285→1984.910MiB、266.723→531.947ms。预加载＋16步循环累计4782.218→4585.752ms，降低4.11%；本轮初始化6971.994→7233.111ms，若把初始化计入字段合计则11754.212→11818.863ms，不能宣称短序列完整启动延迟也降低。首decode吐字间隔约8.51→8.96秒。Prefill算法不变，两组128-token图约2918/2916ms。

两版84个完整输出文件和真实路由逐字节一致；独立重算种子选择、逐步LRU/pin、对齐读取及最终cache ID全部通过。循环内context/注册/RPC buffer生命周期均0，退出81/81 context、472/472注册平衡。前后thermal0、CPU上限边界1843200kHz，未持续采实际频率；单轮短测不作为硬件极限，未重测PPL。复测新目录`run_resident.py --mode warm32 --prompt "什么是注意力机制？" --max-new-tokens 17`，对照`warm16`；原聊天默认未改。详细证据见新原型`REPORT.md`、`RESULTS.json`、`STEPS.csv`和`MANIFEST.json`。

### SM8975：prefill与decode直接复用40层×32专家buffer（2026-10-21）

用户指出不需要复制专家权重。此前保留实验使用四层×64 prefill池，再复制到独立40层×32驻留池；复制来自两套池的实现，不是模型要求。新独立原型`llama-output/qwen36-prefill-reuse32-sm8975-20261021`从prefill开始直接使用最终40层×32 DMA-BUF，AR128按32专家分批计算，每层末尾最近32专家原地保留，阶段交接用FD继承和小型ID/LRU元数据，没有搬移或整理payload。Decode按真实物理槽和LRU时间戳接管，AR1仅按当前token真实Top8更新。

同一128-token中文输入后16步真实贪心，基准为原四层×64 prefill后重读32槽；两组顺序各1轮。直接复用交接额外专家复制0B、额外专家文件读取0B，40块buffer采样指纹一致；消除原1984.910MiB、524.532ms的阶段末预加载，新增FD导出2.505ms、导入校验映射141.883ms（导入已包含于decode setup）。

本轮整体没有性能获益：prefill执行2899.906→3165.860ms，32专家分批使Native调用102→185；decode含ARM循环258.944→294.748ms/token，3.862→3.393token/s。16步内存命中3827/5120＝74.75%、请求读取2067546112B均相同，但专家读取/复制/同步总时间1499.742→2036.242ms，其他计算阶段变化较小。单独DMA-BUF原映射、别名、释放原映射及exec继承后的暖态写入微测未复现此回退，根因尚未确认，不能声称mmap就是原因。首decode Host间隔8845.796→7654.652ms；远程执行开始至第17个生成token19813.061→20081.203ms，不能仅以省掉预加载宣称端到端加速。

两版82份完整prefill输出、84份完整decode输出逐字节一致；独立重放最近32种子选择、实际槽位、逐步LRU/pin、对齐读取及最终缓存通过。新基准与此前resident32的84份输出也逐字节一致。连续计算期间context/IO注册/RPC buffer生命周期均0，退出81/81 context、472/472注册平衡。专家池2010.15625MiB（含40个4KiB头），没有另外一份阶段交接专家池；未施加4GiB总内存限制，不以RSS代替总物理内存峰值。

两组前后thermal0、CPU上限边界1843200kHz，未逐时采实际频率，单轮短测。保留原聊天默认，未改生产llama.py或提交。复测`run_reuse.py --mode reuse32 --prompt "什么是注意力机制？" --max-new-tokens 17`，对照`--mode reload32`。实现、计时边界、内存范围、正确性及原始证据见新原型`REPORT.md`、`RESULTS.json`、`STEPS.csv`、`MANIFEST.json`与`analyze_results.py`。

### SM8975：每层16热专家固定＋16冷槽LRU（2026-10-21）

按用户方案新增独立原型`llama-output/qwen36-hot16-cold16-sm8975-20261021`。真实40层Qwen3.6-35B-A3B，保持AR128 prefill、AR1 decode、原PTE、N4权重、量化网格与DSP算子库，只编译两个ARM runner。每层按本次真实prefill路由频次选Top16（相同频次按ID），其他专家取最近16作为冷区。Prefill调整专家处理批次，让最终32个权重原地保留；通过物理槽hot_mask和继承FD交给decode，额外专家复制/文件重读均0B，没有整理payload。Decode热区固定、冷区LRU，当前Top8全部先pin，只替换冷区未pin槽；40层buffer/context在连续decode中保持驻留。

相同128个真实中文chat-template token后，真实贪心生成到相同EOS，共61个token事件（首token来自prefill，随后60次AR1）。两个策略顺序各1轮、无预热，均直接复用buffer，避免把种子重读混入对照：

| 配置 | 命中率 | UFS请求读取/token | 专家IO/token | 含ARM循环/token | 吞吐 |
|---|---:|---:|---:|---:|---:|
| 32槽LRU | 72.90% | 132.476MiB | 105.226ms | 272.996ms | 3.663 token/s |
| 固定热16＋冷16LRU | 67.95% | 157.798MiB | 109.958ms | 278.602ms | 3.589 token/s |

方案功能成立，但本次热集合选择没有IO收益：请求读取增加19.11%、循环耗时增加2.05%。640个固定热专家中265个在60步里完全未用，40层中31层读取增加。相同热种子及时间戳、仅放开所有32槽淘汰的离线反事实回放降至134.793MiB/token，接近原LRU，支持固定保护prefill热点导致冷区占槽的判断；此为离线回放，没有对应设备耗时。Prefill末尾4～64 token频次等候选的短trace回放也未超过原LRU，未据此改变已测试的策略。

先前16步短测同样增加读取（123.235→137.547MiB/token），但单轮循环293.386→284.382ms，没有据此宣布加速；完整回答随后显示读取与耗时均增加。两种长度均独立验证82份prefill、84份decode完整文件逐字节一致，逐层Top8命中、热/冷分类、pin/LRU、bank对齐读取、最终槽位全部与独立重放相符。40块交接buffer采样指纹一致；热区640个ID未替换、40层热payload采样指纹未改变。全部prefill/decode循环内context/注册/RPC buffer生命周期操作0，decode退出81/81 context、472/472注册平衡。

新增逐层IO日志将reader墙钟等待与DMA同步分开，pread/memcpy则记录并发工作线程时间和。完整回答DMA同步仅0.429→0.452ms/token，不是这轮主要耗时；线程时间和不可直接当作墙钟相加。Prefill每组仍读取8655654912B、5337专家请求、185次Native调用、40960个token-expert任务，无新增专家读取。专家池均2010.15625MiB，没有另一份热区池；未施加4GiB总内存限制，也未将RSS视作完整物理峰值。

两组前后thermal0，CPU上限边界1843200kHz，未逐时采实际频率；LRU完整回答组结束VmSwap17984KiB，固定热区组0，时间为单轮短测，读取量通过重放独立核验。保留原聊天默认，未修改生产llama.py或算子，没有提交/push。复测`run_hotcold.py --mode hot16-cold16 --prompt "什么是注意力机制？" --max-new-tokens 65`，对照`--mode lru32`；17为16步短测。细节、真实答案、全部源代码和证据见新原型`REPORT.md`、`RESULTS.json`、`SHORT_RESULTS.json`、`STEPS.csv`、`IO_LAYERS.csv`、`POLICY_DIFF.json`、`UNLOCKED_REPLAY.json`和`MANIFEST.json`。

### 更正：离线数据集固定热16，prefill与decode共用（2026-10-21）

用户明确热专家应由独立数据构造，并且prefill从一开始就按同一冷热方案执行。前一原型按单次prefill频次选热，不符合这个定义；其性能记录仅描述旧原型，不能代替本节方案。新建`llama-output/qwen36-dataset-hot16-cold16-sm8975-20261021`，保持现有PTE、INT4专家格式、量化网格及DSP算子，只编译ARM缓存管理runner。

读取本地缓存的LLaMA-Factory中英文Alpaca demo（各1000条原始记录），固定种子划分记录池并去除跨组重复内容。使用59条记录打包成64×128＝8192token建表样本，中英各32块；另10条记录组成8×128＝1024token独立验证，中英各4块。无padding，记录内容及token块hash均无跨组交集。已有40层真实PTE严格顺序运行全部72个样本，保持单份context常驻，每样本重置state，检查任务及连续执行生命周期。无需整份35B PyTorch模型驻留或重新导出。

只按建表组2621440次真实Top8分配统计每层频次Top16并冻结，独立验证的327680次分配不参与选择。固定表40×16 int32、2560B，SHA256为`29b49fa041855e788439408d280594489034cedbda0ae837882c777455a3f4ff`；建表路由覆盖26.006%、独立验证21.637%，前32样本和全64样本热集合平均重合84.531%，不将此描述为对所有任务已经收敛。

初始化从UFS一次性读取640热专家，固定在每层前16槽；prefill保留热区，只让后16冷槽参与LRU和加载，热命中原位计算，其他专家最多16个一批。当前prompt不修改热表。结束保留热16和最近冷16，继承同一组40个DMA-BUF给decode，无阶段末专家复制/重读；AR1同样只更新冷区。两个阶段分别验证640热ID和40层热payload采样指纹不变。

同一128-token中文问题“什么是注意力机制？”真实贪心生成到同一EOS，共61个事件、60步AR1，顺序各1轮：

| 配置 | Prefill图 | 首次热加载 | Decode命中 | UFS请求/token | 含ARM循环/token | 吞吐 |
|---|---:|---:|---:|---:|---:|---:|
| 32槽LRU | 3032.951ms | 0 | 72.90% | 132.476MiB | 272.975ms | 3.663token/s |
| 数据集热16＋冷16 | 3056.641ms | 328.562ms | 68.93% | 152.505MiB | 274.072ms | 3.649token/s |

首次热加载1042026496B已包含于prefill setup，不能再重复加到setup；图内专家读取8655654912→7770902528B，但包含热加载后8812929024B，比基准增加1.82%。固定热点544个被当前prefill使用，96个未用仍保留；冷批次容量减半使Native调用185→357，真实token-expert任务仍40960，未修改算子。Decode固定热表命中4421次、冷区8814次，总读取增加15.12%；640热专家中224个在这段decode未用。这份数据表较旧单prompt表在此trace的读取略低，但仍未超过本轮32槽LRU，未宣布IO优化成功。

82份完整prefill和84份完整decode文件逐字节一致；冻结表ID、热初始化读取、prefill实际miss/批次数、冷区初始内容、逐层真实Top8/LRU/pin、所有命中/字节及最终缓存均通过独立重算。阶段交接40块buffer采样指纹一致；循环内context/IO注册/RPC buffer生命周期均0，decode退出81/81 context、472/472注册平衡。专家池仍2010.15625MiB，未施加4GiB完整RAM约束；没有修改原聊天默认、生产llama.py或提交/push。

数据采集后两组前后thermal均1，CPU policy上限边界均1267200/1267200/806400kHz，未持续采实际频率；两组ready/结束VmSwap均0。时间只代表相同降频边界下单轮顺序对照，0.4%耗时差不作稳定结论，不与此前thermal0结果直接算加速。请求字节和路由通过重放独立验证。数据仅覆盖此次8192-token双语指令建表及独立验证，未声称覆盖所有长上下文或多模态工作负载。

复测新原型`run_hotcold.py --mode hot16-cold16 --prompt "什么是注意力机制？" --max-new-tokens 65`，对照`--mode lru32`。建表入口依次为`prepare_dataset.py`、`collect_routes.py --split train`、`build_hot_table.py`，随后独立运行validation采集和`validate_hot_table.py`，验证不更新表。实现、数据来源/内容hash、原始路由和计时见`REPORT.md`、`dataset/MANIFEST.json`、`HOT_TABLE.json`、`HOT_TABLE_VALIDATION.json`、`RESULTS.json`、`SUMMARY.json`及`MANIFEST.json`。

### Qwen3.6：同token跨层和连续token同层的Top8重合度（2026-10-21）

直接分析上述实验保存的真实SM8975路由，未修改模型或缓存策略。重合率定义为无序Top8集合交集个数/8；所有层号从0开始。64×128建表样本、8×128独立验证样本每块都独立重置state，不比较不同块之间的token。真实聊天为128-token输入后60个AR1输入位置；LRU32和热16/冷16的路由相同，只统计一份。

| 数据 | 同token相邻层平均重合 | 连续token同层平均重合 |
|---|---:|---:|
| 建表组8192token | 0.2698/8＝3.37% | 3.2789/8＝40.99% |
| 独立验证1024token | 0.2567/8＝3.21% | 3.1420/8＝39.27% |
| 聊天prefill128token | 0.2612/8＝3.27% | 3.0429/8＝38.04% |
| 聊天decode60token | 0.2667/8＝3.33% | 3.6496/8＝45.62% |

完整枚举40层780个无序层对，发现8～19层与20～31层的对应距离12层对有明显相关：11→23在独立验证53.96%、聊天decode62.29%，14→26分别56.25%、59.79%。只读取原checkpoint40MiB router权重核验，40份tensor hash各异，原prefill导出清单40个router/shared binary hash也各异；上述router权重余弦约0.79～0.82。不同层相同数字ID仍对应不同权重，可作为目标层预取预测的候选，不能直接跨层复用payload，当前未实现或测量这种预取收益。

Decode时间局部性不均匀：前四层平均14.25%、最后十层55.72%，layer0为6.57%、layer36为65.68%。相邻token同层有8.05%的比较完全不重合，只有0.51%完整Top8相同；45.62%是每个Top8中专家重合比例，不是整组路由重复率。Prefill末位置127到decode首输入位置128仅40次比较，均值7.19%，单独列出；decode路由按输入token而非预测输出token对齐。

全部路由范围/唯一性、两策略一致性、raw日志与既有replay、贪心输入链及数据token文件通过核验，另做24000次Python set独立交集检查。结果只描述当前量化PTE、上述双语输入与一段真实decode，未泛化到所有任务。分析脚本、完整均值/分布、逐层/全部层对CSV、逐token原始ID、权重hash/余弦、图和独立manifest在`llama-output/qwen36-dataset-hot16-cold16-sm8975-20261021/route_overlap/REPORT.md`及同目录文件。

### 热4＋冷16及因果跨层预取：逻辑回放（2026-10-21）

用户提出每层固定热4、冷16，并结合上一token和当前token前12/24/36层路由预取。新增独立分析`llama-output/qwen36-dataset-hot16-cold16-sm8975-20261021/prefetch4hot16cold`，没有修改runner或PTE。热4来自原8192-token独立建表组；同组拟合目标层、4位来源组合和前12层Top8名次的条件概率。独立验证组不参与拟合，上一token已缓存权重不重读，目标当前路由在预测完成后才用于核算。源Top8按router分数降序，前列预测质量明显更高；24/36层信号单独较弱，不能统一强制填16个。

60步真实decode逻辑回放的专家池从32槽2010.156MiB减为20槽1256.406MiB，均不含其他内存。完整复现原LRU32、热16/冷16设备日志请求字节。固定热4/冷16不预取，就绪命中60.849%、按需及总读取192.472MiB/token；盲目合并候选每层预取最多4个后，命中60.177%、总读取365.263MiB/token。按来源和源排名筛选，置信度≥0.60、每目标层token最多2个：命中61.891%、按需187.250、总194.116MiB/token；263个预取中217个当次被用（82.51%），46个最终未使用淘汰，未遗留期末未决。总读取仍较无预取增加0.85%，未宣称减少物理IO或设备加速。

独立验证输入上的同一候选策略选择6544个、正确4437个（67.80%）；只排除热4与上一Top8，未考虑更长缓存历史，不能等同设备预取使用率。其他来源概率/名次/阈值敏感性结果均保存在`REPLAY.json`，没有把单聊天最优参数当作已泛化的生产默认。

回放假设所有预测读取在目标使用前完成，就绪命中属于乐观条件；未模拟实际带宽、IO争用、取消或DMA发布等待，setup种子成本未计入decode。`PLAN.md`给出AR128 prefill与AR1 decode分别执行的方案、按需优先队列、真实LRU与投机内容分离、generation/pin/在途去重、当前整块DMA-BUF同步约束和验收计数。下一步需设备端到端验证，不能用命中率替代延迟结果。

### SM8975：同层跨token热4/冷16，异步读取实测（2026-10-21～22）

用户要求在完整模型实测prefill/decode，并明确缓存更新是同一层不同token之间的时间局部性。新增独立版本`llama-output/qwen36-async-hot4-cold16-sm8975-20261021`，真实Qwen3.6-35B-A3B全部40层，SM8975/V85，SDK2.48.1.260715，复用原PTE和DSP算子/量化配置，只编译ARM runner。每层固定数据集热4＋冷16，共800槽、1256.40625MiB专家buffer；prefill按真实128-token路由最后使用位置保留本层最近16非热专家，FD和时间戳直接交给decode；每次AR1只更新当前层自己的LRU，其他层不清空。没有将来源层同ID权重作为目标层权重使用。

最终prefill采用全模型共用的两个16专家DMA bank交替：Native直接读当前bank，8个reader线程填另一个；本层结束才并行保留最近16冷专家到本层持久池。增加约50.258MiB有界临时DMA内存。早期普通staging再发布的额外复制版本被弃用并单独保留；主基准使用原8线程直接读取，不以较慢的中间版计算收益。

同一128个真实中文chat-template输入“什么是注意力机制？”，每轮到相同EOS，共61个生成事件、60步真实AR1，顺序单模型运行：

| 模式 | Prefill128 | Prefill吞吐 | Decode含ARM循环 | Decode吞吐 |
|---|---:|---:|---:|---:|
| 直接同步，第1轮 | 3407.601ms | 37.563token/s | 287.158ms/token | 3.482token/s |
| 直接同步，第2轮 | 3460.060ms | 36.994token/s | 289.464ms/token | 3.455token/s |
| Prefill双缓冲＋decode预测早启动 | 3034.130ms | 42.187token/s | 290.310ms/token | 3.445token/s |
| Prefill双缓冲＋decode预测等待当前按需IO结束 | 3146.948ms | 40.674token/s | 289.378ms/token | 3.456token/s |
| 最终：仅prefill双缓冲，decode同层LRU | 3136.733ms | 40.807token/s | 290.517ms/token | 3.442token/s |

最终prefill比同步两轮均值3433.831ms下降8.65%；IO与Native调用窗口逐批交集共377.540ms，非worker时间相加。表内prefill含输入embedding、40层attention/router/shared/MoE、state反馈及head，含实际外存读取，排除首次加载/context恢复、热4初始化、阶段交接setup及落盘/ADB。Decode为60步完整循环墙钟，含ARM和日志；一次prefill→decode的exec/setup仍存在，没有混入稳定吞吐或宣称优化TTFT。

Decode实验按目标层、同层上一token、当前token前12/24/36层和来源Top8名次查询离线表，置信度≥0.60、每目标层最多2个，只读目标层权重；同层上一Top8已缓存则不重读。后台2个reader额外约3.14MiB staging，按需优先；队列暂停时真实需求可提升并执行，独立线程边界测试通过。改为当前按需IO结束后启动后，预测IO与计算调用重叠由0.966提高到4.988ms/token，预测IO墙钟6.487ms/token，残余等待0.0085ms/token，但整段decode无稳定加速。因此独立脚本默认`prefill-only`关闭decode预测，保留其同层LRU；`async`仅作显式实验。

60步263次预测中217个当次使用，命中60.849%→61.891%；按需192.472→187.250MiB/token，额外预测6.866，总194.116MiB/token，仍增加0.85%。原32槽与当前20槽容量不同，不把缓存缩小描述成免费IO改善。本轮没有测定4GiB总物理峰值或强制RAM上限。

每轮82份完整prefill、84份完整decode文件逐字节一致，也匹配此前原32槽参考；真实Top8、800物理槽、last-use时间戳、逐token每层热/冷命中、预测ID/概率/槽位、每次对齐读取、最终cache全部独立重放。160个热ID、40层热payload采样指纹保持；循环内context/注册/RPC buffer生命周期均0。各轮前后thermal0，CPU三个policy上限边界均为1843200/1843200/1036800kHz，未持续采实际频率；部分轮结束VmSwap约5～7MiB，时间为单prompt短测，小幅decode差异未作为稳定提升。

复现：`python llama-output/qwen36-async-hot4-cold16-sm8975-20261021/run_async.py --mode prefill-only --prompt '什么是注意力机制？' --max-new-tokens 65`；`sync`为同布局直接读取对照，`async`开启预测实验。实现、计时边界、全部中间结果及原始证据见独立目录`REPORT.md`、`RESULTS.csv`、`RESULTS.json`、`analyze_async.py`、`source`、`tests/prefetch_gate.cpp`和`MANIFEST.json`。未修改原聊天默认、生产llama.py或提交/push。

### SM8975：专家预加载与初始化重叠（2026-10-22）

独立原型`llama-output/qwen36-startup-prefetch-sm8975-20261022`，真实40层Qwen3.6-35B-A3B、同一128-token中文输入、既有PTE/INT4专家/DSP库。热4仍由独立8192-token数据构造，冷16用同层训练频次第5～20名初始化，之后可替换。40层800槽、1256.40625MiB专家池不变。

后台4个reader读取热160＋冷640专家，同时恢复81个QNN Context并注册IO；完成后才发布DMA、进入prefill，推理读取切为8线程。跳过专家槽及流水bank约1306.664MiB的冗余清零，选中payload/头部先写再用；GDN/KV state与其它IO初始化不变。首次保留冷种子，prefill结束按真实token路由最后使用顺序保留冷16，避免把按专家分批执行顺序误作LRU。仍直接继承40块DMA-BUF给decode。

两轮基准与两轮最终配置平均值：

| 指标 | 原异步流水基准 | 最终预加载配置 | 耗时下降 |
|---|---:|---:|---:|
| 初始化 | 4.567s | 4.085s | 10.56% |
| 128-token prefill图 | 3.137s | 2.812s | 10.36% |
| 设备初始化＋prefill | 7.704s | 6.897s | 10.48% |
| 启动到首token，含ADB | 7.831s | 6.997s | 10.65% |

最终两轮prefill为2.806～2.818s，首token6.924～7.069s。串行预加载对照虽让prefill降至2.882s，但首token仍7.869s，说明仅移动计时位置没有端到端收益。4/8/16线程均测过，选择初始化4、推理8；各项单轮差值不视为严格独立贡献。完整60步decode最终为287.905～289.910ms/token（3.449～3.473token/s），基准288.677ms/token（3.464token/s），没有稳定decode加速。

热冷预加载1302315008B，最终读取墙钟384.467～389.444ms基本完全重叠，初始化结束后等待0.0095～0.0113ms，均计入setup。640冷种子535个有用、105个未用；图内读取8419127296→7550033920B，含setup总读取8680292352→8852348928B，增加164.086MiB（1.98%）。收益来自关键路径重叠及Native调用385→347，任务仍40960，不是减少总IO。图内reader仍约2.25s，是后续主要瓶颈。

9轮正式对照共304步真实decode；全部82份prefill输出、相同步数的84份decode输出逐字节一致，完整60步也匹配原32槽参考。独立重建训练热点/冷种子，重放实际last-use、物理槽LRU/pin、全部请求字节、命中及最终800槽通过；热160个ID和40层payload采样指纹保持。连续循环内Context/注册/RPC buffer生命周期均0。边界thermal0、CPU上限1843200/1843200/1036800kHz，未持续采实际频率；部分轮有数MiB swap，未强制4GiB总内存上限。仍有一次prefill→decode Context恢复的数秒间隔，本轮未解决阶段切换。未重新做PPL，bitwise核验针对本次调度修改。

复现：`python llama-output/qwen36-startup-prefetch-sm8975-20261022/run_async.py --prompt '什么是注意力机制？' --max-new-tokens 65`，默认使用上述最优组合；`--mode baseline`与`--mode serial`提供对照。详细计时边界、源码、原始记录与核验入口见该目录`REPORT.md`、`RESULTS.csv`、`SUMMARY.json`、`analyze_startup.py`和`MANIFEST.json`。未修改原聊天入口、生产llama.py或提交/push。

### SM8975：取消热区，每层20槽全部LRU（2026-10-22）

新增`llama-output/qwen36-cold20-sm8975-20261022`，并将原`qwen36-chat-sm8975-20261021/run_chat.py`转到新入口。原入口完整备份在新目录`legacy/run_chat.py`。每层20槽全可淘汰，无热4保护；启动预读的20个数据集候选也仅作可淘汰种子。Prefill结束保留真实路由最近20个专家和时间戳，通过同40块DMA-BUF交给decode；decode当前Top8先pin、缺失只淘汰本层LRU，全部hot_mask=0。

同一128-token中文输入，2轮完整60步AR1；另外新跑1轮热4/冷16。发现本轮CPU policy6上限已升到1843200kHz，因此只用这次同条件对照比较时间，三轮边界thermal0、三个policy上限均1843200kHz，未连续采实际频率。全冷20 vs 新对照：decode命中62.11% vs60.85%，读取186.174 vs192.472MiB/token（减少3.27%）；prefill均值2729.641 vs2861.957ms，Native调用291 vs347，图内读取均7550033920B。Decode均值288.938 vs288.854ms/token，未显示稳定加速。旧策略是1轮、新策略2轮，属于短测。

82份完整prefill输出、84份完整decode输出跨新旧策略逐字节一致；独立核对实际最近20、槽位/时间戳、逐请求LRU/pin、热命中=0、所有对齐字节及最终800槽通过。交接40块payload采样指纹相符，循环内Context/注册/RPC buffer生命周期均0。专家池仍1256.40625MiB；prefill双流水bank批次20，比原批次16多12.5625MiB，进入decode前释放。保留初始化异步预读、4/8 reader和专家bank跳过冗余清零；GDN/KV state等不变，没有修改DSP或重新导出模型。

当前命令仍为`python llama-output/qwen36-chat-sm8975-20261021/run_chat.py --prompt '什么是注意力机制？' --max-new-tokens 65`，保留逐token实时输出。证据见新目录`REPORT.md`、`RESULTS.csv`、`CONTROL.json`、`SUMMARY.json`、`analyze_cold20.py`与`MANIFEST.json`。未提交/push。

### SM8975：decode分块读取与状态回写重叠（2026-10-22）

新增`llama-output/qwen36-cold20-io-optimization-sm8975-20261022`，原`qwen36-chat-sm8975-20261021/run_chat.py`已转到新默认配置：热保护仍0、每层LRU20；专家请求按128KiB拆分，由16个ARM线程读取并拷贝不重叠的payload区间。本层GDN/KV反馈在reader工作期间由主线程执行，随后才发布专家buffer并调用MoE。反馈数值、布局、量化映射和实际专家选择不变，本层私有cache下次在下一token消费。

9轮同输入完整回答、共540步AR1。两轮原整块读取基准290.923/286.401ms/token，最终128KiB/16线程两轮230.995/227.849ms/token，均值288.662→229.422ms/token，耗时下降20.52%，吞吐3.464→4.359token/s（提高25.82%）。各轮边界thermal0、CPU三个policy上限1843200kHz，未持续采实际频率，部分结束有数MiB swap。128KiB/8线程复测231.625与247.872，IO有波动；16线程重复后选择16。64/256KiB仅单轮，不声称全局最优。

Decode仍每60步读取11713036288B、7274个专家缺失、命中11926/19200＝62.1146%，186.174MiB/token，全部20槽与淘汰顺序相同。128KiB拆成92724个读取任务，reader批次墙钟约120→88ms/token；pread_worker_ms是并发等待时间和，不能当作总耗时。临时staging由约12.5625MiB降为2MiB；专家池1256.40625MiB与prefill流水bank不变。Prefill路径沿用前版，本轮约2.70～2.79s，不据此宣称加速；仍存在一次prefill→decode Context恢复间隔。

全部82份prefill、84份decode完整输出跨9轮逐字节一致，也匹配原参考；含hidden/logits及80份state/cache。独立重放LRU/pin、读取字节、子任务数和最终800槽通过；runtime检查子块字节总和等于原请求。实际reader类的主机边界/哨兵/同步与8线程/空任务/cached任务/重复epoch测试通过。连续循环内Context/注册/RPC buffer生命周期均0。

曾尝试O_DIRECT直接写rpcmem及继承mmap buffer，地址4096对齐仍都返回EFAULT(errno14)，因此最终保留普通对齐staging→DSP buffer复制；没有借助整份专家文件页缓存驻留。未改变N4算术或DSP库。原命令`python llama-output/qwen36-chat-sm8975-20261021/run_chat.py --prompt '什么是注意力机制？' --max-new-tokens 65`自动使用新版；对照加`--no-feedback-overlap --read-chunk-kib 0 --decode-io-workers 8`。细节与证据见新目录`REPORT.md`、`BREAKDOWN.csv`、`SUMMARY.json`、`analyze_cold20.py`、`TESTS.json`、`direct_probe/probe.log`和`MANIFEST.json`。未提交/push。

### SM8975：全冷LRU由每层20槽增到32槽（2026-10-22）

新增`llama-output/qwen36-cold32-io-optimization-sm8975-20261022`，原聊天入口已转到32槽。真实40层Qwen3.6-35B-A3B，全部1280槽可淘汰；独立数据集Top32仅作启动种子，prefill结束按真实token顺序保留本层最近32专家，直接继承40块DMA-BUF和时间戳；decode当前Top8先pin、缺失只淘汰本层LRU。保留128KiB/16 ARM reader及state反馈重叠，没有冻结热专家，没有改动PTE、DSP或量化规则。

同一128-token输入，按32→20→32顺序，每轮60步真实AR1到同一EOS。20槽新对照235.689ms/token（4.243token/s）；32槽214.072/215.357，均值214.714ms/token（4.657token/s），耗时下降8.90%，吞吐提高9.77%。命中62.1146%→72.9010%，读取186.174→132.476MiB/token（减少28.84%）。20槽之前两轮IO优化均值229.422ms/token，本次采用当前交叉复测作比较。Prefill2721.117→均值2501.082ms，Native调用291→199；启动种子读取变多，含启动的prefill总读取增加2.54%，不能只用图内IO减少宣称总IO下降。

专家池1256.40625→2010.15625MiB，增加753.75MiB；prefill双bank62.82031→100.50781MiB，另增37.6875MiB、decode前释放。Prefill峰值RPC buffer2417.527MiB，不等同整机物理内存；未强制4GiB上限。32槽decode末尾VmSwap约5.6～6.7MiB。各轮边界thermal0、CPU三个policy上限1843200kHz，未持续采实际频率；20槽1轮/32槽2轮是单prompt短测，仍有一次prefill→decode Context恢复间隔。

三轮180步AR1，全部82份完整prefill、84份完整decode文件逐字节一致，也匹配原参考；独立重放全部槽位、last-use、pin/LRU、对齐读取及分块数通过。32槽修复默认指纹掩码`1U<<32`未定义行为，UBSan边界测试确认第32槽参与校验；设备产物hash匹配。循环内Context/注册/RPC buffer生命周期均0。本次没有重新测PPL，调度修改输出数值保持一致。原聊天命令不变，详细记录见新目录REPORT.md、RESULTS.csv、CONTROL.json、SUMMARY.json、analyze_cold32.py、compare_cache.py、TESTS.json与MANIFEST.json。未commit/push。

### SM8975：1024真实输入、连续8×AR128及每层LRU32（2026-10-22）

独立目录`llama-output/qwen36-context1024-cold32-sm8975-20261022`。补齐一次sequence初始化、连续GDN/conv/KV、0/128/.../896位置、warm FA掩码和全序列LRU时间戳。完整1024 KV交给AR1，position1024起生成；每层32专家、共1280槽和2010.15625MiB池保持。10个AR1 FA用冻结QDQ扩到1152容量以容纳后续生成，其他图/权重/qparams/DSP沿用；原PTE执行计划复用，10个FA绑定外部编译context，没有重新校准整个模型。

两轮1024输入、各64步decode：prefill20.249/20.190秒，均值20.220秒（50.64token/s）；decode374.175/380.201ms/token，均值377.188（2.651token/s）。命中71.5674%、139.307MiB/token。相同1152图、128输入对照为370.544ms/token（2.699token/s）；之前快版1024容量/128输入214.714ms/token不能直接作长度对照。10个FA合计从原约32ms/token变为1152容量约146.5ms/token，而30个GDN约34ms/token；额外2048容量实验FA约351.4ms/token、整段605.7ms/token。已定位到FA子图扩容的明显开销，尚未optrace解释内部kernel。原快速聊天入口保留，未用长上下文实验覆盖。

1024图内读取57737969664B（53.773GiB），此为chunk-major：每128token走完整40层，再下一批，32槽不能保存每批全部专家；与此前按层组处理全部token的预填充调度口径不同。Prefill耗时含8次完整执行与IO，排除首次setup和阶段交接；decode含ARM完整循环。实际提示含参考资料、无padding，完整输入和路由记录保存；单prompt短测不外推所有任务。

四轮252步AR1，独立核对全部chunk种子/时间戳/1280物理槽、逐层pin/LRU/字节/分块数通过。两个1152长输入的82份prefill和84份decode完整文件逐字节一致；1152/2048完整hidden/logits一致，128新旧路径的完整prefill/decode hidden/logits也一致。实际Attention类8次更新/完整KV/掩码边界ASan、UBSan通过（ptrace不支持LSan，关闭泄漏检测）。各轮边界thermal0、CPU policy上限1843200kHz；1152长输入结束swap约3.9MiB/0，2048约95MiB。未重新测PPL或强制4GiB总内存上限。

复现：`python llama-output/qwen36-context1024-cold32-sm8975-20261022/run_async.py --prompt-tokens 1024 --decode-capacity 1152 --prompt '什么是注意力机制？' --max-new-tokens 65`。记录和限制见REPORT.md、RESULTS.csv、SUMMARY.json、NUMERICAL_CHECKS.json、analyze_context.py、TESTS.json和MANIFEST.json。未commit/push。

### SM8975：修复长上下文 AR1 FA 性能突降（2026-10-22）

独立目录`llama-output/qwen36-fa-sha-fix-sm8975-20261022`，真实40层Qwen3.6-35B-A3B，保留每层32槽LRU、INT4专家、全部冻结量化参数和原GDN/MoE。原4.66→2.65token/s并非有效上下文自然增长：原快版FA编译容量1024，长输入换1152；即使仍给128输入，1152图也仅2.70token/s。独立第3层原1024/同流程重编1024/1152/2048分别2.933/2.882/14.304/34.259ms；basic profile确认HTP执行本身突增，均6个HVX线程。Detailed/optrace在当前V85系统库返回1003/DSP1100，尚不能断言具体kernel或VTCM spill。

最终仅拆QK/PV两个MatMul的query-head维度，从GQA展开前按`query_head//8`选择KV head，避免2→16 head重复。保留投影、Q/K RMSNorm、RoPE、mask/softmax、所有原Q/DQ及输出投影，只重编10个AR1 FA context并绑定原完整PTE执行计划。通用MHA2SHA虽快，但真实输入key SQNR相对CPU QDQ从32.15降至13.89dB，因此弃用；生产通用pass实验已恢复。最终真实输入单FA14.254→0.528ms，4份完整输出逐字节一致。

同一1024真实输入，按修复→MHA对照→修复串行运行，每轮64步闭环AR1：194.323/377.645/188.480ms/token。修复均值191.402ms/token（5.225token/s），对照2.648token/s，延迟下降49.32%；10个FA合计146.467→7.614ms/token。各轮路由、命中71.5674%和139.307MiB/token读取完全相同。读取等待墙钟存在变化，不能把整段186.24ms收益全部归给MatMul。Prefill保持8×AR128，20.179秒约50.74token/s，长输入chunk-major重复专家IO仍待解决。

两轮修复、新对照与旧MHA参考全部82份prefill和84份decode文件逐字节一致，含完整hidden/logits和80份state/KV。原聊天入口128输入复测187.899ms/token（5.322token/s），与旧快版完整hidden/logits、60份GDN/conv及20份FA cache全部188个有效位置相同。独立重放1280物理槽/每chunk last-use/逐token LRU、全部读入字节和分块数通过；非零随机KV的32/1152/2048容量QDQ回归通过，手机25个产物hash相符。接受对照共252步AR1；弃用SHA/capture的65步未计入最终均值。

原`qwen36-chat-sm8975-20261021/run_chat.py`已转到修复版，默认128输入，1024测试加`--prompt-tokens 1024 --prompt '什么是注意力机制？' --max-new-tokens 65`。保持实时吐字。初始化/阶段切换/结果落盘/ADB不计入稳定吞吐，未重测PPL或强制4GiB总内存上限；单prompt短测、边界thermal0/CPU上限1843200kHz，未连续采实际频率。结构化结果、原始日志、数值核验与限制见新目录REPORT.md、SUMMARY.json、NUMERICAL_CHECKS.json、RESULTS.csv、TESTS.json和MANIFEST.json。未改生产llama.py，未commit/push。

### SM8975：1024输入＋1024步闭环decode长测（2026-10-22）

独立目录`llama-output/qwen36-decode1024-sm8975-20261022`，真实40层Qwen3.6-35B-A3B，沿用每层32槽LRU、128KiB/16 ARM reader和state反馈重叠。只把修复后的10个FA核心拆分图容量从1152编为2048，逐层核对同一冻结QDQ源；其余GDN/MoE/投影精度/量化配置保持。Prefill连续8×AR128共1024真实输入，之后消费位置1024…2047，跑满1024次AR1。Prefill预测首token另计，总共1025个生成事件，没有teacher forcing。长提示词要求完整注意力讲解与代码，本轮全程没有EOS；虽开启固定步数忽略EOS选项，实际未跨EOS。

Prefill循环20.426秒；1024步decode完整循环216.160秒，平均211.093ms/token、4.737token/s。每128步依次4.862/4.709/4.623/4.858/4.769/4.668/4.788/4.635token/s；10个FA均20.37～20.45ms/token，30个GDN约33.28ms/token，没有随位置持续下降的趋势。专家命中68.838%，每token对齐读入152.838MiB，整段152.838GiB；波动与各段命中/读量相符。相比前次短测，提示词、生成内容、FA编译容量和诊断保留方式均不同，不能把5.22→4.74token/s全部归因于有效上下文长度。

每步仍计算全部词表并检查有限值/argmax，保留全部1024份hidden和token ID，只保存最后一步logits，避免累计约970MiB诊断内存干扰长测；最终文件保存/ADB不在循环内。初始context恢复和阶段交接另计。独立重放8个prefill chunk、1280物理专家槽、全部327680次decode专家请求、pin/LRU/对齐读入和分块数通过；输入输出闭环、连续位置、完整hidden及最终logits有限、最终argmax正确。循环内context/注册/RPC buffer生命周期均0，设备25个产物hash匹配。

15秒间隔采集18份设备状态，thermal均0，HAL中NSP类传感器最高62.4℃；末尾进程VmSwap=0。未采DSP实际频率、未强制4GiB总内存上限，未重新做PPL，仅单prompt一次长测。原聊天入口默认不变，复现用`python -u llama-output/qwen36-decode1024-sm8975-20261022/run_long.py`。原始日志、逐token事件和完整输出保存在独立results目录；统计与审计见REPORT.md、SUMMARY.json、WINDOWS.csv、SENSORS.json、TESTS.json、DEVICE_BINARY_VERIFICATION.json和MANIFEST.json。未改生产llama.py，未commit/push。

### SM8975：QNN W2A16真实MoE专家试验（2026-10-22）

独立目录`llama-output/qwen36-moe-w2a16-sm8975-20261022`。现有`QuantDtype.use_16a2w`和SDK2.48.1支持UINT16激活、逐输出通道对称2bit权重（逻辑`[-2,1]`），标准BW_AXIS接口要求静态权重。当前完整MoE原型实际走原生HMX N4×动态A8外存bank，因此本次先验证标准QNN静态W2A16，没有直接将现有native路径改成W2或声称全模型UFS收益。

第0层test记录位置128的真实Top8为112/185/106/43/58/68/236/147，输入hidden2048、GU1024×2048、Down2048×512。双方同一原始权重/输入/路由分数，独立train中均匀128个位置校准A16；16个Linear全部核对UINT16激活网格，权重范围W4`[-7,7]`/W2`[-2,1]`，各为单QNN delegate。SM8975/V85设备按W4→W2→W2→W4，每组预热10次＋128次正式样本：W4均值0.217464ms，W2均值0.145799ms，权重常驻专家分支调用延迟下降32.96%；context12898144→6815584B（12.301→6.500MiB），减少47.16%。不含router/shared/residual、首次加载和UFS换入，不能与当前完整MoE或40层token耗时直接比较。前后thermal0。

同一token手机输出相对FP32：W4 16.314dB、W2 4.487dB；CPU QDQ已为16.315/4.506dB，手机相对CPU QDQ为61.003/30.370dB，主要误差在权重量化前后，非上机才出现。进一步取test前128个token全部真实路由，212个活跃专家、1024次分配，逐专家加载原始BF16并用FP32算术隔离权重误差：routed输出W4逐通道17.349dB、默认W2逐通道4.307dB、独立train驱动的激活加权裁剪W2为6.233dB。裁剪用18种阈值逐输出通道选取，未使用test，未测分组W2/GPTQ；该候选仅CPU误差评估。加残差后SQNR上升不表示专家误差消失。

当前精度不足以直接替换全40层；未修改默认聊天、原生N4 bank或生产量化recipe，未做全模型W2 PPL/文本生成。体积和resident调用收益已实测，外存按需N2格式/载入/原生计算仍需另行适配，不能把context压缩率当作实测UFS减少比例。证据见目录REPORT.md、SUMMARY.json、LAYER128.json、LAYER128_TOKEN.csv、QUANTIZATION_AUDIT.json、device.log、DEVICE_BINARY_VERIFICATION.json和MANIFEST.json。未commit/push。

### SM8975：继续测试 W2A16 的 AR128 性能与完整40层 PPL（2026-10-22）

独立目录 `llama-output/qwen36-w2a16-full-sm8975-20261022`。按要求继续模型级测试。完整40层因果 CUDA QDQ：独立 WikiText-2 train held-out 1024-token 片段，预测1023个真实下一token；每层用另一 train1024 片段均匀128位置校准。只改 routed 专家 GU/down 为逐通道 W2/W4 和 A16 min/max；attention、state/cache、router、shared、embedding、head保持FP32。FP32 PPL=3.9382，W4A16=4.6599，W2A16=13.7361；最后 residual hidden SQNR分别9.111/2.158dB。路由随每层实际量化输出重新计算，非固定FP32路由。它是完整模型的隔离 QDQ 质量评估，不是生产完整PTQ手机PTE的PPL，也不是WikiText官方test全集。逐层逐专家加载，CUDA峰值2943.16MiB，耗时约54.4秒。

新导出第0层真实AR128路由重放图：212个活跃专家、1024次Top8分配，先按专家归组，再GU/SiLU/up/down、恢复顺序与加权求和。FP32重放与缓存参考核对，权重范围W2[-2,1]/W4[-7,7]，A16，双方均一个QNN delegate，use_fp16=False。路由索引固定，所有用到的专家在context常驻，不含router/shared/residual、attention及外存LRU换入。SM8975上W4→W2→W2→W4，各组32次正式AR128：W4均值6.307027ms，W2均值4.973989ms，下降21.14%；context340688928→179486752B（324.906→171.172MiB），减少47.32%。同顺序复测AR1真实Top8，各组128次：0.213556→0.144959ms，下降32.12%。这两者均与现有快速原生N4×动态A8路径不同，未外推全模型token/s。

AR128 context完整O_DIRECT读取两次：W4 335.030/313.177ms，W2 86.749/147.117ms；context首次初始化W4 330.146/342.954ms、W2 192.514/209.944ms，均与稳态执行分开。读取对象是context，非生产专家bank；吞吐波动明显，不作为实测LRU换入收益。手机相对CPU QDQ SQNR：AR1 W4/W2=61.003/30.370dB，AR128=66.147/34.475dB，四份输出有限，实际部署二进制hash核对，前后thermal0。AR128与PPL校准布局不同，PPL不能当作该重放PTE的设备质量。

固定context动态权重更新探针尚未跑通：x86 W2/W4 refinalize均返回6022；补充prepare-only/可更新IO/binary-section配置/V79对照未解决。手机准备探针在获取初始binary大小返回1000，退出清理崩溃，未执行到权重切换。此结果不证明SDK不支持该方案；完整外存流式W2端到端性能仍未测得。未因质量停止实验，亦未替换默认聊天、原N4专家bank或生产recipe。证据为新目录REPORT.md、SUMMARY.json、PERFORMANCE.csv、PPL.json/PPL_LAYERS.csv、手机日志、DEVICE_RUN.json、DEVICE_HASHES.txt与MANIFEST.json。未commit/push。

### SM8975：完整40层 W2A8 外存推理与真实吞吐（2026-10-22）

新增 `llama-output/qwen36-w2-streaming-sm8975-20261022`。W2A16 静态图可运行，当前动态权重更新探针 refinalize 返回6022，不能据此判断SDK不支持。按要求先落地原生W2A8：真实专家GU/down逐通道absmax/1.5、[-2,1]，四个代码/byte，UFS/DRAM保留压缩N2，在VTCM内HVX符号扩展成N4 tile，沿用HMX整数N4×A8计算。不是硬件原生2bit GEMM；没有实现A16高低字节双乘加。40个真实bank共8725626880B，逐层逐专家准备，没有全模型装入主机/GPU。

完整40层PTE计划、attention/GDN/router/shared/embedding/head沿用，外存按需读取，每层32个LRU槽共1280个，128KiB/16 ARM reader与反馈重叠保持。相同128-token prefill、1152 decode容量，每轮48次完整AR1。W4两轮（1闭环＋1用其真实输入重放）、W2固定同一输入两轮：prefill循环均值2651.61→1597.20ms（48.27→80.14token/s，吞吐+66.02%）；decode循环200.05→163.14ms/token（5.00→6.13token/s，+22.62%）。W4范围4.718～5.314，W2固定输入6.104～6.155，W4 IO等待有波动。读入148.85→83.15MiB/token（-44.13%），命中69.69%→68.17%，专家池2010.16→1050.16MiB（-47.76%）。计时包含真实权重读取/复制/同步和所有40层，没有用单层速度外推。

另做两轮W2自由生成、消费自身预测，均48步：5.397/5.623token/s，合并5.508；命中降至56.89%、读取112.63MiB/token。生成文本/路由不同，不能把固定输入6.13当作自然吐字速度。输出出现重复和语义质量下降，本轮未重测W2A8 PPL；前次W2A16 CUDA QDQ 13.74不属于本实现。两轮均未遇EOS；当前只作性能实验。

首次种子读取W4 2083225600B / W2 1095761920B在setup内与context恢复重叠；prefill循环另读6994079744/3614302208B。固定输入两组prefill setup约4.76/4.10秒，命令启动至首token约7.50/5.82秒。初始化、一次prefill→decode恢复交接和最终落盘/ADB不计入稳定吞吐；连续循环生命周期计数均0。缓存数字非整机总内存，未强制4GiB。前后thermal0，W4末尾约7MiB swap、W2为0，未连续采DSP频率，单prompt短测。

相同W2代码分别N2/N4存储，在手机AR1与AR128真实路由输出逐位一致；独立重放六轮完整prefill历史、decode pin/LRU/物理1280槽、全部对齐读入和128KiB子任务通过。同精度同输入的完整hidden/logits/state/KV逐字节一致。40个N2 bank、W4索引、实际kernel/runner指纹核对。一次被中断的W4复测排除并补测。证据见REPORT.md、SUMMARY.json、RESULTS.csv、SMOKE.json、DEVICE_VERIFICATION.json、MANIFEST.json及results原始日志。

原 `llama-output/qwen36-chat-sm8975-20261021/run_chat.py` 新增 `--expert-bits 2`，正常按EOS停止；默认保留W4。示例：`python -u llama-output/qwen36-chat-sm8975-20261021/run_chat.py --expert-bits 2 --max-new-tokens 49`。本次未改生产llama.py/量化recipe，未commit/push。

### SM8975：外存 W2 专家实际接入 GPTQ（2026-10-22）

目录 `llama-output/qwen36-w2-gptq-sm8975-20261022`。原 W2 bank 是逐通道 absmax RTN，没有调用 GPTQ。本轮实际对 40×256 个专家 GU/down 执行 Hessian、Cholesky、逐列及跨 block GPTQ 补偿；先用独立 train 激活做 MSE scale 搜索，GU1024/down512、[-2,1]、damping0.1、block128，匹配现有整数 kernel，不改 router/attention/GDN/shared/embedding/head。每批8专家，约385秒、CUDA峰值1079MiB；N2 bank共8725626880B，80个bank/index文件与手机SHA256核对。

同一独立文本128-token prefill＋127步teacher forcing，共128个next-token预测位置，完整手机PTE实测：FP32参考PPL4.014；W4 PPL4.260、末层hidden归一化SQNR10.09dB；原W2 RTN 35.212/0.72dB；GPTQ＋MSE grids 24.631/2.57dB。不是官方WikiText全集PPL。去幅值归一化为逐token `h / sqrt(mean(h²)+1e-6)`；GPTQ原residual hidden SQNR为−11.37dB，整体RMS为FP32的3.762倍，因此同时记录归一化指标。固定同一设备hidden换FP32 head，PPL依次4.317/33.605/23.964，旧head不是主要差距来源。

同文本prefill为2707.97/1615.79/1732.07ms，127步完整decode为5.123/5.842/5.864token/s，三列依次W4/原W2/GPTQ W2，包含真实UFS IO，排除setup、阶段交接、结果保存；均单轮，不把teacher forcing当自然吐字。GPTQ自由生成异常并在4步decode后EOS，该短次5.608token/s不作为稳定自然生成结论。当前GPTQ W2仍不可作为可用聊天模型，默认保留W4。

相对W4的同输入Top8平均重合率从原W2的66.78%恢复到79.28%；第0层127步全部一致。缓存命中69.18/68.43/68.03%，32槽/层真实LRU、全部物理槽及IO任务独立重放通过，循环生命周期计数均0。N2/N4同权重AR1/AR128设备输出逐位一致；CUDA浮点A8模拟与native并非逐位一致，故质量结论采用手机PPL。

校准仍有限制：train1024有2132个专家未被路由到，使用全局train激活支撑；held-out分配中3.74%落到这些专家。各层采用缓存FP32激活，没有执行量化模型逐层重传播的sequential calibration。40层局部routed SQNR均值原RTN/MSE-RTN/GPTQ＋MSE为0.91/5.31/4.90dB，MSE-RTN在37层高于GPTQ，不能把完整PPL改善全部归功于GPTQ；MSE-only尚未做完整模型对照。

已有聊天入口新增显式实验参数：`python -u llama-output/qwen36-chat-sm8975-20261021/run_chat.py --expert-bits 2 --gptq --max-new-tokens 49`。完整证据见新目录REPORT.md、SUMMARY.json、CALIBRATION.json、LOCAL_LAYER_QUALITY.csv、ROUTING_OVERLAP.csv、HEAD_CHECK.json、GPTQ_PARITY.json及results。未修改生产量化recipe，未commit/push。

### SM8975：每层64个LRU专家缓存（2026-10-22）

新增 `llama-output/qwen36-expert-cache64-sm8975-20261022`，已有聊天入口现在默认每层64个可更新LRU槽，40层共2560槽；`--cache-experts 32`回到原32槽代码和产物。Top8、权重量化、原PTE和DSP kernel保持，ARM prefill/decode runner重编译。Prefill仍AR128，专家流水批次为64；DMA-BUF按40层直接交给decode，交接权重额外读取/复制均0。启动种子沿用原32个前缀，新增32个只按独立train路由频率选择，全部可淘汰，没有固定热专家。交接ABI升级Q36DATA2，ID/stamp数组64，指纹覆盖第63槽，拒绝与32槽元数据混用。

同文本128-token prefill＋127步teacher-forced decode、每种各一轮，32→64槽实测：W4专家池2010→4020MiB、prefill2708→2133ms、decode5.123→6.100token/s（+19.1%）、命中69.18%→84.17%；原W2 RTN池1050→2100MiB、prefill1616→1381ms、decode5.842→6.847（+17.2%）、命中68.43%→85.13%；GPTQ W2池1050→2100MiB、prefill1732→1468ms、decode5.864→6.652（+13.4%）、命中68.03%→83.92%。每token读取依次151.17→76.81、82.47→38.84、83.51→42.01MiB。计时包含真实IO和40层全计算，不含setup、一次阶段交接和结果保存，固定输入速度不能视作自由生成速度，单次对照有IO波动。

三条路径的prefill/decode hidden/logits、全部state/KV、输入输出token和prefill路由，每组167份SHA256对照全部逐位一致。PPL保持4.2597/35.2118/24.6307；64槽不改变W2量化误差。独立重放64槽真实LRU、物理slot、对齐读取和128KiB子任务通过，最终mutable_slots=2560，循环生命周期变化全部0。两条N2无进程swap；W4本轮有170MiB swap，专家池本身约3.93GiB，另有context/IO/scratch，不能声称整模型在4GiB内。前后thermal均0。

使用：`python -u llama-output/qwen36-chat-sm8975-20261021/run_chat.py --expert-bits 2 --gptq --cache-experts 64 --max-new-tokens 49`。W4不传expert-bits；W2 RTN去掉gptq。完整记录见新目录REPORT.md、RESULTS.csv、SUMMARY.json、COLD_SEEDS.json、source/analyze.py和results。原32槽源文件/runner未覆盖，未commit/push。

### SM8975：命中专家计算与缺失读取重叠（2026-10-22）

新增 `llama-output/qwen36-hit-first-sm8975-20261022`，原聊天入口的64槽路径现在默认采用单次DSP调用内的就绪专家调度。命中专家直接参与计算，缺失专家按128KiB分块读取，某个专家全部块完成并发布权重后，独立标记就绪，其对应worker即可继续；其它worker不必等待该层全部缺失专家。每层64槽LRU、Top8、原PTE与所有量化参数保持，最终按原Top8 rank顺序求和，shared/residual各加一次。

相同128-token prefill＋127步teacher-forced完整40层decode，含真实外存读取：GPTQ W2原等待调度150.779ms/token、6.632token/s，单次RPC就绪调度130.752ms、7.648token/s（吞吐+15.3%）；W4为164.535→146.274ms、6.078→6.836token/s（+12.5%）。首版分两次RPC只达到6.806token/s：虽隐藏部分IO，Native调用总时间从41.80升到64.79ms/token，故没有作为默认。

读量和命中不变：W2为42.0096MiB/token、83.9198%，W4为76.8137MiB/token、84.1708%。就绪调度的Native RPC跨度包含其内部等待权重就绪的时间，不能视作纯MoE计算耗时，也不能与重叠IO时间重复相加。W2仍为N2外存/DRAM、VTCM展开N4、HMX N4×A8。各项均单轮固定输入对照，不代表自由生成速度；thermal均0，W2无进程swap，W4两轮最大进程swap约159/66MiB，未约束整模型4GiB。

新增4KiB uncached就绪buffer，只在初始化分配和映射；最后一块通过acquire/release计数观察全部payload写入，DMA-BUF发布后才标为就绪。DSP校验代际号，使用uncached映射、abort和异常超时保护。循环内context/注册/RPC buffer生命周期变化均0。5轮固定输入实验每轮169份prefill/decode hidden/logits/state/KV/token/路由/cache文件相对原64槽版本逐字节一致，独立物理LRU及全部读取块审计通过；既有诊断PPL保持W4 4.2597、GPTQ W2 24.6307，没有改变W2原有精度问题。

原run_chat.py另做W4正常贪心生成17个事件，消费自身预测并遵循EOS，确认默认入口、实时显示及缓存更新有效。详见新目录REPORT.md、RESULTS.csv、SUMMARY.json、ENTRY_SMOKE.json、MANIFEST.json和results。原32槽代码保留，64槽加`--no-hit-first`可对照原等待调度；未改生产llama.py或其它模型recipe，未commit/push。

### SM8975：Native MoE复用CRM资源上下文（2026-10-22）

新增 `llama-output/qwen36-native-moe-opt-sm8975-20261022`，原聊天入口64槽路径已切到本目录，默认仍W4。AR1工作区由5MiB压为2.625MiB（8×320KiB worker＋共享codes/CSR/route输出）；首次调用建立DSP资源管理context，此后逐层`HAP_compute_res_acquire_cached/release_cached`借用和归还VTCM/HMX，会话析构才完整释放。保持命中专家先算、缺失专家单独就绪、单次RPC、原Top8求和顺序。Attention和QNN仍可在Native调用之间使用资源。

同一文本128-token prefill＋127步teacher-forced完整40层decode，含真实外存IO：W2 GPTQ原就绪调度Native57.599ms/token、完整130.752ms、7.648token/s；仅压工作区57.980/130.268/7.676，无明显吞吐收益。加入缓存CRM两次为47.993/121.303/8.244、47.964/121.681/8.218，Native相对57.60下降约16.7%。后一轮与相邻原版56.936/129.605/7.716配对，完整吞吐提升6.5%。W4原66.432/146.274/6.836，本次58.752/140.587/7.113，吞吐+4.0%。不混入32槽数据；Native跨度仍包含专家就绪等待，不能当作纯乘法时间。

16步DSP诊断显示资源申请/等待API约8.590→2.771ms/token、释放4.330→2.594ms；各worker等待、DMA/展开区间有重叠，不能相加。HMX函数调用区间约0.46ms不等于硬件利用率。初始逐调用profile仅作内部诊断；最终累计profile在循环后导出，正式测速关闭打点。此前的QNN context计数不覆盖CRM，这次CRM复用由独立资源句柄管理。

曾试保留最多1GiB多层DSP虚拟权重映射，触发HAP_mmap2失败、Native25，已撤回。另试N2位展开改写，256字节独立解码和完整输出均正确，但完整速度7.810，未稳定超过原展开的8.22～8.24；已恢复原N2Utils实现。失败和候选保留在variants及日志，未进入默认路径。

6轮成功的127步固定输入对照，每轮169份hidden/logits/state/KV/token/route/cache文件与原64槽逐字节一致，物理LRU、pin/淘汰、对齐读量和全部128KiB子任务独立审计通过；W2仍命中83.9198%、42.0096MiB/token，W4为84.1708%、76.8137MiB/token。既有128预测位置PPL保持W4 4.2597、GPTQ W2 24.6307，未修复原W2精度。QNN context/注册/RPC buffer循环变化计数均0。原入口W4正常生成17事件，消费自身预测、遵循EOS，169份文件对旧入口同输入也逐字节一致。

前后thermal均0；W2缓存CRM第一次最大进程swap约2.4MiB、最终复测0，W4有swap，未强制整模型4GiB。以上是固定输入速度，不视作自由生成速度，未给多轮置信区间。完整证据见新目录REPORT.md、RESULTS.csv、SUMMARY.json、PROFILE.json、PROFILE_CACHED.json、VARIANT_HASHES.json、ENTRY_SMOKE.json、MANIFEST.json与results。未改生产llama.py、其它模型recipe或AR128 prefill内核，未commit/push。

### SM8975：异步专家映射和N2查表展开（2026-10-22）

新增 `llama-output/qwen36-native-memory-sm8975-20261022`，原聊天入口64槽路径改到本目录，默认仍W4。保留原量化、Top8和真实LRU。增加常驻DSP线程，用两个映射窗口提前映射下一层专家缓存，与QNN Attention/Router执行重叠，不复制专家权重，也不映射全部40层专家池。N2用HVX的16项查表加交织展开，设备上对全部256种字节独立标量校验；8个专家共享AR1 GU输入tile，VTCM从2.625降为2MiB。

同输入16步内部诊断：DMA/展开的逐层worker最大区间累计4.700→2.620ms/token；前置映射/同步加剩余异步映射等待4.744→0.090ms。映射工作移到后台，其API累计墙钟约8.460ms、入口剩余等待0.022ms，不能说映射本身消失或快了98%。资源获取/归还约2.743/2.622ms，合计仍5.36ms，没有显著下降。并发worker最大区间和后台计时不可直接相加。

128-token prefill＋127步完整40层teacher-forced decode，含真实UFS：W2 GPTQ相邻原版125.711ms、7.955token/s，新版121.742ms、8.214token/s（+3.26%），Native52.642→48.670ms；新版前一轮121.932ms、8.201token/s。历史原版已达到8.218～8.244，不能将不同时间最好成绩混为一组。W4相邻137.813→136.850ms、7.256→7.307token/s（+0.70%），Native59.345→56.834ms；单对W4结果不足以证明稳定整机吞吐提升。

关闭VTCM清零、新版HMX lock3均未见资源API明显收益；非缓存专家映射反而更慢，均未保留。常驻2MiB VTCM的独立HMX context试验返回Native15，改lock3后下一层QNN graphExecute返回1003，已撤回。资源API承担Native/QNN使用权交接，不能直接跳过。继续优化应评估QNN统一资源调度或兼容的VTCM分区。

三轮新版127步实验每轮169份hidden/logits/state/KV/token/route/cache文件对原64槽逐字节一致，物理LRU和全部读取块独立审计通过；W2命中83.9198%、42.0096MiB/token，W4命中84.1708%、76.8137MiB/token，既有PPL不变。循环QNN context/注册/ARM RPC buffer生命周期变化均0。原入口正常W4贪心生成49个事件、48步AR1，使用自身预测并到达配置上限；前16步input/output/hidden/logits和82份prefill文件与旧短参考一致，没有比较不同长度的最终state。

正式对照前后thermal均0；W2新版无swap，W4新版最大进程swap约132MiB、相邻原版约29MiB，未约束整模型4GiB。完整记录见本目录REPORT.md、PROFILE.json、PAIRED_RESULTS.json、RESULTS.csv、SUMMARY.json、ENTRY_SMOKE.json、MANIFEST.json及source/。未修改生产llama.py、其它模型recipe或AR128 prefill内核，未commit/push。


### SM8975：64专家decode分项复测（2026-10-22 19:48）

本次重新运行当前优化版，W2 GPTQ/A8、40层、AR128前缀＋127步teacher forcing；保留前两列历史数据，所有值按每生成一个token平均。

| 环节 | 64专家旧调度（历史） | 64专家首次新调度（历史） | 64专家当前优化版（本次） |
| --- | ---: | ---: | ---: |
| 外存读取处理，RPC外部分 | 32.32 ms | 4.57 ms | 4.58 ms |
| Native MoE调用 | 41.86 ms | 57.60 ms | 49.17 ms |
| Attention | 39.87 ms | 38.43 ms | 38.46 ms |
| Router＋shared expert | 15.96 ms | 15.81 ms | 16.08 ms |
| State反馈、head及其它 | 20.32 ms | 14.35 ms | 13.91 ms |
| **总耗时** | 150.32 ms | 130.75 ms | 122.19 ms |
| **吞吐** | 6.65 token/s | 7.65 token/s | 8.18 token/s |

Native包含内部等待，RPC外部分与Native按原定义统计；State/head/其它是DECODE_LOOP总跨度扣除前四项，含循环和框架处理。169份输出/状态/cache逐字节一致，LRU及全部读取块审计通过，命中83.9198%、42.0096MiB/token不变，thermal前后0、进程VmSwap为0。原始日志及精确值见`llama-output/qwen36-native-memory-sm8975-20261022/BREAKDOWN_RETEST.json`和REPORT.md；前两列不是本轮重跑的相邻A/B。


### SM8975：纠正第二轮优化基线并做A/B/B/A对照（2026-10-22 20:01–20:04）

本轮映射/N2优化应与已经缓存CRM的47.96–47.99ms版本比较。此前只列更早57.60ms作为“优化前”，会掩盖第二轮没有超过历史最好成绩的问题；那个比较只能说明累计演进。

| Native贡献，按每个token归一化 | 历史缓存CRM版 | 此前最新复测 | 变化 |
| --- | ---: | ---: | ---: |
| 2057个全命中层调用的累计贡献/127 | 13.120 ms | 10.706 ms | -2.415 ms |
| 3023个有缺失层调用的累计贡献/127 | 34.843 ms | 38.461 ms | +3.618 ms |
| Native总计 | 47.964 ms | 49.167 ms | +1.203 ms |

全命中层路径确实更快，但此前跨批次比较的有缺失路径增加抵消了收益。内部16步诊断为88.125%命中、31.0234MiB/token；完整127步为83.9198%、42.0096MiB/token。映射主要移到后台，worker区间有并发，不能把两项约6.7ms直接扣减完整decode。

补做相同输入、真实40层、64槽、每次127步的A/B/B/A连续测试：

| 指标 | 缓存CRM基线，本轮两次平均 | 当前版，本轮两次平均 |
| --- | ---: | ---: |
| Native MoE | 52.686 ms/token | 48.659 ms/token |
| 完整decode | 125.601 ms/token | 122.134 ms/token |
| 吞吐 | 7.962 token/s | 8.188 token/s |
| 全命中层Native均值 | 0.816 ms/层 | 0.663 ms/层 |
| 有缺失层Native均值 | 1.658 ms/层 | 1.593 ms/层 |

缓存CRM本轮124.820–126.382ms/token、当前版121.521–122.746ms，平均吞吐+2.84%，未证明超过历史最好8.22–8.24token/s，也未给统计显著性。当前读写条件下旧程序本身也比历史慢。pread worker累计本轮A/B为198.77/198.97ms/token，含并发、阻塞和调度，不能解释为纯UFS硬件时间。四轮各169份结果逐字节一致，LRU及全部读取块审计通过，thermal前后0；B2最大进程VmSwap采样约17MiB，其余0。未修改kernel/调度。详见`llama-output/qwen36-native-memory-sm8975-20261022/REGRESSION_ANALYSIS.json`和REPORT.md末节。


### SM8975 Native内部可加拆分（2026-10-22）

独立计时诊断详见 `llama-output/qwen36-native-breakdown-sm8975-20261022/REPORT.md` 及 `RESULTS.json`/`RESULTS.csv`。相同真实40层、AR128 prefix + 127步固定输入AR1、W2 GPTQ/A8、64槽LRU，计时版基线/当前版Native为53.657/50.336ms/token；它们不是历史47.964/49.167ms的逐项还原。当前版内部为：缺失专家就绪等待24.681ms；ARM/RPC/同步和控制9.740ms；资源获取归还5.528ms；映射/线程启动/锁/分发3.290ms；激活打包与元数据DMA0.915ms；权重DMA提交/剩余等待0.647ms；N2展开1.495ms；bias/HVX后处理0.460ms；HMX请求/完成等待0.102ms；worker其它及共同尾部3.477ms。按最后完成worker接共同尾部统计，分项可加；HMX项不代表纯硬件占用，RPC残差包含统计尾部/运行时。各169份输出哈希一致、同层LRU和读取块审计通过，未修改默认运行入口。


### SM8975 Native RPC复用（2026-10-22）

当前 `llama-output/qwen36-chat-sm8975-20261021/run_chat.py` 的64专家分支已接入 `qwen36-native-handoff-sm8975-20261022`，默认 `mailbox8`。保留原8个HVX worker及原同步/算术kernel，用4KiB uncached命令buffer、单个常驻ARM服务线程和有界Native服务RPC复用调度；每320层命令或3秒续接RPC，context/buffer/映射/工作线程保持不变。每个decode token仍为逐层AR1。

真实40层、AR128 prefix＋127步固定输入AR1、W2 GPTQ/A8、每层64槽LRU的最新同版runner对照：Native 41.016→34.207ms/token，完整decode 117.736→110.602ms/token，8.494→9.041token/s（+6.45%）。保留版另两轮113.482/112.735ms，8.812/8.870token/s；全部旧版三轮117.736–121.329ms，存在读取/调度波动，不宣称固定性能或统计显著性。127步5080次直接Native RPC→16次服务RPC＋5080个共享buffer命令，ARM服务线程只创建1次，循环内context创建释放、内存注册注销和RPC buffer分配释放均为0。三轮保留版各169份输出与参考逐字节一致，LRU/route/全部读取块审计通过，命中及需求读取量不变。

W4默认路径也通过AR128＋16步对照，各169份输出一致；640次直接RPC→2次服务RPC，短测试7.819→8.333token/s，不能与127步W2结果直接比较。尾部屏障融合、减为4 worker都未保留；无限RPC服务触发设备明确的10秒超时，已改成有界服务续接。源码、失败记录、完整计数及所有轮次见 `llama-output/qwen36-native-handoff-sm8975-20261022/REPORT.md`、RESULTS.json/CSV和MANIFEST.json。


### 当前Native版同轮等待/资源开销复核（2026-10-22）

正式无内部profile对照仍为：Native 41.016→34.207ms/token、完整decode 117.736→110.602ms/token、吞吐8.494→9.041token/s。Native下降16.60%，完整decode下降6.06%，吞吐提升6.45%；当前默认仍保留该RPC复用实现。下文41.605658ms来自另一次带内部profile的诊断运行，没有替代这组正式结果；也不能把两轮差值全部归因于计时开销。11.103041ms资源API及16.565603ms非专家等待分项仅适用于该诊断轮，尚不是34.207ms那轮的逐项还原。后续消融中的“当前mailbox基线”也已启用RPC复用。

历史21.389/24.681ms就绪等待和历史34.207ms Native来自不同版本/轮次。34.207ms已经包含内部等待，并不是等待之外额外34ms，不能把这些不同时轮的数字组合成当前版计算分项。

对当前mailbox版本重新增加最后完成worker及HMX锁时间戳，AR128前缀＋127步固定输入AR1、真实40层、W2 GPTQ/A8、每层64槽LRU，同一轮Native41.605658ms/token＝缺失专家就绪等待25.040054＋资源获取2.765508＋资源归还2.688871＋HMX lock2.923977＋HMX unlock2.724685＋其它算术/DMA/打包/调度5.462562。等待之外16.565603ms，四个资源API跨度合计11.103041ms。接口跨度含内部调度等待，不表示都能删除；HMX请求完成时间不是纯硬件占用。该轮带内部profile，不能当作无profile版的回退。169份输出逐字节一致，LRU及读取chunk审计通过。详情：`llama-output/qwen36-native-current-profile-sm8975-20261022/REPORT.md`、RESULTS.json/CSV。

无内部profile的完整127步消融：当前mailbox基线Native40.839/完整114.241ms；HMX直接在服务线程执行41.022/114.198；逐次lock4为42.061/115.719；二者组合41.544/115.042；先取得HMX锁再唤醒HVX为40.902/114.208。各轮169份输出一致，命中/读量不变，未测到明确收益，均未接入默认。持续占有VTCM或跨QNN调用保留HMX注册的短试验失败，不计性能。详情：`llama-output/qwen36-native-dispatch-sm8975-20261022/REPORT.md`。

另编译了约46KB的QNN单节点MoE资源复用原型：EXCLUSIVE声明的context恢复失败5005；相同签名的最小HVX copy图能恢复；HVX上下文内尝试lock4返回1，未运行完整专家计算。资源声明/设备运行时组合尚未调通，没有可报告性能收益，也未替换正常聊天实现。详情：`llama-output/qwen36-qnn-moe-sm8975-20261022/REPORT.md`。当前默认仍为此前通过验证的qwen36-native-handoff-sm8975-20261022/mailbox8。

### RPC复用后的AR1与读取优化（2026-10-22，后续更新）

在原先测到34.207ms Native的RPC复用实现上继续修改。保留AR1直接调度8个Top8位置、共享一次ID/score DMA、4KiB临时索引bank、按ready协议减少重复同步，以及8个读线程×256KiB分块。每层64专家LRU、Top8数量、W2 GPTQ权重格式和HMX N4×A8计算保持原配置。聊天入口`llama-output/qwen36-chat-sm8975-20261021/run_chat.py`的W2 GPTQ／64专家分支已选择`qwen36-native-io-sync-sm8975-20261022`；其余分支使用原目录。

真实40层、AR128前缀＋127步固定输入AR1、context capacity1152、无内部profile，两轮原版与两轮新版均值如下。原版已经启用RPC复用，并非退回更早实现。

| 环节，ms/token | 原版两轮均值 | 新版两轮均值 |
| --- | ---: | ---: |
| Native，含就绪等待 | 41.641 | 37.150 |
| RPC外读取处理 | 5.006 | 1.889 |
| Attention | 38.027 | 38.057 |
| Router＋shared expert | 15.844 | 15.875 |
| State、head及其它，残差项 | 14.286 | 14.231 |
| 完整decode | 114.804 | 107.202 |
| 吞吐，token/s | 8.711 | 9.328 |

Native均值下降10.78%，完整decode下降6.62%，平均吞吐提高7.09%。两轮原版Native41.347/41.934ms、完整decode114.561/115.047ms；新版Native37.124/37.176ms、完整decode106.794/107.610ms，9.364/9.293token/s。历史34.207ms Native／110.602ms完整decode仍有效，但不作为本轮相同条件的原版实测值。本轮没有证明新版Native可稳定低于历史34.207ms；睡眠候选曾为32.205ms，复测回到37.128ms，未采用。新版两轮完整decode均低于历史110.602ms。

四轮各169份输出与原参考逐字节一致；逐层Top8/LRU及全部读取分块核对通过。均命中34105/40640（83.919783%），需求读取5594378240字节/127步（42.009566MiB/token）。读取调用45745→26140，总读量不变。5080层命令仍使用16次服务RPC和1个ARM服务线程，循环内context、内存注册及RPC buffer生命周期计数均为0。临时bank总buffer由13705216降至8192字节，VTCM仍为2MiB。

实际聊天入口另完成8-token自由生成，与原版相同模板/输入的169份结果逐字节一致。两版该W2 GPTQ样本均输出`endah.endah.endah.endah.end`，已有质量问题不由本次性能优化解决；本次没有重测PPL。该短自由生成的吞吐与127步固定输入性能分开记录。手工cache清理、yield、睡眠/计数器退让、资源单线程接管等候选未接入默认。完整消融、源文件、二进制哈希及数值审计见`llama-output/qwen36-native-io-sync-sm8975-20261022/REPORT.md`、RESULTS.json/CSV、SUMMARY.json、MANIFEST.json。
