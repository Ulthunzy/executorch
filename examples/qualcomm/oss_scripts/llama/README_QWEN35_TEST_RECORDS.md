# Qwen3.5 / Lincal3.5 测试记录归档

最初归档：2026-10-15；最近更新：2026-10-17。

当前导出配置、最终性能/PPL和验证范围见[主 README](README.md#validation-record-2026-10-15)。本文保留清理前52份报告中的测试数据、过程和结论；各记录使用当时的配置与测试口径，较早的实验配置和问题描述不代表当前实现。

2026-10-15归档的历史实验脚本、新增测试代码、日志、PTE/PT2、输入输出张量及临时库副本已清理。文中的历史命令与路径仅用于说明当时的参数和证据来源；导出使用主 README 的普通 `llama.py` 命令。已清理产物的链接改为文字，报告间链接指向本文对应记录。后续新增的1K–8K矩阵PTE与测量结果仍保留；本轮仅清理临时测试代码，见[正式入口接入记录](#qwen35-cli-integration-20261017)。算子测量历史见[DeltaNet README](../../custom_op/deltanet_op_package_htp/DeltaNetOpPackage/README.md)。

## 目录

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
