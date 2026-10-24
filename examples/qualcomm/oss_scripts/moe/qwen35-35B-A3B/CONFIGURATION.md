# 配置、导出与运行

量化、QNN 编译、权重分离和推理分别保留独立入口。各步骤使用同一份 `--model-config`，不用把它们合并成一个命令。

## 配置分工

| 文件 | 内容 | 使用位置 |
| --- | --- | --- |
| `configs/model.json` | 量化配方、KV/state 类型、静态容量、thinking、调度 | 校准、导出、打包、推理 |
| `configs/sm8975.json` | 本机模型/产物/构建目录、设备序列号、设备依赖和专家路径 | `run_chat.py --config` |
| `model_config.py` | 默认值、类型检查、支持范围、图配置契约 | 各 Python 入口 |
| `GRAPHS.json` / `IO.json` | 实际编译容量、量化接口、IO 顺序、来源 | 导出时写入，使用时校验 |
| PTE 同名 `.json` | 阶段、层数、prefill 顺序、长度、共享图/IO 指纹、KV 编码指纹 | 打包时写入，运行前校验 |
| `RUN.json` | 本次生效的模型设置、原始 token、设备命令、结果及错误 | 每次推理 |

CLI 覆盖本次运行的模型配置；设备 JSON 不再重复定义专家缓存数等模型参数。模型 JSON 可以只写需要修改的 section/field，未写字段取默认值；未知字段和非法类型会报错。

```bash
conda activate et_npu
export PYTHONPATH="$(dirname "$PWD")"
export MOE_ROOT=examples/qualcomm/oss_scripts/moe/qwen35-35B-A3B
export MOE_MODEL_CONFIG="$MOE_ROOT/configs/model.json"

python "$MOE_ROOT/run_chat.py" --model-config "$MOE_MODEL_CONFIG" --print-config
python "$MOE_ROOT/run_chat.py" --model-config "$MOE_MODEL_CONFIG" --check
```

`--print-config` 不读大模型权重或调用 ADB。`--check` 检查本地编译产物和接口，不进行设备推理。

## 量化配置

| 字段 | 当前默认值 | 实际作用 |
| --- | --- | --- |
| `quantization.attention` | `gptq` | 从 safetensors 顺序校准全部 Attention 投影/滤波器；兼容配方 `source_qdq` 可沿用已有网格 |
| `vision` | `w8a16_gptq` | 视觉投影/滤波器与位置表 GPTQ；W8A16 计算精度保持不变 |
| `head` | `w4a16_gptq` | 从原始 head 生成 W4/group32 GPTQ，并检查共享 head 的来源指纹 |
| `matrix_gptq_damping` / `expert_gptq_only` | `0.01` / `true` | 矩阵 Hessian 阻尼；所有专家 GU/down 强制 GPTQ，不从 RTN 组合择优 |
| `expert_bits` / `expert_activation_bits` | `2` / `8` | 当前 Native ABI 固定：N2 外存、VTCM 展开为 N4×A8 HMX |
| `expert_gu_group` / `expert_down_group` | `1024` / `512` | 专家 bank/index 的实际分组，其他值拒绝运行 |
| `expert_gptq_damping` / `reconstruction_steps` | `0.1` / `80` | `prepare_experts.py` 的 GPTQ 和完整专家输出重构 |
| `router_shared` | `w8a16_gptq` | 五个投影的 per-channel INT8 权重、UINT16 激活；也可选择历史 `fp16` 图 |
| `router_gptq_damping` | `0.01` | `prepare_router.py` 的实际 GPTQ 参数；导出检查校准报告是否匹配 |
| `embedding` | `int4_g32` | group-32 INT4 查表；也可选 `bf16` 使用原表 |

这些是校准/产物选择设置。修改 JSON 不会在推理时把旧权重变成新精度。专家导出记录 `BANK.json.quantization_config`；恢复中断校准时会拒绝使用另一组参数生成的旧 checkpoint。Router 导出既检查 `REPORT.json`，也检查图里的五个 INT8 权重网格。

### 分别生成量化产物

以下路径是输入格式示例，需要替换为实际校准产物。专家校准按层顺序传播误差，只读取当前需要的 checkpoint 分片；Router 校准、QNN 导出也按层/组串行执行。

```bash
# CUDA：顺序校准、专家重构，输出 banks/、indices/、sequential/。
python "$MOE_ROOT/quantization/prepare_experts.py" \
  --model-config "$MOE_MODEL_CONFIG" \
  --model /hunter/model/Qwen/Qwen3.6-35B-A3B \
  --source-dir /path/to/calibrated-prefill-layers \
  --inputs /path/to/INPUTS.pt \
  --output /path/to/reconstructed-experts

# CUDA：Router/shared GPTQ，输出 layerN/router-ar{128,1}-qdq.pt2 和 REPORT.json。
python "$MOE_ROOT/quantization/prepare_router.py" \
  --model-config "$MOE_MODEL_CONFIG" \
  --source-dir /path/to/calibrated-prefill-layers \
  --sequential /path/to/reconstructed-experts/sequential \
  --inputs /path/to/INPUTS.pt \
  --output /path/to/router-qdq \
  --device cuda

# INT4 embedding：读取原始 BF16 表，不加载整个 35B checkpoint。
python "$MOE_ROOT/quantization/prepare_embedding.py" \
  --model-config "$MOE_MODEL_CONFIG" \
  --source /path/to/embedding.bf16 \
  --calibration-inputs /path/to/INPUTS.pt \
  --output /path/to/embedding.int4
```

`INPUTS.pt` 的专家输入为 FP32 `calibration:[8,1024,2048]`、`heldout:[2,1024,2048]`；embedding 另读取 `tokens:[8,1024]`、`heldout_ids`。专家 source-dir 包含每层 `front-qdq.pt2`、`moe-constants.npz`、`reference.pt` 及 cos/sin。Router 读取顺序校准保存的逐层输入。heldout 不用于拟合。

embedding 的 one-hot Hessian 为对角阵，固定网格下 GPTQ 目标退化为逐行最近舍入；它没有普通 Linear 的跨列误差补偿收益。此处明确区分它与专家及 Router 的 GPTQ。

## KV/state 与静态上下文

| 字段 | 默认值 | 当前支持范围 |
| --- | --- | --- |
| `cache.kv_dtype` | `uint16` | 当前 MoE runner 仅 UINT16 |
| `cache.gdn_state_dtype` | `uint16` | 当前 Native/QNN 接口仅 UINT16 |
| `cache.conv_state_dtype` | `uint16` | 当前接口仅 UINT16 |
| `cache.append_only` | `true` | full-attention KV 追加布局，减少整段缓存搬运 |
| `cache.state_swap` | `true` | GDN state 输入/输出 buffer 交换，减少复制 |
| `context.max_seq_len` | `1152` | 已适配 1152、2048；包含输入和生成所需位置 |
| `context.prefill_ar_len` | `128` | 编译 kernel 的静态 AR，当前固定 128 |
| `context.max_prefill_tokens` | `1024` | 128～1024，完整 AR128 块 |

`--max-seq-len` 选择匹配的静态图，不会自动修改旧图容量。旧 `--decode-capacity` 是同一参数的别名。当前没有声明支持 4K/8K 或 UINT8 MoE cache；配置为这些值会提前报错。

`--no-kv-append-only` 保留为布局诊断选项。在同一组 128-token/15 步 teacher 测试中，相对默认路径 hidden SQNR 25.76 dB、logits 26.70 dB，token 一致但浮点数值不是逐字节一致；关闭所有其他优化所得结果与单独关闭 KV append 完全一致。本轮没有逐算子证明该差异的来源，也未为诊断布局重测 PPL，默认继续使用 append-only。

## 导出共享图与外部权重

先构建当前 runner、V85 Native 包及权重回调工具：

```bash
export QNN_SDK_ROOT=/hunter/qnn_sdk/src/backends/qnn/2.48.1.260715
export HEXAGON_SDK_ROOT=/hunter/qnn_sdk/hexagon/6.7.0.0
export HEXAGON_TOOLS_ROOT="$HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/21.0.04/Tools"
export ANDROID_NDK_ROOT=/hunter/android-ndk/android-ndk-r28c
export LD_LIBRARY_PATH="$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/lib/x86_64-unknown-linux-gnu:$QNN_SDK_ROOT/lib/x86_64-linux-clang:$LD_LIBRARY_PATH"
export MOE_BUILD_DIR="$PWD/build/moe/qwen35-35B-A3B-model-config-20261024"
bash "$MOE_ROOT/build.sh"
```

输入是已校准的 Attention QDQ 与上述 Router QDQ，head 也使用已编译产物。这一步不是从原始 HF checkpoint 一键重新校准全部 Attention/head。

```bash
python "$MOE_ROOT/export_shared.py" \
  --model-config "$MOE_MODEL_CONFIG" \
  --all-layers \
  --max-seq-len 1152 \
  --prefill-source llama-output/qwen35-moe-from-safetensors-gptq-20261024/layers \
  --decode-source llama-output/qwen35-moe-from-safetensors-gptq-20261024/layers \
  --router-qdq llama-output/qwen35-moe-from-safetensors-gptq-20261024/router \
  --output llama-output/qwen35-shared-export \
  --qnn-binding llama-output/qwen35-moe-from-safetensors-gptq-20261024/tools/PyQnnManagerAdaptor.cpython-311-x86_64-linux-gnu.so \
  --op-package-host "$MOE_BUILD_DIR/op-package/x86_64-linux-clang/libQnnDeltaNetOpPackage.so" \
  --op-package-device /data/local/tmp/qwen35-35B-A3B/libQnnDeltaNetOpPackage_HTP.so
```

`--all-layers` 串行启动 10 个四层子进程，每组联合编译 AR128 Attention、AR128 Router、AR1 融合图，共 12 个 graph，共享一份常量。编译完成的进程退出后再启动下一组，限制主机内存峰值。加入 `--print-plan` 只打印全部命令；小规模验证可去掉 `--all-layers`，使用 `--layers 0 3` 测 GDN 和 full-attention。

随后在设备上按 SDK 真实权重回调范围分离图和权重；每次只恢复一个 context，完成后释放：

```bash
python "$MOE_ROOT/separate_shared.py" \
  --model-config "$MOE_MODEL_CONFIG" \
  --max-seq-len 1152 \
  --source-dir llama-output/qwen35-shared-export \
  --head-context llama-output/qwen35-moe-from-safetensors-gptq-20261024/head/head.bin \
  --output llama-output/qwen35-shared-separated \
  --build-dir "$MOE_BUILD_DIR" \
  --device A7SR026915000282 \
  --runtime-device-dir /data/local/tmp/qwen35-sharing-corrected-model/prefill
```

产物为 `group0..9/graph.qnn`、`weights.bin`、`WEIGHTS.json`、`GRAPHS.json`、各方法 `.io`，以及 `head/`。此时将设备路径 JSON 的 `shared_model` 指向分离目录，`artifacts` 指向专家 bank/index、启动 profile、cos/sin、cache encoding 所在目录。专家 bank 独立于 QNN 常量共享文件。

### PTE 执行计划

共享模式 `run_chat.py` 会自动根据实际输入长度和顺序生成元数据 PTE；它不重新编译 QNN，也不复制常量。若希望显式保存导出命令：

```bash
python "$MOE_ROOT/pack_pte.py" \
  --model-config "$MOE_MODEL_CONFIG" \
  --phase prefill --prefill-tokens 1024 --prefill-order layer_major \
  --source-dir llama-output/qwen35-moe-from-safetensors-gptq-20261024 \
  --shared-model llama-output/qwen35-shared-separated \
  --head llama-output/qwen35-shared-separated/head/graph.qnn \
  --output llama-output/qwen35-plans/prefill-1024.pte

python "$MOE_ROOT/pack_pte.py" \
  --model-config "$MOE_MODEL_CONFIG" \
  --phase decode \
  --source-dir llama-output/qwen35-moe-from-safetensors-gptq-20261024 \
  --shared-model llama-output/qwen35-shared-separated \
  --head llama-output/qwen35-shared-separated/head/graph.qnn \
  --cache-encoding llama-output/qwen35-moe-from-safetensors-gptq-20261024/assets/cache-encoding.bin \
  --output llama-output/qwen35-plans/decode.pte
```

用 `--local-prefill-pte`、`--local-decode-pte` 选择显式计划；必须保留同名 JSON。图容量、量化/KV 接口、共享 context/IO 指纹、cache encoding 或 prefill 顺序不匹配时，运行会提前拒绝。仅有旧设备 PTE、没有可校验的本地计划时，不允许隐式执行多块 prefill。

## Thinking 与生成

```bash
python -u "$MOE_ROOT/run_chat.py" \
  --model-config "$MOE_MODEL_CONFIG" \
  --max-seq-len 1152 \
  --enable-thinking \
  --system-prompt "你是一个有帮助的人工智能助手。" \
  --prompt "请计算1加1，并解释理由。" \
  --max-new-tokens 128
```

`--no-enable-thinking` 显式关闭。文本和图片都通过模型原生 chat template 传递 `enable_thinking`，不是仅修改日志标签。`max_new_tokens` 包含 thinking 产生的 token。当前采样沿用 greedy，遇 EOS 停止。

文本默认补齐至能容纳实际模板的最小 AR128 倍数，也可用 `--prompt-tokens 1024` 固定长度。补齐放在 system 段，保留 assistant/thinking 后缀；`PROMPT.txt` 保存实际模板内容。`--prefill-token-file` 表示输入已经完成模板编码，不会再套 system/thinking；`RUN.json.input_mode` 与 `template_enable_thinking` 明确记录这一点。

图文输入仍由 `--image-path` 与 `--vision-artifacts` 控制，视觉 W8A16 编码器在整次模型会话内常驻。

## 推理调度

| 模型字段 / CLI | 默认 | 行为 |
| --- | --- | --- |
| `prefill_order` / `--prefill-order` | `layer_major` | 当前层完成全部 AR128 块，再进入下一层；`chunk_major` 每块依次走完 40 层 |
| `cache_experts` / `--cache-experts` | `32` | 每层 32 槽、40 层同时保留；也支持 64 槽 |
| `prefill_async_io` / `--prefill-async-io` | `true` | AR128 专家读取与计算重叠 |
| `startup_prefetch` / `--startup-prefetch` | `overlap` | `overlap`、`sync`、`none`；启动 profile 是候选，不是永久锁定的热专家 |
| `io_workers` / `--io-workers` | `8` | 专家读取 worker，支持 4/8/16 |
| `startup_io_workers` / `--startup-io-workers` | `4` | 启动预加载的 worker 数 |
| `read_chunk_kib` / `--read-chunk-kib` | `256` | 单次专家读取块大小 |
| `decode_hit_first` / `--decode-hit-first` | `true` | 先计算已命中专家，缺失专家同时读取 |
| `decode_ready_io` / `--decode-ready-io` | `true` | 就绪标志驱动缺失专家接续计算，要求 hit-first |
| `overlap_feedback` / `--overlap-feedback` | `true` | 尽量将 state/KV 反馈与后续工作重叠 |
| `context_load_mode` / `--decode-context-load-mode` | `auto` | 旧独立模式的启动装载策略；共享模式统一恢复常驻 context |

布尔 CLI 都有 `--no-...`。关闭 prefill 异步读取时，也需要 `--startup-prefetch none`；空槽以 `id=-1/stamp=0` 安全交接给 decode。命中/缺失调度只改变加载和执行顺序，路由仍由真实 Top8 决定，不改变模型选择的专家。

```bash
python -u "$MOE_ROOT/run_chat.py" \
  --model-config "$MOE_MODEL_CONFIG" \
  --prompt-tokens 1024 --max-new-tokens 128 \
  --prefill-order layer_major \
  --cache-experts 64 \
  --io-workers 8 --read-chunk-kib 256 \
  --decode-hit-first --decode-ready-io --overlap-feedback \
  --prompt "请介绍线性注意力的工作方式。"
```

四层分组用于 QNN 常量共享和串行编译；运行时十组和 head 全部常驻，不按四层反复销毁 context。专家 cache 在 prefill/decode 之间直接交接，缺失专家从 UFS 补入并按 LRU 更新。当前未接入跨 token 的预测预取；基础的缺失读取/命中计算重叠已经启用。

## 配置入口验证范围（全量校准之前）

2026-10-24，SM8975、QNN 2.48.1：

- 默认配置与整理前版本的 86 份 token/hidden/logits/state/KV 文件逐字节一致。
- 第 0 层 GDN、第 3 层 full-attention 真实导出为 6 个 graph/1 个 context；设备回调采集 7 个区间，7 次释放、`live=0`。分离文件重新组合的 SHA256 与源 context 相同。
- 相同 1024-token 输入，`layer_major` 与 `chunk_major` 各 86 份输出逐字节一致。一次设备运行的 prefill 为 6.814 秒与 12.088 秒；这是调度对照，不代表新的稳定吞吐保证。
- 关闭启动预加载后，40 层共有 1185 个空槽；交接及 decode 正常，86 份数值文件与启用预加载一致。
- 原生 tokenizer 的 thinking 开/关后缀不同，带 thinking 的短设备运行及 greedy feedback 校验通过。

本次没有重做 40 层 GPTQ 或全量 QNN 编译，也没有宣称产生新的 PPL 改善；验证的是配置确实传递到校准/编译/运行入口，以及默认数值、实际调度和资源交接保持正确。详细产物位于 `llama-output/qwen35-moe-model-config-20261024`。

## 从 safetensors 重新生成全 GPTQ 产物

`configs/gptq.json` 选择新的原始 checkpoint 路径：Attention 全部投影/卷积、Router/shared、全部 routed 专家的 GU/down、head 和视觉投影都执行 GPTQ。视觉位置表使用实际双线性插值的 Hessian；文本 embedding 的 one-hot Hessian 仍退化为固定网格最近舍入。Norm、bias、GDN 时间参数等非矩阵参数沿用后端原有处理，不把它们标为 GPTQ。未用于当前 40 层推理的 MTP 辅助层不导出。

量化宽度继续使用已验证的混合配置：Attention 为 W4、GDN K/z 与 depthwise 为 W8，Router/shared 和视觉为 W8，head 与 embedding 为 W4，routed experts 为 N2 存储并在 Native 内展开成 N4×A8。QNN 图包含 Attention、Router/shared、head、视觉；routed experts 生成外部 Native bank。

```bash
export MOE_GPTQ_OUTPUT=llama-output/qwen35-moe-from-safetensors-gptq-20261024
export MOE_GPTQ_BINDING="$MOE_GPTQ_OUTPUT/tools/PyQnnManagerAdaptor.cpython-311-x86_64-linux-gnu.so"

python -u "$MOE_ROOT/quantization/prepare_model.py" \
  --model-config "$MOE_ROOT/configs/gptq.json" \
  --model /hunter/model/Qwen/Qwen3.6-35B-A3B \
  --corpus "$MOE_GPTQ_OUTPUT/calibration/wikitext-2-raw-v1-train.jsonl" \
  --output "$MOE_GPTQ_OUTPUT" \
  --expert-batch 32 --qnn-binding "$MOE_GPTQ_BINDING"
```

该命令只负责校准与权重准备，尚不编译 QNN。读取 safetensors 分片中的当前层，按 Attention → Router/shared → 全部 256 专家的顺序推进，下一层使用上一层实际量化输出。每个阶段使用独立进程，便于释放显存并从已完成层恢复；`--last-layer 0` 可先验证一层。原始 FP32 链路另行传播，7 条拟合、1 条选择、2 条留出序列保持分离。输出 `DATA.json`、冻结 `MODEL_CONFIG.json`、逐阶段日志/时间、每个投影 GPTQ 报告、`layers/`、`router/`、`banks/`、`indices/`、`head/`、`embedding.int4` 和启动候选表。

随后分别编译新 head、联合编译新的 AR128/AR1 图：

```bash
python -u "$MOE_ROOT/export_head.py" \
  --model-config "$MOE_GPTQ_OUTPUT/MODEL_CONFIG.json" \
  --source "$MOE_GPTQ_OUTPUT/head/head-qdq.pt2" \
  --output "$MOE_GPTQ_OUTPUT/head" --qnn-binding "$MOE_GPTQ_BINDING"

python -u "$MOE_ROOT/export_shared.py" \
  --model-config "$MOE_GPTQ_OUTPUT/MODEL_CONFIG.json" --all-layers \
  --prefill-source "$MOE_GPTQ_OUTPUT/layers" \
  --decode-source "$MOE_GPTQ_OUTPUT/layers" \
  --router-qdq "$MOE_GPTQ_OUTPUT/router" \
  --output "$MOE_GPTQ_OUTPUT/qnn-shared" --qnn-binding "$MOE_GPTQ_BINDING" \
  --op-package-host "$MOE_BUILD_DIR/op-package/x86_64-linux-clang/libQnnDeltaNetOpPackage.so" \
  --op-package-device /data/local/tmp/qwen35-35B-A3B/libQnnDeltaNetOpPackage_HTP.so

python -u "$MOE_ROOT/export_vision.py" \
  --model-config "$MOE_GPTQ_OUTPUT/MODEL_CONFIG.json" \
  --model /hunter/model/Qwen/Qwen3.6-35B-A3B \
  --output "$MOE_GPTQ_OUTPUT/vision" --qnn-binding "$MOE_GPTQ_BINDING" \
  --calib-images "$MOE_GPTQ_OUTPUT/calibration/images/000000039769.jpg" \
    "$MOE_GPTQ_OUTPUT/calibration/images/Statue-of-Liberty-Island-New-York-Bay.jpg"
```

三条命令串行执行。视觉校准使用真实图片；当前文本校准数据由给定 WikiText 语料生成。库文件和编译 binding 可复用已有 SDK 构建，模型 QDQ、常量、head 和专家 bank 均由本轮 checkpoint 重新生成。

按前述 `separate_shared.py` 流程分离权重时，将 `--source-dir` 改为新 `qnn-shared/`、`--head-context` 改为新 `head/head.bin`。新的 cache 编码也必须从新图读取：

```bash
g++ -O2 -std=c++17 -I"$QNN_SDK_ROOT/include/QNN" \
  "$MOE_ROOT/tools/cache_metadata.cpp" -ldl -o "$MOE_GPTQ_OUTPUT/cache_metadata"
"$MOE_GPTQ_OUTPUT/cache_metadata" "$QNN_SDK_ROOT" \
  "$MOE_GPTQ_OUTPUT/qnn-shared" "$MOE_GPTQ_OUTPUT/assets/cache-encoding.bin"
```

工具通过 QNN System API 读取每层实际 cache 的 scale/offset，按 `.io` 映射解析语义参数，包含 AR128 当前块的完整 KV 交接尺寸；不沿用旧图编码。部署时上传新 `banks/`、`indices/`，设备路径配置指向新 `separated/`、`embedding.int4`、`vision/`，推理仍使用 `run_chat.py`。

`quantization/evaluate_device.py --model ... --artifacts "$MOE_GPTQ_OUTPUT" --run /path/to/device-result` 可评估留出序列的 teacher-forced 结果，报告同一 token 上原始 safetensors FP32 链路与设备的 PPL、最终 hidden SQNR、归一化 hidden SQNR、Top1 一致率及分段变化。需要保存全部 logits，并与 `INPUTS.pt` 中第 8 或 9 条序列使用相同前缀和后续 token。
