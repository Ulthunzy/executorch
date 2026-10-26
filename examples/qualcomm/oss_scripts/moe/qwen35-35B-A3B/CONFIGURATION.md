# 配置、导出与运行

导出统一使用 `export.py --config`，内部按依赖串行完成量化（可选）、QNN 编译、共享权重分离和 PTE 打包。推理继续使用独立的 `run_chat.py`。

## 一条命令导出

```bash
conda activate et_npu
python examples/qualcomm/oss_scripts/moe/qwen35-35B-A3B/export.py \
  --config examples/qualcomm/oss_scripts/moe/qwen35-35B-A3B/configs/export_sm8975.json
```

`export_sm8975.json` 选择产物和路径，引用的 `model.json` 配置量化、KV/state、上下文及调度。主入口不导入 PyTorch/QNN，`--print-plan` 可以只查看命令和输出；`--check` 只检查本地输入，不连接设备、量化或编译。

| 导出配置 | 可选值 / 含义 |
| --- | --- |
| `source` | `calibrated` 复用已有完整 GPTQ 产物；`safetensors` 从原始 checkpoint 顺序校准 |
| `model_config` | 相对于导出配置文件的模型配方路径；所有内部阶段使用同一份快照 |
| `components` | `["text", "vision"]`、仅 `["text"]` 或仅 `["vision"]` |
| `format` | `pte`：共享图/外部权重、完整40层 host PTE 计划；`qnn`：共享 QNN context，不做设备权重采集或 host PTE 打包 |
| `phases` | `pte` 可选择 `prefill` / `decode` 的执行计划；共享 QNN 组始终同时编译 AR128/AR1 方法，维持两阶段常量和量化网格一致 |
| `calibration_dir` | 校准产物根目录；从 safetensors 开始时在此生成新的 GPTQ 产物 |
| `output` | 新的导出目录，与校准目录分开；也可用 `--output` 临时覆盖 |
| `build` / `build_dir` | 文本导出是否先编译 Native、runner 与 op package；`false` 复用指定目录的匹配构建 |
| `qnn_sdk_root` 等工具链路径 | 可以在此配置，也可以使用对应环境变量 |
| `device` / `runtime_device_dir` | `pte` 文本模式的 SDK 共享权重采集需要设备与已部署的匹配运行库；QNN context 导出不连接设备 |
| `corpus` / `images` | 文本和视觉校准数据；视觉只加载该组件所需的权重 |

模型、校准/输出/构建目录和图片的相对路径均以仓库根目录为基准；`model_config` 的相对路径以导出配置文件所在目录为基准。现有配置保持已验证的 W4/W8/N2 混合量化及 UINT16 cache，入口整理不改变这些精度。

默认配置复用保留的 GPTQ 校准产物，输出到 `llama-output/qwen35-35B-A3B-export`。改为 `source=safetensors`、把 `calibration_dir` 指向新的目录，仍使用上面同一条命令，即可从头生成。完整文本顺序校准目前需要 CUDA；每次只处理当前层/四层编译组，避免同时加载多个导出进程。

导出目录包括：

```text
MODEL_CONFIG.json        # 生效的模型配置
EXPORT_PLAN.json         # 内部步骤与命令
EXPORT_STATE.json        # 各阶段完成状态、输入与产物指纹
EXPORT.json              # 最终格式、组件和产物位置
qnn-shared/group0..9/    # 四层一组的共享 QNN context
shared-model/            # PTE 模式：graph.qnn、weights.bin、IO、cache/RoPE
plans/                   # PTE 模式：prefill-N.pte、decode.pte 及 JSON
vision/                  # 选择视觉时：vision.bin、VISION.json，PTE 模式含 vision.pte
head/                    # 选择文本时：编译后的 head
banks、indices、embedding # 指向 calibration_dir 中权重的符号链接，避免额外复制
device.json              # 完整 P/D PTE 导出的推理配置
logs/                    # 各阶段日志
```

保留符号链接所引用的 `calibration_dir`；部署仍按实际文件读取。Routed MoE 专家保持外部 Native bank，不会因为选择 QNN 格式而变成单一标准 QNN MoE 图。中间 `methods.pte` 在每组 context 提取后删除，避免与编译后的权重重复占用磁盘。

重复执行同一命令会核对检查点并跳过已完成阶段。失败阶段及其后续步骤重新执行；配置、源码或输入变化时要求换用新输出目录。`--no-resume` 只用于相同配置下主动重跑。导出目录的文件锁防止同时启动两个大模型导出。

完整 P/D 导出完成后，主机推理可直接选择生成的配置：

```bash
python examples/qualcomm/oss_scripts/moe/qwen35-35B-A3B/run_chat.py \
  --config llama-output/qwen35-35B-A3B-export/device.json \
  --prompt "请用中文介绍一下自己。"
```

`export/head.py`、`export/shared.py`、`export/vision.py` 等现在是内部阶段模块；下文的分步命令供单阶段调试，不需要作为正常导出流程逐条执行。

流程回归以替代阶段检查断点续跑，同时使用真实ExecuTorch序列化器生成并读取两阶段PTE。目录精简后的三份真实PTE和一组共享图/权重文件与已有产物逐字节相同，SM8975文本、图片整模推理通过，推理中没有生成PTE；本轮未重跑GPU校准或40层QNN编译。详情见README中的设备复核记录。

### 设备、芯片与架构

目标可以写入导出配置的 `soc_model` / `dsp_arch`，也可以在命令行覆盖：

```bash
python examples/qualcomm/oss_scripts/moe/qwen35-35B-A3B/export.py \
  --config examples/qualcomm/oss_scripts/moe/qwen35-35B-A3B/configs/export_sm8975.json \
  --soc-model SM8975 --device A7SR026915000282
```

| SoC | Hexagon 架构 | 当前验证范围 |
| --- | --- | --- |
| `SM8750` | `v79` | 专家内核交叉编译通过，整模设备验证待做 |
| `SM8850` | `v81` | 专家内核交叉编译通过，整模设备验证待做 |
| `SM8975` | `v85` | 新导出40层模型的文本、图片、采样控制及多轮设备回归通过 |

SoC 枚举及 SoC→HTP 架构关系直接复用 `backends/qualcomm/serialization/qc_schema.py` 的 `QcomChipset`、`HtpArch` 和 `_soc_info_table`，本目录不再维护第二份映射。Native 只筛选 V79/V81/V85；表中的其他同架构芯片可选择，但未完成整模设备验证。Python 配置与 `build.sh` 使用同一解析逻辑，显式 `--dsp-arch` 必须匹配。

目标传递至 Attention/head/视觉/Router 的QNN编译与校准、Native编译、GDN op package和部署库名，生成的 `MODEL_CONFIG.json`、`TARGET.json` 和图元数据记录实际目标。更换SoC必须使用新的导出/构建目录和匹配的设备QNN运行库。新入口在执行前核对 `ro.soc.model`，不会将V85库作为V79部署。

### 一个函数执行推理

正常运行由 `workflow.inference.inference(args)` 进入设备端 `qwen35_moe_runner` 会话；tokenizer、图片解码、采样和多轮state/KV更新在设备内执行。命令行只需：

```bash
python examples/qualcomm/oss_scripts/moe/qwen35-35B-A3B/run_chat.py \
  --weights-path llama-output/qwen35-35B-A3B-export \
  --device A7SR026915000282 --soc-model SM8975 \
  --temperature 0.7 --max-new-tokens 256 \
  --image-path /path/to/image.jpg \
  --interactive --stream \
  --prompt "请描述图片，然后回答我的追问。"
```

这是新统一导出完成后的用法。正常会话需要本版runner与支持有效长度的AR128图；旧runner缺少新控制项或旧图缺少有效长度输入时，本地校验会要求重建/重新导出，避免静默混用。已经完成的旧benchmark记录仍保留。

| 参数 | 行为 |
| --- | --- |
| `--weights-path` / `--artifacts` | 本地权重根目录，读取其中 `device.json` / `MODEL_CONFIG.json`、共享图、embedding、专家bank |
| `--device` | ADB设备序列号 |
| `--soc-model` / `--soc_model` | 芯片目标；必须匹配已编译图、Native库与手机 |
| `--temperature` / `--temp` | 0使用greedy，正数使用已有ExecuTorch Sampler，top-p沿用0.9 |
| `--ignore-eos` / `--no-ignore-eos` | 是否忽略EOS；输出始终受最大长度及上下文剩余空间限制 |
| `--image-path` | 上传原始JPG/PNG，由设备内解码；省略为文本输入 |
| `--with-vision` / `--no-with-vision` | 保留视觉塔以支持后续 `/image`；初始图片自动开启，纯文本会话默认关闭 |
| `--interactive` / `--no-interactive` | 是否在首轮后继续交互，整次会话复用context、state/KV和专家cache |
| `--stream` / `--no-stream` | 实时逐token刷新，或生成完一轮再显示；不是每轮重新启动模型 |
| `--enable-thinking` / `--no-enable-thinking` | 模型原生thinking模板开关 |
| `--remote-weights` | 设备端UFS上的bank/indices根目录 |
| `--deploy-weights` / `--no-deploy-weights` | 默认用ADB `--sync` 同步专家权重和索引；已有完整权重可关闭同步 |
| `--runner diagnostic` | 保留原单轮张量dump、teacher-forcing、性能诊断链路；固定greedy，不做多轮 |

`temperature`、`ignore_eos`、`interactive` 和 `stream` 也可以放在模型配置的 `generation` 中；CLI优先。原来带 `--teacher-tokens`、`--prefill-token-file` 或张量dump的命令会自动选择诊断链路。`--no-stream` 只改变显示时机，token与计时仍记入设备日志。

`run_chat.py` 还提供 `run_inference(config=None, weights_path=None, device=None, soc_model=None, prompt=..., temperature=None, ignore_eos=None, image_path=None, with_vision=None, interactive=None, stream=None, enable_thinking=None, max_new_tokens=None)`。省略的生成控制项继承配置；交互会话阻塞到 `/exit`，返回本次结果目录。主机输出采用增量UTF-8解码，实时token无需等待换行。每次运行保存 `console.log`、设备 `runner.log` 和 `RUN.json`。

## 配置分工

| 文件 | 内容 | 使用位置 |
| --- | --- | --- |
| `configs/model.json` | 量化配方、KV/state 类型、静态容量、thinking、调度 | 校准、导出、打包、推理 |
| `configs/export_sm8975.json` | 导出组件、格式、源目录、输出目录和 SDK/设备设置 | `export.py --config` |
| `configs/sm8975.json` | 本机模型/产物/构建目录、设备序列号、设备依赖和专家路径 | `run_chat.py --config` |
| `workflow/model_config.py` | 默认值、类型检查、支持范围、图配置契约 | 各 Python 入口 |
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

### 设备端运行

`build.sh` 同时生成 `qwen35_moe_runner`。主机 `run_chat.py` 部署时会上传该程序和 `tokenizer.json`；之后可直接在设备终端交互，无需主机参与文本编码或解码：

```sh
cd /data/local/tmp/qwen35-raw-gptq-20261024/runtime/w2a8-fused-c1152-p128-cache32/prefill
./qwen35_moe_runner --model-root .. --tokenizer-path ../../tokenizer.json --interactive
```

`--model-root` 包含 `prefill/` 和 `decode/`；共享权重默认位于它的上一级 `shared-model/`，也可用 `--shared-model` 指定。运行参数包括 `--prompt`（可重复）、`--system-prompt`、`--max-new-tokens`、`--temperature`、`--enable-thinking`、`--max-seq-len` 和 `--cache-experts`。容量必须匹配已编译图，不能在命令行将 1152 图扩展成 2048。当前默认容量 1152、缓存 32 槽。

终端首轮从 `prefill/plans/PLANS.json` 读取配置的 prefill 顺序，并根据真实 padding 后的 token 数选择 `prefill-{N}.pte`。导出阶段已经生成128、256、…、1024的8个无权重host计划，总计约0.61MB；部署只读取并上传它们。默认 `layer_major` 在当前层处理完全部AR128块再切到下一层，复用该层已加载的专家；选择 `chunk_major` 时需使用该调度对应的导出计划，推理不临时重打包。QNN/Native kernel仍固定AR128，未重新编译成AR256或AR1024。

计划集合必须与实际编译容量一致，缺少manifest或所选PTE时提前报错。当前单层memo最多额外申请210MiB，在进入decode前释放。原Python运行仍使用该次输入对应的显式PTE，不受终端自动选择影响。

新版图的首轮和后续轮输入均按 AR128 分块，尾块补齐并仅提交有效 token 的 state/KV；AR1 用于生成。`/reset` 清空历史 state/KV，保留 context 和专家缓存，`/exit` 结束会话。上下文超限时整条新输入拒绝追加，不会截断一部分问题或自动丢失历史。

同一条请求先组成包含模板、文本和图片 token 的完整序列，再统一按 `ceil(valid_tokens / 128)` 分块，不分别补齐文字与图片。比如总计 253 个有效 token，只执行两个 AR128，第二块有效长度为125；末尾3个位置不推进 state/KV。`CHAT_INPUT` 记录有效总数、文本及模板数量、图片数量、补齐后的总数和块数。2026-10-26 的独立分词样例中，纯文本为27个；图文为226个（文本及模板30＋图片196），后者已经包含文本，不能再加一次27。

设备终端直接读取 JPG/PNG，无需主机预处理：

```sh
./qwen35_moe_runner --model-root .. --tokenizer-path ../../tokenizer.json \
  --image-path ../../test.jpg --prompt "请用中文描述这张图片。" --interactive
```

交互中 `/image "图片路径" 问题` 加入新图片；直接输入问题则继续追问。视觉塔在模型初始化时加载并保留到退出，图像 token 的 embedding 和三维 MRoPE 在内部更新；新版图的后续图片也使用有效长度 AR128 输入路径。

模型根目录需要 `VISION.json`、`model-config.json`、原始 `base-cos.f32` / `base-sin.f32`，以及 `prefill/vision.bin`；当前部署流程自动准备。原始位置表保持只读，混合位置表写入模型根目录，避免覆盖 Python 入口的 `prefill/cos.f32` 和 `decode/cos.f32`。视觉精度仍为 W8A16 GPTQ，图片尺寸、patch/merge/temporal 参数按编译元数据校验。

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

`--max-seq-len` 选择匹配的静态图，不会自动修改旧图容量。旧 `--decode-capacity` 是同一参数的别名。模型配置支持1024～32768的128对齐总上下文，需要导出对应容量；当前MoE接口仍要求UINT16 KV/GDN/conv cache。

`--no-kv-append-only` 保留为布局诊断选项。在同一组 128-token/15 步 teacher 测试中，相对默认路径 hidden SQNR 25.76 dB、logits 26.70 dB，token 一致但浮点数值不是逐字节一致；关闭所有其他优化所得结果与单独关闭 KV append 完全一致。本轮没有逐算子证明该差异的来源，也未为诊断布局重测 PPL，默认继续使用 append-only。

## 导出共享图、权重与 PTE

完整导出只执行 `export.py --config configs/export_sm8975.json`。`format=pte` 时自动完成校准或加载冻结校准结果、共享 QNN 编译、SDK 权重回调范围采集、图/权重分离和 PTE 序列化，不需要额外的手动命令。`format=qnn` 则保留共享 QNN context，不生成 PTE。

| 内部实现 | 职责 |
| --- | --- |
| `export/shared.py` | 编译时就启用 QNN weight sharing，同一四层组包含 prefill/decode 方法 |
| `export/capture.py`、`export/weights.py` | 根据真实 SDK 回调范围将已共享的 context 保存为 `graph.qnn` 与 `weights.bin`；这是文件布局分离，不是在此时才启用共享 |
| `export/program.py` | 序列化完整40层执行计划，122个阶段引用外部共享 context、head和专家缓存 |
| `workflow/plans.py` | 推理时只选择并校验导出的计划，不调用导出或序列化代码 |
| `workflow/vision.py` | 校验视觉元数据，保留离线诊断所需的图片参考处理；正常图片解码在设备C++中完成 |

标准产物为：

```text
output/
├── MODEL_CONFIG.json
├── EXPORT.json
├── device.json
├── plans/
│   ├── PLANS.json
│   ├── decode.pte / decode.json
│   └── prefill-128.pte … prefill-1024.pte（含对应JSON）
├── shared-model/
│   ├── group0 … group9/  # 每组图/权重各一份，供两个阶段共用
│   └── head/            # 公共head
├── assets/              # cache编码和RoPE
├── banks/、indices/、embedding.int4  # 外部专家及embedding
└── vision/              # 选择视觉组件时导出
```

Prefill计划的长度表示主机调度的总输入长度，QNN/Native仍执行AR128块；这些小型PTE不复制模型权重。`run_chat.py --weights-path output` 读取 `plans/`，缺失、容量、量化网格、共享IO签名或调度不匹配时提前报错。调整静态配置或prefill顺序后，使用统一导出入口生成对应产物。运行端不再提供自动补打包的路径。

`--local-prefill-pte`、`--local-decode-pte` 只用于选择已有的诊断计划及其JSON；正常聊天使用完整导出目录。

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

`--no-enable-thinking` 显式关闭。文本和图片都通过模型原生 chat template 传递 `enable_thinking`，不是仅修改日志标签。`max_new_tokens` 包含 thinking 产生的 token。正常会话支持temperature采样及EOS开关，诊断链路保持greedy。

普通文本尾部使用 inactive padding，不向用户提示插入空白文字；只有 `--prompt-tokens 1024` 等显式性能测试才用真实空白补长。`PROMPT.txt` 保存实际模板内容。`--prefill-token-file` 表示输入已经完成模板编码，不会再套 system/thinking；`RUN.json.input_mode` 与 `template_enable_thinking` 明确记录这一点。

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

在 `configs/export_sm8975.json` 中设置 `source` 为 `safetensors`，将 `calibration_dir` 和 `output` 指向新的目录；`corpus`、`images` 指向实际校准数据。仍执行本文开头的同一条 `export.py --config` 命令，无需手动运行 prepare/head/shared/vision/分离/打包。

当前层从 safetensors 读取，按 Attention → Router/shared → 全部256专家的顺序推进；下一层使用上一层实际量化输出。子进程逐阶段释放显存，内部 `prepare_model.py` 的层级检查点与外层导出检查点共同支持续跑。原始 FP32 链路另行传播，拟合、选择和留出序列分离；不更改 GPTQ 配方与数据用途。

统一入口先生成 QDQ、常量、专家 bank、INT4 embedding 和 head，再串行编译新 head、十组共享 AR128/AR1 context 及视觉编码器。cache 编码通过 QNN System API 从新 context 和实际 `.io` 映射读取，RoPE 按选定上下文容量生成，不沿用旧图的 scale/offset。最终自动分离共享权重、包装完整执行计划并生成推理配置。

`quantization/evaluate_device.py --model ... --artifacts "$MOE_GPTQ_OUTPUT" --run /path/to/device-result` 可评估留出序列的 teacher-forced 结果，报告同一 token 上原始 safetensors FP32 链路与设备的 PPL、最终 hidden SQNR、归一化 hidden SQNR、Top1 一致率及分段变化。需要保存全部 logits，并与 `INPUTS.pt` 中第 8 或 9 条序列使用相同前缀和后续 token。
