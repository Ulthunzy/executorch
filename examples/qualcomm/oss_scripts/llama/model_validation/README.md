# 模型量化与部署验证

SafeTensors GPU 基线、校准数据准备、QDQ PT2/PTE 精度测试、对话消融与报告的统一入口。
替换模型时修改配置中的路径和实验参数，不再复制 08/09/10 的实验脚本。

本目录只管理工具、配置示例和小型用例。权重、完整数据集、设备库、日志和输出继续存放在外部路径或 `llama-output/`；移动工具不会移动或覆盖历史实验数据，也不会替换应用 AAR。

## 先明确支持范围

| 任务 | 当前入口 | 边界 |
|---|---|---|
| SafeTensors 对话 | `run.py hf` | CUDA、Qwen3/Lincal ChatML、贪心温度 0；其他架构需新增适配器 |
| PTE 对话 | `run.py pte run` | 手机 QNN、Lincal/Qwen3/Qwen2.5、KV/hybrid；保留完整真实多轮 KV |
| PT2 对话 | `run.py pt2` | CUDA、显式 `lincal-full-sequence` QDQ 布局；原图逐次完整历史回放及因果检查，较慢，不是生产 KV 性能 |
| 标准 PPL | `run.py ppl prepare/run` | 固定文本、token IDs、窗口；各后端能力与驱动限制见 [PPL README](ppl/README.md) |
| 外部测试集 | `run.py sample run` | 已有 sample bundle 格式，详见 [测试集说明](benchmarks/SAMPLE_BENCHMARK.md) |
| 校准准备 | `calibration-build` / `calibration-check` | 现有 Luxi/ChatML think 筛选规则，不是任意架构的通用 SFT 工具 |
| 量化/PTE 导出 | `run.py export` | 转发原 `llama.py`；不修改它的模型注册、参数或导出行为 |
| 消融调度 | `run.py workflow` | JSON 有序阶段、参数矩阵、依赖、日志、检查后续跑 |

“能换模型”首先指相同受支持架构的新权重/导出版本，不意味着任意 `.pt2` 或聊天模板都可直接使用。标准 PPL 的 tokenizer/词表、计分区间必须可比。旧 08/09/10 mask 修补是诊断，不会自动应用到新 PT2。

## 目录

```text
model_validation/
├── run.py                # 统一入口
├── configs/              # 可复制修改的 JSON 流程示例
├── common/               # 工具注册、调度、输入对齐
├── engines/              # HF/PT2/PTE PPL 执行适配
├── calibration/          # 校准数据准备和校验
├── ppl/                  # 标准 PPL、隔离的设备驱动
├── benchmarks/           # 身份/多轮/外部测试集、生成驱动
├── ablation/             # 消融入口与历史归档说明
├── reporting/            # 跨运行汇总
└── tests/                # 新入口及配置回归测试
```

新实验统一使用 `run.py workflow`；硬编码 08/09/10 的旧队列和候选图诊断脚本已归档移出。

## 1. 环境和入口

以下命令从仓库根目录运行：

```bash
cd /hunter/cpp_code/ET/npu/executorch
conda activate et_npu
source env.sh

VALIDATION=examples/qualcomm/oss_scripts/llama/model_validation
python "$VALIDATION/run.py" list
python "$VALIDATION/run.py" hf --help
python "$VALIDATION/run.py" ppl --help
python "$VALIDATION/run.py" pte run --help
```

`list`、配置预览和离线汇总不加载模型。HF/PT2 需要环境中的 CUDA PyTorch；PTE 需要已构建的 QNN runner、ADB 及匹配的设备 QNN 库。工具不自动下载安装或升级依赖。

子工具的参数原样转发，依然可以直接运行新目录内的脚本。相对路径以仓库根目录为准；非交互式终端应先初始化 conda 再激活环境。

## 2. 最常用：新模型的四组对话消融

复制示例到一个新文件后编辑：

```bash
cp "$VALIDATION/configs/generation.example.json" "$VALIDATION/configs/my-model-generation.json"
```

至少修改 `variables` 中的 `hf_model`、`pte_model`、两个 tokenizer 路径、`serial`、`device_lib_dir`；确认 `driver` 存在，并为这次实验设置新的 `output_root`。

示例默认：50 组两轮、100 次回答；无额外 system prompt；温度 0；think 开/关 × RP 1.0/1.1；窗口 128；双保护关闭。身份参考为陆小兮 / Lu Xiaoxi，只用于评分，不注入输入。

先预览全部命令，不启动 GPU/手机、不创建结果文件：

```bash
python "$VALIDATION/run.py" workflow \
  --config "$VALIDATION/configs/my-model-generation.json" \
  --all --dry-run
```

只跑 GPU 基线及其输入准备阶段：

```bash
python "$VALIDATION/run.py" workflow \
  --config "$VALIDATION/configs/my-model-generation.json" \
  --stage hf-ablation --with-dependencies
```

之后跑同一份配置的手机部分：

```bash
python "$VALIDATION/run.py" workflow \
  --config "$VALIDATION/configs/my-model-generation.json" \
  --stage pte-ablation --with-dependencies --resume
```

从头顺序跑全部阶段用 `--all`；中断后在相同命令后加 `--resume`。同一结果根目录同一时间只允许一个调度器；不同配置中相同 `resource` 名称的 GPU/设备任务也互斥，但无法阻止用户直接启动的其他程序。调度器不会停止别的训练或推理进程。

要比较 1.0、1.05、1.1，修改两组阶段的 `matrix.rp` 为 `[1.0, 1.05, 1.1]`，并改用新的 `output_root`。不要在已跑过的目录里改变实验条件继续追加。

### 手机驱动只需提前构建

```bash
python "$VALIDATION/run.py" build-driver \
  --build-folder build-android \
  --output build-android/model_validation/generation \
  --jobs 2
```

构建依赖已有 **CMake Unix Makefiles** 的 `qnn_llama_runner` 编译/链接文件；不是 Ninja 构建的通用替代。只产生隔离测试驱动，不替换 AAR 或常规 runner。

### 单独试两轮或一组问题

```bash
python "$VALIDATION/run.py" hf \
  --model /path/to/safetensors-model \
  --tokenizer /path/to/safetensors-model \
  --device cuda:0 --dtype bfloat16 --attention sdpa \
  --seq-len 4096 --ids I01 \
  --repetition-penalty 1.0 --repeat-last-n 128 \
  --output llama-output/model-validation/my-model/hf-smoke
```

这里 `I01` 是已有题集中的会话 ID。自定义“你好/你是谁”请通过 `--cases /path/to/cases.json` 指定文件；[用例结构和评分](benchmarks/README.md)包含格式说明。HF、PT2、PTE 都应使用相同的 cases。

PTE 单独测试（`TOKENIZER_JSON` 是本地文件，`DEVICE_LIBS` 是手机目录）：

```bash
PTE=/path/to/hybrid_llama_qnn.pte
TOKENIZER_JSON=/path/to/tokenizer.json
SERIAL=YOUR_ADB_SERIAL
DEVICE_LIBS=/data/local/tmp/qnn-libs

python "$VALIDATION/run.py" pte run \
  --model "$PTE" --tokenizer "$TOKENIZER_JSON" \
  --device "$SERIAL" --device-lib-dir "$DEVICE_LIBS" \
  --driver build-android/model_validation/generation/identity_eval_driver \
  --mode hybrid --decoder lincal --seq-len 4096 \
  --repetition-penalty 1.05 --repeat-last-n 128 \
  --repair-think-eos --enable-think-budget --think-budget 512 \
  --ids I01 \
  --output llama-output/model-validation/my-model/pte-protected-smoke
```

不加 `--disable-thinking` 就是开启 think。上例明确开启双保护；四组基线配置则明确关闭。两者不要混为同一组。`--device-model /data/local/tmp/.../model.pte` 可以复用已上传且哈希一致的模型。

PT2 单组测试：

```bash
python "$VALIDATION/run.py" pt2 \
  --model /path/to/decode_qdq.pt2 \
  --tokenizer /path/to/safetensors-model \
  --pt2-layout lincal-full-sequence \
  --device cuda:0 --seq-len 4096 --ids I01 \
  --repetition-penalty 1.0 --repeat-last-n 128 \
  --output llama-output/model-validation/my-model/pt2-smoke
```

这是保守的完整固定图执行，保留自己的全 token 历史，但不是增量 KV 加速。先跑小样本；遇到未来 token 依赖、输入布局或 logits 不符会拒绝，不会悄悄替换成 HF 或改图。可按 `hf-ablation` 的矩阵写一个 `tool: "pt2"` 阶段。旧改图加速实验仍见 [专项说明](ablation/README.md)。

## 3. 标准 PPL

标准 PPL 与生成策略分开：固定参考文本，累计所有目标 token 的 NLL，再计算 `exp(total_nll / total_targets)`。不应用 RP、温度、强制 think 闭合或预算截断；不添加 ChatML/BOS。

准备原始测试文本，例如将固定版本 WikiText-2-raw-v1 **test** 的 `text` 列按双换行连接保存为 UTF-8 TXT。工具按该文件原文编码并记录哈希，不会替你把 train/test 混合，也不会默默截短语料。不要把校准 train 文本冒充 test。

```bash
cp "$VALIDATION/configs/ppl.example.json" "$VALIDATION/configs/my-model-ppl.json"
# 编辑权重、tokenizer、文本、设备路径以及新的 output_root。

python "$VALIDATION/run.py" workflow \
  --config "$VALIDATION/configs/my-model-ppl.json" --all --dry-run

python "$VALIDATION/run.py" workflow \
  --config "$VALIDATION/configs/my-model-ppl.json" \
  --stage hf-ppl --with-dependencies
```

需要测 PT2 时选择 `--stage pt2-ppl --with-dependencies --resume`；PTE 类似。先完成并核对所选后端，再选择 `report`。若只要 HF/PTE，可用 `report` 子工具显式传这两个结果目录，不必伪造 PT2 分数。

PPL 使用专门驱动，不能拿普通生成驱动代替：

```bash
python "$VALIDATION/run.py" build-driver \
  --build-folder build-android \
  --entry-point "$VALIDATION/ppl/ppl_driver.cpp" \
  --define ET_QNN_PPL_EVALUATION \
  --output build-android/model_validation/ppl \
  --jobs 2
```

构建器在独立构建输出内准备 teacher-forcing 头文件覆盖层，不依赖手动修改生产 `decoder_runner.h`；正常生成构建不要加该宏。具体参数、KV/hybrid 支持及 PPL 完整性审计见 [PPL README](ppl/README.md)。

## 4. 校准集准备与量化/PTE 导出

使用独立于测试集的校准问题。已有工具和小型候选题集在 `calibration/`，完整的筛选、复核、合并、校验命令见 [校准 README](calibration/README.md)。

```bash
python "$VALIDATION/run.py" hf \
  --model /path/to/safetensors-model \
  --device cuda:0 --dtype bfloat16 --seq-len 4096 \
  --cases "$VALIDATION/calibration/luxi_multiturn_prompts.json" \
  --output llama-output/model-validation/my-model/calibration-gpu

python "$VALIDATION/run.py" calibration-build inspect-gpu \
  --tokenizer /path/to/safetensors-model \
  --gpu-run llama-output/model-validation/my-model/calibration-gpu \
  --output llama-output/model-validation/my-model/calibration-candidates.json
```

审核候选后再 merge，不把模型自己生成的错误答案自动当标准答案。对话数据保留完整 think、正式回答、EOS；激活统计由量化时的观察器收集，本步骤不是导出所有 HF 中间层数据。历史校准集与部分身份题有重合，旧题集主要用于回归，不能据此宣称独立泛化能力。

量化继续使用既有 `llama.py`，统一入口只做转发：

```bash
python "$VALIDATION/run.py" export --help
```

你已有的 `llama.py` 编译命令可以把开头替换为 `python "$VALIDATION/run.py" export`，后面的参数不变。必须确认当前 decoder 配置/显式 checkpoint 对应的是这次的权重。原 `--checkpoint` 读取转换后的 PyTorch checkpoint，不是直接接收 SafeTensors 文件夹；不要将 HF 测试的 `--model` 参数直接搬给 exporter。

校准文件通过现有 `--calib_text_datasets` 和 `--calib_text_preserve_sources` 接入。需要后续 PT2 测试时 **不要使用 `--skip_qdq_save`**；该开关会跳过 QDQ PT2 评估产物。校准 AR 与部署 prefill AR 分别按原导出逻辑配置，不由测试框架强制绑定。

## 5. 检查输入，查看完整输出和结果

`generation.example.json` 的 `inputs` 阶段冻结 HF 编码的增量 ChatML；手机测试显式读取 `canonical.json`。它不等于已验证手机原生 tokenizer。要核对实际设备编码，单独运行：

```bash
python "$VALIDATION/run.py" check-inputs \
  --tokenizer /path/to/safetensors-model \
  --device-tokenizer "$TOKENIZER_JSON" \
  --cases "$VALIDATION/benchmarks/cases.json" \
  --serial "$SERIAL" --device-lib-dir "$DEVICE_LIBS" \
  --driver build-android/model_validation/generation/identity_eval_driver \
  --output llama-output/model-validation/my-model/native-input-check
```

检查会上传独立驱动、tokenizer 和输入文件；发现差异返回非零状态，保留完整对照。不要把“原生文本输入”与“强制 canonical token 输入”结果混作同一组。即使第二轮用户问题相同，三端第一轮答复不同也会使历史不同；真实多轮质量与固定历史的数值对齐要分开解释。

每个对话运行目录保留：

- `manifest.json`：模型、tokenizer、配置与来源指纹。
- `cases.json`、`sessions/*/input.json`：题目；每轮 JSON 记录实际 prompt/token 信息，字段按执行器区分。
- `sessions/*/*.txt`：原始完整回答，包含 think 和结束标记。
- `ANSWERS.md`、`REPORT.md`、`results.csv`、`summary.json`：便于查看的文本与汇总。
- `review_template.json`：按原始记录指纹绑定的复核模板。

姓名关键词匹配和重复检测只作筛查；未复核时正确率保持空值。请先按 [评分说明](benchmarks/README.md)填写复核文件，再汇总：

```bash
python "$VALIDATION/run.py" pte summarize /path/to/generation-run \
  --reviews /path/to/reviews.json

python "$VALIDATION/run.py" report \
  /path/to/hf-run /path/to/pte-run \
  --output llama-output/model-validation/my-model/comparison-v1
```

`pte summarize` 是共享的离线对话报告入口，也能汇总本框架产生的 HF/PT2 会话，不会启动手机。`report` 可同时收集显式传入的 PPL/对话目录，生成 CSV/JSON/Markdown，缺失、待复核、无标准 PPL 有效性证据的结果不会当作零分或合格分数。

按会话整理“第一轮输入/实际输出/参考答案、第二轮……”的错误表：

```bash
python "$VALIDATION/run.py" errors /path/to/generation-run \
  --reviews /path/to/reviews.json \
  --references /path/to/references.json \
  --output llama-output/model-validation/my-model/errors-v1
```

`references.json` 是 `{"I01T1": "参考答复", "I01T2": "参考答复"}` 这样的 ID→文本映射。两项都可省略：未提供复核的会话列为待复核，不冒充错误；未提供参考答案则标注缺失，不自动编造标准答案。报告包含 Markdown、CSV 和来源哈希，并链接原始完整回答。

## 6. 后续如何添加流程

配置格式、参数矩阵、依赖与新增工具的方法见 [扩展指南](EXTENDING.md)。

只增加已有工具的参数组合时，复制一个 `steps` 项即可；增加新的后端或任务类型时，在相应目录实现独立 CLI 并在 `common/registry.py` 注册。不要把某台设备、新模型版本号写死进调度器。

完整实验至少记录原始权重→PT2→PTE 的对应关系、量化配置和数据来源、tokenizer/template、模式与上下文/AR、生成策略、代码版本及复核口径。单独改图的诊断产物必须使用新的运行名称，不能覆盖原模型的报告。

## 7. 兼容、验证与注意事项

旧 `self_identity_eval/*.py` 和 `calibration/*.py` 保留轻量转发入口；后续维护以本目录为准。旧 `self_identity_eval/ablation/*.py` 已随专项脚本一起归档，不再提供入口。小型旧 JSON 用例路径保留兼容，完整数据集和历史结果未移动。

旧 CLI 兼容不代表旧结果目录可以无条件续跑：迁移后脚本指纹变化时，旧运行会拒绝恢复。保留原始数据；如果持有当时的代码快照，可用它复现旧实验，否则使用新目录重跑。不要删除 manifest 绕过检查。

工作流在 `workflow.json` 记录执行状态，终端输出同步保存到 `workflow-logs/`。退出码成功且声明的输出文件存在、已记录哈希，表示该阶段执行完毕；对话是否语义正确仍需复核，不能仅看调度器的 complete。

对 PPL/对话 `summary.json`，汇总和工作流跳过已完成任务前还会只读核对原始窗口/会话、完成标记、复核绑定与计分结果；删除原始记录后不能仅凭汇总文件继续发布分数。输入准备和报告先写同目录 staging，完成后再发布，失败残留的隐藏目录保留供排查。

HF/PT2/PTE 直接运行同一结果目录也有排他锁。已完成对话 `--resume` 核对后直接返回，不重新推送或加载权重；中途续跑后的汇总保留已绑定的人工复核。回答记录变化时，旧复核会被拒绝，而不是自动套给新回答。

取消会终止本机任务进程组；ADB 连接断开不保证手机 driver 已退出。设备阶段异常后会标记设备状态未知；请先检查手机上的旧进程，确认已停止，再使用 `--resume --device-idle-confirmed`。工具不擅自终止其他设备任务。

回归测试（需激活 `et_npu` 并 `source env.sh`）：

```bash
python -m pytest -q "$VALIDATION/tests" "$VALIDATION/ppl" \
  "$VALIDATION/benchmarks" "$VALIDATION/calibration"
```

设备集成测试必须显式按其文档配置运行，不应把历史长实验当成日常单元测试。不要在整理代码时自动重跑全量 PPL/100 条对话或上传新模型。

已构建上述两个 driver 后，可运行小型真机回归（会上传隔离的小模型和 tokenizer，不替换手机已有模型）：

```bash
ET_THINKING_DEVICE="$SERIAL" \
ET_THINKING_DRIVER="$PWD/build-android/model_validation/generation/identity_eval_driver" \
ET_THINKING_TOKENIZER_DIR=/path/to/local-hf-tokenizer \
ET_THINKING_LIBDIR="$DEVICE_LIBS" \
python -m pytest -q "$VALIDATION/benchmarks/test_thinking_device.py"

ET_PPL_DEVICE="$SERIAL" \
ET_PPL_DRIVER="$PWD/build-android/model_validation/ppl/identity_eval_driver" \
ET_PPL_LIBDIR="$DEVICE_LIBS" \
python -m pytest -q "$VALIDATION/ppl/test_ppl_device.py"
```

前者检查 KV/hybrid、双保护四种组合、长度上限和多轮续接；后者用可解析求出期望值的小模型逐 token 核对 NLL，覆盖 SWA 回绕、AR 尾部、跨窗口 reset 和完成后 resume。合成模型用于验证测试工具，不代表真实模型质量。

本次真实 09 tokenizer 检查中，200 条增量输入有 18 条 HF/设备编码不同（涉及 9 条问题的 think 开/关两种模式，例如“，请”被分成不同 token）。不要忽略 `check-inputs` 的非零返回码；通用消融示例使用明确的 canonical 输入控制此变量，不代表已修复生产 tokenizer。
