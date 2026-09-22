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
| 三端业务集 | `run.py business` | HF/PT2/PTE 共用 plain prompt、ID、参考答案和 scorer；开放题保留人工复核 |
| 标准 PPL | `run.py ppl prepare/run` | 固定文本、token IDs、窗口；各后端能力与驱动限制见 [PPL README](ppl/README.md) |
| 量化质量门禁 | `run.py quality-gate` | 强制审计三端 PPL/生成证据、lineage 和可比性，输出 PASS/FAIL |
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
├── common/               # 工具注册、调度、输入对齐、导出 lineage
├── engines/              # HF/PT2/PTE PPL 执行适配
├── calibration/          # 校准数据准备和校验
├── ppl/                  # 标准 PPL、隔离的设备驱动
├── benchmarks/           # 身份/多轮/通用业务/外部测试集、生成驱动
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
python "$VALIDATION/run.py" business --help
python "$VALIDATION/run.py" quality-gate --help
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

你已有的 `llama.py` 编译命令可以把开头替换为 `python "$VALIDATION/run.py" export`。为了生成可用于质量门禁的完整 lineage，还要显式增加 `--source_safetensors /path/to/original-hf-model`。原 `--checkpoint` 仍读取转换后的 PyTorch checkpoint；`--source_safetensors` 只记录原始 SafeTensors 内容指纹，不改变导出输入。

校准文件通过现有 `--calib_text_datasets` 和 `--calib_text_preserve_sources` 接入。需要后续 PT2 测试时 **不要使用 `--skip_qdq_save`**；该开关会跳过 QDQ PT2 评估产物。导出成功后，artifact 目录新增 `export_lineage.json`，绑定 SafeTensors、checkpoint、tokenizer、本地校准文件、量化/导出参数、代码 commit 与 working-tree 内容指纹、QNN SDK build ID、PT2 与 PTE 内容哈希。只有本次导出调用实际新生成的 `decode_qdq.pt2` 才会进入 lineage，artifact 目录里的旧文件不会冒充本次输出。

缺少本地 SafeTensors、checkpoint、tokenizer、本地且内容可哈希的校准来源或本次 PT2 输出时，manifest 会标记 `complete: false`，不能通过质量门禁。Hub/lm-eval 数据集名称没有固定 revision 和实际样本内容，因此当前只能作为外部描述，不能产生完整 lineage；QAT 的远程训练集同理。需要正式门禁时先物化为本地、受版本控制的输入。校准 AR 与部署 prefill AR 分别按原导出逻辑配置，不由测试框架强制绑定。

## 5. 三端业务集与质量门禁

复制 [business workflow](configs/business.example.json) 和 [业务用例示例](configs/business_cases.example.json)，将自己的 holdout 数据写成同一份 sessions/turns JSON。每个 turn 可声明 `critical: true`，并选择以下 scorer：

- `exact_match`：规范化空白和 Unicode 后精确匹配；可用 `case_sensitive`。
- `contains`：参考字符串全部或任一包含，`mode` 为 `all|any`。
- `choice`：保守提取明确的 A～P 选项。
- `numeric`：保守提取有限十进制或分数并做精确数值比较。
- `manual`：不自动判断语义，必须人工填写 `passed`。

所有 scorer 还会共同检查执行成功、EOS、最终答复、think 闭合、重复和乱码。自动 scorer 无法保守提取时也保持待复核，不猜测通过。HF/PT2/PTE 使用同一用例 ID 和 shared scorer。workflow 会先用 HF tokenizer 冻结 incremental ChatML token IDs，并将 canonical IDs 交给 PTE；门禁还会逐题比较三端实际记录的 `incremental_input_ids`，并要求 PTE 同时记录的原生 tokenizer IDs 完全相同。文本相同但 token 不同会直接判为证据不可比。

业务 workflow 在三端使用相同的 `--max-new-tokens`。示例默认 128；对于 PT2 完整序列回放，必须根据用例预期长度设置有限预算，否则单个未及时输出 EOS 的用例可能运行很久。达到预算会记录为 `length_limit`，不会伪装成正常 EOS：

```bash
cp "$VALIDATION/configs/business.example.json" \
  "$VALIDATION/configs/my-model-business.json"
python "$VALIDATION/run.py" workflow \
  --config "$VALIDATION/configs/my-model-business.json" --all
```

每端运行后，复制其 `review_template.json` 为独立 review 文件，填写待复核项的 `passed: true|false`，保留 `record_hash` 和 `run_fingerprint`，再重新汇总：

```bash
cp /path/to/business/hf/review_template.json /path/to/business/hf/reviews.json
# 编辑 reviews.json；PT2/PTE 同样处理。
python "$VALIDATION/run.py" business summarize /path/to/business/hf \
  --reviews /path/to/business/hf/reviews.json
```

三端业务结果全部完成复核，且三端 PPL 都由新版工具重新产生后，复制 [门禁 policy](configs/quality_gate.example.json) 与 [门禁 workflow](configs/quality_gate.workflow.example.json)，填写六个结果目录及 `export_lineage.json`：

```bash
cp "$VALIDATION/configs/quality_gate.workflow.example.json" \
  "$VALIDATION/configs/my-model-quality-gate.json"
python "$VALIDATION/run.py" workflow \
  --config "$VALIDATION/configs/my-model-quality-gate.json" --all
```

门禁固定执行原始 evidence audit，并强制检查三端 PPL 的 token/target/tokenizer/context/stride/metric 完全可比、三端业务 ID/生成配置/tokenizer.json/逐题实际输入 token 相同、所有题目已评分、每端业务模型与对应 PPL 模型相同、lineage 与实际评测模型哈希一致。业务 evidence 缺少原始 `session.log` 会被拒绝。这些 correctness 项在 policy 中必须保持 `true`，不能用调低阈值的方式绕过。

输出目录包含 `REPORT.md`、`gate.json`、`metrics.json` 和逐题 `regressions.csv`。`gate.json` 内嵌 policy、六端 evidence 指纹和自哈希；resume 会重新审计源目录并重算完整门禁，四个报告文件都受 workflow 输出指纹保护。报告覆盖三端 PPL/mean NLL、PT2→PTE teacher-forcing top-1 一致率、通过率、EOS、重复、乱码、每个转换阶段的新增回归与恢复，以及所有声明/配置的关键 ID 零新增回归。退出码 `0` 表示 PASS，`1` 表示证据有效但指标 FAIL，`2` 表示证据无效或不可比。workflow 会把退出码 1 保存为“执行完成、结论 FAIL”，恢复时复核并继续返回 1，不会把它误作中断任务重跑。旧 PPL 结果没有 `top1_token_ids`，必须重跑，不能补写 summary 冒充新版证据。

## 6. 检查输入，查看完整输出和结果

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

`pte summarize` 是共享的离线身份对话报告入口，也能汇总本框架产生的 HF/PT2 会话，不会启动手机。业务用例使用 `business summarize`。`report` 可同时收集显式传入的 PPL、身份对话或业务目录，生成 CSV/JSON/Markdown，缺失、待复核、无标准 PPL 有效性证据的结果不会当作零分或合格分数。

按会话整理“第一轮输入/实际输出/参考答案、第二轮……”的错误表：

```bash
python "$VALIDATION/run.py" errors /path/to/generation-run \
  --reviews /path/to/reviews.json \
  --references /path/to/references.json \
  --output llama-output/model-validation/my-model/errors-v1
```

`references.json` 是 `{"I01T1": "参考答复", "I01T2": "参考答复"}` 这样的 ID→文本映射。两项都可省略：未提供复核的会话列为待复核，不冒充错误；未提供参考答案则标注缺失，不自动编造标准答案。报告包含 Markdown、CSV 和来源哈希，并链接原始完整回答。

## 7. 后续如何添加流程

配置格式、参数矩阵、依赖与新增工具的方法见 [扩展指南](EXTENDING.md)。

只增加已有工具的参数组合时，复制一个 `steps` 项即可；增加新的后端或任务类型时，在相应目录实现独立 CLI 并在 `common/registry.py` 注册。不要把某台设备、新模型版本号写死进调度器。

完整实验至少记录原始权重→PT2→PTE 的对应关系、量化配置和数据来源、tokenizer/template、模式与上下文/AR、生成策略、代码版本及复核口径。单独改图的诊断产物必须使用新的运行名称，不能覆盖原模型的报告。

## 8. 兼容、验证与注意事项

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

生产 masked-softmax 的因果性回归另见：

```bash
python -m pytest -q backends/qualcomm/tests/test_llama_masked_softmax.py
python -m pytest -q backends/qualcomm/tests/test_mha_to_sha_hadamard.py
```

它覆盖浮点/QDQ、CPU/CUDA、SWA、校准图到部署图的 encoding 传播，以及
AR4096 新导出 PT2 的原图因果检查。CUDA 不可用时跳过 GPU 项。
Hadamard 转换回归覆盖 AR1/4/8/128、1/2/4 个 KV 头及旋转结果直接作为图输出，
检查旋转保留批处理、attention 按头拆分且输出数值一致。
重新导出注意事项见 [masked-softmax 修复说明](../README.md#re-exporting-after-the-masked-softmax-causality-fix)；
旧 PT2 不会因更新测试脚本而自动修复。

设备集成测试必须显式按其文档配置运行，不应把历史长实验当成日常单元测试。不要在整理代码时自动重跑全量 PPL/100 条对话或上传新模型。

masked-softmax 的 QNN 真机回归（默认 Android 构建目录为 `build-android`）：

```bash
ET_MASK_QNN_DEVICE="$SERIAL" ET_MASK_QNN_SOC=SM8750 \
EXECUTORCH_ANDROID_BUILD="$PWD/build-android" \
python -m pytest -q backends/qualcomm/tests/test_llama_masked_softmax.py
```

此项会编译并推送隔离的小型 Lincal W4A16/SWA 模型，核对 AR1/AR128 的
因果性、PT2/PTE 输出、QNN MaskedSoftmax 融合及原生 Hadamard 保留，不覆盖已有模型。
需要主机 QNN SDK、已构建的 `qnn_executor_runner` 和 SDK 的 optrace 工具；
如果主机加载 QNN 时缺少 `libc++.so.1`，可把 NDK 的
`toolchains/llvm/prebuilt/linux-x86_64/lib/x86_64-unknown-linux-gnu`
加入当前进程的 `LD_LIBRARY_PATH`。

本机 SM8750 / QNN 2.49.40 曾在 R3 与 MHA→SHA 同时开启时出现
`q::ConvLayer_s1.opt` 编译失败。转换现在保留批处理的原生 Hadamard，
只拆分其输出，避免逐头复制旋转及上游投影；AR1/AR128 小模型已通过默认路径的
编译、真机数值与算子检查。`ET_MASK_QNN_MHA2SHA` 默认值为 `1`，
设置为 `0` 仅供关闭转换时的对照测试，不改变 `llama.py` 默认设置。
此回归不代表完整 1.7B 模型的 PPL 或生成质量已经验证。

内存紧张时，可先仅测试 AR1 decode 与 AR4 prefill：

```bash
ET_MASK_QNN_DEVICE="$SERIAL" ET_MASK_QNN_PREFILL_AR=4 \
python -m pytest -q backends/qualcomm/tests/test_llama_masked_softmax.py \
  -k 'deployment or exported_qdq'
```

`ET_MASK_QNN_PREFILL_AR` 仅修改此回归的部署 prefill AR 和对应 QDQ 检查，
默认是 `128`；测试用的两层小模型仍以 AR256 校准并传播 encoding 到 AR1/prefill。
上述筛选不会运行 AR4096 测试，也不会修改正式导出参数或重新量化已有模型。

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
