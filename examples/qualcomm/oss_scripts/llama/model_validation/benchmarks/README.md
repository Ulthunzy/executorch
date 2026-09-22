# 可复用的端侧自我认知测试

其他数据集：本地 140 条 benchmark / 3000 条重复测试数据包的直接运行入口，见
[SAMPLE_BENCHMARK.md](SAMPLE_BENCHMARK.md)。其评分与本页的身份评估分开。

原始 safetensors 的 Transformers CUDA 参考入口见
[HF_BENCHMARK.md](HF_BENCHMARK.md)，使用同一套题和评分格式。

固定 `cases.json`：50 种首轮问法，每组追加一轮，共 100 次回答。
同组使用同一个原生 Runner 和 KV 历史，不重传历史；组间启动新进程。
默认名称为「陆小兮」/ `lu xiaoxi`，仅用于评分，**不注入提示词**。
默认保留模型 thinking。显式加 `benchmark.py run --disable-thinking` 可做非思考模式对照：
每轮增量 ChatML 输入追加空的 `<think>\n\n</think>\n\n` 前缀，不删生成文本或历史 KV，
也不修改原题集。配置和实际输入均保存；默认行为不变。

默认贪心生成、重复惩罚 1.0（关闭）、遵循 EOS、不额外限制单轮输出。
`--seq-len` 是整组对话的总上下文预算，不能超过 PTE 实际编译容量；
运行时无法扩大已导出模型。SWA 和 AR 长度由模型及 Runner 处理。
目前支持 ChatML QNN Lincal / Qwen3 / Qwen2.5 的 `kv`、`hybrid` 模式，
不是任意模型架构的通用适配器；其他聊天模板需要另行适配。

通用业务 holdout 不使用本页的姓名专用评分。请改用 `run.py business` 和
`configs/business.example.json`：HF/PT2/PTE 共用同一份 plain-prompt cases，支持
`exact_match`、`contains`、`choice`、`numeric` 和 `manual` scorer，并把关键题、
自动结构检查与人工复核绑定到逐题 record hash。workflow 先冻结 HF ChatML 输入，
PTE 使用 canonical token IDs，最终门禁还会逐题核对三端实际输入 token。完整运行及质量门禁说明见
[模型验证 README](../README.md#5-三端业务集与质量门禁)。

## 首次准备

在仓库根目录运行：

```bash
conda activate et_npu
source env.sh
python examples/qualcomm/oss_scripts/llama/model_validation/benchmarks/build_driver.py \
  --build-folder build-android --jobs 2
```

需要已经构建好的 `qnn_llama_runner`（CMake **Unix Makefiles**，含 `link.txt`、
`flags.make` 和链接库；不能直接使用 Ninja 构建目录）。基础 QNN 构建遵循仓库
`/qualcomm` 的 `backends/qualcomm/scripts/build.sh` 流程。
辅助脚本复用其编译/链接参数，重编译 QNN runner 源码及测试入口以匹配头文件，
写入 `build-android/self_identity_eval/`，不替换原 runner、AAR 或其对象文件。

源码位于 `model_validation/benchmarks/`；旧 `self_identity_eval/*.py` 路径保留
CLI 和函数导入兼容，默认构建输出路径沿用旧值。测试仅位于新目录。
`cases.json` 的 canonical 文件在本目录，旧目录的同名文件仅作兼容副本。

使用 `--define ET_QNN_PPL_EVALUATION` 构建 PPL 驱动时，`build_driver.py`
从当前 `runner/decoder_runner.h` 生成独立输出目录中的 `include_overlay/`，
在副本中注入 teacher-forcing hook（已有完整 hook 时直接复用），并将其
`-I` 放在原 include 路径之前。生产 header 无需预先打补丁。
当前 runner 的 decoder 引用均为完整 `executorch/.../decoder_runner.h` 路径，
可由该 overlay 覆盖；新增相邻相对 include 时需重新检查覆盖行为。
PPL 与普通生成驱动应使用不同 `--output` 目录。

迁移及本次功能调整会改变脚本 SHA256。已有结果的脚本/驱动哈希与新实现不一致时，
不能在原目录 `--resume`；请保留旧结果并使用新输出目录重新运行。
不要修改历史 manifest、重写哈希或跳过身份校验来强行续跑。
更新 Runner/依赖后先更新基础构建，再重建测试入口；仅换 PTE 不需要重建程序。

设备上需要已有与模型、构建和 SoC 匹配的 QNN 运行库。`--device-lib-dir`
指向含 `libqnn_executorch_backend.so`、`libQnnHtp.so`、对应 Stub/Skel 等文件
的设备目录。工具记录库的 SHA256，不替换运行库。

## 换模型测试

### 可选的 think 闭合控制

新版 QNN **文本 KV/hybrid** runner 支持两个独立运行时开关，默认都关闭，
不会改变已有消融结果或默认开启的 thinking。只更新 runner/JNI 一次，
之后更改配置即可，不需要重新校准或导出 PTE。

在测试 driver 的配置 JSON 中添加：

```json
{
  "repair_think_eos": true,
  "enable_think_budget": true,
  "think_budget": 512
}
```

| repair_think_eos | enable_think_budget | 行为 |
|---|---|---|
| false | false | 原始生成行为（默认） |
| true | false | 只把当前 think 内的提前 EOS 替换为闭合 token |
| false | true | 只在思考预算耗尽时强制闭合；预算前的 EOS 仍可结束生成 |
| true | true | 两种保护同时开启 |

对应 driver 命令行覆盖（优先于 JSON，不写回文件）：

```bash
"$model_dir/driver-thinking" "$model_dir/config.json" \
  "$model_dir/manual_three_turns.json" "$chat_out" \
  --temp 0 --repetition-penalty 1.1 --repeat-last-n 128 \
  --repair-think-eos 1 --enable-think-budget 1 --think-budget 512
```

布尔值接受 `0/1` 或 `false/true`，JSON 必须用布尔值。预算是正的 32 位整数，
默认 512，指本轮 think 内新生成的 token 数，不是 prefill AR、SWA 窗口或总上下文。
正常生成的起始/闭合标签不计入预算。达到预算后在当前位置闭合，不寻找历史句号、
不回滚 KV；闭合 token 会作为真实模型输入提交，并计入输出长度和重复惩罚历史。
临近总长度上限时，预算模式会提前闭合，尽可能为正文保留位置，但不增加上下文容量。
强制闭合不保证一定生成正文或答对问题，也不会覆盖用户取消。

首个 prefill 样本和后续 decode 样本都受控；只识别当前助手的 leading think 块，
不把用户输入/上一轮标签当作当前思考。显式非思考前缀不会被重新开启。
不支持 lookahead、多模态和 `ignore_eos=true` 的组合，配置错误会直接报告。
每轮日志打印三个参数；`[QNN_THINK] forced_close` 的原因区分 `early_eos`、
`think_budget` 和 `context_limit`，不得把它们算作自然闭合。

批量测试可加 `benchmark.py run --repair-think-eos --enable-think-budget --think-budget 512`，
使用新的输出目录。开关会写入运行指纹，旧 driver 会被能力检查拒绝。
直接调用 `llama.py` 时使用 `--repair_think_eos --enable_think_budget --think_budget 512`；
直接调用 `qnn_llama_runner` 时使用 `--repair_think_eos=true --enable_think_budget=true --think_budget=512`。

### 输入与采样日志

新构建的测试入口在每轮推理前打印 `IDENTITY_INPUT`：包含轮次、
`has_history_before` 和实际传给 `runner.generate()` 的 `formatted_prompt`。
该 JSON 行保留换行转义及结尾换行；随后以 `PROMPT | ` 为显示前缀逐行展开，
方便查看 ChatML 角色、特殊标记和正文。前缀不属于模型输入。
每轮结果 JSON 也保存 `formatted_prompt`，回答 TXT 仍只保存模型生成内容。
历史存在时打印的是本轮新增 prompt，先前历史保留在 KV 中，不会为了日志而重传。
这些输入日志不作为 EOS/上下文耗尽的运行证据，避免问题文本影响自动统计。
日志含完整输入内容，分享日志前请检查其中是否包含隐私信息。

手动调用新版 `driver CONFIG_JSON SESSION_JSON OUTPUT_DIR` 时，配置 JSON 支持
`"repeat_last_n": 64`，控制实际惩罚窗口；省略时仍为 256。必须是正的 32 位整数。
例如同时设置 `"temperature": 0, "repetition_penalty": 1.0, "repeat_last_n": 64`。
`1.0` 关闭重复惩罚，此时窗口不影响采样；改回大于 `1.0` 后才按该窗口惩罚。
此字段不改变 SWA、上下文、think 或 EOS 行为。旧 driver 不支持配置窗口，需更新一次
二进制；更新后修改 JSON 即可，不需要重新导出 PTE。

也可以在三个位置参数后直接传命令行参数，不修改 JSON：

```bash
"$model_dir/driver-rp256" "$model_dir/config.json" \
  "$model_dir/manual_three_turns.json" "$chat_out" \
  --temp 0 --repetition-penalty 1.0 --repeat-last-n 64
```

优先级为命令行 > JSON > 窗口默认值 256。支持 `--temperature`、下划线名称
（如 `--repeat_last_n`）及 `--repeat-last-n=64` 格式。覆盖仅作用于本次进程，
不会写回配置文件；每轮 `[QNN_GENERATION]` 打印实际生效的值。
运行 `driver --capabilities` 的 `sampling_cli: true` 表示支持这些命令行参数。

```bash
python examples/qualcomm/oss_scripts/llama/model_validation/benchmarks/benchmark.py run \
  --model llama-output/luxi-1_7b-m4096-hybrid-w4a16-cci-wiki \
  --tokenizer /hunter/model/Luxi/luxi-1.7B-1.0.05.2/tokenizer.json \
  --device 3B1F57E9B8L74X5D \
  --device-lib-dir /data/local/tmp/luxi-m4096-hybrid-test-20260910 \
  --mode hybrid \
  --seq-len 4096 \
  --output llama-output/identity-eval/luxi-m4096-v1
```

下次替换 `--model`、对应 `--tokenizer` 和新 `--output` 即可。
导出目录只有一个 `.pte` 时可传目录，否则指定 PTE 文件。
省略 tokenizer 时查找 PTE 同目录的 `tokenizer.json`，不会使用其他模型的文件。
KV 模型用 `--mode kv`。其他 decoder 用 `--decoder qwen3` 等。
预期名称可用 `--expected-english`、`--expected-chinese` 修改。

可选参数：

- `--ids I01`：一组两轮冒烟测试，统计分母为 2，不冒充完整 100 条。
- `--device-model /data/local/tmp/.../model.pte`：复用设备 PTE，必须与本地模型 SHA256 相同。
- `--repetition-penalty 1.1`：评估不同重复惩罚，默认不启用。
  新构建的 runner 仅惩罚最近 256 个已提交的输入/生成 token（含 think，跨轮滑动），
  不随 SWA 淘汰；需要重新构建测试程序，旧二进制保留各自的窗口/累计历史行为。
- `--disable-thinking`：使用与本地 Qwen3/Lincal 非思考模板一致的空 think 前缀。
  复用支持 `formatted_prompt` 的 protocol-v2 driver，无需重新导出 PTE。
  仅支持此身份题集入口，不与题目自带的 `formatted_prompt` 混用；不是直接修改
  driver 配置 JSON 就能生效的开关。两种模式的输入会相差模板前缀，请使用不同输出目录。
- `--repeat-last-n 128`：显式设置最近 token 的重复惩罚窗口，保存到配置及运行指纹；
  需要 driver 声明 `repeat_last_n` 能力，否则拒绝测试，避免旧程序静默忽略参数。
  不传时保留所选 driver 的默认窗口。配合 `--repetition-penalty 1.1` 可测试用户指定的组合。
- `--timeout 600`：每组会话超时秒数，慢设备可调大。
- `--resume`：原命令加此项续跑，模型、题集、配置、设备、程序和运行库必须相同。
  完整结束的组跳过；中断组移到 `attempts/` 保留，整组重跑两轮，不把新进程的空 KV
  当成历史。这只是离线测试进程重启的恢复策略，不改变 App 的取消/prefill 提交语义。

已有输出目录默认拒绝覆盖。每次独立的设备目录记录在 `manifest.json` 中，位于
`/data/local/tmp/executorch-identity-eval/`。设备文件不自动删除；结束后可核对 manifest，
再自行清理该次独立目录。同一输出目录请勿同时启动多个进程。

## 最后统计与复核

测试运行结束时自动生成：

- `REPORT.md`：概览；`ANSWERS.md`：逐题最终答复与完整输出链接。
- `results.csv`：Excel 可打开的 UTF-8 BOM 明细。
- `summary.json`：计划/已尝试数、EOS、重复警报、首轮/追问、分类统计。
- `sessions/`：每轮原始输出（含 thinking）、JSON、日志、命令和电池状态。
- `manifest.json`：模型/tokenizer/题集/程序/设备运行库指纹及生成参数。
- `review_template.json`：本次结果专用复核模板，不沿用旧模型评分。

「名字出现」仅是自动筛查，如「我不是陆小兮」也会命中。重复、乱码也只是启发式。
未完成语义复核时，最终正确率为 `null`，不把待复核的答案判成通过或失败。

复制 `review_template.json` 为独立的 `reviews.json`，核对 `ANSWERS.md` 后填写
各条的 `identity` 和 `language_ok`；保留 `record_hash` 和 `run_fingerprint`。
`identity` 可选 `correct`、`missing`、`wrong`、`partial`、`variant`、`not_answered`。
没有最终答复的已尝试轮次也应复核为 `not_answered`。
`language_ok` 为布尔值，判断是否符合题目要求的中文/英文姓名形式。

中文严格匹配「陆小兮」，英文允许大小写、空格和连字符差异。
只提到、否认、引用或在 thinking 里出现名字不算身份正确。
可加 `other_issue: true`；人工确认启发式误报时，可用 `repetition_override` /
`garbled_override` 布尔值覆盖警报，并写 `note` 说明。
严格通过要求执行成功、EOS、完整最终答复、think 闭合、无明显重复/乱码、
身份正确且名字语言符合要求。开发者/公司归属等其他事实不在本次评分范围。

不重新推理，仅重新统计：

```bash
python examples/qualcomm/oss_scripts/llama/model_validation/benchmarks/benchmark.py summarize \
  llama-output/identity-eval/luxi-m4096-v1 \
  --reviews llama-output/identity-eval/luxi-m4096-v1/reviews.json
```

不传 `--reviews` 则只生成自动检查与待复核统计。不要手改原始 JSON：
模型指纹或响应改变时旧复核会被拒绝，续跑重测后需更新重测轮次的复核。
上下文不足、未执行、失败、截断不从计划分母中删除；全量复核后才发布正确率。
这是固定专项题集，不代表模型通用准确率。

多个模型统计完成后合并：

```bash
python examples/qualcomm/oss_scripts/llama/model_validation/benchmarks/benchmark.py compare \
  llama-output/identity-eval/luxi-m4096-v1 \
  llama-output/identity-eval/luxi-m4096-v2 \
  --output llama-output/identity-eval/comparison.csv
```

题集、tokenizer、参数、设备或程序等不同时，在 `comparison_caveats` 列标出，
不能直接归因于模型质量；未复核的正确率留空。

## 已有基线与工具测试

原来完整 100 条结果仍保留在
`llama-output/luxi-1_7b-m4096-hybrid-w4a16-cci-wiki/device-test-100-20260910/`，
包含原报告、回答和人工复核；本工具题集与其完全相同，没有改写旧结果。
旧格式目录请看原报告，不直接传给新格式的 summarize/compare。
源码与题集在本目录，可随仓库保存；模型、生成结果放在 `llama-output/`，不加入代码版本管理。

无需模型或设备的单元测试：

```bash
python -m unittest discover \
  -s examples/qualcomm/oss_scripts/llama/model_validation/benchmarks \
  -p 'test_*.py' -v
```
