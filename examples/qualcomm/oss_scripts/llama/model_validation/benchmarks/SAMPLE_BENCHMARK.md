# 本地 benchmark 数据包测试

入口：`sample_benchmark.py`，复用自我认知工具的端侧部署、日志、续跑流程，
但使用独立的题目适配和评分。支持当前数据包的 140 条 benchmark 样本，以及
`repeat_test_3000/selected_samples.jsonl` 的 3000 条重复测试输入。
不会运行数据包的 `build_bundle.py`，不会改写下载目录。

## 直接运行 140 条样本

```bash
cd /hunter/cpp_code/ET/npu/executorch
conda activate et_npu
source env.sh

python examples/qualcomm/oss_scripts/llama/model_validation/benchmarks/sample_benchmark.py run \
  --bundle /home/hunter/下载/benchmark_sample_bundle_20260910 \
  --model ./llama-output/luxi-1_7b-m4096-hybrid-w4a16-cci-wiki-03 \
  --device 3B1F57E9B8L74X5D \
  --device-lib-dir /data/local/tmp/luxi-m4096-hybrid-test-20260910 \
  --mode hybrid \
  --seq-len 4096 \
  --output ./llama-output/bundle-eval/luxi-03-140
```

无需手工转换数据。自动使用模型目录里的 `tokenizer.json`，也可以传 `--tokenizer`。
首次运行自动从现有 `build-android` 构建目录编译独立的测试入口，输出到
`build-android/benchmark_bundle/`，不替换旧自我认知测试的二进制、正式 runner 或 AAR。
需要已经构建好的 CMake Unix Makefiles QNN runner；不是从零安装整个后端。
`--build-folder` 可选择其他基础构建目录，`--driver` 可指定已编译的协议 v2 测试入口。

模型自动上传到独立设备目录；运行库沿用 `--device-lib-dir`，必须与模型/SoC 匹配。
如已经上传模型，可传 `--device-model /data/local/tmp/.../model.pte`，校验 SHA256 后复用。
运行期间请勿同时在同一设备上启动其他模型测试。

## 输入和上下文处理

- 使用 `messages`，保留 HUMAN/BOT 的 few-shot 示例及 system/user/assistant 角色。
  每条样本只为最后一道题生成答案，示例中的 BOT 回答是输入，不重新生成。
- `model_response` 是旧 BF16 模型的回答，绝不用于新模型输入，也不复制为新结果。
  `reference` 只留在主机的评分快照中，不发送给设备测试入口。
- 各样本启动独立进程、清空 KV，避免题目之间串历史。
- 默认不关闭 thinking，温度 0，重复惩罚 1.0，遵循 EOS。
- 使用本机 ExecuTorch C++ tokenizer 做长度预检，默认预留 128 token 的生成空间。
  `prompt_tokens + reserve_tokens > seq_len` 的题目标成 `context_too_long`，不截断。
  `--reserve-tokens` 可配置预留空间，它不是生成上限。
- LongBench-v2 本包的输入远超过当前 4096 上下文，会单列跳过并保留在计划分母中。
  预检这些超长文本需要几分钟，终端会显示逐题进度，并不是端侧卡死。
- `--seq-len` 不能扩大 PTE 实际编译容量；设备上的实际输入长度/容量检查仍以 Runner 为准。
- 默认 `--max-new-tokens -1`，生成到 EOS 或上下文限制；可显式改成 512 等值。
  这会改变题目的思考空间，比较模型时应保持一致。

## 选题、重复测试与续跑

在上面命令末尾追加选项，且使用新的输出目录：

- `--benchmarks C-Eval GSM8K`：只跑指定数据集。
- `--ids C-Eval-01 ARC-Challenge-01`：只跑指定样本。
- `--limit 5`：按数据包原顺序取前 5 条，可先验证流程。
- `--dry-run`：只解析、检查 token 长度，不上传、不推理、不创建输出目录。
- `--suite repeat`：改为 3000 条重复测试输入；没有标准答案，不输出知识准确率。
  如需 512-token 上限，可再加 `--max-new-tokens 512`，但仍然保留 thinking。
- `--resume`：使用原命令、原输出目录续跑，完整样本跳过，中断样本的旧文件归档后重跑。
  模型、tokenizer、题集、配置、测试二进制或运行库变化时拒绝混用结果。

数据包中的旧重复测试关闭了 thinking，旧 benchmark 使用 BF16、temperature=0.95、
top_p=0.95、top_k=20；当前脚本不复现这套采样配置，不能直接把差异归因于量化质量。

## 输出与评分

输出目录包含 `REPORT.md`、`ANSWERS.md`、`results.csv`、`summary.json`、
`scored_results.json`、`cases.json`、`manifest.json`，以及 `sessions/` 下的完整输出
（含 thinking）、逐题原始 JSON、输入、日志、运行命令和电池状态。

| 题型 | 当前评分方式 |
|---|---|
| 选择题 | 从明确的 ANSWER/答案、boxed 或独立末行提取 A–P 选项 |
| AIME25/26、GSM8K | 从明确的最终答案中提取数值，使用精确有理数比较 |
| MATH-500 | 数值精确比较、规范化 LaTeX 匹配；未证实的符号等价留待人工复核 |
| IFEval | 调用已安装 lm_eval 的 strict/loose 检查器，主成绩使用 strict |
| 3000 条重复测试 | 生成状态、重复警报、非空最终回答之间的重复计数，不评知识正确率 |

这是本地样本检查，不是完整 OpenCompass 或官方榜单成绩。
`accuracy_on_scored` 只针对已评分子集，必须结合 `scored/planned` 看；
`confirmed_correct_over_planned` 是已确认正确数除以计划题数的下界。
超长、未运行、执行失败、无法自动解析、缺少评分依赖均单独保留，不能悄悄从分母中删除。
`normal_generation` 只判断生成是否正常结束、有最终答复、think 闭合、无明显重复/乱码，
不等于答案正确。重复警报沿用字符/句段启发式，不是数据包旧脚本的 token-loop 指标；
相同的短选项回答也不代表模型塌缩。

IFEval 依赖现有 lm_eval、langdetect、immutabledict 和 NLTK 的 punkt_tab 数据。
本机已经在 `et_npu` 补齐；其他机器如缺少附加依赖，可先执行
`python -m pip install langdetect immutabledict`，再执行
`python -m nltk.downloader punkt_tab`（NLTK 至少 3.9.1）。
脚本不会在评分时偷偷联网下载；依赖缺失时标为 `grader_unavailable`，不是答错。

不重新推理，仅重新统计：

```bash
python examples/qualcomm/oss_scripts/llama/model_validation/benchmarks/sample_benchmark.py summarize \
  ./llama-output/bundle-eval/luxi-03-140
```

如需人工纠正自动判定，将 `review_template.json` 复制为 `reviews.json`，
填写各条 `correct: true/false` 和 `note`，保留两个指纹字段，再运行：

```bash
python examples/qualcomm/oss_scripts/llama/model_validation/benchmarks/sample_benchmark.py summarize \
  ./llama-output/bundle-eval/luxi-03-140 \
  --reviews ./llama-output/bundle-eval/luxi-03-140/reviews.json
```

新工具结果请使用 `sample_benchmark.py summarize`，不要混用身份评估器的评分命令。
