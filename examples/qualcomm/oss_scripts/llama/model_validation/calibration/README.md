# 完整 think 校准数据准备

本流程准备 PTQ 输入，不进行权重训练、不重新导出 PTE、不改变端侧 prefill AR。使用 `/qualcomm` 的校准路径核对样本实际加载行为；保留原有原文校准模式，新增按来源选择的完整对话模式。

## 已完成的 v1

[数据及统计](../../assets/samples/luxi_think_calibration_v1/README.md)：286 组对话，316 个 assistant 回复，286,547 token。中文 R1 256 组，Luxi 自身 GPU 双轮对话 30 组；实际导出 tokenizer 与 HF tokenizer 的 token ID 已逐条核对一致。

公开部分来自 [Congliu/Chinese-DeepSeek-R1-Distill-data-110k](https://huggingface.co/datasets/Congliu/Chinese-DeepSeek-R1-Distill-data-110k)，固定版本 `8520b649430617c2be4490f424d251d09d835ed3`。该数据将 `input`、`reasoning_content`、`content` 分开，数据卡标注 Apache-2.0，并保留上游来源及模型评分。评分不是独立事实保证；使用和再分发前还需核对原始来源的许可与限制。

GPU 来源是本地 `luxi-1.7B-1.0.05.3`，不是从 PTE 反推的浮点权重。默认 BF16/SDPA、贪心生成、think 开启、EOS 生效，同组真实 KV 历史保留原始思考。已生成 56 组候选（112 次回答）；正则只是初筛，之后由 Codex 逐条复核思考和答案，并非人工审核。最终保留 30 组，不改写失败输出来凑数量。

## 复用步骤

在仓库根目录运行，使用 `et_npu` 并 `source env.sh`。所有输出路径必须是新目录或新文件，避免覆盖原始记录。

### 1. 筛选中文 R1

```bash
python examples/qualcomm/oss_scripts/llama/model_validation/calibration/build_think_calibration.py public \
  --tokenizer llama-output/luxi-1_7b-m4096-hybrid-w4a16-cci-wiki-06 \
  --output llama-output/think-calibration/public-new
```

默认流式读取固定版本的前 8000 条，缓存 `source-prefix.jsonl`，不下载整个仓库。以 Luxi tokenizer 计数：输入最多 256 token，最终回答最多 768 token，完整对话最多 4096 token；上游评分至少 9，剔除空字段、特殊标记冲突、乱码、明显复读、外部模型引用和重复题目。至少 60% 的最终公开样本来自通用子集；v1 实际为 179/256。

默认 `--bucket-counts 0 128 112 16` 对应思考长度 32–127、128–511、512–1535、1536–2500 token。扫描的 8000 条中，短于 128 的合格样本只有 1 条，因此 v1 不强行从 R1 填充短思考，短思考由 Luxi GPU 候选补充。样本不足时明确失败，不重复样本、不截短补数。`--max-scanned`、`--min-score` 和四档数量均可调整。

已下载过的前缀可离线重筛：

```bash
python examples/qualcomm/oss_scripts/llama/model_validation/calibration/build_think_calibration.py public \
  --tokenizer llama-output/luxi-1_7b-m4096-hybrid-w4a16-cci-wiki-06 \
  --source-jsonl llama-output/think-calibration/public-r1-v1-20260911/source-prefix.jsonl \
  --output llama-output/think-calibration/public-replayed
```

仅将该版本下载得到的真实前缀用于此回放参数；输入哈希会记录在 manifest。直接使用已经生成的校准 JSONL 不需要重新下载源数据。

### 2. 用本地 Luxi 生成独立多轮候选

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
python examples/qualcomm/oss_scripts/llama/model_validation/benchmarks/hf_benchmark.py \
  --model /hunter/model/Luxi/luxi-1.7B-1.0.05.3 \
  --device cuda:0 --dtype bfloat16 --seq-len 4096 \
  --cases examples/qualcomm/oss_scripts/llama/model_validation/calibration/luxi_multiturn_prompts.json \
  --output llama-output/think-calibration/gpu-new
```

需要代码类补充时，将 cases 换成同目录的 `luxi_multiturn_code_prompts.json`，并使用另一个输出目录。支持原工具的 `--resume`。不要删除生成过程中的 think，不要关闭 EOS。来源记录包含模型配置、safetensors 哈希、生成 token ID 和 KV 历史状态。

这批问题专用于校准候选，不用于报告模型独立准确率。默认排除 `model_validation/benchmarks/cases.json` 中规范化后完全相同的问题；这不保证语义层面完全不重叠，也没有对所有第三方测试集做去重。

### 3. 查看候选并明确复核

```bash
python examples/qualcomm/oss_scripts/llama/model_validation/calibration/build_think_calibration.py inspect-gpu \
  --tokenizer llama-output/luxi-1_7b-m4096-hybrid-w4a16-cci-wiki-06 \
  --gpu-run llama-output/think-calibration/gpu-new \
  --output llama-output/think-calibration/gpu-new-review-candidates.json
```

`--gpu-run` 可同时接收多个运行目录，session ID 必须互不重复。候选快照保存每组的原始消息、正式答复、检查结果和记录哈希。复核时检查完整思考和答案，而非只看关键字：v1 中已排除最终 1400 正确但 think 写成 `1200 × 1.5 = 1400`、答案含目标名称但同时否认自身名称等样本。

复核文件结构参考数据包的 `gpu_reviews.json`：顶层 `run_fingerprints` 对应各 GPU manifest 的 identity 摘要，`reviews` 按 session ID 保存 `accept`、`record_hashes`、`note`。换模型或修改原始记录后，旧复核不能复用；合并会拒绝指纹/哈希不匹配及结构失败。

### 4. 合并

以下路径对应本次保存的已完成运行；如重新生成请换成新路径及新复核文件。

```bash
python examples/qualcomm/oss_scripts/llama/model_validation/calibration/build_think_calibration.py merge \
  --tokenizer llama-output/luxi-1_7b-m4096-hybrid-w4a16-cci-wiki-06 \
  --public llama-output/think-calibration/public-r1-balanced-v1-20260911 \
  --gpu-run \
    llama-output/think-calibration/luxi-gpu-candidates-v1-20260911 \
    llama-output/think-calibration/luxi-gpu-code-candidates-v1-20260911 \
  --gpu-reviews llama-output/think-calibration/gpu-reviews-v1.json \
  --gpu-count 30 \
  --output llama-output/think-calibration/merged-new
```

合并产物包含 `messages.json`、`calibration.chatml.jsonl`、逐条来源、GPU 复核记录、来源 manifest、统计及文件哈希。不会把本轮的问题附加 system 身份提示，也不会修补缺失结束标签的 GPU 输出。公开集由独立推理/答案字段组装标签，GPU 样本保留生成原文。

### 5. 验证实际加载

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
python examples/qualcomm/oss_scripts/llama/model_validation/calibration/validate_think_calibration.py \
  --data-dir examples/qualcomm/oss_scripts/llama/assets/samples/luxi_think_calibration_v1 \
  --tokenizer llama-output/luxi-1_7b-m4096-hybrid-w4a16-cci-wiki-06 \
  --max-context-len 4096
```

校验所有原始文件哈希、消息/ChatML 一致性、两种 tokenizer 的完整 token ID、实际加载器的序列多重集合，以及 think/EOS token 计数。另对首条、最长样本及双轮样本检查校准 collator 的填充结果；不运行模型前向，不进行全量量化。

## 导出时使用

使用新 JSONL 替换旧 `luxi_mixed_think_20`、`luxi_think_20`、`luxi_identity_clean` 补充项，不要同时重复加入。CCI/WikiText 可以继续保留。详见数据包 README 的参数片段。

`--calib_text_preserve_sources` 是本次新增参数，服务器需要同步 `llama.py` 和 `dataset/{config,builders,loaders}.py`。仅复制数据、仍运行旧脚本会报不认识该参数。

## 限制

校准不是 SFT；增大数据量不保证修复 `</think>`。尚未用此数据重新校准或导出 PTE，也未建立精度改善结论。历史校准曾包含部分身份测试题，因此旧测试集主要用于回归比较；新的泛化评价应采用未参与筛选/校准的独立问题。
