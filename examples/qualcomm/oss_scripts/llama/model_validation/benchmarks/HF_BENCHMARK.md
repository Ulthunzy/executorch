# Transformers GPU 自我认知参考测试

同架构新模型可传 `--expected-english "New Assistant" --expected-chinese "新助手"`。
默认仍为 `lu xiaoxi` / `陆小兮`；这些参数只用于评分和结果身份记录，不注入提示词。
温度仍固定为 0，架构仍限于已验证的 Qwen3/Lincal 路径。

迁移及姓名参数功能会改变 `driver_hash`。旧结果与新脚本哈希不匹配时不能在原目录
`--resume`；保留原始结果，选择新输出目录，不修改历史哈希或绕过校验。

`hf_benchmark.py` 用本地 Qwen3/Lincal safetensors 运行同一套 50 组两轮、100 条自我认知测试，不重新量化或导出 PTE。目前验证环境为 `et_npu`、PyTorch 2.9.1+cu128、Transformers 5.8.1；Transformers 4.x 不保证兼容此模型的 `rope_parameters` 配置。

```bash
conda activate et_npu
source env.sh

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python examples/qualcomm/oss_scripts/llama/model_validation/benchmarks/hf_benchmark.py \
  --model /hunter/model/Luxi/luxi-1.7B-1.0.05.3 \
  --device cuda:0 \
  --dtype bfloat16 \
  --seq-len 4096 \
  --output llama-output/identity-eval/luxi-05_3-hf-bf16
```

默认 BF16 权重、SDPA、贪心生成（逻辑温度 0）、重复惩罚 1.0、think 开启。明确覆盖 checkpoint 中的随机采样配置，生成时启用 KV cache；不修改原始 `config.json`。沿用模型声明的混合 full/sliding attention、窗口和 RoPE。支持 `--attention eager`、`--dtype float16/float32`，配置变化需使用新输出目录。

加 `--disable-thinking` 可使用模型自带的 `enable_thinking=False` 模板进行非思考模式对照测试。每轮新增输入包含空的 `<think>\n\n</think>\n\n` 前缀；不删除生成文本，也不清除上一轮 KV。开关写入 manifest 并参与续跑校验，原始 prompt 和 token ID 均保留。默认行为不变；开启与关闭 think 的输入 token ID 会因该前缀而不同。

4096 是整组会话共享的上下文预算，而不是每一轮各 4096。EOS 使用与端侧一致的 `<|im_end|>` 和 `<|endoftext|>`；没有额外单轮输出上限。首轮触顶后，追问记为 `context_full`，仍保留在计划分母中。

## 多轮与复现

使用 Transformers `generate` 和 [DynamicCache](https://huggingface.co/docs/transformers/kv_cache)。同组保持真实 KV，不清除首轮 thinking，不将回答解码后重新分词，不在每轮重新套完整历史模板。每次只编码新增 ChatML 用户消息；`generate` 接收完整 token ID 前缀和旧 cache，通过 cache 长度仅处理未缓存的 token（包括上一轮最后采样但尚未前向的 EOS）。组间重置 cache。小模型测试验证了滑窗越界后的续接 token 与完整前缀重算一致，且第二轮没有重新前向完整历史。

输出格式兼容现有 `benchmark.py summarize`：

- `sessions/` 保存每条原文、完整生成 ID、增量输入 ID、cache 长度、停止原因和计时。
- `manifest.json` 保存 safetensors、模型配置、tokenizer、脚本与依赖版本指纹。
- `model_loading.json` 保存加载检查、实际 dtype、滑窗层和 RoPE 参数。
- `REPORT.md`、`ANSWERS.md`、`results.csv`、`summary.json` 和复核模板与端侧共用。没有语义复核时，不发布最终正确率。

加 `--ids I01` 可跑一组两轮冒烟；加 `--resume` 可续跑同配置，已完成会话校验后跳过，中断会话移入 `attempts/` 后整组重跑。不要向此脚本传入已有端侧输出目录。更换模型只需改 `--model` 和新 `--output`。

```bash
python examples/qualcomm/oss_scripts/llama/model_validation/benchmarks/benchmark.py summarize \
  llama-output/identity-eval/luxi-05_3-hf-bf16 \
  --reviews llama-output/identity-eval/luxi-05_3-hf-bf16/reviews.json
```

## 对照边界

GPU 使用 HF tokenizer；端侧使用 C++ tokenizer，即使 tokenizer 文件一致，仍需核对输入 ID。BF16/SDPA 与 W4A16/QNN 的差异同时包含权重量化、数值精度、算子实现和运行路径，不能仅凭最终回答差异确定是哪一层或哪一种校准策略有问题。此入口不运行 QDQ，也不是中间层精度比较器。

GPU 模型一次加载、跨组复用权重；端侧原测试每组新进程加载模型。GPU 计时排除加载，decode 速度排除首 token；两边环境和计时实现不同，不作为受控硬件速度对比。所有 token 数包含实际生成的 thinking 和 EOS，正式答复单独提取用于语义评分。
