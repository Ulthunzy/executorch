# 固定语料标准 PPL

所有引擎评估 `prepare` 冻结的同一份 token IDs。没有聊天模板、BOS/EOS 注入、采样、重复惩罚、think 修复或 think budget；不使用旧实验 manifest，也不依赖模型版本名称或本机模型目录。

## 完整调用

下面从仓库根目录执行。`MV` 是相对路径变量，模型、tokenizer、评测 TXT 和结果目录由调用者提供。已有 Python 环境需要 `torch`、`numpy`、`transformers`、`tokenizers`；全部模型和 tokenizer 加载均使用本地文件，禁用 remote code。

```bash
MV=examples/qualcomm/oss_scripts/llama/model_validation
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

python "$MV/run.py" ppl prepare \
  --tokenizer /path/to/local-tokenizer-directory \
  --text /path/to/fixed-evaluation.txt \
  --context-len 2048 --stride 1024 --output /path/to/corpus

python "$MV/run.py" ppl run \
  --engine hf --corpus /path/to/corpus \
  --model /path/to/local-hf-model --tokenizer /path/to/local-tokenizer-directory \
  --device cuda:0 --dtype float32 --output /path/to/runs/hf

python "$MV/run.py" ppl run \
  --engine pt2 --corpus /path/to/corpus --model /path/to/decode_qdq.pt2 \
  --tokenizer /path/to/local-tokenizer-directory \
  --pt2-layout lincal-full-sequence --device cuda:0 --dtype float32 \
  --output /path/to/runs/pt2
```

直接调用 `python "$MV/ppl/evaluate.py" prepare/run ...` 使用完全相同的参数。统一 `run.py` 原样转发参数。HF/PT2 均可用 `--device cpu` 运行小型检查；HF 也支持 `--dtype bfloat16`，不同精度会形成不同 run identity。

更换模型时，为 `--model` 提供新路径，使用新的 `--output`。同一 corpus 只适用于词表 token→ID 映射一致的模型；更换 tokenizer 映射后需重新 `prepare`，所得 PPL 不能与原 tokenizer 的 PPL 直接排名。

### HF 配置

默认使用 `AutoConfig` / `AutoModelForCausalLM` 的本地模型配置，不强制任何架构。缺失、多余、不匹配的权重或加载错误均拒绝；不允许随机补齐权重后输出结果。输入 embedding 容量、配置中的位置容量以及输出词表会被检查。

如果本地模型类型未注册，默认明确失败。确需指定兼容架构时，由调用者提供 `--config-override /path/to/override.json`，例如 `{"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}`。该选项只适用于 HF，完整覆盖内容记入 identity，最终配置写入 `engine.json`，不修改磁盘模型。这要求调用者确认权重语义兼容；覆盖不是默认行为。

## PTE：构建、离线计划与实际运行

新 [ppl_driver.cpp](ppl_driver.cpp) 提供独立参数化的 teacher-forcing / NLL 逻辑。历史 driver 不适用于此入口。

已有 QNN runner 的 Android CMake **Unix Makefiles** 构建后，编译独立 driver：

```bash
python "$MV/run.py" build-driver \
  --build-folder build-android \
  --output build-android/model_validation/ppl \
  --entry-point "$MV/ppl/ppl_driver.cpp" \
  --define ET_QNN_PPL_EVALUATION --jobs 2
```

新 driver 需与同目录 `driver_source.sha256` 一起保留；入口检查源码指纹和二进制 hash，旧生成 driver 或旧历史 PPL driver 会被拒绝。构建复用 canonical `benchmarks/build_driver.py`，不替换应用 AAR。

先生成本地计划（**不调用任何 ADB，不推送或运行设备模型**）：

```bash
python "$MV/run.py" ppl run \
  --engine pte --corpus /path/to/corpus --model /path/to/model.pte \
  --tokenizer /path/to/tokenizer.json \
  --serial YOUR_ADB_SERIAL --device-lib-dir /data/local/tmp/qnn-libs \
  --driver build-android/model_validation/ppl/identity_eval_driver \
  --decoder lincal --mode hybrid --dry-run --output /path/to/plans/pte
```

`plan.json` 包含上传列表、driver config 和完整 ADB argv。这里直接调用 `ppl run --dry-run` **会写本地计划文件**；统一入口的 `workflow --dry-run` 则只展示命令，**完全不写文件**。PPL dry-run 的 `summary.json` 标记 `complete: false`、`standard_ppl_eligible: false`，不是评测结果。实际运行使用另一个输出目录，省略 `--dry-run`：

```bash
python "$MV/run.py" ppl run \
  --engine pte --corpus /path/to/corpus --model /path/to/model.pte \
  --tokenizer /path/to/tokenizer.json \
  --serial YOUR_ADB_SERIAL --device-lib-dir /data/local/tmp/qnn-libs \
  --driver build-android/model_validation/ppl/identity_eval_driver \
  --decoder lincal --mode hybrid --output /path/to/runs/pte
```

可加 `--device-model /absolute/device/model.pte` 复用设备已有模型；必须与本地 `--model` SHA256 一致，内容不符直接拒绝。未指定时上传到独立 attempt 目录。实际运行会上传 driver、tokens、tokenizer、config，验证各文件 hash，并核对设备构建标识及库目录全部文件的 hash。设备库需预先就绪，入口不安装 SDK/库。设备 attempt 保留以便排查，不自动删除。

PTE 支持边界：

- `--decoder` 必须显式选择 `lincal|qwen3|qwen2_5|llama3|llama2`，它决定当前 QNN Runner 的 tokenizer/cache 约定。模型必须按相应 runner ABI 导出；这不是任意 ExecuTorch PTE 的通用执行器。
- `--mode hybrid` 要求 `prefill_forward` 与 AR1 `kv_forward`；`--mode kv` 只使用 AR1 `kv_forward`。不支持 lookahead、无 KV 的其他方法命名或不同输出结构。
- 从方法 metadata 的 tokens、全局 attention mask、logits shape 读取 AR、context 和 vocab；要求 batch=1、prompt/decode context 与 logits dtype/vocab 匹配。`seq_len` 取 corpus 的 `context_len`，必须不超过编译 context，不能静默 clamp。无固定 AR/context 数字。
- prompt logits 可为全 context prefill；否则长窗口的 prefill 必须能按 AR 对齐，余下输入用固定参考 token 做 AR1 decode。无合法切分时拒绝。
- 支持 float32 logits 或带有效导出 scale/zero-point 的 uint16 logits；不修改导出 dtype。PTE 不支持 `--dtype bfloat16`。
- 检查完整词表的最大 token ID、逐次 teacher-forcing cursor、logits 字节数和目标对齐。采用原始 logits 计分，不设置 thinking controller。
- 已做 CPU 契约测试和 Android 编译；未用真实设备/模型验证不同 decoder、mode 或 context。支持范围是严格校验的接口范围，不能视作所有这些模型都已实测。

## PT2 适配器与支持边界

共享类位于 [engines/pt2.py](../engines/pt2.py)：

```python
from model_validation.engines.pt2 import PT2Engine

engine = PT2Engine(
    "/path/to/decode_qdq.pt2",
    device="cuda:0", dtype="float32",
    layout="lincal-full-sequence", pad_token_id=0,
)
logits = engine([1, 2, 3])  # [3, vocab]，与模型相同 device
capacity = engine.max_input_length
specs = engine.input_specs
check = engine.last_causality_check
check = engine.validate_causality([1, 2, 3])  # 检查失败抛 ValueError
```

调用方需将 `llama/` 加入 `PYTHONPATH` 才能使用示例的 `model_validation` import；已有生成入口通过 `engines.pt2` 导入同一类也可。所有构造选项均为 keyword-only。

唯一适配布局 `lincal-full-sequence`（CLI 必须显式指定）要求导出保存 flat positional tensor example inputs，顺序为 tokens `[1,S]`、causal mask `[1,1,S,S]`、sliding mask `[1,1,S,S+W-1]`、positions `[1,S]`、所有 K cache、所有 V cache。cache 成对匹配，长度只能是 0 或 `W-1`。`S`、`W`、cache shape、token dtype 全部从导出推导。其他布局、嵌套输入、AR1 导出、动态 shape、非 float32 浮点状态或带输入/buffer mutation 的图不支持。已知 mask 输入语义为允许位置 0、屏蔽位置 -255，position 从 0 开始，cache 清零。

**每次调用**均在原始固定 shape 上执行基准，再在 prefix 的开头、中点、末尾切分，使用全 1 和交替 0/1 两种 masked-future 填充，对切分之前所有 logits 做 bit-exact 比较。即使输入填满 S，也检查内部未来依赖。shape、非有限值或任何因果性比较失败立即拒绝；不做 mask 修补、常量折叠、动态 shape 改写、固定数量权重断言或来源权重替换。

最多需要 7 次 full-sequence forward/窗口（或生成步），速度和显存需求较高；不代表增量 KV 性能。检查是对当前输入和有限填充值的数值验证，不是任意输入下因果性的数学证明。需要可加载此导出的 PyTorch/QDQ operator 环境；未知 custom ops 不能自动适配。PT2 不检查与某个 HF 权重副本逐张量等价，以 PT2 文件 hash 标识实际评测模型。

## Corpus 与词表约束

`prepare` 原样读取 UTF-8 TXT 的字节，包括换行，不下载数据集、不拼接行、不规范化文本；按 `add_special_tokens=False` tokenize 一次。固定输出为 `text.txt`、little-endian uint64 的 `tokens.u64`、`windows.json`、可移动的 `tokenizer/` 快照和版本化 `manifest.json`。

准备过程在最终目录的同父目录内使用 `.名字.prepare-*` staging，写入并校验全部文件后原子 rename 发布。失败或中断留下 staging 供诊断，最终目录不出现半成品，可用原命令重试；已发布的非空 corpus 不会被覆盖。已有旧式半成品最终目录也不会被自动删除，请选择新的输出目录。

至少需要两个 token，`context_len >= 2`，`1 <= stride < context_len`。`context_len` 包含窗口的最后一个目标 token，因此单次模型输入最长是 `context_len-1`。窗口重叠用于上下文，只有新增目标区间计分；整份文本的第 0 个 token 不计分，所有位置 `1..N-1` 恰好计分一次。每个窗口的模型 position/cache 从头开始；这是保留的旧评测口径。

run 校验 corpus 文件 hash、完整 windows 分区及 token 范围。HF/PT2 的 `--tokenizer` 接受目录，PTE 接受本地 `tokenizer.json`；省略时，HF 从模型目录加载，PT2/PTE 使用冻结快照。会比较**完整 token→ID 映射 hash**，并验证固定文本重编码的 IDs 完全一致；目录 tokenizer 还比较特殊 token 配置。PTE 单 JSON 使用其中完整词表和实际编码结果验证，不虚构其缺少的 HF special-token 配置。

如果模型旁边存在 `tokenizer.json`，也会额外验证它，不能通过显式指定另一 tokenizer 跳过模型词表检查。HF 检查模型 embedding 和位置容量，PT2 检查导出 shape 与 logits vocab，PTE 检查编译 metadata。导出文件本身不携带完整 token→字符串映射时，仍需调用者保证传入 tokenizer 是该模型导出时使用的文件；只有 vocab size 不能证明这点。

## 输出、比较与安全 resume

成功运行产生 `metadata.json`、逐窗口 `W*.json`、`engine.json`（HF/PT2）或独立设备 attempt（PTE），最后才发布 `summary.json`。

标准口径为 `PPL = exp(sum(token_nll) / target_count)`，自然对数、逐 token 加权，不能平均各窗口的 PPL。HF/PT2 的交叉熵使用 float32 logits，主机累加使用 float64；PTE 使用 float64 logsumexp，uint16 的等价计算使用导出 scale。PPL 超出 float64 表示范围时保存 `ppl: null`、`ppl_overflow: true` 和有限的 `mean_nll`，不会保存 JSON Infinity。

summary 包含 `complete/count/target_count/nll_sum/mean_nll/ppl`、完整 `metric` 口径、`standard_ppl_eligible`、model/corpus/tokens/targets/tokenizer/identity SHA256、context/stride 和全部 window record hash。只有覆盖完整且验证通过后才标记标准 PPL；PT2 还必须具备每个窗口的因果性检查证据。

比较时先匹配 `tokens_sha256`、`targets_sha256`、`tokenizer_sha256`、context/stride 和 `metric`；再根据目标明确精度差异。`corpus_sha256` 还绑定 tokenizer 快照文件与准备工具版本。仅相同模型名称或 token 数不证明可比。

在完全相同命令末尾追加 `--resume`，可继续已有输出：

```bash
python "$MV/run.py" ppl run \
  --engine hf --corpus /path/to/corpus \
  --model /path/to/local-hf-model --tokenizer /path/to/local-tokenizer-directory \
  --device cuda:0 --dtype float32 --output /path/to/runs/hf --resume
```

resume 绑定模型全部文件内容、固定语料、tokenizer、配置覆盖、引擎选项、评测代码、Python/依赖版本、设备和精度；模型目录不得包含 run 输出。设备运行还绑定 driver、库及 device build。每个缓存窗口验证 target 区间、count、全部 NLL 的长度/有限性/非负性、NLL sum、record hash 和 run identity；已完成 summary 也必须与所有记录一致。任一不符拒绝，不自动覆盖。写入为同文件系统原子 JSON，并使用文件锁拒绝并发写同一 run。

PTE resume 从主机已验证窗口继续，始终创建新的设备 attempt，不信任设备上旧的可变缓存。失败但已拉回且校验有效的完整窗口可复用；硬中断前尚未拉回的设备窗口可能需要重新计算。

## 轻量离线测试

从仓库根目录执行，无模型下载、CUDA 或设备要求：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 \
  python -m unittest discover -s "$MV/ppl" -p 'test_*.py' -v

ppl_test_dir=$(mktemp -d)
c++ -std=c++17 -Wall -Wextra -Werror "$MV/ppl/test_driver_contract.cpp" \
  -o "$ppl_test_dir/test_driver_contract"
"$ppl_test_dir/test_driver_contract"

python "$MV/run.py" ppl prepare --help
python "$MV/run.py" ppl run --help
```

Python 测试只生成随机小型 CPU HF/PT2 模型，覆盖独立 logsumexp 对照、target 分区、逐值 resume 完整性、同大小不同词表拒绝、PT2 形状与未来泄漏拒绝、PTE dry-run 零 ADB 调用和 mock 结果接收。C++ 契约测试覆盖动态 AR/context 切分、float/uint16 NLL 和非有限值拒绝。顶层 pytest 可能先导入仓库的 `llama/__init__.py` 而要求额外 ExecuTorch 环境，以上 unittest 命令可独立执行。

真机逐 token 数值回归见顶层 [设备测试命令](../README.md#7-兼容验证与注意事项)。`test_ppl_device.py` 显式启用后检查 KV/hybrid 的 166 个目标 token、跨 AR/窗口边界与无重算恢复；未设置设备变量时跳过。PPL 的 `repetition_penalty=1.0` 关闭惩罚，但 runner 仍要求合法的正数窗口，因此 driver 传入 128，不用 0 表示关闭。
