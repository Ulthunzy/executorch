# Qwen3.5-35B-A3B MoE，SM8975/V85

本目录维护已验证的 40 层 MoE 推理、融合图导出及专家重构代码。独立于 `llama.py`。默认使用融合后的 Attention＋Router/shared、AR128 prefill、AR1 decode、每层 32 槽 LRU（可用 `--cache-experts 64` 切换）、命中专家先计算与缺失专家异步读取。

当前默认使用 P/D 共享常量、外部权重文件与公共 head；11 个共享 context 在整个会话中常驻，图片输入或显式 `--with-vision` 另保留 1 个视觉 context。Router/shared 为 GPTQ W8A16，embedding 为 group-32 INT4，专家缓存跨阶段交接复用。历史独立权重版本的测试记录仍保留在下文。

导出、量化和推理各自保留入口，共用 [configs/model.json](configs/model.json) 中的量化、KV/state、上下文、thinking 和调度配置；[configs/sm8975.json](configs/sm8975.json) 只管理本机和设备路径。**完整命令、参数生效位置及支持范围见 [CONFIGURATION.md](CONFIGURATION.md)。**

校准源保留在 `llama-output/qwen35-moe-from-safetensors-gptq-20261024`，包含全矩阵 GPTQ 权重、binding、WikiText 语料、两张校准图片及运行库。统一入口的新完整导出位于 `llama-output/qwen35-35B-A3B-export`；2026-10-26 已在 SM8975 验证有效长度 AR128、文本/图片及多轮推理。下文历史命令说明当时测试口径，当前运行使用统一导出及 `CONFIGURATION.md` 中的命令。

分词器继续使用原有 `meta-pytorch/tokenizers` 子模块，锁定上游正式提交 `32af6c1b623639515a08ab9b2b4580ffe570c42a`（上游 PR #198，支持 `ByteLevel use_regex=false`），无需额外的自定义分词器仓库。拉取源码后执行 `git submodule update --init --recursive`。更新版本后应重建对应 Android 分词器静态库，再重新链接 runner/JNI，保证头文件与静态库匹配。

该上游版本的32项 pre-tokenizer 测试及1项 HF ByteLevel 回归通过，主机和 Android C++17 编译兼容性检查通过；对冻结评测的12724条 NFC 输入，token ID 与保存的原设备结果逐项一致。[分词验证记录](docs/tokenizer-upstream-20261026.json)保留版本与输出哈希；本次未重跑模型推理，原129题评测结果继续保留。

## 配置驱动的统一导出

正常导出只需一条命令：

```bash
python examples/qualcomm/oss_scripts/moe/qwen35-35B-A3B/export.py \
  --config examples/qualcomm/oss_scripts/moe/qwen35-35B-A3B/configs/export_sm8975.json
```

导出配置选择文本/视觉组件、PTE/QNN 格式、复用校准产物或从 safetensors 开始，以及输出与工具链路径；引用的 `model.json` 统一管理量化、KV、上下文和调度。`--soc-model` / `--soc_model` 选择 SM8750/V79、SM8850/V81 或 SM8975/V85，`--device` 选择采集设备。内部串行执行构建、量化、分组 QNN 编译、共享权重分离与 PTE 打包，保存检查点以便续跑。`--print-plan` 查看计划，`--check` 检查本地依赖。原分步脚本已移入 `export/`，作为内部实现保留；详细字段、产物格式和推理命令见 [配置说明](CONFIGURATION.md#一条命令导出)。

正常推理统一由 `run_chat.py` / `run_inference(...)` 进入设备端常驻会话：

```bash
python examples/qualcomm/oss_scripts/moe/qwen35-35B-A3B/run_chat.py \
  --weights-path llama-output/qwen35-35B-A3B-export \
  --device A7SR026915000282 --soc-model SM8975 \
  --temperature 0.7 --max-new-tokens 256 \
  --interactive --stream --prompt "请介绍一下自己。"
```

图片使用 `--image-path /path/to/image.jpg`，在设备内解码；后续图片使用 `--with-vision` 保留视觉塔。`--ignore-eos` 控制停止条件，`--no-stream` 改为每轮生成完再显示。旧 teacher-forcing / 张量dump仍进入诊断链路。新会话要求配套的新runner和支持有效长度的AR128图，统一导出会同时生成；旧产物不会被自动覆盖。三种架构的专家内核与GDN包已通过交叉编译，V79/V81整模设备验证尚未执行。

### 2026-10-26：代码复核与新入口设备回归

后续目录精简已移除顶层 `multimodal.py`、`pack_pte.py`、`pack_shared_weights.py`、`separate_shared.py`，配置及构建辅助也下移。顶层只保留 `export.py`、`run_chat.py`、`build.sh` 三个入口。共享在QNN编译时启用，PTE打包和权重文件分离均由导出阶段自动完成；推理只读取 `plans/` 中的现成产物，缺失或配置不匹配时要求重新导出，不再临时打包。具体职责见[代码位置](#代码位置)及[导出格式说明](CONFIGURATION.md#导出共享图权重与-pte)。

目录整理后的54项回归通过；AR128/AR256调度PTE、decode PTE，以及一组真实共享context分离得到的 `graph.qnn`、`weights.bin` 均与已有产物逐字节相同。SM8975用现成PTE完成文本和图片两项整模推理，分别回答“2”和“图中男子穿的上衣是橙色。”，运行日志没有PTE生成记录。此轮没有重跑GPU校准或40层QNN编译。详情见[产物一致性](../../../../../llama-output/qwen35-export-layout-review-20261026/ARTIFACT_EQUIVALENCE.json)及[设备记录](../../../../../llama-output/qwen35-export-layout-review-20261026/DEVICE_RESULTS.json)。

删除无调用方的独立 decode/router 导出器、重复的分组导出调度、未使用的 `--model` 参数和 checkpoint 路径 helper；合并计划打包、视觉校验及重复上传逻辑。部署失败也保存 `RUN.json`。SoC 与架构直接取自 `qc_schema.py` 的 `QcomChipset`、`HtpArch`、`_soc_info_table`；本目录只限制 Native 的 V79/V81/V85 支持范围，构建和导出不再各维护一份映射。

分词探针中的27与226是独立请求，后者已经包含文本。同一请求先合并模板、文本和图片 token，再统一按 AR128 分块；只有尾部的无效位置补齐。设备 token IDs 与 Transformers 完全一致，图片解码、缩放、归一化及 patch 数据与 PIL 参考逐字节一致。

| 分词探针 | 文本及模板 | 图片 token | 有效总数 | AR128 块数 | 无效尾部 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 纯文本 | 27 | 0 | 27 | 1 | 101 |
| 图文 | 30 | 196 | 226 | 2 | 30 |
| 增加27个文本 token 的图文 | 57 | 196 | 253 | 2 | 3 |

整模设备为 SM8975/V85、SDK 2.48.1、40层、1152总上下文、每层32槽，复用冻结文本 GPTQ 校准结果，视觉重新执行 W8A16 GPTQ 校准，视觉 QDQ SQNR 为28.25 dB。新图和新runner串行运行以下六项，均通过；不同文本提示的 token 数与上面的分词探针不同。

| 功能 | 首轮有效输入/AR128块数 | 生成 token 数 | 结果 |
| --- | --- | --- | --- |
| 文本 greedy | 28 / 1 | 62，遇 EOS | 中文介绍正常 |
| 图片 greedy | 226 / 2 | 80，达到上限 | 识别白牛、橙衣男子、草地和帐篷 |
| 温度0.7、非流式 | 30 / 1 | 48，达到上限 | 生成 Python 学习建议 |
| 忽略 EOS | 28 / 1 | 固定12 | 输出长度控制正确 |
| 文本多轮 | 35 / 1 | 2＋12 | 追问正确恢复“蓝鲸42” |
| 253-token 图文及追问 | 253 / 2 | 48＋9 | 追问上衣颜色回答“橙色” |

两组多轮输入的追加日志均为 `ar1_calls=0`，会话为 `model_reloads=0`。253-token 图文的日志为 `CHAT_INPUT valid_tokens=253 text_and_template_tokens=57 image_tokens=196 padded_tokens=256 ar128_blocks=2`，没有将文本和图片分别补齐。

设备回归还定位并修复了两项问题：共享 context 的 IO 预算只统计了 UINT16 state/KV，漏算 FP32 激活及跨 context 注册，原192MiB预算不足。本轮记录的注册量约223.9MiB；改为按全部图IO及页对齐计算安全上界，新产物预算为400MiB，原错误25及8003在后续六项测试中未再出现。非交互 ADB 部署命令改用空stdin，避免抢先读取聊天进程的下一轮输入。另补齐 `quantization/__init__.py`，修复内部阶段脚本的同名模块遮蔽问题。

52项Python回归及C++ context回滚、FP32 IO预算、cache交接和运行参数测试通过；`git diff --check` 通过。原始失败记录、分词探针、六组运行及源码/二进制指纹保留在[设备回归目录](../../../../../llama-output/qwen35-reviewed-device-tests-20261026/DEVICE_RESULTS.json)和新导出的 `RUNTIME_REVIEW.json`。导出检查点保留编译时的原始来源标识，没有因运行时修复而伪造重编译记录。本次为功能回归，未重测峰值内存或完整benchmark，已停止评测的129题记录不变。

## 2026-10-26：2048 上下文图文连续负载测试

设备终端 C++ runner 实测完整40层，统一总上下文2048、prefill AR128、`layer_major`；每层32/64槽LRU，真实Top8不变。Embedding INT4/group32、Router/shared GPTQ W8A16、专家W2/N4×A8、KV/GDN/conv UINT16。每组生成指定数量的输出；图文输入长度包含模板和196个视觉token。同一输入的32/64槽生成ID全部一致。

下表各数字均按 **32槽 / 64槽** 排列，速度单位为token/s。LLM prefill包含专家IO，含视觉prefill另外计入Android内部图片解码、缩放、patch和resident encoder；均排除模型初始化。

| 输入 | 输入/输出token | LLM prefill | 含视觉prefill | Decode | 模型峰值GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 纯文本 | 256/256 | 159.66 / 175.40 | 159.66 / 175.40 | 10.84 / 12.86 | 3.801 / 4.787 |
| 纯文本 | 512/512 | 272.56 / 301.47 | 272.56 / 301.47 | 10.91 / 12.89 | 3.848 / 4.924 |
| 纯文本 | 1024/1024 | 352.44 / 332.09 | 352.44 / 332.09 | 10.02 / 11.60 | 3.968 / 4.994 |
| 图文 | 256/256 | 135.53 / 145.03 | 91.03 / 119.20 | 11.30 / 13.58 | 4.411 / 5.429 |
| 图文 | 512/512 | 227.93 / 234.53 | 165.33 / 167.76 | 10.76 / 12.81 | 4.461 / 5.418 |
| 图文 | 1024/1024 | 341.38 / 315.59 | 265.63 / 251.90 | 9.81 / 12.19 | 4.529 / 5.509 |

这是顺序连续负载下各一轮有效测量，长输出阶段Thermal Status为2–3，不能把跨输入长度的差异全部归因于上下文。64槽本轮decode提升15.8%–24.3%，多占约1GiB；1K prefill没有加速。图片encoder约120ms，图片解码/缩放/patch约257–798ms。64槽的个别轮次SwapPSS达84–145MiB，另列在报告中。

**历史性能对账补充：**旧424.05/140.25 token/s表只计LLM prefill，图文应与本表135.53比较，91.03另外包含了图片处理。手机恢复到Thermal Status 0后，同一2048 PTE、32槽、完全相同1024输入/1024输出、同样开启内存采样，prefill重新测得 **420.57 token/s**、decode **10.427 token/s**、峰值 **3.968560 GiB**；输入和1024个输出ID逐项相同。旧图文提示的LLM prefill复测为 **138.76 token/s**，79个输出ID与历史相同。旧/新模型80份prefill IO描述相同；无采样新提示两轮425.74/422.56 token/s，低热状态下旧提示无采样423.55、开采样423.02。大幅差距主要对应连跑后的设备状态，当前没有约17%的固有prefill回退证据。热状态与同步/调度的独立贡献尚未进一步隔离。详见[历史性能对账及全部控制实验](../../../../../llama-output/qwen35-moe-performance-reconcile-20261026/REPORT.md)。

峰值覆盖初始化、prefill/decode与交接，统计CPU PSS＋去重DMA－PSS中已计DMA＋未映射memfd快照驻留页；目标采样间隔200ms，实际间隔随采样开销变化。这是模型进程占用，未包含整机其它应用或JNI调用方的JVM。12组均跑满目标输出，峰值样本无fdinfo/memfd读取错误；内存曲线未出现decode期间持续增长。详情、初始化耗时、逐token延迟、SwapPSS及原始记录见[完整报告](../../../../../llama-output/qwen35-moe-image-text-performance-20261026/REPORT.md)和[CSV](../../../../../llama-output/qwen35-moe-image-text-performance-20261026/RESULTS_DETAILED.csv)。

固定长度测试可用`--prompt-tokens N`（128的倍数）指定包含模板和视觉token的首轮总输入长度，配合`--max-new-tokens N --ignore-eos`。本次12组均未在目标长度内遇到EOS；这些测试不替代PPL或标准质量评测。

## 32K完整模型质量评测

2026-10-26，完整40层32K PTE已部署到SM8975，中文生成和2920-token长输入预检通过。计划评测AIME 2026共30题、HumanEval共164题、GPQA-Diamond共198题、MMLU-Pro共12032题、IFBench共300题，总计12724题。保持当前GPTQ权重、W2专家、32槽LRU，temperature=0、thinking关闭。每题独立设备进程执行，成功结果逐题保存；五项全部完成前不生成Average。

本次已按用户要求停止。已保存HumanEval/0–128共129题，114题通过，阶段通过率88.37%；HumanEval尚缺35题，其它四项尚未运行，这不是完整评测分数。第130题没有完整结果，不计入已测题目。原始输出、评分、冻结manifest和来源哈希保留在评测目录，并备份到`checkpoints/stopped-20261026/`；[源码内检查点](docs/benchmark-checkpoint-20261026.json)记录所有已测题号及输入/输出哈希。以后续跑同一冻结版本时，`run_device.py`按题号、prompt哈希、输出上限及manifest哈希跳过成功结果，无需重复推理。换用新图或新运行库需要独立结果集，不能合并旧版本分数。

扩容预检同时修复了prefill共享mask固定1024及位置固定896的限制，改为从图的KV窗口取值；memfd快照通过自有句柄读写，普通文件路径继续同步写入。主机和SM8975文件IO回归、32K真实设备预检均通过。完整协议、断点续跑命令和动态结果位于[评测目录](../../../../../llama-output/qwen35-moe-full-benchmarks-20261026/README.md)。

## 设备终端对话

`qwen35_moe_runner` 在设备上完成 ChatML、文本编码、JPG/PNG 解码与预处理、视觉编码、流式吐字及多轮会话，直接读取原始 `tokenizer.json`。模型文件已部署后，在 device shell 执行：

```sh
cd /data/local/tmp/qwen35-raw-gptq-20261024/runtime/w2a8-fused-c1152-p128-cache32/prefill

./qwen35_moe_runner \
  --model-root .. \
  --tokenizer-path ../../tokenizer.json \
  --interactive \
  --max-new-tokens 128
```

输入 `/reset` 清空对话，输入 `/exit` 退出。也可用 `--prompt "第一问" --prompt "第二问"` 连续执行多轮；增加 `--interactive` 可继续从终端读取。`--temperature 0` 为 greedy，`--enable-thinking` 开启思考模板，默认关闭。运行日志及逐 token 时间记录在模型目录的 `chat.log`，可用 `--log-path` 修改。

首轮 Attention/Router 使用 AR128，普通请求仅在序列末尾加入无效 padding；显式 `--prompt-tokens` 的定长 benchmark 才向 system-message 填充真实空格。512–1024位置的同层专家 token 在 Native 内归并，HMX tile 仍最多32行；较短输入的 Native 维持 AR128。终端按实际补齐长度选择无权重的 host PTE，默认按 `layer_major` 让同一层完成全部 AR128 块并复用专家，再进入下一层。顺序来自部署的 `prefill/plans/PLANS.json`，可用 `--prefill-order chunk_major` 显式对照。后续轮将尚未反馈的最后一个生成 token、轮次分隔符及新问题按 AR128 分块追加，仅有效位置更新 state/KV，不重新计算历史；AR1只用于生成。Context、buffer、每层 LRU 和原生服务在会话内常驻；`/reset` 只清空 state/KV 和位置，不重新装载权重或丢弃专家缓存。达到静态上下文容量时拒绝追加，用户可显式 `/reset`。

设备端使用与 `qnn_llama_runner.cpp` 相同的 HFTokenizer/BPE 和 ExecuTorch sampler。Qwen3.5 的 NFC 使用 Android 公共 ICU API（Android 31+），避免现有内置 NFC 丢失重音字符；原 tokenizer 文件保持不变。

图片也可以直接从设备文件系统读取：

```sh
./qwen35_moe_runner \
  --model-root .. \
  --tokenizer-path ../../tokenizer.json \
  --image-path ../../test.jpg \
  --prompt "请用中文描述这张图片。" \
  --interactive \
  --max-new-tokens 128
```

换图使用 `/image /data/local/tmp/another.png 请描述这张图。`；有空格的路径加双引号。每轮支持一张图片，旧图的 state/KV 保留以供追问。`/reset` 同时清除图文历史并恢复文本 RoPE。JPG/PNG 由 Android ImageDecoder 解码，再按校准路径做抗混叠 BICUBIC、RGB 归一化和 patch 排列；形状与 MRoPE 配置来自部署的 `VISION.json` 和 `model-config.json`。当前静态图为 448×448、196 个图像 token，源图最多 32 百万像素。

部署还包括模型根目录的 `base-cos.f32`、`base-sin.f32` 和 `prefill/vision.bin`。图片输入或显式 `--with-vision` 会话的视觉 context 和 IO buffer 随完整模型初始化，所有图文轮次复用，结束会话时统一释放；纯文本会话不加载视觉模块。首轮和后续图文输入均使用有效长度 AR128 分块，旧版 AR1 换图耗时仅保留在下方历史记录中。

SM8975 的 JPG、PNG 预处理、256-token 图文模板和 MRoPE 与 Python 路径逐字节一致；看图描述、跨轮追问、新图片、坏路径恢复和 reset 后再次看图均已验证。整个会话保留 11 个共享 context 和 1 个视觉 context，视觉重复执行不创建或释放 context/buffer。详见[设备图片会话记录](../../../../../llama-output/qwen35-moe-from-safetensors-gptq-20261024/device-image-chat-tests/REPORT.md)。

2026-10-24 调度修复前的设备终端性能复测，32 槽专家缓存、1152 静态容量：三轮 256-token 图文 prefill 为 **3.44–3.60 秒 / 71.19–74.35 token/s**，decode 为 **11.47–11.66 token/s**；三轮均输出相同的 79 个 token（含 EOS）。首次启动到首字 **13.51–14.43 秒**，其中模型/运行时 setup 为 6.46–6.99 秒，视觉执行约 118 ms。当时的 PTE 按两个 AR128 块执行 `chunk_major`，未启用配置 JSON 的 `layer_major`。换图追加 217 个新 token 为 **24.17–24.42 秒**；其后 decode 一轮遇到长停顿为 5.60 token/s，复测为 9.98 token/s。原始数据、统计口径及 I/O 拆分见[图文性能报告](../../../../../llama-output/qwen35-moe-from-safetensors-gptq-20261024/device-image-performance-20261024/REPORT.md)。

### Prefill / Decode 阶段计时

2026-10-24，增加持久 decode 的逐 token 阶段日志后，按相同图文输入再测三轮；输出 token IDs 与上轮一致。每层缓存 32 个专家，未改变图、权重和量化。以下为三轮均值。

Prefill 为 256 token 的总时间，AR128 分两块；不含模型初始化和视觉编码。

| 环节 | ms / 256 token | 占比 |
| --- | ---: | ---: |
| RPC 外读取处理 | 2293.95 | 66.83% |
| Native MoE，含等待 | 543.59 | 15.84% |
| Attention | 357.48 | 10.42% |
| Router/shared | 74.83 | 2.18% |
| State/KV 反馈、head 及其它 | 162.41 | 4.73% |
| **总耗时** | **3432.26** | **100%** |
| **吞吐** | **74.59 token/s** | |

Decode 每轮真正执行 78 步输出 AR1，三轮共 234 步，不把 prefill 直接采样的首 token 算作一次 AR1。

| 环节 | ms/token | 占比 |
| --- | ---: | ---: |
| RPC 外读取处理 | 2.236 | 2.57% |
| Native MoE，含等待 | 39.932 | 45.92% |
| Attention，含融合 Router/shared 计算 | 38.253 | 43.99% |
| Router/shared 输出交接 | 0.047 | 0.05% |
| State/KV 反馈、head 及其它 | 6.491 | 7.46% |
| **总耗时** | **86.958** | **100%** |
| **吞吐** | **11.50 token/s** | |

Decode 的 40 层都已融合 Attention 与 Router/shared；约 0.047 ms 只是结果交接，不能解释为 Router 的计算耗时。30 个 GDN 融合图合计约 30.65 ms/token，10 个 full-attention 融合图合计约 7.51 ms/token，均属于上表 Attention 行的子项。Native 包含内部的专家就绪等待，RPC 外约 2.2 ms 不等于完整的专家 I/O 成本。

同一会话换图后，decode 为 99.984 ms/token、10.00 token/s；Native 从同会话首轮的 39.38 增到 51.49 ms，Attention 融合段仍约 38.4 ms。专家命中率 76.17% → 61.95%，直接读取量 65.28 → 104.23 MB/token。阶段增量主要落在 Native，并伴随缓存命中下降。完整主表、逐 token CSV 和原始日志见[阶段拆分报告](../../../../../llama-output/qwen35-moe-from-safetensors-gptq-20261024/device-stage-breakdown-20261024/REPORT.md)。

### 首轮 prefill 调度修复

上述阶段测试后确认：旧终端入口固定加载 `[1,128]` 的 `prefill.pte`，没有使用配置中的 `layer_major`。256 个输入 token 的第二块和第一块有 77.25% 的逐层专家重合，但每层32槽 LRU 不能保留全部工作集，导致重复读取。现在按首轮实际 padding 长度自动选择128～1024的host计划，QNN与Native的AR仍为128；导出阶段准备8份计划，部署只读取，含JSON共约0.61MB，不复制权重。

SM8975、相同图文输入、同一目录与二进制：

| 指标 | chunk-major 对照 | 修复后 layer-major，3轮均值 |
| --- | ---: | ---: |
| 专家文件读取量 | 9.055 GB | **5.741 GB** |
| RPC 外读取处理 | 2300.98 ms | **1782.16 ms** |
| Prefill 总耗时 | 3438.95 ms | **2899.51 ms** |
| 吞吐 | 74.44 token/s | **88.29 token/s** |

总耗时下降15.69%，文件读取量下降36.60%。路由及最终logits逐字节一致，集成后的三轮输出token也与对照完全一致。额外临时memo为当前层210MiB，进入decode前释放。联合路由实际使用7635个不同专家，扣除预加载命中的929个，剩余6706个专家所需的对齐读取字节数恰好为5.741GB，说明剩余读取量已不是当前样本的重复读；复制、同步及真实外存工作集仍需继续优化。

当前设备已部署更新后的runner与计划集合；已有部署不能只替换二进制而缺少 `prefill/plans/`。完整原因、字节核对、正确性与时间细项见[Prefill I/O 分析报告](../../../../../llama-output/qwen35-moe-from-safetensors-gptq-20261024/prefill-io-analysis-20261024/REPORT.md)。

SM8975 已验证重复 `--prompt`、交互输入、跨轮暗号记忆、超限拒绝后历史不变、`/reset` 与正常退出。8 组编码与 Transformers 逐字节一致；原主机单轮回归的 7 个生成 token 与已有版本完全一致。四轮会话保留 11 个共享 context，换轮与 reset 没有增加 context、IO 注册或 RPC buffer 生命周期操作。详见[设备端会话测试记录](../../../../../llama-output/qwen35-moe-from-safetensors-gptq-20261024/device-chat-tests/REPORT.md)。

### Prefill 共享 DMA memo 与计算批次优化

2026-10-24，以已提交的 `6a4573f0a6` 为基线继续优化，保留 SM8975/V85、40 层、AR128、每层 32 槽 LRU 与同一份全矩阵 GPTQ 权重。这里的计算批次指专家个数，token AR 仍为 128。

- 将当前层约 210 MiB 的临时 memo 改为 DSP 可映射的 uncached rpcmem。缺失专家通过对齐 staging 读取后只复制一次到 memo；同层其余块直接消费已有 payload，去掉 memo→pipeline 的重复复制。
- 持久 decode LRU 保持 32 槽，prefill 缺失专家每批最多处理 128 个；命中专家仍先算，缺失读取与计算重叠。256-token 样本的 Native 调用 **447→203 次**，1024-token 样本 **1484→674 次**。
- 异步读取只修改不重叠的 payload；当前批 Native 返回后才更新专家索引表。多块计划不再申请两块约 26.25 MiB 的 pipeline bank；临时 memo 在交接 decode 前解除映射并释放。

| 输入 | 已提交基线 | 优化后 | 耗时下降 |
| --- | ---: | ---: | ---: |
| 256-token 图文，三轮均值 | 2.916 秒 / 87.78 token/s | **2.646 秒 / 96.73 token/s** | **9.26%** |
| 1024-token 文本，两轮热态均值 | 6.622 秒 / 154.63 token/s | **5.772 秒 / 177.41 token/s** | **12.84%** |

256-token 三轮范围：基线 **2.889–2.945 秒**，优化后 **2.631–2.661 秒**。1024-token 两轮采用 A/B 交替顺序；先前首轮基线有约 120 ms embedding 文件缺页额外成本，未纳入热态均值。两种输入的读取量分别保持 5.741 GB、6.319 GB；没有减少 Top8、删除专家或改变权重量化。

| 环节 | 基线 ms / 256 token | 优化后 ms / 256 token |
| --- | ---: | ---: |
| RPC 外读取处理 | 1799.53 | 1695.58 |
| Native MoE | 546.24 | 377.63 |
| Attention | 348.22 | 356.40 |
| Router/shared | 67.46 | 73.42 |
| State/KV 反馈、head 及其它 | 154.91 | 143.40 |
| 总耗时 | 2916.35 | 2646.43 |

图文与 1K 文本各 **85 份 hidden、logits、40 层 state/KV、路由和历史信息逐字节一致**；图文完整 79 个输出 token（含 EOS）和文本 16 个输出 token 也一致。128-token 文本继续走原单块路径，实测 1.363 秒，正常回答 `1加1等于2。`。图文完整 78 步输出 decode 为 11.47→11.50 token/s，本轮收益在 prefill。

O_DIRECT 直接读 rpcmem 在该设备返回 EFAULT，因此保留一次 staging 复制；仅把 O_DIRECT 目的地换成普通 memo 没有稳定加速，已丢弃。256 个专家一批试验为 2.703 秒，比 128 个一批慢，最终保留 128。循环内 context 创建/释放、QNN IO 注册/注销和 RPC buffer 分配/释放仍全部为零。RPC 外读取处理仍占优化后约 64.1%，包含读取、复制、同步与剩余等待，不能当成纯 UFS 时间。

完整运行口径、阶段 CSV、85 份输出指纹、失败试验和源文件指纹见[共享 DMA memo 测试报告](../../../../../llama-output/qwen35-moe-from-safetensors-gptq-20261024/prefill-direct-io-20261024/REPORT.md)。

### Prefill 就绪分组与跨阶段 I/O 重叠

2026-10-25，以前一节的共享 DMA memo 版本为基线继续优化。SM8975/V85、相同全矩阵 GPTQ 导出、静态容量 1152、token AR128、每层 32 槽、当前层约 210 MiB memo、8 个读取线程均保持相同。三轮采用基线/新调度交替运行，专家文件仍用 O_DIRECT。

1. 以当前层 memo 的有效 payload 判定就绪；已就绪专家不再和缺失专家混批，不再经过 reader 线程屏障。真正缺失的权重在后台读取，就绪批立即计算。
2. 中间 AR128 块只更新 LRU 元数据；本层最后一块才回填常驻 payload。物理槽是否已有同一专家，按本层初始物理槽 ID 判断，不能用中间块已更新的逻辑 LRU ID 跳过复制。
3. Seed 复制与剩余 Attention/Router 段重叠；最终回填与下一层首个 Attention 或最终 head 重叠，覆盖 memo 和交接 decode 前再确认完成。
4. 首块 Router 给出真实路由后即预取首批缺失专家，与后续 Router/shared 及 Native 输入发布重叠；未读取完成的 payload 不进入就绪集合。缺失批次只在消费前等待剩余部分。

| 输入，三轮均值 | 上一版，本轮复测 | 新调度 | 耗时下降 |
| --- | ---: | ---: | ---: |
| 256-token 图文 | 2.650 秒 / 96.61 token/s | **2.227 秒 / 114.95 token/s** | **15.96%** |
| 1024-token 文本 | 5.821 秒 / 175.93 token/s | **4.215 秒 / 242.94 token/s** | **27.58%** |

| 环节，ms / 1024 token | 上一版 | 新调度 |
| --- | ---: | ---: |
| RPC 外读取处理 | 2286.54 | 1048.46 |
| Native MoE 调用 | 1362.95 | 1268.48 |
| Attention | 1401.08 | 1318.14 |
| Router/shared | 239.26 | 197.21 |
| State/KV 反馈、head 及其它 | 530.78 | 382.73 |
| 总耗时 | 5820.62 | 4215.03 |

| RPC 外读取处理子项，ms / 1024 token | 上一版 | 新调度 |
| --- | ---: | ---: |
| 批次剩余等待 | 1553.83 | 1017.83 |
| Seed 复制可见处理 | 75.46 | 0.02 |
| LRU 回填可见处理，含准备/同步 | 604.17 | 23.14 |
| 批次准备/发布 | 53.06 | 7.46 |
| 其它控制/统计 | 0.041 | 0.009 |
| 合计 | 2286.54 | 1048.46 |

1K 的实测重叠时间：Seed 与 Attention/Router **88.83 ms**，LRU 回填与下一层 Attention/head **94.61 ms**，UFS/复制与输入发布 **161.54 ms**，UFS/复制与后续 Router **167.82 ms**。这些是上表中的重叠子项，不能再次加入总耗时；worker 的 pread/copy 累计时间也不是串行墙钟时间。

1K 前两个 AR128 块承担 **97.28%** 的剩余读取等待；起点 640/768/896 的三块，40 层合计各只等 **0.020/0.029/0.016 ms**，说明这些块的缺失读取已基本被计算隐藏。这里的就绪包含同层 memo 命中，不能当作 32 槽跨层 LRU 命中率。

两种输入仍分别读取 5.741 GB / 6.319 GB，没有减少 Top8 或改变权重。图文与1K各85份 hidden/logits/state/KV/路由/历史信息逐字节一致，图文79个输出 token 和文本16个输出 token 一致。128-token 单块、显式 chunk-major、图片追问均通过；chunk-major 的85份产物也与对照一致。图文完整78步输出 decode 为 11.56→11.52 token/s，本轮未修改 decode 调度。

新增 `PREFILL_PROFILE`、`PREFILL_COPY` 和逐层逐块 `PREFILL_PIPE` 明细。Token 循环内 context、QNN IO 和 RPC buffer 的创建/释放计数仍为0，当前设备保留新 runner。报告和 CSV 见[Prefill 并发调度测试](../../../../../llama-output/qwen35-moe-from-safetensors-gptq-20261024/prefill-overlap-20261025/REPORT.md)。

### Prefill 单次 Native 调用与持续读取队列

2026-10-25，继续以前一节的 2.227 秒 / 4.215 秒版本为基线。40 层、AR128、GPTQ 专家权重、Top8、UINT16 state/KV、每层 32 个常驻专家、约 210 MiB 当前层 memo 和 8 个读取线程均保持相同。三轮交替 A/B，专家文件仍通过 O_DIRECT 读取。

1. 同一 AR128 的就绪和缺失专家合入一次 Native 调用，激活 A8 准备、资源申请、RPC 和最终合并各执行一次。就绪专家先计算，缺失专家按读取线程发布的标记继续计算，不等待整批权重。
2. Native 使用 12 MiB VTCM，把原先跨批存入 DDR 的 8 MiB 分路结果留在 VTCM；仍按原 Top8 rank 顺序求和，保持数值一致。DRAM 专家缓存没有扩大；旧分批调用申请 5 MiB VTCM。
3. Router 每完成一个 AR128 块，就把这一层新出现的专家追加到同一读取队列；后续块的权重提前到达。记录数组预分配，读取期间地址保持稳定；本层最后一块完成后才等待队列收尾、回填最终 LRU。
4. ARM 仅在 payload 复制完成后发布就绪状态，DSP 通过绕过 cache 的 DMA 读取标记。读取失败会发布错误；DSP 等待就绪有 5 秒超时。设备注入错误和超时后，同一 handle 的线程与资源均可继续复用。

| 输入，三轮均值 | 上一版，本轮复测 | 本轮优化 | 耗时下降 |
| --- | ---: | ---: | ---: |
| 256-token 图文 | 2.218 秒 / 115.43 token/s | **1.985 秒 / 128.97 token/s** | **10.50%** |
| 1024-token 文本 | 4.225 秒 / 242.35 token/s | **3.677 秒 / 278.49 token/s** | **12.98%** |

| 环节，ms / 1024 token | 上一版 | 本轮优化 |
| --- | ---: | ---: |
| RPC 外读取处理 | 1037.20 | 32.64 |
| Native MoE，含缺失专家剩余等待 | 1270.69 | 1825.55 |
| Attention | 1325.37 | 1312.28 |
| Router/shared | 198.71 | 189.45 |
| State/KV 反馈、head 及其它 | 393.35 | 316.99 |
| 总耗时 | 4225.33 | 3676.91 |

本轮将缺失等待移入 Native，因此不能单看 Native 一项判断计算是否变慢。应比较 **RPC 外处理＋Native**：1K 从 **2307.90 ms 降至 1858.20 ms**。Native 调用数：图文 **186→80**，1K **635→320**。

1K 吞吐提高 **14.92%**。图文和1K各85份 hidden/logits/state/KV/路由/历史信息逐字节一致；图文完整79个输出 token 与文本16个输出 token 一致。128-token 单块、显式 chunk-major 和图片追问均通过。完整78步图文 decode 为 11.47→11.45 token/s，本轮收益在 prefill，decode 调度保持原样。

两种输入读取量分别保持 5.741 GB / 6.319 GB。上述 prefill 包含 embedding、40层计算、反馈和 head；不含初始化、图片预处理、视觉编码或输出 decode。详细记录、阶段 CSV、错误探针和源文件指纹见[单次 Native 与持续队列测试报告](../../../../../llama-output/qwen35-moe-from-safetensors-gptq-20261024/prefill-native-stream-20261025/REPORT.md)。新 runner 与新 prefill DSP skel 必须配套部署；`build.sh` 会一起构建。

### Prefill 1K 提前路由与专家跨块归并

2026-10-25，以前一节约278 token/s的版本为基线，目标为1K输入至少320 token/s。保持相同40层、GPTQ量化、Top8、UINT16 state/KV、每层32槽LRU和8个读取线程；Attention/Router仍使用AR128，静态容量1152。

1. 每块Attention完成即执行该块Router/shared，真实路由立即进入读取队列；不再等同层8块Attention全部结束才开始路由。后续Attention与UFS读取重叠。Formal Router stage接收已保存的结果，避免重复计算。
2. 512–1024个输入位置时，将同层各AR128块中属于同一专家的token归并，HMX仍每次最多计算32行。1K Native调用320→40次，路由项仍为327680，没有减少专家或Top8。短于512的输入继续使用AR128 Native及VTCM分路结果。
3. 实际计算可以归并，LRU仍按原AR128顺序回放。最终物理槽、逐块seed/stamp和decode交接保持一致。大组的Native IO、DSP分路工作区在初始化时一次配置，token循环没有创建context或RPC buffer。

| 输入，三轮交替均值 | 上一版，本轮复测 | 新版 | 耗时下降 |
| --- | ---: | ---: | ---: |
| 256-token 图文 | 1.981 秒 / 129.24 token/s | **1.822 秒 / 140.49 token/s** | **8.00%** |
| 1024-token 文本 | 3.670 秒 / 279.06 token/s | **2.387 秒 / 428.99 token/s** | **34.95%** |

| 环节，ms / 1024 token | 上一版 | 新版 |
| --- | ---: | ---: |
| RPC 外读取处理 | 34.26 | 67.90 |
| Native MoE，含剩余读取等待 | 1840.46 | 508.31 |
| Attention | 1300.96 | 1334.54 |
| Router/shared，含交接复制 | 185.26 | 244.01 |
| State/KV 反馈、head 及其它 | 308.57 | 232.27 |
| 总耗时 | 3669.52 | 2387.03 |

1K新版本范围 **2.384–2.390秒**，三轮均超过320 token/s，吞吐提高 **53.73%**。实际读取/复制与Attention重叠约 **1065.72ms**。计时已排除队列开启但无活跃读任务的空闲区间；这些重叠子项不能再次加入主阶段合计。

图文与1K各85份hidden/logits/state/KV/路由/历史信息逐字节一致；额外512与1K检查各88份产物，覆盖逐块seed/stamp和最终路由，也完全一致。完整79个图文输出token和16个文本输出token一致。AR128单块、chunk-major与图文追问均通过。完整78步图文decode为11.62→11.64 token/s，本轮主要优化prefill。

每层仍只常驻32个专家；1K归并增加约95MiB DDR工作缓冲（按分配尺寸推算，非实测整机峰值），Native VTCM申请由12MiB改为8MiB。256-token图文只增加约2MiB Router交接暂存，保持AR128专家计算。两种输入的专家O_DIRECT读取量仍为5.741/6.319GB。

上表prefill包含embedding、全部40层、state/KV反馈与head，不含模型初始化、图片预处理和视觉编码。512/1K配置新增的RPC接口要求host与prefill skel配套部署，`build.sh`一起构建；读取线程参数可在设备runner使用`--prefill-io-workers`设置，默认仍为8。12/16读取线程、8 HVX线程试验没有稳定收益，均未作为默认加速配置。

完整分析、对照CSV、错误探针与源码指纹见[1K prefill 320 token/s目标验证](../../../../../llama-output/qwen35-moe-from-safetensors-gptq-20261024/prefill-320-20261025/REPORT.md)。

### Decode 阶段拆分与调度复测

2026-10-25，prefill 优化提交为 `d6d877e938`。继续使用真实40层、专家N2存储/N4×A8计算、每层32槽LRU、UINT16 state/KV、静态容量1152，交替测试原调度与“提前读取＋DSP就绪轮询退避”。文本各3轮，图文各2轮；输出至EOS，首token来自prefill，decode速度仅统计后续AR1步进。

| 输入 | 原调度 | 调度试验 | 结论 |
| --- | ---: | ---: | --- |
| 1024文本，57步AR1 | 97.51 ms/token / 10.25 token/s | 97.97 ms/token / 10.21 token/s | 没有稳定收益 |
| 256图文，78步AR1 | 85.97 ms/token / 11.63 token/s | 84.18 ms/token / 11.88 token/s | 仅约2%延迟下降 |

两个调度试验均已回退。默认分块仍为256KiB，缓存容量和读取策略保持原样；仅保留`CHAT_DECODE_STEP`中的读取线程累计`pread/copy/publish`计时，以及关闭会话时的可选Native汇总。`MOE_NATIVE_PROFILE=1 bash build.sh`编译细分计时版本，运行时`MOE_PROFILE_TOTAL=1`在关闭decode会话时输出汇总；普通构建默认不启用DSP细分计时。

1K原调度主要开销为：RPC外处理3.01ms、Native含缺失等待48.93ms、Attention融合图38.31ms、Router输出交接0.06ms、state/head/采样等7.21ms。**Router/shared计算已包含在Attention融合图中，不能重复相加。**直接读取99.83MB/token、命中率63.56%；图文为65.28MB/token、76.17%。输入内容和输出序列不同，两行速度差异不能直接归因于上下文长度。

Native单独profile的图文结果为39.09ms/token，其中最慢worker的专家就绪等待累计33.35ms，HMX执行累计0.45ms；这些是不同并行区间，不能直接相加。读取线程累计时间也不能加进端到端时间。cached/uncached rpcmem的O_DIRECT直读均返回EFAULT，暂存复制仍需保留。

10轮有效对照的输出token、逐token缓存命中及读取字节数一致；未更改量化或专家路由。另有两次prefill初始化映射错误25，包含原版decode对照；失败记录未计入均值，补跑后通过。该映射问题的复现和修复见下一节。完整decode记录见[Decode调度复测报告](../../../../../llama-output/qwen35-moe-from-safetensors-gptq-20261024/decode-optimization-20261025/REPORT.md)。

### Prefill 初始化映射错误 25

2026-10-25，在连续完整1K prefill＋decode会话的第6次启动复现。25是Native自定义错误：约210MiB的当前层专家memo在QNN恢复完成后才通过`HAP_mmap2`建立DSP映射，返回`0xffffffff`。fd有效、注册长度一致，底层日志为`0x80000608 / QuRT Error 12`。SDK把12定义为映射失败，不能直接解释为手机DRAM耗尽。旧代码还缓存失败指针与fd，导致同一handle后续调用持续返回25，并在关闭时尝试解除无效映射。

修复在Native configure时建立cached resource context，先确定VTCM地址布局，再提前映射大memo，之后恢复QNN。每次Native计算仍需`acquire_cached`，完成后`release_cached`将资源归还QNN；关闭会话才最终释放context。仅提前映射memo的中间版本仍曾在首次VTCM申请时返回15，因此最终同时处理两种布局。归并Native的sequence-sized输出与QNN的AR128 chain改用独立fd，消除同一fd注册1MiB/8MiB长度冲突；增加约1MiB工作缓冲。映射缓存只提交有效指针，所有host注册由core统一跟踪并清理。

12轮1K文本与3轮图文完整会话未再出现25、15或清理错误，每轮85份prefill产物和完整生成token均与原版一致。1K prefill为413.56–429.42 token/s，图文为139.05–140.70 token/s，维持原性能范围。AR128/AR1024失败注入确认：旧版同handle重试仍报25，新版能重新映射并正常释放。本轮未改量化、Top8或计算结果，无需重新导出PTE；host runner与prefill skel需配套更新。系统日志尚不能进一步区分VA碎片、页表或底层映射器限制，有限次数通过也不等于任意系统负载下的保证。完整证据、回归、源码指纹与二进制见[错误25定位报告](../../../../../llama-output/qwen35-moe-from-safetensors-gptq-20261024/prefill-map25-20261025/REPORT.md)。

最终排版重建后另补三对原版/修复版回归，覆盖128单块、512归并与chunk-major图文两轮对话，产物和token仍逐字节一致。修复版共18次整模初始化通过，最终配套二进制已留在SM8975测试目录；这些初始化修复已提交为 `858046804b`。

### Decode 提前预测专家 IO（历史试验，CPU 路径已撤回）

2026-10-26，按用户要求模型计算只使用 NPU/DSP，已撤回 CPU INT8 近似路由的运行、准备及部署入口。以下是当时的试验记录；包含 CPU 预测的吞吐不作为当前纯 NPU/DSP 版本的性能基线。异步专家读取、命中先算与 LRU 继续保留。

2026-10-25，初始化错误25修复提交后，继续拆分真实40层AR1。保持每层32槽LRU、原全矩阵GPTQ、Top8、UINT16 state/KV、容量1152与同一QNN图。使用每层Attention输入的CPU INT8近似路由，在Attention之前读取最多两个预测缺失专家到独立scratch；真实路由出来后只复用匹配的读取future，最终复制到实际LRU槽并发布ready。命中专家立即计算，预测错误不驱逐LRU，也不改变实际Router/shared计算与求和。

| 输入 | 关闭提前预取 | 启用提前预取 |
| --- | ---: | ---: |
| 1K文本后57步decode，4轮中位数 | 101.18 ms/token / 9.88 token/s | **87.65 ms/token / 11.41 token/s** |
| 256-token图文后78步decode，3轮中位数 | 87.27 ms/token / 11.46 token/s | **77.97 ms/token / 12.83 token/s** |

文本主阶段为：RPC外处理含预测 **2.62→5.57ms**、Native含剩余等待 **54.05→37.28ms**、Attention融合段含Router/shared **37.97→37.60ms**、路由交接 **0.05→0.06ms**、state/head/采样等 **6.48→7.14ms**。偶数轮采用整轮耗时排序的中间两轮平均，分项取同两轮，保持可加；第三轮基线发生201.92ms/token读盘停顿，原始数据完整保留，未删除异常值。

独立Native profile中，最慢worker专家就绪等待累计 **43.73→30.49ms/token**，HMX活跃计算约0.45ms。真实LRU命中率不变；文本总请求读取量 **99.83→112.23MB/token**，约增加12.43%，以额外预测读取换取计算/IO重叠。辅助矩阵20.039MiB，新增scratch/staging分配约4.922MiB；该数字不是整机内存峰值实测。

14轮普通回归的完整生成ID与逐token实际LRU统计一致，另有1K输入的最终hidden、logits和40×2 state/KV共 **82份文件逐字节一致**。最终保留提前预测读取；集中轮询、Linux AIO和线程数调优没有稳定收益，均恢复原默认。GDN融合图约30.24ms、full-attention融合图约7.27ms仍是后续分析重点，Router/shared已包含其中，不能重复加历史独立图时间。

该试验当时由 `quantization/prepare_early_router.py` 生成辅助矩阵。当前生成、部署和运行入口均已撤回；残留的 `early-router.bin` 不会被新版读取。旧二进制仍可用空变量 `EARLY_ROUTER_WEIGHTS=` 关闭CPU预测做历史对照。完整历史记录见[Decode拆分与提前预取报告](../../../../../llama-output/qwen35-moe-from-safetensors-gptq-20261024/decode-latency-20261025/REPORT.md)。

### Decode 读盘长停顿待查

历史对照中出现过连续 token 的 Native 500–760 ms 长停顿；六轮诊断和一轮普通复测未再次出现，尚未确认根因，也未标记修复。2026-10-25 按用户要求先保留现场，后续再查，本轮转到计算侧优化。[记录与后续安排](../../../../../llama-output/qwen35-moe-from-safetensors-gptq-20261024/decode-io-stall-20261025/REPORT.md)。


### 关闭 CPU 预测后的完整性能与内存验证

2026-10-26，按用户要求撤掉 CPU INT8 近似路由点积，Attention/真实 Router/shared 仍由 QNN 执行，Native MoE 矩阵计算在 DSP/HMX。保留每层32槽、Top8、异步读盘、命中先算、持久 context/state；量化和 PTE 不变，静态容量1152、prefill AR128。上一轮含 CPU 辅助的约11.41 token/s属于历史方案，不作为本轮基准。

下表两列都禁用CPU预测；性能各两轮交替测试，内存另跑一次200ms采样。Prefill计时包含专家IO，排除初始化、图片预处理和encoder；decode不计prefill产生的首token。两个输入都使用视觉context常驻的图文部署。

| 输入 | 指标 | 优化前 | 优化后 |
| --- | --- | ---: | ---: |
| 1024文本 | Prefill | 2.412秒 / 424.61 token/s | 2.415秒 / 424.05 token/s |
| 1024文本 | Decode | 98.048 ms/token / 10.199 token/s | 98.090 ms/token / 10.195 token/s |
| 1024文本 | 全流程峰值 | 4.994724 GiB | 4.994785 GiB |
| 256图文 | Prefill | 1.818秒 / 140.81 token/s | 1.825秒 / 140.25 token/s |
| 256图文 | Decode | 86.267 ms/token / 11.592 token/s | 84.434 ms/token / 11.844 token/s |
| 256图文 | 全流程峰值 | 4.886219 GiB | 4.886658 GiB |

文本性能基本持平，图文decode本轮降低2.12%；两轮不足以认定小幅收益稳定。30个GDN融合图合计由30.671降至29.906ms/token，但Native含剩余IO等待仍约50.6ms，整轮没有同等幅度下降。GDN保留全部量化位，V85启用Karatsuba精确cross重构、双V块系数复用及O3。另修正了构建配置过早展开导致增量编译漏记V85追加参数的问题，标准构建已重新核验。

内存峰值均出现在prefill，DMA分配量完全相同；峰值CPU页差异仅0.0625/0.4492MiB。新版decode文本保持约4.860GiB，图文约4.816GiB，首尾仅增加0.039/0.305MiB，没有持续累积；SwapPSS均为0。退出时所有context/权重回调正常释放。200ms采样是观察峰值，不是任意瞬时分配的严格上界。

**该轮图文部署未满足4GiB预算。** 当时纯文本输入也加载视觉模块：峰值约459.84MiB的`vision.bin` CPU文件页，加约432MiB视觉权重DMA，尚不含视觉工作区及IO；encoder计算仍在NPU。初始化→prefill上升到约5GiB，交接后decode约4.8GiB。原始[内存时间曲线](../../../../../llama-output/qwen35-moe-from-safetensors-gptq-20261024/performance-memory-no-cpu-20261026/MEMORY_TIMELINE.png)保留；与历史纯文本3.867GiB的对账及修复见下一节。

8轮普通性能运行、4轮内存运行全部通过，生成ID和逐token真实LRU/IO相同；最终hidden、logits与40×2 state/KV共82文件逐字节一致。标准构建库另有一次完整1K回归，82文件仍一致。本轮未重新评估FP32 PPL。全部轮次、阶段明细、峰值fdinfo/smaps、配置与二进制SHA见[完整报告](../../../../../llama-output/qwen35-moe-from-safetensors-gptq-20261024/performance-memory-no-cpu-20261026/REPORT.md)。

### 32槽内存回归对账与修复

2026-10-26，历史3.867GiB使用无视觉的纯文本runner，近期约4.995GiB则在纯文本输入时也创建了视觉context；两者比较口径未对齐。缓存仍为40×32槽、1,101,168,640字节，两阶段复用同一批专家fd，没有变成64槽。

已修复两处：非交互纯文本不再因目录中存在`vision.bin`而加载视觉模块；prefill每构建一层就绑定公共激活IO，避免40层临时私有IO在初始化同时存在。state/KV仍逐层私有，图片及可换图的交互会话仍将视觉context与整场会话一起创建、一起释放。

| 输入/部署 | 历史32槽峰值 | 近期修复前 | 修复后 |
| --- | ---: | ---: | ---: |
| 128非交互纯文本 | 3.901541 GiB | — | **3.830641 GiB** |
| 1024非交互纯文本 | 3.867239 GiB | 4.994785 GiB，误加载视觉 | **3.984724 GiB** |
| 256图文 | — | 4.886658 GiB | 4.886780 GiB |

1K修复前后减少**1034.303MiB / 1.010061GiB**：CPU PSS减少460.010MiB，去重DMA减少574.293MiB，后者包含视觉权重、工作区和IO。仅移除视觉的先导运行还捕获4.153GiB初始化峰值；逐层绑定共享IO后，完整峰值降至上述3.985GiB。

1K仍比历史多**120.305MiB**，没有完全回到3.867GiB。DMA净增255.625MiB，包含映射到DMA的210MiB memo、1K Native归并的66MiB DSP分配及32MiB IO，减去取消的52.508MiB pipeline bank等；CPU PSS净减135.320MiB。历史由主机准备输入，当前终端还含分词等工作内存，不能只比较权重或`rpc_buffer_bytes`。

普通性能复测：1K prefill **430.01 token/s**、decode **9.956 token/s**；图文prefill **139.71 token/s**、decode **11.551 token/s**。本轮各一次，上一轮均值分别为424.05/10.195及140.25/11.844，差异包含I/O波动，不据此认定稳定提速或回退。生成ID、真实LRU/IO及82份hidden/logits/state文件与修复前完全一致；纯文本开始的交互会话后续换图也已通过，视觉执行中无context/buffer重建。

统计仍为CPU PSS＋去重DMA－PSS内DMA，200ms采样，SwapPSS为0。1K纯文本距4GiB预算仅约15.64MiB，不能保证其它输入也不超限；**图片或支持后续换图的交互会话仍超4GiB，不能引用非交互纯文本峰值。** 明细、曲线、现场和复跑脚本见[内存回归报告](../../../../../llama-output/qwen35-moe-from-safetensors-gptq-20261024/memory-regression-20261026/REPORT.md)。

### 继续压缩常驻图文内存

2026-10-26，保持全部40层、每层32槽、现有量化与视觉context常驻，完成三项优化：视觉权重通过QNN回调直接读入DMA，回收不再需要的CPU binary驻留页；GDN交换state句柄时不再申请无用的CPU中转数组；SDK堆增长粒度设为4MiB。

| 场景 | 优化前峰值 | 优化后峰值 | 减少 |
| --- | ---: | ---: | ---: |
| 256-token图文 | 4.886780 GiB | **4.382659 GiB** | 516.220 MiB |
| 1K文本＋视觉常驻 | 4.994785 GiB | **4.476583 GiB** | 530.639 MiB |
| 1K纯文本 | 3.984724 GiB | **3.893999 GiB** | 92.902 MiB |

1K视觉常驻使用交互会话初始化，输入为1024个文本token，视觉塔可供后续换图；不是1K图文性能样本。CPU PSS＋去重DMA的统计公式保持一致，SwapPSS均为0。完整含视觉仍超过4GiB预算，没有把纯文本峰值作为图文结果。

原视觉binary的CPU PSS约459.844MiB，优化后28.098MiB，431.750MiB视觉权重DMA继续保留。SDK权重回调现在使用本模块Buffer，`rpc_buffer_bytes`会把原先SDK内部申请的视觉DMA计入，因此该局部计数会增加；完整峰值反而下降。GDN去掉约60MiB无用中转载荷，网格不一致的conv/state转换仍保留。调整堆增长粒度实测只省18MiB，未将全部32MiB块当成空闲。

1K含视觉的最终峰值可对账为：层组/head权重1255.703MiB、专家缓存1050.156MiB、视觉权重431.750MiB、专家memo210.004MiB、其它DMA1373.871MiB、CPU PSS262.537MiB。其它DMA包含state/KV、IO、SDK/DSP堆和中间工作区；SDK编译元数据中各层DDR张量合计430.986MiB，视觉另10.442MiB，算子运行数据合计148.674MiB。这些是已有分配的说明，不能再加到峰值上。

同轮图文无采样性能：prefill **138.96→140.91 token/s**，decode **11.629→11.507 token/s**；encoder **120.71→118.68ms**。各一轮，不能认定百分之一左右的差异是稳定收益或回退。最终1K纯文本为430.91 token/s prefill、10.074 token/s decode。

文本及图文各82份hidden/logits/state文件与优化前逐字节相同，生成ID、真实LRU/IO相同。连续两次图片编码121.22/121.21ms，整场视觉权重只provide一次、退出release一次，执行中context/Buffer不重建。当前设备库拒绝VA优化配合权重回调，以及异步恢复配合deferred graph；持久binary提示和同优先级资源模式没有减少DMA，均未保留。完整分解、SDK元数据、试验记录及源码指纹见[常驻图文内存优化报告](../../../../../llama-output/qwen35-moe-from-safetensors-gptq-20261024/memory-optimization-20261026/REPORT.md)。

## 代码位置

```text
qwen35-35B-A3B/
├── export.py                 # 配置驱动的唯一导出入口
├── run_chat.py               # 推理入口和run_inference函数
├── build.sh                  # Android runner及Native库构建
├── configs/                  # 导出、量化、设备与调度JSON
├── export/
│   ├── shared.py             # 编译时共享prefill/decode权重
│   ├── capture.py            # 导出内部的SDK权重范围采集
│   ├── weights.py            # 图/权重文件布局分离，无独立CLI
│   ├── program.py            # 导出内部的完整PTE序列化
│   ├── head.py、vision.py    # head及视觉编译
│   ├── runtime_assets.py     # 实际cache编码及RoPE
│   └── model.py、graph_utils.py、prefill.py等
├── workflow/
│   ├── model_config.py       # 模型配置及静态图契约
│   ├── target.py             # 复用qc_schema，筛选V79/V81/V85
│   ├── export_config.py、export_pipeline.py # 单命令串行导出与续跑
│   ├── config.py、artifacts.py # 推理参数和产物校验
│   ├── plans.py              # 读取现成PTE，不执行打包
│   ├── vision.py、prompts.py  # 元数据校验及离线诊断输入
│   └── device.py、deployment.py、inference.py、session.py、diagnostic.py等
├── runner/
│   ├── chat/                 # 设备分词、图片解码、MRoPE、多轮和吐字
│   ├── prefill/、decode/     # 模型计算、缓存与专家IO
│   ├── shared/               # 常驻共享context和权重DMA回调
│   └── tests/                # 主机回归
├── native/
│   ├── build_op_package.sh   # GDN包内部构建
│   ├── prefill/、decode/     # HVX/HMX内核与FastRPC
│   └── tests/                # 锁与映射错误注入回归
├── tools/                    # 设备权重范围和cache元数据工具
├── quantization/             # safetensors、GPTQ与专家重构
└── tests/                    # 导出与推理流程回归
```

GDN recurrence 复用[共享 DeltaNetQhpi.cpp](../../../custom_op/deltanet_op_package_htp/DeltaNetOpPackage/src/ops/DeltaNetQhpi.cpp)，本目录不复制该 op。decode 选择已有 DDR kernel；prefill 沿用已有 AR128 图。原 PTE 的 `Q36DECO1`/`Q36PREF1`、`Qwen36*Stage` 名称作为兼容 ABI 保留。

### Python 流程

参考 `llama.py` 的参数解析、产物准备、设备推理与评估职责划分，`run_chat.py` 只串起 `parse_args → check_artifacts → inference`。导出使用 `export.py --config`，由 `workflow/export_pipeline.py` 串行编排校准、QNN 编译、权重分离和 PTE 打包；各内部阶段保留单独调试命令。`--model-config` 在各步骤使用同一配方，修改 GPTQ 参数需要重新校准，修改静态接口或容量需要重新导出。运行入口只消费导出阶段已经生成的PTE及共享权重，根据实际prefill长度选择计划。

`workflow/inference.py` 显式编排输入准备、部署、执行和结果校验；`DevicePaths` 统一管理 prefill、decode 和共享模型路径。独立权重的逐层目录由 `decode_layer_dir` 统一解析，供校验和部署复用。共享权重模式直接校验十组共享产物及 120 份 IO 顺序文件，不要求旧的独立 Attention binary 留在本地。

`AndroidDevice` 封装 ADB 与快照，流式读取异常时回收本次主机 ADB 子进程；`inference` 在失败时也保存 `RUN.json` 的错误信息。`results` 集中检查 handoff 首 token、输出顺序、EOS、greedy feedback 和 teacher forcing。`--help`、`--check` 不初始化模型或调用 ADB。

2026-10-24 回归：32 组主机部署/命令与快照逐项一致；SM8975 上 128 / 1024 输入各 86 份数值文件与重构前逐字节一致。128 输入 prefill 为 81.56 → 81.62 token/s、decode 为 10.76 → 10.81 token/s。单次计时波动不作为稳定加速结论。快照提交为 `3e8e8dc1fd`，完整记录见[架构整理报告](../../../../../llama-output/qwen35-moe-from-safetensors-gptq-20261024/history-reports/qwen35-moe-architecture-20261024/REPORT.md)。

### C++ 接口与实现

`runner/prefill`、`runner/decode` 下的大型方法位于同名 `.cpp`，头文件保留类型、接口和简单访问器。QNN 图和 buffer 管理在 `qnn_front.cpp`，状态更新在 `attention.cpp`，完整调度在 `prefill_session.cpp` / `decode_session.cpp`，专家读取和缓存各自独立编译。公共计时、文件读取统计及交接指纹在 `runner/*.cpp`。

模型入口为 `prefill_handoff_runner`，decode 编译为 `libmoe_decode.so`，通过唯一 C 入口 `qwen35_decode_run` 在同一地址空间执行；不再通过 `exec` 替换进程。动态库隐藏内部 ExecuTorch、RPC stub 和 C++ 符号，避免与 AR128 的接口及 backend 注册表冲突。AR1 的 DSP 库部署为 `libinteger_decode_rpc_skel.so`，区分两种 Native ABI。阶段切换只关闭 AR128 Native worker 并交接专家 buffer 的映射；视觉和 prefill 的 QNN context、buffer、PTE 保留到 decode 结束后统一回收。

`native/*/integer_rpc_impl.cpp` 与 `IntegerMoE.cpp` 独立编译，通过 `integer_moe.h` 对接。N2 展开在 `native/N2Utils.cpp`，后台 bank 映射在 `native/decode/bank_mapper.cpp`。保留内联的主要是 HMX 指令包装、原子操作和简单访问器。可变全局状态只在一个 `.cpp` 定义，头文件使用 `extern` 声明。

资源锁申请使用每次调用独立的 `WorkerStart`：8 个 HVX worker 和 1 个 HMX worker 都报告结果后，才进入计算。任一锁失败，全组跳过计算，成功拿锁的 worker 在自己的线程释放锁，并全部发送完成通知。prefill 的 Native 调用抛异常时，也会先等在途读取结束，保护请求数组的生命周期。

## 当前模型约束

- 40 层，30 层 GDN＋10 层 full attention，hidden=2048。
- 256 专家/层、Top8、专家 intermediate=512。
- GDN：16 个 key head、32 个 value head、K=V=128，UINT16 state `[1,32,128,128]`。
- 专家是四级码本 `[-7,-2,2,7]` 的 **2 bit 存储**，在 VTCM 展开为 N4×A8 HMX 计算；不是原生 W2 HMX 指令。GU group=1024、down group=512，bank 版本为 7。
- Router/shared 默认为 GPTQ W8A16，可选择旧 FP16 图；embedding 默认为 INT4，可选择原 BF16 表。两种选择都检查实际产物，不在运行时自动重量化。
- GDN/conv/KV cache 目前仅支持 UINT16；配置中显式保留三个 dtype，UINT8 在当前 MoE runner 上会报错。
- 已验证静态 decode 容量为 1152、2048；prefill 固定 AR128，支持 128 到 1024 的完整块。文本基准使用 128/1024 两种长度，使用原生 chat template、关闭 thinking。支持原始 token 文件输入。
- 支持单图输入，默认 448×448、196 个视觉 token、视觉塔 W8A16。设备 runner 内部完成图片解码、前处理、视觉编码和完整 MoE 推理；Python 入口保留原主机前处理路径。
- 当前不支持视频、多图、任意模型尺寸和交互式多轮聊天。

本机模型目录名称为 `Qwen3.6-35B-A3B`，其 `config.json` 的 text architecture 是 Qwen3.5 MoE。目录名称通过配置指定。

## 构建

使用已有的 ExecuTorch Android Release 静态库（默认 `build-android`）。需要 QNN 2.48.1、Hexagon SDK 6.7.0.0 和包含 V85 支持的 Hexagon Tools。

```bash
conda activate et_npu
export QNN_SDK_ROOT=/hunter/qnn_sdk/src/backends/qnn/2.48.1.260715
export HEXAGON_SDK_ROOT=/hunter/qnn_sdk/hexagon/6.7.0.0
export HEXAGON_TOOLS_ROOT=$HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/21.0.04/Tools
export ANDROID_NDK_ROOT=/hunter/android-ndk/android-ndk-r28c
export LLAMA_CPP_ROOT=/hunter/cpp_code/ET/npu/llama.cpp

bash examples/qualcomm/oss_scripts/moe/qwen35-35B-A3B/build.sh
```

`LLAMA_CPP_ROOT` 提供已使用的 Qualcomm fork 的 HVX sigmoid 辅助头文件。构建不读取历史实验目录的源码。GDN 直接从仓库共享 op 源码构建，V85 Make 规则在 `native/gdn_v85.mk`。

默认输出到 `build/moe/qwen35-35B-A3B`，可用 `MOE_BUILD_DIR` 覆盖。已有 Android 库目录可用 `EXECUTORCH_ANDROID_BUILD` 覆盖。不同 SDK 重建共享 Python binding 时使用仓库的 `backends/qualcomm/scripts/build.sh`。

锁失败回归测试：

```bash
# 仅主机并发测试，不需要 SDK 或设备。
bash examples/qualcomm/oss_scripts/moe/qwen35-35B-A3B/native/tests/run.sh

# 先完成 build.sh，并设置上述 SDK 环境；额外构建测试库，在 SM8975 注入真实锁错误。
bash examples/qualcomm/oss_scripts/moe/qwen35-35B-A3B/native/tests/run.sh A7SR026915000282
```

故障只通过独立测试库的链接选项 `--wrap` 注入，正式 Native 库没有注入分支。设备测试在专用临时目录运行；每个 handle 先执行一次注入失败的普通调用，再执行一次资源查询，验证错误返回、线程复用和关闭路径。

## 运行

`configs/sm8975.json` 是当前本机已部署产物的配置：模型路径、host artifacts 目录、设备 PTE 和外存专家路径都在这里。历史目录仅作为模型产物输入，不再从中载入 Python/C++ 源码。PTE、权重、构建输出和日志均外置。

先离线检查数据及构建产物：

```bash
python examples/qualcomm/oss_scripts/moe/qwen35-35B-A3B/run_chat.py --check
```

在已配置 SM8975 上运行，实时显示每个 token 和间隔：

```bash
PYTHONPATH="$(dirname "$PWD")" python -u \
  examples/qualcomm/oss_scripts/moe/qwen35-35B-A3B/run_chat.py \
  --prompt "你好，请用中文介绍一下自己，再告诉我1加1等于几。" \
  --max-new-tokens 128
```

1024 输入、默认容量 1152，生成最多 128 个 token：

```bash
PYTHONPATH="$(dirname "$PWD")" python -u \
  examples/qualcomm/oss_scripts/moe/qwen35-35B-A3B/run_chat.py \
  --prompt-tokens 1024 --max-seq-len 1152 \
  --prompt "什么是注意力机制？" --max-new-tokens 128
```

可以用 `--config` 选择设备/产物路径，`--model-config` 选择模型设置，CLI 覆盖本次运行参数。共享模式按实际输入选择导出目录里的现成PTE；也可显式提供 `--local-prefill-pte`、`--local-decode-pte` 及其 JSON sidecar，运行前校验图、IO 顺序、KV 编码和静态配置。`--runtime-device-dir` 指向设备上已部署的 QNN runtime/skel 等依赖，bank 从 `--remote-weights` 读取。

运行会部署本目录构建的 runner、Native MoE 和 GDN 包，并加载 `--artifacts/assets` 的融合图。两个完整 PTE 保留兼容调度容器，decode 的逐层 Attention/Router 由融合 binary 覆盖。不能用普通 llama runner 直接运行这套 PTE。

结果默认写入 `llama-output/qwen35-35B-A3B`，用 `-a` 指定其他目录。`--text-only` 仅显示增量正文，`--trace-layers` 开启逐层计时。正常聊天遇 EOS 停止；`--teacher-tokens`、`--save-all-logits`、`--ignore-eos` 用于显式诊断。

### 图片编码与图文推理

`--vision-artifacts` 指向 `export/vision.py` 生成的目录，默认由 `configs/sm8975.json` 的 `vision_artifacts` 指定。运行入口部署同一编码器的 `vision.bin`；`vision.pte` 同时保留用于检查和其他部署方式。

```bash
PYTHONPATH="$(dirname "$PWD")" python -u \
  examples/qualcomm/oss_scripts/moe/qwen35-35B-A3B/run_chat.py \
  --image-path "/home/hunter/下载/test.jpg" \
  --prompt "请用中文描述这张图片，简短回答。" \
  --max-new-tokens 128
```

448×448 图片先变成 `[1,784,1536]` patches，经设备视觉图生成 `[1,196,2048]` FP32 特征。prefill 的 embedding 阶段按 image token 的出现顺序直接替换对应行，跨 AR128 块继续消费同一份特征。视觉图与全部 prefill 图共用 Runtime，加入同一组异步 context 初始化；像素加载也在初始化阶段完成。编码后不释放视觉 context，穿过 decode 阶段继续常驻，整次模型会话结束才回收。`VisionEncoder::load_image` 可更换同分辨率输入，`encode` 复用已有 context 和 IO。

运行时按实际图文 token 构造三维 MRoPE，decode 继续使用图片压缩位置带来的偏移。短图文输入自动选取最小的 AR128 整数倍，当前样例为 256 token；为了复用完整递推 128 步的现有图，在 system 段补真实空白 token，避免末尾 padding 污染最终 state/logits。输入 token 计数包含这些空白。可显式指定 `--prompt-tokens 1024`，不足以容纳图片和文字的长度会报错。切换到 2048 容量还需要匹配的共享图产物。

日志 `PTE_READY.setup_ms` 包含模型及视觉图的初始化；`VISION_ENCODE` 统计常驻图执行、输出复制和有限值检查，附带 context/IO 创建次数，均应为 0。`MODEL_RESIDENT` 校验 decode 返回后视觉和 prefill context 仍然存活。`PREFILL_TOTAL` 与 `DECODE_LOOP` 分别统计语言模型两阶段，特征保存放在主循环后。结果目录额外保存 `pixels.f32`、`image-features.f32`、`mrope-positions.i32` 和本次使用的 cos/sin 表。

### 单阶段调试：视觉编码器

仅加载 checkpoint 中的 `model.visual.*`，不会把 35B 文本权重加载到主机内存。复用现有 Qwen3.5 静态视觉实现、W8A16 配方与 INT32 bias 保护；每通道 W8 Linear，使用真实图片在 GPU 校准。使用与导出融合图相同的 SDK 环境：

```bash
PYTHONPATH="$(dirname "$PWD")" python -u \
  examples/qualcomm/oss_scripts/moe/qwen35-35B-A3B/export/vision.py \
  --model /hunter/model/Qwen/Qwen3.6-35B-A3B \
  --output llama-output/qwen35-moe-vision-sm8975-20261023 \
  --qnn-binding llama-output/qwen36-decode-optimization-sm8975-20261023/export/PyQnnManagerAdaptor.cpython-311-x86_64-linux-gnu.so \
  --calib-images /path/to/calibration-image-1.jpg /path/to/calibration-image-2.jpg
```

输出包含 `vision.pte`、`vision.bin`、`vision-qdq.pt2`、`VISION.json` 及 FP32/QDQ 参考数据。导出检查视觉图只有一个 QNN delegate，输入输出各一个；运行前检查模型配置指纹和分辨率。`--image-height`、`--image-width` 可选择其他 32 的整数倍分辨率，需要重新导出对应图。

## 完整 PTE 容器与专家重构

以下是历史独立图记录。当前完整执行计划由 `export/program.py` 在统一导出内部生成，每个计划122个阶段，运行时不再手动打包。历史输入目录格式：

- decode：`layerN/attention.bin`、`io.bin`、`moe-front.bin`。
- prefill：`layerN/forward_0.bin`、`moe-front.bin`；full-attention 另需 `cold/layerN/forward_0.bin`。


此历史分步命令已移除；当前统一执行 `export.py --config configs/export_sm8975.json`。


prefill 使用 `--phase prefill` 和对应 AR128 head，不传 `--cache-encoding`。`--prefill-tokens` 默认 128；设为完整输入长度可启用下文的按层复用，QNN/Native 内核仍为 AR128。逐层 QNN 编译和容器包装在内部是两个步骤；正常流程由 `export.py --config` 自动串联。本节的独立图命令用于旧产物调试。

专家重构入口：

```bash
PYTHONPATH="$(dirname "$PWD")" python -u \
  examples/qualcomm/oss_scripts/moe/qwen35-35B-A3B/quantization/prepare_experts.py \
  --model /path/to/Qwen3.5-35B-A3B \
  --source-dir /path/to/calibrated-prefill-layers \
  --inputs /path/to/INPUTS.pt --output /path/to/reconstructed-experts
```

该步骤需要 CUDA，逐层运行，默认每批 8 个专家。`INPUTS.pt` 包含 `calibration:[8,1024,2048]`、`heldout:[2,1024,2048]` 两个 FP32 embedding tensor。source-dir 包含每层的 `front-qdq.pt2`、`moe-constants.npz`、`reference.pt` 以及 `cos.f32`、`sin.f32`。沿用原来的 8192-token 校准流程，7 条拟合、1 条选择方案，2 条 heldout 不参与拟合；继续保留完整专家输出重构及逐层量化误差传播。

## embedding 与 Router/shared 量化

当前默认选择 GPTQ W8A16 Router/shared 和 group-32 INT4 embedding。INT4 每组保存 16 字节 code 和两个 FP16 scale/bias，不展开整张表。两阶段分别 mmap 同一个文件，生成时只解码当前 token 的一行。图片特征继续直接替换 image token 的 embedding。旧 BF16/FP16 方案需要在 `--model-config` 中选择对应配方，并提供匹配产物。

```bash
python examples/qualcomm/oss_scripts/moe/qwen35-35B-A3B/quantization/prepare_embedding.py \
  --source /path/to/embedding.bf16 \
  --calibration-inputs /path/to/INPUTS.pt \
  --output /path/to/embedding.int4
```

embedding 的输入为 one-hot token 索引，标准 GPTQ 的 Hessian 是 token 频率对角阵，跨列误差补偿为零；固定网格下等价于逐行最近舍入。这里明确记录该退化形式，不声称有普通 Linear GPTQ 的补偿收益。输出 JSON 同时记录整表、校准 token、heldout token 的 SQNR；heldout 不参与拟合。文件从 970 MiB 降为 303.125 MiB，实际运行内存须单独采样，不能用文件差值代替。

Router/shared 的五个投影使用真正的固定网格 GPTQ：per-channel W8、UINT16 激活，保留观察器产生的 scale/zero point，沿输入通道做 Hessian 误差补偿。shared down 投影使用已量化 gate/up 的实际中间激活校准。校准输入来自已有逐层 W2 基准的 8 条 1024-token 序列，其中 7 条拟合、1 条留出；另外 2 条 heldout 只评估，不拟合。Attention 图及原 W2 专家保持不变；这一步不重新重构 routed 专家。

```bash
python examples/qualcomm/oss_scripts/moe/qwen35-35B-A3B/quantization/prepare_router.py \
  --source-dir /path/to/calibrated-prefill-layers \
  --sequential /path/to/reconstructed-experts/sequential \
  --inputs /path/to/INPUTS.pt --output /path/to/router-qdq \
  --qnn-binding /path/to/PyQnnManagerAdaptor.so


```

校准按层串行处理，默认使用 CUDA；共享编译使用 `export/shared.py --router-qdq ...`，统一入口随后自动分离图/权重文件。导出检查五个真实 INT8 投影网格及校准报告中的 GPTQ damping，防止配置修改后继续误用旧 QDQ。

旧独立权重模式下，完整 W8 图整批恢复 decode 曾触发 DSP 装载错误，需要 `--decode-context-load-mode serial`。当前共享模式采用 SDK 回调批量恢复 11 个 context，两个阶段都复用这些实例，循环内不创建/销毁。装载方式与逐 token 的专家异步读取是不同设置。

## 测试记录

2026-10-23，回退到已验证的独立权重版本，撤下未通过整模型设备验证的 P/D 常量共享、head/IO 共享及配套 INT4 embedding 实验入口。两个 ARM runner 重新构建后与原已验证二进制的 SHA256 一致，Native MoE 和 V85 GDN 包也重新构建通过。同一张 test.jpg、256-token 图文输入生成相同的 67 个 token（含 EOS），176 份数值文件与原独立权重基准逐字节一致。循环内 context 创建/释放、IO 注册/注销及 RPC buffer 分配/释放均为 0。本轮 prefill 为 2.348 秒，decode 为 14.21 token/s，仅作为回退验证，不据此判断性能提升。核对记录见 [EQUIVALENCE.json](../../../../../llama-output/qwen35-moe-independent-rollback-20261023/EQUIVALENCE.json)。

SM8975，W2/64 槽，同一 1024 输入和 127 步 teacher-forced AR1，容量 2048，前后 thermal status 均为 0：

| 环节，40 层累计 ms/token | 优化前 | 优化后 |
| --- | ---: | ---: |
| Attention＋Router/shared | 53.690 | 34.467 |
| Native MoE，含缺失专家等待 | 39.920 | 29.957 |
| RPC 外读取 | 1.695 | 1.613 |
| State、head 及其它 | 6.713 | 5.893 |
| 完整 decode | 102.018 | 71.930 |
| token/s | 9.802 | 13.902 |

短 WikiText（128 输入＋127 次 teacher forcing，共 128 个预测目标）：PPL 6.130877→6.229842，最后 hidden SQNR 6.187→6.131 dB。FP32 参考 PPL 为 4.014350。这不是完整 WikiText PPL 或 1024 个真实生成 token 的长测。

迁移验证（2026-10-23）：从本目录源码重新构建两个 runner 和 Native MoE 后，中文 greedy 测试生成 31 token（含 EOS），正确回答“1加1等于2”，178 份数值产物与迁移前逐字节一致。循环内 context 创建/释放、IO 注册/注销及 RPC buffer 分配/释放均为 0。

随后使用新导出入口重新编译 GDN 第 0 层、full-attention 第 3 层，均只有一个 QNN delegate；替换这两层并使用本目录构建的 V85 GDN 包运行完整模型，178 份数值产物再次逐字节一致。结果及 `MIGRATION_CHECK.json` 保存在 `llama-output/qwen35-35B-A3B/20261023-094744-944835-w2a8-fused-c1152-p128-cache64/`。该短测 decode 为 11.90 token/s，输入为 128 token，与上表的 1024-token teacher-forced 性能口径不同。本次迁移未重跑全部 40 层校准及 PPL。

C++ 分离编译和锁失败修复验证（2026-10-23）：

- ARM runner、V85 Native MoE 及 GDN 包重新构建通过；25 个 ARM 头文件可独立包含编译。
- 主机 163 组并发测试通过，覆盖全部 HVX/HMX 锁失败位置、多重失败和后续正常调用。
- 设备 prefill/HVX、prefill/HMX、decode/HVX、decode/HMX 四种注入均返回 `1073`，同一 handle 后续资源查询和关闭均返回 `0`。包含失败、查询、关闭的 probe 耗时分别为 10.838、10.787、6.153、4.843 ms；该值不是模型计算耗时。
- 完整模型同一中文输入生成 31 token（含 EOS），178 份数值产物与拆分前逐字节一致。结果位于 `llama-output/qwen35-35B-A3B-cpp-split/20261023-103658-974598-w2a8-fused-c1152-p128-cache64/`，核对记录为 `SPLIT_CHECK.json`。短测 decode 为 12.17 token/s，不据此推断性能提升。

### 分离编译与锁修复前后性能复测

2026-10-23，在同一 SM8975 上串行交错测试，每版本、每输入长度各 2 轮。前后使用相同 PTE、专家权重、token 和 64 槽 LRU；decode 容量统一为 2048，prefill 固定 AR128。每轮第一个输出来自 prefill，随后执行相同的 127 步 teacher-forced AR1。以下为两轮平均耗时及总 token 数除以总耗时得到的吞吐，初始化另计。

| 输入 | 拆分前 prefill | 当前新版 prefill | 拆分前 decode | 当前新版 decode |
| --- | ---: | ---: | ---: | ---: |
| 128 token | 1.525 s / 83.92 token/s | 1.416 s / 90.39 token/s | 13.30 token/s | 13.33 token/s |
| 1024 token，8×AR128 | 10.379 s / 98.66 token/s | 10.254 s / 99.86 token/s | 13.88 token/s | 13.85 token/s |

Decode 平均耗时变化分别为 −0.24% 和 ＋0.19%，处于本轮复测波动范围内。128 输入旧版两轮 prefill 为 1.633 / 1.417 s，新版为 1.387 / 1.446 s；主要差异在专家读取及反馈时间，不能把首轮差距归因为分离编译加速。1024 输入新版 prefill 平均耗时下降 1.20%。本轮未观察到明显性能退化。

全部运行前后 thermal status 均为 0；各输入长度下，每轮 178 份数值文件与旧版基准逐字节一致，专家读取字节数、路由及缓存命中计数一致。prefill/decode 循环内 context 创建/释放、内存注册/注销和 RPC buffer 分配/释放均为 0。

逐轮原始结果、decode 时间分解、输入与构建指纹见[完整报告](../../../../../llama-output/qwen35-moe-from-safetensors-gptq-20261024/history-reports/qwen35-moe-cpp-refactor-performance-20261023/REPORT.md)及同目录 `RESULTS.csv`、`MANIFEST.json`、`EQUIVALENCE.json`。

### 首次视觉编码接入验证（历史独立 Runtime）

2026-10-23，448×448 W8A16 视觉塔导出为一个 QNN delegate，实际在 SM8975 执行。用同一 checkpoint 的 Transformers FP32 独立视觉塔作参考，比较完整的 196×2048 图像特征：

| 图片 | 编码执行 | 编码总耗时 | 编码输出 SQNR | 256-token 图文 prefill | Decode |
| --- | ---: | ---: | ---: | ---: | ---: |
| 双猫，参与视觉校准 | 116.21 ms | 974.20 ms | 25.90 dB | 3.191 s | 14.48 token/s，127 步 |
| 本地 test.jpg，未参与校准 | 116.09 ms | 984.50 ms | 22.72 dB | 3.291 s | 14.46 token/s，66 步 |

双猫样例描述出了虎斑猫、粉红色沙发及遥控器，达到 128 个输出 token 上限。未校准样例描述出了白色公牛、橙衣男子、黑色背包和帐篷，67 个输出 token（包含 EOS）后正常结束。两次运行前后 thermal status 均为 0。输入/输出和路由不同，这两轮的吞吐不用于推断量化性能差异。

MRoPE 的位置和 delta 对照覆盖方形图片、矩形图片及 1024-token 图文输入，与 Transformers 一致；设备所用整张 cos/sin 表与 Transformers 的最大绝对误差均为 0。静态视觉 FP32 相对官方 FP32 的 SQNR 为 51.19 dB。新增视觉入口后的纯文本回归中，178 份数值产物仍与前轮基准逐字节一致，decode 为 13.32 token/s。

本次验证了单图编码和完整 MoE 生成链路，未重做语言模型图文校准或完整视觉问答基准。原始结果、精度数据和样例输出见[视觉接入报告](../../../../../llama-output/qwen35-moe-from-safetensors-gptq-20261024/history-reports/qwen35-moe-vision-sm8975-20261023/REPORT.md)。

### 视觉随模型常驻（当前默认）

2026-10-23，修复独立视觉 Runtime 和阶段 `exec` 后，使用同一 test.jpg 和相同 token 对照：

| 环节 | 原独立编码器 | 当前常驻版本 |
| --- | ---: | ---: |
| 视觉图执行 | 116.09 ms | 119.31 ms |
| 编码总耗时 | 984.50 ms，含单独装载/释放 | 124.51 ms，复用已初始化图和 IO |
| 模型初始化 | 4225.43 ms，视觉初始化另计 | 4765.02 ms，已包含视觉图 |
| 256-token prefill | 3290.56 ms | 3240.70 ms |
| Decode | 14.46 token/s | 14.14 token/s |
| ADB 进程启动到首 token，host 观测 | 8607.25 ms | 8246.77 ms |

编码的热路径消除了约 860 ms 的装载、释放和文件保存开销；视觉装载仍需在全模型初始化支付，不应把两种计时口径混为冷启动收益。原 W8A16 图执行约 116–119 ms，未更改量化或重新编译视觉图。

单个视觉 context 连续编码三次，仅创建 1 次、最后释放 1 次，三份结果逐字节一致；执行分别为 121.82、121.09、123.65 ms。完整模型在 decode 返回后仍持有 82 个 prefill/vision context，过程中释放次数为 0；随后统一释放 82 个，IO 注册/注销各 474 次。新增 GDN state buffer 交换仅在布局和量化网格相同时启用，反馈累计时间由 122.51 降到 78.14 ms。

本轮图文 183 份数值文件、纯文本 178 份数值文件均与原基准逐字节一致；视觉 SQNR 仍为 22.72 dB。上述表格为单轮同输入对照，decode 的小幅差异不能据此判定为稳定退化或加速。

视觉生命周期修复后的上述基准仍需读取约 7.84 GB 专家数据，RPC 外读取处理占 2109.38 ms（约 65%）；后续 I/O 优化见下一节。尝试直接 O_DIRECT 读入 rpcmem buffer 返回 EFAULT，未纳入正式代码。专家 buffer 在两阶段共享，内存计数不能直接相加。完整分解、生命周期和数值证据见[常驻视觉报告](../../../../../llama-output/qwen35-moe-from-safetensors-gptq-20261024/history-reports/qwen35-moe-resident-vision-20261023/REPORT.md)。

### Prefill 按层复用专家

2026-10-23，新增覆盖完整输入的 PTE 调度计划：先处理同一层的全部 AR128 块，再进入下一层。同层用过的专家在临时 memo 中保留，后续块再次使用时跳过 UFS。Attention 仍按块顺序更新 state/KV，Top8、W2 权重、Native 累加顺序和最后的每层 64 槽 LRU 保持一致。

SM8975，真实 40 层，最终同 DSP 库对照：

| 输入 | 专家读取量 | 完整 prefill | 吞吐 |
| --- | ---: | ---: | ---: |
| 256-token 图文，2×AR128 | 7.836 → **5.106 GB** | 3.253 → **2.841 s** | 78.70 → **90.10 token/s** |
| 1024-token 文本，8×AR128 | 22.379 → **5.439 GB** | 10.102 → **5.774 s** | 101.37 → **177.35 token/s** |

读取量分别减少 **34.8% / 75.7%**，prefill 耗时分别下降 **12.7% / 42.8%**。最终实现另一轮为 2.973 / 5.779 秒。RPC 外读取处理分别从 2129.08 / 5990.62 ms 降至 1794.56 / 2166.50 ms，包含主线程等待、复制及同步。初始化预载仍是 2.192 GB，单列在初始化中；完整 prefill 不包含初始化、视觉编码或 decode。

需要额外 **210 MiB** 普通内存复用当前层的 payload，prefill 后释放；永久专家缓存仍是 40×64。PTE 工作区另增加约 4 MiB（256 输入）或 28 MiB（1024 输入），随模型持有到最后。128-token 单块计划不分配临时 memo。后续完整运行内存实测为图文约 7.04 GiB、1K 文本约 6.53 GiB，当前超过 4 GiB 预算，详见下节。

图文、1K 文本的最终两轮及旧 AR128 图兼容性对照，各有 **176 个数值文件逐字节一致**，覆盖 prefill/decode hidden、logits、全部层 state/KV、每块路由和 LRU 记录。直接从新 PTE 提取的 91 份 QNN 二进制与原版本 hash 一致。图文仍生成相同的 67 个 token（含 EOS）。本轮不改变已有 W2 精度，也没有重新评估 PPL。

启用方式：先构建当前 runner，再用原 AR128 图打包与实际输入长度匹配的计划。例如 256-token 图文：


此历史分步命令已移除；当前统一执行 `export.py --config configs/export_sm8975.json`。


1K 输入将打包长度设为 `--prefill-tokens 1024`，运行时选对应 PTE 并传 `--prompt-tokens 1024`。新计划使用 v2 header，需使用新 runner；原 AR128/v1 PTE 可继续运行。默认配置的 AR128 PTE 路径保持原值，选择新计划才会按层复用。

本轮没有修改 decode；短测图文 13.95–14.71 token/s，1K 前缀 9.48–11.29 token/s，尚不能判断稳定变化。完整分解、对照命令、数值核对及内存说明见 [I/O 优化报告](../../../../../llama-output/qwen35-moe-from-safetensors-gptq-20261024/history-reports/qwen35-moe-prefill-io-20261023/REPORT.md)。

### 完整运行内存峰值

2026-10-23，用最新按层复用计划在同一 SM8975 上重跑，以 200 ms 间隔采集 CPU PSS、SwapPSS 和 DMA-BUF。按 inode 去重，并扣除已经包含在 PSS 中的 DMA 映射；prefill/decode 共享的专家 fd 只计算一次。

| 配置 | Prefill 阶段峰值 | 全流程峰值 | 全流程峰值时 DMA / CPU PSS |
| --- | ---: | ---: | ---: |
| 256-token 图文，生成 67 token 含 EOS | **6.31 GiB** | **7.04 GiB**，decode 阶段 | 6.73 GiB / 325 MiB |
| 1024-token 文本，生成 32 token | **5.47 GiB** | **6.53 GiB**，decode 初始化 | 6.23 GiB / 309 MiB |

这是模型进程的采样占用，系统、其它应用及 ZRAM 压缩池另计。图文全流程约 7.56 GB，文本约 7.01 GB，均超过 4 GiB；这里 GiB 为二进制单位。专家缓存 **2.05 GiB 已包含在 DMA 中**。仅用日志里的 `peak_rpc_buffer_bytes`（2.62 GB / 2.44 GiB）会漏掉 QNN 内部的图权重、运行内存等 DMA 分配。

prefill/vision context 和 decode context 同时存活，图文 DMA 占用由 prefill 约 5.04 GiB 上升到 decode 约 6.73 GiB。通过 SDK 的 context 元数据与峰值 DMA inode 对账，图文峰值组成如下：

| 项目 | 峰值同一时刻的占用 |
| --- | ---: |
| 40 层×64 专家，共享缓存 | 2.051 GiB |
| Prefill 图的常量相关 NSW 分配 | 1.426 GiB |
| Decode 图的常量相关 NSW 分配 | 1.406 GiB |
| 视觉图 NSW 分配 | 0.422 GiB |
| 其它 DMA，含 IO/state 和 SDK 运行分配 | 1.422 GiB |
| CPU PSS | 0.318 GiB |
| **合计** | **7.044 GiB** |

实际 123 个 context 的 `constSize` 合计 3211.914 MiB，对应 NSW 分配 3332 MiB；全部 `sharedWeightsSize=0`。Prefill 和 decode 分别保存同源 Attention、Router/shared、head 常量；两份 head 各分配 260 MiB。目前 `SHARE_RESOURCES` 和 `REUSED_IO_LIMIT` 没有启用跨阶段权重共享。NSW 不直接等同于原始权重，1.422 GiB 的其它 DMA 也尚未全部归属到具体 buffer。

“激活 3B”包含选中的 routed 专家，不能全部当作独立 dense 常驻权重后再加专家缓存。当前 Router/shared 保持 FP16，embedding 是 BF16 mmap；并非所有非专家权重都使用 INT4/INT8。专家缓存本身没有重复计算，新增的 210 MiB memo 已在进入 decode 前释放。优先解决两阶段图常量的重复保存，再复用 IO/工作 buffer；仅减少重复常量尚不足以证明满足 4 GiB 预算。

图文/文本的 SwapPSS 峰值约 206/228 MiB；图文期间整机最低 MemAvailable 约 272 MiB，文本约 536 MiB。PSS 和 DMA 的各自最高值可能发生在不同时刻，不能直接把两者单独的最大值相加。内存采样日志不替换无采样的性能基准。

本机 MemTotal 约 10.92 GiB，测试前 MemAvailable 约 5.7 GiB。运行期间整机 Swap 使用分别从 2.42→3.18 GiB、1.89→2.94 GiB，文件缓存也明显减少；系统会调整其它驻留内存，初始可用量并非模型固定配额。PSS＋DMA 分配统计不等于 MemAvailable 的净下降量。本次能运行依赖于实际整机内存及动态回收、交换，不能据此判断可在 4 GiB 物理内存下运行。

原始数据、峰值现场及统计方法见 [运行内存报告](../../../../../llama-output/qwen35-moe-from-safetensors-gptq-20261024/history-reports/qwen35-moe-memory-20261023/REPORT.md)。

### INT4 embedding / W8A16 GPTQ Router/shared 消融

2026-10-23，完整 40 层 SM8975 设备验证。原 W2/N4×A8 routed 专家、每层 64 槽 LRU、Attention、UINT16 GDN/conv/KV 以及独立的 P/D 权重和 head 均沿用基准。新增选项没有切换默认配置。

性能使用相同 1024-token 文本输入、8×AR128 prefill、127 次 AR1，decode 容量 2048；各组使用同一份 teacher token。性能运行与内存采样分别执行。PPL 为独立 WikiText 片段的 128 个 next-token 目标，hidden SQNR 对照 FP32 的最后一层 residual hidden；不是整个测试集 PPL，也不是 logits SQNR。FP32 参考 PPL 为 4.01435。

| 配置 | 片段 PPL | Hidden SQNR | 1K prefill | Decode | 全流程采样峰值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 embedding / FP16 Router/shared | 6.22984 | 6.13 dB | 173.70 token/s | 13.77 token/s | 6.918 GiB |
| INT4 embedding / FP16 Router/shared | 6.27619 | 6.22 dB | 166.24 token/s | 13.27 token/s | 6.982 GiB |
| BF16 embedding / W8A16 GPTQ Router/shared | 6.25557 | 6.09 dB | 168.37 token/s | 13.50 token/s | 7.048 GiB |
| INT4 embedding / W8A16 GPTQ Router/shared | 6.08134 | 6.24 dB | 167.47 token/s | 13.15 token/s | 7.032 GiB |

基准、组合版各两轮，另外两组各一轮。基准两轮 decode 为 13.57–13.97 token/s，组合版为 13.02–13.28 token/s；prefill 存在约 4–8% 的轮间波动。基准首轮 thermal status 为 0，其余为 1，不能把小幅差值解释为稳定的量化加速。组合版此片段 PPL 低 2.38%，不能据此宣称整体语言质量提升。真实 greedy 中文生成正常结束：`我是一个致力于提供准确、简洁信息的人工智能助手。1加1等于2。`

Router/shared 五个投影使用 per-channel W8 GPTQ，damping=0.01，scale 来自实际 observer；不重新用 RTN 覆盖 GPTQ code。40 层 heldout shared 输出 SQNR 最低 26.58 dB、平均 37.28 dB，Router Top8 重合率最低 97.41%。Embedding 为 one-hot 对角 Hessian 下的最近舍入等价形式，详见前面的校准说明；整表 SQNR 22.10 dB，heldout 查表 SQNR 21.25 dB。

INT4 表从 970 MiB 降为 303.125 MiB，减少 666.875 MiB；这是外部查表文件，PTE 调度中的查表在 ARM 按行解包。原 BF16 表已经是按需 mmap，文件缩小不等于峰值运行内存下降。组合版两轮平均 prefill 查表耗时约 51.47 ms，基准约 77.10 ms；Router/shared 则为 197.75 / 166.99 ms，没有测出局部加速，完整 prefill 仍主要受专家 I/O 和 Native 调度限制。

完整 W8 图的整批 decode context 恢复在本机触发过装载错误，实验版使用 `serial` 加 IO 内存预估；原版使用 `async` 资源共享。相同 `serial` 控制下，W8 版 decode 阶段的 DMA 比 FP16 少 320 MiB，但串行装载后额外的资源占用抵消了相对现有部署方式的节省。FP16 `serial` 的全流程峰值为 6.942 GiB；另一次无采样 decode 为 13.55 token/s。当前组合版没有证明能降低全流程峰值，更不满足 4 GiB 预算，保持为可选实验入口。

上述性能、内存均已在重启设备、清除 DSP 重置后相机 HAL/logd 的异常后台占用后复测；重启前的约 6–7 token/s 异常结果不计入。默认路径与原基准 95 份数值文件逐字节一致。所有版本循环内 context 创建/释放、IO 注册/注销和 rpc buffer 分配/释放均为 0。精度使用已有独立运行结果；本轮没有重新评估视觉质量。

完整分解、装载方式控制组、采样方法与复现记录见 [GPTQ 消融报告](../../../../../llama-output/qwen35-moe-from-safetensors-gptq-20261024/history-reports/qwen35-moe-embedding-router-gptq-20261023/REPORT.md)。

### 四层 prefill/decode 共享权重与分离加载

2026-10-23，QNN 2.48.1、SM8975/V85，验证真实第 0–3 层：3 个 GDN 层、1 个 full-attention 层。联合编译为一个 context，包含每层 AR128 Attention、AR128 Router/shared 和融合 AR1 三个入口，共 12 个图；编译时开启 `weightSharingEnabled`。同一层两阶段的公共权重只保存一份，各图仍有少量私有常量。Attention 冻结图、W8A16 GPTQ Router/shared、W2 routed 专家和 UINT16 state/KV 沿用前面的验证版本。

通过 `QnnContext_createFromBinaryWithCallback` 捕获 SDK 的权重 section 范围，再打包为两个文件：

| 产物 | 大小 |
| --- | ---: |
| 关闭去重的四层 context，12 个图 | 199.579 MiB |
| 开启去重的四层 context | 102.539 MiB |
| 分离后的 `graph.qnn`，含图元数据和 section 表 | 2.996 MiB |
| 分离后的 `weights.bin` | 99.547 MiB |

`runner/shared/weight_source.h/.cpp` 预留虚拟地址区间，将图元数据和权重 section 以只读文件映射拼接成原 context 视图，避免复制完整 binary 到匿名内存。重建视图的 SHA256 与原始 context 完全一致。SDK 预读的权重 blob 描述头仍由权重文件映射提供，不修改或解析内部权重格式。

回调将权重读入 DMA，再由 QNN 映射到 DSP。实际初始化共申请 13 个不同 fd：一个 97.582 MiB 公共权重 section，12 个图私有 section；总 DMA 权重 buffer 为 99.547 MiB。AR128 prefill 后连续执行 32 步 AR1，provide 次数仍为 13；切换图和 token 时不重载，最后 contextFree 触发 13 次 release，live=0。没有将两个独立 context 的 weights buffer 强行指向同一块内存。

四层 Attention/Router/shared 组件验证进程的内存采样如下。统计 CPU PSS＋去重 DMA，并扣除已计入 PSS 的 DMA 映射；**本表不包含 routed 专家缓存、embedding 或 head，不是完整模型峰值**。

| 方式 | 进程采样峰值 | 同时刻 DMA | 同时刻 CPU PSS |
| --- | ---: | ---: | ---: |
| 关闭权重去重 | 822.66 MiB | 614.87 MiB | 207.79 MiB |
| 共享权重，普通 context 加载 | 612.17 MiB | 498.87 MiB | 113.29 MiB |
| 共享权重，分离文件＋DMA 回调 | 491.49 MiB | 476.42 MiB | 15.07 MiB |

分离加载相对关闭去重降低 331.18 MiB、40.26%；相对普通共享加载降低 120.68 MiB。该差值包含 CPU 文件映射页和 SDK 内部分配变化，不能直接当作公共权重大小，也不能线性推算 40 层峰值。

三种组件加载方式的 924 份输出逐字节一致，覆盖 prefill、32 步 decode、Top8、shared、GDN state 和 KV。进一步将真实 W2 Native MoE 和残差串入四层，上一层输出直接作为下一层输入，关闭共享与分离加载的 **1056 份产物逐字节一致**。正式 Native 算子源码未修改；损坏包头和截断权重也都在创建 QNN context 前被拒绝。

真实四层串联的两轮无转储均值：关闭共享为 75.130 ms/AR128、7.711 ms/AR1；分离共享为 72.009 ms/AR128、7.967 ms/AR1。为隔离共享机制，该互操作验证预载四层各 256 个 N2 专家，两个阶段复用相同 bank；计时含 Native、state/KV 和桥接复制，排除初始化及 bank 预载。它不是原 64 槽 LRU/UFS 调度的性能基准，也不包含 embedding、head、采样或视觉塔。本轮证明内存下降和数值一致，尚未证明稳定的执行加速。

联合编译入口是 `export/shared.py --layers 0 1 2 3`；传 `--router-qdq` 复用已验证 GPTQ 图，当时的独立常量控制组生成相同12图关闭去重（该对照入口已清理，历史记录保留）。打包入口如下，requests 文件来自设备加载时的公开 DMA 回调：


此历史分步命令已移除；当前统一执行 `export.py --config configs/export_sm8975.json`。


同一 context 最多共享 64 个图；本轮为 12 个。该先导实验之后已完成下节的 40 层扩展；当前默认入口已经切换到通过整模型回归的共享版本。具体限制以 SDK 版本为准，参见[高通 HTP 文档](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-10/htp_backend.html)。

本轮按用户要求清理了 265 个有对应 QNN binary 的历史单图 PTE 中间文件，释放约 5.49 GiB；完整基准 PTE、source QDQ、模型权重与测试记录保留。完整命令、验证驱动、数值对照、内存现场及清理清单见[四层权重共享报告](../../../../../llama-output/qwen35-moe-from-safetensors-gptq-20261024/history-reports/qwen35-moe-sharing-four-layer-20261023/REPORT.md)。

### 完整 40 层共享权重实测

2026-10-23，SM8975/V85、SDK 2.48.1，已按每四层一个 context 扩展到完整 40 层，head 共用一个 context/graph。十个层组共 120 个静态图，加上 head 为 **11 个 context、121 个图**。Prefill/decode 的公共权重在同一层组内去重，分离为 `graph.qnn` 与 `weights.bin`，初始化后一直保留到模型结束。通过 `run_chat.py --shared-model` 显式启用，默认入口继续使用独立权重。

沿用冻结校准结果：embedding INT4/group32、Router/shared 为 GPTQ W8A16、routed 专家保持 W2/N4×A8、GDN/conv/KV 为 UINT16。专家仍是每层 64 槽 LRU，40 层共 2560 槽、约 2.051 GiB，缺失专家走真实 UFS；未采用四层先导测试中的全专家预载方式。本轮为纯文本，未包含视觉 encoder。

| 指标 | 独立权重 | 共享权重 |
| --- | ---: | ---: |
| 128 输入 prefill | 1.376 秒 / 93.01 token/s | 1.372 秒 / 93.27 token/s |
| 随后 127 步 decode | 13.25 token/s | 12.86 token/s |
| 1024 输入，8×AR128 prefill | 6.631 秒 / 154.42 token/s | 6.028 秒 / 169.88 token/s |
| 随后 127 步 decode | 13.45 token/s | 13.04 token/s |
| 128 输入全流程采样峰值 | 7.130 GiB | **4.979 GiB** |
| 128 个 WikiText 预测目标 PPL | 6.081342397 | **6.081342397** |
| 对 FP32 的末层 hidden SQNR | 6.243535 dB | **6.243535 dB** |

128 输入和 1024 输入的两组测试，各有 **86 份 hidden、logits、输出 token 和 40 层 state/KV 文件逐字节一致**。相对原量化版本的 hidden 噪声为零。自由生成中文回答正常、18 个 token 后 EOS：`我是一个致力于提供准确、简洁信息的人工智能助手。1加1等于2。`

峰值降低 **2.151 GiB、30.17%**，口径为 CPU PSS＋按 inode 去重的 DMA 容量－已计入 PSS 的 DMA，200 ms 采样；SwapPSS 另外记录。1K 输入共享版另测峰值 **4.941 GiB**。Prefill 使用 Android `PREFILL_TOTAL.loop_ms`，decode 使用 `DECODE_LOOP`，不含初始化或 ADB 部署。性能为一次完整设备运行，存在设备/I/O 波动，不能将小幅差值视为稳定加速。1K 初始化耗时为 3.400 → 5.624 秒，本轮主要收益是内存下降，decode 并未加速。

整模验证中解决了以下问题：

- 串行加载到第 21 层时，QNN 尝试更换 DSP session，回调权重 offset 校验失败。最终采用 `QnnContext_ParamsV2_t` 的回调批量恢复，并使用 `SHARE_RESOURCES + SEQUENTIAL_WITHOUT_VA_OPTIMIZATION` 共享工作区；SDK 不允许 callback 权重与 `SEQUENTIAL_WITH_VA_OPTIMIZATION` 组合。
- 同一 context 内跨图复用 IO 时，重复注册同一 fd 会触发错误 8002。由共享库按 context/fd/dtype/shape 缓存 memHandle，并引用计数释放。prefill/decode 通过 C ABI 借用 context 与元数据，不跨 DLO 传递布局不同的 C++ `Buffer`。
- QNN 常驻后 Native 专家映射触及地址空间限制，出现 `HAP_mmap2` 错误 25。将 Native prefill 的虚拟映射窗口从四个减为两个；ARM 中的 40×64 专家缓存不变。IO 预算由实际 cache tensor 估算，此容量使用 192 MiB。
- 共享导出必须保留 `replace_prefill_cores` 的 chunked mixed-FP16 GDN。首次遗漏时 PPL 为 6.414，该轮不作为最终结果。补齐后按生成的 `.io` 文件绑定参数，恢复完整数值一致性。上面的四层先导记录对应较早的 recurrent prefill 路径，新导出器以本节的 chunked 路径为准。

最终 token 循环内 context 创建/释放、IO 注册/注销、RPC buffer 分配/释放均为零；130 次层权重回调和一次 head 回调全部正常释放，live=0。默认独立权重路径也已验证新旧 runner 的 86 份结果逐字节一致。

设备上已部署的共享版本可这样运行：

```bash
python examples/qualcomm/oss_scripts/moe/qwen35-35B-A3B/run_chat.py \
  --build-dir build/moe/qwen35-35B-A3B-sharing40 \
  --artifacts llama-output/qwen35-moe-embedding-router-gptq-20261023 \
  --shared-model llama-output/qwen35-moe-sharing-forty-layer-20261023/corrected/separated \
  --embedding-table llama-output/qwen35-moe-embedding-router-gptq-20261023/embedding.int4 \
  --prefill-pte /data/local/tmp/qwen35-sharing-corrected-model/prefill/prefill-seq128.pte \
  --decode-pte /data/local/tmp/qwen35-sharing-corrected-model/decode/decode.pte \
  --remote-root /data/local/tmp/qwen35-sharing-corrected \
  --decode-capacity 1152 \
  --cache-experts 64 \
  --prompt-tokens 128 \
  --max-new-tokens 32 \
  --prompt "你好，请用中文介绍一下自己，再告诉我1加1等于几。"
```

完整数值、初始化耗时、内存现场、SHA256、修复经过与导出调用见[40 层共享权重报告](../../../../../llama-output/qwen35-moe-from-safetensors-gptq-20261024/history-reports/qwen35-moe-sharing-forty-layer-20261023/REPORT.md)。两种 prefill 计划均为 50,304 字节，decode 计划为 56,832 字节，权重在外部文件。经授权另清理了约 1.93 GiB 的重复中间 PTE；原始 binary、冻结 QDQ、权重、日志与对照结果保留。

### 每层 32 / 64 槽对照

2026-10-24，在上述完整 40 层共享权重版本上，仅修改专家缓存容量，Top8、量化配置和图保持一致。Prefill、decode 与交接共同使用实际的 32 / 64 槽容量，pipeline bank 随容量缩小。原每层 64 个候选的启动 profile 保留；32 槽只加载每层前 32 个候选，推理期间按 LRU 更新。

SM8975/V85，纯文本、AR128、decode capacity=1152；每组随后执行相同的 127 步 teacher-forced AR1。下表两列均为本轮同版本 runner 实测，未混用历史吞吐。

| 指标 | 每层 64 槽 | 每层 32 槽 |
| --- | ---: | ---: |
| 128 输入 prefill | 1.363 秒 / 93.92 token/s | 1.569 秒 / 81.56 token/s |
| 随后 decode | 12.90 token/s | 10.76 token/s |
| 128 输入全流程采样峰值 | 4.972 GiB | **3.902 GiB** |
| 1024 输入 prefill，8×AR128 | 6.297 秒 / 162.61 token/s | 7.900 秒 / 129.62 token/s |
| 随后 decode | 12.97 token/s | 10.85 token/s |
| 1024 输入全流程采样峰值 | 4.946 GiB | **3.867 GiB** |

峰值口径为 CPU PSS＋去重 DMA，200 ms 采样，SwapPSS 为 0；不含手机系统及其他应用。128 输入组额外保留全部 logits 用于 PPL，1024 输入组只保留最后一步 logits，同一输入长度的两列完全同口径。

常驻专家缓存从 **2.051 → 1.026 GiB**，加上两个 pipeline bank 缩小，整模峰值减少约 **1.07–1.08 GiB**。128 输入后的 decode 命中率由 **84.24% → 68.56%**，127 步读取量由 **5.48 → 10.94 GB**；1K 输入后命中率由 **85.11% → 72.02%**。额外的缺失读取和 prefill 分批调用导致吞吐下降。

128 / 1024 两组各 86 份 hidden、logits、输出 token、state/KV 逐字节一致。短 WikiText 回归 PPL 均为 **6.081342397**，对 FP32 的末层 hidden SQNR 均为 **6.243535 dB**，没有额外数值损失。

`configs/sm8975.json` 现默认每层 32 槽。需要重编译 runner；本轮已编译目录是 `build/moe/qwen35-35B-A3B-cache32-20261024`。显式传 `--cache-experts 32` 或 `64` 控制两个阶段，无需重新导出 PTE。完整命令、原始日志、峰值 fdinfo/smaps 与数值对照见[32 槽测试报告](../../../../../llama-output/qwen35-moe-from-safetensors-gptq-20261024/history-reports/qwen35-moe-cache32-20261024/REPORT.md)。

### 从 safetensors 全流程 GPTQ 验证

2026-10-24，从 `/hunter/model/Qwen/Qwen3.6-35B-A3B` 的原始分片重新完成 40 层逐层顺序校准、GPTQ、QNN 编译、权重分离与 SM8975 实测，没有复用旧量化权重。量化和导出入口及命令见 [CONFIGURATION.md](CONFIGURATION.md)。本轮冻结配方为 `configs/gptq.json`；历史导出清理后，`configs/model.json` 和设备配置默认指向本轮产物。

Attention 的 160 个投影/卷积滤波器、Router/shared 的 200 个投影、10,240 个 routed 专家的 GU/down、head，以及视觉的 112 个矩阵/滤波器均完成 GPTQ。文本 embedding 使用 INT4/group32，在 one-hot 对角 Hessian 下固定网格 GPTQ 等价于最近舍入；norm、bias 和 GDN 时间向量沿用后端编码。专家为外部 N2 bank，运行时展开为 N4×A8；其他部分编译为 QNN。AR128/AR1 共享校准网格，容量 1152，每层 32 槽 LRU，GDN/conv/KV 保持 UINT16。

下表是同一设备、相同输入和容量的新旧产物对照。性能各一轮，不含初始化与部署；PPL 来自一段独立 WikiText 留出序列的 896 个预测位置，不是完整标准测试集成绩。

| 指标 | 旧混合量化产物 | 本轮全矩阵 GPTQ |
| --- | ---: | ---: |
| 1024 输入 prefill，8×AR128 | 6.834 秒 / 149.84 token/s | 6.722 秒 / 152.33 token/s |
| 随后 127 步 decode | 10.37 token/s | 10.70 token/s |
| 留出序列 PPL，FP32 为 8.8173 | 12.0969 | 11.9839 |
| 最终 hidden SQNR | 7.2391 dB | 6.3706 dB |

中文回答 `1加1等于2。` 和图片描述均正常。视觉 QDQ SQNR 为 28.25 dB，手机执行 118.24 ms；256-token 图文 prefill 为 2.963 秒。本轮性能接近，整体精度没有明显改善，仍需结合逐层误差分析，不能把全部差值归因于 GPTQ。完整覆盖统计、校准边界、六组设备记录及复跑命令见[全流程报告](../../../../../llama-output/qwen35-moe-from-safetensors-gptq-20261024/REPORT.md)。

## Android JNI

新增独立 `Qwen35MoeModule`，支持设备端图片文件解码、流式文本、多轮、取消及重置。原 `LlmModule` 接口保持不变。构建、打包及调用示例见 [JNI 接入说明](../../../../../extension/android/jni/moe/README.md)。


## 2026-10-26：review 修复与有效长度 prefill

Prefill 保持 AR128，首轮、多轮及 reset 后的输入均按 128 分块。129 个真实 token 使用两次 AR128，第二块的有效长度为 1；300 个 token 使用三块，有效长度依次为 128、128、44。尾部用 inactive padding 补齐，不将空白文字插入普通用户的系统提示。仅显式性能测试 `--prompt-tokens` 保留真实空白补长。

GDN 图新增 INT32 `last_token_index`：padding 的 Q/K/V 与 beta 为零、decay 为 1，conv state 按最后有效位置选取。Full attention 的物理窗口为逻辑 context capacity＋128，历史区覆盖完整逻辑上下文；mask 与 KV 反馈只提交真实 token。超过 4K 的 full attention 复用长 decode 的按 head/序列分块 FP16 MatMul，避免整块 GQA 展开放入 VTCM；softmax 仍覆盖全部历史，权重与 UINT16 cache 的量化规则不变。这样尾块不会挤掉历史，也不会回退到 AR1。Native MoE 描述符区分物理长度与有效长度，padding 不参与专家读取、计算路由及 LRU 更新。Head 使用最后一个有效 token 的 hidden，吞吐统计按真实 token 计数。

多轮交接复用原 context、权重及专家 buffer，通过实际校准网格转换 state/KV；已有 decode KV 不重新量化，只追加本轮新 KV。专家 payload 原地复用，只交接槽 ID、LRU stamp 和最后有效 token 的路由。大块 prefill 的工作缓冲在首次阶段切换时缩至 AR128，DSP worker 保留；取消或输入失败后使专家元数据失效并要求 reset，避免使用被部分覆盖的旧槽。

本次同时修复：共享 context 批量初始化失败后清理未借用的 context，Graph 构造失败时回收已取得资源；按目标 context 生成并检查 RoPE；去掉普通 decode 导出入口的 1152/2048 限制；CLI/JNI 从部署计划的 `model_configuration` 读取 embedding、KV、state swap、IO 和调度设置；decode 初始 KV 快照只保存真实前缀，chat 导入后释放快照及 JNI memfd。原 `LlmModule` API 不受影响。

新 shared 导出自动从编译后的图提取 `cache-encoding.bin`，按 checkpoint config 生成足够长度的 cos/sin；内部 `export/capture.py` 一起保留这些文件与 IO metadata。运行时优先使用这套配套资产，拒绝长度不足的 RoPE。旧 PTE、旧 cache metadata 与旧 Native skeleton 不能混用此版 padding ABI，需要重新导出和构建。

本机验证记录：

- `runner/tests/run_host.sh`：共享 context 部分初始化/回滚/重试，KV 的 1、2、63、64、65、127、128 有效长度及 UINT16 网格转换，配置覆盖与 RoPE 越界保护通过；10 个 PyTorch 测试通过，包含 129-token 两块连续 state 更新及长 GQA 分块后的全局 softmax 数值检查。
- 真实已校准第 0 层：完整 AR128 与原 FP16 chunk 实现逐元素一致；有效长度为 1、65、127 时，更改所有 padding 内容不会改变有效输出、conv state 或 recurrent state。
- 四层、2048 逻辑上下文：12 个方法各包含一个 QNN delegate，共享一个 context；prefill 历史为 2048，物理窗口为 2176；编译元数据确认 8 个 cache IO 均为 UINT16，并提取实际 scale/offset。额外 32K full-attention 验证中，3 个方法同样各包含一个 QNN delegate、共享一个 context，物理窗口为 32896；分块处理消除了未分块时 32 MiB InputSlice 无法放入 TCM 的编译错误。
- Android terminal runner、decode DSO、JNI、shared provider、V85 Native 与 GDN op package 构建通过；原有 163 组 HVX/HMX 锁失败与恢复测试通过。

设备完整 benchmark 已停止，本轮没有替换其冻结版本文件。新多轮 AR128 的端侧数值、耗时与内存峰值仍待设备回归；完整历史区会影响静态缓存占用，不能用本机编译结果推断端侧性能或峰值。
