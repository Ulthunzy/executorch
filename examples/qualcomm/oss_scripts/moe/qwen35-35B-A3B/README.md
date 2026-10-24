# Qwen3.5-35B-A3B MoE，SM8975/V85

本目录维护已验证的 40 层 MoE 推理、融合图导出及专家重构代码。独立于 `llama.py`。默认使用融合后的 Attention＋Router/shared、AR128 prefill、AR1 decode、每层 32 槽 LRU（可用 `--cache-experts 64` 切换）、命中专家先计算与缺失专家异步读取。

当前 prefill/decode 各自持有独立的 QNN 图权重及 head。专家缓存仍跨阶段交接复用；视觉 context 在整个模型会话中常驻。Router/shared 为 FP16，embedding 为外部 BF16 表。

## 代码位置

```text
qwen35-35B-A3B/
├── run_chat.py              # 部署、实时 token 输出、EOS、状态交接验证
├── export.py                # 从已校准逐层 QDQ 图导出融合 AR1 图
├── export_shared.py         # 四层 AR128/AR1 联合编译与公共权重去重验证
├── export_router.py         # 导出独立 W8A16 GPTQ Router/shared 图
├── export_vision.py         # 只读取视觉权重，校准并导出 W8A16 编码器
├── multimodal.py            # 图片预处理、image token 展开和三维 MRoPE
├── pack_pte.py              # 将逐层 QNN binary 包装为完整 40 层 PTE 调度计划
├── pack_shared_weights.py   # 按 SDK 回调范围分离图元数据与权重文件
├── build.sh                 # Android model runner、decode 动态库、Native MoE
├── build_op_package.sh      # 复用共享 GDN 源码构建 host/V85 包
├── configs/sm8975.json      # 本机模型、外部产物和设备路径
├── export/
│   ├── model.py             # Router/shared、融合层
│   ├── graph_utils.py       # 静态上下文特化
│   ├── split_attention_core.py
│   └── fp16_attention_core.py
├── runner/
│   ├── prefill/             # 图片编码、AR128、每层专家缓存、向 decode 交接
│   ├── decode/              # 常驻 context/state、32/64 槽 LRU、异步 IO、SIMD 采样
│   ├── shared/              # 共享 context 的外部权重文件映射与 DMA 回调
│   ├── embedding.h/.cpp     # 按行读取 BF16 或 packed INT4 embedding
│   ├── handoff.h
│   └── resident_handoff.h
├── native/
│   ├── prefill/             # AR128 的 N2 解包、A8、N4×A8 HMX
│   ├── decode/              # AR1、mailbox、bank 映射与就绪队列
│   ├── worker_start.h/.cpp   # 所有 worker 锁申请完成后统一进入或取消计算
│   ├── tests/               # 主机并发测试及设备真实锁失败注入
│   └── N2Utils.h/.cpp       # N2→N4 展开接口与实现
└── quantization/
    ├── quantize.py          # Hessian/GPTQ、完整专家重构与 scale 拟合
    ├── prepare_router.py    # 用已有逐层输入校准 W8A16 GPTQ Router/shared
    ├── prepare_embedding.py # one-hot 对角 Hessian、group-32 INT4 表
    └── prepare_experts.py   # 逐层顺序校准、生成 experts.n2 和 bank-index.bin
```

GDN recurrence 复用[共享 DeltaNetQhpi.cpp](../../../custom_op/deltanet_op_package_htp/DeltaNetOpPackage/src/ops/DeltaNetQhpi.cpp)，本目录不复制该 op。decode 选择已有 DDR kernel；prefill 沿用已有 AR128 图。原 PTE 的 `Q36DECO1`/`Q36PREF1`、`Qwen36*Stage` 名称作为兼容 ABI 保留。

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
- Router/shared 保留 FP16，GDN/conv/KV cache 保留 UINT16，embedding 仍为外部 BF16 表。
- 已验证静态 decode 容量为 1152、2048；prefill 固定 AR128，支持 128 到 1024 的完整块。文本基准使用 128/1024 两种长度，使用原生 chat template、关闭 thinking。支持原始 token 文件输入。
- 支持单图输入，默认 448×448、196 个视觉 token、视觉塔 W8A16。图片前处理在主机，视觉编码及完整 MoE prefill/decode 在设备上执行。
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

1024 输入、容量 2048：

```bash
PYTHONPATH="$(dirname "$PWD")" python -u \
  examples/qualcomm/oss_scripts/moe/qwen35-35B-A3B/run_chat.py \
  --prompt-tokens 1024 --decode-capacity 2048 \
  --prompt "什么是注意力机制？" --max-new-tokens 128
```

可以用 `--config` 选择另一份 JSON，或用 `--device`、`--model`、`--artifacts`、`--build-dir`、`--prefill-pte`、`--decode-pte`、`--remote-weights` 覆盖路径。设备 PTE 路径指向**已部署文件**，其他依赖如 embedding、QNN runtime/skel 和 indices 从该 PTE 所在目录获取；bank 从 `--remote-weights` 读取。

运行会部署本目录构建的 runner、Native MoE 和 GDN 包，并加载 `--artifacts/assets` 的融合图。两个完整 PTE 保留兼容调度容器，decode 的逐层 Attention/Router 由融合 binary 覆盖。不能用普通 llama runner 直接运行这套 PTE。

结果默认写入 `llama-output/qwen35-35B-A3B`，用 `-a` 指定其他目录。`--text-only` 仅显示增量正文，`--trace-layers` 开启逐层计时。正常聊天遇 EOS 停止；`--teacher-tokens`、`--save-all-logits`、`--ignore-eos` 用于显式诊断。

### 图片编码与图文推理

`--vision-artifacts` 指向 `export_vision.py` 生成的目录，默认由 `configs/sm8975.json` 的 `vision_artifacts` 指定。运行入口部署同一编码器的 `vision.bin`；`vision.pte` 同时保留用于检查和其他部署方式。

```bash
PYTHONPATH="$(dirname "$PWD")" python -u \
  examples/qualcomm/oss_scripts/moe/qwen35-35B-A3B/run_chat.py \
  --image-path "/home/hunter/下载/test.jpg" \
  --prompt "请用中文描述这张图片，简短回答。" \
  --max-new-tokens 128
```

448×448 图片先变成 `[1,784,1536]` patches，经设备视觉图生成 `[1,196,2048]` FP32 特征。prefill 的 embedding 阶段按 image token 的出现顺序直接替换对应行，跨 AR128 块继续消费同一份特征。视觉图与全部 prefill 图共用 Runtime，加入同一组异步 context 初始化；像素加载也在初始化阶段完成。编码后不释放视觉 context，穿过 decode 阶段继续常驻，整次模型会话结束才回收。`VisionEncoder::load_image` 可更换同分辨率输入，`encode` 复用已有 context 和 IO。

运行时按实际图文 token 构造三维 MRoPE，decode 继续使用图片压缩位置带来的偏移。短图文输入自动选取最小的 AR128 整数倍，当前样例为 256 token；为了复用完整递推 128 步的现有图，在 system 段补真实空白 token，避免末尾 padding 污染最终 state/logits。输入 token 计数包含这些空白。可显式指定 `--prompt-tokens 1024 --decode-capacity 2048`，不足以容纳图片和文字的长度会报错。

日志 `PTE_READY.setup_ms` 包含模型及视觉图的初始化；`VISION_ENCODE` 统计常驻图执行、输出复制和有限值检查，附带 context/IO 创建次数，均应为 0。`MODEL_RESIDENT` 校验 decode 返回后视觉和 prefill context 仍然存活。`PREFILL_TOTAL` 与 `DECODE_LOOP` 分别统计语言模型两阶段，特征保存放在主循环后。结果目录额外保存 `pixels.f32`、`image-features.f32`、`mrope-positions.i32` 和本次使用的 cos/sin 表。

### 导出视觉编码器

仅加载 checkpoint 中的 `model.visual.*`，不会把 35B 文本权重加载到主机内存。复用现有 Qwen3.5 静态视觉实现、W8A16 配方与 INT32 bias 保护；每通道 W8 Linear，使用真实图片在 GPU 校准。使用与导出融合图相同的 SDK 环境：

```bash
PYTHONPATH="$(dirname "$PWD")" python -u \
  examples/qualcomm/oss_scripts/moe/qwen35-35B-A3B/export_vision.py \
  --model /hunter/model/Qwen/Qwen3.6-35B-A3B \
  --output llama-output/qwen35-moe-vision-sm8975-20261023 \
  --qnn-binding llama-output/qwen36-decode-optimization-sm8975-20261023/export/PyQnnManagerAdaptor.cpython-311-x86_64-linux-gnu.so \
  --calib-images /path/to/calibration-image-1.jpg /path/to/calibration-image-2.jpg
```

输出包含 `vision.pte`、`vision.bin`、`vision-qdq.pt2`、`VISION.json` 及 FP32/QDQ 参考数据。导出检查视觉图只有一个 QNN delegate，输入输出各一个；运行前检查模型配置指纹和分辨率。`--image-height`、`--image-width` 可选择其他 32 的整数倍分辨率，需要重新导出对应图。

## 重新导出融合图

输入是已完成校准的逐层数据目录，每层包含 `decode-qdq.pt2`、`moe-constants.npz`。保留原 Q/DQ 网格，按层串行编译，每图要求只有一个 QNN delegate。这个步骤不会重新训练或重新校准 35B 模型。

```bash
export LD_LIBRARY_PATH="$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/lib/x86_64-unknown-linux-gnu:$QNN_SDK_ROOT/lib/x86_64-linux-clang:$LD_LIBRARY_PATH"

PYTHONPATH="$(dirname "$PWD")" python -u \
  examples/qualcomm/oss_scripts/moe/qwen35-35B-A3B/export.py \
  --source-dir /path/to/calibrated-decoder-layers \
  --output /path/to/artifacts/assets \
  --capacities 1152 2048 \
  --op-package-host build/moe/qwen35-35B-A3B/op-package/x86_64-linux-clang/libQnnDeltaNetOpPackage.so
```

可用 `--layers 0 3` 先导出一层 GDN 和一层 full attention。默认使用当前安装的 `PyQnnManagerAdaptor`；若需选择与 QNN SDK 一致的单独构建版本，可传 `--qnn-binding /path/to/PyQnnManagerAdaptor...so`。host op 包由本目录的构建脚本生成。

输出目录为 `gdn-fused/layerN`、`context1152-fused/layerN` 和 `context2048-fused/layerN`，每层包含 `attention.pte`、`attention.bin`、`io.bin`、`IO.json`。

## 完整 PTE 容器与专家重构

`pack_pte.py` 使用已有逐层 QNN binary 和 head 构建完整执行计划，每个计划 122 个阶段。输入目录格式：

- decode：`layerN/attention.bin`、`io.bin`、`moe-front.bin`。
- prefill：`layerN/forward_0.bin`、`moe-front.bin`；full-attention 另需 `cold/layerN/forward_0.bin`。

```bash
python examples/qualcomm/oss_scripts/moe/qwen35-35B-A3B/pack_pte.py \
  --phase decode --source-dir /path/to/base-decoder-layers \
  --head /path/to/decode-head.bin --cache-encoding /path/to/cache-encoding.bin \
  --output /path/to/decode.pte
```

prefill 使用 `--phase prefill` 和对应 AR128 head，不传 `--cache-encoding`。`--prefill-tokens` 默认 128；设为完整输入长度可启用下文的按层复用，QNN/Native 内核仍为 AR128。逐层 QNN 编译和容器包装是两个步骤；本目录尚不提供从原始 Hugging Face 权重直接生成所有基础图的一条命令。

专家重构入口：

```bash
PYTHONPATH="$(dirname "$PWD")" python -u \
  examples/qualcomm/oss_scripts/moe/qwen35-35B-A3B/quantization/prepare_experts.py \
  --model /path/to/Qwen3.5-35B-A3B \
  --source-dir /path/to/calibrated-prefill-layers \
  --inputs /path/to/INPUTS.pt --output /path/to/reconstructed-experts
```

该步骤需要 CUDA，逐层运行，默认每批 8 个专家。`INPUTS.pt` 包含 `calibration:[8,1024,2048]`、`heldout:[2,1024,2048]` 两个 FP32 embedding tensor。source-dir 包含每层的 `front-qdq.pt2`、`moe-constants.npz`、`reference.pt` 以及 `cos.f32`、`sin.f32`。沿用原来的 8192-token 校准流程，7 条拟合、1 条选择方案，2 条 heldout 不参与拟合；继续保留完整专家输出重构及逐层量化误差传播。

## 可选的 embedding 与 Router/shared 量化

独立权重基准仍使用 BF16 embedding、FP16 Router/shared。`run_chat.py --embedding-table /path/to/embedding.int4` 可切换为 group-32 INT4 查表；每组保存 16 字节 code 和两个 FP16 scale/bias，不展开整张表。两阶段分别 mmap 同一个文件，生成时只解码当前 token 的一行。图片特征继续直接替换 image token 的 embedding。

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

python examples/qualcomm/oss_scripts/moe/qwen35-35B-A3B/export_router.py \
  --source-dir /path/to/router-qdq --output /path/to/router-contexts \
  --qnn-binding /path/to/PyQnnManagerAdaptor.so
```

两步均按层串行处理。校准默认使用 CUDA；导出产生独立的 `prefill/layerN/moe-front.bin`、`decode/layerN/moe-front.bin`。融合 AR1 导出沿用前文的 `export.py`，增加 `--router-qdq /path/to/router-qdq`。打包 PTE 时替换 source-dir 中对应的 `moe-front.bin`，Attention 和 head 沿用原版本；运行时 `--artifacts` 指向包含新融合 AR1 图及原专家 bank/index 别名的目录。P/D 的权重及 head 仍然独立。

本机 SDK/设备下，完整 W8 图在保留 prefill context 后整批恢复 decode 会触发 DSP 装载错误。运行 W8 版需用 `--decode-context-load-mode serial`，逐图创建 context 并启用 SDK IO 内存预估；初始化结束后全部常驻，循环内不创建/销毁。`auto` 模式读取 artifact 根目录 `ASSETS.json` 的 `decode_context_load_mode`，无 manifest 时沿用原版 `async`。这与权重共享无关；内存对照需同时注明装载方式。

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

逐轮原始结果、decode 时间分解、输入与构建指纹见[完整报告](../../../../../llama-output/qwen35-moe-cpp-refactor-performance-20261023/REPORT.md)及同目录 `RESULTS.csv`、`MANIFEST.json`、`EQUIVALENCE.json`。

### 首次视觉编码接入验证（历史独立 Runtime）

2026-10-23，448×448 W8A16 视觉塔导出为一个 QNN delegate，实际在 SM8975 执行。用同一 checkpoint 的 Transformers FP32 独立视觉塔作参考，比较完整的 196×2048 图像特征：

| 图片 | 编码执行 | 编码总耗时 | 编码输出 SQNR | 256-token 图文 prefill | Decode |
| --- | ---: | ---: | ---: | ---: | ---: |
| 双猫，参与视觉校准 | 116.21 ms | 974.20 ms | 25.90 dB | 3.191 s | 14.48 token/s，127 步 |
| 本地 test.jpg，未参与校准 | 116.09 ms | 984.50 ms | 22.72 dB | 3.291 s | 14.46 token/s，66 步 |

双猫样例描述出了虎斑猫、粉红色沙发及遥控器，达到 128 个输出 token 上限。未校准样例描述出了白色公牛、橙衣男子、黑色背包和帐篷，67 个输出 token（包含 EOS）后正常结束。两次运行前后 thermal status 均为 0。输入/输出和路由不同，这两轮的吞吐不用于推断量化性能差异。

MRoPE 的位置和 delta 对照覆盖方形图片、矩形图片及 1024-token 图文输入，与 Transformers 一致；设备所用整张 cos/sin 表与 Transformers 的最大绝对误差均为 0。静态视觉 FP32 相对官方 FP32 的 SQNR 为 51.19 dB。新增视觉入口后的纯文本回归中，178 份数值产物仍与前轮基准逐字节一致，decode 为 13.32 token/s。

本次验证了单图编码和完整 MoE 生成链路，未重做语言模型图文校准或完整视觉问答基准。原始结果、精度数据和样例输出见[视觉接入报告](../../../../../llama-output/qwen35-moe-vision-sm8975-20261023/REPORT.md)。

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

视觉生命周期修复后的上述基准仍需读取约 7.84 GB 专家数据，RPC 外读取处理占 2109.38 ms（约 65%）；后续 I/O 优化见下一节。尝试直接 O_DIRECT 读入 rpcmem buffer 返回 EFAULT，未纳入正式代码。专家 buffer 在两阶段共享，内存计数不能直接相加。完整分解、生命周期和数值证据见[常驻视觉报告](../../../../../llama-output/qwen35-moe-resident-vision-20261023/REPORT.md)。

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

```bash
python examples/qualcomm/oss_scripts/moe/qwen35-35B-A3B/pack_pte.py \
  --phase prefill \
  --source-dir llama-output/qwen36-full40-sm8975-20261021 \
  --head llama-output/qwen36-full-prefill-pte-sm8975-20261021/forward_0.bin \
  --prefill-tokens 256 \
  --output llama-output/qwen35-moe-prefill-io-20261023/prefill-seq256.pte

adb -s A7SR026915000282 push \
  llama-output/qwen35-moe-prefill-io-20261023/prefill-seq256.pte \
  /data/local/tmp/qwen36-full-prefill-pte-sm8975-20261021/

PYTHONPATH="$(dirname "$PWD")" python -u \
  examples/qualcomm/oss_scripts/moe/qwen35-35B-A3B/run_chat.py \
  --prefill-pte /data/local/tmp/qwen36-full-prefill-pte-sm8975-20261021/prefill-seq256.pte \
  --prompt-tokens 256 \
  --image-path /home/hunter/下载/test.jpg \
  --prompt "请用中文描述这张图片，简短回答。" \
  --max-new-tokens 128
```

1K 输入将打包长度设为 `--prefill-tokens 1024`，运行时选对应 PTE 并传 `--prompt-tokens 1024`。新计划使用 v2 header，需使用新 runner；原 AR128/v1 PTE 可继续运行。默认配置的 AR128 PTE 路径保持原值，选择新计划才会按层复用。

本轮没有修改 decode；短测图文 13.95–14.71 token/s，1K 前缀 9.48–11.29 token/s，尚不能判断稳定变化。完整分解、对照命令、数值核对及内存说明见 [I/O 优化报告](../../../../../llama-output/qwen35-moe-prefill-io-20261023/REPORT.md)。

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

原始数据、峰值现场及统计方法见 [运行内存报告](../../../../../llama-output/qwen35-moe-memory-20261023/REPORT.md)。

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

完整分解、装载方式控制组、采样方法与复现记录见 [GPTQ 消融报告](../../../../../llama-output/qwen35-moe-embedding-router-gptq-20261023/REPORT.md)。

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

联合编译入口是 `export_shared.py --layers 0 1 2 3`；传 `--router-qdq` 复用已验证 GPTQ 图，`--independent` 生成相同 12 图但关闭去重的控制组。打包入口如下，requests 文件来自设备加载时的公开 DMA 回调：

```bash
python examples/qualcomm/oss_scripts/moe/qwen35-35B-A3B/pack_shared_weights.py \
  --context llama-output/qwen35-moe-sharing-four-layer-20261023/shared/context0.bin \
  --requests llama-output/qwen35-moe-sharing-four-layer-20261023/four-requests.txt \
  --output llama-output/qwen35-moe-sharing-four-layer-20261023/separated
```

同一 context 最多共享 64 个图；本轮为 12 个。该先导实验之后已完成下节的 40 层扩展；完整模型默认入口仍保持独立权重。具体限制以 SDK 版本为准，参见[高通 HTP 文档](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-10/htp_backend.html)。

本轮按用户要求清理了 265 个有对应 QNN binary 的历史单图 PTE 中间文件，释放约 5.49 GiB；完整基准 PTE、source QDQ、模型权重与测试记录保留。完整命令、验证驱动、数值对照、内存现场及清理清单见[四层权重共享报告](../../../../../llama-output/qwen35-moe-sharing-four-layer-20261023/REPORT.md)。

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

完整数值、初始化耗时、内存现场、SHA256、修复经过与导出调用见[40 层共享权重报告](../../../../../llama-output/qwen35-moe-sharing-forty-layer-20261023/REPORT.md)。两种 prefill 计划均为 50,304 字节，decode 计划为 56,832 字节，权重在外部文件。经授权另清理了约 1.93 GiB 的重复中间 PTE；原始 binary、冻结 QDQ、权重、日志与对照结果保留。

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

`configs/sm8975.json` 现默认每层 32 槽。需要重编译 runner；本轮已编译目录是 `build/moe/qwen35-35B-A3B-cache32-20261024`。显式传 `--cache-experts 32` 或 `64` 控制两个阶段，无需重新导出 PTE。完整命令、原始日志、峰值 fdinfo/smaps 与数值对照见[32 槽测试报告](../../../../../llama-output/qwen35-moe-cache32-20261024/REPORT.md)。
