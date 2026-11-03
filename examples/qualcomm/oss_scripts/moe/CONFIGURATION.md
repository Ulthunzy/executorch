# Qwen3.5-35B-A3B：环境、配置、导出与推理

本文对应本目录当前的 `export.py`、`build.sh` 和 `run_chat.py`。首次使用按第1～6节操作；已经准备好相同源码、Python环境和SDK的基础库后，日常导出只需修改配置并执行一次 `export.py`。

SM8850当前整合版的入口为 `configs/sm8850.json`，构建目录为
`build/moe/qwen35-35B-A3B-sm8850`，使用SDK2.51.200.261021及已验证的U16状态模型。
64槽、prefill IO16、decode IO8、512 KiB读取块及现有PTE的直接运行命令见
[当前版本说明](docs/SM8850_OPTIMIZATION.md#current-integrated-version)。运行配置中的
`cache_experts`可覆盖旧包内的槽数默认值，显式 `--cache-experts`仍优先。

先明确构建和导出的关系：

| 入口 | 做什么 | 需要提前准备什么 |
| --- | --- | --- |
| 仓库的 `backends/qualcomm/scripts/build.sh` | 构建主机QNN Python binding、ExecuTorch Android基础库及分词器等依赖 | Python环境、QNN SDK、Android NDK、源码子模块 |
| 本目录的 `build.sh` | 链接基础库，构建MoE runner、Native DSP专家内核、共享加载库、权重采集工具和GDN op package | 已构建的 `build-android`、Hexagon SDK/Tools、Qualcomm llama.cpp辅助头文件 |
| 本目录的 `export.py --config ...` | 串行完成可选GPTQ、QNN编译、设备权重范围采集、共享权重分离、执行计划和单PTE打包 | 导出配置、源权重、匹配的构建依赖；完整文本PTE还需要设备 |

**需要先编译 ExecuTorch/QNN 后端。** 导出配置的 `build=true` 只调用本目录的 `build.sh`，不会替你构建仓库QNN Python接口或 `build-android` 基础库。基础库已经可用时，无需每次重编译。

## 1. 准备源码、Python和SDK

以下主机命令均从 **ExecuTorch仓库根目录**执行。路径是本机示例，换机器时替换成自己的安装位置。

```bash
conda activate et_npu
cd /hunter/cpp_code/ET/npu/executorch

export MOE_ROOT="$PWD/examples/qualcomm/oss_scripts/moe"
export MOE_CONFIG="$MOE_ROOT/configs/export_sm8975.json"
export PYTHONPATH="$(dirname "$PWD")${PYTHONPATH:+:$PYTHONPATH}"

export QNN_SDK_ROOT=/hunter/qnn_sdk/src/backends/qnn/2.48.1.260715
export ANDROID_NDK_ROOT=/hunter/android-ndk/android-ndk-r28c
export HEXAGON_SDK_ROOT=/hunter/qnn_sdk/hexagon/6.7.0.0
export HEXAGON_TOOLS_ROOT="$HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/21.0.04/Tools"
export LLAMA_CPP_ROOT=/hunter/cpp_code/ET/npu/llama.cpp
export EXECUTORCH_ANDROID_BUILD="$PWD/build-android"
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH="$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/lib/x86_64-unknown-linux-gnu:$QNN_SDK_ROOT/lib/x86_64-linux-clang:${LD_LIBRARY_PATH:-}"
```

`HEXAGON_TOOLS_ROOT` 必须指向含 `bin/hexagon-clang++` 的 **`Tools` 子目录**。`LLAMA_CPP_ROOT` 指向Qualcomm fork的llama.cpp，构建读取 `ggml/src/ggml-hexagon/htp/hvx-sigmoid.h` 等辅助头文件。

可以先 `source env.sh`，然后再执行上面的环境变量设置。`env.sh` 含本机路径，其QNN版本及Hexagon Tools路径不一定与本流程一致；不能仅凭已经source就认为版本匹配。

首次拉取源码后初始化子模块；分词器使用仓库锁定的原有tokenizers子模块：

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

如果当前环境还没有安装本仓库的Python包，先执行：

```bash
./install_executorch.sh --editable
```

已有可用的 `et_npu` 环境可跳过安装。环境需要PyTorch CUDA、torchao、transformers、safetensors、NumPy、Pillow及ExecuTorch导出依赖，先检查：

```bash
python -c 'import torch, torchao, transformers, safetensors, numpy, PIL; import executorch.exir; print("torch:", torch.__version__, "CUDA:", torch.cuda.is_available()); print("transformers:", transformers.__version__)'
```

完整文本 `source=safetensors` 流程要求CUDA，并使用当前全GPTQ配方和INT4 embedding。将 `calibration_device` 改成 `cpu` 不能把这套文本流程切成CPU校准；CPU选项可用于允许它的独立视觉阶段。主机为Linux x86_64，QNN编译在主机执行，实际推理在设备HTP/DSP执行。

主机还需可用的C/C++编译器、CMake、make/ninja、ADB和GNU time。量化子流程直接调用 `/usr/bin/time`，设备采集和部署调用PATH里的 `adb`。transformers需能识别源checkpoint的模型配置和tokenizer；只通过普通包导入检查，不等于已验证其支持该checkpoint。

## 2. 先构建 ExecuTorch/QNN 基础依赖

在上述环境中首次构建，或更换QNN SDK后重建：

```bash
bash backends/qualcomm/scripts/build.sh --release --job_number 4
```

脚本默认同时构建x86_64和Android。`--job_number 4` 限制基础库编译并行度，避免内存不足时开满线程；它不控制GPTQ专家批量。

| 构建结果 | 用途 |
| --- | --- |
| `build-x86/backends/qualcomm/Py*.so` | 主机QNN接口，构建脚本自动复制至 `backends/qualcomm/python/` |
| `build-android/libexecutorch*.a`、`extension/*/*.a` | 本项目runner链接的Android ExecuTorch基础库 |
| `build-android/lib/libtokenizers.a`、`libregex_lookahead.a`、`libpcre2-8.a` 等 | 原生分词、正则及其依赖 |
| `build-android/lib/libextension_llm_sampler.a` | 原生采样器 |
| `build-android/include/`、`schema/include/` | 安装及生成头文件 |

已有完整Android构建、只重建主机binding时用 `--skip_linux_android`；只构建Android用 `--skip_x86_64`。默认构建会清理对应的 `build-x86` / `build-android`；`--no_clean` 适用于已有完整构建目录的增量构建。不要混用其他SDK或其他Python ABI构建的旧 `.so`。

使用默认binding时检查实际加载位置：

```bash
python -c 'from executorch.backends.qualcomm.python import PyQnnManagerAdaptor as qnn; print(qnn.__file__)'
```

本目录 `build.sh` 默认链接仓库的 `build-android`。若使用其他基础库目录，设置 `EXECUTORCH_ANDROID_BUILD` 指向该构建根目录。它需要上述实际库与头文件，仅有一个通用 `executor_runner` 可执行文件不够。

导出JSON的 `build=true` 会自动构建本项目，不必再手动运行一次本目录 `build.sh`。设为 `false` 则完全复用 `build_dir`，其中必须已有匹配的runner、Native skeleton、共享库、采集工具和op package；`op_package_host` 默认也是这个目录下的 `op-package/x86_64-linux-clang/libQnnDeltaNetOpPackage.so`。更换源码、SDK、SoC或RPC协议后，不能沿用未匹配的旧构建。

`prefill_gdn_core` 是导出设置，支持默认 `chunked_mixed_fp16`，以及实验性的 `custom_fp16` / `custom_u16`。FP16核心接回原缓存QDQ；U16核心保留原Q/K/V/beta/state网格，decay改用scale=1/65535、zero-point=0以精确表示padding所需的1，并在FP32中屏蔽无效token。U16递推含额外逐token量化，需独立验证误差。各实现应输出到独立目录；层组和PTE sidecar记录选择，同一模型禁止混组。

`gdn_state_vtcm` 默认 `false`；设为 `true` 将同一共享context中的custom GDN映射为 `DeltaNetCoreVtcm`，同时影响decode。该选项不适用于 `custom_fp16`，AR128与AR1应分别计时。

`MOE_NATIVE_PROFILE=1` 构建DSP计时版本，建议使用独立 `MOE_BUILD_DIR`。推理加 `--profile` 才读取累计计数；`TARGET.json` 会验证该构建标记。`MOE_BUILD_OP_PACKAGE=0` 可跳过GDN包构建以复用已验证的同架构算子，默认仍构建完整依赖。分析工具为 `tools/moe_profile.py`，常规吞吐测试应使用无插桩版本。

`MOE_EXPERT_CACHE_TRACE=1` 输出每层、每个有效 AR128 块的专家请求、命中、缺失和新增文件加载次数，默认关闭。每次真实 Top8 使用计一次请求；首次需要加载记一次 miss，同一窗口后续复用常驻或 memo 权重记 hit，重复请求合并等待同一次加载。Native 在权重 ready 后才计算，因此该计数表示加载及复用，不表示 Router 阶段瞬时就绪率。启动预加载另见 `STARTUP_COLD_PRELOAD`。可用 `MOE_EXPERT_CACHE_TRACE_FILE` 指定二进制路由追加文件；每帧包含七个 little-endian int32（magic `0x4d4f4552`、版本 1、层号、块起点、窗口起点、有效 token 数、槽数），后接槽内专家 ID 和有效 Top8 ID。每次独立采样前清空该文件。

## 3. 修改原始 safetensors 路径和导出配置

### 3.1 原始模型路径改哪里

**修改 `configs/export_sm8975.json` 的 `model` 字段。** 该字段传给量化、视觉导出、RoPE生成和最终打包，不是在 `model.json` 或 `gptq.json` 里配置。

```json
"model": "/hunter/model/Qwen/Qwen3.6-35B-A3B"
```

例如权重实际放在 `/data/models/Qwen3.5-35B-A3B`，就改成：

```json
"model": "/data/models/Qwen3.5-35B-A3B"
```

它必须是 **原始Hugging Face checkpoint目录**，不是单个 `.safetensors` 文件、PTE或GPTQ产物目录。代码读取 `model.safetensors.index.json`，据 `weight_map` 找全部分片；不能只下载其中一个分片。

完整文本＋视觉PTE的源目录应保留：

```text
Qwen3.5-35B-A3B/
├── config.json
├── generation_config.json
├── tokenizer.json
├── tokenizer_config.json
├── model.safetensors.index.json
└── *.safetensors                 # 索引引用的全部分片
```

下载目录中的其他tokenizer/processor配置也建议保留。分词器及模型配置使用 `local_files_only=True` 读取，不会替你在线下载缺失模型。

本机目录叫 `Qwen3.6-35B-A3B`，其中权重使用Qwen3.5 MoE结构；**目录名不是结构判断依据**。入口校验 `config.json.text_config` 的40层、hidden=2048、256专家、Top8，后续还依赖对应层布局和张量名。不能仅换路径就用此导出器处理0.8B、2B、4B等dense模型。

### 3.2 配置文件分工

| 文件 | 负责什么 | 是否配置原始权重路径 |
| --- | --- | --- |
| `configs/export_sm8975.json` | 本次任务：源模型、校准/输出/构建目录、SDK、数据、设备和格式 | **是，`model` 字段** |
| `configs/model.json` | 量化、KV/state、上下文、生成和调度 | 否 |
| `configs/gptq.json` | 可选GPTQ模型配方 | 否；只有被 `model_config` 选中才生效，不自动叠加 |
| `configs/sm8975.json` | `run_chat.py` 的设备/产物默认路径 | 其中 `model` 只是推理时默认config/tokenizer来源，不控制导出 |

默认导出JSON引用 `"model_config": "model.json"`。要用 `gptq.json` 就改此引用；要改上下文、thinking、专家槽数，则修改实际被引用的配方。导出生成的 `MODEL_CONFIG.json` 是生效快照，`device.json` 记录该次推理路径，不是源配置文件。

### 3.3 完整导出配置示例

下面内容可放入 `configs/export_sm8975.json`。根据机器替换模型、SDK、数据和设备路径；示例将任务保存在 `llama-output/qwen35-moe-local/`，校准、编译、构建互不重叠。

```json
{
  "version": 1,
  "model_config": "model.json",
  "model": "/hunter/model/Qwen/Qwen3.6-35B-A3B",
  "source": "safetensors",
  "calibration_dir": "llama-output/qwen35-moe-local/calibration",
  "output": "llama-output/qwen35-moe-local/pte",
  "build": true,
  "build_dir": "llama-output/qwen35-moe-local/build",
  "qnn_binding": null,
  "qnn_sdk_root": "/hunter/qnn_sdk/src/backends/qnn/2.48.1.260715",
  "hexagon_sdk_root": "/hunter/qnn_sdk/hexagon/6.7.0.0",
  "hexagon_tools_root": "/hunter/qnn_sdk/hexagon/6.7.0.0/tools/HEXAGON_Tools/21.0.04/Tools",
  "android_ndk_root": "/hunter/android-ndk/android-ndk-r28c",
  "soc_model": "SM8975",
  "device": "A7SR026915000282",
  "runtime_device_dir": "/data/local/tmp/qwen35-runtime-v85",
  "remote_root": "/data/local/tmp/qwen35-moe-export",
  "op_package_device": "/data/local/tmp/qwen35-moe-export/libQnnDeltaNetOpPackage_HTP.so",
  "components": ["text", "vision"],
  "phases": ["prefill", "decode"],
  "format": "pte",
  "corpus": "/data/calibration/wikitext-2-raw-v1-train.jsonl",
  "images": ["/data/calibration/image1.jpg", "/data/calibration/image2.jpg"],
  "calibration_device": "cuda",
  "expert_batch": 8,
  "image_height": 448,
  "image_width": 448
}
```

路径和优先级按 `workflow/export_config.py` 解析：

- `model_config` 相对路径以 **导出JSON所在目录** 为基准，所以示例引用同目录的 `model.json`。
- 模型、校准、输出、构建、语料、图片和工具链的JSON相对路径，以 **ExecuTorch仓库根目录** 为基准，不以终端当前目录或JSON目录为基准。
- JSON不会展开shell变量，不要写 `"$QNN_SDK_ROOT"` 或 `"~/models/..."`；填实际绝对路径或上述相对路径。
- JSON明确配置的SDK路径优先。**省略**对应字段才从 `QNN_SDK_ROOT`、`ANDROID_NDK_ROOT`、`HEXAGON_SDK_ROOT`、`HEXAGON_TOOLS_ROOT` 读取；显式 `null` 不会回退，必需SDK缺失会报错。只改环境变量不会覆盖JSON里的SDK路径。
- `qnn_binding` 的语义不同：`null` 或缺省表示使用第2节编译并复制到源码包的默认接口；非空路径指定另一份匹配的Manager `.so`，但不代替其余后端依赖的构建。
- `export.py` 的CLI可覆盖 `--output`、`--device`、`--soc-model`；原始模型路径改JSON，不能套用dense `llama.py` 的 `--source_safetensors`。

仓库当前配置中的 `qnn_binding`、`corpus`、`images` 等可能指向本机历史产物。迁移机器时逐项替换；这些文件不会因路径写在配置里就自动生成。示例使用 `qnn_binding=null`，前提是第2节已完成。

### 3.4 文本和图片校准数据

`corpus` 是本地 **JSONL文件**，每行至少有字符串字段 `text`：

```json
{"text": "第一段校准文本……"}
{"text": "第二段校准文本……"}
```

统一入口不会自动下载WikiText，也不直接接受 `--calib_text_datasets wikitext-2`。先将公开语料训练集保存为上述格式，再修改 `corpus`。

当前代码按固定行偏移取10段、每段1024 token：前8段用于校准/选择，后2段留出；前段包含从第3500行开始的取样。需要足够多的文本行，且每个截取区间能编码出1024 token。几条短文本不适用，会报 `Corpus segment ... is too short`。语料哈希、取样范围和token哈希写入 `calibration/DATA.json`。

`images` 是主机上真实存在的校准图片列表。选择 `vision` 时不能为空，使用Pillow读取并按 `image_height` / `image_width` 处理；默认448×448对应196个视觉token。这里是校准图片，推理图片另用 `run_chat.py --image-path` 指定。

## 4. 准备目标设备的QNN运行库

### 4.1 为什么导出PTE也要接设备

当前 **完整文本 `format=pte`** 流程在真机通过SDK的 `contextCreateFromBinaryWithCallback` 恢复context，记录公共权重请求区间，随后分离 `graph.qnn` 与 `weights.bin`。因此配置加载时就要求 `device` 和 `runtime_device_dir`，不是只在最后推理时使用。

导出配置设置 `deploy_runtime=true` 时，统一流程按 `qnn_sdk_root` 和目标架构自动上传 Android HTP/System/Prepare、stub 及 DSP skel 到 `runtime_device_dir`，并记录文件 SHA256 到 `build_dir/QNN_RUNTIME.json`。恢复导出时仍会同步运行库，随后复用内容校验通过的编译阶段。SM8850 配置已开启此选项，使用 SDK 2.51.200.261021 与独立的 V81 设备目录。

`deploy_runtime` 默认为 `false`，适合已有正确运行库的设备。此时 `export/capture.py` 只上传新编译的 `capture_weights`、GDN设备包和待采集context，新设备需按下面步骤先准备 `runtime_device_dir`。运行库、编译用SDK及Python QNN binding 必须匹配。

```bash
adb devices -l
export MOE_DEVICE=A7SR026915000282
adb -s "$MOE_DEVICE" shell getprop ro.soc.model
```

配置只选 `soc_model`，架构来自 `backends/qualcomm/serialization/qc_schema.py`，无需再传 `--dsp-arch`：

| SoC | Hexagon架构 | 当前验证范围 |
| --- | --- | --- |
| `SM8750` | `v79` | 专家内核交叉编译通过，整模设备验证待做 |
| `SM8850` | `v81` | SDK2.51.200整模导出通过，32/64槽、256/512/1024输入输出的16次真机验证通过，8192个生成token与基线一致 |
| `SM8975` | `v85` | 当前40层文本、图文、多轮及JNI回归使用的目标 |

换SoC需重编译目标图、Native和GDN包，使用新的导出/构建目录及匹配设备库。入口会核对 `ro.soc.model`。

### 4.2 SM8975/v85测试机示例

这台测试机自身提供以下匹配库，本机QNN 2.48.1 SDK没有对应的V85 stub/skel目录：

```text
/odm/lib64/libQnnHtp.so
/odm/lib64/libQnnHtpV85Stub.so
/odm/lib64/libQnnSystem.so
/odm/lib64/rfsa/libQnnHtpV85Skel.so
```

可将 **设备自己的库** 复制到专用目录：

```bash
adb -s "$MOE_DEVICE" shell 'mkdir -p /data/local/tmp/qwen35-runtime-v85'
adb -s "$MOE_DEVICE" shell 'cp /odm/lib64/libQnnHtp.so /odm/lib64/libQnnHtpV85Stub.so /odm/lib64/libQnnSystem.so /data/local/tmp/qwen35-runtime-v85/'
adb -s "$MOE_DEVICE" shell 'cp /odm/lib64/rfsa/libQnnHtpV85Skel.so /data/local/tmp/qwen35-runtime-v85/'
```

还需使该目录能找到设备真实FastRPC库。若实际位置是 `/vendor/lib64/libcdsprpc.so`：

```bash
adb -s "$MOE_DEVICE" shell 'cp /vendor/lib64/libcdsprpc.so /data/local/tmp/qwen35-runtime-v85/'
```

不同设备的路径和依赖可能不同，按其系统实际位置准备；代码先加载 `libcdsprpc.so`，再尝试 `libadsprpc.so`。**Hexagon SDK中供交叉链接的FastRPC stub不能代替设备实现。** 所用QNN运行库若还需要 `libQnnHtpPrepare.so` 等依赖，也应准备同套匹配文件；正常推理部署会为它建立链接。不要随意混用其他SoC或版本的stub/skel。

将导出JSON的 `runtime_device_dir` 设为此目录。采集阶段的 `LD_LIBRARY_PATH` 指向它，`ADSP_LIBRARY_PATH` 还包含 `/odm/lib64/rfsa` 等DSP目录。已有正确依赖目录可直接复用，无需再次复制。

### 4.3 只导出QNN格式

将 `format` 改成 `qnn`，文本仍完成GPTQ和十组共享QNN context编译，但不执行设备采集、图/权重分离、host PTE计划及最终单PTE打包，此路径不连接设备。仍需主机QNN接口、SDK及 `build=true` 触发的本项目构建依赖。

Routed MoE专家继续使用外部Native bank，不会因选择 `qnn` 就变成通用QNN可直接执行的完整MoE图。日常聊天应选择 `format=pte`，同时保留 `prefill` / `decode` 两阶段；仅选择单阶段用于相应计划导出，不是完整聊天模型。

## 5. 检查配置并执行统一导出

按第3节修改配置后，先查看生效路径、配方和实际阶段：

```bash
python "$MOE_ROOT/export.py" --config "$MOE_CONFIG" --print-plan
python "$MOE_ROOT/export.py" --config "$MOE_CONFIG" --check
```

`--print-plan` 不导入PyTorch/QNN，不编译或连接设备。确认其中 `configuration.model`、`calibration_dir`、`output`、`model_config` 正确。

`--check` 检查模型结构、分片、数据、SDK及部分已有工具/产物，不连接设备，也不验证CUDA、动态库加载或Android基础静态库是否齐全。`EXPORT_PREREQUISITES_OK` 不替代第1、2、4节。

采用第3节示例配置时，执行：

```bash
export MOE_OUT="$PWD/llama-output/qwen35-moe-local/pte"

python -u "$MOE_ROOT/export.py" \
  --config "$MOE_CONFIG" \
  --output "$MOE_OUT" \
  --soc-model SM8975 \
  --device "$MOE_DEVICE" \
  > run-qwen35-moe-export.log 2>&1
```

另一终端可查看：

```bash
tail -f run-qwen35-moe-export.log
```

`--output` 只覆盖编译/打包工作目录，不覆盖 `calibration_dir` 或 `build_dir`。JSON已配置相同输出时可省略此参数。顶层日志放在输出目录之外，避免检查点重置时被清理。

自动执行顺序：

1. `build=true` 时构建本项目runner、Native、共享加载库、采集工具和GDN包。
2. `source=safetensors` 时从原始分片逐层GPTQ，生成Attention/Router QDQ、专家bank、INT4 embedding和head；`calibrated` 则校验并复用完整量化结果。
3. 编译head及十组四层共享QNN context；每组同时包含AR128 prefill和AR1 decode方法。
4. 从新图提取实际cache编码，按目标容量生成RoPE；选择视觉时执行视觉GPTQ和编译。
5. 文本PTE模式连接设备逐组采集SDK权重范围，分离共享图/权重。
6. 生成一个 `prefill.pte` 和一个 `decode.pte` 的host计划，自动打包为 `model.pte`。

不需要手动运行prepare、head、shared、vision、capture或打包命令。一次只运行一个大模型导出。`expert_batch` 控制专家GPTQ批量，显存不足时可减小；它与基础库编译线程数无关。

### 5.1 输出目录、断点复用和重新量化

| 情况 | 实际行为 |
| --- | --- |
| 输出目录不存在 | 创建专用目录和检查点 |
| 配置/来源匹配，阶段完成且产物SHA256匹配 | 打印 `EXPORT_REUSED`，跳过阶段 |
| 某阶段文件缺失或损坏 | 重建该阶段及后续阶段 |
| 检查点缺失/损坏或任务来源不匹配 | 清理专用输出目录内容，重建检查点 |
| 同一目录已有活动导出 | 文件锁拒绝第二个进程 |
| `--no-resume` | 重跑外层构建/编译；量化子流程仍按自身检查点校验和复用 |

`calibration_dir` 和 `output` 不能相同或相互包含。两者都是专用产物目录，不放其他用户文件；量化目录也有独立来源检查点及重置逻辑。源模型、数据、SDK不应放在待重置目录中。

只改编译/交付选项时，已完成且匹配的GPTQ可复用。换checkpoint、语料或量化配方时，使用新的校准和输出目录，便于保留旧版本。强制全新GPTQ也使用新的 `calibration_dir`，不能只依靠 `--no-resume`。

续跑同一任务通常重复原命令即可。明确复用已有完整GPTQ时，设置 `source=calibrated`、`calibration_dir` 指向对应结果；`model` 仍指对应原始checkpoint，提供来源校验、配置/tokenizer及可选视觉权重。必须保留 `QUANTIZATION_COMPLETE.json`、各层QDQ、全部bank/index及Router/head报告，不能只填某个专家文件。

### 5.2 输出和最终交付

```text
qwen35-moe-local/
├── calibration/                 # GPTQ产物及量化子阶段检查点
├── build/                       # 本项目runner、Native、共享库、采集工具、op package
└── pte/                         # --output 目录
    ├── model.pte                # 唯一模型交付文件
    ├── MODEL_CONFIG.json        # 本次生效配方快照
    ├── EXPORT_PLAN.json         # 实际阶段与命令
    ├── EXPORT_STATE.json        # 完成状态及产物指纹
    ├── EXPORT.json              # 完成报告及交付位置
    ├── device.json              # 本机推理配置
    ├── qnn-shared/group0..9/    # 编译后的原始共享context
    ├── shared-model/            # graph.qnn、weights.bin、IO、cache编码和RoPE
    ├── head/、vision/           # 编译组件及metadata
    ├── plans/                   # prefill.pte、decode.pte与JSON
    ├── assets/                  # cache编码和RoPE
    ├── banks/、indices/         # 指向calibration的符号链接
    ├── embedding.int4          # 指向calibration的符号链接
    └── logs/                    # 内部阶段日志
```

完整 `format=pte` 的最终交付是 **一个 `model.pte`**，含执行计划、共享QNN图/权重、全部外存专家、embedding、视觉编码器和tokenizer/config。四层分组和小PTE是内部恢复/诊断产物，不需逐个收集。使用原生named-data segments，不把整个大文件一次性装进DRAM。

**运行程序和平台依赖不在模型PTE内**：还需runner/JNI、Native skeleton、共享库、GDN包及设备QNN运行库。不能把这个包直接交给任意通用 `executor_runner` 当标准端到端QNN图使用。

保留目录运行或断点恢复时，保留其引用的 `calibration_dir`。单PTE数据已写入包内；运行时校验并解包到磁盘缓存，后续复用。主机默认缓存为PTE旁 `.qwen35-model-cache/<id>/`，设备可用 `--package-cache` 指定。磁盘要同时容纳原包和解包资源；单文件交付不代表没有解包空间或运行内存开销。

## 6. 导出后的推理

### 6.1 主机部署并运行

按第5节导出后，推荐选完整PTE，并指定本次构建和设备依赖：

```bash
python -u "$MOE_ROOT/run_chat.py" \
  --weights-path "$MOE_OUT/model.pte" \
  --build-dir "$PWD/llama-output/qwen35-moe-local/build" \
  --device "$MOE_DEVICE" \
  --soc-model SM8975 \
  --runtime-device-dir /data/local/tmp/qwen35-runtime-v85 \
  --temperature 0 --max-new-tokens 128 \
  --no-enable-thinking --interactive --stream \
  --prompt "你好，请用中文介绍一下自己。"
```

PTE包含模型配置和tokenizer，**此用法不要求重新读取原始safetensors**。主机首次解包，再按已有部署逻辑同步到设备，不是每个token解包或上传。专家首次上传较大，后续 `--deploy-weights` 默认用ADB `--sync`。

也可直接使用自动生成的推理配置，选择一种方式即可：

```bash
python -u "$MOE_ROOT/run_chat.py" \
  --config "$MOE_OUT/device.json" \
  --prompt "请用中文介绍一下自己。"
```

`device.json` 记录该次源模型、构建和产物绝对路径，迁移主机要更新路径，或选单PTE方式。只传导出目录时会读取配置记录的模型config/tokenizer来源，可用 `--model /实际checkpoint目录` 覆盖。仓库 `configs/sm8975.json` 的默认路径不会自动跟随任意一次新导出。

图文会话示例：

```bash
python -u "$MOE_ROOT/run_chat.py" \
  --weights-path "$MOE_OUT/model.pte" \
  --build-dir "$PWD/llama-output/qwen35-moe-local/build" \
  --device "$MOE_DEVICE" --soc-model SM8975 \
  --runtime-device-dir /data/local/tmp/qwen35-runtime-v85 \
  --image-path /主机上的图片路径/test.jpg \
  --temperature 0 --max-new-tokens 128 \
  --no-enable-thinking --interactive --stream \
  --prompt "请用中文描述这张图片。"
```

上传原始JPG/PNG，设备C++完成图片解码、前处理、视觉编码、分词和生成。未给初始图片时，纯文本会话默认关闭视觉；要在后续轮次 `/image`，启动时加 `--with-vision`。视觉context随整个session创建并保留到退出。

### 6.2 常用推理参数

| 参数 | 实际行为 |
| --- | --- |
| `--weights-path` / `--artifacts` | 完整 `model.pte` 或导出目录，不是原始safetensors路径 |
| `--config` | 推理设备/产物配置；与 `export.py --config` 的JSON类型不同 |
| `--model-config` | 运行配方，需匹配图和量化接口；不能把旧PTE变成新精度/容量 |
| `--model` | 目录模式下config/tokenizer来源，不触发重量化或编译 |
| `--build-dir` | 本项目runner、Native和共享库等的主机构建目录 |
| `--runtime-device-dir` | 已部署QNN依赖的设备目录 |
| `--device` / `--soc-model` | ADB序列号与芯片，必须匹配图、Native和设备 |
| `--temperature` / `--temp` | 0为greedy，正数使用原生ExecuTorch Sampler，top-p沿用0.9 |
| `--max-new-tokens` | 本轮输出预算，受上下文剩余空间限制，含thinking |
| `--ignore-eos` / `--no-ignore-eos` | 固定输出性能测试可忽略EOS，正常回答默认遇EOS停止 |
| `--enable-thinking` / `--no-enable-thinking` | 原生thinking模板及标记显示 |
| `--interactive` / `--no-interactive` | 首轮后继续对话，复用常驻context、state/KV和专家cache |
| `--stream` / `--no-stream` | 逐token或整轮显示，不改变模型计算 |
| `--image-path` | 初始图片的主机路径 |
| `--with-vision` / `--no-with-vision` | 是否保留视觉塔支持后续 `/image`，初始图片自动开启 |
| `--remote-weights` | 设备UFS专家bank/index路径 |
| `--deploy-weights` / `--no-deploy-weights` | 默认同步权重；关闭时复用已有专家、embedding 和共享图，embedding、head 与各组 graph/weights/metadata/IO 必须与本机文件 SHA256 相同，缺失或不匹配会拒绝运行 |
| `--runner diagnostic` | 张量dump、teacher-forcing与诊断链路，固定greedy、不做交互 |

CLI生成控制覆盖模型JSON的 `generation`。每次结果保存 `console.log`、`runner.log` 和 `RUN.json`。Python也可调用 `run_chat.py` 的 `run_inference(config=None, weights_path=None, device=None, soc_model=None, prompt=..., temperature=None, ignore_eos=None, image_path=None, with_vision=None, interactive=None, stream=None, enable_thinking=None, max_new_tokens=None)`；省略项继承配置，交互模式阻塞至 `/exit`，返回结果目录。

### 6.3 直接在设备终端运行单PTE

先在主机部署运行环境。示例将项目文件和第4节QNN依赖放到同一设备目录：

```bash
export MOE_BUILD="$PWD/llama-output/qwen35-moe-local/build"
adb -s "$MOE_DEVICE" push "$MOE_BUILD/qwen35_moe_runner" /data/local/tmp/qwen35-runtime-v85/
adb -s "$MOE_DEVICE" push "$MOE_BUILD/libmoe_decode.so" /data/local/tmp/qwen35-runtime-v85/
adb -s "$MOE_DEVICE" push "$MOE_BUILD/libqwen35_shared.so" /data/local/tmp/qwen35-runtime-v85/
adb -s "$MOE_DEVICE" push "$MOE_BUILD/native-prefill/libinteger_rpc_skel.so" /data/local/tmp/qwen35-runtime-v85/libqwen35_prefill_skel.so
adb -s "$MOE_DEVICE" push "$MOE_BUILD/native-decode/libinteger_rpc_skel.so" /data/local/tmp/qwen35-runtime-v85/libqwen35_decode_skel.so
adb -s "$MOE_DEVICE" push "$MOE_BUILD/op-package/hexagon-v85/libQnnDeltaNetOpPackage_HTP.so" /data/local/tmp/qwen35-runtime-v85/
adb -s "$MOE_DEVICE" push "$MOE_OUT/model.pte" /data/local/tmp/qwen35-model.pte
```

进入 `adb -s "$MOE_DEVICE" shell` 后，在 **设备shell** 执行：

```sh
cd /data/local/tmp/qwen35-runtime-v85
chmod +x qwen35_moe_runner
export LD_LIBRARY_PATH="$PWD:/odm/lib64"
export ADSP_LIBRARY_PATH="$PWD;/odm/lib64/rfsa;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/dsp"

./qwen35_moe_runner \
  --model-root /data/local/tmp/qwen35-model.pte \
  --package-cache /data/local/tmp/qwen35-model-cache \
  --library-directory "$PWD" \
  --temperature 0 --max-new-tokens 128 \
  --no-vision --no-enable-thinking --interactive --stream \
  --prompt "你好，请用中文介绍一下自己。"
```

`--library-directory` 为包内解出的阶段目录链接共享库及正确prefill/decode skeleton。本例明确用 `--no-vision` 运行纯文本；要使用图片，将它替换为 `--vision`，先把图片放到设备，再加 `--image-path /data/local/tmp/test.jpg`，或在后续轮输入 `/image "设备图片路径" 问题`。`/reset` 清历史、保留模型资源，`/exit` 统一释放。

JNI可用 `new Qwen35MoeModule.Config(modelPte, workspaceDirectory, nativeLibraryDirectory)`；图片接口、线程和APK打包要求见[JNI说明](../../../../../extension/android/jni/moe/README.md)。

SM8850/V81 构建直接加载模型包时，原生 runner/JNI 使用 prefill IO 16、decode IO 8、512 KiB 读取分块；旧包内的导出调度不会使这一路径退回8/256。原生 runner 显式指定 `--model-config` 时保留配置选择，`--prefill-io-workers`、`--decode-io-workers`、`--read-chunk-kib` 的显式参数优先。其他SoC保留原行为。无需为此重做GPTQ、QNN导出或PTE打包。

## 7. 量化、KV/state和上下文

以下属于实际选中的 `model_config`，不是导出JSON的顶层字段。

### 7.1 量化配方

| `quantization` 字段 | 默认 | 实际作用 |
| --- | --- | --- |
| `attention` | `gptq` | Attention投影/卷积GPTQ，已校准兼容路径也支持 `source_qdq` |
| `vision` | `w8a16_gptq` | 视觉矩阵/滤波器及位置表GPTQ，W8A16 |
| `head` | `w4a16_gptq` | head W4/group32 GPTQ |
| `matrix_gptq_damping` / `expert_gptq_only` | `0.01` / `true` | 矩阵Hessian阻尼，专家强制GPTQ而非从RTN组合择优 |
| `expert_bits` / `expert_activation_bits` | `2` / `8` | N2外存、VTCM展开为N4×A8 HMX，不是原生W2 HMX指令 |
| `expert_gu_group` / `expert_down_group` | `1024` / `512` | 专家bank/index实际分组，Native ABI需匹配 |
| `expert_gptq_damping` / `reconstruction_steps` | `0.1` / `80` | 专家GPTQ和完整输出重构 |
| `router_shared` | `w8a16_gptq` | 五个per-channel INT8权重投影、UINT16激活，历史已校准路径可用 `fp16` |
| `router_gptq_damping` | `0.01` | Router/shared GPTQ阻尼及来源校验 |
| `embedding` | `int4_g32` | group32 INT4查表，历史兼容路径可选原 `bf16` 表 |

原始文本统一流程要求GPTQ Attention、GPTQ Router/shared、INT4 embedding和全GPTQ专家，不能把任意历史兼容配方直接用于 `source=safetensors`。

量化宽度保持混合配置：Attention主要W4，GDN K/z和depthwise W8，Router/shared和视觉W8，head/embedding W4，专家N2存储。Norm、bias、GDN时间参数等非矩阵参数沿用后端编码，不标为GPTQ。Embedding的one-hot Hessian为对角阵，固定网格GPTQ退化为最近舍入，没有普通Linear的跨列误差补偿收益。

逐层按Attention → Router/shared → 当前层全部256专家推进，下一层使用上一层实际量化输出，并保留独立FP32参考链和留出序列。MTP辅助层不导出。修改JSON不会在推理时改变旧权重精度，需要重新生成对应量化/编译产物。

### 7.2 KV/state和容量

| 字段 | 默认 | 含义及支持范围 |
| --- | --- | --- |
| `cache.kv_dtype` | `uint16` | 当前MoE接口只支持UINT16 |
| `cache.gdn_state_dtype` | `uint16` | 当前GDN接口只支持UINT16 |
| `cache.conv_state_dtype` | `uint16` | 当前conv state接口只支持UINT16 |
| `cache.append_only` | `true` | KV追加布局，减少整段搬运 |
| `cache.state_swap` | `true` | GDN state输入/输出buffer交换 |
| `context.max_seq_len` | `1152` | 总逻辑上下文，含模板、文本、图片和输出；配置范围1024～32768的128倍数，需编译对应图 |
| `context.prefill_ar_len` | `128` | 当前固定AR128 kernel |

输入1024、输出128，至少需要1152总上下文；输入1024、输出1024，至少2048。输入包含模板和图片token，thinking也占输出预算。运行时 `--max-seq-len` 不能把1152图扩成2048，要先改配方并导出对应图/RoPE/计划。支持配置长容量不代表全部容量已完成性能/精度评测。

同一请求先合并模板、文字和图片，再统一按AR128分块。253个真实token执行两块，第二块125个有效位置；无效padding不推进GDN/conv/KV，不参与路由和LRU。不是文字与图片分别补齐，也不是每128/1024个token多导出一个PTE。后续轮输入同样分块，AR1用于生成；超限拒绝追加，不静默截断历史。

首轮与后续轮都只受总上下文容量约束，实际输入加输出预算不能超过 `max_seq_len`。旧配置中的 `context.max_prefill_tokens` 读取时忽略，新配置不再保存该字段。`PLANS.json` v3 的 `workspace_tokens` 仅描述单次 PTE 的临时工作区。所有 Prefill 入口共用按层调度：每层完成当前窗口内全部 AR128 块后再换层；超过工作区的输入按窗口继续执行同一个 PTE，携带绝对起始位置、有效长度及 KV/GDN state。reset 只重置会话状态，保留图、工作区和专家权重内存，调度方式保持不变。

## 8. Thinking、统计与运行调度

`generation` 默认开启thinking，`max_new_tokens=128`、`temperature=0`、`ignore_eos=false`、`interactive=false`、`stream=true`。旧PTE保存关闭设置时，可在运行时用 `--enable-thinking` 覆盖。输入模板 `<think>` 会补显，模型实际生成的 `</think>` 保留；前缀不重复计入生成token。预算太小时可能只有思考、尚未输出最终答案。

性能直接复用原生 `Stats`、`stats_to_json_string()` 和 `print_report()`。每轮显示 `PyTorchObserver`、加载、prompt evaluation、decode、TTFT和sampling time，并打印全局KV、容量、待写入token及完整上下文。最后采样token（含EOS）可能尚未写入KV，下一轮一起提交；padding不计长度。JNI提供 `kv_cache_tokens`、`pending_tokens`、`context_tokens`、`context_capacity`、`enable_thinking`。

| `schedule` 字段 / CLI | 默认 | 实际行为 |
| --- | --- | --- |
| `prefill_order` / `--prefill-order` | `layer_major` | 统一按层调度；`chunk_major` 参数暂时保留，发出弃用提示后转到同一实现，后续版本移除 |
| `cache_experts` / `--cache-experts` | `32` | 当前支持32/64槽，每层独立LRU、40层保留 |
| `prefill_async_io` / `--prefill-async-io` | `true` | 专家读取与计算重叠 |
| `startup_prefetch` / `--startup-prefetch` | `overlap` | `overlap` / `sync` / `none`，候选profile不代表永久锁定热专家 |
| `io_workers` / `--io-workers` | `8` | 读取worker支持4/8/16 |
| `--prefill-io-workers` | 跟随`io_workers` | chat中独立设置prefill读取worker；SM8850运行配置默认16 |
| `startup_io_workers` / `--startup-io-workers` | `4` | 启动预取worker |
| `read_chunk_kib` / `--read-chunk-kib` | `256` | 单次专家读取块大小 |
| `decode_hit_first` / `--decode-hit-first` | `true` | 命中专家先算，缺失专家同时读 |
| `decode_ready_io` / `--decode-ready-io` | `true` | 就绪标记驱动后续计算，要求hit-first |
| `overlap_feedback` / `--overlap-feedback` | `true` | 重叠state/KV反馈 |
| `context_load_mode` / `--decode-context-load-mode` | `auto` | 共享模式统一恢复常驻context，不用 `serial` 做当前共享文本导出 |

布尔参数均有 `--no-...`。关闭prefill异步读时还需 `--startup-prefetch none`。四层分组用于共享及编译，十组和head随session常驻，不按四层卸载/重建context。专家buffer跨阶段交接，缺失专家从UFS加载、LRU更新，不改变真实Top8路由。当前未接入跨token预测预取。

SM8850 含视觉塔的PTE额外打包 `cold-experts-image.i32` 与来源清单 `cold-experts-image.json`，默认表保存在 `configs/sm8850_image_cold.json`。仅首轮实际传入图片时从该表选择32/64槽前缀。文本导出同时打包 `cold-experts-text.i32/json`，默认profile为 `configs/sm8850_text_cold.json`，首轮无图片时使用，包括仅开启 `--with-vision` 的文本请求。V79/V85或旧包缺当前模态可选表时使用原 `cold-experts.i32`。预载数量及总字节数不变。

现有目录或旧PTE可分别用 `tools/image_cold_profile.py --output-dir <导出目录或PTE所在目录>`、`tools/text_cold_profile.py --output-dir <相同目录>` 添加sidecar；Python入口在旧PTE父目录查找两组可选资产，原生CLI/JNI单PTE需重新打包。表与来源清单必须成对存在。文本固定profile包含三段独立自然正文及SHA、256-token/零额外空格条件、两种预注册排名的留一分数；原基准和独立验证正文均不参与训练。文本工具的可选 `--collection` 从这三段独立校准路由重建并校验输入token SHA。更新任一固定profile只使package检查点失效。`STARTUP_COLD_PROFILE` 日志记录实际所选表。

模型、context、buffer、cache和worker跨用户等待保留；Native RPC在每次forward返回前完成，不跨用户等待保留未返回调用。runner、decode DSO和协议2 DSP skeleton要配套更新。

## 9. 常见问题和检查位置

| 问题 | 检查方法 |
| --- | --- |
| 原始权重路径改哪里 | 导出JSON的 `model`；用 `--print-plan` 确认 `configuration.model` |
| 改 `gptq.json` 没效果 | 看实际 `model_config` 引用，默认是 `model.json` |
| `build=true` 仍缺 `.a` / tokenizer / schema头文件 | 先构建第2节基础库，核对 `EXECUTORCH_ANDROID_BUILD` |
| 缺 `hvx-sigmoid.h` | `LLAMA_CPP_ROOT` 要指向含该头文件的Qualcomm llama.cpp |
| QNN binding找不到/加载失败 | 检查host构建、Python ABI、`qnn_binding`、SDK及动态库路径 |
| `Missing export inputs` / `Missing checkpoint shards` | 检查JSON路径和索引全部分片，历史产物不会自动下载 |
| `Corpus segment ... is too short` | 检查JSONL行数、`text`字段及固定区间是否够1024 token |
| 只改环境变量仍用旧SDK | JSON明确路径优先，修改该路径或删除字段后用环境变量；SDK字段不能用 `null` 回退 |
| GPU可用但校准失败 | 查看具体阶段日志和显存，必要时减小 `expert_batch`；完整文本不能切CPU |
| PTE在 `separate-groupN` 失败 | 查设备、SoC、运行库和该阶段日志，不是重做GPTQ就能解决 |
| 导出时不想接设备 | 选 `format=qnn`，得到离线context及Native专家产物，不生成完整PTE |
| 目录存在是否复用 | 有效检查点和校验匹配才复用，无效专用目录可能重置 |
| `--no-resume` 后仍复用量化 | 外层与量化检查点独立，全新GPTQ用新 `calibration_dir` |
| 单PTE为什么还要 `.so` | PTE含模型数据，不含runner/QNN平台实现/DSP运行环境 |
| KV比完整上下文少1 | 看最后采样token的 `pending_tokens` |

顶层先看最后一条 `EXPORT_START`，再看 `output/logs/<阶段>.log`；量化细节在 `calibration/logs/`，完成声明为 `QUANTIZATION_COMPLETE.json`。`--check` 不证明设备动态库可用，不替代实际推理。

## 10. 对应代码位置

| 实现 | 本文对应逻辑 |
| --- | --- |
| [workflow/export_config.py](workflow/export_config.py) | 导出字段、路径基准、SDK优先级、原始文本/CUDA约束 |
| [workflow/export_pipeline.py](workflow/export_pipeline.py) | 检查、自动构建、串行阶段、环境、推理配置、最终打包 |
| [workflow/checkpoint.py](workflow/checkpoint.py) | 工作目录锁、重置及复用校验 |
| [quantization/prepare_model.py](quantization/prepare_model.py) | 逐层GPTQ子流程和校准检查点 |
| [quantization/prepare_safetensors.py](quantization/prepare_safetensors.py) | 原始分片和语料取样、独立FP32/量化链 |
| [build.sh](build.sh) | 基础库、工具链、辅助头文件和本项目编译产物 |
| [export/binding.py](export/binding.py) | 默认/指定QNN Python接口加载 |
| [export/capture.py](export/capture.py) | 真机权重回调采集、运行库路径 |
| [workflow/model_package.py](workflow/model_package.py) | 单PTE打包、主机解包缓存 |
| [workflow/config.py](workflow/config.py)、[workflow/session.py](workflow/session.py) | 推理覆盖、部署、常驻会话 |
| [runner/model_package.cpp](runner/model_package.cpp) | 设备单PTE解包和运行库链接 |
| [workflow/model_config.py](workflow/model_config.py) | 量化、容量、缓存、生成及调度支持范围 |

历史性能、数值和JNI记录见 [README.md](README.md) 与 `docs/`；本文描述当前操作流程，不用历史测试目录代替必需的环境构建。
