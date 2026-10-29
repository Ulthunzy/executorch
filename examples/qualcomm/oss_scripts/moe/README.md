# Qualcomm MoE

导出入口为 `moe.py`，推理入口为 `run_chat.py`。模型通过 `model/base.py` 和注册表接入；Python 工作流、C++ Runner、JNI 共用同一套部署格式与执行实现。当前内置 Qwen3.5/3.6 35B-A3B 适配器，支持 SM8750/V79、SM8850/V81、SM8975/V85 编译。

本模块服务本仓库的共享导出流程，推理使用配套的共享模型产物；非共享导出不属于支持和验收范围。

## 结构

```text
moe/
├── __init__.py                 # 模型注册与创建
├── moe.py / run_chat.py        # 导出 / 推理入口
├── package.py / checkpoint.py  # PTE 容器与导出恢复
├── model/                     # 模型结构、配置、网络层
├── wrappers/                  # 图拆分、导出组件与工作流
├── quantize/                  # GPTQ、权重打包与量化策略
├── inference/                 # 部署、视觉编码、对话启动
├── runner/                    # 唯一的设备端执行实现
│   ├── runner.h / runner.cpp  # IRunner、会话线程与生命周期
│   ├── model_adapter.*        # 模型接口与工厂
│   ├── qwen35_adapter.cpp     # Qwen 图查找、维度、模板
│   ├── prompt_processor.cpp   # Prefill
│   ├── token_generator.cpp    # Decode
│   ├── state_manager.*        # KV/GDN state 与阶段交接
│   ├── expert_manager.*       # 专家缓存、预取与 IO
│   ├── model_package.cpp      # 设备端 PTE 解包
│   ├── qnn/                   # QNN 图、共享常量、Buffer
│   └── native/                # FastRPC、DSP 服务、bank 映射
├── configs/qwen35_35b_a3b/     # 模型、导出与设备配置
└── docs/
```

GDN 与低比特计算分别调用 [`custom_op/deltanet_op_package_htp`](../../custom_op/deltanet_op_package_htp/) 和 [`custom_op/lowbit`](../../custom_op/lowbit/)。`lowbit` 不负责专家文件映射、RPC 或模型流程。JNI 只转换 Java 参数和回调，调用这里的 `Runner`。

## 两库构建与交付

`build.sh` 输出两个自研库：`libet_infer.so` 在 ARM64 运行，统一包含 JNI、Runner、Prefill/Decode 和共享 context；`libet_qnnop.so` 在 Hexagon 运行，统一包含 GDN、低 bit 内核和两套 FastRPC 服务。两阶段服务使用独立的 `et_prefill_rpc`、`et_decode_rpc` 接口，Native QHPI 适配不覆盖 QNN 的同名函数。V79/V81/V85 按目标分别编译，源码分工保持不变。

命令行 `qwen35_moe_runner` 与诊断入口 `prefill_handoff_runner` 是链接 `libet_infer.so` 的薄入口。JNI 也使用同一库，Java 方法签名保留，加载名为 `et_infer`。已有 PTE 无需因合库重新导出或量化。QNN SDK 运行库单独部署。

环境设置完成后构建并安装：

```bash
MOE_SOC_MODEL=SM8850 MOE_BUILD_DIR="$PWD/build/moe/qwen35-two-libs-sm8850" \
  bash examples/qualcomm/oss_scripts/moe/build.sh
cmake --install build/moe/qwen35-two-libs-sm8850/cmake \
  --prefix "$PWD/build/moe/qwen35-two-libs-sm8850/install"
```

`install/jniLibs/arm64-v8a/` 只安装上述两个自研库；重复安装时清理旧六库文件。C/C++ 命令行入口头文件安装到 `install/include/et_infer/infer_api.h`。Android 工程也可继续通过 `extension/android/jni/moe` CMake 入口构建。

`NATIVE_TARGET.json` 记录实际 Native 架构、SDK 路径、profile 配置及 RPC/库校验值。`MOE_BUILD_NATIVE=0` 仅复用与本次配置一致的完整 Native 构建；旧构建缺少该记录时先完整构建一次。成功完成 ARM 与 DSP 构建后才生成 `TARGET.json`，部署前校验配对产物；同一构建目录禁止并发写入。导出缓存也跟踪 CMake 校验文件和符号导出表。

CLI、C 接口与 JNI 共用进程级会话保护，同时只允许一个活动模型会话。JNI 可在不同线程创建与关闭，关闭并完成当前调用后可重新创建。C 接口返回非零表示执行或参数错误，不向 C 调用方传播标准 C++ 异常。

## SM8850 导出

在仓库根目录初始化环境后直接运行。`env.sh` 默认使用 QNN SDK 2.51，下面显式指定以便核对：

```bash
conda activate et_npu
source env.sh
export CUDA_VISIBLE_DEVICES=0
export QNN_SDK_ROOT=/hunter/qnn_sdk/src/backends/qnn/2.51.200.261021
export PYTHONPATH="$PWD/..:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/lib/x86_64-unknown-linux-gnu:$QNN_SDK_ROOT/lib/x86_64-linux-clang:${LD_LIBRARY_PATH:-}"

python examples/qualcomm/oss_scripts/moe/moe.py \
  -b build-android -m SM8850 -s 3B15AK00GLW00000 \
  --max_seq_len 2048 --prefill_ar_len 128 --cache-experts 64 \
  --calib-image-dataset coco128 \
  -a llama-output/qwen35-sm8850-full-1029
```

`-b` 指预编译的 ExecuTorch Android runtime；`-m/-s/-a` 分别选择 SoC、设备和输出目录。SDK 默认读取 `QNN_SDK_ROOT`；不需要指定 `--source`。文本量化结果保存在 `<output>/quantized/`，Native 构建保存在 `<output>/build/`。有效量化结果自动复用，缺失或损坏时从 safetensors 修复/准备；校验覆盖完整产物、原始 checkpoint、tokenizer、数值配方及校准语料；完整缓存标记必须与量化模型的配方一致。修改 GPTQ 参数、目标或量化版本会使内层断点失效，避免复用旧权重后写入新配方。缓存校验和写入使用同一个工作区锁。导出工作区重建会保留这两个缓存目录，也可识别旧输出记录中的外部校准目录。

视觉编码器默认用 [COCO128](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco128.yaml) 的全部 128 张图片。`--calib-image-dataset coco8` 改用 [COCO8](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8.yaml) 的全部 8 张图片作快速试跑。`vision-data` 阶段从官方发布地址准备数据，下载包和图片通过 SHA256 校验，缓存于 `llama-output/.calibration-datasets/`；有效缓存不会重复下载，校准源和图片列表写入 `VISION.json`。无需安装 Ultralytics。`--image_path` 仍可指定自己的校准图片，与内置数据集选项互斥。文本 GPTQ 使用 `--corpus`，未指定时采用对应 SoC 配方中的语料。

原 `--config examples/qualcomm/oss_scripts/moe/configs/qwen35_35b_a3b/export_sm8850.json --output <directory>` 方式仍受支持；显式配置保留原路径约定，命令行参数可覆盖。`--print-plan` 只查看导出阶段且不会下载数据，`--check` 检查前置条件。实际执行包含数据准备、构建、CUDA GPTQ、文本/视觉图导出、权重捕获与 PTE 打包，中断后重复原命令续跑。`quantize` 日志中的 `QUANTIZATION_REUSED` 表示已直接复用完整量化模型。输出目录内的 `MODEL_CONFIG.json` 是各阶段共用的生效配方。

```bash
PYTHONPATH="$PWD/.." python examples/qualcomm/oss_scripts/moe/run_chat.py \
  --config /your/output/directory/device.json \
  --weights-path /your/output/directory/model.pte \
  --cache-experts 64 --prefill-io-workers 16 --io-workers 8 --read-chunk-kib 512 \
  --prompt "你好，请介绍一下自己。" --max-new-tokens 256 --temperature 0 \
  --no-enable-thinking --no-ignore-eos --no-with-vision --interactive --stream
```

`--build-dir` 可覆盖设备配置中的 runner 构建目录。构建与详细调用链见 [ARCHITECTURE.md](ARCHITECTURE.md)；Android 见 [JNI 文档](../../../../extension/android/jni/moe/README.md)。旧模型包的 entry 名称与 Java 方法签名保持兼容，应用加载库名改为 `et_infer`。

本机两库构建目录为 `build/moe/qwen35-two-libs-sm8850`，运行已有导出包时可增加 `--build-dir build/moe/qwen35-two-libs-sm8850`。JNI 安装产物位于该目录的 `install/jniLibs/arm64-v8a`。性能与内存对照见 [两库验证记录](docs/two-libraries-validation-20261029.json)；此前目录重构记录见 [结构验证](docs/layout-validation-20261029.json)。
