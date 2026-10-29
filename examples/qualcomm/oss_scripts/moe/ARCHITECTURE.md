# MoE 代码边界与调用链

## Python

`moe.py → cli.export_main → create_model(model_family) → ModelWrapper`。
模型适配器提供配置及导出后端；公共 `ExportWorkflow` 负责阶段调度、校验、恢复与产物打包。`wrappers/` 定义图组件，`model/` 组合网络，`quantize/strategy.py` 定义量化参数，`quantize/gptq.py` 实现 GPTQ。GDN 图注册与参考计算从 `custom_op/deltanet_support` 引入。

`run_chat.py → cli.run_main → MoeRuntime → ModelInference`。
`inference/encoder.py` 准备视觉输入，`decoder.py` 启动设备对话；`artifacts.py` 与根目录 `package.py` 校验并读取同一个 PTE 容器。导出与设备部署都使用 `model_family` 选择适配器；未知模型报错。

## C++

CLI 通过 `Runner::run_terminal` 进入终端适配层；JNI 将 Java 请求交给 `Runner::invoke`；公共 C++ 调用可使用 `IRunner::generate`。它们进入同一个 `ModelSession`/`Conversation` 和 Prefill、Decode 执行路径。终端展示与 JNI 回调只在入口层不同，模型线程独占上下文和 state。

`model_adapter.h` 定义模型元数据、模板及图查找接口，工厂在会话创建时选择实现。`qwen35_adapter.cpp` 保存 Qwen 的图命名、特殊 token、布局和对话模板。增加模型需要注册相应 Python/C++ 适配器与模型图；不应复制设备部署、线程、QNN 资源管理或 JNI 会话实现。现有 Native kernel 有明确的已编译张量布局要求，注册表不代表任意模型尺寸自动兼容。

`prompt_processor.cpp` 编排 Prefill，`token_generator.cpp` 编排 Decode；阶段内部的图绑定与流水调度位于 `prefill/`、`decode/`。`state_manager` 负责 KV、GDN state 以及阶段交接，`expert_manager` 负责 Decode 专家缓存和预取。`qnn/` 集中管理 Graph、Buffer、共享 context 和共享常量；JNI 不再维护一套 QNN session。

`Runner` 管理加载、取消、回调、reset、关闭与线程回收。上下文在首次输入时建立，后续轮次复用；`reset()` 保留模型资源并清空对话。`Workspace` 维护私有缓存与 memfd，模型文件只读。进程内仍只允许一个活动 MoE 实例，因为底层阶段入口和环境配置是进程级资源；增加模型种类与并发加载多个模型是不同能力。

## 算子与 Native 服务

`custom_op/htp_arch.h` 统一 `QCOM_HTP_ARCH=79|81|85`，并校验 DSP 编译目标。SM8750/SM8850/SM8975 分别对应 V79/V81/V85。数值或硬件差异使用编译定义维护，不复制整套算子。

`deltanet_op_package_htp/` 保存 GDN HTP 实现；`deltanet_support/` 保存参考、注册和精度支持；`lowbit/` 保存 HVX/HMX 计算、N2 展开与 VTCM 布局。`lowbit` 不引用 MoE Runner、不映射专家文件、不建立 RPC 服务。

`runner/native/` 拥有 IDL、DSP worker、FastRPC 服务、bank 映射和缓存失效处理。它准备计算描述符，调用 `lowbit` 内核。bank 准备发生在 worker 成功启动后、执行内核前；保持旧错误码、同步关系和 profiling 时间戳。原有 W2 打包与计算顺序保持不变。

## 构建

根 `CMakeLists.txt` 构建 `libet_infer.so` 和链接它的 CLI；`runner/CMakeLists.txt` 提供共用 source targets，JNI 使用同一个 Runner。Native 服务使用 Hexagon 编译器，将 `custom_op/lowbit` 和 GDN 链接进 `libet_qnnop.so`。

```bash
export QNN_SDK_ROOT=/hunter/qnn_sdk/src/backends/qnn/2.51.200.261021
export HEXAGON_SDK_ROOT=/hunter/qnn_sdk/hexagon/6.7.0.0
export HEXAGON_TOOLS_ROOT=$HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/21.0.04/Tools
export ANDROID_NDK_ROOT=/hunter/android-ndk/android-ndk-r28c
export MOE_SOC_MODEL=SM8850
export MOE_BUILD_DIR=$PWD/build/moe/qwen35-35B-A3B-sm8850
bash examples/qualcomm/oss_scripts/moe/build.sh
```

切换 `MOE_SOC_MODEL` 选择架构；仅构建 Native 服务可调用 `runner/native/build.sh --arch 81 --output <build-dir>`。SDK 必须与宿主导出 binding、设备 runtime 一致。Python 在进程启动时设置动态库路径，Runner 从部署目录显式加载 QNN 库并检查接口版本，避免 Android 系统库覆盖指定 SDK。

## 兼容性与验证

旧 `model.pte`、图签名、W2 权重、cache 布局和 Java 方法签名保持兼容。两库构建更新了 FastRPC 服务名和 DSO 文件名，设备端需要一起部署配套的 ARM/DSP 库，Java 加载 `et_infer`。此次重构没有改 GDN 数学实现或重新量化权重。旧 `models/`、`common/`、`runtime/`、`custom_op/moe/` 与模型目录软链接已移除，源代码只保留一套入口。

本次验证记录见 `docs/layout-validation-20261029.json`：包括 Python/主机测试、三种 ISA 的 Native 构建、SM8850 旧模型包推理、64 槽文本 1024/1024 与图文 256/256 的吞吐/独立内存测量，以及 JNI 生命周期调用。其它 SoC 的构建通过不代表已完成真机性能验证。
