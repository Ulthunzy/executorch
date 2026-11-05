# MoE 代码边界与调用链

## Python

`moe.py → cli.export_main → create_model(model_family) → ModelWrapper`。
模型适配器提供配置及导出后端；公共 `ExportWorkflow` 负责阶段调度、校验、恢复与产物打包。`wrappers/` 定义图组件，`model/` 组合网络，`quantize/strategy.py` 定义量化参数，`quantize/gptq.py` 实现 GPTQ。GDN 图注册与参考计算从 `custom_op/deltanet_support` 引入。

阶段通过显式依赖和经过校验的产物就绪通知调度。CUDA 量化任务由一路 BF16 原模型激活生成、最多四路独立层权重量化及一路 CPU embedding 量化组成；GPTQ 数学仍保持原精度。原模型激活顺序推进，已就绪的层并行量化，量化验证输出不进入后续校准链。另一路 CPU 在组内所有 Attention/Router 就绪后编译 QNN。就绪通知通过本次运行专有的管道发布，避免复用残留文件；checkpoint 由调度进程加锁写入。Dense/MoE 的误差统计共用 `backends/qualcomm/quantizer/metrics.py`。 校准设备默认自动选择；无 CUDA 或显式选择 CPU 时，原模型校准和全部权重量化按层串行，禁用激活预备及后台 embedding。CPU 量化与 QNN 编译共用调度资源，避免同时运行；CPU 原模型前向使用现有 PyTorch GDN 参考实现。

`run_chat.py → cli.run_main → MoeRuntime → ModelInference`。
`inference/encoder.py` 准备视觉输入，`decoder.py` 启动设备对话；`artifacts.py` 与根目录 `package.py` 校验并读取同一个 PTE 容器。导出与设备部署都使用 `model_family` 选择适配器；未知模型报错。

## C++

CLI 通过 `Runner::run_terminal` 进入终端适配层；JNI 将 Java 请求交给 `Runner::invoke`；公共 C++ 调用可使用 `IRunner::generate`。它们进入同一个 `ModelSession`/`Conversation` 和 Prefill、Decode 执行路径。终端展示与 JNI 回调只在入口层不同，模型线程独占上下文和 state。

`model_adapter.h` 定义模型元数据、模板及图查找接口，工厂在会话创建时选择实现。`qwen35_adapter.cpp` 保存 Qwen 的图命名、特殊 token、布局和对话模板。增加模型需要注册相应 Python/C++ 适配器与模型图；不应复制设备部署、线程、QNN 资源管理或 JNI 会话实现。现有 Native kernel 有明确的已编译张量布局要求，注册表不代表任意模型尺寸自动兼容。

`prompt_processor.cpp` 编排 Prefill，`token_generator.cpp` 编排 Decode；阶段内部的图绑定与流水调度位于 `prefill/`、`decode/`。`state_manager` 负责 KV、GDN state 以及阶段交接，`expert_manager` 负责 Decode 专家缓存和预取。`qnn/` 集中管理 Graph、Buffer、共享 context 和共享常量；JNI 不再维护一套 QNN session。

`Runner` 管理加载、取消、回调、reset、关闭与线程回收。`load()` 等待Prefill、Decode及专家缓存完成初始化后才报告成功，初始化保持空对话；首轮、后续输入和 reset 后的请求均使用按层 Prefill 调度；`reset()` 保留模型资源并清空对话。`Workspace` 维护私有缓存与 memfd，模型文件只读。进程内仍只允许一个活动 MoE 实例，因为底层阶段入口和环境配置是进程级资源；增加模型种类与并发加载多个模型是不同能力。

Prefill 的工作区容量与本次执行长度独立维护。`PrefillSession::append` 导入 KV/GDN state 和实际驻留的专家/LRU 元数据，按已有工作区切分输入；它与 Prefill 工具共用 `execute_plan` 和同一个 PTE 的按层执行器。每层处理完窗口内全部 AR128 块后再换层，超过工作区时继续下一个窗口。`chunk_start` 是会话内的绝对位置，memo/路由暂存及 LRU 重放使用相对 `sequence_start` 的偏移；正常轮次切换不释放或重新申请图、工作区和专家权重内存。中断时先收尾已提交的读取，再允许 reset 恢复。

Prefill 直接在已分配的 KV 输入缓冲中归约、追加有效位置，不保留 CPU 历史镜像或输出暂存。常驻会话仅在图的历史容量足以覆盖整个逻辑上下文时，省去额外的 `history + AR` 交接副本；短历史图及独立 Prefill 工具仍保留交接窗口。内存交接按实际缓冲长度读取新增 token，磁盘交接仍输出原有扩展窗口格式。GDN 保留原有 DMA state 交换；需要重编码时直接写入输入缓冲。这些复用不改变专家槽数、权重、状态编码或图签名。

`qnn/phase_memory.h` 提供串行阶段的缓冲借用：Decode 仅在类型、维度和量化编码完全一致时复用 Prefill 的 GDN 输入/输出以及 head IO，各图仍在自己的 context 内注册句柄。两阶段保留现有 state 交换；导入完整 GDN state 时不先清空目标，避免共享源被清零。Prefill owner 比 Decode 活得更久，关闭时先释放 Decode 引用。FA KV 仅在 append 模式、量化网格相同且历史长度相差一个 token 时共用 Prefill 分配。进入 Prefill 时原地扩展并右对齐有效历史；多个工作窗口之间保持 Prefill 布局，整段输入完成后才压缩成 Decode 的左对齐布局。行搬移顺序避免覆盖未读取数据，空闲位置随后填入量化零；其它不兼容的布局仍保留独立存储。视觉输入、输出在容量足够时复用 Native Prefill 的输入和 residual 工作区；视觉特征复制完成后才进入 Prefill。较小张量借用较大分配时，通过 SDK 的 HTP shared-buffer 描述符注册完整分配容量，避免同一 FD 被映射成不同长度；共享注册缓存同时区分 FD、偏移和张量描述。

`WeightSource` 回调直接读满 DMA 权重，仅补零对齐尾部，随后按已提供的范围回收可丢弃的源文件页，避免所有 context 加载期间累计保留文件副本。共享 QNN context 初始化完成后，再清理 `weights.bin` 的文件映射驻留及剩余可回收页缓存，保留 DMA 权重、图元数据和源文件描述符；专家读盘文件不在清理范围内。日志分别记录兼容 state/工作区减少的 DMA 字节和源文件页驻留量，文件页回收不计入 DMA 节省。

导出 profile 的 `attention_core=valid_online` 将 FA 核心替换为 `custom_op/attention_op_package_htp` 中的有效长度算子，仍使用同一个 Prefill 调度器。`GRAPHS.json` 记录原始 KV 的容量、前缀布局及两阶段量化网格；`qnn/mapped_cache.h` 验证元数据，提供 FD、逻辑字节数和 epoch 描述符。兼容的 Prefill/Decode 共用原始 KV，只追加实际 token；算子按可见历史搬运和计算，reset 或历史重写更新 epoch。QNN 与 Native RPC 使用同一绝对算子库路径，关闭时先释放 DSP 引用再解除 ARM 映射。独立诊断入口从 `PREFILL_VALID_TOKENS` 或 `PREFILL_TOKENS` 恢复前缀，PPL 检查其位置及缓存内容。

QNN 图编译和执行均使用共享 IO。HTP 编译时，`QnnManager` 根据 `shared_buffer` 将应用输入、输出标记为 `MEMHANDLE`，让 SDK 序列化该提示，避免加载后仍为 RAW IO 保留内部 RPC 缓冲。静态权重和图内张量不修改。Attention/Router、head、视觉导出后使用 SDK context utility 检查 IO 内存类型，并写入组件元数据；宿主 binding 未更新时直接报告重新构建要求。旧 RAW context 仍可推理，但需要复用原量化产物重新编译图才能获得这部分内存收益。

专家暂存区只保存当前层的缺失权重，容量为 `(256 - 常驻槽数) × stride + 4 KiB`。已有专家由 Native 直接读取原常驻槽，不再复制到暂存区。两个 DSP 映射分别持有当前层暂存区和常驻槽，offset 高位区分来源；新 ARM/DSP 的 `prefill_expert_stream=2` 必须配套构建。冷缓存的空槽由同一读线程池先填入，首次 Native 调用前等待这些写入完成并统一刷新该 FD；其余缺失专家继续流式加载。最终 LRU 恢复原有 AR128 调度的逐槽排列。跨槽移动的常驻专家先快照到本层已消费完、且不承载最终保留专家的暂存槽，快照全部完成后再并行写回常驻槽，避免覆盖尚未复制的来源；这一过程复用原暂存区，不新增 DMA 分配。停止和异常沿用读线程排空、会话失效及 reset 恢复流程。

启用视觉时，在首个共享图创建前登记 `vision.bin`，并把视觉与文本 context 放入同一次 `SHARE_RESOURCES` 顺序加载批次，允许 SDK 复用二者互斥的 spill/fill 工作区。文本共享权重继续由 `WeightSource` 提供；视觉完整 binary 由 `BinaryWeights` 提供 DMA 权重，两个回调都保留到对应 context 释放后。视觉关闭时不登记或加载视觉 context；运行期间不改变批次成员，不在每轮对话重载视觉权重。实际内存收益以唯一 DMA 分配实测为准，不能直接把图规划量相加当作节省。

编码相同的 cache 反馈不分配恒等 LUT，直接复制原编码；不同网格仍使用原有查表转换。

旧 `chunk_major` 选项保留弃用提示并转到统一调度；原按块循环保留为私有弃用实现，没有运行入口调用，后续版本将移除。JNI、终端和公开 C++ 生成接口不根据首轮、历史或 reset 选择另一套 Prefill 实现。

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
