# Qwen3.5 35B-A3B JNI

`Qwen35MoeModule` 加载统一的 `libet_infer.so`，不修改 `LlmModule`、原 QNN runner 或已有 JNI 注册。复用 MoE 目录中的真实 40 层 PTE、外部专家、共享常量、INT4 embedding、分词器、采样器和 Android 图片解码。没有 subprocess、stdin 协议或 stdout/stderr 重定向。

## 共享 Runner

JNI 入口位于本目录 `jni_layer.cpp`，只转换 Java 参数、UTF-8 与回调。会话、取消、状态和 Workspace 由 `examples/qualcomm/oss_scripts/moe/runner/runner.{h,cpp}` 统一实现，`Runner` 继承 `IRunner`。JNI CMake 复用 MoE 根 CMake 的 `et_infer` 目标，与 CLI 共用 Prefill、Decode、QNN 和模型工厂代码。本目录不再维护独立的 session 实现。

Java 新增 `load()` 接口，现有生成方法签名保持不变，库加载名为 `System.loadLibrary("et_infer")`。项目交付两个自研库：ARM64 的 `libet_infer.so` 包含 JNI、Runner、Prefill/Decode、专家管理和共享 context；Hexagon 的 `libet_qnnop.so` 包含 GDN、低 bit 内核和两套独立命名的 FastRPC 服务。源码仍按算子、调度和 JNI 分目录维护。DSP 架构从 Native 构建的 `TARGET.json` 读取，支持 V79/V81/V85 配套构建。QNN SDK 运行库继续单独提供。

首轮和后续轮复用专家 memo、KV/GDN state 和 QNN 已注册的 AR128 IO；工作区容量与本轮有效 token 数、位置分别维护。分组 Prefill 的 DSP 映射保留至模型关闭，避免后续轮次出现 AR128 请求校验失败或映射长度不匹配。JNI 错误同时写入 Android 日志 `ExecuTorchMoE`，Java 异常类型保持兼容。

默认构建保留已验证的性能请求和 mailbox 等待方式。每轮结束释放性能/QoS 请求以及 mailbox 休眠属于待验证的发热优化，默认关闭；仅在显式设置 `MOE_ENABLE_IDLE_POWER_MANAGEMENT=1` 构建环境变量，或 CMake 的同名选项为 `ON` 时开启。该选项不释放权重、图、专家缓存和 state。JNI 失败同时写入 Android 日志 `ExecuTorchMoE`，原有 Java 异常语义保持不变。

所有生成接口共用按层 Prefill 调度；首轮、后续轮、图片输入和 reset 后的新请求均通过绝对起始位置、有效长度及 KV/GDN state 执行同一个 PTE。超过工作区的输入继续按窗口处理。Prefill 尚未完成时被中断，会先完成已提交 IO 的收尾并返回 `requires_reset=true`；reset 后可继续使用同一模型。旧 `chunk_major` 选项已弃用并转到统一调度，后续版本将移除。

## Android 调用

模型目录使用现有 `run_chat.py` 部署生成的布局：`prefill/`、`decode/`、`base-cos.f32`、`base-sin.f32`；图文另需 `VISION.json`、`model-config.json`、`prefill/vision.bin`。共享权重和 tokenizer 分别传入。模型、专家及权重必须解包成普通文件，不能直接传 APK asset URI。App UID 必须能读取这些路径。

导出器现在默认生成单个 `model.pte`，可使用 `new Qwen35MoeModule.Config(modelPte, workspaceDirectory, nativeLibraryDirectory)`，内部校验、缓存解包并定位共享权重和tokenizer；旧的五参数目录构造接口仍保留。解包需要相应磁盘空间。`enableVision=false` 会覆盖包内默认值，纯文本使用此选项可避免加载视觉context。

两种构造方式的 `contextLength` 与 `expertsPerLayer` 默认均为 `0`，表示读取导出元数据。现有旧包若嵌入32槽，可显式设置 `expertsPerLayer=64`。

SM8850/V81 构建默认采用 prefill IO 16、decode IO 8、512 KiB 读取分块，直接加载旧PTE时也生效，无需Python启动参数或重新打包模型。V79/V85保持原调度默认值。

当前SM8850整合版使用 `CMakePresets.json` 的 `sm8850` 入口，从
`build/moe/qwen35-35B-A3B-sm8850` 安装已验证的Native MoE和GDN库，
JNI、Prefill、Decode 和共享 context 链接进同一个 ARM64 库，统一缓存 ABI。
SDK固定为2.51.200.261021。Java接口签名不变；旧PTE内可能仍记录32槽，
需要显式选择本轮验证过的2048上下文和64槽：

```java
Qwen35MoeModule.Config config = new Qwen35MoeModule.Config(
    modelPte, context.getCacheDir().getAbsolutePath(),
    context.getApplicationInfo().nativeLibraryDir);
config.contextLength = 2048;
config.expertsPerLayer = 64;
config.enableVision = false;
Qwen35MoeModule model = new Qwen35MoeModule(config);
model.load(); // 后台完成实际加载，再开放聊天入口
```

从仓库根目录构建并打包当前8850版本：

```bash
cmake --preset sm8850 -S extension/android/jni/moe
cmake --build build/moe/qwen35-jni-sm8850 -j 3
cmake --install build/moe/qwen35-jni-sm8850 \
  --prefix build/moe/qwen35-jni-sm8850/install
```

安装目录 `install/jniLibs/arm64-v8a` 只包含本项目的 `libet_infer.so` 和 `libet_qnnop.so`。原六库安装中的旧文件会在安装时清理。应用另外部署同一 SDK 的 QNN runtime 和设备匹配的 HTP stub/skel。目录构造和单 PTE 构造都从 `nativeLibraryDirectory` 定位两个自研库。

```java
import org.pytorch.executorch.extension.llm.Qwen35MoeModule;

Qwen35MoeModule.Config config = new Qwen35MoeModule.Config(
    modelDirectory, sharedWeightsDirectory, tokenizerPath,
    context.getCacheDir().getAbsolutePath(),
    context.getApplicationInfo().nativeLibraryDir);
config.contextLength = 2048; // 当前8850；必须与实际导出 PLANS.json 一致
config.expertsPerLayer = 64; // 32 / 64，与现有运行时一致
config.enableVision = true;
Qwen35MoeModule model = new Qwen35MoeModule(config);
model.load(); // 后台完成实际加载，再开放聊天入口

// 在后台线程调用；回调就在该调用线程执行，更新 UI 时自行 post 到主线程。
Qwen35MoeModule.Callback callback = new Qwen35MoeModule.Callback() {
  public void onResult(String text) { System.out.print(text); }
  public void onStats(String json) { System.out.println(json); }
};
model.generate("你好，请介绍自己。", 128, 0.0f, false, callback);
model.generateWithImage(imagePath, "请用中文描述这张图片。", 128, 0.0f, false, callback);
model.generate("图片中最显眼的是什么？", 64, 0.0f, false, callback);
model.resetContext();
// model.stop() 可从另一个线程或 token 回调调用。
model.close();
```

构造函数检查部署并创建独立会话目录；**在后台调用 `load()`，等待Prefill/Decode图、共享权重和专家缓存资源就绪，再进入聊天界面**。加载阶段不消耗对话token，`generate()` 负责输入Prefill和输出生成；后续复用所有 context/state/专家缓存，`close()` 统一释放。`resetContext()` 清理历史而不卸载模型。首轮和后续轮都支持任意有效输入长度，实际输入加输出预算不能超过模型的上下文容量。超过单次 PTE 工作区的长输入重复执行 AR128；不足 128 的尾块补齐，通过有效长度屏蔽 padding 的 GDN/conv/KV 更新，位置只按真实 token 推进；AR1 用于输出生成。`PLANS.json` v3 的 `workspace_tokens` 仅为工作区大小，旧 v2 的 `max_prefill_tokens` 也只按工作区解释，不再限制整段输入。短输入保留原有批处理优化。图片解码/encoder 始终在内部完成。旧图如果没有完整上下文 KV/mask 或 GDN `last_token_index`，仍需重新导出；更新 Java 配置无法补齐这些图接口。

单 PTE 构造函数首次调用还会同步解包，应在后台线程调用。除原始 PTE 外，工作目录需要容纳解包资源及 1 GiB 余量；解包前按未缓存资源核算空间，写入各资源前再次检查。完整缓存可直接复用，空间不足会抛异常，部分已完成资源可在后续继续使用。

JNI 的 prefill/decode cache 快照保存在会话私有 memfd，避免把阶段交接写入 UFS；decode 完成导入后立即关闭这些临时快照，后续轮次直接交接常驻 cache view 与专家 LRU 元数据，不再重复建立快照。模型权重仍通过原有只读文件映射与专家缓存读取。长生成后关闭再在同一进程重新创建曾出现 QNN 5005，完整评测暂采用每题独立进程，并保存运行失败记录；不能把短轮次的关闭重建测试视为长上下文已验证。

快照I/O复用自有memfd句柄，使用pread/pwrite且保持原句柄的偏移和生命周期，适配Android对`/proc/self/fd`路径重开的权限限制。`tests/test_moe_file_io.cpp`在主机和SM8975覆盖普通文件、相对符号链接、截断及只读句柄保护。冻结旧部署已通过中文生成和2920-token首轮输入预检；新增AR128多轮/尾块路径的1152容量端侧回归见下文，不能外推到全部上下文容量。

`stop()` 在 decode 步之间及后续轮次的 AR128 块之间响应；首次模型初始化和首次 prefill 为同步阶段，完成当前阶段后响应。若在后续轮次的输入提交中取消，返回统计中的 `requires_reset=true`，下轮前应调用 `resetContext()`。统计 `generation_ms` 只包含生成，不包括模型加载和输入提交；端到端耗时需在 Java 调用外测量。EOS 计入 `generated_tokens`，不传给文本回调。

`generate` / `generateWithImage` 的 `thinking=true` 使用原生思考模板，回调先收到模板中的 `<think>\n`，再收到模型生成的思考、`</think>` 和答案；起始标记不计入生成 token。传 `false` 保持直接回答。思考和答案共用 `maxNewTokens` 预算。`onStats` 另含累计 `kv_cache_tokens`（已写入）、`pending_tokens`（最后一个尚未写入的采样 token）、`context_tokens`（两者之和）、`context_capacity` 和 `enable_thinking`，`resetContext()` 后重新累计；AR128 无效 padding 不计入。

同一实例的 generate/reset 并发调用会立即抛异常；close 会停止并等待模型线程。回调异常会取消当前输出并原样返回 Java。BPE 输出经增量 UTF-8 解码，支持中文和补充平面字符，不使用 JNI modified UTF-8 拼接 token。

当前 MoE 底层为单实例且使用进程环境配置，每进程只允许打开一个 `Qwen35MoeModule`；不要与另一个 QNN 模型同时运行。结束时恢复所修改的环境变量。此限制只作用于新增 MoE 接口，已有接口签名和逻辑不变。

## 构建与打包

MoE 的 `build.sh` 一次构建两个自研库，已经包含 JNI 入口。可直接安装该构建树；独立 JNI CMake 入口复用同一套目标。
SM8850/V81使用上面的preset；下面保留SM8975/V85的构建示例。
使用有效长度AR128路径需要配套prefill图；已有GPTQ权重可复用，不需要重新量化：

```bash
export QNN_SDK_ROOT=/hunter/qnn_sdk/src/backends/qnn/2.51.200.261021
export HEXAGON_SDK_ROOT=/hunter/qnn_sdk/hexagon/6.7.0.0
export HEXAGON_TOOLS_ROOT=$HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/21.0.04/Tools
export ANDROID_NDK_ROOT=/hunter/android-ndk/android-ndk-r28c
export MOE_BUILD_DIR=$PWD/build/moe/qwen35-35B-A3B
bash examples/qualcomm/oss_scripts/moe/build.sh

cmake -S extension/android/jni/moe -B build/moe/qwen35-jni -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-30 -DANDROID_STL=c++_static \
  -DMOE_BUILD_DIR="$MOE_BUILD_DIR" \
  -DEXECUTORCH_ANDROID_BUILD="$PWD/build-android" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/moe/qwen35-jni -j 3
cmake --install build/moe/qwen35-jni --prefix /path/to/app/src/main
```

也可在原 Android CMake 配置中显式开启 `EXECUTORCH_BUILD_QWEN35_MOE_JNI=ON`，默认关闭。新增 DSO 使用独立的 ExecuTorch/backend 注册表及隐藏符号，避免与原 `libexecutorch.so` 中的 runner 同名符号互相覆盖。

安装输出 `jniLibs/arm64-v8a/` 包含两个自研库。`libet_qnnop.so` 由 DSP 加载，Java 只加载 `et_infer`。Prefill 和 Decode 的 RPC 接口分别为 `et_prefill_rpc`、`et_decode_rpc`，都定位到同一个 DSP 文件。Java 类在 Android 工程源码中；对应 ProGuard 保留规则已经加入。

另需部署与设备兼容的 `libQnnHtp.so`、`libQnnSystem.so`、V85 stub，以及设备侧 V85 HTP skel（该测试机已在 `/odm/lib64/rfsa` 提供）。SDK 2.48.1 没有 V85 stub 时应使用此前已验证的设备匹配版本。**SDK 用于链接的 `libcdsprpc.so` 是 stub，不能打进 APK 充当设备 FastRPC 实现。** 使用系统公开的 vendor 库，target SDK 31 及以上在 application 内声明：

```xml
<uses-native-library android:name="libcdsprpc.so" android:required="true" />
```

设置 `android:extractNativeLibs="true"`，保证 nativeLibraryDir 中有真实文件供 QNN/op package 和 DSP 加载。不要将整个 `/vendor/lib64` 塞进 App 的库搜索路径，否则可能把 vendor 私有依赖错误加载到 App namespace。普通 APK 的具体 DSP 访问权限由设备系统决定；设备 Java 进程验证与实际 APK 安装验证应区分记录。

## 回归

2026-10-29 重构 Review：最终 JNI、decode 和共享 provider 同源构建，SM8850 Java 进程的目录接口通过文本、多轮、元数据默认值、加载前后 reset、回调取消、回调异常原样返回、并发 reset 拒绝、单活跃会话限制、禁用图片检查、幂等 close 和关闭后重开；重开时另验证了从元数据默认 32 槽切换到显式 64 槽。`../tests/Qwen35MoeLifecycleTest.java` 支持 `model.pte workspace nativeLibraryDir` 或 `model shared tokenizer workspace nativeLibraryDir` 两种参数形式。

本轮完整 PTE 首次解包遇到设备空间耗尽、进程被终止，尚未完成此路径的推理验证；已清理本次副本并补上空间检查。普通 APK 两种安装方式均返回 `Failure [-99]`，未通过安装验证。目录 Java 进程测试不能证明普通 APK 的 namespace/DSP 权限可用。CLI 的 64 槽文本 1024→1024、图文 256→256 性能及独立内存对照见 [Review 验收记录](../../../../examples/qualcomm/oss_scripts/moe/docs/review-validation-20261029.json)。

2026-10-28当前SM8850/V81、SDK2.51整合版已通过真机Java进程短测：目录构造接口、
文本流式回调、多轮上下文、reset、close及关闭后拒绝调用。上下文为2048、每层64槽；
三轮分别生成18/10/2 tokens，末轮reset后回答“2”。JNI、decode和共享权重提供者同源构建，
修复了新JNI搭配旧decode库的缓存ABI不一致；目录接口同时覆盖已安装的共享权重提供者。
本轮没有重测图片、单PTE构造或普通APK安装。
库指纹及范围见 [JNI验证记录](../../../../build/moe/qwen35-jni-sm8850/VALIDATION.json)，
原始日志见 `llama-output/qwen35-sm8850--1028/jni-mainline-20261028/`。

`../tests/Qwen35MoeSmokeTest.java` 使用真实模型，覆盖中文/emoji 分片、文本流式、多轮、图片路径解码、坏图恢复、取消、回调异常、并发拒绝、reset、close 和同进程重开。与其它 JNI 测试一样可编译成 dex 后用 `app_process` 执行；参数为 model、shared、tokenizer、workspace、nativeLibraryDir、imagePath。模型加载必须串行测试。

单PTE回归使用四个参数：`model.pte workspace nativeLibraryDir imagePath`。若JNI与QNN依赖分别放在两个目录，`-Djava.library.path` 和 `LD_LIBRARY_PATH` 均需包含两者；仅设置后者不足以保证Android classloader加载QNN依赖。

2026-10-26 已在 SM8975（A7SR026915000282）通过上述真实模型 Java/JNI 回归，包括在 token 回调内关闭模型、关闭后同进程重开并进行首轮图文 prefill。相同输入下 JNI 与终端 CLI 的 19 个生成 ID 完全一致；现有 Android 显式源码列表测试通过。该轮是 `app_process` 实测，尚未进行普通 APK 安装验证。日志和指纹见 [回归记录](../../../../llama-output/qwen35-moe-from-safetensors-gptq-20261024/android-jni-20261026/REPORT.md)。

后续 benchmark 压力测试补充：首题连续生成512 token后，同进程重开模型仍复现一次共享context映射错误5005。此前短对话的重开回归通过不能外推为长会话压力稳定。当前批量质量评测采用每题独立设备进程；应用侧应优先长期持有模型并用reset清历史，长输出后的close/reopen稳定性仍待修复。

2026-10-27：单PTE真实Java/JNI回归通过，包含12秒token回调等待和12秒轮间等待、累计KV长度、thinking的起止标记与正常EOS、图文、取消、回调异常及关闭重开。Native RPC在每次forward返回前完成，context、buffer和worker保持常驻，等待用户期间不保留未返回的RPC；需配套更新协议2的decode skeleton和DSO。

同模型/量化/调度的旧显示实现与新实现对照，性能各2轮、固定输出128 token、32槽/层；28轮包含性能及独立内存补测，全部运行前后thermal status为0，同组生成ID逐项一致。

| 输入 | Prefill旧→新 token/s | Decode旧→新 token/s | 峰值内存旧→新 GiB |
| --- | ---: | ---: | ---: |
| 128文本 | 85.40→85.66 | 12.291→12.335 | 3.8897→3.9136 |
| 1024文本 | 434.91→436.55 | 11.148→11.151 | 3.98248→3.98250 |
| 256图文 | 137.87→138.08 | 10.572→10.559 | 4.4606→4.4785 |

Prefill只计LLM，不含加载、图片解码和encoder；decode取原生Stats。开启thinking补充组吞吐变化低于0.6%，同一实例第二轮生成64 token为11.225→11.227 token/s。性能轮无采样器；独立内存测试计入JVM、PSS、去重DMA与短暂交接memfd（56.875MiB），正式JNI库没有测试插桩。峰值差低于1%，未发现明显回退；不视为内存优化或完整质量评测。完整口径与指纹见[验证记录](../../../../examples/qualcomm/oss_scripts/moe/docs/chat-jni-validation-20261027.json)，原始记录保留在 `llama-output/qwen35-jni-performance-regression-20261027/`。本轮仍为 `app_process`，未做普通APK安装测试。
