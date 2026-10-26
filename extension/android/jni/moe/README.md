# Qwen3.5 35B-A3B JNI

新增独立的 `Qwen35MoeModule` 和 `libqwen35_moe_jni.so`，不修改 `LlmModule`、原 QNN runner 或已有 JNI 注册。复用 MoE 目录中的真实 40 层 PTE、外部专家、共享常量、INT4 embedding、分词器、采样器和 Android 图片解码。没有 subprocess、stdin 协议或 stdout/stderr 重定向。

## Android 调用

模型目录使用现有 `run_chat.py` 部署生成的布局：`prefill/`、`decode/`、`base-cos.f32`、`base-sin.f32`；图文另需 `VISION.json`、`model-config.json`、`prefill/vision.bin`。共享权重和 tokenizer 分别传入。模型、专家及权重必须解包成普通文件，不能直接传 APK asset URI。App UID 必须能读取这些路径。

```java
import org.pytorch.executorch.extension.llm.Qwen35MoeModule;

Qwen35MoeModule.Config config = new Qwen35MoeModule.Config(
    modelDirectory, sharedWeightsDirectory, tokenizerPath,
    context.getCacheDir().getAbsolutePath(),
    context.getApplicationInfo().nativeLibraryDir);
config.contextLength = 1152; // 必须与导出 PLANS.json 一致，不能只改此数值扩上下文
config.expertsPerLayer = 32; // 16 / 32 / 64，与现有运行时一致
config.enableVision = true;
Qwen35MoeModule model = new Qwen35MoeModule(config);

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

构造函数检查部署并创建独立会话目录；**首次 generate 才加载模型并执行 AR128 prefill**，后续复用所有 context/state/专家缓存，`close()` 统一释放。`resetContext()` 清理历史而不卸载模型。新导出支持后续轮次按 AR128 分块：不足 128 的尾块补齐，通过有效长度屏蔽 padding 的 GDN/conv/KV 更新，位置只按真实 token 推进；AR1 用于输出生成。此功能需要重新导出带 `last_token_index` 的 prefill 图并重新构建 Native skeleton，旧部署不会自动获得该语义。图片解码/encoder 始终在内部完成。首轮输入上限取自 `PLANS.json` 的 `max_prefill_tokens`，旧部署默认为 1024（包含模板和图片 token，padding 不计入逻辑上下文），并需给输出留出上下文空间。输出长度按实际 context capacity 校验。更长首轮输入需重新编译 full-attention 的 cache/mask 并打包 chunk_major 计划，不能仅修改 Java 配置或 JSON。

JNI 的 prefill/decode cache 快照保存在会话私有 memfd，避免把阶段交接写入 UFS；decode 完成导入后立即关闭这些临时快照，后续轮次直接交接常驻 cache view 与专家 LRU 元数据，不再重复建立快照。模型权重仍通过原有只读文件映射与专家缓存读取。长生成后关闭再在同一进程重新创建曾出现 QNN 5005，完整评测暂采用每题独立进程，并保存运行失败记录；不能把短轮次的关闭重建测试视为长上下文已验证。

快照I/O复用自有memfd句柄，使用pread/pwrite且保持原句柄的偏移和生命周期，适配Android对`/proc/self/fd`路径重开的权限限制。`tests/test_moe_file_io.cpp`在主机和SM8975覆盖普通文件、相对符号链接、截断及只读句柄保护。冻结旧部署已通过中文生成和2920-token首轮输入预检；本次新增 AR128 多轮/尾块路径尚待端侧回归。

`stop()` 在 decode 步之间及后续轮次的 AR128 块之间响应；首次模型初始化和首次 prefill 为同步阶段，完成当前阶段后响应。若在后续轮次的输入提交中取消，返回统计中的 `requires_reset=true`，下轮前应调用 `resetContext()`。统计 `generation_ms` 只包含生成，不包括模型加载和输入提交；端到端耗时需在 Java 调用外测量。EOS 计入 `generated_tokens`，不传给文本回调。

同一实例的 generate/reset 并发调用会立即抛异常；close 会停止并等待模型线程。回调异常会取消当前输出并原样返回 Java。BPE 输出经增量 UTF-8 解码，支持中文和补充平面字符，不使用 JNI modified UTF-8 拼接 token。

当前 MoE 底层为单实例且使用进程环境配置，每进程只允许打开一个 `Qwen35MoeModule`；不要与另一个 QNN 模型同时运行。结束时恢复所修改的环境变量。此限制只作用于新增 MoE 接口，已有接口签名和逻辑不变。

## 构建与打包

先用 MoE 的 `build.sh` 构建匹配的 V85 native、decode DSO 和 op package（如前一版本已有构建，也需重建 host 库以包含生命周期修复），随后独立构建本 JNI。使用有效长度 AR128 路径需要重新导出配套 prefill 图；已有 GPTQ 权重可复用，不需要重新量化：

```bash
export QNN_SDK_ROOT=/hunter/qnn_sdk/src/backends/qnn/2.48.1.260715
export HEXAGON_SDK_ROOT=/hunter/qnn_sdk/hexagon/6.7.0.0
export HEXAGON_TOOLS_ROOT=$HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/21.0.04/Tools
export ANDROID_NDK_ROOT=/hunter/android-ndk/android-ndk-r28c
export MOE_BUILD_DIR=$PWD/build/moe/qwen35-35B-A3B
bash examples/qualcomm/oss_scripts/moe/qwen35-35B-A3B/build.sh

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

安装输出 `jniLibs/arm64-v8a/` 包含 JNI、decode、共享权重提供者、GDN 包及两阶段 DSP skel。两个 native MoE skel 在 APK 中重命名为 `libqwen35_prefill_skel.so` / `libqwen35_decode_skel.so`；会话目录分别映射回各阶段所需的 `libinteger_rpc_skel.so`。Java 类在 Android 工程源码中；对应 ProGuard 保留规则已经加入。

另需部署与设备兼容的 `libQnnHtp.so`、`libQnnSystem.so`、V85 stub，以及设备侧 V85 HTP skel（该测试机已在 `/odm/lib64/rfsa` 提供）。SDK 2.48.1 没有 V85 stub 时应使用此前已验证的设备匹配版本。**SDK 用于链接的 `libcdsprpc.so` 是 stub，不能打进 APK 充当设备 FastRPC 实现。** 使用系统公开的 vendor 库，target SDK 31 及以上在 application 内声明：

```xml
<uses-native-library android:name="libcdsprpc.so" android:required="true" />
```

设置 `android:extractNativeLibs="true"`，保证 nativeLibraryDir 中有真实文件供 QNN/op package 和 DSP 加载。不要将整个 `/vendor/lib64` 塞进 App 的库搜索路径，否则可能把 vendor 私有依赖错误加载到 App namespace。普通 APK 的具体 DSP 访问权限由设备系统决定；设备 Java 进程验证与实际 APK 安装验证应区分记录。

## 回归

`../tests/Qwen35MoeSmokeTest.java` 使用真实模型，覆盖中文/emoji 分片、文本流式、多轮、图片路径解码、坏图恢复、取消、回调异常、并发拒绝、reset、close 和同进程重开。与其它 JNI 测试一样可编译成 dex 后用 `app_process` 执行；参数为 model、shared、tokenizer、workspace、nativeLibraryDir、imagePath。模型加载必须串行测试。

2026-10-26 已在 SM8975（A7SR026915000282）通过上述真实模型 Java/JNI 回归，包括在 token 回调内关闭模型、关闭后同进程重开并进行首轮图文 prefill。相同输入下 JNI 与终端 CLI 的 19 个生成 ID 完全一致；现有 Android 显式源码列表测试通过。该轮是 `app_process` 实测，尚未进行普通 APK 安装验证。日志和指纹见 [回归记录](../../../../llama-output/qwen35-moe-from-safetensors-gptq-20261024/android-jni-20261026/REPORT.md)。

后续 benchmark 压力测试补充：首题连续生成512 token后，同进程重开模型仍复现一次共享context映射错误5005。此前短对话的重开回归通过不能外推为长会话压力稳定。当前批量质量评测采用每题独立设备进程；应用侧应优先长期持有模型并用reset清历史，长输出后的close/reopen稳定性仍待修复。
