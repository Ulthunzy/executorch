# Qwen3.5 dense JNI

`Qwen35DenseModule` uses the existing `libexecutorch.so` and QNN multimodal
runner. Its generation methods match `Qwen35MoeModule`; the existing
`LlmModule` and MoE methods keep their descriptors and behavior.

Build Android arm64 with `EXECUTORCH_BUILD_LLAMA_JNI=ON`,
`EXECUTORCH_BUILD_QNN=ON`, `EXECUTORCH_BUILD_KERNELS_LLM=ON` and
`QNN_SDK_ROOT=/hunter/qnn_sdk/src/backends/qnn/2.48.1.260715`.
Package the matching QNN backend, HTP/System/Prepare libraries, the CPU and
HTP DeltaNet packages, and the Stub/Skel for the device architecture.
SM8975 uses the device-provided V85 Stub/Skel because that SDK distribution
has no V85 binaries. The existing `fbjni:0.7.0` Gradle dependency supplies
`libfbjni.so` and `libc++_shared.so`; include them explicitly in a standalone
`app_process` test. Use `jniLibs.useLegacyPackaging = true` so the DSP
libraries exist as files in `applicationInfo.nativeLibraryDir`.

After compiling `executorch_jni`, place
`<build>/extension/android/libexecutorch_jni.so` in the packaging directory as
`arm64-v8a/libexecutorch.so`, together with the matching QNN backend and device
libraries. Point Gradle at that directory so new Java methods are packaged with
the new native implementation, rather than a previous prebuilt library:

```bash
JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 \
  ./extension/android/gradlew -p extension/android \
  :executorch_android:assembleRelease \
  -PexecutorchJniLibsDir=/absolute/path/to/jniLibs
```

Put these exported files in an app-readable model directory:

- `hybrid_llama_qnn.pte`: `kv_forward` and padded `prefill_forward`.
- `tok_embedding_qnn.pte`: token embedding methods.
- `vision_encoder_qnn.pte`: image encoder `forward`.
- The model's original `tokenizer.json`.

The source model/PTE directory is read only. The native backend temporarily
resolves embedded op-package filenames against `nativeLibraryDirectory`;
there is no requirement to create the exporter's `/data/local/tmp` paths.
The DSP search path and backend directory option are restored after each call.
Run one QNN model at a time in a process, since those options are process-wide.
Dense calls on different threads reject overlapping temporary directory changes;
all previous backend and DSP search settings are restored on completion or error.

```java
Qwen35DenseModule.Config config = new Qwen35DenseModule.Config(
    modelDirectory, getApplicationInfo().nativeLibraryDir);
config.systemPrompt = "你是一个有帮助的人工智能助手。";
config.contextLength = 0; // Infer usable prefill capacity, excluding padding slots.
// Defaults: 448×448, patch 16, temporal 2, merge 2. Encoder metadata is checked.
config.imageHeight = 448;
config.imageWidth = 448;

try (Qwen35DenseModule model = new Qwen35DenseModule(config)) {
  Qwen35DenseModule.Callback callback = new Qwen35DenseModule.Callback() {
    @Override public void onResult(String piece) { /* Post to the UI thread. */ }
    @Override public void onStats(String json) { /* Native Stats JSON. */ }
  };
  model.generate("你好。", 128, 0.7f, false, callback);
  model.generateWithImage(imagePath, "请描述图片。", 256, 0.7f, true, callback);
  model.generate("刚才的图片里有什么？", 128, 0.7f, false, callback);
  model.resetContext();
}
```

Construction reads metadata; the first generation loads resident model
resources. Generate on a background thread; callbacks run synchronously on
that thread, with UTF-8 characters assembled across token boundaries.
`thinking` selects the Qwen chat-template prefix and preserves reasoning
markers. `temperature=0` selects greedy sampling. `maxNewTokens` counts the
first sample and EOS. By default EOS stops generation; the additional
`generateWithImage(..., thinking, ignoreEos, callback)` overload can ignore EOS.

Later turns retain KV, GDN state and M-RoPE position offsets. The final sampled
token is fed back on the next turn, and only valid tail tokens count toward
context. No conversation history is silently truncated. When the compiled
prefill capacity is exhausted, call `resetContext()` or export a larger model.
Reset clears cache contents using their quantization zero points without
unloading modules. Statistics use the native Stats serializer/printer, with
`global_kv_tokens` for committed positions, `pending_tokens` for the last
sample still awaiting feedback, and `context_length` for the logical limit.

`stop()` is safe from another thread or the token callback. It takes effect
between graph executions; an in-flight graph cannot be interrupted. An
interrupted prefill or model/callback failure requires `resetContext()` before
retrying. A completed decode cancellation can continue with a new turn.
Concurrent generate/reset calls are rejected. `close()` requests stop, waits
for a running call and releases resources; call it outside the callback.

Device regression test: `../tests/Qwen35DenseSmokeTest.java` covers text/image,
EOS, budgets, reasoning markers, UTF-8, multi-turn cache growth, deterministic
reset, cancellation and concurrent close. It accepts model directory, native
library directory and JPG path through `app_process`.

Validation on 2026-10-27 used the real Qwen3.5-2B hybrid PTE on SM8850,
SDK 2.48.1, and V81 device libraries. Text, internal JPG decode, multi-turn
history, EOS, output budgets, temperature sampling, UTF-8, reset, invalid input,
callback failure, cancellation and concurrent close passed. JNI also compiled
with QNN disabled; the dense API then reports unsupported. The AAR retained
all 67 legacy public classes and 646 member descriptors. The full device test
also passed with the production fbjni 0.7.0 dependency. An old Java
runtime without the dense class loaded the new JNI successfully. SM8975 was
not connected during this JNI validation. Detailed logs and artifact hashes
are in `llama-output/qwen35-dense-jni-20261027/REPORT.md` at the repository root.

The 2026-10-27 review additionally verified concurrent callback `stop()` while
`close()` is waiting, direct C++ runner reuse after stop, cancellation published
before entering generate, the `seq_len=0` fallback and invalid QNN option types.
Decoder input rank/dtype and integer capacity metadata are checked before loading
the model graphs. Full Buck checking remains blocked by the existing missing
`shim_et//tools/build_defs/fbsource_utils.bzl`; SDK 2.48.1 CMake builds, AAR API
checks and SM8850 device regressions passed.
