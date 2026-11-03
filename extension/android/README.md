# ExecuTorch Android

This directory contains the Android Java/Kotlin binding. The final product is an AAR,
which contains the `.so` libraries for c++ runtime, and `.jar` for Java API, and required
metadata `AndroidManifest.xml`.

## Core contents

Under `extension/android/`,

- `executorch_android/` is the root for the Java `org.pytorch.executorch` package
  - `src/`
    - `androidTest/` contains the android instrumentation test source
    - `main/` contains the Java source
    - `test/` contains the Java unit test source
  - `build.gradle` is the rule to build the Java package.
- `jni/` contains the JNI layer code, which depends on the ExecuTorch c++ runtime library.
- `CMakeLists.txt` is the rule for building the JNI library.

## Build

`scripts/build_android_library.sh` is a helper script to build the Java library (into .jar), native library (into .so), and the packaged AAR file.

The usage is:
```sh
export ANDROID_SDK=/path/to/sdk
export ANDROID_NDK=/path/to/ndk
sh scripts/build_android_library.sh
```

The AAR file will be `extension/android/executorch_android/build/outputs/aar/executorch_android-debug.aar`.
If you set an environment variable `BUILD_AAR_DIR`, then the AAR will be copied to `$BUILD_AAR_DIR/executorch.aar`.
Later, you can copy `$BUILD_AAR_DIR/executorch.aar` to your app directory to use as a library.

Please see [Android building from source](https://pytorch.org/executorch/main/using-executorch-android#building-from-source) for details

## Test

After the library is built,

```sh
# Set up models for testing
sh executorch_android/android_test_setup.sh

# Run unit test
./gradlew :executorch_android:testDebugUnitTest

# Run instrumentation test
./gradlew :executorch_android:connectedAndroidTest
```

## QNN text generation (Lincal / Qwen)

### Legacy App AAR compatibility

The legacy public JVM signatures are retained: `load()`, all `generate()`
overloads, and `Module.loadMethod()` return `int`; the six array/text prefill
overloads return `long` (zero on success). Generation returns native error codes
and also calls `onError`; load/prefill may still throw, as in the legacy AAR.
The legacy `extension.textencoder.TextEncoderModule` is also available when
Llama JNI is built.

```java
LlmModule module = new LlmModule(
    4, modelPath, tokenizerPath, systemPrompt, "lincal", 0.0f,
    Collections.emptyList());
int status = module.load();
if (status == 0) {
  status = module.generateWithChat("你好，你是谁？", 256, callback, false);
}
module.unload(); // Frees the runner; subsequent load/generate starts a new context.
module.resetNative(); // Permanently destroys the wrapper.
```

The legacy constructor automatically selects KV or hybrid from the PTE methods.
QNN repetition penalty defaults to `1.05` over the latest `128` tokens, including
legacy constructors and reloads. Use `qnnRepetitionPenalty(1.0f)` to disable it or
`qnnRepeatLastN(256)` to change the window in the module config builder.
`generateWithChat()` takes **raw user text** and adds the Qwen/Lincal/Gemma3
template. By default there is no system prompt: omitted, null and empty values
do not insert a system message. A nonempty system prompt explicitly passed to
the constructor is sent only when there is no committed history; reset/unload
restores that explicit prompt. Qwen/Lincal ChatML starts subsequent turns with
one newline, matching the identity-evaluation driver. Plain `generate()` still
accepts the caller's formatted input without adding a template.
In the QNN chat API, the integer argument is the
**new-token budget**, while total context is taken from the PTE. In the plain
`generate()` API, `seqLen` remains the total sequence limit.

The final boolean is **echo**, not an enable-thinking switch. Neither value
disables thinking or inserts an empty `<think></think>` block. Stop and wait for
generation to return before unloading or closing. AR-level cancellation keeps
the already committed prefix. During lazy QNN loading, `stop()` is retained and
prevents prefill/generation after loading completes; it does not interrupt QNN's
in-flight model initialization. A subsequent generation call can proceed normally.

Encrypted QNN PTEs from the legacy App are accepted by the existing constructors
(the eight-byte `FF` header, embedded AES-256 key, nonce, ciphertext, and GCM tag).
The embedded key is a compatibility format, not secure key storage. `ETENC1`
containers instead require `.qnnModelKeyHex(key)` on `LlmModuleConfig.create()`;
the key must contain 32 or 64 hexadecimal digits (AES-128/256). Whitespace and a
leading `0x` are accepted. The same option can override a legacy embedded key.
Authentication failures are rejected before model execution. Decryption allocates
one full plaintext model buffer plus a small chunk buffer, without writing plaintext
to disk; budget RAM accordingly. Encrypted containers currently require embedded
weights (no separate `.ptd` data files) and do not use the mmap load mode.
QNN JNI builds now also require `git submodule update --init third-party/boringssl`.
The crypto library is linked statically; its notices are included in the AAR.

Do not combine a new classes.jar with an old JNI library. When using an
incremental CMake build, stage the matching libraries by ABI and pass
`-PexecutorchJniLibsDir=/absolute/path/to/staged-libs` to Gradle. This overrides
the default `cmake-out-android-so` directory, which may contain stale binaries.
The staging directory should contain e.g. `arm64-v8a/libexecutorch.so`,
`libqnn_executorch_backend.so`, and the matching QNN runtime libraries. Only
include ABIs whose JNI library has been rebuilt.

### SM8850 Qwen35 Dense and MoE AAR

The combined SM8850 release uses QNN SDK `2.51.200.261021` and V81 libraries.
Its staging directory must contain `libexecutorch.so`,
`libqnn_executorch_backend.so`, `libQnnDeltaNetOpPackage.so`,
`libQnnDeltaNetOpPackage_HTP.so`, `libet_infer.so`, `libet_qnnop.so`, and the
matching QNN runtime libraries. The MoE runtime keeps its two-library layout;
the combined AAR also carries the separate Dense runtime.

Build from that complete staging directory with native-content validation:

```bash
./extension/android/gradlew -p extension/android \
  :executorch_android:assembleRelease \
  -PexecutorchJniLibsDir="$PWD/llama-output/qwen35-sm8850-aar/jniLibs" \
  -PexecutorchQwen35Modules=dense,moe
```

The additional property checks both Java APIs, required native libraries,
ARM64/Hexagon ELF types and 16 KB Android `LOAD` segment alignment, and rejects
V79/V85 runtime libraries. It catches a MoE-only staging directory or a 4 KB
Android library before the combined release is accepted. Hexagon ELF32 DSP
libraries are checked separately for their architecture; Android ARM64 page
alignment requirements apply to the host libraries.

When consuming a local AAR file, retain the application's `fbjni:0.7.0` and
`nativeloader:0.10.5` dependencies, plus Kotlin stdlib. The fbjni dependency
supplies `libfbjni.so` and `libc++_shared.so`; these are not duplicated in the
combined AAR. The app must extract its DSP libraries as described in the
[Dense JNI](jni/dense/README.md) and [MoE JNI](jni/moe/README.md) instructions.
Local `files(...)` AAR dependencies do not resolve Maven dependencies, so the
consumer must declare these dependencies explicitly and use compatible native
libraries in the final APK. CameraX belongs to the consuming application and
is outside this AAR's dependency set.

Compare a new AAR against an app's existing one:

```bash
python extension/android/jni/tests/check_aar_api.py old.aar new.aar
```

### Configurable generation

Build with QNN and Llama JNI enabled. Ship the Kotlin classes and JNI library
from the same build; private native signatures are not compatible with an older
Java/Kotlin JAR. Existing public `LlmModule` constructors and `generate`
overloads remain available.

```kotlin
val module = LlmModule(
    LlmModuleConfig.create()
        .modulePath("$modelDir/kv_llama_qnn.pte")
        .tokenizerPath("$modelDir/tokenizer.json")
        .modelType(LlmModuleConfig.MODEL_TYPE_QNN_LLAMA)
        .qnnDecoderModel("lincal") // "qwen3" for Qwen3
        .qnnEvalMode(LlmModuleConfig.QNN_EVAL_MODE_KV)
        .qnnSharedBuffer(false)
        .qnnRepetitionPenalty(1.05f) // Default; use 1.0f to disable.
        .qnnRepeatLastN(128) // Default window, in tokens.
        .temperature(0.0f)
        .build()
)
module.load()

val prompt = "<|im_start|>user\n你好，你是谁？<|im_end|>\n<|im_start|>assistant\n"
val generation = LlmGenerationConfig.create()
    .seqLen(1024)
    .maxNewTokens(256)
    .echo(false)
    .ignoreEos(false)
    .build()
module.generate(prompt, generation, callback)
// module.stop() may be called from another thread during generate().
module.resetContext() // Call between independent conversations, not between turns.
module.close()
```

Thinking protections default to enabled for QNN **Lincal/Qwen3 text KV/hybrid**,
including legacy generate/chat calls. Other decoders/backends retain their old
defaults. Explicit protections require single-token ChatML/think markers and
a QNN text runner. Per-generation settings can override each switch:

```kotlin
val generation = LlmGenerationConfig.create()
    .repairThinkEos(true)
    .enableThinkBudget(true)
    .thinkBudget(512)
    .build()
module.generate(formattedPrompt, generation, callback)
```

Both switches default to `true` and can be disabled independently. Set both to
`false` to disable protection without disabling thinking. The budget
defaults to 512 newly generated reasoning tokens; it is not the prefill AR or
SWA window. EOS repair replaces premature EOS with the real closing token,
while the budget switch closes reasoning at its limit. Tokens are fed through
the model and retained in KV/history, not patched only in display text. Neither
switch disables thinking, expands context, overrides cancellation, or guarantees
a correct/nonempty final answer. `ignoreEos(true)` disables unspecified protections
for fixed-length benchmarks; explicitly enabling a protection with it is invalid.
Unspecified protections do not alter multimodal or non-QNN generation;
explicitly requesting them on a non-QNN runner returns `NotSupported`.
The old public methods/constructor and `generateNative` signature are retained;
enabled controls use the additive `generateWithThinkingNative` entry point.

QNN module options (independent of the per-generation settings above):

| Builder method | Meaning / default |
| --- | --- |
| `qnnDecoderModel` | Runner model identifier, default `llama3`; Lincal uses `lincal`. |
| `qnnEvalMode` | `QNN_EVAL_MODE_KV` or `QNN_EVAL_MODE_HYBRID` (default). Must match the PTE methods. |
| `qnnSharedBuffer` | Shared QNN I/O buffers, default `false`. |
| `qnnRepetitionPenalty` | Penalty over recent prompt/generated tokens (including thinking, across turns); default `1.05`, set `1.0` to disable. |
| `qnnRepeatLastN` | Positive repetition window; default `128` tokens. Persists across unload/reload. |
| `qnnModelKeyHex` | AES key for ETENC1; empty by default, permitting legacy embedded keys. |

`seqLen` includes the prompt and prior turns and cannot enlarge a compiled PTE's
context. `maxNewTokens` limits this call's output, including thinking and EOS.
The smaller limit wins. Zero new tokens is a no-op. Generation temperature and
BOS/EOS counts default to `-1` to inherit the module settings; explicit zero
counts suppress BOS/EOS insertion. QNN applies BOS only at context position zero.
`numEos` appends EOS to the **input**; `ignoreEos` controls stopping on **output**.
Counts are limited to 0–127 (`-1` inherits the module setting). QNN inserts the
requested IDs explicitly, independently of the HF tokenizer's postprocessor.
Qwen/Lincal ChatML models have no separate BOS: use `numBos(0)` (the module
default); requesting a BOS is an error. Their input EOS is `<|im_end|>`.
`warming(true)` suppresses the performance report but still consumes context;
call `resetContext()` before starting an independent conversation afterward.

For multi-turn generation, pass only the new formatted turn. The runner retains
the last output token, including EOS, and feeds it into the KV cache with the
next prompt. Do not repeat that token yourself. `maxNewTokens(0)` leaves this
pending token and the existing context unchanged.

QNN native Info logs include `[QNN_GENERATION]` with the effective per-call
temperature, repetition penalty and the effective window. `[QNN_INPUT]` describes
existing KV history and input BOS/EOS insertion; `[QNN_PROMPT]` prints the complete
formatted input in JSON-escaped chunks with byte offsets, even with `echo(false)`.
The logs never enter `onResult`. Rebuild/update the AAR once to get these logs;
the PTE is unchanged. Prompt logs contain conversation content, so disable Info
logging in production builds that must not retain it.

`stop()` during decode preserves the context for subsequent turns. Prefill
cancellation also succeeds without resetting or resending history: each AR
block is committed independently. For Lincal/True-SWA, a stop observed after
inference but before cache commit discards that block's output; earlier blocks
remain committed. Cancellation does not undo the whole input turn. `onStats`
reports the number of committed new input tokens in `prompt_tokens` (excluding
the prior pending output token), and zero `generated_tokens` for prefill-only
cancellation. Unprocessed input is discarded, not queued for automatic replay.
Keep the app's conversation state consistent with this consumed prefix, which
can end inside a chat message. No second KV copy is allocated. In-place recurrent
cache updates or attention-sink eviction must finish the current block's commit
before stopping. Execution errors, unlike ordinary cancellation, can still
invalidate the context and require an explicit reset. A full-context
prefill graph (`prefill AR == context length`) requires a reset between calls;
use a smaller prefill AR or KV mode for incremental multi-turn generation.

For plain `generate()`, pass prompts formatted with the model's chat template;
`generateWithChat()` accepts raw user text instead.
Lincal thinking remains enabled. SWA layout and window size come from the PTE.
For QNN, text-only KV/hybrid generation is supported here; standalone `prefillPrompt`
and image/audio prefill still require a runner that implements `IRunner::prefill`.

See [the QNN JNI smoke test](jni/tests/README.md) for device validation.
