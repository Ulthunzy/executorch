package org.pytorch.executorch.extension.llm;

import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.nio.charset.CharsetDecoder;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.util.Objects;

/**
 * Qwen3.5 35B-A3B external-expert runtime. Independent of LlmModule.
 *
 * <p>Construct and generate on a background thread. Callbacks run on that same caller thread.
 * The first generation loads the model; subsequent turns reuse context, KV and
 * experts until close(). A bundled PTE is extracted and cached internally. Use an
 * app-private cache directory for workspaceDirectory. Only one instance may be
 * open per process; do not run another QNN model concurrently with this runtime.
 */
public final class Qwen35MoeModule implements AutoCloseable {
  static {
    System.loadLibrary("et_infer");
  }

  public static final class Config {
    public final String modelDirectory, sharedWeightsDirectory, tokenizerPath;
    public final String workspaceDirectory, nativeLibraryDirectory;
    public String systemPrompt = "你是一个有帮助的人工智能助手。";
    /** Zero inherits the exported model metadata; positive values override it. */
    public int contextLength = 0;
    public int expertsPerLayer = 0;
    public boolean enableVision = true;

    public Config(String modelDirectory, String sharedWeightsDirectory, String tokenizerPath,
        String workspaceDirectory, String nativeLibraryDirectory) {
      this.modelDirectory = Objects.requireNonNull(modelDirectory);
      this.sharedWeightsDirectory = Objects.requireNonNull(sharedWeightsDirectory);
      this.tokenizerPath = Objects.requireNonNull(tokenizerPath);
      this.workspaceDirectory = Objects.requireNonNull(workspaceDirectory);
      this.nativeLibraryDirectory = Objects.requireNonNull(nativeLibraryDirectory);
    }

    /** Load one complete model PTE; infer context and expert slots from its metadata. */
    public Config(String modelPte, String workspaceDirectory, String nativeLibraryDirectory) {
      this(modelPte, "", "", workspaceDirectory, nativeLibraryDirectory);
    }
  }

  public interface Callback {
    void onResult(String text);
    default void onStats(String json) {}
  }

  private volatile long handle;

  public Qwen35MoeModule(Config config) {
    Objects.requireNonNull(config);
    handle = createNative(config.modelDirectory, config.sharedWeightsDirectory,
        config.tokenizerPath, config.workspaceDirectory, config.nativeLibraryDirectory,
        Objects.requireNonNull(config.systemPrompt), config.contextLength, config.expertsPerLayer,
        config.enableVision);
  }

  /** Returns JSON statistics. Retains prior turns; use resetContext() for a new conversation. */
  public String generate(
      String prompt, int maxNewTokens, float temperature, boolean thinking, Callback callback) {
    return generateWithImage("", prompt, maxNewTokens, temperature, thinking, callback);
  }

  /** Decode a JPG/PNG file, preprocess and run the resident vision encoder internally. */
  public String generateWithImage(String imagePath, String prompt, int maxNewTokens,
      float temperature, boolean thinking, Callback callback) {
    Stream stream = new Stream(Objects.requireNonNull(callback));
    String stats = generateNative(requireHandle(), Objects.requireNonNull(prompt),
        Objects.requireNonNull(imagePath), maxNewTokens, temperature, thinking, stream);
    stream.finish();
    callback.onStats(stats);
    return stats;
  }

  /** Thread safe; checked between decode steps. Partial input cancellation requires reset. */
  public void stop() {
    stopNative(requireHandle());
  }

  /** Clears history without unloading model/expert buffers. Rejects a concurrent generation. */
  public void resetContext() {
    resetNative(requireHandle());
  }

  /** Stops generation, joins the model thread and releases resources. Idempotent. */
  @Override
  public synchronized void close() {
    if (handle != 0) {
      long previous = handle;
      handle = 0;
      closeNative(previous);
    }
  }

  private long requireHandle() {
    long value = handle;
    if (value == 0)
      throw new IllegalStateException("MoE module is closed");
    return value;
  }

  // BPE can split a UTF-8 character across tokens. JNI passes bytes rather than
  // using modified UTF-8 NewStringUTF, so Chinese and supplementary characters survive.
  private static final class Stream {
    private final Callback callback;
    private final CharsetDecoder decoder = StandardCharsets.UTF_8.newDecoder()
                                               .onMalformedInput(CodingErrorAction.REPLACE)
                                               .onUnmappableCharacter(CodingErrorAction.REPLACE);
    private ByteBuffer pending = ByteBuffer.allocate(0);
    Stream(Callback callback) {
      this.callback = callback;
    }
    private void onBytes(byte[] bytes) {
      emit(bytes, false);
    }
    void finish() {
      emit(new byte[0], true);
    }
    private void emit(byte[] bytes, boolean end) {
      ByteBuffer input = ByteBuffer.allocate(pending.remaining() + bytes.length);
      input.put(pending).put(bytes).flip();
      CharBuffer text = CharBuffer.allocate(input.remaining() + 2);
      decoder.decode(input, text, end);
      pending = ByteBuffer.allocate(input.remaining());
      pending.put(input).flip();
      if (end)
        decoder.flush(text);
      text.flip();
      if (text.hasRemaining())
        callback.onResult(text.toString());
    }
  }

  private static native long createNative(String model, String shared, String tokenizer,
      String workspace, String libraries, String system, int capacity, int experts, boolean vision);
  private static native String generateNative(long handle, String prompt, String image, int count,
      float temperature, boolean thinking, Stream callback);
  private static native void stopNative(long handle);
  private static native void resetNative(long handle);
  private static native void closeNative(long handle);
}
