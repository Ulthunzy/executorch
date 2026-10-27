package org.pytorch.executorch.extension.llm;

import com.facebook.jni.HybridData;
import com.facebook.jni.annotations.DoNotStrip;
import java.io.File;
import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.nio.charset.CharsetDecoder;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.util.Objects;
import org.pytorch.executorch.ExecuTorchRuntime;

/**
 * Resident QNN Qwen3.5 dense runtime, with the same generation methods as Qwen35MoeModule.
 * Construct/generate on a background thread. Callbacks execute on the calling thread.
 * Do not run another QNN model concurrently in this process.
 */
@DoNotStrip
public final class Qwen35DenseModule implements AutoCloseable {
  public static final class Config {
    public String decoderPath, embeddingPath, encoderPath, tokenizerPath;
    public final String nativeLibraryDirectory;
    public String systemPrompt = "你是一个有帮助的人工智能助手。";
    /** Zero infers the logical prefill capacity, excluding padding slots. */
    public int contextLength = 0;
    public int imageHeight = 448, imageWidth = 448;

    public Config(String modelDirectory, String nativeLibraryDirectory) {
      Objects.requireNonNull(modelDirectory);
      decoderPath = new File(modelDirectory, "hybrid_llama_qnn.pte").getPath();
      embeddingPath = new File(modelDirectory, "tok_embedding_qnn.pte").getPath();
      encoderPath = new File(modelDirectory, "vision_encoder_qnn.pte").getPath();
      tokenizerPath = new File(modelDirectory, "tokenizer.json").getPath();
      this.nativeLibraryDirectory = Objects.requireNonNull(nativeLibraryDirectory);
    }
  }

  public interface Callback {
    void onResult(String text);
    default void onStats(String json) {}
  }

  @DoNotStrip private final HybridData mHybridData;
  private Thread running;
  private boolean closing;
  private final Object lifecycle = new Object();

  public Qwen35DenseModule(Config config) {
    Objects.requireNonNull(config);
    ExecuTorchRuntime.getRuntime();
    if (!isAvailableNative()) throw new UnsupportedOperationException("Build Llama JNI with QNN");
    ExecuTorchRuntime.validateFilePath(config.decoderPath, "decoder PTE");
    ExecuTorchRuntime.validateFilePath(config.embeddingPath, "embedding PTE");
    ExecuTorchRuntime.validateFilePath(config.encoderPath, "encoder PTE");
    ExecuTorchRuntime.validateFilePath(config.tokenizerPath, "tokenizer");
    if (config.systemPrompt == null || config.contextLength < 0)
      throw new IllegalArgumentException("Invalid system prompt or context length");
    mHybridData = initHybrid(config.decoderPath, config.embeddingPath, config.encoderPath,
        config.tokenizerPath, config.nativeLibraryDirectory, config.systemPrompt,
        config.contextLength, config.imageHeight, config.imageWidth);
  }

  public String generate(String prompt, int count, float temperature, boolean thinking, Callback callback) {
    return generateWithImage("", prompt, count, temperature, thinking, callback);
  }

  public String generateWithImage(String image, String prompt, int count, float temperature,
      boolean thinking, Callback callback) {
    return generateWithImage(image, prompt, count, temperature, thinking, false, callback);
  }

  /** Setting ignoreEos runs to the token budget, subject to compiled context capacity. */
  public String generateWithImage(String image, String prompt, int count, float temperature,
      boolean thinking, boolean ignoreEos, Callback callback) {
    Objects.requireNonNull(prompt);
    Objects.requireNonNull(image);
    Objects.requireNonNull(callback);
    if (count <= 0 || !Float.isFinite(temperature) || temperature < 0)
      throw new IllegalArgumentException("Invalid token budget or temperature");
    if (!image.isEmpty()) ExecuTorchRuntime.validateFilePath(image, "image");
    synchronized (lifecycle) {
      requireOpen();
      if (running != null) throw new IllegalStateException("Generation is already running");
      prepareNative();
      running = Thread.currentThread();
    }
    try {
      Stream stream = new Stream(callback);
      String stats = generateNative(prompt, image, count, temperature, thinking, ignoreEos, stream);
      stream.finish();
      callback.onStats(stats);
      return stats;
    } finally {
      synchronized (lifecycle) { running = null; lifecycle.notifyAll(); }
    }
  }

  public void stop() {
    synchronized (lifecycle) {
      if (!mHybridData.isValid()) throw new IllegalStateException("Dense module is closed");
      if (!closing) stopNative();
    }
  }

  /** Clear KV/GDN history while retaining loaded modules and buffers. */
  public void resetContext() {
    synchronized (lifecycle) {
      requireOpen();
      if (running != null) throw new IllegalStateException("Stop generation before resetting");
      resetNative();
    }
  }

  @Override public void close() {
    boolean interrupted = false;
    synchronized (lifecycle) {
      if (!mHybridData.isValid()) return;
      if (running == Thread.currentThread())
        throw new IllegalStateException("Cannot close from a generation callback");
      closing = true;
      stopNative();
      while (running != null) {
        try { lifecycle.wait(); } catch (InterruptedException e) { interrupted = true; }
      }
      mHybridData.resetNative();
    }
    if (interrupted) Thread.currentThread().interrupt();
  }

  private void requireOpen() {
    if (closing || !mHybridData.isValid()) throw new IllegalStateException("Dense module is closed");
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


  private static native boolean isAvailableNative();
  private static native HybridData initHybrid(String decoder, String embedding, String encoder,
      String tokenizer, String libraries, String system, int capacity, int height, int width);
  private native String generateNative(String prompt, String image, int count, float temperature,
      boolean thinking, boolean ignoreEos, Stream callback);
  private native void stopNative();
  private native void prepareNative();
  private native void resetNative();
}
