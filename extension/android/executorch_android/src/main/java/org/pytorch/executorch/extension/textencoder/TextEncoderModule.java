package org.pytorch.executorch.extension.textencoder;

import com.facebook.jni.HybridData;
import com.facebook.jni.annotations.DoNotStrip;
import org.pytorch.executorch.ExecuTorchRuntime;
import org.pytorch.executorch.annotations.Experimental;

/** Compatibility wrapper for the legacy sentence-embedding encoder. */
@Experimental
public class TextEncoderModule implements AutoCloseable {
  @DoNotStrip private final HybridData mHybridData;

  public TextEncoderModule(String modulePath, int hiddenSize, int maxSeqLen) {
    ExecuTorchRuntime.getRuntime();
    ExecuTorchRuntime.validateFilePath(modulePath, "model path");
    mHybridData = initHybrid(modulePath, hiddenSize, maxSeqLen);
  }

  @DoNotStrip
  private static native HybridData initHybrid(String modulePath, int hiddenSize, int maxSeqLen);

  @DoNotStrip
  private native float[] encodeNative(int[] inputIds, float[] attentionMask);

  @DoNotStrip
  private native float[] encodeTextNative(String tokenizerPath, String text);

  public synchronized float[] encodeText(String tokenizerPath, String text) {
    if (!mHybridData.isValid()) {
      throw new IllegalStateException("TextEncoderModule has been destroyed");
    }
    ExecuTorchRuntime.validateFilePath(tokenizerPath, "tokenizer path");
    return encodeTextNative(tokenizerPath, text);
  }

  public synchronized float[] encode(int[] inputIds, float[] attentionMask) {
    if (!mHybridData.isValid()) {
      throw new IllegalStateException("TextEncoderModule has been destroyed");
    }
    if (inputIds == null || attentionMask == null || inputIds.length == 0
        || inputIds.length != attentionMask.length) {
      throw new IllegalArgumentException("inputIds and attentionMask must have equal nonzero lengths");
    }
    return encodeNative(inputIds, attentionMask);
  }

  public synchronized void resetNative() {
    mHybridData.resetNative();
  }

  @Override
  public void close() {
    resetNative();
  }
}
