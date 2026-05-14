package org.pytorch.executorch.extension.face;

import android.util.Log;
import androidx.annotation.NonNull;
import com.facebook.jni.annotations.DoNotStrip;
import java.nio.ByteBuffer;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import org.pytorch.executorch.ExecuTorchRuntime;
import org.pytorch.executorch.annotations.Experimental;

/** JNI wrapper for RetinaFace + ArcFace camera inference on top of ExecuTorch. */
@Experimental
public final class FaceRecognitionModule implements AutoCloseable {

  private static final String TAG = "FaceRecognitionModule";
  private static final float DEFAULT_SCORE_THRESHOLD = 0.6f;
  private static final float DEFAULT_NMS_THRESHOLD = 0.4f;

  private final Lock mLock = new ReentrantLock();
  private long mNativeHandle;

  public FaceRecognitionModule(@NonNull String modelPath) {
    this(modelPath, DEFAULT_SCORE_THRESHOLD, DEFAULT_NMS_THRESHOLD);
  }

  public FaceRecognitionModule(
      @NonNull String modelPath, float scoreThreshold, float nmsThreshold) {
    ExecuTorchRuntime.getRuntime();
    ExecuTorchRuntime.validateFilePath(modelPath, "model path");
    mNativeHandle = nativeCreate(modelPath, scoreThreshold, nmsThreshold);
  }

  public int load() {
    mLock.lock();
    try {
      ensureHandle();
      return nativeLoad(mNativeHandle);
    } finally {
      mLock.unlock();
    }
  }

  public FaceRecognitionResult analyzeFromRgba(
      @NonNull ByteBuffer rgbaBuffer,
      int width,
      int height,
      int rowStride,
      int rotationDegrees,
      boolean mirrorHorizontally) {
    mLock.lock();
    try {
      ensureHandle();
      return nativeAnalyzeFromRgba(
          mNativeHandle,
          rgbaBuffer,
          width,
          height,
          rowStride,
          rotationDegrees,
          mirrorHorizontally);
    } finally {
      mLock.unlock();
    }
  }

  public void destroy() {
    mLock.lock();
    try {
      if (mNativeHandle != 0L) {
        nativeDestroy(mNativeHandle);
        mNativeHandle = 0L;
      }
    } finally {
      mLock.unlock();
    }
  }

  @Override
  public void close() {
    destroy();
  }

  private void ensureHandle() {
    if (mNativeHandle == 0L) {
      throw new IllegalStateException("FaceRecognition native handle is not valid.");
    }
  }

  @Override
  protected void finalize() throws Throwable {
    try {
      if (mNativeHandle != 0L) {
        Log.w(TAG, "FaceRecognitionModule finalized without explicit destroy().");
        destroy();
      }
    } finally {
      super.finalize();
    }
  }

  @DoNotStrip
  private static native long nativeCreate(String modelPath, float scoreThreshold, float nmsThreshold);

  @DoNotStrip
  private static native void nativeDestroy(long nativeHandle);

  @DoNotStrip
  private static native int nativeLoad(long nativeHandle);

  @DoNotStrip
  private static native FaceRecognitionResult nativeAnalyzeFromRgba(
      long nativeHandle,
      ByteBuffer rgbaBuffer,
      int width,
      int height,
      int rowStride,
      int rotationDegrees,
      boolean mirrorHorizontally);
}
