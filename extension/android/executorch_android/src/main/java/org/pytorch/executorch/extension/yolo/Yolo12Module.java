package org.pytorch.executorch.extension.yolo;

import android.util.Log;
import androidx.annotation.NonNull;
import com.facebook.jni.annotations.DoNotStrip;
import java.nio.ByteBuffer;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import org.pytorch.executorch.ExecuTorchRuntime;
import org.pytorch.executorch.annotations.Experimental;

/**
 * JNI wrapper for YOLO12 camera inference on top of ExecuTorch.
 *
 * <p>Warning: These APIs are experimental and subject to change without notice.
 */
@Experimental
public final class Yolo12Module implements AutoCloseable {

  private static final String TAG = "Yolo12Module";
  private static final float DEFAULT_SCORE_THRESHOLD = 0.25f;
  private static final float DEFAULT_NMS_THRESHOLD = 0.45f;

  private final Lock mLock = new ReentrantLock();
  private long mNativeHandle;

  public Yolo12Module(@NonNull String modelPath) {
    this(modelPath, DEFAULT_SCORE_THRESHOLD, DEFAULT_NMS_THRESHOLD);
  }

  public Yolo12Module(@NonNull String modelPath, float scoreThreshold, float nmsThreshold) {
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

  public Yolo12Result detectFromRgba(
      @NonNull ByteBuffer rgbaBuffer,
      int width,
      int height,
      int rowStride,
      int rotationDegrees,
      boolean mirrorHorizontally) {
    mLock.lock();
    try {
      ensureHandle();
      return nativeDetectFromRgba(
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

  public Yolo12FrameResult detectFrameFromRgba(
      @NonNull ByteBuffer rgbaBuffer,
      int width,
      int height,
      int rowStride,
      int rotationDegrees,
      boolean mirrorHorizontally,
      long timestampNs) {
    mLock.lock();
    try {
      ensureHandle();
      return nativeDetectFrameFromRgba(
          mNativeHandle,
          rgbaBuffer,
          width,
          height,
          rowStride,
          rotationDegrees,
          mirrorHorizontally,
          timestampNs);
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
      throw new IllegalStateException("YOLO12 native handle is not valid.");
    }
  }

  @Override
  protected void finalize() throws Throwable {
    try {
      if (mNativeHandle != 0L) {
        Log.w(TAG, "Yolo12Module finalized without explicit destroy().");
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
  private static native Yolo12Result nativeDetectFromRgba(
      long nativeHandle,
      ByteBuffer rgbaBuffer,
      int width,
      int height,
      int rowStride,
      int rotationDegrees,
      boolean mirrorHorizontally);

  @DoNotStrip
  private static native Yolo12FrameResult nativeDetectFrameFromRgba(
      long nativeHandle,
      ByteBuffer rgbaBuffer,
      int width,
      int height,
      int rowStride,
      int rotationDegrees,
      boolean mirrorHorizontally,
      long timestampNs);
}
