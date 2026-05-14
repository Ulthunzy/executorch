package org.pytorch.executorch.extension.yolo;

import org.pytorch.executorch.annotations.Experimental;

/** One YOLO12 inference result packaged with the corresponding RGBA frame. */
@Experimental
public final class Yolo12FrameResult {

  public static final Yolo12FrameResult EMPTY =
      new Yolo12FrameResult(new byte[0], 0, 0, 0, 0L, new Yolo12Detection[0], 0.0f);

  public final byte[] rgbaBytes;
  public final int imageWidth;
  public final int imageHeight;
  public final int rowStride;
  public final long timestampNs;
  public final Yolo12Detection[] detections;
  public final float latencyMs;

  public Yolo12FrameResult(
      byte[] rgbaBytes,
      int imageWidth,
      int imageHeight,
      int rowStride,
      long timestampNs,
      Yolo12Detection[] detections,
      float latencyMs) {
    this.rgbaBytes = rgbaBytes;
    this.imageWidth = imageWidth;
    this.imageHeight = imageHeight;
    this.rowStride = rowStride;
    this.timestampNs = timestampNs;
    this.detections = detections;
    this.latencyMs = latencyMs;
  }
}
