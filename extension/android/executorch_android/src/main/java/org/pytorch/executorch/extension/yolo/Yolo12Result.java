package org.pytorch.executorch.extension.yolo;

import org.pytorch.executorch.annotations.Experimental;

/** Result for one YOLO12 inference pass. */
@Experimental
public final class Yolo12Result {

  public static final Yolo12Result EMPTY = new Yolo12Result(new Yolo12Detection[0], 0, 0, 0.0f);

  public final Yolo12Detection[] detections;
  public final int imageWidth;
  public final int imageHeight;
  public final float latencyMs;

  public Yolo12Result(
      Yolo12Detection[] detections, int imageWidth, int imageHeight, float latencyMs) {
    this.detections = detections;
    this.imageWidth = imageWidth;
    this.imageHeight = imageHeight;
    this.latencyMs = latencyMs;
  }
}
