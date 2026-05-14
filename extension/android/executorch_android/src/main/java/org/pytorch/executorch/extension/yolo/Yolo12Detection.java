package org.pytorch.executorch.extension.yolo;

import org.pytorch.executorch.annotations.Experimental;

/** One YOLO12 detection in source-image coordinates. */
@Experimental
public final class Yolo12Detection {

  public final int classId;
  public final float score;
  public final int left;
  public final int top;
  public final int right;
  public final int bottom;

  public Yolo12Detection(int classId, float score, int left, int top, int right, int bottom) {
    this.classId = classId;
    this.score = score;
    this.left = left;
    this.top = top;
    this.right = right;
    this.bottom = bottom;
  }
}
