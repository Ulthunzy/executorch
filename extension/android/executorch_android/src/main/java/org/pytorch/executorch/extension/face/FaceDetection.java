package org.pytorch.executorch.extension.face;

import org.pytorch.executorch.annotations.Experimental;

/** One face detection in source-image coordinates. */
@Experimental
public final class FaceDetection {

  public final float score;
  public final int left;
  public final int top;
  public final int right;
  public final int bottom;
  public final boolean primary;
  public final boolean registrationEligible;
  public final String qualityMessage;

  public FaceDetection(
      float score,
      int left,
      int top,
      int right,
      int bottom,
      boolean primary,
      boolean registrationEligible,
      String qualityMessage) {
    this.score = score;
    this.left = left;
    this.top = top;
    this.right = right;
    this.bottom = bottom;
    this.primary = primary;
    this.registrationEligible = registrationEligible;
    this.qualityMessage = qualityMessage;
  }
}
