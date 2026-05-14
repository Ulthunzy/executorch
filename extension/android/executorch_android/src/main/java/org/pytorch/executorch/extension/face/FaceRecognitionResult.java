package org.pytorch.executorch.extension.face;

import org.pytorch.executorch.annotations.Experimental;

/** Result for one face-recognition inference pass. */
@Experimental
public final class FaceRecognitionResult {

  public static final FaceRecognitionResult EMPTY =
      new FaceRecognitionResult(new FaceDetection[0], 0, 0, 0.0f, null, false, null);

  public final FaceDetection[] detections;
  public final int imageWidth;
  public final int imageHeight;
  public final float latencyMs;
  public final float[] embedding;
  public final boolean primaryRegistrationEligible;
  public final String primaryQualityMessage;

  public FaceRecognitionResult(
      FaceDetection[] detections,
      int imageWidth,
      int imageHeight,
      float latencyMs,
      float[] embedding,
      boolean primaryRegistrationEligible,
      String primaryQualityMessage) {
    this.detections = detections;
    this.imageWidth = imageWidth;
    this.imageHeight = imageHeight;
    this.latencyMs = latencyMs;
    this.embedding = embedding;
    this.primaryRegistrationEligible = primaryRegistrationEligible;
    this.primaryQualityMessage = primaryQualityMessage;
  }
}
