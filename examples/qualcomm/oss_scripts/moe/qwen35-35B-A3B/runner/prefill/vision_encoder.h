#pragma once

#include "qnn_front.h"

struct VisionEncoder {
  Graph graph;

  VisionEncoder(Runtime& runtime, const std::string& root);

  void load_image(const std::string& pixels_path);

  std::vector<float> encode();
};
