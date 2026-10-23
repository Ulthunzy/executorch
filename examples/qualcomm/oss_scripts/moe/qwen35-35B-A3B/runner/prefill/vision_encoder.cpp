#include "vision_encoder.h"

#include "qnn_front.h"
#include "../runtime_utils.h"
#include <cmath>

VisionEncoder::VisionEncoder(Runtime& runtime, const std::string& root)
    : graph(runtime, root + "/vision.bin") {
  if (graph.inputs.size() != 1 || graph.outputs.size() != 1 ||
      FIELD(graph.inputs[0], dataType) != QNN_DATATYPE_FLOAT_32 ||
      FIELD(graph.outputs[0], dataType) != QNN_DATATYPE_FLOAT_32)
    throw std::runtime_error("vision requires one FP32 input and output");
  load_image(root + "/pixels.f32");
}

void VisionEncoder::load_image(const std::string& pixels_path) {
  const auto pixels = bytes(pixels_path);
  if (pixels.size() != Graph::size(graph.inputs[0]))
    throw std::runtime_error("vision pixel input size");
  memcpy(graph.input(0), pixels.data(), pixels.size());
}

std::vector<float> VisionEncoder::encode() {
  const auto begin = Clock::now();
  auto& runtime = graph.runtime;
  const auto creates = runtime.context_creates, frees = runtime.context_frees,
             registrations = runtime.memory_registers,
             allocations = runtime.buffer_allocations;
  const auto execute = Clock::now();
  graph.execute();
  const double execute_ms = elapsed(execute);
  const size_t count = Graph::elements(graph.outputs[0]);
  if (!count || count % 2048)
    throw std::runtime_error("vision output hidden dimension");
  const auto* output = static_cast<const float*>(graph.output(0));
  std::vector<float> features(output, output + count);
  for (float value : features)
    if (!std::isfinite(value))
      throw std::runtime_error("nonfinite vision features");
  if (runtime.context_creates != creates || runtime.context_frees != frees ||
      runtime.memory_registers != registrations ||
      runtime.buffer_allocations != allocations)
    throw std::runtime_error("vision changed context or buffer lifetime");
  printf(
      "VISION_ENCODE tokens=%zu hidden=2048 resident=1 execute_ms=%.6f total_ms=%.6f context_creates=0 context_frees=0 memory_registers=0 buffer_allocations=0\n",
      features.size() / 2048,
      execute_ms,
      elapsed(begin));
  return features;
}
