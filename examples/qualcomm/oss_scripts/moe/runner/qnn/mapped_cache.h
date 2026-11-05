#pragma once

#include "qnn_front.h"
#include "qnn_cache.h"
#include <nlohmann/json.hpp>
#include <array>
#include <cmath>
#include <limits>

struct MappedCache {
  bool active = false;
  Qnn_Tensor_t metadata{};
  std::array<uint32_t, 4> dimensions{};
  std::shared_ptr<Buffer> storage;
  uint32_t epoch = 0;

  void initialize(const Qnn_Tensor_t& output, int layer, int cache, bool decode) {
    const char* root = getenv("SHARED_MODEL_ROOT");
    if (!root || cache < 0 || cache > 1 || FIELD(output, rank) != 4 ||
        FIELD(output, dataType) != QNN_DATATYPE_UFIXED_POINT_16)
      throw std::runtime_error("Invalid mapped KV output or model root");
    const auto path = std::string(root) + "/group" + std::to_string(layer / 4) +
        "/GRAPHS.json";
    const auto raw = bytes(path);
    const auto manifest = nlohmann::json::parse(raw.begin(), raw.end());
    const auto& document = manifest.at("attention_cache").at(std::to_string(layer));
    if (document.at("descriptor_words").get<int>() != 3 ||
        document.at("layout").get<std::string>() != "prefix")
      throw std::runtime_error("Unsupported mapped KV descriptor");
    const auto& encoding = document.at(decode ? "decode" : "prefill").at(cache);
    const auto capacity = document.at("capacity").get<uint32_t>();
    std::copy(FIELD(output, dimensions), FIELD(output, dimensions) + 4, dimensions.begin());
    dimensions[cache ? 2 : 3] = capacity;
    size_t elements = 1;
    for (auto value : dimensions) {
      if (!value || elements > ((1u << 30) - 1) / 2 / value)
        throw std::runtime_error("Mapped KV descriptor size overflow");
      elements *= value;
    }
    const float scale = encoding.at("scale").get<float>();
    const int32_t offset = encoding.at("offset").get<int32_t>();
    if (encoding.at("elements").get<size_t>() != elements ||
        !std::isfinite(scale) || scale <= 0 || offset > 0 || offset < -65535)
      throw std::runtime_error("Invalid mapped KV encoding");
    metadata = output;
    FIELD(metadata, dimensions) = dimensions.data();
    FIELD(metadata, rank) = 4;
    FIELD(metadata, quantizeParams).scaleOffsetEncoding.scale = scale;
    FIELD(metadata, quantizeParams).scaleOffsetEncoding.offset = offset;
    active = true;
    invalidate();
  }

  void invalidate() {
    if (epoch == std::numeric_limits<uint32_t>::max())
      throw std::runtime_error("Mapped KV generation exhausted");
    ++epoch;
  }

  void publish(Graph& graph, int input, bool exact) const {
    const auto logical_bytes = Graph::size(metadata);
    if (!active || !storage || logical_bytes > storage->size ||
        logical_bytes >= (1u << 30) ||
        Graph::elements(graph.inputs.at(input)) != 3 ||
        FIELD(graph.inputs.at(input), dataType) != QNN_DATATYPE_INT_32)
      throw std::runtime_error("Mapped KV storage is not ready");
    const int32_t descriptor[] = {
        storage->fd,
        int32_t(logical_bytes) | (exact ? int32_t(1u << 30) : 0),
        int32_t(epoch)};
    memcpy(graph.input(input), descriptor, sizeof(descriptor));
  }
};
