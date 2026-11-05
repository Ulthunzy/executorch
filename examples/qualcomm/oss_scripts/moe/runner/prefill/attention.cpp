#include "attention.h"
#include "../qnn/qnn_cache.h"

Attention::Attention(
    Runtime& r,
    const std::string& root,
    int layer,
    bool is_linear,
    bool cold_fa,
    int resident_capacity)
    : graph(
          r,
          root +
              (cold_fa && !is_linear ? "/cold/layer" + std::to_string(layer)
                                     : "/layer" + std::to_string(layer)) +
              "/forward_0.bin"),
      linear(is_linear),
      cold(cold_fa && !is_linear) {
  if (cold)
    cache_graph = std::make_unique<Graph>(
        r, root + "/layer" + std::to_string(layer) + "/forward_0.bin", true);
  // 编译图的参数次序不同，统一映射成 hidden、RoPE/mask、cache 的逻辑次序。
  std::vector<int> im = linear ? std::vector<int>{0, 2, 1}
      : cold                   ? std::vector<int>{0, 1, 2, 3}
                               : std::vector<int>{0, 1, 2, 5, 4, 3};
  std::vector<int> om =
      linear ? std::vector<int>{3, 2, 0, 1} : std::vector<int>{3, 2, 1, 0};
  if (linear && graph.inputs.size() == 4)
    im.push_back(3);
  if (graph.borrowed_context) {
    const auto mapping = bytes(
        std::string(getenv("SHARED_MODEL_ROOT")) + "/group" +
        std::to_string(layer / 4) + "/layer" + std::to_string(layer) +
        "_prefill_attention.io");
    const auto* order = reinterpret_cast<const uint32_t*>(mapping.data());
    if (mapping.size() !=
            4 * (2 + graph.inputs.size() + graph.outputs.size()) ||
        order[0] != graph.inputs.size() || order[1] != graph.outputs.size())
      throw std::runtime_error("shared prefill IO order");
    im.assign(order + 2, order + 2 + order[0]);
    om.assign(order + 2 + order[0], order + 2 + order[0] + order[1]);
  }
  for (int i : im)
    input.push_back(graph.index(true, i));
  for (int i : om)
    output.push_back(graph.index(false, i));
  for (int i = 0; i < 2; i++) {
    cache_input[i] = cold ? cache_graph->index(true, i ? 3 : 4)
                          : input[i + (linear ? 1 : 4)];
    if (!linear && FIELD(graph.inputs[cache_input[i]], dataType) == QNN_DATATYPE_INT_32) {
      mapped[i].initialize(graph.outputs[output[i + 2]], layer, i, false);
      mapped[i].storage = std::make_shared<Buffer>(r, Graph::size(mapped[i].metadata), false, false);
      mapped[i].storage->shared_views = true;
    }
    cache_maps[i] = remap(graph.outputs[output[i + 2]], cache_tensor(i));
    identity[i] = true;
    for (size_t j = 0; j < cache_maps[i].size(); j++)
      if (cache_maps[i][j] != j) {
        identity[i] = false;
        break;
      }
    const auto& in = cache_tensor(i);
    const auto& out = graph.outputs[output[i + 2]];
    swap_state[i] = linear && identity[i] &&
        FIELD(in, dataType) == FIELD(out, dataType) &&
        FIELD(in, rank) == FIELD(out, rank) &&
        std::equal(FIELD(in, dimensions),
                   FIELD(in, dimensions) + FIELD(in, rank),
                   FIELD(out, dimensions));
    if (!linear) {
      const auto elements = Graph::elements(in);
      if (elements % 512 || !elements)
        throw std::runtime_error("prefill KV history layout");
      const int current = elements / 512;
      if (i && current != history)
        throw std::runtime_error("prefill K/V history mismatch");
      history = current;
      if (!mapped[i].active && (resident_capacity <= 0 || history < resident_capacity))
        handoff_cache[i].resize(512 * (history + 128));
    }
  }
}

size_t Attention::cache_bytes(int c) {
  return mapped[c].active ? mapped[c].storage->size : Graph::size(cache_owner().inputs[cache_input[c]]);
}

size_t Attention::export_bytes(int c) {
  return handoff_cache[c].empty() ? cache_bytes(c)
                                : handoff_cache[c].size() * 2;
}

void Attention::save_cache(const std::string& path, int c) {
  if (mapped[c].active) {
    const auto source = qnn_cache_view(cache_tensor(c), cache_data(c));
    std::vector<uint16_t> ordered(source.elements);
    QuantizedCacheView destination{ordered.data(), ordered.size(), source.scale, source.offset};
    clear_cache(destination);
    const size_t valid = mapped_start + valid_tokens;
    copy_cache_range(source, destination, c ? CacheLayout::Value : CacheLayout::Key, 0, history-valid, valid);
    save(path, ordered.data(), ordered.size()*2);
    return;
  }
  if (linear || !handoff_cache[c].empty()) {
    save(path, export_data(c), export_bytes(c));
    return;
  }
  // Existing diagnostic files include an extra AR128 window. A resident
  // conversation keeps all valid tokens in the live history instead.
  const auto source = qnn_cache_view(
      cache_owner().inputs[cache_input[c]], cache_data(c));
  std::vector<uint16_t> expanded(512 * (history + 128));
  const QuantizedCacheView destination{
      expanded.data(), expanded.size(), source.scale, source.offset,
      source.dtype};
  clear_cache(destination);
  copy_cache_range(
      source,
      destination,
      c ? CacheLayout::Value : CacheLayout::Key,
      0,
      128,
      history);
  save(path, expanded.data(), expanded.size() * 2);
}

void Attention::reset() {
  for (int c = 0; c < 2; c++) {
    if (mapped[c].active) {
      clear_cache(qnn_cache_view(cache_tensor(c), cache_data(c)));
      mapped[c].invalidate();
    } else reset_quant(cache_owner(), cache_input[c]);
  }
}

void Attention::prefill(
    const float* hidden,
    int start,
    const float* cos,
    const float* sin,
    int valid,
    bool reuse) {
  if (valid < 1 || valid > 128 || start < 0)
    throw std::runtime_error("invalid AR128 valid length");
  valid_tokens = valid;
  mapped_start = start;
  if (linear) {
    if (input.size() == 4)
      *static_cast<int32_t*>(graph.input(input[3])) = valid - 1;
    else if (valid != 128)
      throw std::runtime_error("Re-export GDN with AR128 padding support");
  }
  if (!reuse)
    memcpy(
        hidden_alias ? hidden_alias->data : graph.input(input[0]),
        hidden,
        128 * 2048 * 4);
  if (!linear) {
    auto* cos_input = static_cast<float*>(graph.input(input[1]));
    auto* sin_input = static_cast<float*>(graph.input(input[2]));
    std::copy(cos + start * 32, cos + (start + valid) * 32, cos_input);
    std::copy(sin + start * 32, sin + (start + valid) * 32, sin_input);
    std::fill(cos_input + valid * 32, cos_input + 128 * 32, 1.f);
    std::fill(sin_input + valid * 32, sin_input + 128 * 32, 0.f);
    const int history = cold ? 0 : this->history, columns = history + 128;
    if (start < 0 || start > history)
      throw std::runtime_error("prefill position exceeds KV history");
    if (mapped[0].active) {
      const int lengths[] = {valid, 0, start};
      memcpy(graph.input(input[3]), lengths, sizeof(lengths));
    } else {
    float* mask = (float*)graph.input(input[3]);
    std::fill(mask, mask + 128 * columns, -255.f);
    for (int t = 0; t < valid; t++) {
      std::fill(
          mask + t * columns + history - start,
          mask + t * columns + history,
          0.f);
      std::fill(
          mask + t * columns + history,
          mask + t * columns + history + t + 1,
          0.f);
    }
    // Padding rows still need a finite softmax, but never enter saved KV.
    for (int t = valid; t < 128; ++t)
      mask[t * columns + history] = 0.f;
    }
    if (mapped[0].active)
      for (int c = 0; c < 2; ++c)
        mapped[c].publish(graph, cache_input[c], identity[0] && identity[1]);
  }
  graph.execute();
}

void Attention::feedback() {
  // 输出与下一次输入可能使用不同的量化网格，必须先查表重映射。
  // Handoff includes the current block; the next graph input retains history.
  for (int c = 0; c < 2; c++) {
    if (mapped[c].active) {
      copy_cache_range(qnn_cache_view(graph.outputs[output[c+2]], graph.output(output[c+2])),
          qnn_cache_view(cache_tensor(c), cache_data(c)), c ? CacheLayout::Value : CacheLayout::Key,
          0, mapped_start, valid_tokens);
      continue;
    }
    if (swap_state[c]) {
      const int in = cache_input[c], out = output[c + 2];
      // 两端已注册且布局、UINT16 网格相同，复用输出 buffer 作为下一块 state。
      std::swap(
          FIELD(graph.inputs[in], memHandle),
          FIELD(graph.outputs[out], memHandle));
      std::swap(graph.inbuf[in], graph.outbuf[out]);
      graph.bindings[FIELD(graph.inputs[in], id)] = graph.inbuf[in];
      graph.bindings[FIELD(graph.outputs[out], id)] = graph.outbuf[out];
      continue;
    }
    const size_t count = Graph::elements(graph.outputs[output[c + 2]]);
    auto* device_input = (uint16_t*)cache_data(c);
    const auto* from = static_cast<const uint16_t*>(
        graph.output(output[c + 2]));
    if (linear && identity[c]) {
      memcpy(device_input, from, count * 2);
      continue;
    }
    auto& lut = cache_maps[c];
    if (linear) {
      for (size_t i = 0; i < count; i++)
        device_input[i] = lut[from[i]];
    } else {
      const auto q = FIELD(cache_owner().inputs[cache_input[c]], quantizeParams)
                         .scaleOffsetEncoding;
      append_prefill_kv(
          {device_input, cache_bytes(c) / 2, q.scale, q.offset},
          from,
          lut.data(),
          handoff_cache[c].empty() ? nullptr : handoff_cache[c].data(),
          c ? CacheLayout::Value : CacheLayout::Key,
          valid_tokens);
    }
  }
}
