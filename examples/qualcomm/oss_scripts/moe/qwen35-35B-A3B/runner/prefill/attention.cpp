#include "attention.h"

Attention::Attention(
    Runtime& r,
    const std::string& root,
    int layer,
    bool is_linear,
    bool cold_fa)
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
    cache_maps[i] = remap(
        graph.outputs[output[i + 2]], cache_owner().inputs[cache_input[i]]);
    identity[i] = true;
    for (int j = 0; j < 65536; j++)
      if (cache_maps[i][j] != j) {
        identity[i] = false;
        break;
      }
    const auto& in = cache_owner().inputs[cache_input[i]];
    const auto& out = graph.outputs[output[i + 2]];
    swap_state[i] = linear && identity[i] &&
        FIELD(in, dataType) == FIELD(out, dataType) &&
        FIELD(in, rank) == FIELD(out, rank) &&
        std::equal(FIELD(in, dimensions),
                   FIELD(in, dimensions) + FIELD(in, rank),
                   FIELD(out, dimensions));
    if (!linear)
      handoff_cache[i].resize(512 * 1024);
    cache_host[i].resize(Graph::elements(cache_owner().inputs[cache_input[i]]));
    cache_temporary[i].resize(Graph::elements(graph.outputs[output[i + 2]]));
  }
}

size_t Attention::cache_bytes(int c) {
  return Graph::size(cache_owner().inputs[cache_input[c]]);
}

size_t Attention::export_bytes(int c) {
  return linear ? cache_bytes(c) : handoff_cache[c].size() * 2;
}

void Attention::reset() {
  for (int c = 0; c < 2; c++) {
    reset_quant(cache_owner(), cache_input[c]);
    if (!linear)
      memcpy(cache_host[c].data(), cache_data(c), cache_host[c].size() * 2);
  }
}

void Attention::prefill(
    const float* hidden,
    int start,
    const float* cos,
    const float* sin,
    bool reuse) {
  if (!reuse)
    memcpy(
        hidden_alias ? hidden_alias->data : graph.input(input[0]),
        hidden,
        128 * 2048 * 4);
  if (!linear) {
    memcpy(graph.input(input[1]), cos + start * 32, 128 * 32 * 4);
    memcpy(graph.input(input[2]), sin + start * 32, 128 * 32 * 4);
    const int history = cold ? 0 : 896, columns = history + 128;
    float* mask = (float*)graph.input(input[3]);
    std::fill(mask, mask + 128 * columns, -255.f);
    for (int t = 0; t < 128; t++) {
      std::fill(
          mask + t * columns + history - start,
          mask + t * columns + history,
          0.f);
      std::fill(
          mask + t * columns + history,
          mask + t * columns + history + t + 1,
          0.f);
    }
  }
  graph.execute();
}

void Attention::feedback() {
  // 输出与下一次输入可能使用不同的量化网格，必须先查表重映射。
  // handoff_cache 保留完整 1024 窗口供 decode 导入，graph 输入则仅保留历史
  // 896。
  for (int c = 0; c < 2; c++) {
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
    const size_t count = cache_temporary[c].size();
    auto* device_input = (uint16_t*)cache_data(c);
    if (linear && identity[c]) {
      memcpy(device_input, graph.output(output[c + 2]), count * 2);
      continue;
    }
    memcpy(cache_temporary[c].data(), graph.output(output[c + 2]), count * 2);
    const uint16_t* from = cache_temporary[c].data();
    auto* to = cache_host[c].data();
    auto& lut = cache_maps[c];
    if (linear) {
      for (size_t i = 0; i < count; i++)
        to[i] = lut[from[i]];
    } else if (c == 0)
      for (int row = 0; row < 512; row++) {
        auto* full = handoff_cache[c].data() + row * 1024;
        memcpy(full, to + row * 896, 896 * 2);
        for (int t = 0; t < 128; t++)
          full[896 + t] = lut[from[row * 128 + t]];
        memcpy(to + row * 896, full + 128, 896 * 2);
      }
    else
      for (int h = 0; h < 2; h++) {
        auto* full = handoff_cache[c].data() + h * 1024 * 256;
        memcpy(full, to + h * 896 * 256, 896 * 256 * 2);
        for (int i = 0; i < 128 * 256; i++)
          full[896 * 256 + i] = lut[from[h * 128 * 256 + i]];
        memcpy(to + h * 896 * 256, full + 128 * 256, 896 * 256 * 2);
      }
    memcpy(device_input, to, cache_host[c].size() * 2);
  }
}
