#include "attention.h"
#include "../qnn/qnn_cache.h"

namespace moe::decode {

Attention::Attention(
    Runtime& runtime,
    const std::string& root,
    int layer,
    bool is_linear)
    : graph(
          runtime,
          root + "/layer" + std::to_string(layer) + "/attention.bin"),
      linear(is_linear) {
  const auto& order = layer_io[layer];
  if (order.size() < 2 || order.size() != 2 + order[0] + order[1] ||
      order[0] != (linear ? 3 : 6) || (order[1] != 4 && order[1] != 7))
    throw std::runtime_error("AR1 IO order");
  for (unsigned i = 0; i < order[0]; i++)
    input.push_back(graph.index(true, order[2 + i]));
  for (unsigned i = 0; i < order[1]; i++)
    output.push_back(graph.index(false, order[2 + order[0] + i]));
  for (int c = 0; c < 2; c++) {
    const int target = input[c + (linear ? 1 : 4)];
    state[c] = graph.inbuf[target];
    if (!linear)
      history = Graph::elements(graph.inputs[target]) / 512;
    const auto source = read_cache_encoding(cache_encoding, layer * 2 + c);
    auto old = bytes(
        root + "/cache-prefill/layer" + std::to_string(layer) + "-cache" +
        std::to_string(c) + ".u16");
    if (old.size() != source.elements * 2)
      throw std::runtime_error("prefill cache metadata");
    const auto destination = qnn_cache_view(graph.inputs[target], nullptr);
    state_elements[c] = Graph::elements(graph.inputs[target]);
    const int source_window = linear ? 0 : source.elements / 512;
    initial_tokens = getenv("PREFILL_VALID_TOKENS")
        ? atoi(getenv("PREFILL_VALID_TOKENS"))
        : getenv("PREFILL_TOKENS") ? atoi(getenv("PREFILL_TOKENS"))
                                   : 128;
    if (linear) {
      if (state_elements[c] != source.elements)
        throw std::runtime_error("GDN state layout");
      initial[c].resize(source.elements);
    } else {
      if (!source_window || source.elements % 512 ||
          state_elements[c] != 512 * size_t(history) || initial_tokens < 1 ||
          initial_tokens > source_window || initial_tokens > history)
        throw std::runtime_error("KV cache layout or prefix length");
      // Store only real prefix tokens; unused decode capacity is quantized
      // zero.
      initial[c].resize(512 * size_t(initial_tokens));
    }
    copy_cache_range(
        {reinterpret_cast<uint16_t*>(old.data()),
         source.elements,
         source.scale,
         source.offset, source.dtype},
        {initial[c].data(), initial[c].size(), destination.scale,
         destination.offset, destination.dtype},
        linear ? CacheLayout::State
               : (c ? CacheLayout::Value : CacheLayout::Key),
        linear ? 0 : source_window - initial_tokens,
        0,
        linear ? source.elements : initial_tokens);
    maps[c] = grid_map(graph.outputs[output[c + 2]], graph.inputs[target]);
    identity[c] = true;
    for (size_t i = 0; i < maps[c].size(); i++)
      if (maps[c][i] != i) {
        identity[c] = false;
        break;
      }
  }
}

QuantizedCacheView Attention::cache_view(int c) {
  if (c < 0 || c >= 2)
    throw std::runtime_error("Invalid cache index");
  return qnn_cache_view(
      graph.inputs[input[c + (linear ? 1 : 4)]], state[c]->data);
}

void Attention::reset() {
  last_position = initial_tokens - 1;
  for (int c = 0; c < 2; ++c) {
    auto destination = cache_view(c);
    clear_cache(destination);
    copy_cache_range(
        {initial[c].data(),
         initial[c].size(),
         destination.scale,
         destination.offset, destination.dtype},
        destination,
        linear ? CacheLayout::State
               : (c ? CacheLayout::Value : CacheLayout::Key),
        0,
        linear || append_kv ? 0 : history - initial_tokens,
        linear ? initial[c].size() : initial_tokens);
  }
}

void Attention::clear() {
  last_position = -1;
  for (int c = 0; c < 2; ++c)
    clear_cache(cache_view(c));
}

void Attention::execute(
    const float* hidden,
    int position,
    const float* cos,
    const float* sin) {
  last_position = position;
  memcpy(alias->data, hidden, 2048 * 4);
  if (!linear) {
    if (position < 0 || position > history)
      throw std::runtime_error("FA position");
    memcpy(graph.input(input[1]), cos + position * 32, 32 * 4);
    memcpy(graph.input(input[2]), sin + position * 32, 32 * 4);
    auto* mask = static_cast<float*>(graph.input(input[3]));
    std::fill(mask, mask + history + 1, -255.f);
    if (append_kv) {
      std::fill(mask, mask + position, 0.f);
      mask[history] = 0.f;
    } else
      std::fill(mask + history - position, mask + history + 1, 0.f);
  }
  if (const char* directory = getenv("FA_CAPTURE_ROOT"); directory && !linear &&
      position == 1024 &&
      graph.name.find("/layer3/attention.bin") != std::string::npos) {
    for (size_t i = 0; i < graph.inputs.size(); i++) {
      auto name = std::string(FIELD(graph.inputs[i], name));
      save(
          std::string(directory) + "/" + name + ".raw",
          graph.input(i),
          Graph::size(graph.inputs[i]));
    }
  }
  if (const char* directory = getenv("GDN_CAPTURE"); directory && linear &&
      position ==
          (getenv("PREFILL_VALID_TOKENS") ? atoi(getenv("PREFILL_VALID_TOKENS"))
               : getenv("PREFILL_TOKENS") ? atoi(getenv("PREFILL_TOKENS"))
                                          : 128) &&
      graph.name.find("/layer0/attention.bin") != std::string::npos) {
    for (size_t i = 0; i < graph.inputs.size(); i++)
      save(
          std::string(directory) + "/" + FIELD(graph.inputs[i], name) + ".raw",
          graph.input(i),
          Graph::size(graph.inputs[i]));
  }
  graph.execute();
}

void Attention::feedback() {
  // 只有布局和量化网格都一致时，GDN 才能交换句柄而不复制 state。
  // append 模式只写当前 token 的 K/V；历史槽无需每步整体搬移。
  for (int c = 0; c < 2; c++) {
    auto* from = static_cast<const uint16_t*>(graph.output(output[c + 2]));
    auto* to = static_cast<uint16_t*>(state[c]->data);
    if (linear && identity[c] && swap_state) {
      const int in = input[c + 1], out = output[c + 2];
      if (Graph::size(graph.inputs[in]) != Graph::size(graph.outputs[out]) ||
          FIELD(graph.inputs[in], dataType) !=
              FIELD(graph.outputs[out], dataType))
        throw std::runtime_error("state swap layout");
      if (FIELD(graph.inputs[in], rank) != FIELD(graph.outputs[out], rank) ||
          !std::equal(
              FIELD(graph.inputs[in], dimensions),
              FIELD(graph.inputs[in], dimensions) +
                  FIELD(graph.inputs[in], rank),
              FIELD(graph.outputs[out], dimensions)))
        throw std::runtime_error("state swap dimensions");
      // Both memory handles already belong to this context and have identical
      // layouts.
      std::swap(
          FIELD(graph.inputs[in], memHandle),
          FIELD(graph.outputs[out], memHandle));
      std::swap(graph.inbuf[in], graph.outbuf[out]);
      state[c] = graph.inbuf[in];
      graph.bindings[FIELD(graph.inputs[in], id)] = graph.inbuf[in];
      graph.bindings[FIELD(graph.outputs[out], id)] = graph.outbuf[out];
    } else if (linear) {
      if (identity[c])
        memcpy(to, from, state_elements[c] * 2);
      else
        for (size_t i = 0; i < state_elements[c]; i++)
          to[i] = maps[c][from[i]];
    } else if (append_kv && last_position < history) {
      if (c == 0)
        for (int row = 0; row < 512; row++)
          to[row * history + last_position] =
              identity[c] ? from[row] : maps[c][from[row]];
      else
        for (int h = 0; h < 2; h++)
          for (int i = 0; i < 256; i++)
            to[h * history * 256 + last_position * 256 + i] =
                identity[c] ? from[h * 256 + i] : maps[c][from[h * 256 + i]];
    } else if (c == 0)
      for (int row = 0; row < 512; row++) {
        memmove(to + row * history, to + row * history + 1, (history - 1) * 2);
        to[row * history + history - 1] =
            identity[c] ? from[row] : maps[c][from[row]];
      }
    else
      for (int h = 0; h < 2; h++) {
        memmove(
            to + h * history * 256,
            to + h * history * 256 + 256,
            (history - 1) * 256 * 2);
        if (identity[c])
          memcpy(
              to + h * history * 256 + (history - 1) * 256,
              from + h * 256,
              256 * 2);
        else
          for (int i = 0; i < 256; i++)
            to[h * history * 256 + (history - 1) * 256 + i] =
                maps[c][from[h * 256 + i]];
      }
  }
}

void Attention::save_cache(const std::string& path, int c) {
  // 文件交接口径统一为有效历史靠右，屏蔽内部 append 布局与窗口布局的差异。
  if (linear || !append_kv) {
    save(path, state[c]->data, state_elements[c] * 2);
    return;
  }
  const int valid = std::min(history, last_position + 1);
  const auto q =
      FIELD(graph.inputs[input[c + 4]], quantizeParams).scaleOffsetEncoding;
  std::vector<uint16_t> ordered(state_elements[c], uint16_t(-q.offset));
  const auto* from = static_cast<const uint16_t*>(state[c]->data);
  if (c == 0)
    for (int row = 0; row < 512; row++)
      memcpy(
          ordered.data() + row * history + history - valid,
          from + row * history,
          valid * 2);
  else
    for (int h = 0; h < 2; h++)
      memcpy(
          ordered.data() + h * history * 256 + (history - valid) * 256,
          from + h * history * 256,
          valid * 256 * 2);
  save(path, ordered.data(), ordered.size() * 2);
}

std::vector<uint8_t> cache_encoding;
std::array<std::vector<uint32_t>, 40> layer_io;

} // namespace moe::decode
