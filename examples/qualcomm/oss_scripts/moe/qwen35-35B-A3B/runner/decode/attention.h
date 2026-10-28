#pragma once

#include "qnn_front.h"
#include "cache_utils.h"
#include "../runtime_utils.h"
#include "../cache_transfer.h"
#include "../cache_encoding.h"
#include <array>
#include <cmath>

extern std::vector<uint8_t> cache_encoding;
extern std::array<std::vector<uint32_t>, 40> layer_io;

// AR1 的 conv/recurrent state 或 full-attention K/V；这些内存跨 token 常驻。
struct Attention {
  Graph graph;
  bool linear;
  int history = 0;
  std::vector<int> input, output;
  std::vector<uint16_t> initial[2], maps[2];
  size_t state_elements[2]{};
  int initial_tokens = 0;
  bool identity[2];
  bool swap_state =
      getenv("GDN_STATE_SWAP") && std::string(getenv("GDN_STATE_SWAP")) == "1";
  bool append_kv =
      getenv("KV_APPEND_ONLY") && std::string(getenv("KV_APPEND_ONLY")) == "1";
  int last_position = -1;
  std::shared_ptr<Buffer> alias, state[2];

  Attention(
      Runtime& runtime,
      const std::string& root,
      int layer,
      bool is_linear);

  void reset();

  void clear();

  QuantizedCacheView cache_view(int c);

  void execute(
      const float* hidden,
      int position,
      const float* cos,
      const float* sin);

  void feedback();

  void save_cache(const std::string& path, int c);
};
