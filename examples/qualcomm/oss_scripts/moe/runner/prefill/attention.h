#pragma once

#include "../qnn/qnn_front.h"
#include "cache_utils.h"
#include "../runtime_utils.h"
#include "../state_manager.h"
#include <array>
#include <cmath>

// 单层 AR128 Attention，隐藏维度 2048；GDN 的两个 cache 是 conv/recurrent
// state。full attention 的历史长度来自编译图，再追加本批 128 个位置。
struct Attention {
  Graph graph;
  std::shared_ptr<Buffer> hidden_alias;
  std::unique_ptr<Graph> cache_graph;
  int cache_input[2];
  bool linear, cold;
  int history = 0;
  int valid_tokens = 128;
  std::vector<int> input, output;
  std::vector<uint16_t> cache_maps[2], cache_host[2], cache_temporary[2];
  bool identity[2], swap_state[2]{};
  std::vector<uint16_t> handoff_cache[2];

  Attention(
      Runtime& r,
      const std::string& root,
      int layer,
      bool is_linear,
      bool cold_fa = false);

  Graph& cache_owner() {
    return cold ? *cache_graph : graph;
  }

  void* cache_data(int c) {
    return cache_owner().input(cache_input[c]);
  }

  size_t cache_bytes(int c);

  void* export_data(int c) {
    return linear ? cache_data(c) : handoff_cache[c].data();
  }

  size_t export_bytes(int c);

  void reset();

  void prefill(
      const float* hidden,
      int start,
      const float* cos,
      const float* sin,
      int valid = 128,
      bool reuse = false);

  void feedback();

  float* hidden() {
    return (float*)graph.output(output[0]);
  }

  float* residual() {
    return (float*)graph.output(output[1]);
  }
};
