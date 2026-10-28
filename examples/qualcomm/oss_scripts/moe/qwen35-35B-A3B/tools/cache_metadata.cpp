#include "System/QnnSystemInterface.h"
#include "../runner/cache_encoding.h"
#include "../runner/qnn_cache.h"
#include <algorithm>
#include <dlfcn.h>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#define FIELD(t, f) ((t).version == QNN_TENSOR_VERSION_1 ? (t).v1.f : (t).v2.f)

static void check(Qnn_ErrorHandle_t code) {
  if (code)
    throw std::runtime_error("QnnSystem error " + std::to_string(code));
}

static std::vector<char> read(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file)
    throw std::runtime_error("open " + path);
  std::vector<char> data(file.tellg());
  file.seekg(0);
  file.read(data.data(), data.size());
  if (!file)
    throw std::runtime_error("read " + path);
  return data;
}

int main(int argc, char** argv) try {
  if (argc != 4 && argc != 5)
    throw std::runtime_error(
        "cache_metadata SDK SHARED_EXPORT OUTPUT [GROUPS]");
  const int groups = argc == 5 ? std::stoi(argv[4]) : 10;
  if (groups < 1 || groups > 10)
    throw std::runtime_error("cache metadata needs 1 through 10 groups");
  void* library = dlopen(
      (std::string(argv[1]) + "/lib/x86_64-linux-clang/libQnnSystem.so")
          .c_str(),
      RTLD_NOW);
  if (!library)
    throw std::runtime_error(dlerror());
  auto providers_fn = reinterpret_cast<Qnn_ErrorHandle_t (*)(
      const QnnSystemInterface_t***, uint32_t*)>(
      dlsym(library, "QnnSystemInterface_getProviders"));
  if (!providers_fn)
    throw std::runtime_error("system provider symbol");
  const QnnSystemInterface_t** providers = nullptr;
  uint32_t count = 0;
  check(providers_fn(&providers, &count));
  if (!count)
    throw std::runtime_error("system provider missing");
  auto system = providers[0]->QNN_SYSTEM_INTERFACE_VER_NAME;
  std::ofstream output(argv[3], std::ios::binary);
  std::vector<CacheEncoding> encodings;
  for (int group = 0; group < groups; group++) {
    const auto root = std::string(argv[2]) + "/group" + std::to_string(group);
    const auto binary = read(root + "/context0.bin");
    QnnSystemContext_Handle_t context = nullptr;
    check(system.systemContextCreate(&context));
    const QnnSystemContext_BinaryInfo_t* info = nullptr;
    Qnn_ContextBinarySize_t size = 0;
    check(system.systemContextGetBinaryInfo(
        context,
        const_cast<char*>(binary.data()),
        binary.size(),
        &info,
        &size));
    auto* graphs = info->version == 1 ? info->contextBinaryInfoV1.graphs
        : info->version == 2          ? info->contextBinaryInfoV2.graphs
                                      : info->contextBinaryInfoV3.graphs;
    const auto graph_count = info->version == 1
        ? info->contextBinaryInfoV1.numGraphs
        : info->version == 2 ? info->contextBinaryInfoV2.numGraphs
                             : info->contextBinaryInfoV3.numGraphs;
    for (int layer = group * 4; layer < group * 4 + 4; layer++) {
      const auto name = "layer" + std::to_string(layer) + "_prefill_attention";
      const QnnSystemContext_GraphInfoV1_t* selected = nullptr;
      for (unsigned index = 0; index < graph_count; index++)
        if (name == graphs[index].graphInfoV1.graphName)
          selected = &graphs[index].graphInfoV1;
      if (!selected)
        throw std::runtime_error("missing graph " + name);
      const auto order_bytes = read(root + "/" + name + ".io");
      if (order_bytes.size() < 8)
        throw std::runtime_error("short IO order");
      const auto* order = reinterpret_cast<const uint32_t*>(order_bytes.data());
      for (int cache = 0; cache < 2; cache++) {
        const auto semantic = (layer % 4 != 3 ? 1 : 4) + cache;
        if (unsigned(semantic) >= order[0])
          throw std::runtime_error("cache IO index");
        const auto prefix =
            "input_" + std::to_string(order[2 + semantic]) + "_";
        bool found = false;
        for (unsigned index = 0; index < selected->numGraphInputs; index++) {
          const auto& t = selected->graphInputs[index];
          if (std::string(FIELD(t, name)).rfind(prefix, 0))
            continue;
          const auto view = qnn_cache_view(t, nullptr);
          if (view.dtype == CacheDType::Float16 && (layer % 4 == 3 || cache != 1))
            throw std::runtime_error("Only GDN recurrent state may use FP16 cache");
          uint64_t elements = 1;
          for (unsigned dim = 0; dim < FIELD(t, rank); dim++)
            elements *= FIELD(t, dimensions)[dim];
          if (layer % 4 == 3) {
            // Handoff stores the complete history, including the current AR128
            // block.
            const auto history = FIELD(t, dimensions)[cache ? 2 : 3];
            if (!history)
              throw std::runtime_error("empty KV history");
            elements += elements / history * 128;
          }
          encodings.push_back({view.scale, view.offset, elements, view.dtype});
          found = true;
        }
        if (!found)
          throw std::runtime_error("missing cache input " + prefix);
      }
    }
    check(system.systemContextFree(context));
  }
  const bool typed = std::any_of(encodings.begin(), encodings.end(), [](const auto& e) {
    return e.dtype != CacheDType::AffineUint16;
  });
  output.write("Q36CACHE", 8);
  const uint32_t header[] = {typed ? 2U : 1U, uint32_t(groups * 4)};
  output.write(reinterpret_cast<const char*>(header), sizeof(header));
  for (const auto& encoding : encodings) {
    output.write(reinterpret_cast<const char*>(&encoding.scale), 4);
    output.write(reinterpret_cast<const char*>(&encoding.offset), 4);
    output.write(reinterpret_cast<const char*>(&encoding.elements), 8);
    if (typed) {
      const uint32_t extra[] = {uint32_t(encoding.dtype), 0};
      output.write(reinterpret_cast<const char*>(extra), sizeof(extra));
    }
  }
  if (!output)
    throw std::runtime_error("write cache metadata");
  printf(
      "CACHE_ENCODINGS_READY layers=%d tensors=%d\n", groups * 4, groups * 8);
  return 0;
} catch (const std::exception& error) {
  fprintf(stderr, "%s\n", error.what());
  return 1;
}
