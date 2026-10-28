#include "../runner/prefill/cold_start.h"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <string>

int main(int argc, char** argv) {
  assert(argc == 2);
  const std::filesystem::path root = argv[1];
  std::filesystem::create_directories(root);
  const auto text = (root / "cold-experts.i32").string();
  const auto image = (root / "cold-experts-image.i32").string();
  const auto manifest = root / "cold-experts-image.json";
  std::ofstream(text) << "text";
  assert(cold_start_table(root.string(), false) == text);
  assert(cold_start_table(root.string(), true) == text);
  std::ofstream(image) << "image";
  bool rejected = false;
  try {
    cold_start_table(root.string(), true);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
#if defined(SM_MOE_DSP_ARCH) && SM_MOE_DSP_ARCH == 81
  assert(rejected);
#else
  assert(!rejected);
#endif
  std::ofstream(manifest) << "{}";
  assert(cold_start_table(root.string(), false) == text);
#if defined(SM_MOE_DSP_ARCH) && SM_MOE_DSP_ARCH == 81
  assert(cold_start_table(root.string(), true) == image);
#else
  assert(cold_start_table(root.string(), true) == text);
#endif
  const auto text_table = (root / "cold-experts-text.i32").string();
  const auto text_manifest = root / "cold-experts-text.json";
  std::ofstream(text_manifest) << "{}";
  rejected = false;
  try {
    cold_start_table(root.string(), false);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
#if defined(SM_MOE_DSP_ARCH) && SM_MOE_DSP_ARCH == 81
  assert(rejected);
#else
  assert(!rejected);
#endif
  std::ofstream(text_table) << "natural text";
#if defined(SM_MOE_DSP_ARCH) && SM_MOE_DSP_ARCH == 81
  assert(cold_start_table(root.string(), false) == text_table);
  assert(cold_start_table(root.string(), true) == image);
#else
  assert(cold_start_table(root.string(), false) == text);
  assert(cold_start_table(root.string(), true) == text);
#endif
  const auto text_target = root / "text-resource";
  std::filesystem::rename(text_table, text_target);
  std::filesystem::create_symlink(text_target, text_table);
#if defined(SM_MOE_DSP_ARCH) && SM_MOE_DSP_ARCH == 81
  assert(cold_start_table(root.string(), false) == text_table);
#else
  assert(cold_start_table(root.string(), false) == text);
#endif
  const auto target = root / "image-resource";
  std::filesystem::rename(image, target);
  std::filesystem::create_symlink(target, image);
#if defined(SM_MOE_DSP_ARCH) && SM_MOE_DSP_ARCH == 81
  assert(cold_start_table(root.string(), true) == image);
#else
  assert(cold_start_table(root.string(), true) == text);
#endif
}
