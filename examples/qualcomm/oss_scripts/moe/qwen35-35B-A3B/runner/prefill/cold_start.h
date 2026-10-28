#pragma once

#include <stdexcept>
#include <string>
#include <sys/stat.h>

inline std::string cold_start_table(const std::string& root, bool initial_image) {
#if defined(SM_MOE_DSP_ARCH) && SM_MOE_DSP_ARCH == 81
  const std::string modality = initial_image ? "image" : "text";
  const std::string table = root + "/cold-experts-" + modality + ".i32";
  const std::string manifest = root + "/cold-experts-" + modality + ".json";
  struct stat info {};
  const bool has_table = lstat(table.c_str(), &info) == 0;
  const bool has_manifest = lstat(manifest.c_str(), &info) == 0;
  if (has_table != has_manifest)
    throw std::runtime_error(modality + " cold startup requires table and manifest");
  if (has_table)
    return table;
#else
  (void)initial_image;
#endif
  return root + "/cold-experts.i32";
}
