#pragma once

#include <string>
#include <vector>

namespace example {

// Pixels use the same bicubic resize and patch order as the Qwen processor.
std::vector<float> qwen_image_patches(
    const std::string& path,
    int height,
    int width,
    int patch = 16,
    int merge = 2,
    int temporal = 2);

} // namespace example
