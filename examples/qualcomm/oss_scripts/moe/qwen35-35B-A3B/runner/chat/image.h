#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

struct VisionEncoder;

namespace moe_chat {
struct ImageLayout {
  int height, width, patch, merge, temporal;
  std::array<int, 3> sections;
  explicit ImageLayout(const std::string& root);
  int tokens() const;
};

std::vector<float> image_patches(
    const std::string& path,
    const ImageLayout& layout);
std::vector<float> encode_image(
    VisionEncoder* vision,
    const std::vector<float>& pixels);

class RotaryPositions {
 public:
  explicit RotaryPositions(const std::string& root);
  void reset();
  void append(
      const std::vector<uint64_t>& ids,
      size_t start,
      const ImageLayout* image);
  void save(const std::string& root) const;
  std::vector<float> cos, sin;

 private:
  std::vector<float> base_cos, base_sin;
  int delta = 0;
};
} // namespace moe_chat
