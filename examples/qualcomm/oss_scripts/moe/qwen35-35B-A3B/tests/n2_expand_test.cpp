#include "../native/N2Utils.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main() {
  constexpr int maximum = 32768;
  constexpr int guard = 128;
  constexpr uint8_t canary = 0xa5;
  constexpr int levels[] = {-7, -2, 2, 7};
  alignas(128) std::array<uint8_t, 2 * maximum + 2 * guard> storage;
  auto* buffer = storage.data() + guard;
  unsigned cases = 0;
  auto check = [&](const std::vector<uint8_t>& source) {
    const int bytes = source.size();
    std::vector<uint8_t> expected(2 * bytes, 0);
    for (int i = 0; i < bytes; ++i)
      for (int code = 0; code < 4; ++code) {
        const int weight = levels[(source[i] >> (2 * code)) & 3];
        const auto nibble = uint8_t(weight < 0 ? weight + 16 : weight);
        expected[2 * i + code / 2] |= nibble << (4 * (code % 2));
      }
    storage.fill(canary);
    if (bytes)
      std::memcpy(buffer, source.data(), bytes);
    n2_expand_in_place(buffer, bytes);
    for (int i = 0; i < 2 * bytes; ++i)
      if (buffer[i] != expected[i]) {
        std::fprintf(
            stderr,
            "arch=%d bytes=%d case=%u index=%d actual=%u expected=%u\n",
            __HEXAGON_ARCH__, bytes, cases, i, buffer[i], expected[i]);
        std::exit(1);
      }
    for (int i = 0; i < guard; ++i)
      if (storage[i] != canary)
        std::abort();
    for (std::size_t i = guard + 2 * bytes; i < storage.size(); ++i)
      if (storage[i] != canary)
        std::abort();
    ++cases;
  };
  check({});
  for (int bytes = 128; bytes <= maximum; bytes += 128) {
    std::vector<uint8_t> source(bytes);
    for (int pattern = 0; pattern < 8; ++pattern) {
      uint32_t random = 0x6e326f70u ^ uint32_t(bytes * 17 + pattern);
      for (int i = 0; i < bytes; ++i) {
        random ^= random << 13;
        random ^= random >> 17;
        random ^= random << 5;
        switch (pattern) {
          case 0:
            source[i] = uint8_t(i);
            break;
          case 1:
            source[i] = uint8_t(i + 128);
            break;
          case 2:
            source[i] = 0;
            break;
          case 3:
            source[i] = 255;
            break;
          case 4:
            source[i] = i % 2 ? 0xf0 : 0x0f;
            break;
          default:
            source[i] = uint8_t(random);
        }
      }
      check(source);
    }
  }
  for (int bytes : {128, 256, 384, 512, 640})
    for (int code = 0; code < 256; ++code)
      check(std::vector<uint8_t>(bytes, uint8_t(code)));
  std::printf("PASS N2 arch=%d cases=%u\n", __HEXAGON_ARCH__, cases);
}
