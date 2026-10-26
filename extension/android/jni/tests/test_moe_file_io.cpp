#include "executorch/examples/qualcomm/oss_scripts/moe/qwen35-35B-A3B/runner/runtime_utils.h"

#include <cassert>
#include <fcntl.h>
#include <filesystem>
#include <linux/memfd.h>
#include <stdexcept>
#include <sys/syscall.h>
#include <unistd.h>

int main() {
  char temporary[] = "/tmp/qwen35-file-io-XXXXXX";
#ifdef __ANDROID__
  char android_temporary[] = "/data/local/tmp/qwen35-file-io-XXXXXX";
  const std::filesystem::path root(mkdtemp(android_temporary));
#else
  const std::filesystem::path root(mkdtemp(temporary));
#endif
  const auto ordinary = (root / "ordinary.bin").string();
  const std::vector<uint8_t> payload{0, 1, 2, 0, 254, 255};
  save(ordinary, payload.data(), payload.size());
  assert(bytes(ordinary) == payload);
  save(ordinary, payload.data(), 2);
  assert(bytes(ordinary) == std::vector<uint8_t>({0, 1}));
  save(ordinary, nullptr, 0);
  assert(bytes(ordinary).empty());

  const int fd = syscall(SYS_memfd_create, "qwen35-file-io-test", MFD_CLOEXEC);
  assert(fd >= 0);
  std::filesystem::create_directory(root / "prefill");
  std::filesystem::create_directory(root / "decode");
  std::filesystem::create_symlink(
      "/proc/self/fd/" + std::to_string(fd), root / "prefill/cache");
  std::filesystem::create_symlink("../prefill/cache", root / "decode/cache");
  assert(lseek(fd, 37, SEEK_SET) == 37);
  save((root / "prefill/cache").string(), payload.data(), payload.size());
  assert(bytes((root / "decode/cache").string()) == payload);
  assert(lseek(fd, 0, SEEK_CUR) == 37);
  assert(fcntl(fd, F_GETFD) >= 0);
  save((root / "decode/cache").string(), payload.data(), 1);
  assert(bytes((root / "prefill/cache").string()) ==
         std::vector<uint8_t>({0}));
  assert(lseek(fd, 0, SEEK_CUR) == 37);

  save(ordinary, payload.data(), payload.size());
  const int readonly = open(ordinary.c_str(), O_RDONLY | O_CLOEXEC);
  assert(readonly >= 0);
  const auto readonly_path = "/proc/self/fd/" + std::to_string(readonly);
  bool rejected = false;
  try {
    save(readonly_path, payload.data(), 1);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  assert(rejected);
  assert(bytes(ordinary) == payload);
  assert(fcntl(readonly, F_GETFD) >= 0);
  close(readonly);
  close(fd);
  std::filesystem::remove_all(root);
}
