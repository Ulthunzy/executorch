#include "runtime_utils.h"
#include <cerrno>
#include <charconv>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <fstream>
#include <limits.h>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

namespace {
struct BinaryFile {
  int fd = -1;
  bool snapshot = false;

  BinaryFile(const std::string& name, bool writable) {
    std::filesystem::path path(name);
    const std::string prefix = "/proc/self/fd/";
    for (int depth = 0; depth < 16; ++depth) {
      const auto target = path.string();
      if (target.compare(0, prefix.size(), prefix) == 0) {
        int original = -1;
        const auto end = target.data() + target.size();
        const auto parsed =
            std::from_chars(target.data() + prefix.size(), end, original);
        if (parsed.ec != std::errc{} || parsed.ptr != end || original < 0)
          throw std::runtime_error("invalid snapshot descriptor: " + name);
        // Reopening proc fd links can be denied on Android. Dup preserves the
        // owned memfd's access rights; pread/pwrite leave its offset untouched.
        fd = fcntl(original, F_DUPFD_CLOEXEC, 0);
        snapshot = true;
        break;
      }
      char link[PATH_MAX];
      const auto size = readlink(target.c_str(), link, sizeof(link));
      if (size < 0)
        break;
      if (size == sizeof(link))
        throw std::runtime_error("snapshot link too long: " + name);
      const std::filesystem::path next(std::string(link, size));
      path = (next.is_absolute() ? next : path.parent_path() / next)
                 .lexically_normal();
    }
    if (!snapshot)
      fd = open(
          name.c_str(),
          (writable ? O_WRONLY | O_CREAT : O_RDONLY) | O_CLOEXEC,
          0666);
    if (fd < 0)
      throw std::runtime_error("open " + name + ": " + std::strerror(errno));
  }

  ~BinaryFile() {
    close(fd);
  }

  BinaryFile(const BinaryFile&) = delete;
  BinaryFile& operator=(const BinaryFile&) = delete;
};
} // namespace

std::vector<uint8_t> bytes(const std::string& name) {
  BinaryFile file(name, false);

  struct stat info {};

  if (fstat(file.fd, &info) || info.st_size < 0)
    throw std::runtime_error("stat " + name);
  std::vector<uint8_t> result(static_cast<size_t>(info.st_size));
  size_t done = 0;
  while (done < result.size()) {
    const auto count =
        pread(file.fd, result.data() + done, result.size() - done, done);
    if (count < 0 && errno == EINTR)
      continue;
    if (count <= 0)
      throw std::runtime_error("read " + name);
    done += count;
  }
  return result;
}

void save(const std::string& name, const void* data, size_t size) {
  BinaryFile file(name, true);
  if (ftruncate(file.fd, size))
    throw std::runtime_error("truncate " + name);
  size_t done = 0;
  while (done < size) {
    const auto count = pwrite(
        file.fd, static_cast<const uint8_t*>(data) + done, size - done, done);
    if (count < 0 && errno == EINTR)
      continue;
    if (count <= 0)
      throw std::runtime_error("write " + name);
    done += count;
  }
  if (!file.snapshot && fsync(file.fd))
    throw std::runtime_error("sync " + name);
}

uint64_t physical_reads() {
  std::ifstream file("/proc/self/io");
  std::string key;
  uint64_t value;
  while (file >> key >> value)
    if (key == "read_bytes:")
      return value;
  return 0;
}

void validate_rope_tables(
    size_t cos_bytes,
    size_t sin_bytes,
    size_t positions) {
  constexpr size_t row_bytes = 32 * sizeof(float);
  if (!positions || cos_bytes != sin_bytes || cos_bytes % row_bytes ||
      cos_bytes / row_bytes < positions)
    throw std::runtime_error(
        "RoPE tables do not cover the compiled context: rows=" +
        std::to_string(cos_bytes / row_bytes) +
        " required=" + std::to_string(positions));
}
