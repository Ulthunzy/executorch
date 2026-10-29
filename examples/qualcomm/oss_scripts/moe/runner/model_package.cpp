#include "model_package.h"

#include <executorch/schema/extended_header.h>
#include <executorch/schema/program_generated.h>
#include <nlohmann/json.hpp>
#include <zlib.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <cerrno>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <vector>

namespace {
namespace fs = std::filesystem;
using Json = nlohmann::json;

class File {
 public:
  File(const fs::path& path, int flags)
      : fd(::open(path.c_str(), flags, 0600)) {
    if (fd < 0)
      throw std::runtime_error("Cannot open model file: " + path.string());
  }

  ~File() {
    ::close(fd);
  }

  File(const File&) = delete;
  File& operator=(const File&) = delete;
  int fd;
};

void read_at(int fd, uint64_t offset, void* data, size_t size) {
  auto* out = static_cast<char*>(data);
  while (size) {
    const auto n = pread(fd, out, size, offset);
    if (n < 0 && errno == EINTR)
      continue;
    if (n <= 0)
      throw std::runtime_error("Truncated model PTE");
    out += n;
    offset += n;
    size -= n;
  }
}

void write_all(int fd, const void* data, size_t size) {
  auto* input = static_cast<const char*>(data);
  while (size) {
    const auto n = write(fd, input, size);
    if (n < 0 && errno == EINTR)
      continue;
    if (n <= 0)
      throw std::runtime_error("Cannot write model cache");
    input += n;
    size -= n;
  }
}

fs::path relative_name(const std::string& name) {
  fs::path path(name);
  if (name.empty() || name.rfind("EXTRACT_STATE.json", 0) == 0 ||
      path.is_absolute() || path.lexically_normal() != path ||
      path.generic_string() != name || name.find('\\') != std::string::npos ||
      name.find('\0') != std::string::npos)
    throw std::runtime_error("Invalid resource path in model PTE");
  for (const auto& part : path)
    if (part == ".." || part == ".")
      throw std::runtime_error("Invalid resource path in model PTE");
  return path;
}

Json stamp(const fs::path& path) {
  struct stat info {};

  if (lstat(path.c_str(), &info) || !S_ISREG(info.st_mode))
    return Json();
  return Json{
      {"bytes", info.st_size},
      {"mtime_ns",
       int64_t(info.st_mtim.tv_sec) * 1000000000 + info.st_mtim.tv_nsec}};
}

void save_json(const fs::path& path, const Json& value) {
  const auto temporary = path.string() + ".tmp";
  {
    File file(temporary, O_WRONLY | O_CREAT | O_TRUNC | O_NOFOLLOW | O_CLOEXEC);
    const auto text = value.dump(2) + "\n";
    write_all(file.fd, text.data(), text.size());
    if (fsync(file.fd))
      throw std::runtime_error("Cannot sync model cache manifest");
  }
  fs::rename(temporary, path);
}
} // namespace

class ModelPackageLock {
 public:
  explicit ModelPackageLock(const fs::path& path)
      : file(path, O_RDWR | O_CREAT | O_NOFOLLOW | O_CLOEXEC) {
    if (flock(file.fd, LOCK_EX | LOCK_NB))
      throw std::runtime_error("This model workspace is already in use");
  }

 private:
  File file;
};

std::shared_ptr<ModelPackageLock> lock_package_model(const std::string& root) {
  return std::make_shared<ModelPackageLock>(fs::path(root) / ".inference.lock");
}

std::string unpack_model_package(
    const std::string& filename,
    const std::string& cache_directory) {
  File source(filename, O_RDONLY | O_CLOEXEC);

  struct stat info {};

  if (fstat(source.fd, &info) || !S_ISREG(info.st_mode) || info.st_size < 64)
    throw std::runtime_error("Invalid model PTE file");
  const uint64_t file_size = info.st_size;
  char head[64];
  read_at(source.fd, 0, head, sizeof(head));
  const auto parsed =
      executorch::runtime::ExtendedHeader::Parse(head, sizeof(head));
  if (!parsed.ok())
    throw std::runtime_error("Model PTE has no valid segment header");
  const auto header = parsed.get();
  if (header.program_size < 64 || header.program_size > 16 * 1024 * 1024 ||
      header.program_size > file_size ||
      header.segment_base_offset > file_size ||
      header.segment_data_size > file_size - header.segment_base_offset)
    throw std::runtime_error("Truncated model PTE segments");
  std::vector<uint8_t> program_bytes(header.program_size);
  read_at(source.fd, 0, program_bytes.data(), program_bytes.size());
  flatbuffers::Verifier verifier(program_bytes.data(), program_bytes.size());
  if (!executorch_flatbuffer::VerifyProgramBuffer(verifier))
    throw std::runtime_error("Invalid model PTE program");
  const auto* program = executorch_flatbuffer::GetProgram(program_bytes.data());
  std::map<std::string, std::pair<uint64_t, uint64_t>> entries;
  if (!program->named_data() || !program->segments())
    throw std::runtime_error("Model PTE does not contain bundled resources");
  for (const auto* named : *program->named_data()) {
    if (!named || !named->key() ||
        named->segment_index() >= program->segments()->size())
      throw std::runtime_error("Invalid PTE named data");
    const auto* segment = program->segments()->Get(named->segment_index());
    if (segment->offset() > file_size - header.segment_base_offset ||
        segment->size() >
            file_size - header.segment_base_offset - segment->offset())
      throw std::runtime_error("Truncated PTE named data");
    if (!entries
             .emplace(
                 named->key()->str(),
                 std::make_pair(
                     header.segment_base_offset + segment->offset(),
                     segment->size()))
             .second)
      throw std::runtime_error("Duplicate PTE named data key");
  }
  const auto found = entries.find("__qwen35_package_manifest__");
  if (found == entries.end())
    throw std::runtime_error("PTE is not a complete MoE model package");
  const auto index = found->second;
  if (index.second > 8 * 1024 * 1024)
    throw std::runtime_error("Oversized model PTE manifest");
  std::string manifest_data(index.second, '\0');
  read_at(source.fd, index.first, manifest_data.data(), manifest_data.size());
  const auto manifest = Json::parse(manifest_data);
  const std::string id = manifest.at("id");
  if (manifest.at("version") != 1 || id.size() != 64 ||
      id.find_first_not_of("0123456789abcdef") != std::string::npos ||
      entries.size() != manifest.at("files").size() + 1)
    throw std::runtime_error("Invalid model PTE manifest");
  for (const auto& item : manifest.at("files").items()) {
    relative_name(item.key());
    if (entries.at(item.key()).second !=
        item.value().at("bytes").get<uint64_t>())
      throw std::runtime_error("Model resource length mismatch");
  }
  fs::path cache = cache_directory.empty()
      ? fs::absolute(filename).parent_path() / ".qwen35-model-cache"
      : fs::path(cache_directory);
  fs::create_directories(cache);
  cache = fs::canonical(cache);
  auto root = cache / id;
  if (fs::is_symlink(root))
    throw std::runtime_error("Model cache root cannot be a symlink");
  File lock(
      cache / ("." + id + ".export.lock"),
      O_RDWR | O_CREAT | O_NOFOLLOW | O_CLOEXEC);
  if (flock(lock.fd, LOCK_EX))
    throw std::runtime_error("Cannot lock model cache");
  const auto state_path = root / "EXTRACT_STATE.json";
  const Json identity{{"package", id}};
  Json state;
  try {
    std::ifstream file(state_path);
    file >> state;
  } catch (const std::exception&) {
  }
  if (!state.is_object() || state.value("identity", Json()) != identity ||
      !state.contains("stages") || !state["stages"].is_object()) {
    fs::remove_all(root);
    state = Json{{"identity", identity}, {"stages", Json::object()}};
  }
  fs::create_directories(root);
  uint64_t missing_bytes = 0;
  for (const auto& item : manifest.at("files").items()) {
    const auto current = stamp(root / relative_name(item.key()));
    if (!current.is_null() &&
        current == state["stages"].value(item.key(), Json()) &&
        current.at("bytes") == item.value().at("bytes"))
      continue;
    const auto size = entries.at(item.key()).second;
    if (size > UINT64_MAX - missing_bytes)
      throw std::runtime_error("Model cache size overflow");
    missing_bytes += size;
  }
  constexpr uint64_t reserve = 1024 * 1024 * 1024;
  const auto available = fs::space(cache).available;
  if (missing_bytes &&
      (missing_bytes > available || available - missing_bytes < reserve))
    throw std::runtime_error("Not enough free space to extract model package");
  std::vector<char> buffer(8 * 1024 * 1024);
  for (const auto& item : manifest.at("files").items()) {
    const auto output = root / relative_name(item.key());
    auto current = stamp(output);
    if (!current.is_null() &&
        current == state["stages"].value(item.key(), Json()) &&
        current.at("bytes") == item.value().at("bytes"))
      continue;
    const auto available = fs::space(cache).available;
    const auto required = entries.at(item.key()).second;
    if (required > available || available - required < reserve)
      throw std::runtime_error(
          "Not enough free space to extract model resource: " + item.key());
    for (auto parent = output.parent_path(); parent != root;
         parent = parent.parent_path())
      if (fs::is_symlink(parent))
        throw std::runtime_error("Symlink in model cache path");
    fs::create_directories(output.parent_path());
    const auto temporary = output.string() + ".tmp";
    try {
      {
        File target(
            temporary, O_WRONLY | O_CREAT | O_TRUNC | O_NOFOLLOW | O_CLOEXEC);
        auto offset = entries.at(item.key()).first;
        auto remaining = entries.at(item.key()).second;
        uLong crc = 0;
        while (remaining) {
          const size_t amount = std::min<uint64_t>(remaining, buffer.size());
          read_at(source.fd, offset, buffer.data(), amount);
          crc =
              crc32(crc, reinterpret_cast<const Bytef*>(buffer.data()), amount);
          write_all(target.fd, buffer.data(), amount);
          remaining -= amount;
          offset += amount;
        }
        if (uint32_t(crc) != item.value().at("crc32").get<uint32_t>())
          throw std::runtime_error("Corrupt model resource: " + item.key());
        if (fsync(target.fd))
          throw std::runtime_error("Cannot sync model resource");
      }
      fs::rename(temporary, output);
    } catch (...) {
      fs::remove(temporary);
      throw;
    }
    state["stages"][item.key()] = stamp(output);
    save_json(state_path, state);
  }
  for (const auto& item : manifest.at("links").items()) {
    const auto output = root / relative_name(item.key());
    const std::string link = item.value();
    const auto target = (output.parent_path() / link).lexically_normal();
    const auto relative = target.lexically_relative(root);
    if (fs::path(link).is_absolute() || relative.empty() ||
        *relative.begin() == "..")
      throw std::runtime_error("Unsafe model resource link");
    fs::create_directories(output.parent_path());
    if (fs::is_symlink(output)) {
      if (fs::read_symlink(output) == link)
        continue;
      fs::remove(output);
    }
    if (fs::exists(output))
      throw std::runtime_error("Resource link conflicts with a model file");
    fs::create_symlink(link, output);
  }
  save_json(state_path, state);
  return fs::canonical(root).string();
}

void link_model_libraries(
    const std::string& root,
    const std::string& directory) {
  if (directory.empty())
    return;
  const fs::path libraries(directory);
  for (const char* stage : {"prefill", "decode"}) {
    const auto target = fs::path(root) / stage;
    fs::create_directories(target);
    for (const auto& item : fs::directory_iterator(libraries)) {
      if (item.path().extension() != ".so")
        continue;
      const auto name = item.path().filename();
      fs::remove(target / name);
      fs::create_symlink(fs::absolute(item.path()), target / name);
    }

  }
}
