#include "bank.h"

Bank::Bank(const std::string& root, int layer)
    : index(bytes(
          root + "/indices/layer" + std::to_string(layer) +
          "/bank-index.bin")) {
  if (index.size() != 6144)
    throw std::runtime_error("bank index size");
  const bool n2 =
      getenv("EXPERT_BITS") && std::string(getenv("EXPERT_BITS")) == "2";
  fd = open(
      (root + "/compact/layer" + std::to_string(layer) +
       (n2 ? "/experts.n2" : "/experts.n4"))
          .c_str(),
      O_RDONLY | O_DIRECT);
  if (fd < 0)
    throw std::runtime_error("expert bank open");

  struct stat statbuf {};

  if (fstat(fd, &statbuf))
    throw std::runtime_error("expert bank stat");
  for (int i = 0; i < 256; i++) {
    if (offset(i) + size(i) > size_t(statbuf.st_size))
      throw std::runtime_error("bank range");
    maximum = std::max(maximum, size(i));
  }
}

size_t Bank::offset(int e) const {
  return reinterpret_cast<const uint64_t*>(index.data() + 32)[e];
}

size_t Bank::size(int e) const {
  return reinterpret_cast<const uint64_t*>(index.data() + 4096)[e];
}

Bank::~Bank() {
  if (fd >= 0)
    close(fd);
}
