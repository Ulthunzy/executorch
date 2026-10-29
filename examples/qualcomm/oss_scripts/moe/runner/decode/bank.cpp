#include "bank.h"

namespace moe::decode {

Bank::Bank(const std::string& root, int layer)
    : index(bytes(
          root + "/indices/layer" + std::to_string(layer) +
          "/bank-index.bin")) {
  if (index.size() != 4096 + 256 * 8)
    throw std::runtime_error("bank index length");
  const bool n2 =
      getenv("EXPERT_BITS") && std::string(getenv("EXPERT_BITS")) == "2";
  fd = open(
      (root + "/banks/layer" + std::to_string(layer) +
       (n2 ? "/experts.n2" : "/experts.n4"))
          .c_str(),
      O_RDONLY | O_DIRECT);
  if (fd < 0)
    throw std::runtime_error("open bank");

  struct stat st {};

  if (fstat(fd, &st))
    throw std::runtime_error("bank stat");
  size = st.st_size;
  for (int e = 0; e < 256; e++)
    maximum = std::max(maximum, length(e));
}

size_t Bank::offset(int e) const {
  return reinterpret_cast<const uint64_t*>(index.data() + 32)[e];
}

size_t Bank::length(int e) const {
  return reinterpret_cast<const uint64_t*>(index.data() + 4096)[e];
}

Bank::~Bank() {
  close(fd);
}

} // namespace moe::decode
