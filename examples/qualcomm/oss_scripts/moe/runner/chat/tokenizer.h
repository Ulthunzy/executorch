#pragma once

#include <pytorch/tokenizers/hf_tokenizer.h>

namespace moe_chat {
class DeviceTokenizer final : public tokenizers::HFTokenizer {
 public:
  tokenizers::Error load(const std::string& path) override;
  tokenizers::Result<std::vector<uint64_t>> encode(
      const std::string& text,
      int8_t bos = 0,
      int8_t eos = 0) const override;
};
} // namespace moe_chat
