#pragma once
#include <cstdint>
#include "../../../../../custom_op/lowbit/decode/integer_moe.h"

namespace moe::dsp::decode {

struct BankMapping {
  int fd = -1, generation = -1;
  uint8_t* pointer = nullptr;
  unsigned bytes = 0;
};

void prepare_integer_bank(Sync* sync, const int32_t* descriptor);
int release_expert_mappings();

} // namespace moe::dsp::decode
