#pragma once

#include "decode_session.h"

namespace moe::decode {

extern std::string model_root;
extern std::unique_ptr<DecodeSession> session;

} // namespace moe::decode
