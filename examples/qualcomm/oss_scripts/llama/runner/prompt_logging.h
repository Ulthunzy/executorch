/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/platform/log.h>
#include <string>

namespace example {

inline void log_prompt(const std::string& prompt) {
#if ET_LOG_ENABLED
  ET_LOG(Info, "[QNN_PROMPT] begin: bytes=%zu", prompt.size());
  size_t offset = 0;
  while (offset < prompt.size()) {
    const size_t start = offset;
    std::string chunk;
    // Leave room in ET_LOG's 256-byte buffer for metadata and UTF-8 boundaries.
    while (offset < prompt.size() &&
           (chunk.size() < 160 ||
            (static_cast<unsigned char>(prompt[offset]) & 0xc0) == 0x80)) {
      const auto byte = static_cast<unsigned char>(prompt[offset++]);
      switch (byte) {
        case '\\':
          chunk += "\\\\";
          break;
        case '"':
          chunk += "\\\"";
          break;
        case '\n':
          chunk += "\\n";
          break;
        case '\r':
          chunk += "\\r";
          break;
        case '\t':
          chunk += "\\t";
          break;
        default:
          if (byte < 0x20 || byte == 0x7f) {
            constexpr char hex[] = "0123456789abcdef";
            chunk += "\\u00";
            chunk += hex[byte >> 4];
            chunk += hex[byte & 0xf];
          } else {
            chunk += static_cast<char>(byte);
          }
      }
    }
    ET_LOG(Info, "[QNN_PROMPT] offset=%zu text=\"%s\"", start, chunk.c_str());
  }
  ET_LOG(Info, "[QNN_PROMPT] end");
#else
  (void)prompt;
#endif
}

} // namespace example
