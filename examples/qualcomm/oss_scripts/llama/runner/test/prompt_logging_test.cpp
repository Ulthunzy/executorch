/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/prompt_logging.h>
#include <executorch/runtime/platform/platform.h>
#include <cassert>
#include <iostream>
#include <iterator>

namespace executorch::runtime {

et_timestamp_t pal_current_ticks() {
  return 0;
}

void pal_emit_log_message(
    et_timestamp_t,
    et_pal_log_level_t,
    const char*,
    const char*,
    size_t,
    const char* message,
    size_t length) {
  assert(length < 255);
  std::cout.write(message, length);
  std::cout << '\n';
}

} // namespace executorch::runtime

int main() {
  const std::string prompt(
      std::istreambuf_iterator<char>{std::cin},
      std::istreambuf_iterator<char>{});
  example::log_prompt(prompt);
}
