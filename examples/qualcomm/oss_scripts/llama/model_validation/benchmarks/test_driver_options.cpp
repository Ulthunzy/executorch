/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "driver_options.h"

#include <iostream>

int main(int argc, char** argv) {
  try {
    const auto config = nlohmann::json::parse(std::cin);
    std::cout << example::identity_eval::parse_sampling_options(
                     config, argc - 1, argv + 1)
                     .dump()
              << '\n';
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
}
