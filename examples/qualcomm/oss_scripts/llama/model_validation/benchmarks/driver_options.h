/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/repetition_penalty.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace example::identity_eval {

inline nlohmann::json parse_sampling_options(
    nlohmann::json settings,
    int argc,
    const char* const* argv) {
  for (int i = 0; i < argc; ++i) {
    const std::string argument = argv[i];
    const auto equals = argument.find('=');
    std::string option = argument.substr(0, equals);
    if (option.rfind("--", 0) != 0) {
      throw std::invalid_argument("Unexpected argument: " + option);
    }
    option.erase(0, 2);
    std::replace(option.begin(), option.end(), '-', '_');
    if (option == "temp") {
      option = "temperature";
    }
    if (option != "temperature" && option != "repetition_penalty" &&
        option != "repeat_last_n" && option != "repair_think_eos" &&
        option != "enable_think_budget" && option != "think_budget") {
      throw std::invalid_argument("Unknown option: " + argument);
    }
    std::string value;
    if (equals != std::string::npos) {
      value = argument.substr(equals + 1);
    } else if (++i < argc) {
      value = argv[i];
    }
    auto parsed = nlohmann::json::parse(value, nullptr, false);
    const bool boolean =
        option == "repair_think_eos" || option == "enable_think_budget";
    if (boolean && parsed.is_number_integer() && (parsed == 0 || parsed == 1)) {
      parsed = parsed == 1;
    }
    if (boolean ? !parsed.is_boolean() : !parsed.is_number()) {
      if (boolean) {
        throw std::invalid_argument(
            "Expected true/false or 1/0 for --" + option);
      }
      throw std::invalid_argument("Expected a numeric value for --" + option);
    }
    settings[option] = parsed;
  }

  for (const auto* option : {"temperature", "repetition_penalty"}) {
    const auto& value = settings.at(option);
    const float minimum = std::string(option) == "temperature" ? 0.f : 1.f;
    if (!value.is_number() || !std::isfinite(value.get<float>()) ||
        value.get<float>() < minimum) {
      throw std::invalid_argument(
          std::string(option) +
          " must be finite and >= " + std::to_string(minimum));
    }
  }
  if (!settings.contains("repeat_last_n")) {
    settings["repeat_last_n"] = RepetitionPenalty::kDefaultWindowSize;
  }
  const auto& window = settings.at("repeat_last_n");
  if (!window.is_number_integer() || window.get<int64_t>() <= 0 ||
      window.get<int64_t>() > std::numeric_limits<int32_t>::max()) {
    throw std::invalid_argument(
        "repeat_last_n must be a positive 32-bit integer");
  }
  for (const auto* option : {"repair_think_eos", "enable_think_budget"}) {
    if (settings.contains(option) && !settings.at(option).is_boolean()) {
      throw std::invalid_argument(
          std::string(option) + " must be a JSON boolean");
    }
  }
  if (settings.contains("think_budget")) {
    const auto& budget = settings.at("think_budget");
    if (!budget.is_number_integer() || budget.get<int64_t>() <= 0 ||
        budget.get<int64_t>() > std::numeric_limits<int32_t>::max()) {
      throw std::invalid_argument(
          "think_budget must be a positive 32-bit integer");
    }
  }
  return settings;
}

} // namespace example::identity_eval
