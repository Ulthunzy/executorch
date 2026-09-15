/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/runner.h>
#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <nlohmann/json.hpp>

#include "driver_options.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using json = nlohmann::json;

int main(int argc, char** argv) {
  if (argc == 5 && std::string(argv[1]) == "--tokenize") {
    auto tokenizer = executorch::extension::llm::load_tokenizer(argv[2]);
    if (!tokenizer) {
      return 3;
    }
    std::ifstream input(argv[3]);
    json output = json::array();
    for (const auto& item : json::parse(input)) {
      auto encoded = tokenizer->encode(item.at("prompt"), 0, 0);
      if (!encoded.ok()) {
        return 3;
      }
      output.push_back({{"id", item.at("id")}, {"input_ids", encoded.get()}});
    }
    std::ofstream result(argv[4]);
    result << output.dump() << '\n';
    return result.good() ? 0 : 3;
  }
  if (argc == 2 && std::string(argv[1]) == "--capabilities") {
    std::cout
        << R"({"protocol_version":2,"formatted_prompt":true,"repeat_last_n":true,"sampling_cli":true,"tokenized_prompt":true,"thinking_controls":true})"
        << std::endl;
    return 0;
  }
  const bool help = argc == 2 &&
      (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h");
  if (argc < 4 || help) {
    std::cerr
        << "Usage: identity_eval_driver CONFIG_JSON SESSION_JSON OUTPUT_DIR\n"
        << "  [--temp FLOAT] [--repetition-penalty FLOAT] [--repeat-last-n INT]\n"
        << "  [--repair-think-eos 0|1] [--enable-think-budget 0|1] [--think-budget INT]\n"
        << "CLI values override JSON for this run only. Underscore names and --name=value are also accepted.\n";
    return help ? 0 : 2;
  }
  std::ifstream config_input(argv[1]);
  std::ifstream session_input(argv[2]);
  json settings;
  json session;
  try {
    settings = example::identity_eval::parse_sampling_options(
        json::parse(config_input), argc - 4, argv + 4);
    session = json::parse(session_input);
  } catch (const std::exception& error) {
    std::cerr << "Invalid driver input: " << error.what() << '\n';
    return 2;
  }
  const std::string output_dir = argv[3];
  const std::string model_path = settings.at("model");
  auto tokenizer = executorch::extension::llm::load_tokenizer(settings.at("tokenizer"));
  if (!tokenizer) {
    std::cerr << "Cannot load test tokenizer" << std::endl;
    return 3;
  }
  auto* tokenizer_view = tokenizer.get();
  auto module = std::make_unique<executorch::extension::Module>(
      model_path,
      executorch::extension::Module::LoadMode::MmapUseMlockIgnoreErrors);
  example::Runner runner(
      std::move(module),
      settings.at("decoder"),
      model_path,
      settings.at("tokenizer"),
      "",
      "",
      settings.at("temperature"),
      settings.at("mode") == "hybrid" ? example::Runner::kHybrid
                                      : example::Runner::kKVCached,
      false,
      0,
      0,
      0,
      std::move(tokenizer),
      nullptr,
      settings.at("repetition_penalty"),
      settings.at("repeat_last_n").get<size_t>());
  executorch::extension::llm::GenerationConfig config;
  config.echo = false;
  config.ignore_eos = false;
  config.max_new_tokens = settings.value("max_new_tokens", -1);
  config.seq_len = settings.at("seq_len");
  config.temperature = settings.at("temperature");
  config.num_bos = 0;
  config.num_eos = 0;
  config.repair_think_eos = settings.value("repair_think_eos", false);
  config.enable_think_budget = settings.value("enable_think_budget", false);
  config.think_budget = settings.value("think_budget", 512);

  for (const auto& turn : session.at("turns")) {
    const std::string id = turn.at("id");
    const std::string question = turn.at("prompt");
    const bool history_before = runner.has_history();
    const std::string prompt = turn.contains("formatted_prompt")
        ? turn.at("formatted_prompt").get<std::string>()
        : (history_before ? "\n" : "") + std::string("<|im_start|>user\n") +
            question + "<|im_end|>\n<|im_start|>assistant\n";
    const auto encoded = tokenizer_view->encode(prompt, 0, 0);
    if (!encoded.ok()) {
      std::cerr << "Cannot encode test prompt" << std::endl;
      return 3;
    }
    auto input_ids = encoded.get();
    std::string runner_input = prompt;
    const bool tokenized_prompt = turn.contains("input_ids");
    if (tokenized_prompt) {
      const auto& ids = turn.at("input_ids");
      if (!ids.is_array() || ids.empty()) {
        std::cerr << "input_ids must be a nonempty array" << std::endl;
        return 2;
      }
      input_ids.clear();
      for (const auto& value : ids) {
        if (!value.is_number_unsigned() || value.get<uint64_t>() >= tokenizer_view->vocab_size()) {
          std::cerr << "Invalid input token ID" << std::endl;
          return 2;
        }
        input_ids.push_back(value.get<uint64_t>());
      }
      runner_input = output_dir + "/" + id + ".tokens.u64";
      std::ofstream tokens(runner_input, std::ios::binary);
      tokens.write(reinterpret_cast<const char*>(input_ids.data()), input_ids.size() * sizeof(uint64_t));
      tokens.close();
      if (!tokens) {
        std::cerr << "Cannot save prompt token IDs" << std::endl;
        return 3;
      }
    }
    std::cout << "IDENTITY_TURN_BEGIN " << id << std::endl;
    std::cout << "IDENTITY_INPUT "
              << json({{"id", id},
                       {"has_history_before", history_before},
                       {"formatted_prompt", prompt}})
                     .dump()
              << '\n';
    std::istringstream prompt_lines(prompt);
    for (std::string line; std::getline(prompt_lines, line);) {
      std::cout << "PROMPT | " << line << '\n';
    }
    std::cout << std::flush;
    std::ofstream transcript(output_dir + "/" + id + ".txt", std::ios::binary);
    if (!transcript) {
      std::cerr << "Cannot open transcript for " << id << std::endl;
      return 3;
    }
    std::string response;
    json stats = nullptr;
    size_t callbacks = 0;
    const auto started = std::chrono::steady_clock::now();
    const auto error = runner.generate_from_prompt_or_file(
        runner_input,
        tokenized_prompt,
        config,
        [&](const std::string& piece) {
          response += piece;
          transcript.write(piece.data(), piece.size());
          if (++callbacks % 32 == 0) {
            transcript.flush();
          }
        },
        [&](const executorch::llm::Stats& value) {
          stats = json::parse(
              executorch::extension::llm::stats_to_json_string(value));
        });
    transcript.flush();
    const json record = {
        {"id", id},
        {"session_id", session.at("id")},
        {"turn", turn.at("turn")},
        {"prompt", question},
        {"formatted_prompt", prompt},
        {"incremental_input_ids", input_ids},
        {"native_input_ids", encoded.get()},
        {"tokenized_prompt", tokenized_prompt},
        {"has_history_before", history_before},
        {"has_history_after", runner.has_history()},
        {"error_code", static_cast<uint32_t>(error)},
        {"wall_seconds",
         std::chrono::duration<double>(
             std::chrono::steady_clock::now() - started)
             .count()},
        {"stats", stats},
        {"response", response}};
    std::ofstream result(output_dir + "/" + id + ".json");
    result << record.dump(-1, ' ', false, json::error_handler_t::replace)
           << '\n';
    result.flush();
    if (!result || !transcript) {
      std::cerr << "Cannot save result for " << id << std::endl;
      return 3;
    }
    std::cout << "IDENTITY_TURN_END " << id << std::endl;
  }
  return 0;
}
