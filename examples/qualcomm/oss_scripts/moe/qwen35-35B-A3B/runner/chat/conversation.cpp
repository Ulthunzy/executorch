#include "conversation.h"
#include "terminal.h"
#include "../prefill_api.h"
#include "../runtime_utils.h"

#include <executorch/extension/llm/sampler/sampler.h>
#include <algorithm>
#include <cmath>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <unistd.h>

namespace moe_chat {
using executorch::extension::llm::time_in_ms;

Conversation::Conversation(Options settings, Client* transport)
    : options(std::move(settings)), logits(248320), client(transport) {
  stats.model_load_start_ms = time_in_ms();
  if (client)
    return;
  console = fdopen(dup(STDOUT_FILENO), "w");
  if (!console)
    throw std::runtime_error("Cannot open chat output");
}

Conversation::~Conversation() {
  if (close)
    close();
  if (console)
    fclose(console);
}

void Conversation::report_error(const std::string& message) {
  if (client) {
    client->fail(message);
    return;
  }
  fprintf(
      console,
      "推理失败：%s；运行日志：%s\n",
      message.c_str(),
      options.log_path.c_str());
  fflush(console);
}

std::vector<uint64_t> Conversation::encode(const std::string& text) {
  auto result = tokenizer.encode(text, 0, 0);
  if (!result.ok())
    throw std::runtime_error("Text tokenization failed");
  auto ids = std::move(result.get());
  if (std::any_of(ids.begin(), ids.end(), [](auto id) {
        return id >= 248320;
      }))
    throw std::runtime_error("Tokenizer produced an out-of-vocabulary token");
  return ids;
}

std::string
Conversation::prompt(const std::string& text, bool first, bool image) const {
  std::string result;
  if (first)
    result = "<|im_start|>system\n" + options.system_prompt + "<|im_end|>\n";
  else
    result = ended ? "\n" : "<|im_end|>\n";
  result += "<|im_start|>user\n";
  if (image)
    result += "<|vision_start|><|image_pad|><|vision_end|>";
  result += text + "<|im_end|>\n<|im_start|>assistant\n";
  result += options.thinking ? "<think>\n" : "<think>\n\n</think>\n\n";
  return result;
}

std::vector<uint64_t>
Conversation::prompt_tokens(const std::string& text, bool first, bool image) {
  std::string content = text;
  for (auto at = image ? content.find("<image>") : std::string::npos;
       at != std::string::npos;
       at = content.find("<image>"))
    content.erase(at, 7);
  const auto begin = content.find_first_not_of(" \t\r\n");
  if (image)
    content = begin == std::string::npos
        ? ""
        : content.substr(
              begin, content.find_last_not_of(" \t\r\n") - begin + 1);
  auto ids = encode(prompt(content, first, image));
  const auto images = std::count(ids.begin(), ids.end(), 248056);
  if (images != int(image))
    throw std::runtime_error(
        "Use --image-path or /image for image input; do not insert vision markers manually");
  if (image) {
    if (!image_layout)
      throw std::runtime_error("Vision assets are not deployed");
    const auto at = std::find(ids.begin(), ids.end(), 248056);
    ids.insert(at, image_layout->tokens() - 1, 248056);
  }
  return ids;
}

void Conversation::prepare() {
  if (!client) {
    fflush(stdout);
    fflush(stderr);
    FILE* log = fopen(options.log_path.c_str(), "w");
    if (!log || dup2(fileno(log), STDOUT_FILENO) < 0 ||
        dup2(fileno(log), STDERR_FILENO) < 0)
      throw std::runtime_error("Cannot redirect runtime diagnostics");
    fclose(log);
  }
  if (tokenizer.load(options.tokenizer_path) != tokenizers::Error::Ok)
    throw std::runtime_error(
        "Cannot load tokenizer: " + options.tokenizer_path);
  for (const auto& marker : std::vector<std::pair<std::string, uint64_t>>{
           {"<|im_start|>", 248045},
           {"<|im_end|>", 248046},
           {"<|endoftext|>", 248044},
           {"<think>", 248068},
           {"</think>", 248069}}) {
    const auto ids = encode(marker.first);
    if (ids.size() != 1 || ids[0] != marker.second)
      throw std::runtime_error("Tokenizer does not match the exported model");
  }
  rotary = std::make_unique<RotaryPositions>(options.model_root);
  validate_rope_tables(
      rotary->cos.size() * sizeof(float),
      rotary->sin.size() * sizeof(float),
      options.capacity);
  if (options.vision_enabled &&
      std::ifstream(options.model_root + "/prefill/vision.bin"))
    image_layout = std::make_unique<ImageLayout>(options.model_root);
  if (!options.image_path.empty() && !image_layout)
    throw std::runtime_error(
        "Deploy vision.bin, VISION.json and model-config.json before using --image-path");
  if (options.prompts.empty()) {
    std::string line;
    const auto waiting = time_in_ms();
    if (!read_terminal_prompt(std::cin, console, line) || line == "/exit")
      return;
    stats.model_load_start_ms += time_in_ms() - waiting;
    options.prompts.push_back(line);
  }
  if (!normalize_terminal_line(options.prompts.front()))
    throw std::runtime_error("The first prompt is blank");
  first_tokens =
      prompt_tokens(options.prompts.front(), true, !options.image_path.empty());
  stats.token_encode_end_ms = time_in_ms();
  const size_t target = options.prompt_tokens
      ? size_t(options.prompt_tokens)
      : (first_tokens.size() + 127) / 128 * 128;
  const size_t valid = options.prompt_tokens ? target : first_tokens.size();
  if (target < first_tokens.size() ||
      target > size_t(options.max_prefill_tokens) ||
      valid + options.max_new_tokens > size_t(options.capacity))
    throw std::runtime_error(
        "Initial prompt and requested output exceed the compiled context");
  // Explicit benchmark lengths add real whitespace. Ordinary prompts retain
  // their original tokenization and use inactive AR128 padding instead.
  if (options.prompt_tokens) {
    auto end = std::find(first_tokens.begin(), first_tokens.end(), 248046);
    const auto space = encode(" ");
    if (end == first_tokens.end() || space.size() != 1)
      throw std::runtime_error("Invalid ChatML system padding");
    first_tokens.insert(end, target - first_tokens.size(), space[0]);
  }
  rotary->append(first_tokens, 0, image_layout.get());
  last_input_token = first_tokens.back();
  rotary->save(options.model_root);
  if (!options.image_path.empty()) {
    const auto started = Clock::now();
    first_pixels = image_patches(options.image_path, *image_layout);
    initial_prompt_ms += elapsed(started);
    if (options.tokenize_only) {
      std::ofstream pixels(
          options.model_root + "/prefill/pixels.f32", std::ios::binary);
      pixels.write(
          reinterpret_cast<const char*>(first_pixels.data()),
          first_pixels.size() * sizeof(float));
      if (!pixels)
        throw std::runtime_error("Cannot write image preprocessing output");
    }
  }
  std::ofstream file(
      options.model_root + "/prefill/tokens.i64", std::ios::binary);
  auto padded = first_tokens;
  padded.resize(target, 248044);
  file.write(reinterpret_cast<const char*>(padded.data()), padded.size() * 8);
  if (!file)
    throw std::runtime_error("Cannot write prefill tokens");
  file.close();
  configure_environment(options, padded.size(), first_tokens.size());
  const size_t image_tokens =
      std::count(first_tokens.begin(), first_tokens.end(), 248056);
  printf(
      "CHAT_INPUT valid_tokens=%zu text_and_template_tokens=%zu image_tokens=%zu padded_tokens=%zu ar128_blocks=%zu\n",
      first_tokens.size(),
      first_tokens.size() - image_tokens,
      image_tokens,
      padded.size(),
      padded.size() / 128);
  if (options.tokenize_only) {
    fprintf(console, "Encoded %zu tokens\n", first_tokens.size());
    return;
  }
  if (!client) {
    fprintf(console, "正在加载模型…\n");
    fflush(console);
  }
}

std::vector<float> Conversation::initial_image(VisionEncoder* encoder) {
  vision = encoder;
  if (first_pixels.empty())
    return {};
  const auto started = Clock::now();
  stats.on_model_execution_begin();
  auto features = encode_image(vision, first_pixels);
  stats.on_model_execution_end();
  initial_prompt_ms += elapsed(started);
  if (features.size() != size_t(image_layout->tokens()) * 2048)
    throw std::runtime_error("Vision output differs from image token count");
  first_pixels.clear();
  first_pixels.shrink_to_fit();
  return features;
}

void Conversation::initial_prefill_complete(double milliseconds) {
  stats.num_prompt_tokens = first_tokens.size();
  initial_prompt_ms += milliseconds;
  stats.aggregate_model_execution_time_ms += static_cast<long>(milliseconds);
  stats.model_execution_end_ms = time_in_ms();
  stats.model_execution_start_ms =
      stats.model_execution_end_ms - static_cast<long>(milliseconds);
}

void Conversation::consume(int64_t token) {
  stats.on_model_execution_begin();
  if (step(token, logits.data(), logits.size()))
    throw std::runtime_error(error());
  stats.on_model_execution_end();
}

bool Conversation::prefill(
    const std::vector<uint64_t>& ids,
    const std::vector<float>& features) {
  if (client && client->stopped()) {
    needs_reset = true;
    return false;
  }
  Qwen35SessionState state;
  if (export_state(&state))
    throw std::runtime_error(error());
  std::vector<int64_t> tokens;
  if (pending >= 0)
    tokens.push_back(pending);
  tokens.insert(tokens.end(), ids.begin(), ids.end());
  stats.num_prompt_tokens = tokens.size();
  needs_reset = true;
  try {
    stats.on_model_execution_begin();
    qwen35_prefill_append(
        tokens.data(),
        tokens.size(),
        features.data(),
        features.size(),
        rotary->cos.data(),
        rotary->sin.data(),
        rotary->cos.size(),
        state,
        logits.data(),
        [this] {
          return client && client->stopped();
        });
    stats.on_model_execution_end();
  } catch (...) {
    // A partially filled slot may no longer contain its former expert. Mark
    // all slots cold before reset; never publish stale payload identities.
    state.aborted = true;
    for (auto& layer : state.layer) {
      std::fill(std::begin(layer.experts), std::end(layer.experts), -1);
      std::fill(std::begin(layer.stamps), std::end(layer.stamps), 0);
    }
    if (import_state(&state))
      throw std::runtime_error(error());
    if (client && client->stopped())
      return false;
    throw;
  }
  if (import_state(&state))
    throw std::runtime_error(error());
  pending = -1;
  needs_reset = false;
  last_input_token = ids.back();
  return true;
}

void Conversation::generate() {
  const int begin_position = position();
  const int limit =
      std::min(options.max_new_tokens, options.capacity - begin_position);
  executorch::extension::llm::Sampler sampler(
      248320, options.temperature, 0.9f, 42);
  uint64_t previous = last_input_token;
  generated.clear();
  // The opening marker is already in the prompt, so display it without
  // counting it as a generated token or feeding it back into the cache.
  std::string buffered = options.thinking ? "<think>\n" : "";
  const auto started = Clock::now();
  stats.prompt_eval_end_ms = time_in_ms();
  if (client) {
    if (!buffered.empty())
      client->token(buffered);
  } else {
    fprintf(console, "助手> %s", options.stream ? buffered.c_str() : "");
    fflush(console);
  }
  ended = false;
  for (int i = 0; i < limit; ++i) {
    if (client && client->stopped()) {
      stop_reason = "stopped";
      break;
    }
    stats.on_sampling_begin();
    pending = sampler.sample(logits.data());
    stats.on_sampling_end();
    if (i == 0)
      stats.first_token_ms = time_in_ms();
    generated.push_back(pending);
    ended = pending == 248046 || pending == 248044;
    printf(
        "CHAT_TOKEN turn=%d index=%d id=%d position=%d elapsed_ms=%.6f\n",
        turns,
        i,
        pending,
        position(),
        elapsed(started));
    if (ended && !options.ignore_eos) {
      stop_reason = "eos";
      break;
    }
    // HFTokenizer also filters non-special added tokens. Preserve Qwen's
    // thinking boundaries while continuing to hide other control tokens.
    const bool thinking_marker = pending == 248068 || pending == 248069;
    auto piece = tokenizer.decode(previous, pending, !thinking_marker);
    if (!piece.ok())
      throw std::runtime_error("Output detokenization failed");
    const auto& text = piece.get();
    if (client)
      client->token(text);
    else if (options.stream) {
      fwrite(text.data(), 1, text.size(), console);
      fflush(console);
    } else
      buffered += text;
    previous = pending;
    if (i + 1 < limit && !(client && client->stopped())) {
      consume(pending);
      pending = -1;
    }
  }
  if (!client) {
    if (!options.stream)
      fwrite(buffered.data(), 1, buffered.size(), console);
    fprintf(console, "\n");
    fflush(console);
  }
  ++turns;
  empty = false;
  stats.num_generated_tokens = generated.size();
  stats.inference_end_ms = time_in_ms();
  if (stop_reason.empty())
    stop_reason = client && client->stopped() ? "stopped"
        : limit < options.max_new_tokens      ? "context_limit"
                                              : "max_new_tokens";
  printf(
      "CHAT_TURN turn=%d input_position=%d output_position=%d generated=%zu eos=%d elapsed_ms=%.6f\n",
      turns,
      begin_position,
      position(),
      generated.size(),
      ended,
      elapsed(started));
  auto report = nlohmann::json::parse(
      executorch::extension::llm::stats_to_json_string(stats));
  const int kv_tokens = position();
  const int pending_tokens = pending >= 0 ? 1 : 0;
  report.update(
      {{"position", kv_tokens},
       {"kv_cache_tokens", kv_tokens},
       {"pending_tokens", pending_tokens},
       {"context_tokens", kv_tokens + pending_tokens},
       {"context_capacity", options.capacity},
       {"enable_thinking", options.thinking},
       {"generation_ms", elapsed(started)},
       {"stop_reason", stop_reason},
       {"ignore_eos", options.ignore_eos},
       {"eos", stop_reason == "eos"},
       {"stopped", client && client->stopped()},
       {"requires_reset", false}});
  printf("CHAT_STATS turn=%d %s\n", turns, report.dump().c_str());
  if (client)
    client->complete(report.dump());
  else {
    ScopedConsoleOutput output(console);
    executorch::extension::llm::print_report(stats);
    printf(
        "KV cache: %d / %d tokens; pending: %d; context length: %d; thinking: %s\n",
        kv_tokens,
        options.capacity,
        pending_tokens,
        kv_tokens + pending_tokens,
        options.thinking ? "on" : "off");
  }
}

void Conversation::reset_context() {
  if (reset())
    throw std::runtime_error(error());
  pending = -1;
  ended = false;
  empty = true;
  needs_reset = false;
  rotary->reset();
  if (set_inputs(
          nullptr,
          0,
          rotary->cos.data(),
          rotary->sin.data(),
          rotary->cos.size()))
    throw std::runtime_error(error());
}

void Conversation::client_turn(const Request& request) {
  if (request.kind == Request::Kind::Reset) {
    reset_context();
    client->complete("{\"reset\":true}");
    return;
  }
  if (needs_reset)
    throw std::runtime_error("Reset the conversation after interrupted input");
  if (request.max_new_tokens < 1 || request.max_new_tokens > options.capacity ||
      !std::isfinite(request.temperature) || request.temperature < 0)
    throw std::runtime_error("Invalid generation length or temperature");
  const bool image = !request.image_path.empty();
  if (!image &&
      request.prompt.find_first_not_of(" \t\r\n\f\v") == std::string::npos)
    throw std::runtime_error("Prompt is blank");
  stats.reset();
  stats.inference_start_ms = time_in_ms();
  stop_reason.clear();
  const auto ids = prompt_tokens(request.prompt, empty, image);
  stats.token_encode_end_ms = time_in_ms();
  if (position() + int(ids.size()) + (pending >= 0) >= options.capacity)
    throw std::runtime_error("Context is full; reset the conversation");
  std::vector<float> features;
  if (image) {
    if (!vision || !image_layout)
      throw std::runtime_error(
          "Vision was not enabled when creating this module");
    const auto pixels = image_patches(request.image_path, *image_layout);
    stats.on_model_execution_begin();
    features = encode_image(vision, pixels);
    stats.on_model_execution_end();
    if (features.size() != size_t(image_layout->tokens()) * 2048)
      throw std::runtime_error("Vision output differs from image token count");
  }
  rotary->append(ids, position() + (pending >= 0), image_layout.get());
  if (set_inputs(
          features.data(),
          features.size(),
          rotary->cos.data(),
          rotary->sin.data(),
          rotary->cos.size()))
    throw std::runtime_error(error());
  if (!prefill(ids, features)) {
    client->complete("{\"stopped\":true,\"requires_reset\":true}");
    return;
  }
  options.max_new_tokens = request.max_new_tokens;
  options.temperature = request.temperature;
  generate();
}

bool Conversation::turn(std::string text) {
  if (!normalize_terminal_line(text))
    return true;
  if (text == "/exit")
    return false;
  if (text == "/reset") {
    reset_context();
    fprintf(console, "对话已清空。\n");
    fflush(console);
    printf("CHAT_RESET position=%d resources_retained=1\n", position());
    return true;
  }
  stats.reset();
  stats.inference_start_ms = time_in_ms();
  stop_reason.clear();
  std::string message = text, path;
  if (text.rfind("/image ", 0) == 0) {
    std::istringstream input(text.substr(7));
    input >> std::quoted(path);
    std::getline(input, message);
    if (path.empty() || !vision || !image_layout) {
      fprintf(console, "请使用 /image 图片路径 问题，并先部署视觉模型。\n");
      fflush(console);
      return true;
    }
  }
  const auto ids = prompt_tokens(message, empty, !path.empty());
  stats.token_encode_end_ms = time_in_ms();
  if (position() + int(ids.size()) + (pending >= 0) >= options.capacity) {
    fprintf(console, "上下文已满，请输入 /reset 开始新对话。\n");
    fflush(console);
    return true;
  }
  const auto started = Clock::now();
  std::vector<float> features;
  if (!path.empty()) {
    // Decode failures leave KV, pending token and position untouched.
    try {
      const auto pixels = image_patches(path, *image_layout);
      stats.on_model_execution_begin();
      features = encode_image(vision, pixels);
      stats.on_model_execution_end();
      if (features.size() != size_t(image_layout->tokens()) * 2048)
        throw std::runtime_error(
            "Vision output differs from image token count");
    } catch (const std::exception& e) {
      fprintf(console, "图片读取失败：%s\n", e.what());
      fflush(console);
      return true;
    }
  }
  rotary->append(ids, position() + (pending >= 0), image_layout.get());
  if (set_inputs(
          features.data(),
          features.size(),
          rotary->cos.data(),
          rotary->sin.data(),
          rotary->cos.size()))
    throw std::runtime_error(error());
  // The final sampled token (including EOS) has not been fed back yet. Commit
  // it before appending the next user turn; never reset or repad existing KV.
  prefill(ids, features);
  printf(
      "CHAT_APPEND turn=%d tokens=%zu position=%d elapsed_ms=%.6f state_reset=0\n",
      turns + 1,
      ids.size(),
      position(),
      elapsed(started));
  generate();
  return true;
}

int Conversation::run(void* library) {
  open = reinterpret_cast<decltype(open)>(dlsym(library, "qwen35_decode_open"));
  step = reinterpret_cast<decltype(step)>(dlsym(library, "qwen35_decode_step"));
  reset =
      reinterpret_cast<decltype(reset)>(dlsym(library, "qwen35_decode_reset"));
  position = reinterpret_cast<decltype(position)>(
      dlsym(library, "qwen35_decode_position"));
  close =
      reinterpret_cast<decltype(close)>(dlsym(library, "qwen35_decode_close"));
  error =
      reinterpret_cast<decltype(error)>(dlsym(library, "qwen35_decode_error"));
  set_inputs = reinterpret_cast<decltype(set_inputs)>(
      dlsym(library, "qwen35_decode_set_inputs"));
  export_state = reinterpret_cast<decltype(export_state)>(
      dlsym(library, "qwen35_decode_export_state"));
  import_state = reinterpret_cast<decltype(import_state)>(
      dlsym(library, "qwen35_decode_import_state"));
  if (!open || !step || !reset || !position || !close || !error ||
      !set_inputs || !export_state || !import_state)
    throw std::runtime_error("Decode library does not support persistent chat");

  struct DecoderLifetime {
    decltype(close)& release;

    ~DecoderLifetime() {
      if (release)
        release();
      release = nullptr;
    }
  } lifetime{close};

  if (open((options.model_root + "/decode").c_str(), 4096))
    throw std::runtime_error(error());
  if (client)
    client->ready();
  {
    std::ifstream file(
        options.model_root + "/prefill/pte-0-logits.f32", std::ios::binary);
    file.read(
        reinterpret_cast<char*>(logits.data()), logits.size() * sizeof(float));
    if (!file)
      throw std::runtime_error("Cannot read first prefill logits");
  }
  // Prefill runs while resident sessions are being constructed. Keep its
  // measured work in prompt evaluation rather than model-load time.
  stats.model_load_end_ms = time_in_ms() - static_cast<long>(initial_prompt_ms);
  stats.inference_start_ms = stats.model_load_end_ms;
  stats.token_encode_end_ms = stats.inference_start_ms;
  generate();
  if (client) {
    while (true) {
      const auto request = client->next();
      if (request.kind == Request::Kind::Close)
        break;
      options.thinking = request.thinking;
      try {
        client_turn(request);
      } catch (const std::exception& e) {
        client->fail(e.what());
      }
    }
    close();
    close = nullptr;
    return 0;
  }
  bool quit = false;
  for (size_t i = 1; i < options.prompts.size(); ++i)
    if (!turn(options.prompts[i])) {
      quit = true;
      break;
    }
  if (options.interactive && !quit) {
    std::string line;
    while (read_terminal_prompt(std::cin, console, line))
      if (!turn(line))
        break;
  }
  printf(
      "CHAT_SESSION turns=%d final_position=%d model_reloads=0\n",
      turns,
      position());
  close();
  close = nullptr;
  return 0;
}
} // namespace moe_chat
