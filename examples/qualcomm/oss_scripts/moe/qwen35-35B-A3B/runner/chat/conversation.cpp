#include "conversation.h"
#include "../runtime_utils.h"

#include <executorch/extension/llm/sampler/sampler.h>
#include <algorithm>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <unistd.h>

namespace moe_chat {
Conversation::Conversation(Options settings)
    : options(std::move(settings)), logits(248320) {
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
  fflush(stdout);
  fflush(stderr);
  FILE* log = fopen(options.log_path.c_str(), "w");
  if (!log || dup2(fileno(log), STDOUT_FILENO) < 0 ||
      dup2(fileno(log), STDERR_FILENO) < 0)
    throw std::runtime_error("Cannot redirect runtime diagnostics");
  fclose(log);
  if (tokenizer.load(options.tokenizer_path) != tokenizers::Error::Ok)
    throw std::runtime_error(
        "Cannot load tokenizer: " + options.tokenizer_path);
  for (const auto& marker : std::vector<std::pair<std::string, uint64_t>>{
           {"<|im_start|>", 248045},
           {"<|im_end|>", 248046},
           {"<|endoftext|>", 248044}}) {
    const auto ids = encode(marker.first);
    if (ids.size() != 1 || ids[0] != marker.second)
      throw std::runtime_error("Tokenizer does not match the exported model");
  }
  rotary = std::make_unique<RotaryPositions>(options.model_root);
  if (std::ifstream(options.model_root + "/prefill/vision.bin"))
    image_layout = std::make_unique<ImageLayout>(options.model_root);
  if (!options.image_path.empty() && !image_layout)
    throw std::runtime_error(
        "Deploy vision.bin, VISION.json and model-config.json before using --image-path");
  if (options.prompts.empty()) {
    fprintf(console, "用户> ");
    fflush(console);
    std::string line;
    if (!std::getline(std::cin, line) || line == "/exit")
      return;
    if (line.empty())
      throw std::runtime_error("The first prompt is empty");
    options.prompts.push_back(line);
  }
  first_tokens =
      prompt_tokens(options.prompts.front(), true, !options.image_path.empty());
  const size_t target = (first_tokens.size() + 127) / 128 * 128;
  if (target > 1024 ||
      target + options.max_new_tokens > size_t(options.capacity) + 1)
    throw std::runtime_error(
        "Initial prompt and requested output exceed the compiled context");
  // Match the calibrated host path: whitespace lives inside the system message,
  // leaving the final cache/logits at the actual assistant generation boundary.
  auto end = std::find(first_tokens.begin(), first_tokens.end(), 248046);
  const auto space = encode(" ");
  if (end == first_tokens.end() || space.size() != 1)
    throw std::runtime_error("Invalid ChatML system padding");
  first_tokens.insert(end, target - first_tokens.size(), space[0]);
  rotary->append(first_tokens, 0, image_layout.get());
  rotary->save(options.model_root);
  if (!options.image_path.empty()) {
    first_pixels = image_patches(options.image_path, *image_layout);
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
  file.write(
      reinterpret_cast<const char*>(first_tokens.data()),
      first_tokens.size() * 8);
  if (!file)
    throw std::runtime_error("Cannot write prefill tokens");
  file.close();
  configure_environment(options, first_tokens.size());
  if (options.tokenize_only) {
    fprintf(console, "Encoded %zu tokens\n", first_tokens.size());
    return;
  }
  fprintf(console, "正在加载模型…\n");
  fflush(console);
}

std::vector<float> Conversation::initial_image(VisionEncoder* encoder) {
  vision = encoder;
  if (first_pixels.empty())
    return {};
  auto features = encode_image(vision, first_pixels);
  if (features.size() != size_t(image_layout->tokens()) * 2048)
    throw std::runtime_error("Vision output differs from image token count");
  first_pixels.clear();
  first_pixels.shrink_to_fit();
  return features;
}

void Conversation::consume(int64_t token) {
  if (step(token, logits.data(), logits.size()))
    throw std::runtime_error(error());
}

void Conversation::generate() {
  const int begin_position = position();
  const int limit =
      std::min(options.max_new_tokens, options.capacity + 1 - begin_position);
  executorch::extension::llm::Sampler sampler(
      248320, options.temperature, 0.9f, 42);
  uint64_t previous = first_tokens.back();
  generated.clear();
  const auto started = Clock::now();
  fprintf(console, "助手> ");
  fflush(console);
  ended = false;
  for (int i = 0; i < limit; ++i) {
    pending = sampler.sample(logits.data());
    generated.push_back(pending);
    ended = pending == 248046 || pending == 248044;
    printf(
        "CHAT_TOKEN turn=%d index=%d id=%d position=%d elapsed_ms=%.6f\n",
        turns,
        i,
        pending,
        position(),
        elapsed(started));
    if (ended)
      break;
    auto piece = tokenizer.decode(previous, pending, true);
    if (!piece.ok())
      throw std::runtime_error("Output detokenization failed");
    const auto& text = piece.get();
    fwrite(text.data(), 1, text.size(), console);
    fflush(console);
    previous = pending;
    if (i + 1 < limit)
      consume(pending);
  }
  fprintf(console, "\n");
  fflush(console);
  ++turns;
  empty = false;
  printf(
      "CHAT_TURN turn=%d input_position=%d output_position=%d generated=%zu eos=%d elapsed_ms=%.6f\n",
      turns,
      begin_position,
      position(),
      generated.size(),
      ended,
      elapsed(started));
}

bool Conversation::turn(const std::string& text) {
  if (text == "/exit")
    return false;
  if (text == "/reset") {
    if (reset())
      throw std::runtime_error(error());
    pending = -1;
    ended = false;
    empty = true;
    rotary->reset();
    if (set_inputs(
            nullptr,
            0,
            rotary->cos.data(),
            rotary->sin.data(),
            rotary->cos.size()))
      throw std::runtime_error(error());
    fprintf(console, "对话已清空。\n");
    fflush(console);
    printf("CHAT_RESET position=%d resources_retained=1\n", position());
    return true;
  }
  if (text.empty())
    return true;
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
  if (position() + int(ids.size()) + (pending >= 0) > options.capacity) {
    fprintf(console, "上下文已满，请输入 /reset 开始新对话。\n");
    fflush(console);
    return true;
  }
  const auto started = Clock::now();
  std::vector<float> features;
  if (!path.empty()) {
    // Decode failures leave KV, pending token and position untouched.
    try {
      features = encode_image(vision, image_patches(path, *image_layout));
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
  if (pending >= 0)
    consume(pending);
  for (const auto id : ids)
    consume(id);
  pending = -1;
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
  if (!open || !step || !reset || !position || !close || !error || !set_inputs)
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
  {
    std::ifstream file(
        options.model_root + "/prefill/pte-0-logits.f32", std::ios::binary);
    file.read(
        reinterpret_cast<char*>(logits.data()), logits.size() * sizeof(float));
    if (!file)
      throw std::runtime_error("Cannot read first prefill logits");
  }
  generate();
  bool quit = false;
  for (size_t i = 1; i < options.prompts.size(); ++i)
    if (!turn(options.prompts[i])) {
      quit = true;
      break;
    }
  if (options.interactive && !quit) {
    std::string line;
    while (fprintf(console, "用户> "),
           fflush(console),
           std::getline(std::cin, line))
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
