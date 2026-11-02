#include "chat.h"
#include "conversation.h"

#include <memory>

namespace moe_chat {
namespace {
std::unique_ptr<ModelSession> conversation;
// Tokenizer/Abseil archives contain native TLS. ARM64 Bionic requires the
// executable's PT_TLS segment to have at least 64-byte alignment.
alignas(64) thread_local volatile int tokenizer_tls_alignment;
} // namespace

int converse() {
  return conversation->run();
}

bool preloading() {
  return conversation->options.preload;
}

std::vector<float> initial_image(VisionEncoder* encoder) {
  return conversation->initial_image(encoder);
}

void initial_prefill_complete(double milliseconds) {
  conversation->initial_prefill_complete(milliseconds);
}

int run_client(Options options, Client& client, PrefillMain prefill) try {
  tokenizer_tls_alignment = 1;
  const auto family = options.model_family;
  register_conversation(family);
  conversation = create_model_session(family, std::move(options), &client);
  conversation->prepare();
  std::string root = conversation->options.model_root + "/prefill";
  char name[] = "qwen35_moe_jni", one[] = "1", zero[] = "0";
  char* arguments[] = {name, root.data(), one, zero};
  const int result = prefill(4, arguments);
  conversation.reset();
  return result;
} catch (const std::exception& error) {
  client.fail(error.what());
  conversation.reset();
  return 1;
}

int main(int argc, char** argv, PrefillMain prefill) try {
  tokenizer_tls_alignment = argc;
  auto options = parse_options(argc, argv);
  if (options.help_requested)
    return 0;
  const auto family = options.model_family;
  register_conversation(family);
  conversation = create_model_session(family, std::move(options), nullptr);
  conversation->prepare();
  if (conversation->options.prompts.empty() ||
      conversation->options.tokenize_only) {
    conversation.reset();
    return 0;
  }
  std::string root = conversation->options.model_root + "/prefill";
  char one[] = "1", zero[] = "0";
  char* arguments[] = {argv[0], root.data(), one, zero};
  const int result = prefill(4, arguments);
  if (result)
    conversation->report_error("模型执行失败");
  conversation.reset();
  return result;
} catch (const std::exception& error) {
  // Keep the terminal usable even when runtime diagnostics were redirected.
  if (conversation)
    conversation->report_error(error.what());
  else
    fprintf(stderr, "Chat failed: %s\n", error.what());
  conversation.reset();
  return 1;
}
} // namespace moe_chat
