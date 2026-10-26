#pragma once

#include <vector>
struct VisionEncoder;

namespace moe_chat {
struct Options;
class Client;
using PrefillMain = int (*)(int, char**);
int main(int argc, char** argv, PrefillMain prefill);
int run_client(Options options, Client& client, PrefillMain prefill);
int converse(void* decoder);
std::vector<float> initial_image(VisionEncoder* encoder);
} // namespace moe_chat
