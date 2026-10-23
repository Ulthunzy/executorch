#pragma once

extern "C" __attribute__((visibility("default")))
int qwen35_decode_run(int argc, const char** argv);

using DecodeMain = decltype(&qwen35_decode_run);
