#pragma once

#include <cstddef>
#include <cstdint>

extern "C" __attribute__((visibility("default"))) int qwen35_decode_run(
    int argc,
    const char** argv);

using DecodeMain = decltype(&qwen35_decode_run);

// The prefill owner must outlive this API: its shared contexts and expert
// buffers are borrowed by decode for the entire conversation.
extern "C" {
__attribute__((visibility("default"))) int qwen35_decode_open(
    const char* root,
    int cache_mib);
__attribute__((visibility("default"))) int
qwen35_decode_step(int64_t token, float* logits, size_t count);
__attribute__((visibility("default"))) int qwen35_decode_set_inputs(
    const float* features,
    size_t feature_count,
    const float* cos,
    const float* sin,
    size_t table_count);
__attribute__((visibility("default"))) int qwen35_decode_reset();
__attribute__((visibility("default"))) int qwen35_decode_position();
__attribute__((visibility("default"))) void qwen35_decode_close();
__attribute__((visibility("default"))) const char* qwen35_decode_error();
}
