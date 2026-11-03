#pragma once

#include "session_state.h"
#include <functional>

void qwen35_prefill_append(
    const int64_t* tokens,
    size_t count,
    const float* features,
    size_t feature_count,
    const float* cos,
    const float* sin,
    size_t table_count,
    Qwen35SessionState& state,
    float* logits,
    const std::function<bool()>& stopped);

void qwen35_prefill_set_power_active(bool active);
