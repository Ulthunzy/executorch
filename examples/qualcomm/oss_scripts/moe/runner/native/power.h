#pragma once
#include "HAP_dcvs.h"
#include "HAP_power.h"

namespace moe::dsp {
class PowerVote {
  void* context_;
  bool voted_ = false;

 public:
  explicit PowerVote(void* context) : context_(context) {}
  PowerVote(const PowerVote&) = delete;
  PowerVote& operator=(const PowerVote&) = delete;
  ~PowerVote() {
    set(false);
  }

  int set(bool active) {
    if (!active) {
      if (!voted_)
        return 0;
      const int error = HAP_power_destroy(context_);
      if (!error)
        voted_ = false;
      return error;
    }
    if (voted_)
      return 0;
    HAP_power_request_t power{};
    power.type = HAP_power_set_apptype;
    power.apptype = HAP_POWER_COMPUTE_CLIENT_CLASS;
    int error = HAP_power_set(context_, &power);
    if (error) {
      set(false);
      return error;
    }
    voted_ = true;
    power = {};
    power.type = HAP_power_set_DCVS_v3;
    auto& dcvs = power.dcvs_v3;
    dcvs.set_dcvs_enable = 1;
    dcvs.dcvs_enable = 0;
    dcvs.set_bus_params = 1;
    dcvs.bus_params.min_corner = dcvs.bus_params.target_corner =
        dcvs.bus_params.max_corner = HAP_DCVS_VCORNER_MAX;
    dcvs.set_core_params = 1;
    dcvs.core_params.min_corner = dcvs.core_params.target_corner =
        dcvs.core_params.max_corner = HAP_DCVS_VCORNER_MAX;
    dcvs.set_sleep_disable = 1;
    dcvs.sleep_disable = 1;
    HAP_set_dcvs_v3_protected_bus_corners(&power, 1);
    error = HAP_power_set(context_, &power);
    if (error) {
      set(false);
      return error;
    }
    power = {};
    power.type = HAP_power_set_HVX;
    power.hvx.power_up = 1;
    error = HAP_power_set(context_, &power);
    if (error) {
      set(false);
      return error;
    }
    power = {};
    power.type = HAP_power_set_HMX_v2;
    power.hmx_v2.set_power = 1;
    power.hmx_v2.power_up = 1;
    power.hmx_v2.set_clock = 1;
    power.hmx_v2.target_corner = power.hmx_v2.min_corner =
        power.hmx_v2.max_corner = HAP_DCVS_EXP_VCORNER_MAX;
    power.hmx_v2.perf_mode = HAP_CLK_PERF_HIGH;
    error = HAP_power_set(context_, &power);
    if (error) {
      set(false);
      return error;
    }
    return 0;
  }
};
} // namespace moe::dsp
