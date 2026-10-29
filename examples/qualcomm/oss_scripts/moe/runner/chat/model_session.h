#pragma once

#include "options.h"
#include "client.h"
#include <memory>
#include <utility>

struct VisionEncoder;

namespace moe_chat {
class ModelSession {
 public:
  explicit ModelSession(Options settings) : options(std::move(settings)) {}
  virtual ~ModelSession() = default;
  virtual void prepare() = 0;
  virtual int run() = 0;
  virtual void report_error(const std::string& message) = 0;
  virtual std::vector<float> initial_image(VisionEncoder* encoder) = 0;
  virtual void initial_prefill_complete(double milliseconds) = 0;
  Options options;
};

using SessionFactory = std::unique_ptr<ModelSession> (*)(Options, Client*);
void register_model_session(const std::string& family, SessionFactory factory);
std::unique_ptr<ModelSession> create_model_session(
    const std::string& family, Options options, Client* client = nullptr);
} // namespace moe_chat
