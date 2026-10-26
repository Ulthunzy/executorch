#pragma once

#include <string>

namespace moe_chat {
struct Request {
  enum class Kind { Generate, Reset, Close };
  Kind kind = Kind::Generate;
  std::string prompt, image_path;
  int max_new_tokens = 128;
  float temperature = 0;
  bool thinking = false;
};

// The application owns scheduling and callbacks; the model thread owns all
// context/state operations. No terminal IO or process-wide stream redirection.
class Client {
 public:
  virtual ~Client() = default;
  virtual Request next() = 0;
  virtual bool stopped() const = 0;

  virtual void ready() {}

  virtual void token(const std::string& utf8) = 0;
  virtual void complete(const std::string& stats) = 0;
  virtual void fail(const std::string& message) = 0;
};
} // namespace moe_chat
