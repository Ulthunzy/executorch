#pragma once

#include <functional>
#include <memory>
#include <string>

namespace example {
class QNNMultimodalRunner;
}

namespace executorch::android::dense {

struct Config {
  std::string decoder, embedding, encoder, tokenizer, libraries, system;
  int context_length = 0;
  int image_height = 448;
  int image_width = 448;
};

class Session {
 public:
  explicit Session(Config config);
  ~Session();
  void prepare();
  std::string generate(
      const std::string& prompt,
      const std::string& image,
      int count,
      float temperature,
      bool thinking,
      bool ignore_eos,
      const std::function<void(const std::string&)>& callback);
  void stop();
  void reset();

 private:
  Config config_;
  std::unique_ptr<example::QNNMultimodalRunner> runner_;
  bool started_ = false;
  bool ended_with_eos_ = false;
};

} // namespace executorch::android::dense
