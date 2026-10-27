#pragma once

#include <cstdio>
#include <istream>
#include <string>

namespace moe_chat {
class ScopedConsoleOutput {
 public:
  explicit ScopedConsoleOutput(FILE* output);
  ~ScopedConsoleOutput();
  ScopedConsoleOutput(const ScopedConsoleOutput&) = delete;
  ScopedConsoleOutput& operator=(const ScopedConsoleOutput&) = delete;

 private:
  void restore();
  int stdout_fd, stderr_fd;
};

bool normalize_terminal_line(std::string& line);
bool read_terminal_prompt(std::istream& input, FILE* output, std::string& line);
} // namespace moe_chat
