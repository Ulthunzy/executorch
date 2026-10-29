#include "terminal.h"

#include <stdexcept>
#include <unistd.h>

namespace moe_chat {
ScopedConsoleOutput::ScopedConsoleOutput(FILE* output) {
  fflush(nullptr);
  stdout_fd = dup(STDOUT_FILENO);
  stderr_fd = dup(STDERR_FILENO);
  if (stdout_fd < 0 || stderr_fd < 0) {
    if (stdout_fd >= 0)
      close(stdout_fd);
    if (stderr_fd >= 0)
      close(stderr_fd);
    throw std::runtime_error("Cannot save runtime log streams");
  }
  if (output &&
      (dup2(fileno(output), STDOUT_FILENO) < 0 ||
       dup2(fileno(output), STDERR_FILENO) < 0)) {
    restore();
    throw std::runtime_error("Cannot route the native report to the console");
  }
}

void ScopedConsoleOutput::restore() {
  fflush(nullptr);
  dup2(stdout_fd, STDOUT_FILENO);
  dup2(stderr_fd, STDERR_FILENO);
  close(stdout_fd);
  close(stderr_fd);
}

ScopedConsoleOutput::~ScopedConsoleOutput() {
  restore();
}

bool normalize_terminal_line(std::string& line) {
  // getline removes LF but leaves the CR in CRLF input forwarded by ADB.
  while (!line.empty() && line.back() == '\r')
    line.pop_back();
  return line.find_first_not_of(" \t\r\n\f\v") != std::string::npos;
}

bool read_terminal_prompt(
    std::istream& input,
    FILE* output,
    std::string& line) {
  while (true) {
    fprintf(output, "用户> ");
    fflush(output);
    if (!std::getline(input, line))
      return false;
    if (normalize_terminal_line(line))
      return true;
  }
}
} // namespace moe_chat
