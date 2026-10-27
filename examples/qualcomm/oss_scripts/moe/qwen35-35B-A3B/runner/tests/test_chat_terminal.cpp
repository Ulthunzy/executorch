#include "../chat/terminal.h"

#include <cassert>
#include <sys/stat.h>
#include <unistd.h>
#include <sstream>

int main() {
  FILE* output = tmpfile();
  assert(output);
  std::istringstream input("\r\n \t\r\n\n  保留缩进\r\n/exit\r\n");
  std::string line;
  assert(moe_chat::read_terminal_prompt(input, output, line));
  assert(line == "  保留缩进");
  assert(moe_chat::read_terminal_prompt(input, output, line));
  assert(line == "/exit");
  assert(!moe_chat::read_terminal_prompt(input, output, line));
  std::istringstream blank("\r\n \t\n");
  assert(!moe_chat::read_terminal_prompt(blank, output, line));

  struct stat before_stdout, before_stderr, after_stdout, after_stderr;
  assert(fstat(STDOUT_FILENO, &before_stdout) == 0);
  assert(fstat(STDERR_FILENO, &before_stderr) == 0);
  {
    moe_chat::ScopedConsoleOutput report(output);
    fputs("native stdout report\n", stdout);
    fputs("native stderr report\n", stderr);
  }
  assert(fstat(STDOUT_FILENO, &after_stdout) == 0);
  assert(fstat(STDERR_FILENO, &after_stderr) == 0);
  assert(
      before_stdout.st_ino == after_stdout.st_ino &&
      before_stdout.st_dev == after_stdout.st_dev);
  assert(
      before_stderr.st_ino == after_stderr.st_ino &&
      before_stderr.st_dev == after_stderr.st_dev);
  rewind(output);
  std::string printed;
  char buffer[256];
  while (fgets(buffer, sizeof(buffer), output))
    printed += buffer;
  assert(printed.find("native stdout report") != std::string::npos);
  assert(printed.find("native stderr report") != std::string::npos);
  fclose(output);
}
