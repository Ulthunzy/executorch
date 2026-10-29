#include "infer_api.h"
#include <cstdio>

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
#ifdef MOE_DIAGNOSTIC_RUNNER
  return et_infer_run_prefill(argc, argv);
#else
  return et_infer_run_chat(argc, argv);
#endif
}
