#pragma once

#ifdef __cplusplus
extern "C" {
#endif
__attribute__((visibility("default"))) int et_infer_run_chat(int argc, char** argv);
__attribute__((visibility("default"))) int et_infer_run_prefill(int argc, char** argv);
#ifdef __cplusplus
}
#endif
